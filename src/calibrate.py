"""Threshold calibration for the single-stream face models.

Why we need this
----------------
``eval.py`` thresholds the sigmoid output at 0.5 because that's the
"textbook" default, but 0.5 is arbitrary — the model's probability scale
depends on its training data, class balance, and calibration. The actual
threshold that maximises macro-F1 (or hits a given drowsy-recall target)
might be 0.42 or 0.58.

This script sweeps thresholds on the **val** set, picks two operating
points, and re-scores **test** at each. We never choose thresholds on the
test set — that would leak test labels into our model-selection process.

Two operating points we report
------------------------------
1. **Best macro-F1 on val**   — the balanced "best overall" threshold.
2. **Drowsy recall ≥ 0.95 on val** — the "safety-first" threshold, because
   in drowsiness detection a missed microsleep is more costly than a
   false alarm.

Both are then evaluated on test and written to
``artifacts/<model>/calibration.json`` plus a ``threshold_sweep.png``.

Usage
-----
    py -m src.calibrate --model mobilenet_v2 --num-workers 0 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_single_stream import FaceStreamDataset
from .models import build_model


# ---------------------------------------------------------------------------
# Scoring a split (just runs the model, collects sigmoid scores + labels)
# ---------------------------------------------------------------------------

def _score_split(model, db_path, split, device, batch_size, num_workers
                 ) -> tuple[np.ndarray, np.ndarray]:
    ds = FaceStreamDataset(db_path, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    scores, labels = [], []
    with torch.inference_mode():
        for imgs, ys in tqdm(loader, desc=f"score {split}", dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            scores.append(torch.sigmoid(logits).cpu().numpy().ravel())
            labels.append(ys.cpu().numpy().ravel())
    return np.concatenate(scores), np.concatenate(labels).astype(np.int64)


# ---------------------------------------------------------------------------
# Metrics at a given threshold (sklearn-free, matches eval.py conventions)
# ---------------------------------------------------------------------------

def _metrics_at(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    preds = (scores >= t).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    pos_p = tp / (tp + fp) if (tp + fp) else 0.0
    pos_r = tp / (tp + fn) if (tp + fn) else 0.0
    pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r) if (pos_p + pos_r) else 0.0
    neg_p = tn / (tn + fn) if (tn + fn) else 0.0
    neg_r = tn / (tn + fp) if (tn + fp) else 0.0
    neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r) if (neg_p + neg_r) else 0.0
    macro_f1 = 0.5 * (pos_f1 + neg_f1)
    return {
        "threshold": float(t), "n": int(n), "accuracy": acc,
        "precision_drowsy": pos_p, "recall_drowsy": pos_r,
        "f1_drowsy": pos_f1, "f1_alert": neg_f1, "macro_f1": macro_f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sweep(thresholds: np.ndarray, macro_f1: np.ndarray,
                drowsy_recall: np.ndarray, picks: dict, out_path: Path,
                title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, macro_f1, label="macro-F1 (val)", linewidth=2)
    ax.plot(thresholds, drowsy_recall, label="drowsy recall (val)",
            linewidth=2, linestyle="--")
    for name, t in picks.items():
        ax.axvline(t, color="grey", linestyle=":", alpha=0.7)
        ax.text(t, 0.02, name, rotation=90, fontsize=8, va="bottom",
                ha="right", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.5,
               label="default 0.5")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate decision threshold on val.")
    p.add_argument("--model", required=True,
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--recall-target", type=float, default=0.95,
                   help="Target drowsy recall for the safety-first operating point.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.artifacts) / args.model / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(args.model, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Score both splits once — we'll reuse these arrays for every threshold.
    val_scores, val_labels = _score_split(
        model, args.db, "val", device, args.batch_size, args.num_workers,
    )
    test_scores, test_labels = _score_split(
        model, args.db, "test", device, args.batch_size, args.num_workers,
    )

    # Threshold sweep on val
    thresholds = np.linspace(0.05, 0.95, 91)
    mF1_curve = np.array([
        _metrics_at(val_scores, val_labels, t)["macro_f1"] for t in thresholds
    ])
    recall_curve = np.array([
        _metrics_at(val_scores, val_labels, t)["recall_drowsy"] for t in thresholds
    ])

    # Operating point 1: best macro-F1 on val
    t_best_mf1 = float(thresholds[int(np.argmax(mF1_curve))])

    # Operating point 2: lowest threshold that still hits recall-target on val
    # (lowest threshold = highest recall; we want the least aggressive threshold
    # that still catches the target fraction of drowsy samples)
    ok = recall_curve >= args.recall_target
    if ok.any():
        # thresholds is ascending, higher t → lower recall → we want the
        # largest t where recall still ≥ target (best precision at that recall)
        t_safety = float(thresholds[np.where(ok)[0].max()])
    else:
        t_safety = float(thresholds[0])  # can't hit target, use lowest threshold

    # Re-evaluate test at each operating point (and 0.5 default for reference)
    summary = {
        "model": args.model,
        "ckpt": str(ckpt_path),
        "val_n": int(val_labels.size),
        "test_n": int(test_labels.size),
        "operating_points": {
            "default_0.5": {
                "threshold": 0.5,
                "val":  _metrics_at(val_scores,  val_labels,  0.5),
                "test": _metrics_at(test_scores, test_labels, 0.5),
            },
            "best_macro_f1_on_val": {
                "threshold": t_best_mf1,
                "val":  _metrics_at(val_scores,  val_labels,  t_best_mf1),
                "test": _metrics_at(test_scores, test_labels, t_best_mf1),
            },
            f"recall_≥_{args.recall_target:.2f}_on_val": {
                "threshold": t_safety,
                "val":  _metrics_at(val_scores,  val_labels,  t_safety),
                "test": _metrics_at(test_scores, test_labels, t_safety),
            },
        },
    }

    # Write artifacts
    out_dir = Path(args.artifacts) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "calibration.json"
    json_path.write_text(json.dumps(summary, indent=2))
    png_path = out_dir / "threshold_sweep.png"
    _plot_sweep(thresholds, mF1_curve, recall_curve,
                picks={"bestF1": t_best_mf1, "safety": t_safety},
                out_path=png_path,
                title=f"{args.model} — threshold sweep on val")

    # Pretty-print
    print(f"\n[calibrate] {args.model}")
    print(f"  val N={val_labels.size}  test N={test_labels.size}")
    for name, op in summary["operating_points"].items():
        print(f"\n  ── {name}  (threshold={op['threshold']:.2f}) ──")
        v, t = op["val"], op["test"]
        print(f"    val  : acc={v['accuracy']:.3f}  mF1={v['macro_f1']:.3f}  "
              f"drowsy_r={v['recall_drowsy']:.3f}  drowsy_p={v['precision_drowsy']:.3f}")
        print(f"    test : acc={t['accuracy']:.3f}  mF1={t['macro_f1']:.3f}  "
              f"drowsy_r={t['recall_drowsy']:.3f}  drowsy_p={t['precision_drowsy']:.3f}")
        c = t["confusion"]
        print(f"    test confusion: [[{c['tn']:>5d}, {c['fp']:>5d}], "
              f"[{c['fn']:>5d}, {c['tp']:>5d}]]")
    print(f"\n[calibrate] → {json_path}")
    print(f"[calibrate] → {png_path}")


if __name__ == "__main__":
    main()
