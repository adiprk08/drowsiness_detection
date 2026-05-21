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

Combined models
---------------
``--combined`` calibrates a model trained by ``src.train_combined``
(folder ``artifacts/<model>_combined/``). The DDD val/test splits come
from the SQLite bundle and the UTA val/test splits are reconstructed
from the checkpoint's stored args (seed / val-frac / test-frac), so the
calibration sees exactly the same held-out data the model was selected
on. The threshold is swept on the **combined** val set — the same
selection signal ``train_combined`` used — and every operating point is
then reported per domain (DDD / UTA / combined).

Usage
-----
    py -m src.calibrate --model mobilenet_v2 --num-workers 0 --batch-size 256
    py -m src.calibrate --model mobilenet_v2 --combined --num-workers 0
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
from .uta_rldd import UtaRldDataset, split_uta_subjects


# ---------------------------------------------------------------------------
# Scoring a dataset (just runs the model, collects sigmoid scores + labels)
# ---------------------------------------------------------------------------

def _score_dataset(model, dataset, device, batch_size, num_workers,
                   desc: str = "score") -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    scores, labels = [], []
    with torch.inference_mode():
        for imgs, ys in tqdm(loader, desc=desc, dynamic_ncols=True):
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
# Threshold sweep on val → two operating points
# ---------------------------------------------------------------------------

def _sweep_and_pick(val_scores: np.ndarray, val_labels: np.ndarray,
                    recall_target: float) -> dict:
    """Sweep thresholds on val and pick the best-macro-F1 and safety points.

    Returns a dict with the sweep curves (for plotting) and the two
    chosen thresholds.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    mF1_curve = np.array([
        _metrics_at(val_scores, val_labels, t)["macro_f1"] for t in thresholds
    ])
    recall_curve = np.array([
        _metrics_at(val_scores, val_labels, t)["recall_drowsy"] for t in thresholds
    ])

    # Operating point 1: best macro-F1 on val.
    t_best_mf1 = float(thresholds[int(np.argmax(mF1_curve))])

    # Operating point 2: largest threshold that still hits the recall target
    # on val (thresholds ascend → recall descends, so the largest passing
    # threshold gives the best precision at the target recall).
    ok = recall_curve >= recall_target
    if ok.any():
        t_safety = float(thresholds[np.where(ok)[0].max()])
    else:
        t_safety = float(thresholds[0])  # can't hit target, use lowest threshold

    return {
        "thresholds": thresholds,
        "mF1_curve": mF1_curve,
        "recall_curve": recall_curve,
        "t_best_mf1": t_best_mf1,
        "t_safety": t_safety,
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
# Pretty-printing an operating point
# ---------------------------------------------------------------------------

def _print_op(name: str, threshold: float, val: dict, test: dict) -> None:
    print(f"\n  -- {name}  (threshold={threshold:.2f}) --")
    print(f"    val  : acc={val['accuracy']:.3f}  mF1={val['macro_f1']:.3f}  "
          f"drowsy_r={val['recall_drowsy']:.3f}  drowsy_p={val['precision_drowsy']:.3f}")
    print(f"    test : acc={test['accuracy']:.3f}  mF1={test['macro_f1']:.3f}  "
          f"drowsy_r={test['recall_drowsy']:.3f}  drowsy_p={test['precision_drowsy']:.3f}")
    c = test["confusion"]
    print(f"    test confusion: [[{c['tn']:>5d}, {c['fp']:>5d}], "
          f"[{c['fn']:>5d}, {c['tp']:>5d}]]")


# ---------------------------------------------------------------------------
# Calibration: DDD-only single-stream model
# ---------------------------------------------------------------------------

def _calibrate_single(args: argparse.Namespace, device: torch.device) -> None:
    ckpt_path = (Path(args.ckpt) if args.ckpt
                 else Path(args.artifacts) / args.model / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(args.model, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Score both splits once — we'll reuse these arrays for every threshold.
    val_scores, val_labels = _score_dataset(
        model, FaceStreamDataset(args.db, split="val"),
        device, args.batch_size, args.num_workers, desc="score val",
    )
    test_scores, test_labels = _score_dataset(
        model, FaceStreamDataset(args.db, split="test"),
        device, args.batch_size, args.num_workers, desc="score test",
    )

    sweep = _sweep_and_pick(val_scores, val_labels, args.recall_target)
    t_best_mf1, t_safety = sweep["t_best_mf1"], sweep["t_safety"]

    # Re-evaluate test at each operating point (and 0.5 default for reference).
    def _op(t):
        return {
            "threshold": t,
            "val":  _metrics_at(val_scores,  val_labels,  t),
            "test": _metrics_at(test_scores, test_labels, t),
        }

    safety_key = f"recall_≥_{args.recall_target:.2f}_on_val"
    summary = {
        "model": args.model,
        "combined": False,
        "ckpt": str(ckpt_path),
        "val_n": int(val_labels.size),
        "test_n": int(test_labels.size),
        "operating_points": {
            "default_0.5": _op(0.5),
            "best_macro_f1_on_val": _op(t_best_mf1),
            safety_key: _op(t_safety),
        },
    }

    out_dir = Path(args.artifacts) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "calibration.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    png_path = out_dir / "threshold_sweep.png"
    _plot_sweep(sweep["thresholds"], sweep["mF1_curve"], sweep["recall_curve"],
                picks={"bestF1": t_best_mf1, "safety": t_safety},
                out_path=png_path,
                title=f"{args.model} - threshold sweep on val")

    print(f"\n[calibrate] {args.model}")
    print(f"  val N={val_labels.size}  test N={test_labels.size}")
    for name, op in summary["operating_points"].items():
        _print_op(name, op["threshold"], op["val"], op["test"])
    print(f"\n[calibrate] → {json_path}")
    print(f"[calibrate] → {png_path}")


# ---------------------------------------------------------------------------
# Calibration: combined (DDD + UTA-RLDD) model from src.train_combined
# ---------------------------------------------------------------------------

def _calibrate_combined(args: argparse.Namespace, device: torch.device) -> None:
    model_dir = Path(args.artifacts) / f"{args.model}_combined"
    ckpt_path = Path(args.ckpt) if args.ckpt else model_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"combined checkpoint not found: {ckpt_path} "
            f"(train it with `py -m src.train_combined --model {args.model}`)"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(args.model, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Reproduce the exact held-out splits the combined model was selected on.
    # The DDD splits are fixed in the SQLite bundle; the UTA subject split is
    # deterministic given seed / val-frac / test-frac, which the checkpoint
    # records — so we must read them from there, not from CLI defaults.
    train_args = ckpt.get("args", {})
    seed = int(train_args.get("seed", 42))
    val_frac = float(train_args.get("val_frac", 0.15))
    test_frac = float(train_args.get("test_frac", 0.15))
    print(f"[calibrate] combined splits from ckpt: seed={seed} "
          f"val_frac={val_frac} test_frac={test_frac}")

    _, val_subjects, test_subjects = split_uta_subjects(
        args.uta_frames, val_frac=val_frac, test_frac=test_frac, seed=seed,
    )

    eval_sets = {
        ("val", "ddd"):   FaceStreamDataset(args.db, split="val"),
        ("test", "ddd"):  FaceStreamDataset(args.db, split="test"),
        ("val", "uta"):   UtaRldDataset(args.uta_frames, subjects=val_subjects),
        ("test", "uta"):  UtaRldDataset(args.uta_frames, subjects=test_subjects),
    }

    # Score every (split, domain) once; combined = the two domains pooled.
    scores: dict[tuple[str, str], np.ndarray] = {}
    labels: dict[tuple[str, str], np.ndarray] = {}
    for (split, domain), ds in eval_sets.items():
        s, l = _score_dataset(model, ds, device, args.batch_size,
                              args.num_workers, desc=f"score {split}/{domain}")
        scores[(split, domain)] = s
        labels[(split, domain)] = l
    for split in ("val", "test"):
        scores[(split, "combined")] = np.concatenate(
            [scores[(split, "ddd")], scores[(split, "uta")]])
        labels[(split, "combined")] = np.concatenate(
            [labels[(split, "ddd")], labels[(split, "uta")]])

    domains = ["combined", "ddd", "uta"]

    # Sweep on the COMBINED val set — the same signal train_combined used to
    # pick the best epoch — then report every operating point per domain.
    sweep = _sweep_and_pick(scores[("val", "combined")],
                            labels[("val", "combined")], args.recall_target)
    t_best_mf1, t_safety = sweep["t_best_mf1"], sweep["t_safety"]

    def _op(t):
        return {
            "threshold": t,
            "val":  {d: _metrics_at(scores[("val", d)],  labels[("val", d)],  t)
                     for d in domains},
            "test": {d: _metrics_at(scores[("test", d)], labels[("test", d)], t)
                     for d in domains},
        }

    safety_key = f"recall_≥_{args.recall_target:.2f}_on_val"
    summary = {
        "model": args.model,
        "combined": True,
        "ckpt": str(ckpt_path),
        "selection_key": ckpt.get("selection_key", "combined"),
        "swept_on": "combined val",
        "recall_target": args.recall_target,
        "val_n": {d: int(labels[("val", d)].size) for d in domains},
        "test_n": {d: int(labels[("test", d)].size) for d in domains},
        "operating_points": {
            "default_0.5": _op(0.5),
            "best_macro_f1_on_val": _op(t_best_mf1),
            safety_key: _op(t_safety),
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "calibration.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    png_path = model_dir / "threshold_sweep.png"
    _plot_sweep(sweep["thresholds"], sweep["mF1_curve"], sweep["recall_curve"],
                picks={"bestF1": t_best_mf1, "safety": t_safety},
                out_path=png_path,
                title=f"{args.model}_combined - threshold sweep on combined val")

    print(f"\n[calibrate] {args.model}_combined  (swept on combined val)")
    for d in domains:
        print(f"  {d:>8}: val N={summary['val_n'][d]:>6d}  "
              f"test N={summary['test_n'][d]:>6d}")
    for name, op in summary["operating_points"].items():
        print(f"\n  ==== {name}  (threshold={op['threshold']:.2f}) ====")
        for d in domains:
            _print_op(d, op["threshold"], op["val"][d], op["test"][d])
    print(f"\n[calibrate] → {json_path}")
    print(f"[calibrate] → {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate decision threshold on val.")
    p.add_argument("--model", required=True,
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"])
    p.add_argument("--combined", action="store_true",
                   help="Calibrate a src.train_combined model — reads "
                        "artifacts/<model>_combined/ and reconstructs the "
                        "DDD+UTA val/test splits from the checkpoint args.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--uta-frames", default="data/uta_rldd_frames",
                   help="On-disk UTA face-crop tree (combined mode only).")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--recall-target", type=float, default=0.95,
                   help="Target drowsy recall for the safety-first operating point.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.combined:
        _calibrate_combined(args, device)
    else:
        _calibrate_single(args, device)


if __name__ == "__main__":
    main()
