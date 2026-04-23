"""Evaluate a trained single-stream checkpoint on the held-out test split.

Usage
-----
    py -m src.eval --model mobilenet_v2
    py -m src.eval --model alexnet --ckpt artifacts/alexnet/best.pt

What it reports
---------------
- Loss, accuracy, precision, recall, F1 (positive=drowsy), macro-F1, ROC-AUC
- Confusion matrix (printed + saved as PNG)
- A bundle JSON at ``artifacts/<model>/test_metrics.json``

ROC-AUC is computed with a tiny sort-based implementation so we don't have
to pull in scikit-learn just for one number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_single_stream import FaceStreamDataset
from .models import build_model


# ---------------------------------------------------------------------------
# Metrics (sklearn-free)
# ---------------------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney-U formulation; undefined if one class is missing."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # rank of combined scores, average ties
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    combined = np.concatenate([pos, neg])[order]
    # handle ties with average rank
    i = 0
    rank_vals = np.empty(combined.size, dtype=np.float64)
    while i < combined.size:
        j = i
        while j + 1 < combined.size and combined[j + 1] == combined[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # ranks are 1-based
        rank_vals[i:j + 1] = avg
        i = j + 1
    ranks[order] = rank_vals
    sum_ranks_pos = ranks[:pos.size].sum()
    u = sum_ranks_pos - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def _confusion(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """2×2 confusion with rows=true, cols=pred, order [alert, drowsy]."""
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["alert", "drowsy"])
    ax.set_yticks([0, 1], labels=["alert", "drowsy"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained face-stream model.")
    p.add_argument("--model", required=True,
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"])
    p.add_argument("--ckpt", default=None,
                   help="Path to checkpoint. Defaults to artifacts/<model>/best.pt.")
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.artifacts) / args.model / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # Build an architecturally-matching model. ``pretrained=False`` avoids
    # needlessly downloading ImageNet weights — we're about to overwrite them.
    model = build_model(args.model, pretrained=False,
                        freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = FaceStreamDataset(args.db, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"[eval] {args.model} on {args.split} (N={len(ds)}) from {ckpt_path}")

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.inference_mode():
        for imgs, labels in tqdm(loader, desc=f"eval {args.split}",
                                 dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            logits = model(imgs)
            loss = criterion(logits, labels)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            all_scores.append(torch.sigmoid(logits).cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(np.int64)
    preds = (scores >= 0.5).astype(np.int64)

    cm = _confusion(preds, labels)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = cm.sum()
    acc = (tp + tn) / total
    pos_p = tp / (tp + fp) if (tp + fp) else 0.0
    pos_r = tp / (tp + fn) if (tp + fn) else 0.0
    pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r) if (pos_p + pos_r) else 0.0
    neg_p = tn / (tn + fn) if (tn + fn) else 0.0
    neg_r = tn / (tn + fp) if (tn + fp) else 0.0
    neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r) if (neg_p + neg_r) else 0.0
    macro_f1 = 0.5 * (pos_f1 + neg_f1)
    auc = _roc_auc(scores, labels)
    mean_loss = total_loss / max(total_n, 1)

    print(f"[eval] loss      : {mean_loss:.4f}")
    print(f"[eval] accuracy  : {acc:.4f}")
    print(f"[eval] precision : {pos_p:.4f}  (drowsy)")
    print(f"[eval] recall    : {pos_r:.4f}  (drowsy)")
    print(f"[eval] F1        : {pos_f1:.4f}  (drowsy)")
    print(f"[eval] macro-F1  : {macro_f1:.4f}")
    print(f"[eval] ROC-AUC   : {auc:.4f}")
    print( "[eval] confusion matrix (rows=true, cols=pred, order=[alert, drowsy]):")
    print(f"        [[{tn:>6d}, {fp:>6d}],")
    print(f"         [{fn:>6d}, {tp:>6d}]]")

    out_dir = Path(args.artifacts) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_png = out_dir / f"confusion_{args.split}.png"
    _plot_confusion(cm, cm_png, title=f"{args.model} — {args.split}")

    metrics_path = out_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps({
        "model": args.model,
        "split": args.split,
        "ckpt": str(ckpt_path),
        "n": int(total),
        "loss": mean_loss,
        "accuracy": acc,
        "precision_drowsy": pos_p,
        "recall_drowsy": pos_r,
        "f1_drowsy": pos_f1,
        "f1_alert": neg_f1,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }, indent=2))
    print(f"[eval] confusion matrix → {cm_png}")
    print(f"[eval] metrics          → {metrics_path}")


if __name__ == "__main__":
    main()
