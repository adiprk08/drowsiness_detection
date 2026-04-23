"""Evaluate the two-stream (eye + face) model on a held-out split.

Reports **per-branch** metrics because our data has no paired samples —
every test image is either an MRL eye crop or a DDD face crop, never
both. So the eye branch's numbers are over MRL test samples only, and
the face branch's numbers are over DDD test samples only.

This is the right way to compare:

    face-only MobileNetV2 (from ``src/train.py``)
        vs.
    face branch of two-stream model (evaluated on the same DDD test subset)

The face branch's test metrics are directly comparable to the single-stream
MobileNetV2 results, which tells us whether jointly training with the eye
branch helped, hurt, or was neutral for face-stream generalisation.

Usage
-----
    py -m src.eval_fusion --num-workers 0 --batch-size 256
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

from .datasets import SQLiteDrowsinessDataset
from .eval import _plot_confusion, _roc_auc
from .models import build_model


def _per_branch_metrics(scores: np.ndarray, labels: np.ndarray, name: str,
                        out_dir: Path, split: str) -> dict:
    if scores.size == 0:
        return {"n": 0, "note": f"no {name}-stream samples in {split}"}
    preds = (scores >= 0.5).astype(np.int64)
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
    auc = _roc_auc(scores, labels)

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    cm_path = out_dir / f"{name}_confusion_{split}.png"
    _plot_confusion(cm, cm_path, title=f"two_stream {name} branch — {split}")

    return {
        "n": int(n), "accuracy": acc,
        "precision_drowsy": pos_p, "recall_drowsy": pos_r,
        "f1_drowsy": pos_f1, "f1_alert": neg_f1,
        "macro_f1": macro_f1, "roc_auc": auc,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_png": str(cm_path),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the two-stream model.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.artifacts) / "two_stream" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model("two_stream", pretrained=False,
                        freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = SQLiteDrowsinessDataset(args.db, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"[eval_fusion] two_stream on {args.split} (N={len(ds)}) from {ckpt_path}")

    eye_scores: list[np.ndarray] = []
    eye_labels: list[np.ndarray] = []
    face_scores: list[np.ndarray] = []
    face_labels: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval {args.split}", dynamic_ncols=True):
            eye = batch["eye"].to(device, non_blocking=True)
            face = batch["face"].to(device, non_blocking=True)
            em = batch["eye_mask"].cpu().numpy().ravel()
            fm = batch["face_mask"].cpu().numpy().ravel()
            y = batch["label"].cpu().numpy().ravel()

            out = model(eye, face)
            e_p = torch.sigmoid(out["eye_logit"]).cpu().numpy().ravel()
            f_p = torch.sigmoid(out["face_logit"]).cpu().numpy().ravel()

            # Split scores by which branch each sample actually has data for
            eye_sel = em > 0.5
            face_sel = fm > 0.5
            if eye_sel.any():
                eye_scores.append(e_p[eye_sel])
                eye_labels.append(y[eye_sel].astype(np.int64))
            if face_sel.any():
                face_scores.append(f_p[face_sel])
                face_labels.append(y[face_sel].astype(np.int64))

    out_dir = Path(args.artifacts) / "two_stream"
    out_dir.mkdir(parents=True, exist_ok=True)

    eye_arr = np.concatenate(eye_scores) if eye_scores else np.empty(0)
    eye_lab = np.concatenate(eye_labels) if eye_labels else np.empty(0, dtype=np.int64)
    face_arr = np.concatenate(face_scores) if face_scores else np.empty(0)
    face_lab = np.concatenate(face_labels) if face_labels else np.empty(0, dtype=np.int64)

    eye_m = _per_branch_metrics(eye_arr, eye_lab, "eye", out_dir, args.split)
    face_m = _per_branch_metrics(face_arr, face_lab, "face", out_dir, args.split)

    def _print(name: str, m: dict) -> None:
        if m.get("n", 0) == 0:
            print(f"\n  {name} branch: {m.get('note', 'no samples')}")
            return
        print(f"\n  {name} branch (N={m['n']}):")
        print(f"    acc       : {m['accuracy']:.4f}")
        print(f"    macro-F1  : {m['macro_f1']:.4f}")
        print(f"    F1 drowsy : {m['f1_drowsy']:.4f}")
        print(f"    recall    : {m['recall_drowsy']:.4f}  (drowsy)")
        print(f"    precision : {m['precision_drowsy']:.4f}  (drowsy)")
        print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
        c = m["confusion"]
        print(f"    confusion : [[{c['tn']:>5d}, {c['fp']:>5d}],")
        print(f"                 [{c['fn']:>5d}, {c['tp']:>5d}]]")

    print(f"\n[eval_fusion] {args.split} results")
    _print("eye", eye_m)
    _print("face", face_m)

    metrics_path = out_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps({
        "model": "two_stream",
        "split": args.split,
        "ckpt": str(ckpt_path),
        "eye_branch":  eye_m,
        "face_branch": face_m,
    }, indent=2))
    print(f"\n[eval_fusion] metrics → {metrics_path}")


if __name__ == "__main__":
    main()
