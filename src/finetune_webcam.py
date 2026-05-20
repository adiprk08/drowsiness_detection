"""Fine-tune MobileNetV2 on the webcam calibration set + DDD anchor.

Why this script exists
----------------------
The pretrained MobileNetV2 (``artifacts/mobilenet_v2/best.pt``) was
trained on cabin-camera footage from DDD and doesn't transfer to a
laptop webcam — the face branch saturates near 1.0 regardless of state
(see the handoff doc for the full story). Fine-tuning on a small
in-domain set is the standard fix.

What this does
--------------
1. Loads the trained checkpoint as starting weights.
2. Builds a webcam training set from ``data/webcam_calibration/`` (run
   ``src.collect_webcam`` first to populate it).
3. Mixes in a small balanced sample of the original DDD training data as
   an "anchor" to prevent catastrophic forgetting of the original
   distribution. Default 500 samples per class.
4. Fine-tunes for a few epochs at a tiny LR (default 1e-5), with the
   final two MobileNetV2 blocks + classifier unfrozen.
5. Saves the best epoch to ``artifacts/mobilenet_v2_finetuned/best.pt``
   (best on a held-out webcam val split).
6. Evaluates the fine-tuned model on the held-out DDD test split so we
   can report whether we forgot the original distribution. Target: stay
   within ~2 percentage points of original test macro-F1 (0.705).

Usage
-----
    py -m src.finetune_webcam
    py -m src.finetune_webcam --epochs 5 --lr 5e-6
    py -m src.finetune_webcam --webcam-root data/webcam_calibration --anchor-per-class 800

Output
------
    artifacts/mobilenet_v2_finetuned/
        best.pt
        history.json
        ddd_test_metrics.json   (sanity check vs original 0.705)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .data_single_stream import FaceStreamDataset
from .datasets import DrowsinessDataset, _letterbox
from .models import build_model
from .train import _binary_metrics, _run_epoch


# ---------------------------------------------------------------------------
# Webcam folder dataset
# ---------------------------------------------------------------------------

class WebcamFolderDataset(Dataset):
    """Reads the alert/ and drowsy/ folders produced by ``src.collect_webcam``.

    Each sample yields ``(Tensor[3, 224, 224], Tensor[1])`` — identical
    shape to :class:`FaceStreamDataset`, so the two can be concat-ed and
    fed to the same training loop unchanged.
    """

    _MEAN = DrowsinessDataset._MEAN
    _STD = DrowsinessDataset._STD

    def __init__(self, root: str | Path,
                 augment_fn=None):
        self.root = Path(root)
        self.augment_fn = augment_fn
        self.items: list[tuple[Path, int]] = []
        for cls_name, label in (("alert", 0), ("drowsy", 1)):
            folder = self.root / cls_name
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.jpg")):
                self.items.append((p, label))
        if not self.items:
            raise ValueError(
                f"No images found under {self.root}. "
                f"Run `py -m src.collect_webcam` to populate it first."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            # Fall back to the next sample rather than crashing the loop.
            return self.__getitem__((idx + 1) % len(self))
        # collect_webcam already letterboxed to 224 before writing, but call
        # _letterbox again defensively in case someone drops in a raw JPEG.
        if bgr.shape[:2] != (224, 224):
            bgr = _letterbox(bgr, 224)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment_fn is not None:
            rgb = self.augment_fn(rgb)
        img = rgb.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(img),
                torch.tensor([float(label)], dtype=torch.float32))

    def class_counts(self) -> dict[str, int]:
        out = {"alert": 0, "drowsy": 0}
        for _, lab in self.items:
            out["drowsy" if lab == 1 else "alert"] += 1
        return out


# ---------------------------------------------------------------------------
# DDD anchor — small balanced sample of the original training distribution
# ---------------------------------------------------------------------------

def _build_ddd_anchor(db_path: str, per_class: int,
                      seed: int) -> Subset:
    """Pull a balanced ``per_class`` sample from DDD train so the fine-tune
    doesn't catastrophically forget the original cabin-camera distribution.
    No augmentation — we want the anchor stable across epochs."""
    ddd_train = FaceStreamDataset(db_path, split="train", augment=False)
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, s in enumerate(ddd_train.samples):
        by_label[int(s.label)].append(i)
    chosen: list[int] = []
    for lab in (0, 1):
        pool = by_label[lab]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    rng.shuffle(chosen)
    return Subset(ddd_train, chosen)


# ---------------------------------------------------------------------------
# Webcam val split — leave the last N per class out for early stopping
# ---------------------------------------------------------------------------

def _split_webcam(ds: WebcamFolderDataset, val_per_class: int,
                  seed: int) -> tuple[Subset, Subset]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, (_, lab) in enumerate(ds.items):
        by_label[lab].append(i)
    train_idx, val_idx = [], []
    for lab in (0, 1):
        pool = by_label[lab]
        rng.shuffle(pool)
        v = min(val_per_class, max(1, len(pool) // 5))  # cap at 20% if pool tiny
        val_idx.extend(pool[:v])
        train_idx.extend(pool[v:])
    rng.shuffle(train_idx)
    return Subset(ds, train_idx), Subset(ds, val_idx)


# ---------------------------------------------------------------------------
# Fine-tune target unfreeze — last 2 inverted-residual blocks + classifier
# ---------------------------------------------------------------------------

def _unfreeze_last_blocks(model: nn.Module, unfreeze_from: int = 17) -> None:
    """MobileNetV2 has 19 blocks in ``features``. Unfreeze from
    ``unfreeze_from`` onward plus the classifier head. Default 17 = last
    two inverted-residuals + the 1x1 conv tail."""
    # Freeze everything first, then re-open the tail.
    for p in model.parameters():
        p.requires_grad = False
    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on webcam calibration data.")
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--webcam-root", default="data/webcam_calibration")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--source-model", default="mobilenet_v2",
                   help="Subfolder under --artifacts to load the starting checkpoint from.")
    p.add_argument("--out-name", default="mobilenet_v2_finetuned",
                   help="Subfolder under --artifacts to write the fine-tuned checkpoint to.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Tiny LR — we're nudging a trained checkpoint, not retraining.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--anchor-per-class", type=int, default=500,
                   help="DDD samples per class mixed in as anti-forgetting anchor. "
                        "0 disables.")
    p.add_argument("--webcam-val-per-class", type=int, default=10,
                   help="Webcam samples per class held out for early stopping.")
    p.add_argument("--unfreeze-from", type=int, default=17,
                   help="MobileNetV2 features[] block index to start unfreezing from "
                        "(default 17 = last 2 IRBs + tail).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stopping patience on webcam-val macro-F1.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}")

    artifacts = Path(args.artifacts)
    src_ckpt = artifacts / args.source_model / "best.pt"
    if not src_ckpt.exists():
        sys.exit(f"source checkpoint not found: {src_ckpt} — "
                 f"run `py -m src.train --model mobilenet_v2` first")
    out_dir = artifacts / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data --------------------------------------------------------------
    webcam_full = WebcamFolderDataset(args.webcam_root)
    print(f"[finetune] webcam set: {len(webcam_full)} samples  "
          f"(counts={webcam_full.class_counts()})")

    webcam_train, webcam_val = _split_webcam(
        webcam_full, args.webcam_val_per_class, args.seed,
    )
    print(f"[finetune] webcam split: train={len(webcam_train)}  val={len(webcam_val)}")

    train_parts: list[Dataset] = [webcam_train]
    if args.anchor_per_class > 0:
        anchor = _build_ddd_anchor(args.db, args.anchor_per_class, args.seed)
        print(f"[finetune] DDD anchor: {len(anchor)} samples "
              f"({args.anchor_per_class} per class)")
        train_parts.append(anchor)
    train_ds: Dataset = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        webcam_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model -------------------------------------------------------------
    model = build_model("mobilenet_v2", pretrained=False, freeze_backbone=False).to(device)
    state = torch.load(src_ckpt, map_location=device)["state_dict"]
    model.load_state_dict(state)
    print(f"[finetune] loaded starting weights from {src_ckpt}")

    _unfreeze_last_blocks(model, unfreeze_from=args.unfreeze_from)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] params: {trainable:,} trainable / {total:,} total "
          f"(unfreeze_from features[{args.unfreeze_from}])")

    # No pos_weight here — the webcam set is balanced by construction (50/50)
    # and the DDD anchor is balanced by sampling, so plain BCE is appropriate.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Train loop --------------------------------------------------------
    ckpt_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, criterion, device,
                             optimizer=optimizer,
                             desc=f"epoch {epoch:02d}/{args.epochs} train")
        val_m = _run_epoch(model, val_loader, criterion, device,
                           desc=f"epoch {epoch:02d}/{args.epochs}   val")
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"train loss={train_m.loss:.4f} acc={train_m.acc:.3f} f1={train_m.f1:.3f} | "
            f"webcam-val loss={val_m.loss:.4f} acc={val_m.acc:.3f} "
            f"f1={val_m.f1:.3f} mF1={val_m.macro_f1:.3f} | {dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": train_m.as_dict(),
            "webcam_val": val_m.as_dict(),
        })

        if val_m.macro_f1 > best_f1:
            best_f1 = val_m.macro_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_name": "mobilenet_v2",
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "args": vars(args),
                "source_checkpoint": str(src_ckpt),
            }, ckpt_path)
            print(f"[finetune]   ↳ new best webcam-val macro-F1 ({best_f1:.3f}) → saved {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[finetune] early stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch}, mF1={best_f1:.3f})")
                break

    history_path.write_text(json.dumps({
        "source_checkpoint": str(src_ckpt),
        "best_epoch": best_epoch,
        "best_webcam_val_macro_f1": best_f1,
        "args": vars(args),
        "history": history,
    }, indent=2))
    print(f"[finetune] history → {history_path}")

    # ---- Forgetting check: re-eval best ckpt on held-out DDD test ----------
    if best_epoch > 0:
        print(f"\n[finetune] forgetting check: evaluating {ckpt_path} on DDD test split")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
        ddd_test_ds = FaceStreamDataset(args.db, split="test")
        ddd_test_loader = DataLoader(
            ddd_test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        ddd_m = _run_epoch(model, ddd_test_loader, criterion, device,
                           desc="DDD test (post-finetune)")
        ddd_metrics = ddd_m.as_dict()
        print(f"[finetune] DDD test (post-finetune): "
              f"acc={ddd_metrics['acc']:.3f}  mF1={ddd_metrics['macro_f1']:.3f}  "
              f"(original mobilenet_v2 test mF1 was 0.705)")
        (out_dir / "ddd_test_metrics.json").write_text(
            json.dumps(ddd_metrics, indent=2)
        )
        print(f"[finetune] DDD test metrics → {out_dir / 'ddd_test_metrics.json'}")

    print(f"\n[finetune] done. live-demo:\n"
          f"    py -m src.realtime_demo --face-ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
