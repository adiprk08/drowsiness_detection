"""Unified training script for the three single-stream face models.

Usage
-----
    py -m src.train --model baseline_cnn --epochs 15
    py -m src.train --model alexnet       --epochs 10
    py -m src.train --model mobilenet_v2  --epochs 10

What this does
--------------
1. Loads the face-only view of the shared SQLite bundle
   (``data/drowsiness.db``) via :class:`FaceStreamDataset`.
2. Builds the requested model via :func:`src.models.build_model`.
3. Trains with:
     - ``BCEWithLogitsLoss(pos_weight=N_neg/N_pos)``
     - ``WeightedRandomSampler`` on the training loader
     - AdamW + cosine LR schedule
     - Early stopping on val **macro-F1** (patience configurable)
4. Checkpoints the best epoch to ``artifacts/<model>/best.pt`` along with
   the training-run metadata (hparams + per-epoch metrics) in
   ``artifacts/<model>/history.json``.

Notes
-----
- The SQLite dataset already applies ImageNet mean/std normalisation, so
  pretrained weights see the input distribution they expect without any
  extra transform in this script.
- Labels come out as ``Tensor[B, 1]`` from :class:`FaceStreamDataset`, so
  the BCE loss and the ``Tensor[B, 1]`` model logits line up directly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_single_stream import FaceStreamDataset
from .datasets import compute_pos_weight, make_weighted_sampler
from .models import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BinaryMetrics:
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float
    macro_f1: float

    def as_dict(self) -> dict:
        return self.__dict__


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    loss: float) -> BinaryMetrics:
    """Compute per-batch-aggregated binary metrics. Positive class = drowsy."""
    preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    y = labels.long().view(-1)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    # per-class F1 (positive = drowsy, negative = alert) then macro-avg
    def _f1(tp_: int, fp_: int, fn_: int) -> float:
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    pos_p = tp / (tp + fp) if (tp + fp) else 0.0
    pos_r = tp / (tp + fn) if (tp + fn) else 0.0
    pos_f1 = _f1(tp, fp, fn)
    neg_f1 = _f1(tn, fn, fp)
    macro_f1 = 0.5 * (pos_f1 + neg_f1)

    return BinaryMetrics(
        loss=loss, acc=acc, precision=pos_p, recall=pos_r,
        f1=pos_f1, macro_f1=macro_f1,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               device: torch.device, *,
               optimizer: torch.optim.Optimizer | None = None,
               desc: str = "") -> BinaryMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_n = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    pbar = tqdm(loader, desc=desc or ("train" if training else "val"),
                leave=False, dynamic_ncols=True)

    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Live rolling-average loss + accuracy on the progress bar.
            running_acc = (
                (torch.sigmoid(torch.cat(all_logits)) >= 0.5).long().view(-1)
                == torch.cat(all_labels).long().view(-1)
            ).float().mean().item()
            pbar.set_postfix(loss=f"{total_loss / total_n:.4f}",
                             acc=f"{running_acc:.3f}")

    pbar.close()
    mean_loss = total_loss / max(total_n, 1)
    return _binary_metrics(torch.cat(all_logits), torch.cat(all_labels), mean_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single-stream face model.")
    p.add_argument("--model", required=True,
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"])
    p.add_argument("--db", default="data/drowsiness.db",
                   help="Path to the combined SQLite bundle.")
    p.add_argument("--artifacts", default="artifacts",
                   help="Directory to write checkpoints + history under.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stopping patience on val macro-F1.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable ImageNet weights for alexnet / mobilenet_v2.")
    p.add_argument("--no-freeze", action="store_true",
                   help="Fine-tune the whole backbone (for the TL models).")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable train-time augmentation.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  model={args.model}")

    # Augmentation pipeline is lazy-imported so CPU-only runs don't pay for
    # OpenCV imports twice.
    augment_fn = None
    if not args.no_augment:
        from .augmentations import AugPipeline
        augment_fn = AugPipeline()

    train_ds = FaceStreamDataset(args.db, split="train",
                                 augment=True, augment_fn=augment_fn)
    val_ds = FaceStreamDataset(args.db, split="val")
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    sampler = make_weighted_sampler(train_ds.samples)
    pos_weight = torch.tensor([compute_pos_weight(train_ds.samples)],
                              dtype=torch.float32, device=device)
    print(f"[train] pos_weight={pos_weight.item():.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {trainable:,} trainable / {total:,} total")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.artifacts) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
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
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"train loss={train_m.loss:.4f} acc={train_m.acc:.3f} f1={train_m.f1:.3f} | "
            f"val loss={val_m.loss:.4f} acc={val_m.acc:.3f} "
            f"f1={val_m.f1:.3f} mF1={val_m.macro_f1:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": train_m.as_dict(),
            "val": val_m.as_dict(),
            "lr": scheduler.get_last_lr()[0],
        })

        if val_m.macro_f1 > best_f1:
            best_f1 = val_m.macro_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[train]   ↳ new best val macro-F1 ({best_f1:.3f}) → saved {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch}, mF1={best_f1:.3f})")
                break

    history_path.write_text(json.dumps({
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "args": vars(args),
        "history": history,
    }, indent=2))
    print(f"[train] history written → {history_path}")
    print(f"[train] best checkpoint → {ckpt_path}  (epoch {best_epoch}, mF1={best_f1:.3f})")


if __name__ == "__main__":
    main()
