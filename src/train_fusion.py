"""Two-stream (eye + face) training with masked BCE loss.

This is the second training entry point in the project (the first is
``src/train.py`` for single-stream face models). It trains
:class:`TwoStreamModel` — an ``EyeStateCNN`` + face-MobileNetV2 pair —
on the **full** SQLite bundle (MRL eye crops + DDD face crops, no
filtering).

Loss
----
Each batch item comes with ``eye_mask`` / ``face_mask`` (one is 0 and
the other is 1 in our data — no paired samples). We compute per-branch
BCE logits and mask the per-sample contribution, then normalise by the
number of valid samples per branch:

    L = (Σ_i eye_mask[i]  * BCE(eye_logit[i],  y[i])) / Σ eye_mask
      + (Σ_i face_mask[i] * BCE(face_logit[i], y[i])) / Σ face_mask

That way a batch dominated by face samples doesn't crowd out the tiny
eye-branch signal (or vice-versa).

What we report per epoch
------------------------
- Per-branch BCE loss
- Per-branch macro-F1 (computed only over the samples that had that
  stream's mask=1 — so the eye-branch F1 is MRL-only, face-branch F1 is
  DDD-only)
- Combined best metric for early stopping — the mean of the two macro-F1s,
  which keeps both branches honest during training.

Usage
-----
    py -m src.train_fusion --epochs 10 --num-workers 0 --batch-size 128
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import (
    SQLiteDrowsinessDataset, compute_pos_weight, make_weighted_sampler,
)
from .models import build_model


# ---------------------------------------------------------------------------
# Per-branch metrics
# ---------------------------------------------------------------------------

@dataclass
class BranchMetrics:
    n: int
    loss: float
    acc: float
    f1: float
    macro_f1: float

    def as_dict(self) -> dict:
        return self.__dict__


def _branch_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor, loss_sum: float) -> BranchMetrics:
    """Metrics over the subset where ``mask == 1``."""
    sel = mask.view(-1) > 0.5
    n = int(sel.sum().item())
    if n == 0:
        return BranchMetrics(n=0, loss=0.0, acc=0.0, f1=0.0, macro_f1=0.0)
    logits = logits.view(-1)[sel]
    y = labels.view(-1)[sel].long()
    preds = (torch.sigmoid(logits) >= 0.5).long()

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    acc = (tp + tn) / n

    def _f1(tp_: int, fp_: int, fn_: int) -> float:
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    pos_f1 = _f1(tp, fp, fn)
    neg_f1 = _f1(tn, fn, fp)
    macro_f1 = 0.5 * (pos_f1 + neg_f1)
    return BranchMetrics(n=n, loss=loss_sum / n, acc=acc,
                         f1=pos_f1, macro_f1=macro_f1)


# ---------------------------------------------------------------------------
# Masked BCE — one per branch
# ---------------------------------------------------------------------------

def _masked_bce(logit: torch.Tensor, label: torch.Tensor, mask: torch.Tensor,
                pos_weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Returns (mean_loss_for_masked_samples, sum_loss_for_metrics)."""
    # BCE with logits, per-sample, no reduction.
    per = F.binary_cross_entropy_with_logits(
        logit.view(-1), label.view(-1),
        pos_weight=pos_weight, reduction="none",
    )
    m = mask.view(-1)
    n = m.sum()
    loss_sum = (per * m).sum()
    if n.item() == 0:
        return torch.tensor(0.0, device=logit.device, requires_grad=True), 0.0
    return loss_sum / n, float(loss_sum.detach().item())


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------

def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
               eye_pw: torch.Tensor, face_pw: torch.Tensor, *,
               optimizer: torch.optim.Optimizer | None = None,
               desc: str = "") -> tuple[BranchMetrics, BranchMetrics]:
    training = optimizer is not None
    model.train(training)

    all_eye_logits, all_face_logits, all_labels = [], [], []
    all_eye_masks, all_face_masks = [], []
    eye_loss_sum, face_loss_sum = 0.0, 0.0

    pbar = tqdm(loader, desc=desc or ("train" if training else "val"),
                leave=False, dynamic_ncols=True)
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for batch in pbar:
            eye = batch["eye"].to(device, non_blocking=True)
            face = batch["face"].to(device, non_blocking=True)
            em = batch["eye_mask"].to(device, non_blocking=True)
            fm = batch["face_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float()

            out = model(eye, face)
            e_loss, e_sum = _masked_bce(out["eye_logit"], y, em, eye_pw)
            f_loss, f_sum = _masked_bce(out["face_logit"], y, fm, face_pw)
            loss = e_loss + f_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            eye_loss_sum += e_sum
            face_loss_sum += f_sum
            all_eye_logits.append(out["eye_logit"].detach().cpu())
            all_face_logits.append(out["face_logit"].detach().cpu())
            all_labels.append(y.detach().cpu())
            all_eye_masks.append(em.detach().cpu())
            all_face_masks.append(fm.detach().cpu())

            pbar.set_postfix(
                e_loss=f"{e_loss.item():.3f}",
                f_loss=f"{f_loss.item():.3f}",
            )
    pbar.close()

    eye_logits = torch.cat(all_eye_logits)
    face_logits = torch.cat(all_face_logits)
    labels = torch.cat(all_labels)
    eye_masks = torch.cat(all_eye_masks)
    face_masks = torch.cat(all_face_masks)

    eye_m = _branch_metrics(eye_logits, labels, eye_masks, eye_loss_sum)
    face_m = _branch_metrics(face_logits, labels, face_masks, face_loss_sum)
    return eye_m, face_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the two-stream fusion model.")
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _split_pos_weights(samples) -> tuple[float, float]:
    """Compute per-stream pos_weight for BCE. Each stream has its own class
    imbalance — MRL is roughly balanced, DDD is not."""
    eye = [s for s in samples if s.stream == "eye"]
    face = [s for s in samples if s.stream == "face"]
    return compute_pos_weight(eye), compute_pos_weight(face)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fusion] device={device}")

    augment_fn = None
    if not args.no_augment:
        from .augmentations import AugPipeline
        augment_fn = AugPipeline()

    train_ds = SQLiteDrowsinessDataset(
        args.db, split="train", augment=True, augment_fn=augment_fn,
    )
    val_ds = SQLiteDrowsinessDataset(args.db, split="val")

    eye_count = sum(1 for s in train_ds.samples if s.stream == "eye")
    face_count = len(train_ds.samples) - eye_count
    print(f"[fusion] train: {len(train_ds)} total ({eye_count} eye, {face_count} face)")
    print(f"[fusion] val  : {len(val_ds)} total")

    eye_pw_val, face_pw_val = _split_pos_weights(train_ds.samples)
    eye_pw = torch.tensor([eye_pw_val], dtype=torch.float32, device=device)
    face_pw = torch.tensor([face_pw_val], dtype=torch.float32, device=device)
    print(f"[fusion] pos_weight — eye={eye_pw_val:.3f}, face={face_pw_val:.3f}")

    sampler = make_weighted_sampler(train_ds.samples)
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
        "two_stream",
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[fusion] params: {trainable:,} trainable / {total:,} total")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.artifacts) / "two_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"

    history: list[dict] = []
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_eye, tr_face = _run_epoch(
            model, train_loader, device, eye_pw, face_pw,
            optimizer=optimizer,
            desc=f"epoch {epoch:02d}/{args.epochs} train",
        )
        va_eye, va_face = _run_epoch(
            model, val_loader, device, eye_pw, face_pw,
            desc=f"epoch {epoch:02d}/{args.epochs}   val",
        )
        scheduler.step()
        dt = time.time() - t0

        # Early-stop criterion: mean of the two val macro-F1s so one branch
        # can't mask the other's collapse.
        combined = 0.5 * (va_eye.macro_f1 + va_face.macro_f1)

        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"train eye mF1={tr_eye.macro_f1:.3f} face mF1={tr_face.macro_f1:.3f} | "
            f"val eye mF1={va_eye.macro_f1:.3f} face mF1={va_face.macro_f1:.3f} "
            f"combined={combined:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": {"eye": tr_eye.as_dict(), "face": tr_face.as_dict()},
            "val":   {"eye": va_eye.as_dict(), "face": va_face.as_dict()},
            "val_combined_macro_f1": combined,
            "lr": scheduler.get_last_lr()[0],
        })

        if combined > best_score:
            best_score = combined
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_name": "two_stream",
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_combined_macro_f1": best_score,
                "args": vars(args),
            }, ckpt_path)
            print(f"[fusion]   ↳ new best combined ({best_score:.3f}) → saved {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[fusion] early stopping at epoch {epoch} "
                      f"(best was {best_epoch}, combined={best_score:.3f})")
                break

    history_path.write_text(json.dumps({
        "model": "two_stream",
        "best_epoch": best_epoch,
        "best_val_combined_macro_f1": best_score,
        "args": vars(args),
        "history": history,
    }, indent=2))
    print(f"[fusion] history → {history_path}")
    print(f"[fusion] best   → {ckpt_path}  (epoch {best_epoch}, combined={best_score:.3f})")


if __name__ == "__main__":
    main()
