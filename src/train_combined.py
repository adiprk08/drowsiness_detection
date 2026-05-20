"""Train a single-stream face model on the combined DDD + UTA-RLDD distribution.

Why this exists
---------------
The original ``src.train`` trains on the SQLite bundle (DDD face stream
only, for the face-CNN models). That model saturates at deployment on a
laptop webcam because the cabin-camera training distribution doesn't
match. ``src/uta_rldd.py`` extracts UTA-RLDD — self-recorded webcam /
phone subjects — to face JPEGs on disk. This script trains any of the
three single-stream architectures (baseline_cnn / alexnet /
mobilenet_v2) on the union of:

  - the DDD face stream from ``data/drowsiness.db``  (existing pipeline)
  - the UTA face crops from ``data/uta_rldd_frames`` (new, on disk)

both wrapped in PyTorch datasets that yield the same
``(Tensor[3, 224, 224], Tensor[1])`` shape, so a single ``ConcatDataset``
is enough.

What it does
------------
1. Splits UTA subject-disjointly into train / val / test (DDD already is).
2. Concatenates train sets; evaluates val and test separately on DDD,
   UTA, and combined, so we can see per-domain performance.
3. Trains with the same loop as ``src.train`` (BCE-with-logits,
   weighted-random sampler, AdamW + cosine, early stopping on combined
   val macro-F1).
4. Saves to ``artifacts/<model>_combined/`` with full per-domain
   metrics in ``history.json`` and ``test_metrics.json``.

Usage
-----
    # primary deployment models — one per architecture, on the combined set
    py -m src.train_combined --model baseline_cnn --epochs 15
    py -m src.train_combined --model alexnet      --epochs 10
    py -m src.train_combined --model mobilenet_v2 --epochs 10

    # ablations (out-name auto-suffixes to <model>_uta_only / <model>_ddd_only)
    py -m src.train_combined --model mobilenet_v2 --uta-only
    py -m src.train_combined --model mobilenet_v2 --ddd-only

    py -m src.train_combined --model mobilenet_v2 --no-augment
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .data_single_stream import FaceStreamDataset
from .datasets import Sample
from .models import build_model
from .train import _run_epoch
from .uta_rldd import UtaRldDataset, split_uta_subjects


# ---------------------------------------------------------------------------
# Weighted sampler over a combined set, using a flat label list
# ---------------------------------------------------------------------------

def _build_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    arr = np.array(labels)
    class_counts = np.bincount(arr, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[arr]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


def _pos_weight(labels: list[int]) -> float:
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    return (n_neg / n_pos) if n_pos else 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a single-stream face model on the combined "
                    "DDD + UTA-RLDD distribution.",
    )
    p.add_argument("--model", default="mobilenet_v2",
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"],
                   help="Which single-stream architecture to train.")
    p.add_argument("--db", default="data/drowsiness.db",
                   help="Path to the DDD-containing SQLite bundle.")
    p.add_argument("--uta-frames", default="data/uta_rldd_frames",
                   help="Path to the on-disk UTA face-crop tree (output of `src.uta_rldd extract`).")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out-name", default=None,
                   help="Artifacts subfolder name. Defaults to '<model>_combined' "
                        "(or '<model>_uta_only' / '<model>_ddd_only' for the ablations).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stopping patience on combined-val macro-F1.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-freeze", action="store_true",
                   help="Fine-tune the whole MobileNetV2 backbone.")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddd-only", action="store_true",
                   help="Ablation: ignore UTA, train on DDD alone (matches src.train).")
    p.add_argument("--uta-only", action="store_true",
                   help="Ablation: ignore DDD, train on UTA alone.")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="UTA val fraction (DDD splits come from the SQLite bundle and are fixed).")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="UTA test fraction.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.ddd_only and args.uta_only:
        raise SystemExit("--ddd-only and --uta-only are mutually exclusive")

    # Default the artifacts subfolder name from the model + ablation mode.
    if args.out_name is None:
        suffix = ("ddd_only" if args.ddd_only
                  else "uta_only" if args.uta_only
                  else "combined")
        args.out_name = f"{args.model}_{suffix}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_combined] device={device}  model={args.model}  out={args.out_name}")

    augment_fn = None
    if not args.no_augment:
        from .augmentations import AugPipeline
        augment_fn = AugPipeline()

    # ---- DDD via SQLite ----------------------------------------------------
    ddd_train = ddd_val = ddd_test = None
    if not args.uta_only:
        ddd_train = FaceStreamDataset(args.db, split="train",
                                      augment=True, augment_fn=augment_fn)
        ddd_val = FaceStreamDataset(args.db, split="val")
        ddd_test = FaceStreamDataset(args.db, split="test")
        print(f"[train_combined] DDD: train={len(ddd_train)} "
              f"val={len(ddd_val)} test={len(ddd_test)}")

    # ---- UTA via disk ------------------------------------------------------
    uta_train = uta_val = uta_test = None
    if not args.ddd_only:
        train_subjects, val_subjects, test_subjects = split_uta_subjects(
            args.uta_frames, val_frac=args.val_frac,
            test_frac=args.test_frac, seed=args.seed,
        )
        uta_train = UtaRldDataset(args.uta_frames, subjects=train_subjects,
                                  augment_fn=augment_fn)
        uta_val = UtaRldDataset(args.uta_frames, subjects=val_subjects)
        uta_test = UtaRldDataset(args.uta_frames, subjects=test_subjects)
        print(f"[train_combined] UTA: train={len(uta_train)} "
              f"val={len(uta_val)} test={len(uta_test)} "
              f"(train/val/test subjects: "
              f"{len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)})")

    # ---- Combined train + per-domain val/test ------------------------------
    train_parts = [d for d in (ddd_train, uta_train) if d is not None]
    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    # Sampler needs a flat label list across the combined train set.
    train_labels: list[int] = []
    if ddd_train is not None:
        train_labels.extend(int(s.label) for s in ddd_train.samples)
    if uta_train is not None:
        train_labels.extend(int(s.label) for s in uta_train.samples)
    sampler = _build_weighted_sampler(train_labels)
    pos_weight_val = _pos_weight(train_labels)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"[train_combined] combined train: {len(train_labels)} samples  "
          f"pos_weight={pos_weight_val:.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Build evaluation loaders for each domain we have, plus combined.
    def _loader(ds, shuffle=False):
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    val_loaders: dict[str, DataLoader] = {}
    test_loaders: dict[str, DataLoader] = {}
    if ddd_val is not None: val_loaders["ddd"] = _loader(ddd_val)
    if uta_val is not None: val_loaders["uta"] = _loader(uta_val)
    if len(val_loaders) > 1:
        val_loaders["combined"] = _loader(ConcatDataset(
            [d for d in (ddd_val, uta_val) if d is not None]
        ))
    if ddd_test is not None: test_loaders["ddd"] = _loader(ddd_test)
    if uta_test is not None: test_loaders["uta"] = _loader(uta_test)
    if len(test_loaders) > 1:
        test_loaders["combined"] = _loader(ConcatDataset(
            [d for d in (ddd_test, uta_test) if d is not None]
        ))

    # ---- Model -------------------------------------------------------------
    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train_combined] params: {trainable:,} trainable / {total:,} total")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.artifacts) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"

    # Which val loader drives early stopping? Prefer combined; otherwise
    # whichever single domain we have.
    selection_key = ("combined" if "combined" in val_loaders
                     else next(iter(val_loaders)))
    print(f"[train_combined] early stopping on val/{selection_key} macro-F1")

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, criterion, device,
                             optimizer=optimizer,
                             desc=f"epoch {epoch:02d}/{args.epochs} train")
        val_metrics: dict[str, dict] = {}
        for name, loader in val_loaders.items():
            val_m = _run_epoch(model, loader, criterion, device,
                               desc=f"epoch {epoch:02d}/{args.epochs} val/{name}")
            val_metrics[name] = val_m.as_dict()
        scheduler.step()
        dt = time.time() - t0

        sel_m = val_metrics[selection_key]
        msg = (f"[epoch {epoch:02d}/{args.epochs}] "
               f"train loss={train_m.loss:.4f} f1={train_m.f1:.3f} | ")
        for name, m in val_metrics.items():
            msg += (f"val/{name} mF1={m['macro_f1']:.3f} "
                    f"acc={m['acc']:.3f} | ")
        msg += f"lr={scheduler.get_last_lr()[0]:.2e}  {dt:.1f}s"
        print(msg)

        history.append({
            "epoch": epoch,
            "train": train_m.as_dict(),
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        })

        if sel_m["macro_f1"] > best_f1:
            best_f1 = sel_m["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "selection_key": selection_key,
                "args": vars(args),
            }, ckpt_path)
            print(f"[train_combined]   ↳ new best val/{selection_key} "
                  f"mF1={best_f1:.3f} → saved {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train_combined] early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, mF1={best_f1:.3f})")
                break

    # ---- Final test evaluation on the best checkpoint ----------------------
    print(f"\n[train_combined] reloading best ({ckpt_path}) for test evaluation")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_metrics: dict[str, dict] = {}
    for name, loader in test_loaders.items():
        m = _run_epoch(model, loader, criterion, device,
                       desc=f"test/{name}")
        # Carry the sample count so src.compare can show a meaningful N.
        test_metrics[name] = {**m.as_dict(), "n": len(loader.dataset)}
        print(f"[train_combined] test/{name}: "
              f"n={len(loader.dataset)}  acc={m.acc:.3f}  "
              f"mF1={m.macro_f1:.3f}  f1={m.f1:.3f}")

    history_path.write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "selection_key": selection_key,
        "args": vars(args),
        "history": history,
        "test_metrics": test_metrics,
    }, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"[train_combined] history → {history_path}")
    print(f"[train_combined] test metrics → {out_dir / 'test_metrics.json'}")

    print(f"\n[train_combined] done. live demo:\n"
          f"    py -m src.realtime_demo --face-ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
