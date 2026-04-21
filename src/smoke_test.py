"""
smoke_test.py
-------------
Verifies the data-integration pipeline works end-to-end without needing
the actual datasets downloaded. Creates a tiny synthetic MRL-style and
DDD-style directory tree, runs the loader, and prints diagnostics.

Run:
    python -m src.smoke_test
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import cv2
from torch.utils.data import DataLoader

from .datasets import build_datasets, make_weighted_sampler
from .augmentations import AugPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")


def _make_synthetic_mrl(root: Path, n_subjects: int = 5, per_subject: int = 20):
    """MRL filenames: sNNNN_MMMMM_G_g_e_r_l_.png"""
    rng = np.random.default_rng(0)
    for subj in range(1, n_subjects + 1):
        for i in range(per_subject):
            state = int(rng.integers(0, 2))     # 0=closed, 1=open
            gender = int(rng.integers(0, 2))
            glasses = int(rng.integers(0, 2))
            reflect = int(rng.integers(0, 3))
            light = int(rng.integers(0, 2))
            fname = f"s{subj:04d}_{i:05d}_{gender}_{glasses}_{state}_{light}_{reflect}_.png"
            # MRL-like small grayscale eye crop, varied size
            size = int(rng.integers(40, 120))
            img = rng.integers(0, 256, (size, size), dtype=np.uint8)
            cv2.imwrite(str(root / fname), img)


def _make_synthetic_ddd(root: Path, per_class: int = 25):
    rng = np.random.default_rng(1)
    for cls in ["yawn", "no_yawn", "Closed", "Open"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # Face-size-ish color image, varied aspect
            h = int(rng.integers(120, 360))
            w = int(rng.integers(120, 360))
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{cls.lower()}_{i:03d}.jpg"), img)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="drowsy_smoke_"))
    mrl_root = tmp / "mrl"
    ddd_root = tmp / "ddd"
    mrl_root.mkdir()
    ddd_root.mkdir()

    try:
        log.info("Creating synthetic MRL at %s", mrl_root)
        _make_synthetic_mrl(mrl_root)
        log.info("Creating synthetic DDD at %s", ddd_root)
        _make_synthetic_ddd(ddd_root)

        aug = AugPipeline(seed=0)
        train_ds, val_ds, test_ds, info = build_datasets(
            mrl_root=mrl_root,
            ddd_root=ddd_root,
            augment_fn=aug,
            val_frac=0.2,
            test_frac=0.2,
        )

        log.info("Source counts:         %s", info["source_counts"])
        log.info("Split label counts:    %s", info["counts"])
        log.info("pos_weight (train):    %.3f", info["pos_weight"])

        # Try one batch with weighted sampling
        sampler = make_weighted_sampler(train_ds.samples)
        loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
        batch = next(iter(loader))

        log.info("Batch shapes:")
        log.info("  eye:       %s", tuple(batch["eye"].shape))
        log.info("  face:      %s", tuple(batch["face"].shape))
        log.info("  eye_mask:  %s", batch["eye_mask"].tolist())
        log.info("  face_mask: %s", batch["face_mask"].tolist())
        log.info("  labels:    %s", batch["label"].tolist())
        log.info("  sources:   %s", batch["source"])

        # Sanity assertions — exactly one stream active per sample
        both_on = ((batch["eye_mask"] > 0) & (batch["face_mask"] > 0)).sum().item()
        both_off = ((batch["eye_mask"] == 0) & (batch["face_mask"] == 0)).sum().item()
        assert both_on == 0, "A sample activated both streams — should be impossible"
        assert both_off == 0, "A sample activated no stream — loader bug"

        log.info("✅ Smoke test passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
