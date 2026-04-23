"""Single-stream (face-only) view of the unified SQLite bundle.

The multi-stream SQLiteDrowsinessDataset returns a dict with eye + face
tensors and per-sample masks. For the three comparable single-stream models
(BaselineCNN, AlexNet, MobileNetV2) we want a simpler (image, label) tuple
and only the DDD face frames — MRL eye crops belong in a separate eye-state
classifier (future work).

This wrapper filters the bundle to ``stream == "face"`` samples and exposes
the same (image, label) shape any standard classification Dataset does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import SQLiteDrowsinessDataset, Sample


class FaceStreamDataset(Dataset):
    """DDD face frames only. Returns ``(Tensor[3,224,224], Tensor[1])``.

    Keeps ``.samples`` compatible with :func:`compute_pos_weight` and
    :func:`make_weighted_sampler`.
    """

    def __init__(
        self,
        db_path: str | Path,
        split: Literal["train", "val", "test"],
        augment: bool = False,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._base = SQLiteDrowsinessDataset(
            db_path, split=split, augment=augment, augment_fn=augment_fn,
        )
        self._indices = [
            i for i, s in enumerate(self._base.samples) if s.stream == "face"
        ]
        if not self._indices:
            raise ValueError(
                f"No face-stream samples in split={split!r}. Make sure the "
                f"bundle contains DDD samples (check source_counts in meta)."
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._base[self._indices[idx]]
        # Face is (3, 224, 224). Label is a 0-d float tensor; return as (1,)
        # so DataLoader stacking yields (B, 1) to match our (B, 1) logits.
        return out["face"], out["label"].unsqueeze(-1)

    @property
    def samples(self) -> list[Sample]:
        """Filtered Sample records — for compute_pos_weight / sampler."""
        return [self._base.samples[i] for i in self._indices]
