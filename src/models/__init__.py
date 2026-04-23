"""Single-stream face-classification models.

Three architectures we compare head-to-head on the DDD face frames:

    - ``baseline_cnn``  — small from-scratch CNN (Model A). The "floor".
    - ``alexnet``       — torchvision AlexNet pretrained on ImageNet,
                          classifier head replaced (Model B). Classical TL.
    - ``mobilenet_v2``  — torchvision MobileNetV2 pretrained on ImageNet,
                          classifier head replaced (Model C). Real-time
                          target for webcam inference.

Each model takes a ``Tensor[B, 3, 224, 224]`` and returns raw **logits** of
shape ``Tensor[B, 1]`` — we train with ``BCEWithLogitsLoss`` so the sigmoid
lives in the loss, not the model. Evaluation applies sigmoid at report time.

Use :func:`build_model` as the single entry point so ``train.py`` / ``eval.py``
don't need to know the concrete class for each ``--model`` flag.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .baseline_cnn import BaselineCNN
from .alexnet_tl import build_alexnet
from .mobilenet_v2 import build_mobilenet_v2

ModelName = Literal["baseline_cnn", "alexnet", "mobilenet_v2"]

__all__ = [
    "BaselineCNN",
    "build_alexnet",
    "build_mobilenet_v2",
    "build_model",
    "ModelName",
]


def build_model(name: ModelName, *, pretrained: bool = True,
                freeze_backbone: bool = True) -> nn.Module:
    """Factory. ``pretrained`` / ``freeze_backbone`` are ignored for the
    from-scratch baseline."""
    name = name.lower()
    if name == "baseline_cnn":
        return BaselineCNN()
    if name == "alexnet":
        return build_alexnet(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if name == "mobilenet_v2":
        return build_mobilenet_v2(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(
        f"Unknown model {name!r}. Choose from: baseline_cnn, alexnet, mobilenet_v2."
    )
