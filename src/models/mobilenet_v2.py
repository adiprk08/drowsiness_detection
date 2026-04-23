"""Model C — MobileNetV2 transfer learning (real-time target).

MobileNetV2 is the deployment target: ~3.5 M parameters, ~300 MFLOPs per
224×224 forward pass, which is small enough to run in real time on a
laptop CPU alongside a webcam capture loop.

Strategy
--------
- ImageNet-pretrained weights.
- ``freeze_backbone=True`` (default) freezes the stem + most of the
  inverted-residual blocks (``features[:14]``) and fine-tunes the last
  few blocks + classifier. That keeps compute low while letting the
  high-level features adapt to cabin-camera faces (illumination, pose,
  motion blur) which differ from ImageNet.
- Set ``freeze_backbone=False`` to fine-tune end-to-end.

Input / output
--------------
    in : Tensor[B, 3, 224, 224]  (ImageNet-normalised in the training loop)
    out: Tensor[B, 1]            (raw logits)
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


# MobileNetV2 has 19 blocks in features[]. Freezing the first 14 leaves
# the last 5 inverted-residuals + final 1×1 conv trainable — a good
# "feature adaptation" depth for a ~40k-image downstream dataset.
_FREEZE_UP_TO = 14


def build_mobilenet_v2(*, pretrained: bool = True, freeze_backbone: bool = True,
                       num_classes: int = 1) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for i, block in enumerate(net.features):
            if i < _FREEZE_UP_TO:
                for p in block.parameters():
                    p.requires_grad = False

    # Original head: Dropout(0.2) → Linear(1280, 1000)
    # Replace with binary head. Keep it simple — MobileNetV2's features
    # are strong enough that a linear probe usually suffices.
    in_features = net.classifier[-1].in_features  # 1280
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return net
