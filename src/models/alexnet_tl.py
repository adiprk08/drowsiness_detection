"""Model B — AlexNet transfer learning.

We take torchvision's ImageNet-pretrained AlexNet, freeze the conv feature
extractor, and replace the 1000-way classifier with a 2-layer MLP that
outputs a single logit for the binary {alert, drowsy} task.

Why AlexNet here
----------------
It's the classical "first deep CNN" — a standard comparison point in
coursework-style ML projects. Compared to the from-scratch baseline we
keep ImageNet low-level features (edges, textures) instead of re-learning
them from ~40k face frames.

Input / output
--------------
    in : Tensor[B, 3, 224, 224]  (ImageNet-normalised in the training loop)
    out: Tensor[B, 1]            (raw logits)

Freezing
--------
``freeze_backbone=True`` (default) locks ``features`` and ``avgpool`` so
only the new classifier trains. Pass ``False`` to fine-tune the whole
network — slower, more overfitting risk, but sometimes a small lift if
you have enough data / epochs.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_alexnet(*, pretrained: bool = True, freeze_backbone: bool = True,
                  num_classes: int = 1) -> nn.Module:
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.alexnet(weights=weights)

    if freeze_backbone:
        for p in net.features.parameters():
            p.requires_grad = False
        # avgpool has no parameters, no need to freeze

    # Original head: Dropout → Linear(9216, 4096) → ReLU → Dropout
    #              → Linear(4096, 4096) → ReLU → Linear(4096, 1000)
    # Replace with a slimmer head suited to binary classification.
    net.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(256 * 6 * 6, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )
    return net
