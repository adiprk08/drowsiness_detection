"""Model A — small from-scratch CNN baseline.

Purpose
-------
Establishes a "floor" accuracy to compare against the pretrained models.
If AlexNet / MobileNetV2 can't beat this, transfer learning isn't helping
and something is wrong with the training setup.

Architecture (224×224 RGB → 1 logit)
------------------------------------
    Block 1:  Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)      # 112×112
    Block 2:  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)     #  56×56
    Block 3:  Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)    #  28×28
    Block 4:  Conv(128→256, 3×3) → BN → ReLU → MaxPool(2)   #  14×14
    Head:     AdaptiveAvgPool(1) → Flatten
              → Dropout(0.5) → Linear(256→128) → ReLU
              → Dropout(0.3) → Linear(128→1)

~1.1 M parameters — deliberately small so the comparison with the 3–60 M
pretrained nets is meaningful (capacity vs. transferred features).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """4-block CNN → GAP → MLP → 1 logit."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
