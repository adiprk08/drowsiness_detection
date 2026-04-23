"""Two-stream (eye + face) fusion model.

Architecture
------------
Two independent branches with no weight sharing — eye and face images are
visually different enough (tight 64×64 grayscale crops vs full 224×224
colour faces) that a shared backbone is unlikely to help.

    eye  (B, 3,  64,  64)  →  EyeStateCNN   →  logit_eye   (B, 1)
    face (B, 3, 224, 224)  →  MobileNetV2   →  logit_face  (B, 1)

At inference the "combined" prediction averages the two sigmoid outputs
when both streams are present (which, in our training data, never
happens — MRL is eye-only, DDD is face-only). At deployment, a face
detector (MediaPipe) would produce *both* crops from a single webcam
frame, which is where fusion becomes meaningful.

Training strategy (implemented in ``src/train_fusion.py``)
---------------------------------------------------------
Each sample carries an ``eye_mask`` and ``face_mask`` indicating which
stream has valid data. The loss is the mask-weighted sum of per-branch
BCE losses. For an MRL eye sample the face branch's loss contribution is
zero (mask=0) so gradients only flow through the eye branch — and vice
versa for DDD. The two branches therefore effectively train in parallel
on disjoint subject pools, sharing only the optimiser's batch schedule.

What this buys us
-----------------
- ~37 new subjects from MRL that the face-only models never saw.
- A subject-independent signal (eye-state) that complements the face
  appearance signal.
- A single model that answers both "is this eye closed?" and "does this
  face look drowsy?", which is what a real cabin pipeline needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mobilenet_v2 import build_mobilenet_v2


# ---------------------------------------------------------------------------
# Eye branch — small CNN, 64×64 input, ~90k parameters.
# Deliberately small: MRL has 85k eye crops, so a big model overfits fast.
# ---------------------------------------------------------------------------

def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class EyeStateCNN(nn.Module):
    """3 conv+pool blocks → GAP → 2-layer MLP → 1 logit."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),    # 64 → 32
            _conv_block(32, 64),   # 32 → 16
            _conv_block(64, 128),  # 16 →  8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
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
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# Combined two-stream model.
# ---------------------------------------------------------------------------

class TwoStreamModel(nn.Module):
    """Eye + face branches, independent weights, per-branch logits."""

    def __init__(self, *, pretrained_face: bool = True,
                 freeze_face_backbone: bool = True) -> None:
        super().__init__()
        self.eye_branch = EyeStateCNN()
        self.face_branch = build_mobilenet_v2(
            pretrained=pretrained_face,
            freeze_backbone=freeze_face_backbone,
        )

    def forward(self, eye: torch.Tensor, face: torch.Tensor) -> dict:
        """Returns per-branch logits.

        Both branches *always* run — the caller applies the per-sample
        mask downstream. Running branches on zero-tensor inputs is cheap
        and keeps the graph static, which makes mixed-precision and
        torch.compile behave predictably.
        """
        return {
            "eye_logit": self.eye_branch(eye),
            "face_logit": self.face_branch(face),
        }

    @staticmethod
    def fuse(eye_logit: torch.Tensor, face_logit: torch.Tensor,
             eye_mask: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
        """Average sigmoid outputs from whichever branches are valid.

        - Both streams valid  → mean of the two probabilities.
        - Only one valid      → that branch's probability.
        - Neither valid       → 0.5 (undefined; shouldn't happen in practice).

        Returns a probability (not a logit) so it's directly thresholdable.
        """
        eye_p = torch.sigmoid(eye_logit)
        face_p = torch.sigmoid(face_logit)
        eye_mask = eye_mask.view(-1, 1)
        face_mask = face_mask.view(-1, 1)

        weighted = eye_p * eye_mask + face_p * face_mask
        denom = (eye_mask + face_mask).clamp(min=1e-6)
        fused = weighted / denom

        # Fallback for the (both-masked-out) edge case
        both_zero = (eye_mask + face_mask) < 1e-6
        fused = torch.where(both_zero, torch.full_like(fused, 0.5), fused)
        return fused
