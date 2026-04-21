"""
augmentations.py
----------------
Data augmentation for drowsiness detection.

Choices here are driven by the deployment domain (driver-facing cabin camera):
  - horizontal flip: OK, driver position is not inherently left/right asymmetric
  - rotation: small (±10°), head tilts exist but not upside-down
  - brightness/contrast: aggressive — sun, shadows, night driving
  - motion blur: small; cars move
  - cutout: simulates occlusion by hand, hair, sunglasses (small patches only —
    a large cutout could erase the eye signal we're trying to read)
  - NO vertical flip, NO hue shift (skin-tone invariance we want from data, not aug)
"""

from __future__ import annotations

import cv2
import numpy as np


class AugPipeline:
    def __init__(
        self,
        p_flip: float = 0.5,
        rotate_deg: float = 10.0,
        brightness: float = 0.3,
        contrast: float = 0.3,
        p_blur: float = 0.2,
        blur_kernel: int = 3,
        p_cutout: float = 0.25,
        cutout_frac: float = 0.15,
        seed: int | None = None,
    ):
        self.p_flip = p_flip
        self.rotate_deg = rotate_deg
        self.brightness = brightness
        self.contrast = contrast
        self.p_blur = p_blur
        self.blur_kernel = blur_kernel
        self.p_cutout = p_cutout
        self.cutout_frac = cutout_frac
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # Horizontal flip
        if self.rng.random() < self.p_flip:
            img = img[:, ::-1, :].copy()

        # Rotation
        if self.rotate_deg > 0:
            angle = self.rng.uniform(-self.rotate_deg, self.rotate_deg)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h),
                borderMode=cv2.BORDER_REFLECT_101,
            )

        # Brightness + contrast (applied in float space, then clipped)
        if self.brightness > 0 or self.contrast > 0:
            alpha = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
            beta = 255.0 * self.rng.uniform(-self.brightness, self.brightness)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Motion blur (light)
        if self.rng.random() < self.p_blur:
            k = self.blur_kernel
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Cutout — small only
        if self.rng.random() < self.p_cutout:
            ch = int(h * self.cutout_frac)
            cw = int(w * self.cutout_frac)
            if ch > 0 and cw > 0:
                y = self.rng.integers(0, h - ch)
                x = self.rng.integers(0, w - cw)
                img[y:y + ch, x:x + cw] = 0

        return img
