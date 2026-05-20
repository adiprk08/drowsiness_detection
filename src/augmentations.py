"""
augmentations.py
----------------
Data augmentation for drowsiness detection.

Choices here are driven by the deployment domain (driver-facing cabin camera):
  - horizontal flip: partial — drivers don't face left/right 50/50, but some
    asymmetry (head turn, right-hand driving) is realistic; p=0.3
  - rotation: small (±10°), head tilts exist but not upside-down
  - brightness/contrast: aggressive — sun, shadows, night driving
  - **motion blur: directional** — cars move, the driver's head moves; a
    directional kernel models this far better than an isotropic Gaussian
  - **JPEG compression artefacts** — webcams / dashcams encode frames at
    variable quality, and our training data is already compressed at source;
    re-encoding teaches the model to ignore artefacts vs. real signal
  - **shift + scale jitter** — at inference a face detector (MediaPipe / Haar)
    won't centre the crop perfectly; this simulates that crop variance
  - cutout: simulates occlusion by hand, hair, sunglasses (small patches only —
    a large cutout could erase the eye signal we're trying to read)
  - **colour-temperature cast** — scales the red<->blue axis to simulate warm
    (incandescent / "yellow") vs cool (daylight) lighting. This is an
    *illumination* property, not an identity one, so augmenting it is the
    right call — it teaches white-balance invariance the training data
    (mostly neutral-lit) never showed. Distinct from a hue shift of skin
    tone, which we still avoid (see below).
  - NO vertical flip, NO arbitrary hue shift of skin tone (skin-tone
    diversity we want from data, not aug)

The pipeline is intentionally a single hand-written class rather than an
``albumentations`` / ``torchvision.transforms`` stack so the project has one
fewer heavy dependency, and every transform is auditable in ~100 lines.
"""

from __future__ import annotations

import cv2
import numpy as np


class AugPipeline:
    """Cabin-camera-aware augmentation pipeline.

    All transforms operate on a HxWx3 uint8 BGR or RGB image (they're
    channel-agnostic) and return the same shape + dtype. The image is
    letterboxed to a fixed size *before* augmentation by the dataset, so
    ``h`` and ``w`` are known constants during training.
    """

    def __init__(
        self,
        # --- geometry ---
        p_flip: float = 0.3,             # lowered from 0.5: drivers face forward-ish
        rotate_deg: float = 10.0,
        p_shift_scale: float = 0.5,      # NEW: simulate imperfect face-detector crops
        shift_max_frac: float = 0.06,    # shift up to 6% of width/height
        scale_range: tuple[float, float] = (0.90, 1.10),
        # --- photometric ---
        brightness: float = 0.3,
        contrast: float = 0.3,
        color_cast: float = 0.20,        # NEW: warm/cool white-balance jitter
        # --- cabin-specific noise ---
        p_motion_blur: float = 0.25,     # NEW: directional motion blur
        motion_blur_kernel_max: int = 9,
        p_jpeg: float = 0.30,            # NEW: re-encode at random JPEG quality
        jpeg_quality_range: tuple[int, int] = (35, 85),
        # --- occlusion ---
        p_cutout: float = 0.25,
        cutout_frac: float = 0.15,
        seed: int | None = None,
    ):
        self.p_flip = p_flip
        self.rotate_deg = rotate_deg
        self.p_shift_scale = p_shift_scale
        self.shift_max_frac = shift_max_frac
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.color_cast = color_cast
        self.p_motion_blur = p_motion_blur
        self.motion_blur_kernel_max = motion_blur_kernel_max
        self.p_jpeg = p_jpeg
        self.jpeg_quality_range = jpeg_quality_range
        self.p_cutout = p_cutout
        self.cutout_frac = cutout_frac
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Individual transforms — kept as methods so each is unit-testable.
    # ------------------------------------------------------------------

    def _motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Directional linear blur — models camera/head motion along an axis.
        Kernel size is odd, small (3–9), and the line direction is random."""
        max_k = max(3, self.motion_blur_kernel_max | 1)  # force odd
        k = int(self.rng.integers(3, max_k + 1, endpoint=False)) | 1
        kernel = np.zeros((k, k), dtype=np.float32)
        # random angle in [0, 180) — axis-symmetric so 180+ is redundant
        angle = self.rng.uniform(0, 180)
        theta = np.deg2rad(angle)
        centre = k // 2
        # draw a thin line across the kernel at `angle` from centre
        for i in range(k):
            t = i - centre
            x = int(round(centre + t * np.cos(theta)))
            y = int(round(centre + t * np.sin(theta)))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        s = kernel.sum()
        if s > 0:
            kernel /= s
        return cv2.filter2D(img, -1, kernel)

    def _jpeg_compress(self, img: np.ndarray) -> np.ndarray:
        q_lo, q_hi = self.jpeg_quality_range
        q = int(self.rng.integers(q_lo, q_hi + 1))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _shift_scale_rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Combined affine: rotation around centre + scale + translation.
        Done in one warpAffine call so we only resample the image once."""
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2
        scale = self.rng.uniform(*self.scale_range)
        tx = self.rng.uniform(-self.shift_max_frac, self.shift_max_frac) * w
        ty = self.rng.uniform(-self.shift_max_frac, self.shift_max_frac) * h
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _apply_color_cast(self, img: np.ndarray) -> np.ndarray:
        """Simulate warm/cool illumination (colour-temperature shift).

        Scales the two outer channels in opposite directions. Under either
        BGR or RGB ordering that is a blue<->red cast — exactly the axis
        along which incandescent ('yellow') vs daylight light differ. The
        shift is symmetric and random, so the *set* of augmentations is
        identical regardless of channel order: this stays channel-agnostic
        like the rest of the pipeline. The middle (green) channel — closest
        to luminance — is left untouched.

        This is what makes the model robust to warm indoor lighting; the
        training data (DDD + UTA) is mostly neutral-lit, so without this
        the model sees a yellow-lit face as out-of-distribution.
        """
        shift = self.rng.uniform(-self.color_cast, self.color_cast)
        img = img.astype(np.float32)
        img[:, :, 0] *= (1.0 + shift)
        img[:, :, 2] *= (1.0 - shift)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # Horizontal flip
        if self.rng.random() < self.p_flip:
            img = img[:, ::-1, :].copy()

        # Combined rotate + shift + scale (one resample)
        want_rotate = self.rotate_deg > 0
        want_shift = self.rng.random() < self.p_shift_scale
        if want_rotate or want_shift:
            angle = (self.rng.uniform(-self.rotate_deg, self.rotate_deg)
                     if want_rotate else 0.0)
            if want_shift:
                img = self._shift_scale_rotate(img, angle)
            else:
                # rotation only (no scale / shift) — keep old behaviour
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101,
                )

        # Brightness + contrast (in float space, then clipped)
        if self.brightness > 0 or self.contrast > 0:
            alpha = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
            beta = 255.0 * self.rng.uniform(-self.brightness, self.brightness)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Colour-temperature cast (warm <-> cool white balance)
        if self.color_cast > 0:
            img = self._apply_color_cast(img)

        # Motion blur (directional)
        if self.rng.random() < self.p_motion_blur:
            img = self._motion_blur(img)

        # JPEG compression artefacts
        if self.rng.random() < self.p_jpeg:
            img = self._jpeg_compress(img)

        # Cutout — small only, so eye/mouth regions aren't wiped
        if self.rng.random() < self.p_cutout:
            ch = int(h * self.cutout_frac)
            cw = int(w * self.cutout_frac)
            if ch > 0 and cw > 0:
                y = self.rng.integers(0, h - ch)
                x = self.rng.integers(0, w - cw)
                img[y:y + ch, x:x + cw] = 0

        return img
