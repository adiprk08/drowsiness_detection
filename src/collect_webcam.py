"""Collect a small in-domain calibration set for webcam fine-tuning.

Why this script exists
----------------------
``src/realtime_demo.py`` loads MobileNetV2 trained on the DDD cabin-camera
dataset and runs it on a laptop webcam. The two distributions don't match
(camera angle, focal length, lighting, indoor vs cabin), and the face
branch saturates near 1.0 regardless of true state — the classic
distribution-shift failure mode.

The fix is to collect a small in-domain dataset and fine-tune the trained
checkpoint on it (see ``src/finetune_webcam.py``). This script is the
"collect" half: it opens the webcam, runs MediaPipe to crop the face the
same way the live demo does, and saves the face crop to disk under a
class folder of the user's choosing.

What you do
-----------
Run the script, then sit in front of the camera and:

    a   save the current face crop as ALERT      (label 0)
    d   save the current face crop as DROWSY     (label 1, eyes closed / yawning)
    q   quit

Target: ~50 frames per class, varied across pose / lighting / expression
so the fine-tuned model generalises beyond one specific moment. The
handoff notes ~100 frames per person is enough to close most of the
distribution gap.

Output layout
-------------
    data/webcam_calibration/
        alert/   frame_0000.jpg, frame_0001.jpg, ...
        drowsy/  frame_0000.jpg, frame_0001.jpg, ...

Frame numbers continue from the highest existing file in that folder, so
re-running the script appends rather than overwrites.

Notes
-----
- Faces of teammates should NOT be checked into git. The repo's
  .gitignore already excludes ``data/``, so this layout is safe.
- Saves the 224×224 letterboxed face crop (matches the model input), not
  the full frame. We don't need anything else for fine-tuning.
- The MediaPipe FaceLandmarker model file is shared with realtime_demo.py
  at ``artifacts/face_landmarker.task`` and is auto-downloaded on first
  run.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .datasets import _letterbox
from .realtime_demo import FACE_LANDMARKER_URL, _crop_with_pad


CLASS_KEYS = {ord("a"): "alert", ord("d"): "drowsy"}


def _next_frame_index(folder: Path) -> int:
    """Return one past the highest ``frame_NNNN.jpg`` index in ``folder``."""
    if not folder.exists():
        return 0
    existing = [p.stem for p in folder.glob("frame_*.jpg")]
    nums = []
    for stem in existing:
        try:
            nums.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(nums) + 1 if nums else 0


def _save_crop(crop_bgr: np.ndarray, root: Path, label_name: str,
               frame_idx: int) -> Path:
    """Letterbox to 224 and write as JPEG. Returns the path written."""
    folder = root / label_name
    folder.mkdir(parents=True, exist_ok=True)
    img224 = _letterbox(crop_bgr, 224)
    out_path = folder / f"frame_{frame_idx:04d}.jpg"
    cv2.imwrite(str(out_path), img224, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def _draw_overlay(frame: np.ndarray, face_bbox: tuple[int, int, int, int] | None,
                  counts: dict[str, int], last_saved: str | None,
                  flash_until: float) -> None:
    h, w = frame.shape[:2]
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    lines = [
        "keys:  a = save ALERT     d = save DROWSY     q = quit",
        f"saved: alert {counts.get('alert', 0)}   drowsy {counts.get('drowsy', 0)}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    if last_saved and time.time() < flash_until:
        cv2.putText(frame, last_saved, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect webcam calibration set.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument("--out", default="data/webcam_calibration",
                   help="Output root. Subfolders alert/ and drowsy/ are created under it.")
    p.add_argument("--artifacts", default="artifacts",
                   help="Where to find / cache face_landmarker.task.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        sys.exit("mediapipe not installed — run: py -m pip install mediapipe")

    landmarker_path = Path(args.artifacts) / "face_landmarker.task"
    if not landmarker_path.exists():
        landmarker_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[collect] downloading face landmarker model → {landmarker_path}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, landmarker_path)
    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )
    frame_idx = 0  # monotonic timestamp for Tasks API

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        sys.exit(f"could not open camera index {args.camera}")
    print(f"[collect] webcam {args.camera} open")

    out_root = Path(args.out)
    next_idx = {name: _next_frame_index(out_root / name)
                for name in ("alert", "drowsy")}
    counts = {name: 0 for name in ("alert", "drowsy")}
    print(f"[collect] writing to {out_root.resolve()}")
    print(f"[collect] starting indices: alert={next_idx['alert']} drowsy={next_idx['drowsy']}")
    print("[collect] keys: a=save ALERT, d=save DROWSY, q=quit")

    last_saved_msg: str | None = None
    flash_until = 0.0
    latest_face_crop: np.ndarray | None = None
    latest_face_bbox: tuple[int, int, int, int] | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[collect] camera stream ended")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_mesh.detect_for_video(mp_image, frame_idx * 33)
            frame_idx += 1

            latest_face_crop = None
            latest_face_bbox = None
            if results.face_landmarks:
                lms = results.face_landmarks[0]
                xs = np.array([lm.x for lm in lms]) * w
                ys = np.array([lm.y for lm in lms]) * h
                latest_face_crop, latest_face_bbox = _crop_with_pad(
                    frame, int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max()),
                    pad_frac=0.15,
                )

            _draw_overlay(frame, latest_face_bbox, counts,
                          last_saved_msg, flash_until)
            cv2.imshow("collect — a=ALERT  d=DROWSY  q=quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in CLASS_KEYS:
                label_name = CLASS_KEYS[key]
                if latest_face_crop is None or latest_face_crop.size == 0:
                    last_saved_msg = "no face detected — skipped"
                    flash_until = time.time() + 1.0
                    continue
                out_path = _save_crop(
                    latest_face_crop, out_root, label_name, next_idx[label_name],
                )
                counts[label_name] += 1
                next_idx[label_name] += 1
                last_saved_msg = f"saved {label_name}: {out_path.name}"
                flash_until = time.time() + 1.0
                print(f"[collect] {last_saved_msg}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

    print(f"[collect] done. saved this session: "
          f"alert={counts['alert']}  drowsy={counts['drowsy']}")
    for name in ("alert", "drowsy"):
        total = _next_frame_index(out_root / name)
        print(f"[collect]   {out_root / name}: {total} files total")


if __name__ == "__main__":
    main()
