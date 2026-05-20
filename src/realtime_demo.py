"""Real-time drowsiness detection on a webcam.

What this does
--------------
1. Opens the webcam and reads frames at whatever FPS the camera delivers.
2. Runs the MediaPipe ``FaceLandmarker`` (Tasks API, 478 landmarks) on each
   frame to locate the face + eye regions.
3. Crops the face (224×224 letterboxed) and one eye (64×64 letterboxed).
4. Feeds the face crop to the single-stream MobileNetV2 (``artifacts/mobilenet_v2/best.pt``).
5. Feeds the eye crop to the two-stream model's eye branch (``artifacts/two_stream/best.pt``).
6. Fuses the two probabilities (simple mean of sigmoids).
7. Maintains a rolling buffer of the last ``window`` predictions — the
   **temporal smoothing** that our project's write-up has promised all
   along. A blink (1–3 frames of "drowsy") gets averaged out; sustained
   drowsiness (15+ of 30 frames above threshold) is held onto.
8. Applies **hysteresis**: we don't flip the alarm state on a single
   smoothed sample crossing the threshold — we require ``hysteresis``
   consecutive samples above/below the threshold. Stops flickering.
9. Draws a live overlay: face bounding box, instantaneous probability,
   smoothed probability, current state (ALERT / DROWSY).

Usage
-----
    py -m src.realtime_demo                          # default webcam (index 0)
    py -m src.realtime_demo --camera 1               # second camera
    py -m src.realtime_demo --threshold 0.5          # decision threshold
    py -m src.realtime_demo --window 30              # temporal smoothing window
    py -m src.realtime_demo --record out.mp4         # also save an MP4
    py -m src.realtime_demo --video clip.mp4         # run on a file instead of webcam
    py -m src.realtime_demo --face-ckpt artifacts/mobilenet_v2_finetuned/best.pt
                                                     # use a fine-tuned webcam-domain checkpoint

Press ``q`` or ``Esc`` in the window to quit.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .datasets import DrowsinessDataset, _letterbox
from .models import build_model


# ---------------------------------------------------------------------------
# MediaPipe landmark indices — FaceLandmarker 478-point model.
# ---------------------------------------------------------------------------
# The full mesh gives us hundreds of points; we only need a few.
# Right eye from the *person's* perspective (camera's left if front-facing).
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382,
                362, 398, 384, 385, 386, 387, 388, 466]

# 6-point subsets used for Eye Aspect Ratio (Soukupova & Cech 2016).
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
# Open eye is ~0.25–0.35, closed eye drops below ~0.18.
#            p1=outer  p2,p3=upper lid   p4=inner  p5,p6=lower lid
EAR_RIGHT = (33,       160, 158,         133,      153, 144)
EAR_LEFT  = (263,      387, 385,         362,      380, 373)

# Mouth Aspect Ratio (yawn signal). Same vertical/horizontal idea applied to
# the lips. Closed mouth gives MAR ≈ 0.02–0.10; a yawn opens the jaw enough
# to push it past ~0.5. Indices are MediaPipe FaceLandmarker:
#   61, 291  -> mouth corners (horizontal)
#   13, 14   -> centre upper/lower lip (inner mouth opening)
#   78, 308  -> inner corners (extra horizontal anchor)
MAR_INDICES = (61, 291, 13, 14)

# Auto-downloaded on first run if missing. The float16 bundle is ~3 MB and
# gives us 478 landmarks (including refined lips/eyes), which is what the
# older ``solutions.face_mesh`` API returned internally anyway.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# ---------------------------------------------------------------------------
# Preprocessing — must match what the training dataset does to the image.
# ---------------------------------------------------------------------------

_MEAN = DrowsinessDataset._MEAN  # (0.485, 0.456, 0.406)
_STD = DrowsinessDataset._STD    # (0.229, 0.224, 0.225)


def _crop_with_pad(frame: np.ndarray, x_min: int, y_min: int,
                   x_max: int, y_max: int, pad_frac: float = 0.15) -> np.ndarray:
    """Crop [x_min,y_min]..[x_max,y_max] with ``pad_frac`` of the box
    padded on every side. Clamps to frame boundaries."""
    h, w = frame.shape[:2]
    box_w, box_h = x_max - x_min, y_max - y_min
    pad_x, pad_y = int(box_w * pad_frac), int(box_h * pad_frac)
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x)
    y1 = min(h, y_max + pad_y)
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _eye_aspect_ratio(lms, indices: tuple, w: int, h: int) -> float:
    """EAR for one eye from six landmark indices (p1..p6 in the standard order)."""
    def pt(i: int) -> np.ndarray:
        lm = lms[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    p1, p2, p3, p4, p5, p6 = (pt(i) for i in indices)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(vertical / (2.0 * horizontal))


def _mouth_aspect_ratio(lms, w: int, h: int) -> float:
    """MAR — yawn signal. Vertical inner-lip distance / horizontal corner distance."""
    def pt(i: int) -> np.ndarray:
        lm = lms[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    left, right, top, bot = (pt(i) for i in MAR_INDICES)
    horizontal = np.linalg.norm(left - right)
    if horizontal < 1e-6:
        return 0.0
    vertical = np.linalg.norm(top - bot)
    return float(vertical / horizontal)


def _to_model_input(img_bgr: np.ndarray, target: int,
                    device: torch.device) -> torch.Tensor:
    """BGR → RGB → letterbox(target) → normalise → (1, 3, target, target) tensor."""
    if img_bgr.size == 0:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = _letterbox(img, target)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Temporal smoothing + hysteresis
# ---------------------------------------------------------------------------

class Smoother:
    """Rolling-average of the last ``window`` probabilities, plus hysteresis
    on the alarm state so a single borderline frame doesn't flip it."""

    def __init__(self, window: int = 30, threshold: float = 0.5,
                 hysteresis: int = 3) -> None:
        self.buf: collections.deque[float] = collections.deque(maxlen=window)
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.state_drowsy = False
        self._streak_above = 0
        self._streak_below = 0

    def push(self, p: float) -> tuple[float, bool]:
        """Returns (smoothed_prob, alarm_state). Call once per frame."""
        self.buf.append(p)
        smooth = sum(self.buf) / len(self.buf)

        if smooth >= self.threshold:
            self._streak_above += 1
            self._streak_below = 0
            if self._streak_above >= self.hysteresis:
                self.state_drowsy = True
        else:
            self._streak_below += 1
            self._streak_above = 0
            if self._streak_below >= self.hysteresis:
                self.state_drowsy = False

        return smooth, self.state_drowsy

    def reset(self) -> None:
        self.buf.clear()
        self.state_drowsy = False
        self._streak_above = 0
        self._streak_below = 0


# ---------------------------------------------------------------------------
# UI overlay
# ---------------------------------------------------------------------------

def _draw_overlay(frame: np.ndarray, face_bbox: tuple[int, int, int, int] | None,
                  eye_bboxes: list[tuple[int, int, int, int]],
                  p_face: float | None, p_eye: float | None,
                  ear: float | None, mar: float | None,
                  inst_prob: float | None, smooth_prob: float | None,
                  drowsy: bool, fps: float) -> None:
    h, w = frame.shape[:2]
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for bbox in eye_bboxes:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)

    # Top-left: FPS + per-branch probabilities + fused
    lines = [f"FPS: {fps:5.1f}"]
    lines.append(f"P face: {p_face:.2f}" if p_face is not None else "P face: --")
    lines.append(f"P eye : {p_eye:.2f}" if p_eye is not None else "P eye : --")
    lines.append(f"EAR   : {ear:.2f}" if ear is not None else "EAR   : --")
    lines.append(f"MAR   : {mar:.2f}" if mar is not None else "MAR   : --")
    if inst_prob is not None:
        lines.append(f"inst   P: {inst_prob:.2f}")
    if smooth_prob is not None:
        lines.append(f"smooth P: {smooth_prob:.2f}")
    else:
        lines.append("smooth P: --")
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    # Centre-top: big state badge
    badge_text = "DROWSY" if drowsy else "ALERT"
    colour = (0, 0, 255) if drowsy else (0, 200, 0)
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    pad = 12
    bx0 = (w - tw) // 2 - pad
    by0 = 10
    bx1 = bx0 + tw + 2 * pad
    by1 = by0 + th + 2 * pad
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), colour, -1)
    cv2.putText(frame, badge_text, (bx0 + pad, by1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _load_models(device: torch.device, artifacts: Path,
                 face_ckpt: Path | None = None,
                 eye_ckpt: Path | None = None,
                 ) -> tuple[torch.nn.Module, torch.nn.Module]:
    face_ckpt_path = Path(face_ckpt) if face_ckpt else artifacts / "mobilenet_v2" / "best.pt"
    two_stream_ckpt_path = Path(eye_ckpt) if eye_ckpt else artifacts / "two_stream" / "best.pt"
    if not face_ckpt_path.exists():
        sys.exit(f"face checkpoint not found: {face_ckpt_path} — run src.train first")
    if not two_stream_ckpt_path.exists():
        sys.exit(f"two-stream checkpoint not found: {two_stream_ckpt_path} — run src.train_fusion first")

    print(f"[demo] loading face model from {face_ckpt_path}")
    face_model = build_model("mobilenet_v2", pretrained=False,
                             freeze_backbone=False).to(device).eval()
    face_model.load_state_dict(torch.load(face_ckpt_path, map_location=device)["state_dict"])

    print(f"[demo] loading two-stream (for eye branch) from {two_stream_ckpt_path}")
    two_stream = build_model("two_stream", pretrained=False,
                             freeze_backbone=False).to(device).eval()
    two_stream.load_state_dict(torch.load(two_stream_ckpt_path, map_location=device)["state_dict"])
    # We only need the eye branch — but keeping the whole model around is
    # cheap (~8 MB) and means we don't have to repackage the checkpoint.
    return face_model, two_stream.eye_branch


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live drowsiness detection demo.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument("--video", default=None, help="Run on a video file instead of webcam.")
    p.add_argument("--record", default=None,
                   help="Path to save the overlay video (mp4). Optional.")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--face-ckpt", default=None,
                   help="Override the face model checkpoint path. "
                        "Default: artifacts/mobilenet_v2/best.pt. Use this to "
                        "point at the fine-tuned webcam checkpoint, e.g. "
                        "artifacts/mobilenet_v2_finetuned/best.pt.")
    p.add_argument("--eye-ckpt", default=None,
                   help="Override the two-stream checkpoint path "
                        "(eye branch is taken from it). "
                        "Default: artifacts/two_stream/best.pt.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on smoothed probability.")
    p.add_argument("--window", type=int, default=30,
                   help="Temporal smoothing window (frames).")
    p.add_argument("--hysteresis", type=int, default=3,
                   help="Consecutive smoothed samples required to switch alarm state.")
    p.add_argument("--show-fps", action="store_true",
                   help="Print FPS to stdout each second.")
    p.add_argument("--face-only", action="store_true",
                   help="Use only the face model (disable eye branch).")
    p.add_argument("--eye-only", action="store_true",
                   help="Use only the eye branch (disable face model).")
    p.add_argument("--use-ear", action="store_true",
                   help="Drive the alarm decision from Eye Aspect Ratio "
                        "(MediaPipe landmarks) instead of the CNN fusion. "
                        "CNN probs remain visible on the overlay.")
    p.add_argument("--ear-threshold", type=float, default=0.20,
                   help="Eye Aspect Ratio below this value counts as "
                        "'eyes closed / drowsy' (default 0.20). Typical open "
                        "eye is 0.25–0.35; closed eye falls below ~0.18.")
    p.add_argument("--mar-threshold", type=float, default=0.55,
                   help="Mouth Aspect Ratio above this value counts as "
                        "'yawning' (default 0.55). Closed mouth is ~0.05; "
                        "a full yawn pushes past ~0.6.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        sys.exit("mediapipe not installed — run: py -m pip install mediapipe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device}")

    face_model, eye_model = _load_models(
        device, Path(args.artifacts),
        face_ckpt=args.face_ckpt, eye_ckpt=args.eye_ckpt,
    )
    smoother = Smoother(window=args.window, threshold=args.threshold,
                        hysteresis=args.hysteresis)

    # MediaPipe FaceLandmarker (Tasks API). The legacy ``mp.solutions.face_mesh``
    # module is no longer shipped on Python 3.13 wheels — the Tasks API is the
    # supported path forward. Output is still 478 landmarks, same indices.
    landmarker_path = Path(args.artifacts) / "face_landmarker.task"
    if not landmarker_path.exists():
        landmarker_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"[demo] downloading face landmarker model → {landmarker_path}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, landmarker_path)
    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )
    frame_idx = 0  # used to feed the Tasks API a monotonically increasing timestamp

    # Source: webcam or video file
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[demo] video file: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        print(f"[demo] webcam index: {args.camera}")
    if not cap.isOpened():
        sys.exit("could not open video source")

    writer = None
    if args.record:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h),
        )
        print(f"[demo] recording overlay → {args.record}")

    print("[demo] press 'q' or Esc to quit")

    # FPS estimator — EMA over inter-frame intervals
    last_t = time.time()
    fps_ema = 0.0

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[demo] end of video / camera stream")
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # Tasks API in VIDEO mode needs a monotonically increasing
                # timestamp — frame count × nominal 33 ms is fine.
                results = face_mesh.detect_for_video(mp_image, frame_idx * 33)
                frame_idx += 1

                face_bbox = None
                eye_bbox = None  # crop fed to the CNN eye branch (right eye)
                eye_bboxes: list[tuple[int, int, int, int]] = []
                inst_prob: float | None = None
                smooth_prob: float | None = None
                p_face: float | None = None
                p_eye: float | None = None
                ear: float | None = None
                mar: float | None = None

                if results.face_landmarks:
                    lms = results.face_landmarks[0]
                    # Eye Aspect Ratio — classical drowsiness signal from the
                    # landmarks themselves, independent of the CNNs.
                    ear_r = _eye_aspect_ratio(lms, EAR_RIGHT, w, h)
                    ear_l = _eye_aspect_ratio(lms, EAR_LEFT, w, h)
                    ear = 0.5 * (ear_r + ear_l)
                    # Mouth Aspect Ratio — yawn signal.
                    mar = _mouth_aspect_ratio(lms, w, h)
                    # Face bbox from *all* landmarks
                    xs = np.array([lm.x for lm in lms]) * w
                    ys = np.array([lm.y for lm in lms]) * h
                    face_crop, face_bbox = _crop_with_pad(
                        frame, int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()),
                        pad_frac=0.15,
                    )
                    # Eye bboxes — both eyes drawn on the overlay so it
                    # visibly tracks blinks on either side. Only the right
                    # eye's crop is fed to the CNN eye branch (it was
                    # trained on single-eye crops, so feeding both would
                    # require either two forward passes or stitching).
                    for idx_set in (RIGHT_EYE_IDX, LEFT_EYE_IDX):
                        ex = np.array([lms[i].x for i in idx_set]) * w
                        ey = np.array([lms[i].y for i in idx_set]) * h
                        crop, bbox = _crop_with_pad(
                            frame, int(ex.min()), int(ey.min()),
                            int(ex.max()), int(ey.max()),
                            pad_frac=0.35,
                        )
                        eye_bboxes.append(bbox)
                        if idx_set is RIGHT_EYE_IDX:
                            eye_crop = crop
                            eye_bbox = bbox

                    face_tensor = _to_model_input(face_crop, 224, device)
                    eye_tensor = _to_model_input(eye_crop, 64, device)

                    p_face = None
                    p_eye = None
                    probs: list[float] = []
                    if face_tensor is not None and not args.eye_only:
                        p_face = torch.sigmoid(face_model(face_tensor)).item()
                        probs.append(p_face)
                    if eye_tensor is not None and not args.face_only:
                        p_eye = torch.sigmoid(eye_model(eye_tensor)).item()
                        probs.append(p_eye)

                    if probs:
                        inst_prob = sum(probs) / len(probs)

                    # Alarm driver: classical (EAR + MAR) if requested, else
                    # the CNN fusion. EAR/MAR are converted to BINARY events
                    # (eye-closed = 1, otherwise = 0) so the smoothed value is
                    # *literal PERCLOS* — the fraction of recent frames in
                    # which eyes were closed, which is the well-defined
                    # drowsiness signal from the trucking-industry literature.
                    # A typical blink (3–5 frames out of 30) contributes only
                    # ~10–17% PERCLOS, well below the default --threshold 0.5.
                    if args.use_ear and ear is not None:
                        ear_closed = 1.0 if ear < args.ear_threshold else 0.0
                        yawning = 0.0
                        if mar is not None:
                            yawning = 1.0 if mar > args.mar_threshold else 0.0
                        drive = max(ear_closed, yawning)
                        smooth_prob, _ = smoother.push(drive)
                    elif inst_prob is not None:
                        smooth_prob, _ = smoother.push(inst_prob)
                else:
                    # No face detected — don't pollute the buffer with a stale
                    # reading, just leave the smoother alone.
                    pass

                # FPS
                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now
                fps_inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * fps_inst if fps_ema else fps_inst

                _draw_overlay(frame, face_bbox, eye_bboxes,
                              p_face, p_eye, ear, mar,
                              inst_prob, smooth_prob,
                              smoother.state_drowsy, fps_ema)

                if writer is not None:
                    writer.write(frame)
                cv2.imshow("drowsiness demo — q/Esc to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.show_fps and int(now) != int(now - dt):
                    print(f"[demo] fps={fps_ema:.1f}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
