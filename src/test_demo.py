"""Live demo of the trained MobileNetV2 on the held-out DDD test set.

Why this script exists
----------------------
The webcam demo (``src/realtime_demo.py``) hits a distribution-shift gap:
the CNNs were trained on cabin-camera footage and don't transfer cleanly
to a laptop webcam. This script removes that gap by running the trained
model on the data it was actually validated on — the held-out DDD test
subjects in our SQLite bundle.

The result is a "live" slideshow that proves the trained model works:

    - random test image displayed full-window
    - prediction overlaid (probability + class)
    - true label visible for ground truth
    - running accuracy counter accumulating across the session
    - rolling confusion-matrix counts in the corner

This is a more honest demonstration of the trained CNN than the webcam
demo for the verbal defence, because it shows the model on the
distribution it was trained for. The webcam demo can still be shown
afterwards to motivate the distribution-shift discussion.

Usage
-----
    py -m src.test_demo                                # default MobileNetV2
    py -m src.test_demo --model alexnet                # different model
    py -m src.test_demo --fps 2                        # 2 frames per second
    py -m src.test_demo --threshold 0.05               # safety threshold
    py -m src.test_demo --shuffle-seed 42              # reproducible order

Press ``q`` or ``Esc`` to quit. Press ``space`` to pause / resume.
Press ``n`` to advance one frame while paused.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .data_single_stream import FaceStreamDataset
from .models import build_model


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _denormalize(img_chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return an HxWx3 BGR uint8 image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_chw.cpu() * std + mean).clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _draw_overlay(frame: np.ndarray, *, prob: float, pred: int, label: int,
                  threshold: float, idx: int, total: int,
                  tn: int, fp: int, fn: int, tp: int) -> np.ndarray:
    h, w = frame.shape[:2]
    pad = 24
    canvas = cv2.copyMakeBorder(frame, pad, pad + 110, pad, pad,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20))
    H, W = canvas.shape[:2]

    correct = (pred == label)
    box_colour = (0, 200, 0) if correct else (0, 0, 200)
    cv2.rectangle(canvas, (pad - 2, pad - 2),
                  (pad + w + 2, pad + h + 2), box_colour, 2)

    # Centre-bottom: predicted class + true label
    pred_text = "DROWSY" if pred == 1 else "ALERT"
    label_text = "DROWSY" if label == 1 else "ALERT"
    pred_col = (0, 0, 255) if pred == 1 else (0, 200, 0)
    label_col = (0, 0, 255) if label == 1 else (0, 200, 0)

    info_y = pad + h + 32
    cv2.putText(canvas, f"pred:  {pred_text}  ({prob:.2f})",
                (pad, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                pred_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"truth: {label_text}",
                (pad, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                label_col, 2, cv2.LINE_AA)

    # Right-side: running stats
    n = tn + fp + fn + tp
    acc = (tn + tp) / n if n else 0.0
    drowsy_recall = tp / (tp + fn) if (tp + fn) else 0.0
    stats = [
        f"frame {idx}/{total}",
        f"thr {threshold:.2f}",
        f"acc {acc:.3f}",
        f"recall(drowsy) {drowsy_recall:.3f}",
        f"TP {tp}  FN {fn}",
        f"TN {tn}  FP {fp}",
    ]
    sx = W - 230
    sy = pad + 18
    for line in stats:
        cv2.putText(canvas, line, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        sy += 22

    # Top: model identity
    cv2.putText(canvas, "MobileNetV2 — DDD test set",
                (pad, pad - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_model(name: str, device: torch.device,
                artifacts: Path) -> torch.nn.Module:
    ckpt_path = artifacts / name / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path} -- run src.train --model {name} first")
    print(f"[test_demo] loading {name} from {ckpt_path}")
    model = build_model(name, pretrained=False, freeze_backbone=False).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    return model


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trained-CNN demo on DDD test set.")
    p.add_argument("--model", default="mobilenet_v2",
                   choices=["baseline_cnn", "alexnet", "mobilenet_v2"])
    p.add_argument("--db", default="data/drowsiness.db")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on sigmoid output.")
    p.add_argument("--fps", type=float, default=1.5,
                   help="Target playback frame rate.")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="Seed for the random sample order.")
    p.add_argument("--record", default=None,
                   help="Optional path to save the slideshow as MP4.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_demo] device={device}")

    model = _load_model(args.model, device, Path(args.artifacts))
    ds = FaceStreamDataset(args.db, split="test")
    print(f"[test_demo] DDD test split: {len(ds)} samples")

    indices = list(range(len(ds)))
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(indices)

    writer = None
    if args.record:
        # We'll size the writer once we render the first frame.
        writer_path = args.record

    tn = fp = fn = tp = 0
    paused = False
    advance_one = False
    frame_interval = 1.0 / max(args.fps, 0.1)
    last_render_t = 0.0

    print("[test_demo] keys: q/Esc=quit, space=pause/resume, n=next frame while paused")

    with torch.inference_mode():
        i = 0
        while i < len(indices):
            now = time.time()
            if paused and not advance_one:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = False
                if key == ord("n"):
                    advance_one = True
                continue
            if not paused and (now - last_render_t) < frame_interval:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = True
                continue

            sample_idx = indices[i]
            img_tensor, label_tensor = ds[sample_idx]
            label = int(label_tensor.item())

            x = img_tensor.unsqueeze(0).to(device)
            logit = model(x)
            prob = float(torch.sigmoid(logit).item())
            pred = 1 if prob >= args.threshold else 0

            # Update confusion counts
            if pred == 1 and label == 1: tp += 1
            elif pred == 0 and label == 0: tn += 1
            elif pred == 1 and label == 0: fp += 1
            else: fn += 1

            display = _denormalize(img_tensor)
            canvas = _draw_overlay(
                display, prob=prob, pred=pred, label=label,
                threshold=args.threshold, idx=i + 1, total=len(indices),
                tn=tn, fp=fp, fn=fn, tp=tp,
            )

            if args.record and writer is None:
                H, W = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    writer_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, (W, H),
                )
                print(f"[test_demo] recording -> {writer_path}")
            if writer is not None:
                writer.write(canvas)

            cv2.imshow(f"trained {args.model} on DDD test - q/Esc to quit",
                       canvas)
            last_render_t = now
            advance_one = False
            i += 1

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = True

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    n = tn + fp + fn + tp
    if n:
        print(f"\n[test_demo] {n} frames shown")
        print(f"  accuracy:        {(tn+tp)/n:.4f}")
        print(f"  drowsy recall:   {tp/(tp+fn) if (tp+fn) else 0:.4f}")
        print(f"  drowsy precision:{tp/(tp+fp) if (tp+fp) else 0:.4f}")


if __name__ == "__main__":
    main()
