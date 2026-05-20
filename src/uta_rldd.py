"""UTA-RLDD video → face-crop extractor and on-disk dataset.

Why
---
UTA-RLDD (Ghoddoosian et al. 2019) is a 60-subject self-recorded drowsiness
dataset filmed on phones and webcams — much closer to our deployment
domain (a laptop webcam) than the cabin-camera DDD set. Adding it to the
training distribution is the proper-engineering fix for the
distribution-shift problem documented in handoff.md.

The raw download is ~85 GB of MP4. This module turns it into a few GB of
face JPEGs at ~1 fps so the existing training loop can consume it
alongside DDD.

Pipeline
--------
1. CLI (``py -m src.uta_rldd extract``) walks the Kaggle-distributed
   layout::

       data/uta-rldd/                    (Kaggle: rishab260/uta-reallife-drowsiness-dataset)
           Fold1_part1/                  (Kaggle's outer zip-extract dir)
               Fold1_part1/              (duplicate of outer, also zip artefact)
                   01/  0.mov  5.mov  10.MOV
                   02/  ...
                   ...                   (~6 subjects per part)
           Fold1_part2/  Fold1_part2/
           Fold2_part1/  Fold2_part1/
           Fold2_part2/  Fold2_part2/
           Fold3_part1/  Fold3_part1/
           Fold3_part2/  Fold3_part2/
           Fold4_part1/  Fold4_part1/    (Kaggle mirror has 4 folds × 2 parts
           Fold4_part2/  Fold4_part2/     ≈ 48 subjects total — fewer than the
                                          60-subject original release)

   The Kaggle mirror has a mix of ``.mp4`` and ``.mov`` files; both work
   identically here (OpenCV decodes via FFmpeg). For each
   ``0.{mp4,mov}`` / ``10.{mp4,mov}`` (we drop the ambiguous ``5``
   "low vigilance" class — see handoff doc):

     - opens the video with OpenCV
     - samples every Nth frame (default 30 ≈ 1 fps)
     - runs MediaPipe FaceLandmarker on each sampled frame
     - crops the face with 15% padding (same as the live demo)
     - letterboxes to 224 × 224
     - writes to ``data/uta_rldd_frames/<subject_id>/<label>/<stem>_<frame>.jpg``

   Per-video resumable: re-running skips videos whose output folder is
   already non-empty. Logs progress per video so a multi-hour extract is
   interruptible.

2. :class:`UtaRldDataset` reads the on-disk JPEG tree and yields
   ``(Tensor[3, 224, 224], Tensor[1])`` — same shape as
   :class:`FaceStreamDataset`, so it concat-ables with the existing
   training set with zero pipeline changes.

3. :func:`split_uta_subjects` produces a subject-disjoint train / val /
   test split over UTA subjects, so we never train and evaluate on the
   same person.

Subject IDs
-----------
Path-derived: ``uta_<fold>_<part>_<subject_num>`` (e.g.
``uta_fold1_part1_03``). Globally unique across the 60 subjects even if
two parts of the same fold happen to share a numeric subject folder
(they don't in the official release, but the prefix is cheap insurance).

Usage
-----
    py -m src.uta_rldd extract                                # default paths
    py -m src.uta_rldd extract --root D:/uta_rldd --out D:/uta_frames
    py -m src.uta_rldd extract --every 60                     # 0.5 fps
    py -m src.uta_rldd extract --max-videos 4                 # quick smoke test
    py -m src.uta_rldd stats                                  # report what's extracted
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import DrowsinessDataset, Sample, _letterbox

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout walking
# ---------------------------------------------------------------------------

# Label encoding for UTA filenames. Class 5 ("low vigilance") is intentionally
# omitted — see handoff.md / discussion: it's an ambiguous middle class and we
# train on the unambiguous endpoints.
_UTA_LABEL_MAP: dict[str, int] = {
    "0":  0,   # alert
    "10": 1,   # drowsy
}


@dataclass(frozen=True)
class _VideoTask:
    video_path: Path
    subject_id: str   # e.g. "uta_fold1_part1_03"
    label: int        # 0 or 1
    label_name: str   # "alert" or "drowsy"
    out_dir: Path     # where this video's frames will be written


def _iter_videos(root: Path, out_root: Path) -> Iterator[_VideoTask]:
    """Yield one _VideoTask per (subject, class) MP4 found under ``root``.

    Handles the Kaggle layout where each fold/part is wrapped in a duplicate
    directory: ``fold1_part1/fold1_part1/01/0.mp4``. We just walk in and let
    the path tell us the fold / part / subject.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"UTA root not found: {root}")

    for fold_part_outer in sorted(root.iterdir()):
        if not fold_part_outer.is_dir():
            continue
        # Look one level deeper for the duplicated wrapper, then fall back to
        # treating ``fold_part_outer`` itself as the subject-containing dir.
        candidates = [d for d in fold_part_outer.iterdir() if d.is_dir()]
        # The duplicated wrapper has the same name as its parent; if we find
        # exactly that, descend into it.
        deeper = [d for d in candidates if d.name == fold_part_outer.name]
        subject_parent = deeper[0] if len(deeper) == 1 else fold_part_outer
        fold_part_key = fold_part_outer.name  # e.g. "fold1_part1"

        for subject_dir in sorted(subject_parent.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = f"uta_{fold_part_key}_{subject_dir.name}"
            for video_path in sorted(subject_dir.iterdir()):
                # Kaggle mirror has a mix of .mp4 / .mov / .m4v; OpenCV
                # decodes all of them via FFmpeg so we just whitelist the
                # container suffix.
                if video_path.suffix.lower() not in {
                    ".mp4", ".mov", ".m4v", ".avi", ".mkv",
                }:
                    continue
                # Filename stem encodes the class: "0", "5", "10". A few
                # subjects' drowsy clip is split into two files named
                # "10_1" / "10_2" — take the part before the first
                # underscore so both still resolve to class "10".
                stem = video_path.stem               # "0", "5", "10", "10_1"
                base = stem.split("_", 1)[0]          # "0", "5", "10", "10"
                if base not in _UTA_LABEL_MAP:
                    continue
                label = _UTA_LABEL_MAP[base]
                label_name = "drowsy" if label == 1 else "alert"
                out_dir = (out_root / subject_id / label_name /
                           f"{video_path.stem}")
                yield _VideoTask(
                    video_path=video_path,
                    subject_id=subject_id,
                    label=label,
                    label_name=label_name,
                    out_dir=out_dir,
                )


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def _crop_face_from_frame(frame_bgr: np.ndarray, lms_result, pad_frac: float = 0.15
                          ) -> np.ndarray | None:
    """Crop the first detected face from ``frame_bgr`` using MediaPipe
    landmark results. Returns the cropped BGR uint8 image, or None if no
    face was found."""
    if not lms_result.face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    lms = lms_result.face_landmarks[0]
    xs = np.array([lm.x for lm in lms]) * w
    ys = np.array([lm.y for lm in lms]) * h
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w <= 0 or box_h <= 0:
        return None
    pad_x, pad_y = int(box_w * pad_frac), int(box_h * pad_frac)
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x)
    y1 = min(h, y_max + pad_y)
    return frame_bgr[y0:y1, x0:x1].copy()


def _extract_one_video(task: _VideoTask, every: int, face_mesh,
                       mp_image_cls, mp_image_format,
                       start_ts_ms: int = 0,
                       progress_every: int = 60) -> tuple[dict, int]:
    """Extract face crops from a single video.

    MediaPipe's ``detect_for_video`` requires monotonically increasing
    timestamps **across the lifetime of the FaceLandmarker** — not just
    within a single video — otherwise it raises
    ``ValueError: Input timestamp must be monotonically increasing.``
    So we take ``start_ts_ms`` from the caller and return the next safe
    timestamp; the caller threads it through across videos.

    Returns ``(stats_dict, next_start_ts_ms)``.
    """
    out_dir = task.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(task.video_path))
    if not cap.isOpened():
        log.warning("could not open %s", task.video_path)
        return ({"saved": 0, "no_face": 0, "frames": 0, "skipped": True},
                start_ts_ms)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved = 0
    no_face = 0
    frame_idx = 0
    sample_idx = 0  # per-video sample counter, only used for progress / logging
    t0 = time.time()
    last_ts_ms = start_ts_ms

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_image_cls(image_format=mp_image_format, data=rgb)
                # Global monotonic timestamp: previous + 33 ms (matches the
                # nominal 30 fps cadence MediaPipe expects).
                ts_ms = last_ts_ms + 33
                results = face_mesh.detect_for_video(mp_image, ts_ms)
                last_ts_ms = ts_ms
                sample_idx += 1
                crop = _crop_face_from_frame(frame, results)
                if crop is None or crop.size == 0:
                    no_face += 1
                else:
                    img224 = _letterbox(crop, 224)
                    out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), img224,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved += 1
                if sample_idx % progress_every == 0:
                    dt = time.time() - t0
                    log.info(
                        "  %s: %d / %d frames processed (%d saved, %d no-face, %.1fs)",
                        task.video_path.name, frame_idx, total_frames, saved, no_face, dt,
                    )
            frame_idx += 1
    finally:
        cap.release()

    return ({
        "saved": saved, "no_face": no_face, "frames": frame_idx,
        "skipped": False, "elapsed_s": time.time() - t0,
    }, last_ts_ms)


# ---------------------------------------------------------------------------
# CLI: extract
# ---------------------------------------------------------------------------

def _cmd_extract(args: argparse.Namespace) -> None:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        sys.exit("mediapipe not installed — run: py -m pip install mediapipe")

    landmarker_path = Path(args.artifacts) / "face_landmarker.task"
    if not landmarker_path.exists():
        # Lazy-download via the same URL the live demo uses.
        from .realtime_demo import FACE_LANDMARKER_URL
        import urllib.request
        landmarker_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading face landmarker → %s", landmarker_path)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, landmarker_path)
    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = list(_iter_videos(root, out_root))
    log.info("found %d videos under %s", len(tasks), root)
    if args.max_videos:
        tasks = tasks[: args.max_videos]
        log.info("limiting to first %d videos (--max-videos)", args.max_videos)

    totals = {"saved": 0, "no_face": 0, "videos_done": 0, "videos_skipped": 0}
    t_all = time.time()
    # Global monotonically-increasing timestamp threaded across videos so
    # MediaPipe doesn't reject the input when we move on to the next clip.
    next_ts_ms = 0

    try:
        for i, task in enumerate(tasks, 1):
            # Resumable: skip only videos that wrote a .done marker — i.e.
            # finished cleanly. Partial output (Ctrl-C mid-video) leaves
            # jpgs without a marker, so we re-process from scratch.
            done_marker = task.out_dir / ".done"
            if done_marker.exists() and not args.force:
                log.info(
                    "[%d/%d] %s — already extracted, skipping",
                    i, len(tasks), task.video_path.relative_to(root),
                )
                totals["videos_skipped"] += 1
                continue

            # Wipe any partial frames from a previous interrupted run so we
            # don't end up with a mix of "frames from old crashed run" and
            # "frames from this run" in the same dir.
            if task.out_dir.exists():
                for stale in task.out_dir.glob("*.jpg"):
                    stale.unlink()

            log.info("[%d/%d] %s  → %s",
                     i, len(tasks),
                     task.video_path.relative_to(root),
                     task.out_dir.relative_to(out_root))
            stats, next_ts_ms = _extract_one_video(
                task, args.every, face_mesh, mp.Image, mp.ImageFormat.SRGB,
                start_ts_ms=next_ts_ms,
            )
            # 1-second gap between clips so adjacent videos' timestamps
            # are clearly separated — pure paranoia, no functional need.
            next_ts_ms += 1000
            totals["saved"] += stats["saved"]
            totals["no_face"] += stats["no_face"]
            totals["videos_done"] += 1
            log.info("    saved=%d  no_face=%d  frames=%d  %.1fs",
                     stats["saved"], stats["no_face"], stats["frames"],
                     stats.get("elapsed_s", 0.0))
            # Mark this video complete only after the loop above finished
            # cleanly — Ctrl-C during the loop never reaches this line, so
            # no marker is written and the video re-runs on the next launch.
            done_marker.touch()
    finally:
        face_mesh.close()

    dt = time.time() - t_all
    log.info(
        "[done] %d videos processed, %d skipped, %d frames saved, "
        "%d no-face, %.1f min",
        totals["videos_done"], totals["videos_skipped"], totals["saved"],
        totals["no_face"], dt / 60,
    )


# ---------------------------------------------------------------------------
# CLI: stats
# ---------------------------------------------------------------------------

def _cmd_stats(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    if not out_root.is_dir():
        sys.exit(f"no extracted-frames dir at {out_root}")
    subjects = sorted(p for p in out_root.iterdir() if p.is_dir())
    print(f"[uta] {len(subjects)} subjects under {out_root}")
    n_alert = n_drowsy = 0
    for s in subjects:
        a = sum(1 for _ in (s / "alert").rglob("*.jpg")) if (s / "alert").exists() else 0
        d = sum(1 for _ in (s / "drowsy").rglob("*.jpg")) if (s / "drowsy").exists() else 0
        n_alert += a
        n_drowsy += d
        if args.per_subject:
            print(f"  {s.name}: alert={a}  drowsy={d}")
    print(f"[uta] totals: alert={n_alert}  drowsy={n_drowsy}  "
          f"(total={n_alert + n_drowsy})")


# ---------------------------------------------------------------------------
# Indexer (for use from datasets.py or train scripts)
# ---------------------------------------------------------------------------

def index_uta_rldd(out_root: str | Path) -> list[Sample]:
    """Walk an extracted-frames tree and emit :class:`Sample` records.

    Mirrors :func:`src.datasets.index_ddd` so the split logic in
    ``_group_split`` accepts the result unchanged.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise FileNotFoundError(f"UTA frames root not found: {out_root}")

    samples: list[Sample] = []
    for subject_dir in sorted(out_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        for label_name, label in (("alert", 0), ("drowsy", 1)):
            cls_dir = subject_dir / label_name
            if not cls_dir.exists():
                continue
            for path in sorted(cls_dir.rglob("*.jpg")):
                samples.append(Sample(
                    path=path, label=label, source="uta",
                    stream="face", subject_id=subject_id,
                ))
    log.info("UTA: indexed %d frames across %d subjects",
             len(samples), len({s.subject_id for s in samples}))
    return samples


# ---------------------------------------------------------------------------
# Dataset class — concat-able with FaceStreamDataset
# ---------------------------------------------------------------------------

class UtaRldDataset(Dataset):
    """Reads extracted face crops from disk and yields the same
    ``(Tensor[3, 224, 224], Tensor[1])`` shape as
    :class:`FaceStreamDataset`, so it concats cleanly into the training
    loop.

    Honours the ``subjects`` whitelist for subject-disjoint splits.
    """

    _MEAN = DrowsinessDataset._MEAN
    _STD = DrowsinessDataset._STD

    def __init__(
        self,
        out_root: str | Path,
        subjects: Iterable[str] | None = None,
        augment_fn=None,
    ):
        self.out_root = Path(out_root)
        all_samples = index_uta_rldd(self.out_root)
        if subjects is not None:
            allowed = set(subjects)
            all_samples = [s for s in all_samples if s.subject_id in allowed]
        if not all_samples:
            raise ValueError(
                f"No UTA samples under {out_root} (subjects filter may be too narrow). "
                f"Run `py -m src.uta_rldd extract` first."
            )
        self.samples = all_samples
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        bgr = cv2.imread(str(s.path), cv2.IMREAD_COLOR)
        if bgr is None:
            return self.__getitem__((idx + 1) % len(self))
        if bgr.shape[:2] != (224, 224):
            bgr = _letterbox(bgr, 224)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment_fn is not None:
            rgb = self.augment_fn(rgb)
        img = rgb.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(img),
                torch.tensor([float(s.label)], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Subject-disjoint split over UTA only
# ---------------------------------------------------------------------------

def split_uta_subjects(out_root: str | Path,
                       val_frac: float = 0.15,
                       test_frac: float = 0.15,
                       seed: int = 42,
                       ) -> tuple[list[str], list[str], list[str]]:
    """Return (train_subjects, val_subjects, test_subjects) where each list
    is a disjoint partition of UTA subject ids. Uses simple random group
    assignment — UTA has 60 subjects, enough that random splits balance
    out without needing the deficit-dealer used for DDD's few groups.
    """
    out_root = Path(out_root)
    subjects = sorted(p.name for p in out_root.iterdir() if p.is_dir())
    if not subjects:
        raise ValueError(f"no extracted subjects under {out_root}")
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_total = len(subjects)
    n_test = max(1, int(round(n_total * test_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    test = subjects[:n_test]
    val = subjects[n_test:n_test + n_val]
    train = subjects[n_test + n_val:]
    log.info("UTA split: train=%d  val=%d  test=%d subjects",
             len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UTA-RLDD video → face-crop extractor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract face JPEGs from MP4s.")
    pe.add_argument("--root", default="data/uta-rldd",
                    help="Root of the downloaded UTA-RLDD layout. "
                         "The Kaggle mirror lays it out under "
                         "data/uta-rldd/Fold{1..N}_part{1,2}/...")
    pe.add_argument("--out", default="data/uta_rldd_frames",
                    help="Where to write extracted face JPEGs.")
    pe.add_argument("--artifacts", default="artifacts",
                    help="Where to cache the MediaPipe face_landmarker model.")
    pe.add_argument("--every", type=int, default=30,
                    help="Sample every Nth frame (default 30 ≈ 1 fps).")
    pe.add_argument("--max-videos", type=int, default=0,
                    help="If >0, process only the first N videos (smoke test).")
    pe.add_argument("--force", action="store_true",
                    help="Re-extract videos whose output dir already has frames.")

    ps = sub.add_parser("stats", help="Report extracted-frame counts per subject.")
    ps.add_argument("--out", default="data/uta_rldd_frames")
    ps.add_argument("--per-subject", action="store_true")

    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)
    if args.cmd == "extract":
        _cmd_extract(args)
    elif args.cmd == "stats":
        _cmd_stats(args)
    else:
        sys.exit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
