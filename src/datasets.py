"""
datasets.py
-----------
Data integration for the drowsiness detection system.

Loads two heterogeneous public datasets and exposes them through a single
unified PyTorch `Dataset` interface with a common binary label schema
{0: alert, 1: drowsy}.

Datasets
--------
1. MRL Eye Dataset      — eye-region crops, label encoded in filename
2. Driver Drowsiness DS — full-face crops, label encoded in folder name

Both are reconciled to a two-stream sample:
    {
        "eye":       Tensor[3, 64, 64]   or zeros + mask=0,
        "face":      Tensor[3, 224, 224] or zeros + mask=0,
        "eye_mask":  1.0 if eye stream is valid,
        "face_mask": 1.0 if face stream is valid,
        "label":     0 (alert) or 1 (drowsy),
        "source":    "mrl" | "ddd",
    }

Downstream training code uses `eye_mask` / `face_mask` so each sample only
contributes loss to the stream it actually has data for.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

ALERT, DROWSY = 0, 1
LABEL_NAMES = {ALERT: "alert", DROWSY: "drowsy"}

# MRL filename field index for eye-state (0=closed, 1=open)
_MRL_EYE_STATE_FIELD = 4

# DDD folder-name → unified label. Folder names are compared lowercased and
# with whitespace→underscore, so "Non Drowsy" matches "non_drowsy". Covers the
# two common public layouts:
#   4-class release: {Open, Closed, yawn, no_yawn}
#   2-class release: {Drowsy, Non Drowsy}  (Ismail Nasri / similar)
_DDD_LABEL_MAP = {
    # 4-class release
    "open":         ALERT,
    "no_yawn":      ALERT,
    "closed":       DROWSY,
    "yawn":         DROWSY,
    # 2-class release
    "non_drowsy":   ALERT,
    "alert":        ALERT,
    "drowsy":       DROWSY,
}


# ---------------------------------------------------------------------------
# Sample record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    path: Path
    label: int               # ALERT or DROWSY
    source: Literal["mrl", "ddd"]
    stream: Literal["eye", "face"]   # which input branch this image feeds
    subject_id: str | None = None    # for subject-wise splitting (MRL only)


# ---------------------------------------------------------------------------
# Index builders — one per dataset, both emit List[Sample]
# ---------------------------------------------------------------------------

def index_mrl(root: Path) -> list[Sample]:
    """
    MRL files look like:  s0001_00001_0_0_1_0_0_.png
    Field indices (0-based): [subject, imgid, gender, glasses, state, light, reflect]
    We only need: subject (0) and state (4).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MRL root not found: {root}")

    samples: list[Sample] = []
    skipped = 0
    for path in root.rglob("*.png"):
        parts = path.stem.split("_")
        if len(parts) < 7:
            skipped += 1
            continue
        try:
            state = int(parts[_MRL_EYE_STATE_FIELD])  # 0=closed, 1=open
        except ValueError:
            skipped += 1
            continue
        label = ALERT if state == 1 else DROWSY
        samples.append(Sample(
            path=path,
            label=label,
            source="mrl",
            stream="eye",
            subject_id=parts[0],
        ))

    log.info("MRL: loaded %d samples (%d skipped due to malformed names)",
             len(samples), skipped)
    return samples


def _default_ddd_group_key(path: Path) -> str:
    """Pseudo-subject id for DDD frames, used for group-wise splitting.

    Recognises two common filename conventions:
      1. Underscore-separated: ``subject01_123.jpg``  → ``subject01``
      2. Alpha-prefix + index: ``A0001.png``         → ``A``
         (the Ismail Nasri DDD release is this style — each subject / video
          gets a distinct leading letter)

    Override ``group_key_fn`` in :func:`index_ddd` if your release differs.
    """
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0].lower()
    # Leading alphabetic run — captures 'A' from 'A0001', 'subj' from 'subj42'.
    # Case-folded so that releases that use 'A' in one class folder and 'a' in
    # the other (same physical subject, different state) group together.
    i = 0
    while i < len(stem) and stem[i].isalpha():
        i += 1
    if 0 < i < len(stem):
        return stem[:i].lower()
    # Coarse fallback: class folder + first 3 chars.
    return f"{path.parent.name}:{stem[:3].lower()}"


def index_ddd(
    root: Path,
    blacklist: set[str] | None = None,
    group_key_fn: Callable[[Path], str] = _default_ddd_group_key,
) -> list[Sample]:
    """
    DDD layout:
        root/{yawn,no_yawn,Closed,Open}/*.{jpg,jpeg,png}
    Folder names are case-insensitive here.

    ``group_key_fn`` extracts a pseudo-subject id from each filename so that
    :func:`split_samples` can split at the person/video level rather than
    frame-wise (which would leak near-duplicate frames between train and val).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DDD root not found: {root}")

    blacklist = blacklist or set()
    samples: list[Sample] = []
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        key = cls_dir.name.strip().lower().replace(" ", "_")
        if key not in _DDD_LABEL_MAP:
            log.warning("DDD: skipping unknown class folder %s", cls_dir.name)
            continue
        label = _DDD_LABEL_MAP[key]
        for path in cls_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if path.name in blacklist:
                continue
            samples.append(Sample(
                path=path,
                label=label,
                source="ddd",
                stream="face",
                subject_id=group_key_fn(path),
            ))

    n_groups = len({s.subject_id for s in samples})
    log.info("DDD: loaded %d samples across %d groups", len(samples), n_groups)
    if n_groups < 5 and len(samples) > 20:
        log.warning(
            "DDD: only %d groups for %d samples — grouping key is likely "
            "wrong for this release; override group_key_fn or val/test will "
            "leak near-duplicate frames.", n_groups, len(samples))
    return samples


# ---------------------------------------------------------------------------
# Image loading helpers — aspect-preserving letterbox
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, target: int, pad_value: int = 0) -> np.ndarray:
    """Resize keeping aspect ratio, pad the short side to `target`x`target`."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    if resized.ndim == 2:
        canvas = np.full((target, target), pad_value, dtype=resized.dtype)
    else:
        canvas = np.full((target, target, resized.shape[2]),
                         pad_value, dtype=resized.dtype)

    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def _postprocess_image(
    img: np.ndarray | None, target: int, force_rgb: bool = True
) -> np.ndarray | None:
    """Shared pipeline: channel-normalise, letterbox, optionally BGR→RGB."""
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:                                  # grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                            # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = _letterbox(img, target)
    if force_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_image(path: Path, target: int, force_rgb: bool = True) -> np.ndarray | None:
    """Read image from disk, letterbox to target size, return HxWx3 uint8.
    Returns None on failure so the caller can skip the sample."""
    return _postprocess_image(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED), target, force_rgb
    )


def _decode_image(data: bytes, target: int, force_rgb: bool = True) -> np.ndarray | None:
    """Decode raw image bytes (from SQLite BLOB), letterbox, return HxWx3 uint8.
    Returns None on failure so the caller can skip the sample."""
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return _postprocess_image(
        cv2.imdecode(arr, cv2.IMREAD_UNCHANGED), target, force_rgb
    )


# ---------------------------------------------------------------------------
# Unified Dataset
# ---------------------------------------------------------------------------

class DrowsinessDataset(Dataset):
    """Yields two-stream samples from MRL + DDD with masked streams."""

    EYE_SIZE = 64
    FACE_SIZE = 224
    # ImageNet stats — the face branch uses MobileNetV3 pretrained weights
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        samples: list[Sample],
        augment: bool = False,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.samples = samples
        self.augment = augment
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.samples)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))             # HWC → CHW
        return torch.from_numpy(img)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        eye = torch.zeros(3, self.EYE_SIZE, self.EYE_SIZE)
        face = torch.zeros(3, self.FACE_SIZE, self.FACE_SIZE)
        eye_mask = 0.0
        face_mask = 0.0

        if s.stream == "eye":
            img = _load_image(s.path, self.EYE_SIZE)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            if self.augment and self.augment_fn is not None:
                img = self.augment_fn(img)
            eye = self._to_tensor(img)
            eye_mask = 1.0
        else:  # face stream
            img = _load_image(s.path, self.FACE_SIZE)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            if self.augment and self.augment_fn is not None:
                img = self.augment_fn(img)
            face = self._to_tensor(img)
            face_mask = 1.0

        return {
            "eye": eye,
            "face": face,
            "eye_mask": torch.tensor(eye_mask, dtype=torch.float32),
            "face_mask": torch.tensor(face_mask, dtype=torch.float32),
            "label": torch.tensor(s.label, dtype=torch.float32),
            "source": s.source,
        }


# ---------------------------------------------------------------------------
# Splitting — subject-wise for MRL, random for DDD
# ---------------------------------------------------------------------------

def _group_split(
    samples: list[Sample],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split ``samples`` so all frames sharing a ``subject_id`` land in the
    same partition. Falls back to the file path when ``subject_id`` is None,
    which degenerates to a frame-wise split for that sample.

    Uses a stratified group split: groups are sorted by their drowsy-fraction
    and then dealt round-robin into train / val / test buckets. This keeps each
    partition's class balance close to the global balance, which matters when
    there are few groups (e.g. ~37 MRL subjects) — a purely random subject
    shuffle can produce a val set that's 70% alert and a test set that's 70%
    drowsy purely by chance.
    """
    by_group: dict[str, list[Sample]] = {}
    for s in samples:
        key = s.subject_id if s.subject_id is not None else str(s.path)
        by_group.setdefault(key, []).append(s)

    # Shuffle group order, then let the dual-target (sample + drowsy count)
    # deficit dealer below balance things. Any systematic ordering (e.g.
    # sorted by drowsy-fraction) biases which bucket sees which groups first
    # and the small buckets fill with whatever came in early before the
    # deficit logic can counter-balance.
    groups = list(by_group.keys())
    rng.shuffle(groups)

    # Targets are now sample-count and drowsy-count proportions — not group
    # counts. MRL subjects vary 5× in size and 0→100% in drowsy-fraction, so
    # balancing by group count alone leaves partition class distributions
    # skewed. Deal each group to whichever bucket has the largest combined
    # (samples, drowsy) deficit relative to target.
    train_frac = 1.0 - val_frac - test_frac
    n_total    = sum(len(by_group[g]) for g in groups)
    n_drowsy   = sum(sum(1 for s in by_group[g] if s.label == DROWSY)
                     for g in groups)

    targets_n = {"train": n_total  * train_frac,
                 "val":   n_total  * val_frac,
                 "test":  n_total  * test_frac}
    targets_d = {"train": n_drowsy * train_frac,
                 "val":   n_drowsy * val_frac,
                 "test":  n_drowsy * test_frac}
    filled_n  = {"train": 0, "val": 0, "test": 0}
    filled_d  = {"train": 0, "val": 0, "test": 0}
    assigned  = {"train": [], "val": [], "test": []}

    for g in groups:
        lst = by_group[g]
        n_g = len(lst)
        d_g = sum(1 for s in lst if s.label == DROWSY)

        def deficit(b: str) -> float:
            # Sum of (1 - filled/target) across both axes. Largest deficit wins.
            samp = 1.0 - (filled_n[b] / targets_n[b] if targets_n[b] else 1.0)
            drow = 1.0 - (filled_d[b] / targets_d[b] if targets_d[b] else 1.0)
            return samp + drow

        bucket = max(filled_n, key=lambda b: (deficit(b), rng.random()))
        assigned[bucket].append(g)
        filled_n[bucket] += n_g
        filled_d[bucket] += d_g

    # Guarantee non-empty val / test even on tiny datasets.
    for required in ("val", "test"):
        if not assigned[required] and assigned["train"]:
            assigned[required].append(assigned["train"].pop())

    train_g = set(assigned["train"])
    val_g   = set(assigned["val"])
    test_g  = set(assigned["test"])

    train, val, test = [], [], []
    for g, lst in by_group.items():
        if g in test_g:   test.extend(lst)
        elif g in val_g:  val.extend(lst)
        else:             train.extend(lst)
    return train, val, test


def split_samples(
    mrl: list[Sample],
    ddd: list[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Group-wise split for both datasets — MRL by subject, DDD by the
    pseudo-subject id assigned in :func:`index_ddd`."""
    rng = random.Random(seed)

    mrl_train, mrl_val, mrl_test = _group_split(mrl, val_frac, test_frac, rng)
    ddd_train, ddd_val, ddd_test = _group_split(ddd, val_frac, test_frac, rng)

    train = mrl_train + ddd_train
    val   = mrl_val   + ddd_val
    test  = mrl_test  + ddd_test
    rng.shuffle(train)

    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# Class balancing utilities
# ---------------------------------------------------------------------------

def compute_pos_weight(samples: list[Sample]) -> float:
    """pos_weight for BCEWithLogitsLoss = N_neg / N_pos.
    Drowsy = positive class."""
    n_pos = sum(1 for s in samples if s.label == DROWSY)
    n_neg = len(samples) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def make_weighted_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    """Over-samples the minority class so each batch is roughly balanced."""
    labels = np.array([s.label for s in samples])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(samples),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# One-call convenience builder
# ---------------------------------------------------------------------------

def build_datasets(
    mrl_root: str | Path,
    ddd_root: str | Path,
    augment_fn: Callable | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[DrowsinessDataset, DrowsinessDataset, DrowsinessDataset, dict]:
    """
    Returns (train_ds, val_ds, test_ds, info_dict).
    info_dict contains pos_weight and label distributions for logging.
    """
    mrl = index_mrl(Path(mrl_root))
    ddd = index_ddd(Path(ddd_root))

    train_s, val_s, test_s = split_samples(
        mrl, ddd, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    train_ds = DrowsinessDataset(train_s, augment=True, augment_fn=augment_fn)
    val_ds   = DrowsinessDataset(val_s,   augment=False)
    test_ds  = DrowsinessDataset(test_s,  augment=False)

    info = {
        "pos_weight": compute_pos_weight(train_s),
        "counts": {
            "train": _count_labels(train_s),
            "val":   _count_labels(val_s),
            "test":  _count_labels(test_s),
        },
        "source_counts": {
            "mrl": len(mrl),
            "ddd": len(ddd),
        },
    }
    return train_ds, val_ds, test_ds, info


def _count_labels(samples: list[Sample]) -> dict[str, int]:
    out = {"alert": 0, "drowsy": 0}
    for s in samples:
        out[LABEL_NAMES[s.label]] += 1
    return out


# ---------------------------------------------------------------------------
# Combined-dataset SQLite bundle
# ---------------------------------------------------------------------------
#
# Packs both source datasets + the unified split into a single .db file that
# can be copied between team members without needing MRL/DDD folders alongside.
# Images are stored as raw bytes (still JPEG/PNG compressed) so preprocessing
# stays configurable at load time.
#
# Schema:
#   samples(id, split, image_bytes BLOB, label, source, stream, subject_id,
#           filename)
#   meta(key TEXT PRIMARY KEY, value TEXT)
#
# Typical size for MRL (84,898 PNG) + DDD (41,793 JPG): ~2 GB.

import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    split       TEXT    NOT NULL,
    image_bytes BLOB    NOT NULL,
    label       INTEGER NOT NULL,
    source      TEXT    NOT NULL,
    stream      TEXT    NOT NULL,
    subject_id  TEXT,
    filename    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def export_to_sqlite(
    mrl_root: str | Path,
    ddd_root: str | Path,
    db_path: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Build the unified dataset and write it to a single SQLite file.

    Returns the same ``info`` dict as :func:`build_datasets` plus the output
    path and total on-disk size.
    """
    db_path = Path(db_path)
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{db_path} already exists; pass overwrite=True to replace it."
            )
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    mrl = index_mrl(Path(mrl_root))
    ddd = index_ddd(Path(ddd_root))
    train_s, val_s, test_s = split_samples(
        mrl, ddd, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        # Larger page cache + WAL off for bulk insert; safe because we are
        # writing a fresh file.
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA cache_size = -200000;")  # ~200 MB

        def _ingest(split_name: str, samples: list[Sample]) -> None:
            batch: list[tuple] = []
            BATCH = 500
            for i, s in enumerate(samples, 1):
                try:
                    data = s.path.read_bytes()
                except OSError as exc:
                    log.warning("skipping unreadable file %s: %s", s.path, exc)
                    continue
                batch.append((
                    split_name, data, int(s.label), s.source, s.stream,
                    s.subject_id, s.path.name,
                ))
                if len(batch) >= BATCH:
                    conn.executemany(
                        "INSERT INTO samples "
                        "(split, image_bytes, label, source, stream, subject_id, filename) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
                if i % 5000 == 0:
                    log.info("  %s: %d / %d written", split_name, i, len(samples))
            if batch:
                conn.executemany(
                    "INSERT INTO samples "
                    "(split, image_bytes, label, source, stream, subject_id, filename) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()

        log.info("Writing SQLite bundle → %s", db_path)
        _ingest("train", train_s)
        _ingest("val",   val_s)
        _ingest("test",  test_s)

        meta = {
            "version":         "1",
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "seed":            str(seed),
            "val_frac":        str(val_frac),
            "test_frac":       str(test_frac),
            "eye_size":        str(DrowsinessDataset.EYE_SIZE),
            "face_size":       str(DrowsinessDataset.FACE_SIZE),
            "n_train":         str(len(train_s)),
            "n_val":           str(len(val_s)),
            "n_test":          str(len(test_s)),
            "n_mrl":           str(len(mrl)),
            "n_ddd":           str(len(ddd)),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            list(meta.items()))
        conn.commit()
    finally:
        conn.close()

    return {
        "db_path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "counts": {
            "train": _count_labels(train_s),
            "val":   _count_labels(val_s),
            "test":  _count_labels(test_s),
        },
        "source_counts": {"mrl": len(mrl), "ddd": len(ddd)},
        "pos_weight": compute_pos_weight(train_s),
    }


class SQLiteDrowsinessDataset(Dataset):
    """Two-stream dataset backed by a single .db file produced by
    :func:`export_to_sqlite`. Drop-in replacement for
    :class:`DrowsinessDataset` — same sample dict, same ``.samples`` list,
    same label schema — so :func:`compute_pos_weight` and
    :func:`make_weighted_sampler` work unchanged.
    """

    EYE_SIZE = DrowsinessDataset.EYE_SIZE
    FACE_SIZE = DrowsinessDataset.FACE_SIZE
    _MEAN = DrowsinessDataset._MEAN
    _STD = DrowsinessDataset._STD

    def __init__(
        self,
        db_path: str | Path,
        split: Literal["train", "val", "test"],
        augment: bool = False,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.db_path = str(db_path)
        self.split = split
        self.augment = augment
        self.augment_fn = augment_fn

        # Thread-/process-local SQLite handle. A single handle can't be shared
        # across forked DataLoader workers — we open lazily in __getitem__.
        self._local = threading.local()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, label, source, stream, subject_id, filename "
                "FROM samples WHERE split = ? ORDER BY id", (split,),
            ).fetchall()

        # Re-hydrate as Sample objects so downstream utilities keep working.
        # `path` holds the original filename for debugging only; actual bytes
        # come from the DB.
        self._row_ids: list[int] = []
        self.samples: list[Sample] = []
        for rid, label, source, stream, subject_id, filename in rows:
            self._row_ids.append(rid)
            self.samples.append(Sample(
                path=Path(filename),
                label=int(label),
                source=source,
                stream=stream,
                subject_id=subject_id,
            ))

        if not self.samples:
            raise ValueError(f"No samples found in split={split!r} of {db_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path)
            # Read-only is enforced by never running writes; this avoids
            # WAL-file side effects when multiple workers open the same DB.
            self._local.conn = c
        return c

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        rid = self._row_ids[idx]

        row = self._conn().execute(
            "SELECT image_bytes FROM samples WHERE id = ?", (rid,)
        ).fetchone()
        data = row[0] if row else b""

        eye = torch.zeros(3, self.EYE_SIZE, self.EYE_SIZE)
        face = torch.zeros(3, self.FACE_SIZE, self.FACE_SIZE)
        eye_mask = 0.0
        face_mask = 0.0

        if s.stream == "eye":
            img = _decode_image(data, self.EYE_SIZE)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            if self.augment and self.augment_fn is not None:
                img = self.augment_fn(img)
            eye = self._to_tensor(img)
            eye_mask = 1.0
        else:
            img = _decode_image(data, self.FACE_SIZE)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            if self.augment and self.augment_fn is not None:
                img = self.augment_fn(img)
            face = self._to_tensor(img)
            face_mask = 1.0

        return {
            "eye": eye,
            "face": face,
            "eye_mask": torch.tensor(eye_mask, dtype=torch.float32),
            "face_mask": torch.tensor(face_mask, dtype=torch.float32),
            "label": torch.tensor(s.label, dtype=torch.float32),
            "source": s.source,
        }


def read_sqlite_meta(db_path: str | Path) -> dict[str, str]:
    """Return the metadata stored alongside the samples (schema version,
    creation timestamp, split sizes, etc.)."""
    with sqlite3.connect(str(db_path)) as conn:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
