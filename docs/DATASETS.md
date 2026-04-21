# Dataset Specifications

This document describes the two datasets used in the drowsiness detection system, their native structures, and how we unify them.

---

## 1. MRL Eye Dataset

**Source:** Media Research Lab, VŠB – Technical University of Ostrava
**URL:** http://mrl.cs.vsb.cz/eyedataset

### Structure
- **~84,898 eye-region images** from 37 subjects (33 male, 4 female)
- Captured under varied conditions (glasses / no glasses, low / high illumination, IR / RGB sensors)
- Grayscale, cropped tightly to the eye region, variable resolutions (most near 86×86)

### File-naming convention
Each filename encodes seven attributes separated by underscores:
```
sNNNN_MMMMM_G_g_e_r_l_.png
  │     │    │ │ │ │ │
  │     │    │ │ │ │ └─ reflections (0=none, 1=low, 2=high)
  │     │    │ │ │ └─── lighting (0=bad, 1=good)
  │     │    │ │ └───── eye state (0=closed, 1=open)  ← our label
  │     │    │ └─────── glasses (0=no, 1=yes)
  │     │    └───────── gender (0=male, 1=female)
  │     └────────────── image id within subject
  └──────────────────── subject id (s0001…s0037)
```

### Label we extract
- **Field 5 (`e`)**: `0` → closed, `1` → open
- Rough class balance: ~48% closed / ~52% open (near-balanced)

---

## 2. Driver Drowsiness Dataset (DDD)

**Source:** Publicly released driver drowsiness dataset; several variants on Kaggle. The loader auto-detects which release you have by inspecting the class folder names.

### Supported layouts

**2-class release** (e.g. Ismail Nasri "Driver Drowsiness Dataset" on Kaggle — the one currently in use):
```
DDD/
├── Drowsy/          # closed eyes / yawning / fatigued driver
└── Non Drowsy/      # alert driver
```

**4-class release** (older YawDD-derived releases):
```
DDD/
├── yawn/            # mouth open wide, yawning
├── no_yawn/         # neutral / talking / closed mouth
├── Closed/          # eyes closed (full-face crop)
└── Open/            # eyes open (full-face crop)
```

Folder-name matching is case-insensitive and normalises spaces → underscores
(`Non Drowsy` → `non_drowsy`), so both releases work with the same code.

### Properties
- **~40,000–45,000 images** for the 2-class release; ~2,900 for the 4-class release
- Full-face RGB crops, JPG, variable resolution (~145×145 up to 1280×720)
- Real drivers filmed in cars → realistic illumination, pose variation, motion blur
- Filenames encode a per-subject prefix (e.g. `A0001.png, A0002.png, …`)
  which the loader uses as a pseudo-subject id for group-wise splitting

---

## 3. Unification Strategy

The two datasets disagree on everything: resolution, color space, crop (eye-only vs full-face), and label schema. We reconcile them with a two-stage unification.

### 3.1 Label mapping → binary `{alert, drowsy}`

| Source dataset | Native label | Unified label | Rationale |
|---|---|---|---|
| MRL Eye        | `open` (e=1)      | `alert`  | Open eye is the single strongest alertness signal |
| MRL Eye        | `closed` (e=0)    | `drowsy` | Sustained closure ≈ microsleep |
| DDD (2-class)  | `Non Drowsy`      | `alert`  | Frames from alert-state recordings |
| DDD (2-class)  | `Drowsy`          | `drowsy` | Frames from drowsy-state recordings |
| DDD (4-class)  | `Open`            | `alert`  | Matches MRL `open` |
| DDD (4-class)  | `Closed`          | `drowsy` | Matches MRL `closed` |
| DDD (4-class)  | `no_yawn`         | `alert`  | Neutral facial state |
| DDD (4-class)  | `yawn`            | `drowsy` | Yawning is a validated fatigue indicator |

**Caveat we explicitly document:** a single closed-eye frame is not drowsiness — it's a blink. Binary per-frame labels are the *training* target; the runtime system is expected to apply temporal smoothing (planned, not yet implemented) so brief blinks aren't false positives.

### 3.2 Spatial unification → two-stream architecture

Eye-only crops (MRL) and full-face crops (DDD) are **not interchangeable**. Resizing a 640×480 face to 86×86 destroys the eye signal; upscaling an 86×86 eye to 224×224 wastes capacity on nothing.

We therefore split inputs into two streams that are merged **after** feature extraction:

```
                   ┌──────────────────────┐
eye crop (64×64) ──│ EyeStateCNN          │──┐
                   └──────────────────────┘  │
                                             ├── concat ── MLP ── {alert, drowsy}
                   ┌──────────────────────┐  │
face crop (224×224)│ FaceMobileNetV3      │──┘
                   └──────────────────────┘
```

At inference time, a face detector (MediaPipe / Haar cascade) is expected to produce both crops from a single webcam frame (not yet implemented). During training, each image contributes to whichever stream it naturally belongs to via a per-sample mask (`eye_mask`, `face_mask`) that zeroes out the inactive stream's loss contribution.

### 3.3 Handling inconsistencies

| Issue | Handling |
|---|---|
| Variable image sizes | Aspect-preserving letterbox resize (not squish) to 64×64 (eye) and 224×224 (face) — see `src/datasets.py` `_letterbox` |
| Grayscale vs RGB | MRL grayscale is broadcast to 3 channels; face stream stays RGB |
| Class imbalance | `compute_pos_weight()` for `BCEWithLogitsLoss(pos_weight=…)` and `make_weighted_sampler()` for batch-level rebalancing |
| Duplicate subjects (MRL) | Grouped by `subject_id` (parsed from filename) and split group-wise — same person never appears in both train and val |
| Duplicate subjects (DDD) | Grouped by the leading letter(s) of the filename (case-insensitive), which correspond to subject/video IDs in the Ismail-Nasri-style release. Override `group_key_fn` in `index_ddd` for other releases. |
| Non-uniform split balance | Stratified group-wise split: sample- and drowsy-count deficits are tracked per partition so all three splits see similar class distributions |
| Corrupt / zero-byte images | `_load_image` / `_decode_image` return `None`; `__getitem__` rolls over to the next sample |
| Label noise in DDD | Teammate-B task (see `data/explore_images.py`) flags suspect files; a `configs/ddd_blacklist.txt` file is passed to `index_ddd(blacklist=…)` if present (not yet populated) |

### 3.4 Sharing the unified dataset

`export_dataset.py` packs the combined + split dataset into a single SQLite file (`data/drowsiness.db`, ~3 GB) containing every image's raw bytes plus metadata. Teammates load it through `SQLiteDrowsinessDataset(db_path, split=…)` and never touch the original MRL / DDD folders. The split is deterministic — `seed=42` is stored in the bundle's `meta` table so re-runs and teammates' machines all see the same partitioning.
