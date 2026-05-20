# Dataset Specifications

This document describes the three public datasets used in the drowsiness
detection system, their native structures, and how we unify them.

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

## 3. UTA Real-Life Drowsiness Dataset (UTA-RLDD)

**Source:** Ghoddoosian, Galib & Athitsos, *"A Realistic Dataset and Baseline
Temporal Model for Early Drowsiness Detection"*, CVPRW 2019.
**Project page:** https://sites.google.com/view/utarldd/home

UTA-RLDD is added to bridge a distribution-shift gap: DDD is cabin-camera
footage, but the deployment target is a consumer laptop/phone webcam.
UTA-RLDD is exactly that domain — subjects self-recorded themselves on
their own phones/webcams at roughly arm's length.

### Structure
- The original release is **60 subjects, ~30 h of RGB video, 180 videos**
  (3 per subject). The Kaggle mirror used here
  (`rishab260/uta-reallife-drowsiness-dataset`, ~85 GB) contains **48
  subjects** laid out as 4 folds × 2 parts.
- Each subject recorded **three ~10-minute videos**, one per drowsiness
  state, named by a KSS-derived class index:
  - `0`  → **alert**
  - `5`  → **low vigilance** (subtle / borderline)
  - `10` → **drowsy**
- Self-recorded → varied real-world backgrounds, lighting, camera angles,
  and frame rates (always < 30 fps).

### Label we extract
- `0` → `alert`, `10` → `drowsy`.
- **Class `5` (low vigilance) is dropped** — it is an ambiguous middle
  state and we train on the unambiguous endpoints only.

### Frame extraction (`src/uta_rldd.py`)
The raw videos are far too large and redundant to use directly. The
extractor:
- samples **every 30th frame** (≈ 1 fps),
- runs MediaPipe FaceLandmarker to crop the face (15% padding),
- letterboxes each crop to **224×224** and writes it as JPEG to
  `data/uta_rldd_frames/<subject>/<alert|drowsy>/`.

This yields **~54,090 face frames** across 48 subjects, near class-balanced
(~27k alert / ~27k drowsy). Extraction is resumable per video.

---

## 4. Unification Strategy

The datasets disagree on resolution, colour space, crop (eye-only vs
full-face), and label schema. We reconcile them as follows.

### 4.1 Label mapping → binary `{alert, drowsy}`

| Source dataset | Native label   | Unified label | Rationale |
|---|---|---|---|
| MRL Eye        | `open` (e=1)      | `alert`  | Open eye is the strongest alertness signal |
| MRL Eye        | `closed` (e=0)    | `drowsy` | Sustained closure ≈ microsleep |
| DDD (2-class)  | `Non Drowsy`      | `alert`  | Frames from alert-state recordings |
| DDD (2-class)  | `Drowsy`          | `drowsy` | Frames from drowsy-state recordings |
| DDD (4-class)  | `Open` / `no_yawn`| `alert`  | Neutral facial state |
| DDD (4-class)  | `Closed` / `yawn` | `drowsy` | Eye closure / yawning are fatigue indicators |
| UTA-RLDD       | `0`               | `alert`  | Subject's alert-state recording |
| UTA-RLDD       | `10`              | `drowsy` | Subject's drowsy-state recording |
| UTA-RLDD       | `5`               | *(dropped)* | Ambiguous low-vigilance middle class |

**Caveat we explicitly document:** a single closed-eye frame is not
drowsiness — it's a blink. Binary per-frame labels are the *training*
target; the runtime system applies temporal smoothing
(`src/realtime_demo.py`) so brief blinks aren't false positives.

For UTA-RLDD there is an additional caveat: labels are assigned at the
*video* level, so a fraction of frames in a "drowsy" video show an
alert-looking face. This frame-level label noise caps single-frame
accuracy on UTA — the original paper addresses it with a temporal model.

### 4.2 Spatial unification → two-stream architecture (MRL + DDD)

Eye-only crops (MRL) and full-face crops (DDD) are **not interchangeable**.
Resizing a 640×480 face to 86×86 destroys the eye signal; upscaling an
86×86 eye to 224×224 wastes capacity.

We therefore split inputs into two streams merged **after** feature
extraction:

```
                   ┌──────────────────────┐
eye crop (64×64) ──│ EyeStateCNN          │──┐
                   └──────────────────────┘  │
                                             ├── concat ── MLP ── {alert, drowsy}
                   ┌──────────────────────┐  │
face crop (224×224)│ face CNN branch      │──┘
                   └──────────────────────┘
```

During training, each image contributes to whichever stream it naturally
belongs to via a per-sample mask (`eye_mask`, `face_mask`) that zeroes out
the inactive stream's loss. At inference, the live demo uses MediaPipe
FaceLandmarker to locate the face from a webcam frame.

The two-stream model is one of the four architectures evaluated; the
single-stream MobileNetV2 (below) is the deployment model.

### 4.3 UTA-RLDD integration → combined face-stream training

UTA-RLDD is **full-face** imagery, so it belongs to the face stream — it
is *not* added to the two-stream eye branch (which stays MRL-only). Nor is
it packed into the SQLite bundle; the extracted JPEG tree is read directly
from disk.

`src/train_combined.py` concatenates two face-stream datasets:
- DDD face frames — `FaceStreamDataset`, filtered from the SQLite bundle;
- UTA-RLDD face crops — `UtaRldDataset`, from `data/uta_rldd_frames/`.

Both yield identical `(Tensor[3,224,224], label)` tuples, so a single
`ConcatDataset` feeds the training loop. Test metrics are reported
separately on DDD, UTA, and the combined set so per-domain generalisation
is visible. UTA-RLDD subjects are split subject-disjointly (34 train /
7 val / 7 test).

### 4.4 Handling inconsistencies

| Issue | Handling |
|---|---|
| Variable image sizes | Aspect-preserving letterbox resize (not squish) to 64×64 (eye) and 224×224 (face) — see `src/datasets.py` `_letterbox` |
| Grayscale vs RGB | MRL grayscale is broadcast to 3 channels; face streams stay RGB |
| Colour temperature | Augmentation includes a warm/cool colour-cast jitter so the model is not thrown by indoor lighting it never saw in training |
| Class imbalance | `compute_pos_weight()` for `BCEWithLogitsLoss(pos_weight=…)` and `make_weighted_sampler()` for batch-level rebalancing |
| Duplicate subjects (MRL) | Grouped by `subject_id` (parsed from filename) and split group-wise |
| Duplicate subjects (DDD) | Grouped by the leading letter(s) of the filename (case-insensitive) |
| Duplicate subjects (UTA) | Grouped by fold/part/subject-number; split subject-disjointly |
| Non-uniform split balance | Stratified group-wise split: sample- and drowsy-count deficits tracked per partition |
| Corrupt / zero-byte images | `_load_image` / `_decode_image` return `None`; `__getitem__` rolls over to the next sample |

### 4.5 Sharing the unified dataset

`export_dataset.py` packs the combined + split MRL/DDD dataset into a
single SQLite file (`data/drowsiness.db`, ~3 GB) containing every image's
raw bytes plus metadata. The split is deterministic — `seed=42` is stored
in the bundle's `meta` table so re-runs and teammates' machines all see
the same partitioning.

UTA-RLDD is *not* in the bundle (it would add tens of GB). It is
downloaded separately, extracted once with `py -m src.uta_rldd extract`,
and read from `data/uta_rldd_frames/` thereafter.
