# Driver Drowsiness Detection

ENGG2112 group project. A deep-learning pipeline that detects driver drowsiness
from a cabin-facing camera, using two public datasets (MRL Eye + Driver
Drowsiness Dataset) unified into a single training stream.

## Status

| Stage | What's here | Done? |
|---|---|---|
| Data integration | unified `{alert, drowsy}` schema across MRL + DDD, two-stream sample format, stratified subject-wise split, class-balancing utilities, single-file SQLite bundle for sharing | ✅ |
| Data exploration | starter scripts for quantitative analysis and visual inspection (`data/explore_*.py`) | in progress |
| Augmentations | cabin-camera-appropriate pipeline (flip, small rotation, brightness/contrast, blur, cutout) | ✅ |
| Models | two-stream: `EyeStateCNN` + `FaceMobileNetV3` + fusion head | pending |
| Training loop | masked BCE, weighted sampling, early stopping on val macro-F1 | pending |
| Real-time inference | MediaPipe face-mesh → eye+face crops → temporal smoothing | pending |

## Layout

```
drowsiness_detection/
├── README.md                     this file
├── .gitignore
├── check_real_data.py            quick sanity check against real datasets
├── export_dataset.py             pack MRL + DDD into a single .db bundle
├── data/
│   ├── drowsiness.db             3 GB combined dataset (NOT committed)
│   ├── README.txt                teammate onboarding
│   ├── explore_stats.py          Teammate A starter (quantitative)
│   └── explore_images.py         Teammate B starter (visual)
├── docs/
│   └── DATASETS.md               dataset specs + unification rationale
└── src/
    ├── __init__.py
    ├── datasets.py               two-stream Dataset, SQLite bundle, splitting
    ├── augmentations.py          training-time augmentations
    └── smoke_test.py             end-to-end pipeline check on synthetic data
```

## Quick start

```powershell
py -m pip install torch opencv-python-headless numpy matplotlib

# 1. Verify the pipeline works on synthetic data (no real datasets needed)
py -m src.smoke_test

# 2. (optional) Verify against your real MRL + DDD downloads
py check_real_data.py

# 3. Pack both datasets into a single 3 GB shareable file
py export_dataset.py --overwrite
```

## Loading the combined dataset

Once `data/drowsiness.db` exists (either because you ran step 3 above, or a
teammate shared it with you):

```python
from torch.utils.data import DataLoader
from src.datasets import SQLiteDrowsinessDataset, make_weighted_sampler

train_ds = SQLiteDrowsinessDataset("data/drowsiness.db", split="train", augment=True)
val_ds   = SQLiteDrowsinessDataset("data/drowsiness.db", split="val")
test_ds  = SQLiteDrowsinessDataset("data/drowsiness.db", split="test")

sampler = make_weighted_sampler(train_ds.samples)
loader  = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
```

Each sample is a dict:

```python
{
    "eye":       Tensor[3, 64, 64],      # zeros if sample is face-stream
    "face":      Tensor[3, 224, 224],    # zeros if sample is eye-stream
    "eye_mask":  1.0 or 0.0,              # 1.0 → eye stream is valid
    "face_mask": 1.0 or 0.0,
    "label":     0.0 (alert) or 1.0 (drowsy),
    "source":    "mrl" or "ddd",
}
```

The masks are what let a single model train on two structurally different
datasets — the loss for a sample is only computed on the stream whose mask is 1.

## Dataset details

See [`docs/DATASETS.md`](docs/DATASETS.md) for:
- native structure of each source dataset (MRL naming convention, DDD folder
  layouts — both 4-class and 2-class releases are supported)
- label unification (`{alert, drowsy}`) with rationale
- two-stream input strategy (eye-only 64×64 vs full-face 224×224)
- split strategy (stratified group-wise — same subject never crosses partitions)
- handling of resolution / colour / label-noise inconsistencies

