# Driver Drowsiness Detection

ENGG2112 group project. A deep-learning pipeline that detects driver
drowsiness from a cabin-facing camera, with a real-time webcam demo
combining trained CNN classifiers and classical landmark-based signals.

## Headline result

MobileNetV2 with cabin-camera-aware augmentation reached **70.5% macro-F1**
and **99.4% drowsy recall at the safety operating point** on a fully
held-out, subject-disjoint test set of 7,504 frames.

| Model              | Threshold | Macro-F1 | Drowsy recall | ROC-AUC |
| ------------------ | --------- | -------- | ------------- | ------- |
| BaselineCNN        | 0.50      | 0.637    | 0.671         | 0.601   |
| AlexNet (TL)       | 0.50      | 0.520    | 0.468         | 0.531   |
| **MobileNetV2 (TL)**   | **0.50**  | **0.705**| **0.870**     | **0.663** |
| MobileNetV2 (TL)   | safety    | 0.566    | **0.994**     | —       |
| Two-stream (eye)   | 0.50      | 0.902    | 0.948         | 0.977   |
| Two-stream (face)  | 0.50      | 0.474    | 0.657         | 0.597   |

Full table at [`artifacts/comparison.md`](artifacts/comparison.md).
Re-generate with `py -m src.compare`.

## Layout

```
drowsiness_detection/
├── README.md                     this file
├── .gitignore
├── check_real_data.py            sanity-check the raw MRL + DDD downloads
├── export_dataset.py             pack MRL + DDD into a single SQLite bundle
├── data/
│   ├── drowsiness.db             ~3 GB combined dataset (NOT committed)
│   ├── README.txt                teammate onboarding
│   └── explore_*.py              early data-exploration starters
├── docs/
│   └── DATASETS.md               source-dataset specs + unification rationale
├── src/
│   ├── datasets.py               two-stream Dataset + SQLite reader + splitting
│   ├── data_single_stream.py     face-only view used by Models 1–3
│   ├── augmentations.py          cabin-camera-aware augmentation pipeline
│   ├── models/
│   │   ├── baseline_cnn.py       Model 1 — from-scratch CNN
│   │   ├── alexnet_tl.py         Model 2 — AlexNet transfer learning
│   │   ├── mobilenet_v2.py       Model 3 — MobileNetV2 transfer learning
│   │   └── two_stream.py         Model 4 — eye + face fusion
│   ├── train.py                  trains BaselineCNN / AlexNet / MobileNetV2
│   ├── train_fusion.py           trains the two-stream fusion model
│   ├── eval.py                   test-set evaluation for single-stream models
│   ├── eval_fusion.py            per-branch test-set evaluation for two-stream
│   ├── calibrate.py              threshold sweep + safety operating point
│   ├── compare.py                renders artifacts/comparison.md
│   ├── realtime_demo.py          live webcam demo with EAR/MAR + smoothing
│   └── smoke_test.py             synthetic-data sanity check
└── artifacts/                    saved metrics, confusion matrices, calibration
    ├── baseline_cnn/
    ├── alexnet/
    ├── mobilenet_v2/
    ├── two_stream/
    └── comparison.md             side-by-side test results
```

Model checkpoints (`*.pt`) and the SQLite bundle (`*.db`) are gitignored
and shared out-of-band.

## Setup

```powershell
py -m pip install torch torchvision opencv-python numpy matplotlib tqdm mediapipe
```

For GPU training, install the CUDA-enabled wheels of torch and torchvision
matching your driver. Example for CUDA 12.x:

```powershell
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## How to run

### 1. Build the unified dataset bundle

```powershell
py export_dataset.py --overwrite
```

This packs the raw MRL Eye and Driver Drowsiness datasets into a single
SQLite file at `data/drowsiness.db` (~3 GB). The file is gitignored and
shared between teammates out-of-band.

### 2. Train a single-stream face model

```powershell
py -m src.train --model mobilenet_v2 --epochs 15 --batch-size 128 --num-workers 0
```

Replace `mobilenet_v2` with `baseline_cnn` or `alexnet` for the other two
architectures. Use `--no-augment` to disable the cabin-camera augmentation
pipeline (used to measure its contribution: +0.029 macro-F1 on MobileNetV2).
Results are saved to `artifacts/<model>/`.

### 3. Train the two-stream fusion model

```powershell
py -m src.train_fusion --epochs 15 --batch-size 128 --num-workers 0
```

### 4. Evaluate on the test set

```powershell
py -m src.eval --model mobilenet_v2 --num-workers 0 --batch-size 256
py -m src.eval_fusion --num-workers 0 --batch-size 256
```

Generates `test_metrics.json` and a confusion-matrix PNG per model.

### 5. Calibrate thresholds (single-stream models only)

```powershell
py -m src.calibrate --model mobilenet_v2 --num-workers 0
```

Sweeps thresholds from 0.05 to 0.95 on validation, picks two operating
points (best macro-F1 and the largest threshold keeping val drowsy recall
≥ 0.95), and re-evaluates on test at each.

### 6. Render the comparison table

```powershell
py -m src.compare
```

Reads every `artifacts/<model>/test_metrics.json` and writes
`artifacts/comparison.md`.

### 7. Run the real-time demo

```powershell
py -m src.realtime_demo --use-ear
```

On first run, downloads MediaPipe's `face_landmarker.task` (~3 MB) into
`artifacts/`. Press `q` or `Esc` to quit. Useful flags:

| Flag                  | Default | Purpose                                                                |
| --------------------- | ------- | ---------------------------------------------------------------------- |
| `--use-ear`           | off     | Drive the alarm decision from EAR/MAR instead of CNN fusion            |
| `--threshold`         | 0.5     | Smoothed-PERCLOS threshold for the alarm                               |
| `--window`            | 30      | Smoothing window in frames (≈ 1 second at 30 fps)                      |
| `--hysteresis`        | 3       | Consecutive frames required to switch ALERT ↔ DROWSY                   |
| `--ear-threshold`     | 0.20    | Eye Aspect Ratio below this counts as eyes-closed                      |
| `--mar-threshold`     | 0.55    | Mouth Aspect Ratio above this counts as yawning                        |
| `--face-only`         | off     | Disable the eye branch                                                 |
| `--eye-only`          | off     | Disable the face model                                                 |
| `--camera N`          | 0       | Webcam index                                                           |
| `--video file.mp4`    | —       | Run on a video file instead of webcam                                  |
| `--record out.mp4`    | —       | Save the annotated overlay to disk                                     |

## Live demo pipeline

```
webcam frame
   │
   ▼
MediaPipe FaceLandmarker (478 points)
   │
   ├─► face crop (224x224)  ──► MobileNetV2  ──► P_face
   │
   ├─► eye crop (64x64)     ──► EyeStateCNN  ──► P_eye
   │
   ├─► EAR per eye          ──► binary "eye closed" event
   │
   └─► MAR                  ──► binary "yawn" event
                                       │
                                       ▼
                       max(events)  →  rolling smoother (PERCLOS)
                                       │
                                       ▼
                              hysteresis  →  ALERT / DROWSY
```

## Limitations

The CNN models were trained on cabin-camera (DDD) and controlled-condition
(MRL) imagery. On a laptop webcam at desk distance the visual distribution
differs enough that both models saturate — `P_face` pins near 1.0 and
`P_eye` near 0.1 regardless of actual state. EAR and MAR, computed
geometrically from MediaPipe landmarks, remain accurate. The deployed demo
uses `--use-ear` to drive the alarm decision from the classical signals
while keeping the CNN probabilities visible for diagnostic purposes.

A real cabin deployment would not have this gap because the deployment
camera matches the training distribution. Closing it for consumer webcams
would require training on more diverse cameras and lighting conditions —
flagged as future work.

## Loading the dataset programmatically

```python
from torch.utils.data import DataLoader
from src.datasets import SQLiteDrowsinessDataset, make_weighted_sampler

train_ds = SQLiteDrowsinessDataset("data/drowsiness.db", split="train", augment=True)
val_ds   = SQLiteDrowsinessDataset("data/drowsiness.db", split="val")
test_ds  = SQLiteDrowsinessDataset("data/drowsiness.db", split="test")

sampler = make_weighted_sampler(train_ds.samples)
loader  = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
```

Each sample is a dict:

```python
{
    "eye":       Tensor[3, 64, 64],       # zeros if sample is face-stream
    "face":      Tensor[3, 224, 224],     # zeros if sample is eye-stream
    "eye_mask":  1.0 or 0.0,              # 1.0 → eye stream is valid
    "face_mask": 1.0 or 0.0,              # 1.0 → face stream is valid
    "label":     0.0 (alert) or 1.0 (drowsy),
    "source":    "mrl" or "ddd",
}
```

The masks are what let a single model train on two structurally different
datasets — the loss for a sample is only computed on the stream whose mask
is 1.

For face-only training (Models 1–3), use `FaceStreamDataset` from
`src/data_single_stream.py`, which filters the bundle to DDD face frames
and returns plain `(image, label)` tuples.

## Dataset details

See [`docs/DATASETS.md`](docs/DATASETS.md) for source-dataset specs, label
unification, two-stream input strategy, and split rationale.
