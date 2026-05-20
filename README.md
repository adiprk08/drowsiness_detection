# Driver Drowsiness Detection

ENGG2112 group project. A deep-learning pipeline that detects driver
drowsiness from face imagery, with a real-time webcam demo built on a
MobileNetV2 classifier trained across two complementary public datasets.

## Results

The project ran two experiments.

**Experiment 1 — architecture comparison (DDD only).** Four architectures
trained and evaluated on the cabin-camera Driver Drowsiness Dataset (DDD):

| Model                     | Macro-F1 | Drowsy recall | ROC-AUC |
| ------------------------- | -------- | ------------- | ------- |
| BaselineCNN               | 0.637    | 0.671         | 0.601   |
| AlexNet (TL)              | 0.520    | 0.468         | 0.531   |
| MobileNetV2 (TL)          | 0.705    | 0.870         | 0.663   |
| Two-stream — eye branch   | 0.902    | 0.948         | 0.977   |
| Two-stream — face branch  | 0.474    | 0.657         | 0.597   |

**Experiment 2 — combined training (DDD + UTA-RLDD).** The DDD-trained
models saturate on consumer-webcam footage — a distribution-shift gap. We
integrated UTA-RLDD (48 subjects self-recorded on phones/webcams) and
retrained the three single-stream architectures on the union of the two
datasets, reporting test macro-F1 separately per domain:

| Model               | DDD test | UTA test | Combined |
| ------------------- | -------- | -------- | -------- |
| BaselineCNN         | 0.889    | 0.666    | 0.776    |
| AlexNet (TL)        | 0.938    | 0.595    | 0.776    |
| **MobileNetV2 (TL)**| **0.887**| **0.740**| **0.813**|

All numbers are macro-F1 at threshold 0.5 on fully held-out,
subject-disjoint test sets. Adding UTA-RLDD and tuning regularisation
(dropout, weight decay, label smoothing) lifted DDD-test performance well
past the Experiment-1 baseline (MobileNetV2 0.705 → 0.887), and AlexNet's
Experiment-1 collapse turned out to be data-bound rather than
architectural — it recovers to 0.776 with the larger subject pool.

The **deployment model is MobileNetV2 trained on the combined set**: the
best combined score (0.813) and by far the strongest on the held-out
webcam domain (UTA 0.740, vs ≤0.67 for the others). AlexNet edges it on
cabin-camera DDD but transfers poorly to webcam footage — the domain that
matters for deployment. Full table at
[`artifacts/comparison.md`](artifacts/comparison.md); regenerate with
`py -m src.compare`.

## Layout

```
drowsiness_detection/
├── README.md                     this file
├── .gitignore
├── check_real_data.py            sanity-check the raw MRL + DDD downloads
├── export_dataset.py             pack MRL + DDD into a single SQLite bundle
├── data/                         drowsiness.db + UTA-RLDD frames — see below
├── docs/
│   └── DATASETS.md               source-dataset specs + unification rationale
├── src/
│   ├── datasets.py               two-stream Dataset + SQLite reader + splitting
│   ├── data_single_stream.py     face-only view of the SQLite bundle
│   ├── uta_rldd.py               UTA-RLDD video → face-crop extractor + dataset
│   ├── augmentations.py          geometric + photometric + colour-temperature augmentation
│   ├── models/
│   │   ├── baseline_cnn.py       Model 1 — from-scratch CNN
│   │   ├── alexnet_tl.py         Model 2 — AlexNet transfer learning
│   │   ├── mobilenet_v2.py       Model 3 — MobileNetV2 transfer learning
│   │   └── two_stream.py         Model 4 — eye + face fusion
│   ├── train.py                  trains single-stream models on DDD
│   ├── train_combined.py         trains single-stream models on DDD + UTA-RLDD
│   ├── train_fusion.py           trains the two-stream fusion model
│   ├── eval.py                   test-set evaluation for single-stream models
│   ├── eval_fusion.py            per-branch test-set evaluation for two-stream
│   ├── calibrate.py              threshold sweep + safety operating point
│   ├── compare.py                renders artifacts/comparison.md
│   ├── realtime_demo.py          live single-stream webcam demo
│   └── smoke_test.py             synthetic-data sanity check
└── artifacts/                    saved metrics, confusion matrices, calibration
    ├── baseline_cnn/  alexnet/  mobilenet_v2/        DDD-only runs
    ├── baseline_cnn_combined/  alexnet_combined/  mobilenet_v2_combined/
    ├── two_stream/
    └── comparison.md             side-by-side test results
```

Model checkpoints (`*.pt`) and the SQLite bundle (`*.db`) are gitignored
because of size.

## Datasets

This project uses three public datasets. See
[`docs/DATASETS.md`](docs/DATASETS.md) for full specs.

**MRL Eye + DDD** are packed into a single SQLite bundle:

📦 **Download `drowsiness.db` (~3 GB):** [Google Drive](https://drive.google.com/drive/folders/16uuGogxat70HFd7qErt2KXN_fA6tz-Rv?usp=drive_link)

After cloning, create `data/` in the repo root and drop `drowsiness.db`
into it — every script expects the bundle at `data/drowsiness.db`.

**UTA-RLDD** is a separate ~85 GB video dataset
([Kaggle: `rishab260/uta-reallife-drowsiness-dataset`](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)).
Download and extract it to `data/uta-rldd/`, then run the frame extractor
(step 4 below) to produce `data/uta_rldd_frames/`. Cite the original
paper, not the Kaggle mirror: Ghoddoosian, Galib & Athitsos, *"A Realistic
Dataset and Baseline Temporal Model for Early Drowsiness Detection"*,
CVPRW 2019.

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

### 1. Build the unified MRL + DDD bundle

```powershell
py export_dataset.py --overwrite
```

Packs the raw MRL Eye and Driver Drowsiness datasets into a single SQLite
file at `data/drowsiness.db` (~3 GB).

### 2. Experiment 1 — train the single-stream models on DDD

```powershell
py -m src.train --model mobilenet_v2 --epochs 15
```

Replace `mobilenet_v2` with `baseline_cnn` or `alexnet`. `--no-augment`
disables the augmentation pipeline. Results land in `artifacts/<model>/`.

### 3. Train the two-stream fusion model

```powershell
py -m src.train_fusion --epochs 15
```

### 4. Extract UTA-RLDD face frames

```powershell
py -m src.uta_rldd extract          # data/uta-rldd/ → data/uta_rldd_frames/
py -m src.uta_rldd stats            # report extracted-frame counts
```

Samples each video at ~1 fps, crops the face with MediaPipe, and writes
224×224 JPEGs. Resumable per-video. ~54k frames across 48 subjects.

### 5. Experiment 2 — train the combined (DDD + UTA-RLDD) models

```powershell
py -m src.train_combined --model mobilenet_v2 --epochs 10
```

Trains on the union of DDD (from the SQLite bundle) and UTA-RLDD (from the
extracted frames), reporting test metrics per domain. Output goes to
`artifacts/<model>_combined/`. `--uta-only` / `--ddd-only` run the
single-dataset ablations.

### 6. Evaluate and calibrate (Experiment-1 models)

```powershell
py -m src.eval --model mobilenet_v2
py -m src.eval_fusion
py -m src.calibrate --model mobilenet_v2
```

`calibrate` sweeps the decision threshold on validation and picks a safety
operating point (largest threshold keeping val drowsy recall ≥ 0.95).

### 7. Render the comparison table

```powershell
py -m src.compare
```

Auto-discovers every `artifacts/*/test_metrics.json` — DDD-only runs,
combined runs, and the two-stream model — and writes
`artifacts/comparison.md`.

### 8. Run the real-time demo

```powershell
py -m src.realtime_demo
```

Runs the deployment model (`artifacts/mobilenet_v2_combined/best.pt`) live
on the webcam. On first run, downloads MediaPipe's `face_landmarker.task`
(~3 MB) into `artifacts/`. Press `q` or `Esc` to quit. Useful flags:

| Flag                | Default                | Purpose                                          |
| ------------------- | ---------------------- | ------------------------------------------------ |
| `--face-ckpt PATH`  | mobilenet_v2_combined  | Override the face-model checkpoint               |
| `--threshold`       | 0.5                    | Decision threshold on the smoothed probability   |
| `--window`          | 30                     | Smoothing window in frames (≈ 1 second at 30 fps)|
| `--hysteresis`      | 3                      | Consecutive frames required to switch ALERT ↔ DROWSY |
| `--camera N`        | 0                      | Webcam index                                     |
| `--video file.mp4`  | —                      | Run on a video file instead of webcam            |
| `--record out.mp4`  | —                      | Save the annotated overlay to disk               |
| `--show-fps`        | off                    | Print FPS to stdout each second                  |

## Live demo pipeline

```
webcam frame
   │
   ▼
MediaPipe FaceLandmarker (478 points)  ──►  face crop (224×224)
   │
   ▼
MobileNetV2 (DDD + UTA-RLDD combined)  ──►  P(drowsy)
   │
   ▼
rolling smoother (window frames)  ──►  smoothed probability
   │
   ▼
hysteresis  ──►  ALERT / DROWSY
```

Temporal smoothing averages out brief blinks (3–5 frames in a 30-frame
window is only ~10–17%) while holding onto sustained eye closure;
hysteresis requires several consecutive smoothed samples to cross the
threshold before the alarm state flips, which stops flickering.

## Limitations

- **Distribution shift, largely closed.** The Experiment-1 models, trained
  only on cabin-camera (DDD) imagery, saturate on consumer webcams.
  Integrating UTA-RLDD — consumer webcam/phone footage — closed most of
  this gap: the combined MobileNetV2 reaches 0.746 macro-F1 on held-out
  UTA subjects and the live demo tracks eye state under normal lighting.
- **Subtle drowsiness is hard frame-wise.** UTA-RLDD labels whole videos,
  so individual "drowsy" frames are noisy, and its drowsiness is subtle by
  design. Single-frame classification has a ceiling here; temporal
  modelling (à la the original UTA-RLDD paper) is the natural next step.
  The inference-time smoother partially compensates.
- **Colour-temperature sensitivity.** The model is robust under neutral
  light; strong warm ("yellow") indoor lighting can still degrade it
  despite colour-temperature augmentation. Per-device calibration would
  close this — standard practice for production driver-monitoring systems.

## Loading the dataset programmatically

```python
from torch.utils.data import DataLoader
from src.datasets import SQLiteDrowsinessDataset, make_weighted_sampler

train_ds = SQLiteDrowsinessDataset("data/drowsiness.db", split="train", augment=True)
val_ds   = SQLiteDrowsinessDataset("data/drowsiness.db", split="val")
test_ds  = SQLiteDrowsinessDataset("data/drowsiness.db", split="test")

sampler = make_weighted_sampler(train_ds.samples)
loader  = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
```

Each SQLite sample is a dict:

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

The masks let a single model train on two structurally different datasets
— a sample's loss is computed only on the stream whose mask is 1.

Convenience wrappers, both yielding plain `(image, label)` tuples:

- `FaceStreamDataset` (`src/data_single_stream.py`) — DDD face frames from
  the SQLite bundle, used by the single-stream models.
- `UtaRldDataset` (`src/uta_rldd.py`) — UTA-RLDD face crops from the
  extracted-frames tree, with a subject-disjoint split.

`src/train_combined.py` concatenates the two for combined training.

## Dataset details

See [`docs/DATASETS.md`](docs/DATASETS.md) for source-dataset specs, label
unification, the two-stream input strategy, UTA-RLDD integration, and
split rationale.
