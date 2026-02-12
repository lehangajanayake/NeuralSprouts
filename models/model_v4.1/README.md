# model_v4.1 — RGB-only dry-weight regressor + viewer

## Overview

Model v4.1 streamlines the earlier multi-branch concept into a single, lightweight RGB regressor. Depth images, logits, and fusion stages were removed to simplify deployment and reduce preprocessing costs. The entire pipeline now operates on aligned RGB crops resized to 96×96 and predicts lettuce dry weight directly with one MAE-optimized head.

Primary metric: **MAE on dry weight**.

## Architecture

`model.py` implements a compact CNN:

- Input: RGB tensor shaped `(N, 3, 96, 96)`.
- Backbone: four `ConvBlock`s (32→64→128→256 channels) with stride-2 pooling per block.
- Head: flatten → `Linear(256*6*6, 512)` → ReLU/Dropout → `Linear(512, 1)`.
- Output: `(N,)` dry-weight predictions.

`predict_dry_weight()` wraps `forward()` and keeps the legacy API name for tooling compatibility.

## Data pipeline

`preprocess.py`

- Reads `datasets/Training/Train.csv` (or augmented splits).
- Mandatory steps per image:
  1. Center-crop to **1000×1000** (fallback: min side crop if images are smaller).
  2. Resize to **96×96**.
  3. Save RGB PNGs plus a refreshed CSV with deterministic augmented IDs.
- Generates `num_aug_per_image` variants via flips/rotations + RGB-only color jitter. Depth outputs were removed.

`dataloader.py`

- `PlantDatasetV4`: loads RGB tensors (float32 in `[0,1]`), optional deterministic augment, and dry-weight targets.
- `TestPlantDatasetV4`: inference-only loader returning IDs + RGB tensors.
- Both datasets share the new crop/resize defaults and no longer assume `Variety` labels or depth files.

## Training

`train.py`

- Configuration: `TrainConfig` exposes dataset paths, batch size, epochs, learning rate, patience, etc.
- Loader setup: caches preprocessed tensors on CPU, splits into train/val via `random_split`, uses worker seeding for determinism.
- Optimization: single-stage MAE training with AdamW + early stopping. Best checkpoint saved as `best_model_v4.pth` inside `out_dir`.
- Usage:

```cmd
cd models/model_v4.1
python preprocess.py   # optional if augmented data already exists
python train.py
```

Notes:

- Defaults point to the augmented dataset under `datasets/Training/Augmented/`.
- Dataloaders default to `num_workers=0` on Windows for stability; enable more workers on Linux for throughput.

## Evaluation

`eval.py` loads the trained checkpoint, constructs `PlantDatasetV4` in non-augmented mode, and reports validation MAE:

```cmd
python eval.py --checkpoint best_model_v4.pth
```

You can override CSV/image paths via the dataclass parameters or CLI flags (see `argparse` help inside the script if added).

## Prediction workflow

`predict.py`

- Consumes RGB-only test data (`datasets/Test/RGBImages`, `datasets/Test/Test.csv`).
- Runs the regressor in eval mode and writes `DryWeightShoot` predictions to `Test_with_predictions_v4.csv` by default.

Example:

```cmd
python predict.py --model-path best_model_v4.pth --output-csv submissions/v4_1.csv
```

## Viewer (optional)

`view_rgb_depth_side_by_side.py` still exists for historical debugging of RGB/Depth pairs. The script remains unchanged and continues to expect both modalities, but it is no longer required for training or inference.

Run from this directory:

```cmd
python view_rgb_depth_side_by_side.py --split Training
```

Refer to in-script help (`-h`) for keyboard shortcuts, augmentation preview toggles, and dependency notes (uses OpenCV).

## Repository checklist

- `preprocess.py`: RGB-only augmentation pipeline (1000→96).
- `dataloader.py`: RGB loaders for train/test.
- `model.py`: single-branch CNN regressor.
- `train.py`: single-stage MAE training loop.
- `eval.py`: MAE reporting.
- `predict.py`: CSV export.
- `docs/`, `tests/`, and the viewer remain for archival reference; update them as needed if you rely on those assets.


version 4.1.1 MAE (dry weight): 0.155983 20 aug with 90's factor rotation 
version 4.1.2 MAE (dry weight): 0.100910 30 aug random rot 
