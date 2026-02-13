# model_v4.2 — dual RGB/RGBD regressors with fusion

## Overview

Version 4.2 now keeps two clean regression branches—one fed only RGB, the other fed RGB+Depth—and fuses their scalar predictions into a single dry-weight estimate. Classification logits and variety supervision were removed to simplify the loss surface and focus purely on MAE. The preprocessing pipeline crops every image pair to 1000×1000 before resizing to 96×96 so it stays consistent with the pared-down architecture.

Primary metric: **MAE on the fused dry-weight output**.

## Architecture (`model.py`)

- **RGB regressor**: four shared `ConvBlock`s (32→64→128→256) ending in a linear head `256*6*6 → 256 → 1`.
- **RGBD regressor**: same topology but accepts 4-channel stacks (RGB concatenated with depth).
- **Fusion MLP**: takes the two scalar predictions `[rgb_pred, rgbd_pred]` and refines them through a 2-layer MLP to a single dry-weight value.
- `forward(rgb, rgbd)` returns `(rgb_pred, rgbd_pred, fusion_pred)`. `predict_dry_weight()` simply returns the fused output.
- `set_requires_grad()` remains for staged freezing/unfreezing.

## Data pipeline

### `preprocess.py`

- Reads `Train.csv`, paired RGB/Depth folders, and creates deterministic augmentations.
- Center-crops **1000×1000**, resizes to **96×96**, and exports aligned RGB/Depth PNGs plus `Train_aug.csv`.
- Augmentations keep modalities aligned (flips, 90° rotations) and apply color jitter to RGB only.

### `dataloader.py`

- `PlantDatasetV4` now only requires `id` and `DryWeightShoot` columns. Each sample returns `rgb`, `rgbd`, and `dry_weight` tensors.
- `TestPlantDatasetV4` mirrors the same preprocessing for inference datasets.
- Deterministic caching/augmentation options are unchanged from earlier versions.

## Training (`train.py`)

Training still happens in three stages, but every phase optimizes MAE:

1. **Stage 1** — Train the RGB regressor while freezing RGBD + fusion (target: dry weight).
2. **Stage 2** — Train the RGBD regressor while freezing RGB + fusion.
3. **Stage 3** — Unfreeze everything and train the fusion head (still MAE on the fused prediction).

Each stage uses AdamW with independent LRs/patience values and saves `best_rgb_branch_v4.pth`, `best_rgbd_branch_v4.pth`, and `best_model_v4.pth`. Defaults assume the augmented dataset under `datasets/Training/Augmented/`.

Usage:

```cmd
cd models/model_v4.2
python preprocess.py   # optional if augmented assets already exist
python train.py
```

## Evaluation (`eval.py`)

Runs on the base training split (no augmentation) and reports MAE for the fused prediction:

```cmd
python eval.py --checkpoint best_model_v4.pth
```

## Prediction (`predict.py`)

Consumes paired RGB/Depth test folders and writes `DryWeightShoot` predictions to `Test_with_predictions_v4.csv` by default (override with `--output-csv`). Example:

```cmd
python predict.py --model-path best_model_v4.pth --output-csv submissions/v4_2.csv
```

## Viewer

`view_rgb_depth_side_by_side.py` is still available for sanity-checking alignment/augmentations, though it is not required for training or inference.

## Repository checklist

- `preprocess.py`: RGB+Depth augmentation (1000→96) with deterministic IDs.
- `dataloader.py`: RGB/RGBD loaders returning regression targets only.
- `model.py`: dual regression branches feeding a fusion MLP.
- `train.py`: MAE-only staged training and checkpointing.
- `eval.py`: MAE reporting.
- `predict.py`: submission helper using the new regression-only API.
- `tests/`: updated shape checks for RGB/RGBD inputs at 96×96.
