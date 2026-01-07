# model_v4 — multi-branch CNN + viewer

This folder contains **Model v4** (multi-branch CNN: RGB classification + RGBD regression + fusion dry-weight prediction) and a fast viewer to inspect paired RGB/Depth images side-by-side.

## Model v4 (training + evaluation)

Implements the staged-training model described in `docs/prompt1_improved.txt`:

- **RGB branch**: lettuce variety classification (4 classes)
- **RGBD branch**: dry-weight regression
- **Fusion network**: concatenates (4 logits + 1 reg) → final dry-weight prediction

Primary metric: **MAE on dry weight** (fusion output).

### Files

- `model.py`: architecture + `set_requires_grad` helper
- `dataloader.py`: dataset returning both `rgb` (3×64×64) and `rgbd` (4×64×64)
- `preprocess.py`: creates augmented dataset + `Train_aug.csv`
- `train.py`: 3-stage training (freeze/unfreeze) + early stopping + checkpoints
- `eval.py`: evaluation reporting MAE (and optional classification accuracy)

### Data assumptions

Expected dataset structure (matches existing repo conventions):

- `datasets/Training/RGBImages/RGB_<id>.png`
- `datasets/Training/DepthImages/Depth_<id>.png`
- CSV columns:
  - `image_id` (or `id`)
  - `Variety` (string; mapped to class indices)
  - `DryWeightShoot` (float)

Mandatory preprocessing (train + eval):

- center crop **900×900**
- resize **64×64**

### Run

From `models/model_v4`:

```cmd
python preprocess.py
python train.py
python eval.py
```

Notes:

- Training defaults to augmented data under `datasets/Training/Augmented/...`.
- On Windows, dataloader workers default to 0 for stability.

## Viewer

- Script: `view_rgb_depth_side_by_side.py`
- Matches image pairs by plant id, using files:
  - `RGB_{id}.png`
  - `Depth_{id}.png`

Default dataset locations (relative to this folder):

- Training:
  - `../../datasets/Training/Train.csv`
  - `../../datasets/Training/RGBImages/`
  - `../../datasets/Training/DepthImages/`
- Test:
  - `../../datasets/Test/Test.csv`
  - `../../datasets/Test/RGBImages/`
  - `../../datasets/Test/DepthImages/`

## Run

From `models/model_v4`:

```cmd
python view_rgb_depth_side_by_side.py --split Training
```

If you’re using the repo venv:

```cmd
C:\Users\gajan\Documents\Projects\NeuralSprouts\.venv\Scripts\python.exe view_rgb_depth_side_by_side.py --split Training
```

## Controls

- `A` / `D`: prev / next image
- `R`: random image
- `J`: jump to a plant id
- `C`: cycle depth colormap
- `T`: toggle augmented preview (aligned flip/rot for RGB+Depth)
- `K`: re-roll augmentation for the current image
- `W` / `S`: increase / decrease depth contrast clipping
- `Q` or `ESC`: quit

## Augmented preview

Start with augmented preview enabled:

```cmd
python view_rgb_depth_side_by_side.py --split Training --aug
```

Optional: add light RGB-only color jitter in preview:

```cmd
python view_rgb_depth_side_by_side.py --split Training --aug --aug-color
```

## Dependency

This viewer uses OpenCV:

- `opencv-python`
