# model_v4.2 — dual-branch fusion baseline

Model v4.2 is the first “clean” regression-only baseline in the current family.
It keeps two lightweight convolutional branches—RGB and RGBD—and lets a single
fusion head learn how to mix their feature embeddings. There is no variety
classification, deep supervision, or attention: the goal is a dependable MAE
baseline that trains quickly and produces reproducible splits.

## Highlights

- **Two parallel stacks** of `ConvBlock` layers (32→64→128→256) with adaptive
	average pooling and a 256-d embedding per branch.
- **Fusion MLP** consumes the concatenated embeddings (512 dims) and outputs the
	dry-weight prediction; the loss is pure MAE on this fused scalar.
- **Deterministic preprocessing** crops every RGB/Depth pair between 1000 px and
	the full frame, applies ±100 px center shifts, aligned flips/rotations, and RGB
	color jitter before resizing to 96×96.
- **Group-aware data splits**: training uses either a held-out split or K-folds
	while keeping augmentations from the same original plant together.

## Architecture (`model.py`)

- `RGBRegressionBranch` / `RGBDRegressionBranch` share the same 4-layer ConvBlock
	topology; each returns both a scalar prediction and its pooled embedding.
- `FusionMLP` (256→256→1 with dropout) learns how to mix the two embeddings.
- `forward(rgb, rgbd)` returns `(rgb_pred, rgbd_pred, fusion_pred)`, but only the
	fused tensor is used for training. `predict_dry_weight()` is a convenience wrapper.

## Data pipeline (`preprocess.py`, `dataloader.py`)

1. `preprocess.py` reads `datasets/Training/Train.csv`, pairs `RGB_*.png` and
	 `Depth_*.png`, and exports 1 + `num_aug_per_image` aligned crops to
	 `datasets/Training/Augmented/{RGBImages,DepthImages}` with a matching
	 `Train_aug.csv` (the script stores the `original_id` column for grouped splits).
2. `PlantDatasetV4` loads those PNGs, applies optional on-the-fly flips/rotations,
	 and returns tensors `{'id', 'rgb', 'rgbd', 'dry_weight'}`. `TestPlantDatasetV4`
	 mirrors the same transforms for inference.

## Training workflow (`train.py`)

- Single-stage AdamW training with ReduceLROnPlateau and early stopping on
	validation MAE.
- Optional GPU preloading to keep entire splits resident in memory when VRAM allows.
- Group-aware splits determined by `original_id` (or `outputs_per_original` when
	IDs are missing) so augmented views of a plant never leak across folds.
- Default checkpoints: `best_model_v4.pth` (or `best_model_v4_<fold>.pth`).

Usage:

```cmd
cd models/model_v4.2
python preprocess.py   # one-time augmentation step
python train.py        # trains and saves best_model_v4.pth
```

## Evaluation & inference

- `eval.py --checkpoint best_model_v4.pth` runs on the original (non-augmented)
	training CSV, prints MAE, and writes per-sample predictions.
- `predict.py --model-path best_model_v4.pth --output-csv submissions/v4_2.csv`
	scores the public Test set (expects paired RGB/Depth folders).
- `view_rgb_depth_side_by_side.py` remains available for quick sanity checks of
	the augmented crops.

## File map

- `preprocess.py` — deterministic RGB/Depth augmentation pipeline.
- `dataloader.py` — training + inference datasets returning RGB, RGBD, ids, targets.
- `model.py` — two ConvBlock branches plus fusion MLP.
- `train.py` — grouped single-stage MAE training.
- `eval.py`, `predict.py` — evaluation and submission helpers.
- `tests/` — forward-shape regression tests at 96×96.
