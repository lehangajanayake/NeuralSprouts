# model_v7 — Phase 1 Improvements (Multi-branch CNN v2)

## Overview

**Model v7** is an improved version of model_v4 with Phase 1 optimization improvements targeting a reduction in MAE from **0.6456 to approximately 0.55**.

This model maintains the same multi-branch architecture as v4 but implements critical improvements in data augmentation, depth normalization, and learning rate scheduling for better convergence.

## Architecture (Same as v4)

The model consists of three coordinated components:

1. **RGB Classification Branch**
   - Input: 3×64×64 RGB images
   - Architecture: 4 convolutional blocks (32→64→128→256 filters)
   - Output: 4-class lettuce variety logits
   - Role: Extracts variety-specific features

2. **RGBD Regression Branch**
   - Input: 4×64×64 (RGB + Depth concatenated)
   - Architecture: 4 convolutional blocks (32→64→128→256 filters)
   - Output: Initial dry weight estimate (single scalar)
   - Role: Direct regression from multimodal input

3. **Fusion Network**
   - Input: Concatenated features (4 variety logits + 1 regression value = 5 features)
   - Architecture: 2-layer MLP (hidden_dim=32)
   - Output: Final refined dry weight prediction
   - Role: Combines classification and regression information for improved accuracy

## Phase 1 Improvements

### 1. **Increased Data Augmentation** ⭐⭐
- **Change**: Augmentation increased from 20 → **50 variants per original image**
- **Why**: Larger, more diverse training dataset improves generalization and reduces overfitting
- **File Modified**: `preprocess.py`
- **Expected Impact**: +3-8% MAE improvement
- **Training set**: ~231 original → ~11,550 augmented samples

### 2. **Better Depth Normalization** ⭐
- **Change**: Depth normalized by **actual min/max per image** instead of fixed 255
- **Why**: Different depth sensors/images have different value ranges; per-image normalization ensures features are learned correctly
- **File Modified**: `dataloader.py`
- **Implementation**:
  ```python
  # Old: depth_np = depth_np / 255.0
  
  # New: Normalize per-image
  depth_min = depth_np.min()
  depth_max = depth_np.max()
  if depth_max > depth_min:
      depth_np = (depth_np - depth_min) / (depth_max - depth_min)
  ```
- **Expected Impact**: +2-5% MAE improvement

### 3. **Learning Rate Scheduling** ⭐⭐⭐
- **Change**: Added warmup + cosine annealing for all three training stages
- **Why**: Better convergence by:
  - Warmup: Gradually ramp up LR (10% → 100%) to stabilize early training
  - Cosine Annealing: Smoothly decrease LR over time to find better local minima
- **File Modified**: `train.py`
- **Implementation Details**:
  - **Warmup**: LinearLR for 5 epochs (start_factor=0.1)
  - **Annealing**: CosineAnnealingLR for remaining epochs (eta_min=1e-5)
  - Applied to all 3 stages with stage-specific epoch counts
- **Expected Impact**: +5-10% MAE improvement

## Expected Performance Gain

| Metric | model_v4 | model_v7 | Improvement |
|--------|----------|----------|-------------|
| **MAE** | 0.6456 | ~0.55 | **~14-15%** |
| Strategy | Baseline | Phase 1 optimizations | Cumulative |

**Breakdown**:
- Augmentation: 0.6456 → 0.62-0.63 (+3-8%)
- Depth normalization: 0.63 → 0.61-0.62 (+2-5%)  
- LR scheduling: 0.62 → 0.55 (+5-10%)
- **Total: 0.6456 → ~0.55 MAE** ✅

## Training Strategy

Identical to model_v4 (3-stage training with improvements):

1. **Stage 1**: Train RGB branch only (variety classification)
   - Loss: Cross-Entropy
   - Learning Rate: 1e-3 **with warmup + cosine annealing** ⭐
   - Max Epochs: 100 (early stopping, patience=7)

2. **Stage 2**: Train RGBD branch only (dry weight regression)
   - Loss: MAE (L1)
   - Learning Rate: 1e-3 **with warmup + cosine annealing** ⭐
   - Max Epochs: 100 (early stopping, patience=10)

3. **Stage 3**: Train fusion network (fine-tune all branches)
   - Loss: MAE on fusion output
   - Learning Rate: 5e-4 **with warmup + cosine annealing** ⭐
   - Max Epochs: 200 (early stopping, patience=12)

## Files Changed from model_v4

```
model_v7/
├── preprocess.py          ← num_aug_per_image: 20 → 50
├── dataloader.py          ← Improved depth normalization (per-image min/max)
├── train.py               ← Added LR scheduling (LinearLR + CosineAnnealingLR)
├── model.py               (unchanged - same architecture)
├── eval.py                (unchanged)
├── predict.py             (unchanged)
└── README.md              (this file)
## Data Assumptions

Expected dataset structure:
- `datasets/Training/Augmented/RGBImages/RGB_*.png`
- `datasets/Training/Augmented/DepthImages/Depth_*.png`
- `datasets/Training/Augmented/Train_aug.csv`

CSV columns:
- `image_id` or `id`
- `Variety` (string; 4 classes: Aphylion, Lugano, Salanova, Satine)
- `DryWeightShoot` (float; in grams)

Preprocessing:
- Center crop: 900×900
- Resize: 64×64
- RGB normalization: ÷ 255
- Depth normalization: **Per-image (min-max normalization)** ⭐ (NEW)
- Augmentations: 50 variants per image (NEW)

## Run Instructions

```bash
cd models/model_v7

# Step 1: Preprocess & create augmented dataset (creates Train_aug.csv)
python preprocess.py

# Step 2: Train model (3-stage training with LR scheduling)
python train.py

# Step 3: Evaluate on test set
python eval.py

# Step 4: Generate predictions
python predict.py
```

## Output Files

- `best_rgb_branch_v4.pth` - Best RGB branch checkpoint (stage 1)
- `best_rgbd_branch_v4.pth` - Best RGBD branch checkpoint (stage 2)
- `best_model_v4.pth` - Final best model checkpoint (stage 3)
- `Test_with_predictions_v4.csv` - Predictions on test set

## Key Improvements Summary

| Component | model_v4 | model_v7 | Impact |
|-----------|----------|----------|--------|
| Augmentation | 20/image | **50/image** | +3-8% |
| Depth norm | Fixed ÷255 | **Per-image** | +2-5% |
| LR schedule | Fixed | **Warmup+Cosine** | +5-10% |
| Expected MAE | 0.6456 | **~0.55** | **-14-15%** |

## Comparison to model_v4

- **Architecture**: Identical (no structural changes)
- **Training stages**: Same 3-stage approach
- **Early stopping**: Unchanged (same patience values)
- **Main differences**: Better data, better normalization, smarter learning rate
- **Training time**: Slightly longer (~2-3x due to 50 aug variants)
- **Memory**: Same as v4
- **Performance**: Expected +14-15% MAE improvement

## Notes

- All Phase 1 improvements are **low-risk** (no architectural changes)
- **Backward compatible** with model_v4 structure
- Learning rate scheduling handled automatically
- No manual hyperparameter adjustment needed
- Larger augmentation dataset requires preprocessing time (~5-10 min)

## When to Move to Phase 2

If Phase 1 achieves **MAE < 0.58** with stable validation curves, move to Phase 2 improvements:
- Deeper networks (5+ conv blocks)
- Huber/Smooth L1 loss
- End-to-end joint training
- Batch normalization in fusion network
- Multi-task auxiliary heads

---

**Version**: model_v7 (Phase 1)  
**Target MAE**: **0.55** (from 0.6456)  
**Status**: ✅ Ready to train
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
