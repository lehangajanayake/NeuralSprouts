# Model v7 - Quick Reference Card

## 🎯 Goal
Improve MAE from **0.6456 → ~0.55** (+14-15% improvement)

## 📊 What Changed

| Aspect | v4 | v7 | Impact |
|--------|----|----|--------|
| **Augmentations** | 20/image | **50/image** | +3-8% ↓ MAE |
| **Depth normalization** | Fixed ÷255 | **Per-image min-max** | +2-5% ↓ MAE |
| **Learning rate** | Fixed | **Warmup + Cosine annealing** | +5-10% ↓ MAE |
| **Expected MAE** | 0.6456 | **~0.55** | **+14-15% improvement** |

## 🚀 Quick Start

```bash
cd models/model_v7

# 1. Preprocess (creates augmented dataset)
python preprocess.py  # ~5-10 minutes

# 2. Train (3-stage with LR scheduling)
python train.py       # ~30-60 minutes (GPU)

# 3. Evaluate
python eval.py

# 4. Predict
python predict.py
```

## 📝 Files Modified (Only 3!)

### 1. `preprocess.py` (1 line changed)
```python
num_aug_per_image: int = 50  # was 20
```

### 2. `dataloader.py` (11 lines changed)
```python
# Normalize depth by actual min/max instead of fixed 255
depth_min = depth_np.min()
depth_max = depth_np.max()
if depth_max > depth_min:
    depth_np = (depth_np - depth_min) / (depth_max - depth_min)
```

### 3. `train.py` (4 imports + 27 lines added)
```python
# Import schedulers
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Added to each stage:
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
main_scheduler = CosineAnnealingLR(optimizer, T_max=..., eta_min=1e-5)

# In training loop:
if epoch < 5:
    warmup_scheduler.step()
else:
    main_scheduler.step()
```

## ⚙️ Technical Details

### Depth Normalization
**Why changed**: Different depth sensors have different value ranges
- **Old**: `depth_np / 255.0` (assumes all values 0-255)
- **New**: `(depth - min) / (max - min)` (per-image normalization)
- **Benefit**: Better learned features, more consistent training

### Learning Rate Scheduling
**Why added**: Better convergence to lower MAE values
- **Warmup (5 epochs)**: Slowly increase LR from 10% to 100%
  - Prevents instability at start of training
- **Cosine Annealing**: Smoothly decrease LR over remaining epochs
  - Finds better local minima
  - Allows more fine-grained adjustments near optimum

### More Augmentation
**Why increased**: 231 samples → ~11,550 samples after augmentation
- **Old**: 20 variants per image (4,620 total)
- **New**: 50 variants per image (11,550 total)
- **Benefit**: More diverse training data = less overfitting

## 📈 Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Current MAE** | 0.6456 | Baseline from model_v4 |
| **Target MAE** | ~0.55 | Phase 1 goal (+14-15%) |
| **Very likely** | <0.58 | Conservative estimate |
| **If excellent** | <0.50 | With good augmentation effect |

## ✅ Validation Checklist

After training, check:
- [ ] Validation MAE decreases smoothly (no spikes)
- [ ] Final MAE < 0.58 (conservative), ideally ~0.55
- [ ] Both training and validation curves converge
- [ ] Early stopping activates around epoch 50-100 (not too early)
- [ ] Learning rate scheduling is working (check logs for changes)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `batch_size` in config (64 → 32) |
| Training too slow | Check `num_workers` setting (use 0 on CPU, 2-4 on GPU) |
| MAE not improving | Check if warmup is too aggressive (reduce from 5 → 3 epochs) |
| High variance | Increase `weight_decay` (1e-4 → 5e-4) |

## 📚 Documentation

- **README.md**: Comprehensive overview + architecture details
- **PHASE1_IMPROVEMENTS.md**: Detailed implementation notes

## 🚨 Important Notes

- **No architectural changes**: Same as v4
- **Backward compatible**: Can use v4 and v7 interchangeably
- **All other models untouched**: v4, v5, v6 remain unchanged
- **Low risk**: Only optimizations, no new dependencies
- **Reproducible**: Same seed handling as v4

## 💡 Next Phase (Phase 2)

If Phase 1 achieves MAE < 0.58, move to Phase 2:
1. Add 5th conv block
2. Use Huber Loss
3. End-to-end training
4. Deeper fusion network

Target: **0.45 MAE**

---

**Version**: model_v7 (Phase 1)  
**Expected MAE**: **0.55** (±0.05)  
**Status**: ✅ Ready to train  
**Training time**: 30-60 minutes (GPU)
