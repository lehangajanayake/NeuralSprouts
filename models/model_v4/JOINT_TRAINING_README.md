# Joint (End-to-End) Training for Model V4

## Overview
`train_joint.py` provides an alternative to the staged training approach in `train.py`. Instead of training branches separately (Stage 1→2→3), it trains all branches (RGB classifier, RGBD regressor, Fusion head) **simultaneously from scratch**.

## Why Joint Training?

### Potential Benefits
1. **Direct optimization for the target task**: All parameters are optimized end-to-end for dry_weight MAE from the start
2. **Simpler training**: No need to manage checkpoints between stages or decide when to freeze/unfreeze branches
3. **Better feature alignment**: RGB and RGBD branches learn representations jointly, which may lead to better fusion
4. **Avoids degradation**: In staged training, Stage 3 can degrade the RGB classifier's accuracy; joint training balances both tasks throughout

### Trade-offs
- **Slower convergence**: Training from scratch without pre-training the RGB classifier takes more epochs
- **Hyperparameter sensitivity**: Loss weights (classification vs regression) need tuning
- **Resource usage**: Trains all branches every epoch (no freezing), so slightly more compute per epoch

## Configuration

Key parameters in `JointTrainConfig`:

```python
# Multi-task loss weights
mae_weight: float = 1.0      # Primary: dry_weight MAE (competition metric)
cls_weight: float = 0.0      # Auxiliary: RGB classification (helps regularize)

# Set cls_weight=0.0 for pure regression (no classification loss)
# Increase cls_weight (e.g., 0.5) to emphasize classification more

# Training params
num_epochs: int = 150        # Joint training may need more epochs than Stage 3
lr: float = 1e-3             # Learning rate for AdamW
patience: int = 15           # Early stopping patience
```

## Usage

### Basic Run
```bash
cd models/model_v4
python train_joint.py
```

This will:
- Train all branches jointly for 150 epochs (or until early stopping)
- Use multi-task loss: `loss = 1.0 * MAE + 0.1 * CrossEntropy`
- Save best model to `best_joint_v4.pth`
- Log to `debug_joint.log`
- Generate plots: `joint_mae_curve.png` and `joint_cls_curve.png`

### Pure Regression Mode (No Classification)
To optimize **only** for dry_weight MAE without any classification loss:

```python
from train_joint import JointTrainConfig, main
cfg = JointTrainConfig(mae_weight=1.0, cls_weight=0.0)
# Then modify main() to accept cfg or edit the script
```

Or edit `train_joint.py` line 31:
```python
cls_weight: float = 0.0  # Pure regression
```

### Custom Configuration
```python
from train_joint import JointTrainConfig

cfg = JointTrainConfig(
    num_epochs=200,
    lr=5e-4,              # Lower LR for stability
    patience=20,
    mae_weight=1.0,
    cls_weight=0.2,       # Emphasize classification more
    batch_size=32,        # Smaller batch if GPU memory is tight
)
# Then run with this config
```

## Outputs

After training completes:
- **Checkpoint**: `best_joint_v4.pth` (best validation MAE)
- **Logs**: `debug_joint.log` (all epochs, metrics, confusion matrix)
- **Plots**:
  - `joint_mae_curve.png`: Train/val MAE curves with best epoch marker
  - `joint_cls_curve.png`: Train/val classification loss + accuracy (if `cls_weight > 0`)

## Evaluation

Use the updated `eval.py` to evaluate the joint-trained model:

```bash
python -c "from eval import EvalConfig, main; main(EvalConfig(checkpoint='best_joint_v4.pth', head='fusion'))"
```

This will show:
- Validation MAE on fusion output
- RGB classification accuracy (should be much higher than Stage 3 if `cls_weight > 0`)
- Confusion matrix

## Comparison: Staged vs Joint

| Aspect | Staged Training (`train.py`) | Joint Training (`train_joint.py`) |
|--------|------------------------------|-----------------------------------|
| **Training time** | ~100 epochs total (3 stages) | ~150 epochs (single run) |
| **Complexity** | 3-stage checkpoints, freeze/unfreeze logic | Single training loop, no freezing |
| **RGB accuracy** | High in Stage 1, degrades in Stage 3 | Maintained throughout (if `cls_weight > 0`) |
| **MAE** | Optimized in Stages 2 & 3 | Optimized from epoch 1 |
| **Best for** | Faster convergence, stable initialization | Simpler pipeline, direct end-to-end optimization |

### When to Use Joint Training
- You want simpler training without managing stages
- You care about maintaining RGB classification accuracy while regressing dry_weight
- You want to experiment with multi-task learning weights
- Staged training's RGB degradation is problematic for your use case

### When to Use Staged Training
- You want faster initial convergence (Stage 1 RGB pre-training is quick)
- You only care about final MAE, not RGB classification
- You have limited compute (staged approach uses fewer epochs)

## Tips for Best Results

1. **Start with default weights**: `mae_weight=1.0, cls_weight=0.1` is a good baseline
2. **Monitor both losses**: Check `debug_joint.log` to see if classification is too weak/strong
3. **Adjust cls_weight**: If val_acc stays low (<80%), increase to 0.2-0.5; if MAE suffers, decrease to 0.05
4. **Longer patience**: Joint training benefits from longer patience (15-20) since early epochs are noisy
5. **Learning rate**: If training is unstable, reduce LR to 5e-4 or add a scheduler

## Expected Results

With default config (`mae_weight=1.0, cls_weight=0.1`):
- **Validation MAE**: Should converge to ~0.65-0.75 (competitive with staged training)
- **RGB accuracy**: Should maintain ~85-95% (much better than Stage 3's 28%)
- **Training time**: ~150 epochs @ ~2 sec/epoch on GPU = ~5 minutes total

The joint approach should give you similar or better MAE than staged training, while keeping the RGB classifier functional.
