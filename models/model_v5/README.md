# Model V5: Triple-Branch Fusion Architecture

A multimodal deep learning model for plant dry weight prediction that leverages three independent branches (RGB, RGBD, Depth) with late fusion through a fully connected layer.

## Architecture Overview

### Core Design Philosophy

Model V5 adopts a **late fusion strategy** with three independent feature extraction branches:

```
                      ┌─────────────────┐
                      │  RGB Input      │
                      │  (3, 128, 128)  │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  RGB Branch     │
                      │  (Conv Blocks)  │
                      │  → 256 features │
                      └────────┬────────┘
                               │
        ┌──────────────┐       │       ┌──────────────┐
        │ RGBD Input   │       │       │ Depth Input  │
        │(4, 128, 128) │       │       │(1, 128, 128) │
        └──────┬───────┘       │       └──────┬───────┘
               │               │              │
        ┌──────▼───────┐       │       ┌──────▼───────┐
        │ RGBD Branch  │       │       │ Depth Branch │
        │ (Conv Blocks)│       │       │ (Conv Blocks)│
        │→ 256 features│       │       │→ 256 features│
        └──────┬───────┘       │       └──────┬───────┘
               │               │              │
               └───────────────┼──────────────┘
                               │
                        ┌──────▼──────┐
                        │ Concatenate │
                        │ (768 dims)  │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │ Fusion FC    │
                        │ (2 hidden    │
                        │  layers)     │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Output: 1  │
                        │(Dry Weight) │
                        └─────────────┘
```

### Three Branches

1. **RGB Branch (3 channels)**
   - Input: Color information from plant images
   - Independent Conv blocks: 3→32→64→128→256
   - Purpose: Capture visible features (color, shape, texture)

2. **RGBD Branch (4 channels)**
   - Input: RGB + Depth information
   - Independent Conv blocks: 4→32→64→128→256
   - Purpose: Combine spatial structure with color information
   - Note: **NO weight sharing** with RGB branch

3. **Depth Branch (1 channel)**
   - Input: Depth/3D structure only
   - Independent Conv blocks: 1→32→64→128→256
   - Purpose: Capture structural/volumetric information
   - Note: **NO weight sharing** with other branches

### Fusion Layer

- **Concatenation**: Combine 256-dim features from each branch → 768-dim vector
- **FC Layers**:
  - 768 → 256 (ReLU, Dropout)
  - 256 → 128 (ReLU, Dropout)
  - 128 → 1 (output: dry weight)

## Key Features

### 1. **128×128 Input Images**
- Higher resolution than previous models (64×64)
- Better detail preservation for fine plant structures
- Input preprocessing: center crop 900×900 → resize to 128×128

### 2. **VRAM-Efficient Training**
- All images preloaded into GPU memory at initialization
- Eliminates I/O overhead during training
- Optimized for 6GB VRAM GPUs with batch_size=16
- No data loading delays → faster epochs

### 3. **Loss Functions**
- **Training Loss**: RMSE (Root Mean Squared Error) via MSELoss
  - Penalizes larger errors more heavily
  - Good for regression tasks with varied scales
- **Validation Metric**: MAE (Mean Absolute Error)
  - **Competition metric** for final evaluation
  - More interpretable (directly in dry weight units)
  - Tracked separately from training loss

### 4. **Learning Rate Scheduling**
- Automatic learning rate adjustment during training
- Options available:
  - `CosineAnnealingLR`: Smooth decay to near-zero
  - `StepLR`: Step-wise decay every N epochs
  - `ExponentialLR`: Exponential decay
  - `ReduceLROnPlateau`: Adaptive reduction on validation plateau
- **Logged per epoch** for debugging

### 5. **Keyboard Interrupt Handling**
- Gracefully handles Ctrl+C during training
- Saves current model state to `checkpoint_interrupted.pth`
- Allows resuming training or inspection

### 6. **Comprehensive Logging**
- Dual output: console (INFO+) and file (DEBUG+)
- Per-epoch metrics: Loss, MAE, Learning Rate
- Batch-level debugging info
- Training history saved as JSON

### 7. **Metrics Visualization**
- Automatic plotting of training curves
- Separate subplots for RMSE Loss and MAE
- Saved as `training_metrics.png`

## Pros and Cons of Triple-Branch Fusion

### ✅ Pros

1. **Multimodal Learning**
   - Each modality (RGB, RGBD, D) provides unique information
   - Complementary features improve generalization
   - Better handles diverse plant growth patterns

2. **Independent Feature Extraction**
   - No weight sharing → branches learn specialized representations
   - More expressive model (higher capacity)
   - Each modality optimized for its characteristics

3. **Late Fusion Benefits**
   - Allows flexible combination of learned features
   - Easier to debug individual branch contributions
   - Can be extended with attention mechanisms later

4. **Robustness**
   - Model can handle missing modality gracefully (with modification)
   - Combines complementary information sources
   - Reduces overfitting through diversity

5. **Interpretability**
   - Can analyze branch outputs separately
   - Understand which modality contributes most
   - Easier to identify failure modes

### ❌ Cons

1. **Higher Parameter Count**
   - ~3x parameters compared to single-branch model
   - More data needed for generalization
   - Longer training time (3 branches process data sequentially)
   - Risk of overfitting on limited datasets

2. **Memory Overhead**
   - Stores 3 separate branches in VRAM
   - Higher GPU memory requirements
   - Limits maximum batch size or input resolution

3. **Training Complexity**
   - Hyperparameters must balance 3 branches fairly
   - Potential for one branch to dominate
   - May require careful initialization

4. **Computational Cost at Inference**
   - 3× forward passes through conv blocks
   - Slightly slower inference compared to single-branch
   - Not ideal for real-time applications

5. **Data Augmentation Complexity**
   - Must maintain alignment across RGB, Depth, RGBD
   - Augmentations must be applied consistently to all modalities

### When This Architecture Excels

✓ Sufficient training data (1000+ samples)
✓ Multimodal information available and complementary
✓ Inference speed not critical
✓ GPU memory available (6GB+)
✓ High accuracy priority over efficiency

### When Single-Branch is Better

✗ Limited training data (<500 samples)
✗ Need fast inference (mobile/edge)
✗ Very restricted VRAM (<2GB)
✗ Only one modality available

## Usage

### 1. Preprocessing

```bash
python preprocess.py
```

Creates 128×128 images from original dataset:
- Center crops to 900×900
- Resizes to 128×128
- Generates augmented variants (20 per original)
- Outputs to `datasets/Training/Augmented/`

**Note**: Modify `PreprocessConfig` in `preprocess.py` to change:
- `image_size`: 128 (fixed for this model)
- `num_aug_per_image`: 20 (number of augmented variants)
- `crop_size`: 900 (center crop before resize)

### 2. Training

```bash
python train.py
```

**Key outputs:**
- `best_model_v5.pth`: Best model (lowest validation MAE)
- `checkpoint_epoch_*.pth`: Periodic checkpoints
- `training_metrics.png`: Loss and MAE curves
- `training_history.json`: Numerical training history
- `logs/train_v5_*.log`: Detailed training logs

**Configuration** (modify `TrainConfig` in `train.py`):

```python
cfg = TrainConfig()
cfg.batch_size = 16           # VRAM-friendly for 6GB GPU
cfg.num_epochs = 200          # Total training epochs
cfg.lr = 1e-3                 # Learning rate for Adam
cfg.scheduler_type = 'cosine' # 'cosine', 'step', 'exponential', 'plateau'
cfg.device = 'cuda'           # 'cuda' or 'cpu'
cfg.dropout = 0.2             # Dropout probability
```

**Stopping early:**
- Ctrl+C saves `checkpoint_interrupted.pth`
- Can resume from checkpoint (modify code to load state_dict)

### 3. Evaluation

```bash
python eval.py \
  --model best_model_v5.pth \
  --csv ../../datasets/Training/Augmented/Train_aug.csv \
  --rgb_dir ../../datasets/Training/Augmented/RGBImages \
  --depth_dir ../../datasets/Training/Augmented/DepthImages \
  --device cuda \
  --batch_size 32
```

Reports:
- MAE (competition metric)
- RMSE (training loss metric)
- Prediction statistics

### 4. Prediction

```bash
python predict.py \
  --model best_model_v5.pth \
  --output predictions.csv \
  --rgb_dir ../../datasets/Test/RGBImages \
  --depth_dir ../../datasets/Test/DepthImages \
  --input_csv ../../datasets/Test/Test.csv
```

Creates CSV with columns: `image_id`, `DryWeightShoot`

Or use programmatically:

```python
from model import PlantV5TripleBranch
from predict import load_image_pair, predict_single_sample

model = PlantV5TripleBranch()
model.load_state_dict(torch.load('best_model_v5.pth'))
model = model.to('cuda')

rgb, rgbd, depth = load_image_pair('RGB_1.png', 'Depth_1.png', device='cuda')
pred = predict_single_sample(model, rgb, rgbd, depth)
print(f"Predicted dry weight: {pred:.2f}")
```

## Model Details

### Architecture Parameters

| Component | Input | Output | Details |
|-----------|-------|--------|---------|
| RGB Branch | (N,3,128,128) | (N,256) | 4×ConvBlock(3,32,64,128,256) |
| RGBD Branch | (N,4,128,128) | (N,256) | 4×ConvBlock(4,32,64,128,256) |
| Depth Branch | (N,1,128,128) | (N,256) | 4×ConvBlock(1,32,64,128,256) |
| Fusion FC | (N,768) | (N,1) | 256→128→1 with ReLU+Dropout |

### ConvBlock Details

```
Conv2d(kernel=3, padding=1)
  ↓
BatchNorm2d
  ↓
ReLU(inplace=True)
  ↓
MaxPool2d(kernel=2)
```

### Spatial Reduction

After 4 MaxPool2d layers with kernel=2:
- 128×128 → 64×64 → 32×32 → 16×16 → 8×8
- Final feature map: (256, 8, 8) = 16,384 values
- After flatten + FC: 256 features per branch

### Total Parameters

~2.5M trainable parameters:
- RGB Branch: ~450K
- RGBD Branch: ~500K  
- Depth Branch: ~400K
- Fusion FC: ~200K

## Training Tips

### 1. **Memory Optimization**
- If OOM with batch_size=16, reduce to 8
- Model stays in VRAM between epochs (no reload)
- ~1.5GB for all images + model

### 2. **Learning Rate**
- Default: 1e-3 (good for 6GB GPU)
- Try 5e-4 if training is unstable
- Try 2e-3 if convergence is slow
- Monitor learning rate printout per epoch

### 3. **Scheduler Selection**
- `cosine`: Recommended default (smooth decay)
- `step`: Good if you know when to reduce LR
- `plateau`: Adaptive but can be tricky
- None: Fixed LR (not recommended)

### 4. **Data Augmentation**
- Currently: rotation, flip, brightness, contrast
- Modify `preprocess.py` for more augmentations
- Balance between variants per original

### 5. **Hyperparameter Search**
Edit `TrainConfig` and experiment:
```python
for lr in [1e-3, 5e-4, 2e-3]:
    for batch_size in [8, 16, 32]:
        cfg.lr = lr
        cfg.batch_size = batch_size
        main(cfg)
```

## Logs and Debugging

### Log File Location
`logs/train_v5_YYYYMMDD_HHMMSS.log`

### Debug Output
- Per-batch metrics: `cfg.debug = True`
- Model architecture: Printed at start
- Data loading progress: Shown during preprocessing
- Per-epoch learning rate: Logged during training

### Common Issues

**OOM (Out of Memory)**
- Reduce `batch_size` (try 8)
- Clear cache: `torch.cuda.empty_cache()`

**Slow training**
- Check if images in VRAM: Print cache size at start
- Verify GPU usage: `nvidia-smi` during training

**Poor convergence**
- Check learning rate in logs (should decay over time)
- Verify scheduler type
- Check data range (images normalized to [0,1])

**Training stops abruptly**
- Check `checkpoint_interrupted.pth` for Ctrl+C
- Verify disk space for logs

## Comparison with Previous Models

| Aspect | V4 | V5 |
|--------|----|----|
| Input Resolution | 64×64 | 128×128 |
| Branches | 2 (RGB, RGBD) | 3 (RGB, RGBD, Depth) |
| Training Strategy | 3 stages | Single (dry weight) |
| Parameters | ~1.5M | ~2.5M |
| Training Loss | Classification + Regression | Pure RMSE |
| Validation Metric | Custom | MAE (standard) |
| VRAM Caching | Partial | Full (all images) |
| Scheduler | Basic | Configurable |
| Logging | Basic | Comprehensive |

## Future Improvements

1. **Attention Mechanism** (V5.1)
   - Channel attention between branches
   - Spatial attention within branches
   - Cross-branch attention

2. **Ensemble Methods**
   - Multiple model checkpoints
   - Weighted averaging
   - Voting mechanism

3. **Transfer Learning**
   - Pretrained ImageNet weights
   - Fine-tuning strategies
   - Domain adaptation

4. **Uncertainty Estimation**
   - Monte Carlo Dropout
   - Epistemic/aleatoric uncertainty
   - Confidence intervals

5. **Temporal Information**
   - Multi-timestep sequences
   - Growth trajectory modeling
   - LSTM/Transformer fusion

## References

- **Late Fusion**: Baltrušaitis et al., "Multimodal Machine Learning: A Survey and Taxonomy"
- **MSE/RMSE Loss**: Standard regression loss for continuous targets
- **MAE**: Interpretable mean absolute error metric
- **Data Augmentation**: Improves generalization with limited data

## License and Citation

Model V5 - Plant Dry Weight Prediction using Triple-Branch Fusion (2025)

If you use this model, please cite:
```
@model{PlantV5TripleBranch2025,
  author = {NeuralSprouts},
  title = {Model V5: Triple-Branch Fusion for Plant Dry Weight Prediction},
  year = {2025}
}
```

## Contact

For questions or issues, refer to the training logs and debugging output.
