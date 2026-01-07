# Complete Tutorial: From Setup to Submission

This tutorial walks through the entire workflow of using the Multimodal Fusion Model for lettuce dry weight prediction.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Configuration](#3-configuration)
4. [Training](#4-training)
5. [Prediction](#5-prediction)
6. [Evaluation](#6-evaluation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Step 1.1: Create Virtual Environment

```bash
# Navigate to project directory
cd models/multimodal_fusion

# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 1.2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import timm; print('TIMM:', timm.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 1.3: Automated Setup (Alternative)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

---

## 2. Data Preparation

### Step 2.1: Understand Data Structure

Your data should follow this structure:

```
data/
├── train/
│   ├── rgb/              # RGB images
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── depth/            # Depth images (aligned to RGB)
│   │   ├── 0001.png      # or .npy, .tif
│   │   ├── 0002.png
│   │   └── ...
│   ├── masks/            # Binary segmentation masks (optional)
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── labels.csv        # Ground truth
├── test/
│   ├── rgb/
│   │   ├── test_001.png
│   │   └── ...
│   └── depth/
│       ├── test_001.png
│       └── ...
└── sample_submission.csv (optional)
```

### Step 2.2: Create labels.csv

Format:
```csv
id,dry_weight
0001,2.45
0002,3.12
0003,1.89
0004,4.23
...
```

**Important:**
- `id` should match image filenames (without extension)
- `dry_weight` is the target variable
- No header for IDs, just the data

### Step 2.3: Image Format Requirements

**RGB Images:**
- Format: PNG, JPG, JPEG
- Channels: 3 (RGB)
- Size: Any (will be resized to IMAGE_SIZE)

**Depth Images:**
- Format: PNG (single channel), NPY (numpy array), TIF/TIFF
- Channels: 1
- Values: Any range (will be normalized)

**Masks (optional):**
- Format: PNG, JPG
- Channels: 1 (grayscale)
- Values: 0 (background), 255 (plant)

### Step 2.4: Verify Data

```bash
python verify_data.py
```

**Expected output:**
```
✓ Data directory: data
✓ Training directory: data/train
✓ Training RGB images: data/train/rgb
  Files: 1000
✓ Training depth images: data/train/depth
  Files: 1000
✓ Training labels CSV: data/train/labels.csv
  Rows: 1000
  Columns: ['id', 'dry_weight']
  Dry weight range: [0.5000, 8.5000]
  
✓ Training data looks good!
```

---

## 3. Configuration

### Step 3.1: Review config.py

Key settings to check:

```python
# Data paths (adjust if needed)
DATA_DIR = Path("data")

# Model architecture
RGB_BACKBONE = "convnext_tiny"    # Options: efficientnetv2_s, resnet50
IMAGE_SIZE = 384                   # Larger = better quality, slower
BATCH_SIZE = 8                     # Reduce if out of memory

# Training
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_FOLDS = 5                      # For cross-validation

# Features
USE_PHENOTYPE_FEATURES = True      # Set False if no masks
```

### Step 3.2: Customize for Your System

**Limited GPU memory?**
```python
BATCH_SIZE = 4
IMAGE_SIZE = 256
```

**Fast prototyping?**
```python
NUM_EPOCHS = 20
NUM_FOLDS = 2
```

**No masks available?**
```python
USE_PHENOTYPE_FEATURES = False
```

---

## 4. Training

### Step 4.1: Test Model Build

Before full training, verify the model builds correctly:

```bash
python test_model.py
```

**Expected output:**
```
Building model...
✓ Model built successfully

Model Statistics:
  Total parameters: 28,589,057
  Trainable parameters: 28,589,057
  
✓ Forward pass successful
✓ Backward pass successful
🎉 All tests passed!
```

### Step 4.2: Start Training

```bash
python train.py
```

**What happens:**
1. Loads training data
2. Splits into 5 folds
3. Trains one model per fold
4. Saves best checkpoint per fold
5. Reports fold-wise metrics

**Training progress:**
```
Epoch 1/100 [Train]: 100%|████████| 125/125 [00:58<00:00]
Train Loss: 2.4567

Epoch 1/100 [Val]: 100%|████████| 31/31 [00:12<00:00]
Val Loss: 2.1234, MAE: 1.5678, RMSE: 2.0123, R²: 0.8456
✓ Best model saved (MAE: 1.5678)

...

Early stopping triggered at epoch 45
Best epoch was 30 with MAE: 0.8234
```

### Step 4.3: Monitor Training

**Metrics to watch:**
- **MAE**: Primary metric, lower is better
- **RMSE**: Penalizes large errors
- **R²**: Goodness of fit (closer to 1.0 is better)

**Expected training time:**
- Small dataset (100 samples): ~30 minutes
- Medium dataset (1000 samples): ~2-4 hours
- Large dataset (10000 samples): ~1-2 days

### Step 4.4: Check Outputs

After training:
```
output/
├── checkpoints/
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   ├── fold_2_best.pth
│   ├── fold_3_best.pth
│   └── fold_4_best.pth
└── logs/
```

**K-Fold Summary:**
```
K-Fold Cross-Validation Summary
Fold 1: MAE = 0.8234
Fold 2: MAE = 0.8456
Fold 3: MAE = 0.8123
Fold 4: MAE = 0.8567
Fold 5: MAE = 0.8345

Mean MAE: 0.8345 ± 0.0167
```

---

## 5. Prediction

### Step 5.1: Prepare Test Data

Ensure test data is ready:
```
data/test/
├── rgb/
│   ├── test_001.png
│   └── ...
└── depth/
    ├── test_001.png
    └── ...
```

### Step 5.2: Generate Predictions

```bash
python predict.py
```

**Process:**
1. Loads all 5 fold checkpoints
2. Generates predictions with each model
3. Averages predictions (ensemble)
4. Saves submission file

**Output:**
```
Found 5 fold checkpoints

Loading fold 1 checkpoint...
Loaded from epoch 30, MAE: 0.8234
Predicting: 100%|████████| 63/63 [00:15<00:00]

...

Ensemble predictions generated for 500 samples
Mean prediction std: 0.1234

✓ Submission saved to: output/submission.csv
```

### Step 5.3: Check Predictions

**submission.csv:**
```csv
id,dry_weight
test_001,2.4567
test_002,3.1234
test_003,1.8901
...
```

**Sanity checks:**
```python
import pandas as pd
df = pd.read_csv('output/submission.csv')

print(f"Predictions: {len(df)}")
print(f"Range: [{df['dry_weight'].min():.2f}, {df['dry_weight'].max():.2f}]")
print(f"Mean: {df['dry_weight'].mean():.2f}")
```

---

## 6. Evaluation

### Step 6.1: Analyze Validation Results

If you have ground truth for validation:

```python
# Create results CSV with columns: id, actual, predicted
import pandas as pd

# Load validation predictions
val_df = pd.read_csv('output/validation_results.csv')
```

### Step 6.2: Run Evaluation

```bash
python evaluate.py output/validation_results.csv
```

**Generates:**
- Prediction vs Actual scatter plot
- Residual analysis
- Error distribution
- Detailed metrics

**Output plots:**
```
output/evaluation/
├── predictions_vs_actual.png
└── residuals.png
```

### Step 6.3: Interpret Results

**Good signs:**
- R² > 0.85
- MAE < 10% of mean target value
- Residuals normally distributed around 0
- No systematic bias in predictions

**Warning signs:**
- Large outliers in residual plot
- Systematic over/under prediction
- High prediction uncertainty (std)

---

## 7. Troubleshooting

### Problem: Out of Memory

**Solution:**
```python
# In config.py
BATCH_SIZE = 4  # or 2
IMAGE_SIZE = 256
USE_AMP = True  # Enable mixed precision
```

### Problem: Slow Training

**Solutions:**
```python
# Reduce workers if CPU bottleneck
NUM_WORKERS = 2

# Use smaller backbone
RGB_BACKBONE = "efficientnetv2_s"

# Enable AMP
USE_AMP = True
```

### Problem: Poor Performance

**Checklist:**
1. ✓ Check data quality (aligned RGB/depth?)
2. ✓ Verify normalization (depth range?)
3. ✓ Increase augmentation
4. ✓ Try different loss weights
5. ✓ Train longer (more epochs)
6. ✓ Use larger backbone

### Problem: Masks Not Available

```python
# In config.py
USE_PHENOTYPE_FEATURES = False
```

Model will use only deep regression head.

### Problem: Model Not Learning

**Debug steps:**
```bash
# 1. Test with tiny dataset (10 samples)
# Should overfit quickly

# 2. Check loss values
# Should decrease consistently

# 3. Verify gradients
python test_model.py

# 4. Try higher learning rate
# LEARNING_RATE = 1e-3
```

### Problem: Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install torch torchvision timm albumentations
```

---

## Advanced Usage

### Custom Backbone

```python
# In config.py
RGB_BACKBONE = "efficientnetv2_m"  # Larger, more accurate
# or
RGB_BACKBONE = "resnet50"  # Classic, well-tested
```

### Adjust Loss Weights

```python
# In config.py
LAMBDA_SEG = 1.0      # Increase for better masks
LAMBDA_FINAL = 3.0    # Emphasize final prediction
```

### Fixed Alpha (No Learning)

```python
# In config.py
LEARNABLE_ALPHA = False
FIXED_ALPHA = 0.8  # 80% deep, 20% phenotype
```

### Export to ONNX

```python
import torch
from model import build_model
from config import Config

model = build_model(Config())
model.eval()

dummy_rgb = torch.randn(1, 3, 384, 384)
dummy_depth = torch.randn(1, 1, 384, 384)

torch.onnx.export(
    model,
    (dummy_rgb, dummy_depth),
    "model.onnx",
    input_names=['rgb', 'depth'],
    output_names=['prediction']
)
```

---

## Tips for Best Results

1. **Data Quality > Quantity**
   - Ensure RGB/depth alignment
   - Clean annotations
   - Representative samples

2. **Start Simple**
   - Train on subset first
   - Use default hyperparameters
   - Verify pipeline works

3. **Iterate Systematically**
   - Change one thing at a time
   - Track all experiments
   - Compare fold-wise metrics

4. **Ensemble is Key**
   - Always use all folds
   - Consider test-time augmentation
   - Blend multiple runs if time permits

5. **Monitor Validation**
   - Don't overfit
   - Use early stopping
   - Check prediction distribution

---

## Next Steps

✅ Completed training? Consider:
- Hyperparameter tuning
- Different architectures
- Additional modalities
- Test-time augmentation
- Model distillation

📝 Share results:
- Document your findings
- Compare with baselines
- Publish leaderboard scores

🚀 Deploy model:
- Convert to ONNX/TensorRT
- Build REST API
- Create web demo

---

**Questions?** Open an issue or check the documentation:
- README.md
- PROJECT_OVERVIEW.md
- QUICK_START.md

Good luck! 🌱🚀
