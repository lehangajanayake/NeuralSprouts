# Quick Start Guide

Get up and running with the Multimodal Fusion Model in minutes!

## 1. Installation (2 minutes)

```bash
# Navigate to project directory
cd models/multimodal_fusion

# Install dependencies
pip install -r requirements.txt
```

## 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 3. Prepare Your Data (5 minutes)

Create this structure:

```
data/
  train/
    rgb/           # Put your RGB images here (e.g., 0001.png, 0002.png)
    depth/         # Put your depth images here (e.g., 0001.png, 0002.png)
    masks/         # Put binary masks here (optional)
    labels.csv     # Create CSV with: id,dry_weight
```

**Example labels.csv:**
```csv
id,dry_weight
0001,2.45
0002,3.12
0003,1.89
```

## 4. Verify Data Structure

```bash
python verify_data.py
```

This will check:
- ✓ All directories exist
- ✓ Images can be loaded
- ✓ labels.csv is properly formatted

## 5. Configure (Optional)

Edit `config.py` if needed:

```python
# Quick tweaks
BATCH_SIZE = 8          # Reduce if out of memory
IMAGE_SIZE = 384        # Try 256 for faster training
NUM_EPOCHS = 100        # Reduce for quick testing
NUM_FOLDS = 5           # Keep at 5 for best results
```

## 6. Train (1-24 hours depending on data size)

```bash
python train.py
```

**What happens:**
- 5-fold cross-validation
- Saves best model per fold
- Prints progress and metrics
- Early stopping if no improvement

**Expected output:**
```
Epoch 1/100
Train Loss: 2.4567
Val Loss: 2.1234, MAE: 1.5678, RMSE: 2.0123, R²: 0.8456
✓ Best model saved (MAE: 1.5678)
...

K-Fold Cross-Validation Summary
Fold 1: MAE = 1.5678
Fold 2: MAE = 1.6234
...
Mean MAE: 1.5890 ± 0.0456
```

## 7. Generate Predictions

Prepare test data:
```
data/
  test/
    rgb/           # Test RGB images
    depth/         # Test depth images
```

Run inference:
```bash
python predict.py
```

**Output:**
- `output/submission.csv` - Final predictions
- `output/submission_with_uncertainty.csv` - With prediction std

## 8. Evaluate (Optional)

If you have ground truth for validation set:

```bash
python evaluate.py path/to/results.csv
```

This generates:
- Prediction vs Actual plot
- Residual analysis
- Detailed metrics

---

## Common Issues & Solutions

### Out of Memory
```python
# In config.py
BATCH_SIZE = 4  # or 2
IMAGE_SIZE = 256
```

### Slow Training
```python
# In config.py
USE_AMP = True  # Enable mixed precision
NUM_WORKERS = 4  # Adjust based on CPU cores
```

### No Masks Available
```python
# In config.py
USE_PHENOTYPE_FEATURES = False
```

### Different Image Formats
Dataset loader automatically handles:
- RGB: .png, .jpg, .jpeg
- Depth: .png, .jpg, .npy, .tif
- Masks: .png, .jpg

---

## Quick Test (5 minutes)

Test with a small subset:

1. Create mini dataset (e.g., 20 samples)
2. Set in config.py:
   ```python
   NUM_EPOCHS = 5
   NUM_FOLDS = 2
   ```
3. Run: `python train.py`
4. Verify it runs without errors

---

## Next Steps

✅ Trained successfully? Try:
- Tuning hyperparameters
- Different backbones (efficientnetv2_s, resnet50)
- Adjusting loss weights
- More aggressive augmentation

📊 Analyze results:
- Check prediction distribution
- Identify failure cases
- Iterate on data quality

🚀 Production:
- Train on full dataset
- Use all folds for ensemble
- Generate final submission

---

**Need help?** Check the full README.md or open an issue.

Happy training! 🌱
