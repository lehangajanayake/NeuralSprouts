# Multimodal Fusion Model for Lettuce Dry Weight Prediction

## Model Description

**Multimodal Fusion Model** is the most sophisticated architecture in the NeuralSprouts project, implementing state-of-the-art deep learning techniques for robust lettuce phenotyping. This model combines modern backbone architectures, semantic segmentation, and phenotype-based feature extraction in an end-to-end learnable framework.

### Architecture Overview

1. **Dual Encoder Design**
   - **RGB Encoder**: ConvNeXt or EfficientNet backbone
   - **Depth Encoder**: ConvNeXt or EfficientNet backbone
   - Both encoders process inputs independently and extract high-level features
   
2. **Mid-level Fusion Module**
   - Combines RGB and Depth features at intermediate representations
   - Enables multimodal interaction before task-specific heads
   - Preserves modality-specific information while learning cross-modal patterns

3. **Multi-task Heads**
   - **Segmentation Head**: Predicts binary lettuce masks (foreground/background)
     - Loss: Combined BCE + Dice loss for robust segmentation
     - Output: Pixel-wise lettuce probability maps
   
   - **Regression Head**: Predicts dry weight from fused features
     - Loss: Huber loss (robust to outliers)
     - Output: Direct dry weight estimate

4. **Phenotype Feature Extraction**
   - Extracts interpretable features from predicted segmentation masks:
     - Area fraction (percentage of image covered)
     - Bounding box dimensions (width, height)
     - Equivalent diameter (circular approximation)
     - Depth statistics (mean, std, median within mask region)
   
5. **Learnable Blending Network**
   - Input: Deep regression prediction + 7 phenotype features
   - 2-layer MLP combines deep learning and traditional features
   - Output: Final refined dry weight prediction
   - Learns optimal weighting between model-based and feature-based predictions

### Key Innovations
- **Hybrid Prediction**: Combines deep learning with classical phenotype features
- **Robust Training**: Huber loss, gradient clipping, mixed precision (AMP)
- **K-fold Cross-Validation**: 5-fold CV for reliable performance estimation
- **Ensemble Inference**: Averages predictions from all folds
- **Modern Backbones**: ConvNeXt/EfficientNet for strong feature extraction
- **Interpretability**: Phenotype features provide explainable predictions

### Training Strategy
- Multi-task optimization (segmentation + regression)
- Cosine annealing learning rate schedule
- Early stopping with patience
- Gradient clipping for stability
- Mixed precision training for efficiency

A complete PyTorch implementation of a multimodal multi-task deep learning system for predicting lettuce dry shoot weight from RGB and Depth images.

## Features

🎯 **Advanced Architecture**
- Dual encoder design (RGB + Depth) using modern backbones (ConvNeXt, EfficientNet)
- Mid-level fusion for effective multimodal integration
- Multi-task learning: Segmentation + Regression
- Phenotype feature extraction from predicted masks
- Learnable blending of deep and phenotype predictions

💪 **Robust Training**
- Huber loss for robust regression
- Combined BCE + Dice loss for segmentation
- K-fold cross-validation (5 folds)
- Mixed precision training (AMP)
- Gradient clipping and cosine annealing
- Early stopping

🔬 **Phenotype Features**
- Area fraction
- Bounding box dimensions
- Equivalent diameter
- Depth statistics (mean, std, median) within mask

📊 **Evaluation**
- MAE (Mean Absolute Error) - primary metric
- RMSE (Root Mean Squared Error)
- R² Score
- Ensemble predictions from all folds

## Project Structure

```
multimodal_fusion/
├── config.py           # Configuration and hyperparameters
├── dataset.py          # PyTorch Dataset and data loading
├── model.py            # Model architecture
├── losses.py           # Loss functions
├── utils.py            # Utility functions
├── train.py            # K-fold training script
├── predict.py          # Ensemble inference script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. **Clone the repository**
```bash
cd models/multimodal_fusion
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision timm numpy pandas opencv-python albumentations scikit-learn pyyaml tqdm
```

## Data Preparation

Structure your data as follows:

```
data/
├── train/
│   ├── rgb/              # RGB images (e.g., 0001.png)
│   ├── depth/            # Depth images (e.g., 0001.png or 0001.npy)
│   ├── masks/            # Binary masks (optional, e.g., 0001.png)
│   └── labels.csv        # Columns: id, dry_weight
├── test/
│   ├── rgb/              # Test RGB images
│   └── depth/            # Test depth images
└── sample_submission.csv # (optional) Template for submission
```

**labels.csv format:**
```csv
id,dry_weight
0001,2.45
0002,3.12
...
```

**Notes:**
- RGB images should be standard image formats (PNG, JPG)
- Depth images can be PNG (single channel) or NPY (numpy array)
- Masks should be binary (0/255) single-channel images
- If masks are not available, set `USE_PHENOTYPE_FEATURES = False` in `config.py`

## Configuration

Edit `config.py` to customize:

**Data paths:**
```python
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
```

**Model architecture:**
```python
RGB_BACKBONE = "convnext_tiny"  # Options: convnext_tiny, efficientnetv2_s, etc.
IMAGE_SIZE = 384
FUSION_CHANNELS = 256
```

**Training:**
```python
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_FOLDS = 5
USE_AMP = True  # Mixed precision
```

**Phenotype features:**
```python
USE_PHENOTYPE_FEATURES = True  # Set False if masks unavailable
LEARNABLE_ALPHA = True  # Learnable blending weight
```

## Training

Run k-fold cross-validation training:

```bash
python train.py
```

This will:
1. Load training data
2. Perform 5-fold cross-validation
3. Train a model for each fold
4. Save best checkpoint per fold to `output/checkpoints/`
5. Print fold-wise and average MAE

**Training logs:**
```
Epoch 1/100
Train Loss: 2.4567
Val Loss: 2.1234, MAE: 1.5678, RMSE: 2.0123, R²: 0.8456
✓ Best model saved (MAE: 1.5678)
```

**Output:**
- `output/checkpoints/fold_0_best.pth`
- `output/checkpoints/fold_1_best.pth`
- ...
- `output/checkpoints/fold_4_best.pth`

## Inference

Generate predictions for test set:

```bash
python predict.py
```

This will:
1. Load all fold checkpoints
2. Generate predictions for each fold
3. Average predictions across folds
4. Save submission file

**Output:**
- `output/submission.csv` - Standard submission format
- `output/submission_with_uncertainty.csv` - Includes prediction std

**Example output:**
```
id,dry_weight
test_001,2.45
test_002,3.12
...
```

## Model Architecture Details

### Dual Encoders
- **RGB Encoder**: ConvNeXt-Tiny (pretrained on ImageNet)
- **Depth Encoder**: ConvNeXt-Tiny (1-channel input)
- Extract multi-scale features at 4 levels

### Fusion Strategy
- Concatenate RGB and Depth features at each scale
- Apply 1×1 convolution to reduce channels
- Use deepest fused features for both tasks

### Segmentation Head
- UNet-style decoder with upsampling
- Produces binary mask (1 channel)
- Trained with BCE + Dice loss

### Regression Heads
1. **Deep Regression**: Global pooling → MLP → dry_weight
2. **Phenotype Regression**: Phenotype features → MLP → dry_weight
3. **Final Prediction**: α × deep + (1-α) × phenotype

### Phenotype Features (7 features)
1. Area fraction (normalized)
2. Bounding box height (normalized)
3. Bounding box width (normalized)
4. Equivalent diameter
5. Mean depth in mask
6. Std depth in mask
7. Median depth in mask

## Loss Function

Multi-task loss:
```
Total Loss = λ_seg × (BCE + Dice) 
           + λ_deep × Huber(deep_pred, y)
           + λ_phen × Huber(phen_pred, y)
           + λ_final × Huber(final_pred, y)
```

Default weights:
- λ_seg = 0.5
- λ_deep = 1.0
- λ_phen = 1.0
- λ_final = 2.0

## Metrics

- **MAE (Mean Absolute Error)**: Primary metric for competition
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Goodness of fit

## Tips for Better Performance

1. **Data Quality**
   - Ensure RGB and depth images are properly aligned
   - Handle missing/invalid depth values
   - Verify mask quality if using segmentation

2. **Hyperparameter Tuning**
   - Adjust learning rate and batch size based on GPU memory
   - Try different backbones: `efficientnetv2_s`, `resnet50`, `convnext_base`
   - Tune loss weights (λ values)

3. **Augmentation**
   - Enable/disable based on dataset size
   - Adjust augmentation strength in config

4. **Model Size**
   - For smaller datasets, use smaller backbones
   - Add dropout for regularization

5. **Training Strategy**
   - Increase patience if overfitting slowly
   - Use learning rate warmup for stability

## Troubleshooting

**Out of memory?**
- Reduce `BATCH_SIZE` in config
- Reduce `IMAGE_SIZE`
- Use smaller backbone

**Poor performance?**
- Check data normalization
- Verify RGB/depth alignment
- Try different loss weights
- Increase augmentation

**Masks unavailable?**
- Set `USE_PHENOTYPE_FEATURES = False`
- Model will use only deep regression

**Slow training?**
- Enable AMP: `USE_AMP = True`
- Reduce `NUM_WORKERS` if CPU bottleneck
- Use smaller image size

## Advanced Usage

### Custom Backbone
```python
# In config.py
RGB_BACKBONE = "efficientnetv2_m"
DEPTH_BACKBONE = "resnet50"
```

### Fixed Alpha
```python
# In config.py
LEARNABLE_ALPHA = False
FIXED_ALPHA = 0.7  # 70% deep, 30% phenotype
```

### Per-Image Depth Normalization
```python
# In config.py
DEPTH_NORM_STRATEGY = "per_image"  # or "global", "percentile"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_fusion_lettuce,
  title={Multimodal Fusion Model for Lettuce Dry Weight Prediction},
  author={NeuralSprouts Team},
  year={2026},
  url={https://github.com/yourusername/NeuralSprouts}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- TIMM library for backbone models
- Albumentations for data augmentation
- PyTorch team for the framework

---

**Questions or issues?** Please open an issue on GitHub or contact the team.

Happy training! 🌱🚀
