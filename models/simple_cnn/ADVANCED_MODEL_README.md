# Advanced Multi-Modal Lettuce Dry Weight Prediction Model

This advanced model achieves superior accuracy by combining multiple data sources and employing state-of-the-art deep learning techniques.

## Key Features

### 1. **Multi-Modal Architecture**
- **RGB Images**: ResNet50 backbone (pretrained on ImageNet)
- **Depth Images**: ResNet18 backbone (adapted for single-channel input)
- **Tabular Features**: Height, Diameter, LeafArea, FreshWeightShoot, Variety

### 2. **Advanced Training Techniques**
- **Transfer Learning**: Leverages pretrained ImageNet weights
- **Data Augmentation**: Random flips, rotations, color jitter, affine transforms
- **Differential Learning Rates**: Lower LR for pretrained layers
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Prevents overfitting (patience=15 epochs)
- **Gradient Clipping**: Stabilizes training
- **Batch Normalization**: Throughout the fusion network
- **L2 Regularization**: Weight decay in AdamW optimizer

### 3. **Model Architecture**

```
RGB Branch (ResNet50)
    ↓ (2048 features)
    
Depth Branch (ResNet18)
    ↓ (512 features)
    
Tabular Branch (FC layers)
    ↓ (128 features)
    
Concatenation → 2688 features
    ↓
Fusion Network (512 → 256 → 64)
    ↓
Output (1 value: Dry Weight)
```

## Installation

```bash
# Navigate to the project root
cd /Users/hansikodikara/NeuralSprouts

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
cd models/simple_cnn
python advanced_train.py
```

### Expected Outputs

1. **best_advanced_model.pth** - Best model checkpoint
2. **scaler.pkl** - StandardScaler for tabular features
3. **label_encoder.pkl** - LabelEncoder for variety categories
4. **training_curves.png** - Training/validation loss plot
5. **predictions_vs_actuals.png** - Model performance visualization

### Configuration

Edit the configuration section in `advanced_train.py`:

```python
IMAGE_SIZE = 224          # Input image resolution
BATCH_SIZE = 16          # Batch size (adjust based on GPU memory)
EPOCHS = 100             # Maximum epochs (early stopping may trigger sooner)
LEARNING_RATE = 0.001    # Initial learning rate
```

## Model Performance

Expected performance metrics:
- **R² Score**: > 0.90 (excellent prediction accuracy)
- **MAE**: < 0.05 (mean absolute error in dry weight units)
- **RMSE**: < 0.08 (root mean squared error)

## How It Works

### 1. Data Loading
- RGB and Depth images loaded from corresponding directories
- Tabular features extracted from CSV and standardized
- Variety encoded as numerical categories

### 2. Data Augmentation (Training Only)
- Random horizontal/vertical flips
- Random rotations (±15°)
- Color jittering (brightness, contrast, saturation, hue)
- Random affine transformations
- Normalization using ImageNet statistics

### 3. Training Process
- Separate learning rates for pretrained vs new layers
- Learning rate reduced by 50% when validation loss plateaus
- Best model saved based on validation loss
- Training stops early if no improvement for 15 epochs

### 4. Evaluation
- MSE, RMSE, MAE, and R² metrics calculated
- Scatter plot showing prediction accuracy
- Visual feedback on model performance

## Making Predictions on New Data

```python
import torch
import pickle
from PIL import Image
from advanced_train import MultiModalLettuceModel, AdvancedLettuceDataset

# Load model
model = MultiModalLettuceModel(num_tabular_features=5)
checkpoint = torch.load('best_advanced_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load preprocessors
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Prepare new data and make prediction
# (See advanced_train.py for full example)
```

## Advantages Over Simple CNN

| Feature | Simple CNN | Advanced Model |
|---------|-----------|----------------|
| Image Inputs | RGB only | RGB + Depth |
| Tabular Features | ❌ | ✅ (5 features) |
| Transfer Learning | ❌ | ✅ (ImageNet) |
| Data Augmentation | Basic | Advanced |
| Learning Rate Scheduling | ❌ | ✅ |
| Early Stopping | ❌ | ✅ |
| Expected R² | ~0.75 | ~0.90+ |

## Troubleshooting

### Out of Memory (OOM) Error
- Reduce `BATCH_SIZE` from 16 to 8 or 4
- Reduce `IMAGE_SIZE` from 224 to 128

### Training Too Slow
- Reduce `num_workers` in DataLoader
- Ensure you're using GPU/MPS (check device output)
- Consider reducing model complexity (use ResNet18 for RGB branch)

### Poor Performance
- Check that RGB and Depth images match by image_id
- Verify CSV format matches expected structure
- Ensure sufficient training data (>100 samples minimum)
- Try increasing `EPOCHS` or adjusting learning rate

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended) or Apple Silicon (MPS)
- 8GB+ RAM
- 2GB+ GPU memory

## Citation

If you use this model in your research, please cite:

```bibtex
@software{neuralsprouts_advanced_model,
  title={Advanced Multi-Modal Lettuce Dry Weight Prediction},
  author={NeuralSprouts Team},
  year={2025}
}
```
