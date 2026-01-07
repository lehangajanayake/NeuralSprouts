# Simple CNN Model

## Model Description

**Simple CNN** is a lightweight, standalone baseline model designed for quick prototyping and experimentation. This model provides a minimal but functional architecture for lettuce dry weight prediction, serving as an entry point for understanding the dataset and establishing baseline performance.

### Architecture Overview
- **Type**: Basic Convolutional Neural Network
- **Task**: Single-task regression (dry weight prediction only)
- **Input**: RGB images (configurable size, default 224×224)
- **Simplicity**: All code contained in a single file for easy modification

### Network Structure
1. **Convolutional Layers** (3 blocks):
   - Conv1: 3 → 32 filters (3×3 kernels)
   - Conv2: 32 → 64 filters (3×3 kernels)
   - Conv3: 64 → 128 filters (3×3 kernels)
   - Each followed by ReLU activation and MaxPooling

2. **Fully Connected Layers** (3 layers):
   - FC1: Flattened features → 256 units
   - FC2: 256 → 64 units
   - FC3: 64 → 1 (dry weight output)
   - Dropout regularization between layers

### Key Features
- **Lightweight**: Minimal parameters, fast training
- **Standalone**: Single-file implementation for easy understanding
- **Customizable**: Simple architecture allows easy modifications
- **Baseline Model**: Establishes minimum performance expectations
- **Educational**: Clear code structure for learning PyTorch

### Use Cases
- Quick baseline experiments
- Testing data preprocessing pipelines
- Learning PyTorch model development
- Rapid prototyping of new ideas
- Sanity checking dataset quality

### Training Strategy
- MSE loss for regression
- Adam optimizer with configurable learning rate
- Basic training loop without advanced techniques
- Model checkpointing saves best weights

A standalone PyTorch implementation of a simple CNN for lettuce dry weight prediction.

## Model Architecture

- 3 Convolutional layers (32, 64, 128 filters)
- Max pooling after each conv layer
- 3 Fully connected layers (256, 64, 1)
- Dropout for regularization

## Usage

1. **Prepare your data**: Add images and labels to `../../datasets/`

2. **Update the dataset loader** in `train.py`:
   ```python
   class LettuceDataset(Dataset):
       def __init__(self, image_dir, labels_file, image_size=224):
           # Load your data here
           import pandas as pd
           df = pd.read_csv(labels_file)
           self.image_paths = df['image_name'].tolist()
           self.labels = df['dry_weight'].values
   ```

3. **Train the model**:
   ```bash
   cd models/simple_cnn
   python train.py
   ```

## Files

- `train.py` - Complete training script with model, dataset, and training loop
- `best_model.pth` - Saved weights of best model (created during training)

## Customization

All the code is in a single file (`train.py`) for simplicity. You can easily modify:
- Model architecture (change layers in `SimpleCNN` class)
- Training parameters (epochs, learning rate, batch size)
- Data augmentation (add transforms in `LettuceDataset`)

## Requirements

```bash
pip install torch torchvision numpy pillow
```
