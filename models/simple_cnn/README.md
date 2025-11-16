# Simple CNN Model

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
