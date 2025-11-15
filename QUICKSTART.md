# Quick Start Guide 🚀

This guide will help you get started with the NeuralSprouts lettuce dry weight prediction framework.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Your lettuce image dataset with dry weight labels

## Installation

### 1. Set up Python Environment

```bash
# Clone the repository
git clone https://github.com/lehangajanayake/NeuralSprouts.git
cd NeuralSprouts

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your data in the following structure:

```
data/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── labels.csv
├── val/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── labels.csv
└── test/
    ├── image_001.jpg
    ├── image_002.jpg
    └── labels.csv
```

## Your First Model Training

### Step 1: Implement Your Dataset

Edit `example_usage.py` and implement the `LettuceDataset` class based on your data format:

```python
class LettuceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Example: Load from CSV
        import pandas as pd
        labels_df = pd.read_csv(f"{data_path}/labels.csv")
        
        self.image_paths = [f"{data_path}/{img}" for img in labels_df['image_name']]
        self.labels = labels_df['dry_weight'].values
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        # Get label
        label = self.labels[idx]
        
        # Apply augmentation
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Step 2: Configure Your Experiment

The default configuration is in `configs/cnn_v1_config.yaml`. You can modify it or create a new one:

```yaml
experiment:
  name: my_lettuce_experiment
  version: v1
  description: My first CNN model

model:
  name: SimpleCNN
  version: v1
  input_channels: 3
  input_size: 224
  dropout_rate: 0.5

data:
  train_path: data/train
  val_path: data/val
  test_path: data/test
  batch_size: 32
  num_workers: 4

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
```

### Step 3: Run Training

```bash
python example_usage.py
```

This will:
- Load your configuration
- Create the SimpleCNN model
- Set up data augmentation
- Start training
- Save checkpoints and logs to `experiments/`

### Step 4: Monitor Training

Training progress is automatically logged to:
- `experiments/lettuce_dry_weight_prediction/logs/` - Metric logs
- `experiments/lettuce_dry_weight_prediction/checkpoints/` - Model checkpoints
- `experiments/lettuce_dry_weight_prediction/results/` - Final results

## Running Tests

To verify everything is working correctly:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Next Steps

### 1. Experiment with Data Augmentation

Try different augmentation strategies:

```python
# Use more aggressive augmentation
from src.data_augmentation.private_pipeline_template import PrivatePipelineTemplate

train_augmentation = PrivatePipelineTemplate(
    name="aggressive",
    image_size=224,
    is_training=True
)
```

### 2. Create Your Own Model (v2)

Create a new file `src/models/cnn_v2.py`:

```python
from src.models.base_model import BaseModel
import torch.nn as nn

class ImprovedCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your improved architecture here
        
    def forward(self, x):
        # Your forward pass
        return x
```

Register it in `src/models/__init__.py`:

```python
from .cnn_v2 import ImprovedCNN

__all__ = ["BaseModel", "SimpleCNN", "ImprovedCNN"]
```

### 3. Compare Model Iterations

Use the experiment tracker to compare different runs:

```python
from src.utils import ExperimentTracker

tracker = ExperimentTracker("lettuce_dry_weight_prediction")

# List all runs
runs = tracker.list_runs(status='completed')
for run in runs:
    print(f"{run['run_id']}: val_loss = {run['results']['best_val_loss']:.4f}")

# Get the best run
best_run = tracker.get_best_run(metric='val_loss', minimize=True)
print(f"Best model: {best_run['run_id']}")
```

## Common Issues

### GPU Memory Issues

Reduce batch size in config:
```yaml
data:
  batch_size: 16  # or 8
```

### Training Too Slow

- Reduce image size: `input_size: 128`
- Reduce number of workers: `num_workers: 2`
- Use GPU if available

### Overfitting

- Increase dropout: `dropout_rate: 0.6`
- Use more aggressive augmentation
- Add weight decay: `weight_decay: 0.001`
- Reduce model complexity

## Project Structure Summary

```
NeuralSprouts/
├── src/
│   ├── models/              # Model architectures
│   ├── data_augmentation/   # Augmentation pipelines
│   ├── config/             # Configuration management
│   └── utils/              # Utilities and experiment tracking
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── experiments/            # Training results (auto-created)
├── data/                   # Your dataset (you create this)
├── example_usage.py        # Example training script
└── requirements.txt        # Dependencies
```

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Look at `example_usage.py` for a complete example
- Review `tests/` for usage examples
- Check experiment logs in `experiments/*/logs/`

## Tips for Success

1. **Start Simple**: Begin with the baseline SimpleCNN model
2. **Iterate Quickly**: Make small changes and test them
3. **Track Everything**: Use the experiment tracker to organize your runs
4. **Compare Fairly**: Use the same data splits and metrics across experiments
5. **Document**: Keep notes on what works and what doesn't
6. **Visualize**: Plot your metrics to understand training dynamics

Good luck with your lettuce dry weight predictions! 🌱🎯
