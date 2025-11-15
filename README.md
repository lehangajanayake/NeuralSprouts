# NeuralSprouts 🌱

A comprehensive framework for predicting lettuce dry weight using deep learning. This repository is designed for iterative model development with organized experiment tracking and flexible data augmentation pipelines.

## 📋 Overview

This project provides a structured framework for building and iterating on AI models to predict lettuce dry weight from images. The architecture supports:

- **Multiple Model Iterations**: Organized structure for versioning and comparing different model architectures
- **Generic Data Augmentation**: Shared augmentation pipelines that can be used across models
- **Private Pipelines**: Model-specific augmentation strategies for fine-tuning
- **Experiment Tracking**: Comprehensive tracking of training runs, metrics, and results
- **Baseline CNN**: Simple CNN (v1) as a starting point for iterative improvements

## 🏗️ Project Structure

```
NeuralSprouts/
├── src/
│   ├── models/                    # Model architectures
│   │   ├── base_model.py         # Abstract base class for all models
│   │   └── cnn_v1.py             # Simple CNN (v1) - baseline model
│   ├── data_augmentation/        # Data augmentation pipelines
│   │   ├── base_pipeline.py      # Abstract base class for pipelines
│   │   └── generic_pipeline.py   # Generic augmentation pipeline
│   ├── config/                   # Configuration management
│   │   └── config.py             # Config class for managing settings
│   └── utils/                    # Utility functions
│       ├── experiment_tracker.py # Experiment tracking and versioning
│       └── trainer.py            # Generic training loop
├── configs/                      # Configuration files
│   └── cnn_v1_config.yaml       # Config for CNN v1
├── experiments/                  # Experiment results (created during training)
├── data/                        # Data directory (not included in repo)
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt             # Python dependencies
└── example_usage.py            # Example training script
```

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lehangajanayake/NeuralSprouts.git
cd NeuralSprouts
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Prepare your data**: Organize your lettuce images and labels in the `data/` directory
   - `data/train/` - Training images and labels
   - `data/val/` - Validation images and labels
   - `data/test/` - Test images and labels

2. **Configure your experiment**: Edit `configs/cnn_v1_config.yaml` or create a new config file

3. **Train the model**: Run the example script (after implementing your dataset loader)
```bash
python example_usage.py
```

## 🧠 Model Iterations

### Current Models

#### SimpleCNN (v1) - Baseline
- **Architecture**: 3 convolutional blocks + 2 fully connected layers
- **Purpose**: Establish baseline performance and test pipeline
- **Features**: 
  - Batch normalization
  - Dropout regularization
  - MaxPooling for downsampling

### Adding New Model Iterations

To add a new model version:

1. Create a new file in `src/models/` (e.g., `cnn_v2.py`)
2. Inherit from `BaseModel`
3. Implement the `forward()` method
4. Register in `src/models/__init__.py`
5. Create a new config file in `configs/`

Example:
```python
from src.models.base_model import BaseModel
import torch.nn as nn

class ImprovedCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Define your architecture
        
    def forward(self, x):
        # Implement forward pass
        return x
```

## 🎨 Data Augmentation

### Generic Pipeline

The `GenericAugmentationPipeline` provides standard augmentations suitable for plant images:

- Geometric: Rotation (±15°), horizontal/vertical flips, shift/scale
- Color: Brightness, contrast, hue, saturation adjustments
- Quality: Blur, noise
- Normalization: ImageNet statistics

**Training mode**: Applies all augmentations
**Validation mode**: Only resizing and normalization

### Creating Private Pipelines

For model-specific augmentation strategies:

```python
from src.data_augmentation import AugmentationPipeline
import albumentations as A

class CustomPipeline(AugmentationPipeline):
    def __init__(self, name="custom", config=None):
        super().__init__(name, config)
        self.transform = A.Compose([
            # Your custom augmentations
        ])
    
    def __call__(self, image, **kwargs):
        return self.transform(image=image)['image']
    
    def get_description(self):
        return "Custom augmentation pipeline for model X"
```

## 📊 Experiment Tracking

The `ExperimentTracker` automatically organizes your experiments:

```
experiments/
└── lettuce_dry_weight_prediction/
    ├── checkpoints/       # Saved model weights
    ├── configs/          # Run configurations
    ├── logs/             # Training metrics
    ├── results/          # Final results
    └── metadata.json     # Experiment metadata
```

### Key Features

- **Automatic organization**: Each run gets a unique ID with timestamp
- **Metric logging**: Track training/validation metrics per epoch
- **Best model saving**: Automatically saves the best performing checkpoint
- **Run comparison**: Compare different runs based on any metric
- **Configuration versioning**: Each run's configuration is saved

### Example Usage

```python
from src.utils import ExperimentTracker

tracker = ExperimentTracker("lettuce_dry_weight_prediction")

# Start a new run
run_id = tracker.start_run("cnn_v1_run1", config_dict)

# Log metrics during training
tracker.log_metrics(run_id, {"train_loss": 0.5, "val_loss": 0.6}, epoch=1)

# End the run
tracker.end_run(run_id, final_results, status="completed")

# Find the best run
best_run = tracker.get_best_run(metric='val_loss', minimize=True)
```

## ⚙️ Configuration

Configuration files use YAML format. Key sections:

### Model Configuration
```yaml
model:
  name: SimpleCNN
  version: v1
  input_channels: 3
  input_size: 224
  dropout_rate: 0.5
```

### Training Configuration
```yaml
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: reduce_on_plateau
  early_stopping_patience: 15
```

### Data Augmentation Configuration
```yaml
augmentation:
  pipeline: generic
  image_size: 224
  augmentation_probability: 0.5
```

## 🔬 Implementing Your Dataset

You need to implement a custom `Dataset` class based on your data format:

```python
from torch.utils.data import Dataset

class LettuceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load your image paths and labels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = load_image(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

## 🎯 Workflow for Iterative Development

1. **Baseline (v1)**: Start with SimpleCNN to establish baseline performance
2. **Experiment with augmentation**: Try different augmentation strategies
3. **Iterate on architecture**: Create v2, v3, etc. with improved architectures
4. **Fine-tune**: Use private pipelines for model-specific optimizations
5. **Compare**: Use experiment tracker to compare all iterations
6. **Select best**: Choose the best performing model for deployment

## 📈 Next Steps

- Implement your dataset loader based on your data format
- Train the baseline SimpleCNN (v1)
- Analyze results and identify areas for improvement
- Create improved model versions (v2, v3, etc.)
- Experiment with different augmentation strategies
- Fine-tune hyperparameters
- Ensemble multiple models for better performance

## 🤝 Contributing

This is a competition repository. Good luck! 🍀

## 📝 License

See LICENSE file for details.

---

**Note**: This framework is designed for flexibility and organization. Adapt it to your specific needs and data format. 
