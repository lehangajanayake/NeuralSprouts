# CONFIG.md

## Configuration Guide for Model_v6

All parameters for preprocessing, augmentation, training, and model architecture are defined in `config.py` as a Python class. This allows for easy experimentation and version tracking.

## Preprocessing Parameters

### Image Resizing
```python
CENTER_CROP_SIZE = 1000      # Center crop to 1000x1000 pixels
RESIZE_SIZE = 96             # Final resize to 96x96 pixels
```

## Augmentation Parameters

### Enable/Disable Augmentations
```python
AUGMENTATIONS_ENABLED = True  # Enable all augmentations
```

### Horizontal Flip
```python
HORIZONTAL_FLIP_ENABLED = True
HORIZONTAL_FLIP_PROB = 0.5    # 50% probability
```

### Vertical Flip
```python
VERTICAL_FLIP_ENABLED = True
VERTICAL_FLIP_PROB = 0.5       # 50% probability
```

### Rotation
```python
ROTATION_ENABLED = True
ROTATION_ANGLE_RANGE = (-15, 15)  # Range in degrees
ROTATION_PROB = 0.5                # 50% probability
```

### Horizontal Shift
```python
HORIZONTAL_SHIFT_ENABLED = True
HORIZONTAL_SHIFT_MAX = 0.1    # 10% of image width
HORIZONTAL_SHIFT_PROB = 0.5   # 50% probability
```

### Vertical Shift
```python
VERTICAL_SHIFT_ENABLED = True
VERTICAL_SHIFT_MAX = 0.1      # 10% of image height
VERTICAL_SHIFT_PROB = 0.5     # 50% probability
```

## Logging Parameters

```python
LOG_DIR = "./logs"                          # Log directory
AUGMENTATION_LOG_FILE = "augmentations.csv" # Log filename
KEEP_IMAGE_ID = True                        # Retain original image IDs
```

## Training Parameters

```python
BATCH_SIZE = 32           # Batch size for training
LEARNING_RATE = 0.001     # Learning rate
EPOCHS = 100              # Number of training epochs
OPTIMIZER = "adam"        # Optimizer (adam, sgd)
LOSS_FUNCTION = "mse"     # Loss function (mse, mae)
DEVICE = "cuda"           # Device (cuda, cpu)
NUM_WORKERS = 4           # Number of data loading workers
```

## Model Architecture Parameters

```python
RGB_CHANNELS = 3                    # RGB image channels
RGBD_CHANNELS = 4                   # RGBD image channels
OUTPUT_DIM = 1                      # Output dimension (1 for single value prediction)
NUM_CONV_LAYERS = 3                 # Number of convolutional layers per branch
INITIAL_FILTERS = 32                # Initial number of filters
FILTER_MULTIPLIER = 2               # Filter multiplier between layers
FUSION_HIDDEN_DIM = 256             # Fusion layer hidden dimension
DROPOUT_RATE = 0.5                  # Dropout rate
```

## Versioning Parameters

```python
VERSION = "6.1"                     # Current version
EXPERIMENT_DIR = "./experiments/6.1" # Experiment directory
```

## How to Configure

1. **Edit config.py** directly to change parameters
2. **For different experiments**, create new version folders:
   ```bash
   python main.py setup --version 6.2
   # Edit config.py with new parameters
   python main.py preprocess
   python main.py train
   ```
3. **Compare results** across versions using visualization tools

---
For more details, see the `config.py` file and QUICK_START.md guide.