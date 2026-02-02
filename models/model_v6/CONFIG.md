# CONFIG.md

## Configurable Parameters

All preprocessing and augmentation parameters for model_v6 are configurable. Recommended to use a YAML file or Python class for easy experimentation.

### Preprocessing
- Center crop size: default 1000x1000
- Resize size: default 96x96

### Augmentations
- Horizontal flip: enable/disable, probability
- Vertical flip: enable/disable, probability
- Rotation: enable/disable, angle range
- Horizontal shift: enable/disable, max pixels
- Vertical shift: enable/disable, max pixels

### Logging
- Augmentation log file path
- Image ID retention

### Training
- Batch size
- Learning rate
- Epochs
- GPU usage (GTX 1660ti)

---
Edit this file to set your desired configuration before running preprocessing or training.