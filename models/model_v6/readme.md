# Model_v6: Dual-Branch CNN for Dry Weight Prediction

## Overview
Model_v6 is designed to predict plant dry weight using a dual-branch convolutional neural network (CNN) architecture:
- **RGB Branch:** Processes RGB images.
- **RGBD Branch:** Processes RGBD images (RGB + Depth).
- **Fusion Layer:** Combines features from both branches before final prediction.
- **Output:** Predicts only the dry weight.

## Preprocessing Pipeline
- **Center Crop:** All images are center-cropped to 1000x1000 pixels.
- **Resize:** Cropped images are resized to 96x96 pixels before input to the model.
- **Augmentations:**
  - Random horizontal and vertical flips
  - Random rotations
  - Random vertical and horizontal shifts
- **Configurable Augmentations:** All augmentation parameters are configurable for experimentation.
- **Preprocessing Before Training:** Augmentations and resizing are performed before training to speed up the training loop and maximize GPU usage (GTX 1660ti).
- **Augmentation Logging:** Each image retains its original ID, and all applied augmentations are logged for traceability and outlier analysis.

## Configuration
- All preprocessing and augmentation parameters are configurable via a dedicated config file (recommended: YAML or Python class).
- Enables systematic testing of how each parameter affects model performance.

## Visualization & Debugging
- Tools/scripts provided to visualize:
  - Model attention maps (what the model "sees" in each image)
  - Prediction results and errors
  - Augmentation effects
- Debugging utilities to inspect model behavior and identify if the model focuses on correct image regions.

## Logging & Versioning
- All experiments, logs, and checkpoints are organized in versioned subfolders (e.g., `6.1/`, `6.2/`).
- Each run is tracked for reproducibility and analysis.
- Augmentation logs are linked to image IDs for performance tracking and outlier detection.

## GPU Utilization
- Preprocessing is performed before training to fully utilize the GTX 1660ti GPU and accelerate training.

## Documentation Files
- `CONFIG.md`: Details all configurable parameters and usage.
- `LOGGING.md`: Describes logging/versioning strategy and augmentation tracking.
- `VISUALIZATION.md`: Guides visualization and debugging tools.

## Getting Started
1. Review and edit `CONFIG.md` for desired preprocessing and augmentation settings.
2. Prepare your dataset as described above.
3. Run preprocessing scripts to generate augmented/resized images and logs.
4. Start training using the provided training scripts.
5. Use visualization/debugging tools to analyze model predictions and attention.
6. Track all experiments in versioned subfolders for easy comparison.

---
For more details, see the supporting documentation files and code comments.
