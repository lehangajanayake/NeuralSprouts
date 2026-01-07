# Model v2 for NeuralSprouts

## Model Description

**Model v2** extends Model v1 by introducing a third auxiliary branch dedicated to leaf area prediction, creating a comprehensive three-branch architecture for multi-task lettuce phenotyping.

### Architecture Overview
- **RGBD Branch (Primary)**: 6-layer CNN for dry weight regression
  - Input: 4×64×64 (RGB + Depth concatenated)
  - Output: Dry shoot weight (single scalar)
  
- **RGB Branch (Auxiliary 1)**: 5-layer CNN for variety classification
  - Input: 3×64×64 (RGB only)
  - Output: 3-class variety prediction
  
- **Depth Branch (Auxiliary 2)**: 6-layer CNN for leaf area regression
  - Input: 1×64×64 (Depth only)
  - 6 convolutional layers (16→32→64→128→256→256 filters)
  - Output: Leaf area prediction (single scalar)

### Key Improvements over v1
- **New depth-only branch** for leaf area estimation
- Three separate loss objectives for joint optimization
- Enhanced multi-task learning with complementary phenotype predictions
- Shared architectural patterns across branches for consistency

### Training Strategy
- Multi-task loss combining:
  - MSE loss for dry weight (RGBD branch)
  - Cross-Entropy loss for variety classification (RGB branch)
  - MSE loss for leaf area (Depth branch)
- Weighted loss aggregation for balanced training

This version adds an auxiliary branch for leaf area prediction using depth images, in addition to the main regression and classification branches. The dataloader and training/testing scripts are updated accordingly.

## Files

- `model.py`: Model definition with RGBD, RGB, and Depth (leaf area) branches.
- `dataloader.py`: Loads RGB, depth images, and leaf area labels.
- `train.py`: Training script for all branches.
- `test.py`: Testing script for all outputs.

## Data

- Expects CSVs for train/test with columns for RGB and depth image filenames, and optionally a CSV for leaf area labels.
- Update paths in scripts as needed.

## Usage

- Train: `python train.py`
- Test: `python test.py`

## Note

- Ensure all required CSVs and image folders exist.
- Adjust batch size, epochs, and learning rate as needed.
