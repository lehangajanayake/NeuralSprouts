# Model v2 for NeuralSprouts

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
