# Model V3: Simple ResNet Baseline

This is a simple baseline model using a pretrained ResNet18 for both regression (DryWeightShoot) and classification (Variety) tasks.

## Files

- `dataloader.py`: Loads RGB images and labels, applies normalization for ResNet.
- `model.py`: Defines a simple model using ResNet18 as a feature extractor with two heads.
- `train.py`: Training script with MAE and accuracy plots.

## Usage

- Update the paths in `train.py` if needed.
- Run training: `python train.py`

## Requirements

- torch
- torchvision
- pandas
- matplotlib

## Notes

- Only RGB images are used (no depth or leaf area).
- Images are resized and normalized for ResNet.
- The model is simple and intended for quick experimentation.
