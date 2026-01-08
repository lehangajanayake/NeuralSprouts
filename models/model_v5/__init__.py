"""Model V5: Triple-Branch Fusion for Plant Dry Weight Prediction."""

from .model import PlantV5TripleBranch, RGBBranch, RGBDBranch, DepthBranch, FusionFC
from .dataloader import PlantDatasetV5, group_aware_train_val_split

__all__ = [
    'PlantV5TripleBranch',
    'RGBBranch',
    'RGBDBranch',
    'DepthBranch',
    'FusionFC',
    'PlantDatasetV5',
    'group_aware_train_val_split',
]
