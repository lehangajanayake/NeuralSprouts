"""
Multimodal Fusion Model Package

A complete PyTorch implementation for multimodal multi-task learning
to predict lettuce dry weight from RGB + Depth images.
"""

__version__ = '1.0.0'
__author__ = 'NeuralSprouts Team'

from .model import build_model, MultimodalFusionModel
from .dataset import LettuceDataset, create_dataloaders
from .losses import MultiTaskLoss, HuberLoss, DiceLoss
from .config import Config
from .utils import set_seed, calculate_metrics, save_checkpoint, load_checkpoint

__all__ = [
    'build_model',
    'MultimodalFusionModel',
    'LettuceDataset',
    'create_dataloaders',
    'MultiTaskLoss',
    'HuberLoss',
    'DiceLoss',
    'Config',
    'set_seed',
    'calculate_metrics',
    'save_checkpoint',
    'load_checkpoint',
]
