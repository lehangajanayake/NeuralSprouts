"""
Model_v6: Dual-Branch CNN for Plant Dry Weight Prediction

A comprehensive framework for training and evaluating a dual-branch CNN model
that predicts plant dry weight using RGB and RGBD images.

Features:
- Configurable preprocessing with center crop and resize
- Augmentation pipeline with logging
- Dual-branch CNN (RGB + RGBD) with fusion layer
- Comprehensive visualization and debugging tools
- Version-based experiment tracking
"""

from .config import Config
from .model import create_model, DualBranchCNN
from .dataloader import create_dataloader, DualBranchDataset, PreprocessedDataset
from .preprocess import ImagePreprocessor, BatchPreprocessor
from .train import Trainer, train_model
from .predict import Predictor, predict_on_test_set
from .visualize import ModelVisualizer, create_comprehensive_visualization

__version__ = "6.0"
__all__ = [
    "Config",
    "create_model",
    "DualBranchCNN",
    "create_dataloader",
    "DualBranchDataset",
    "PreprocessedDataset",
    "ImagePreprocessor",
    "BatchPreprocessor",
    "Trainer",
    "train_model",
    "Predictor",
    "predict_on_test_set",
    "ModelVisualizer",
    "create_comprehensive_visualization"
]
