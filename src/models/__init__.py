"""
Models package for lettuce dry weight prediction.
"""

from .base_model import BaseModel
from .cnn_v1 import SimpleCNN

__all__ = ["BaseModel", "SimpleCNN"]
