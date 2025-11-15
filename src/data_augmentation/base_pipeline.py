"""
Base augmentation pipeline interface.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class AugmentationPipeline(ABC):
    """
    Abstract base class for data augmentation pipelines.
    
    This allows for creating both generic pipelines (shared across models)
    and model-specific private pipelines for later iterations.
    """
    
    def __init__(self, name: str, config: dict = None):
        """
        Initialize the augmentation pipeline.
        
        Args:
            name: Name/identifier for this pipeline
            config: Configuration dictionary for augmentation parameters
        """
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply augmentation pipeline to an image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional arguments for augmentation
            
        Returns:
            Augmented image as numpy array
        """
        pass
    
    def get_name(self) -> str:
        """Get the pipeline name."""
        return self.name
    
    def get_config(self) -> dict:
        """Get the pipeline configuration."""
        return self.config
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the augmentation pipeline.
        
        Returns:
            String describing the augmentations applied
        """
        pass
