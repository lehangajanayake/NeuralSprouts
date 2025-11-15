"""
Template for creating model-specific private augmentation pipelines.

This file serves as a template for creating custom augmentation pipelines
that are tailored to specific models or later iterations. Copy this file
and modify it according to your needs.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from .base_pipeline import AugmentationPipeline


class PrivatePipelineTemplate(AugmentationPipeline):
    """
    Template for a private augmentation pipeline.
    
    Use this as a starting point for creating model-specific augmentation
    strategies that you've found work well for particular architectures
    or during later iterations of model development.
    
    Example use cases:
    - Heavier augmentation for deeper models
    - Domain-specific augmentations discovered during experimentation
    - Fine-tuning augmentation parameters for specific model architectures
    """
    
    def __init__(self, 
                 name: str = "private_template",
                 image_size: int = 224,
                 is_training: bool = True):
        """
        Initialize private pipeline.
        
        Args:
            name: Pipeline name (customize this)
            image_size: Target image size
            is_training: Whether this is for training or validation
        """
        config = {
            'image_size': image_size,
            'is_training': is_training,
            'pipeline_type': 'private'
        }
        super().__init__(name, config)
        
        self.image_size = image_size
        self.is_training = is_training
        self.transform = self._build_pipeline()
        
    def _build_pipeline(self) -> A.Compose:
        """
        Build the augmentation pipeline.
        
        Customize this method to implement your specific augmentation strategy.
        
        Returns:
            Composed augmentation pipeline
        """
        if self.is_training:
            # Example: More aggressive augmentation strategy
            pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.6),  # More aggressive rotation
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.6
                ),
                
                # Advanced geometric transformations
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(p=0.3),
                
                # Color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
                
                # Quality and style augmentations
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.4),
                
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                    A.ISONoise(p=1.0),
                ], p=0.4),
                
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.Sharpen(p=0.3),
                
                # Cutout/Coarse dropout for regularization
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    p=0.3
                ),
                
                # Normalization (use ImageNet statistics or compute your own)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation pipeline
            pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            
        return pipeline
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply augmentation pipeline to an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Augmented image as tensor
        """
        augmented = self.transform(image=image)
        return augmented['image']
    
    def get_description(self) -> str:
        """
        Get description of the pipeline.
        
        Returns:
            String describing the augmentations
        """
        if self.is_training:
            return (
                f"Private augmentation pipeline (training mode):\n"
                f"- Image size: {self.image_size}x{self.image_size}\n"
                f"- Aggressive geometric transformations (rotation up to ±30°)\n"
                f"- Elastic and grid distortions\n"
                f"- Extensive color augmentations\n"
                f"- Multiple blur and noise options\n"
                f"- CLAHE and sharpening\n"
                f"- Coarse dropout for regularization\n"
                f"- ImageNet normalization\n"
                f"\nNote: This is an example template - customize for your needs!"
            )
        else:
            return f"Private augmentation pipeline (validation mode): Resize + Normalize"


# Example: Create another specialized pipeline
class ConservativePipeline(AugmentationPipeline):
    """
    A more conservative augmentation pipeline.
    
    Use this for models that are sensitive to heavy augmentation
    or when you want to maintain image fidelity.
    """
    
    def __init__(self, name: str = "conservative", image_size: int = 224, is_training: bool = True):
        config = {'image_size': image_size, 'is_training': is_training}
        super().__init__(name, config)
        
        self.image_size = image_size
        self.is_training = is_training
        self.transform = self._build_pipeline()
    
    def _build_pipeline(self) -> A.Compose:
        """Build conservative augmentation pipeline."""
        if self.is_training:
            pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),  # Minimal rotation
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return pipeline
    
    def __call__(self, image: np.ndarray, **kwargs):
        return self.transform(image=image)['image']
    
    def get_description(self) -> str:
        return "Conservative pipeline: minimal augmentation for sensitive models"
