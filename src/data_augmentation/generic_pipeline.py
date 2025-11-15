"""
Generic augmentation pipeline for use across multiple models.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional
from .base_pipeline import AugmentationPipeline


class GenericAugmentationPipeline(AugmentationPipeline):
    """
    Generic data augmentation pipeline that can be shared across models.
    
    This pipeline includes common augmentations suitable for plant/lettuce images:
    - Geometric transformations (rotation, flip, shift)
    - Color adjustments (brightness, contrast, hue, saturation)
    - Quality adjustments (blur, noise)
    """
    
    def __init__(self, 
                 name: str = "generic",
                 image_size: int = 224,
                 is_training: bool = True,
                 augmentation_probability: float = 0.5):
        """
        Initialize generic augmentation pipeline.
        
        Args:
            name: Pipeline name
            image_size: Target image size for resizing
            is_training: Whether this is for training (applies augmentations) or validation
            augmentation_probability: Probability of applying each augmentation
        """
        config = {
            'image_size': image_size,
            'is_training': is_training,
            'augmentation_probability': augmentation_probability
        }
        super().__init__(name, config)
        
        self.image_size = image_size
        self.is_training = is_training
        self.aug_prob = augmentation_probability
        
        # Build the augmentation pipeline
        self.transform = self._build_pipeline()
        
    def _build_pipeline(self) -> A.Compose:
        """
        Build the albumentations pipeline.
        
        Returns:
            Composed augmentation pipeline
        """
        if self.is_training:
            # Training augmentations
            pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=self.aug_prob),
                A.VerticalFlip(p=self.aug_prob * 0.5),  # Less common for plants
                A.Rotate(limit=15, p=self.aug_prob),  # Small rotations
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=self.aug_prob
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=self.aug_prob
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=self.aug_prob * 0.7
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=self.aug_prob * 0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=self.aug_prob * 0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]    # ImageNet stds
                ),
                ToTensorV2()
            ])
        else:
            # Validation/test pipeline (no augmentation, just normalization)
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
            return (f"Generic augmentation pipeline (training mode) with:\n"
                   f"- Image size: {self.image_size}x{self.image_size}\n"
                   f"- Horizontal/vertical flips\n"
                   f"- Random rotations (±15°)\n"
                   f"- Shift, scale, rotate transformations\n"
                   f"- Brightness and contrast adjustments\n"
                   f"- Hue, saturation, value adjustments\n"
                   f"- Blur and noise (30% probability)\n"
                   f"- ImageNet normalization")
        else:
            return (f"Generic augmentation pipeline (validation mode) with:\n"
                   f"- Image size: {self.image_size}x{self.image_size}\n"
                   f"- ImageNet normalization only")
