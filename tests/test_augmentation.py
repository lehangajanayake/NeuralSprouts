"""
Tests for data augmentation pipelines.
"""

import numpy as np
import pytest
from src.data_augmentation import GenericAugmentationPipeline, AugmentationPipeline


def test_generic_pipeline_initialization():
    """Test GenericAugmentationPipeline initialization."""
    pipeline = GenericAugmentationPipeline(
        name="test_pipeline",
        image_size=224,
        is_training=True
    )
    
    assert pipeline is not None
    assert pipeline.get_name() == "test_pipeline"
    assert isinstance(pipeline, AugmentationPipeline)


def test_generic_pipeline_training_mode():
    """Test GenericAugmentationPipeline in training mode."""
    pipeline = GenericAugmentationPipeline(
        image_size=224,
        is_training=True
    )
    
    # Create dummy image (RGB)
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Apply augmentation
    augmented = pipeline(dummy_image)
    
    # Check output is a tensor with correct shape
    assert augmented.shape == (3, 224, 224)  # CHW format


def test_generic_pipeline_validation_mode():
    """Test GenericAugmentationPipeline in validation mode."""
    pipeline = GenericAugmentationPipeline(
        image_size=224,
        is_training=False
    )
    
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    augmented = pipeline(dummy_image)
    
    assert augmented.shape == (3, 224, 224)


def test_pipeline_description():
    """Test pipeline description generation."""
    pipeline = GenericAugmentationPipeline(
        image_size=224,
        is_training=True
    )
    
    description = pipeline.get_description()
    assert isinstance(description, str)
    assert len(description) > 0
    assert "224" in description


def test_pipeline_different_image_sizes():
    """Test pipeline with different image sizes."""
    for size in [128, 224, 256]:
        pipeline = GenericAugmentationPipeline(
            image_size=size,
            is_training=False
        )
        
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        augmented = pipeline(dummy_image)
        
        assert augmented.shape == (3, size, size)


def test_pipeline_config():
    """Test pipeline configuration retrieval."""
    pipeline = GenericAugmentationPipeline(
        image_size=224,
        is_training=True,
        augmentation_probability=0.7
    )
    
    config = pipeline.get_config()
    assert config['image_size'] == 224
    assert config['is_training'] == True
    assert config['augmentation_probability'] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
