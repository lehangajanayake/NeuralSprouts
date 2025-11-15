"""
Tests for model implementations.
"""

import torch
import pytest
from src.models import SimpleCNN, BaseModel


def test_simple_cnn_initialization():
    """Test SimpleCNN can be initialized with config."""
    config = {
        'version': 'v1',
        'input_channels': 3,
        'input_size': 224,
        'dropout_rate': 0.5
    }
    
    model = SimpleCNN(config)
    assert model is not None
    assert model.get_version() == 'v1'
    assert isinstance(model, BaseModel)


def test_simple_cnn_forward_pass():
    """Test SimpleCNN forward pass with dummy input."""
    config = {
        'version': 'v1',
        'input_channels': 3,
        'input_size': 224,
        'dropout_rate': 0.5
    }
    
    model = SimpleCNN(config)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_simple_cnn_parameter_count():
    """Test SimpleCNN parameter counting."""
    config = {
        'version': 'v1',
        'input_channels': 3,
        'input_size': 224,
        'dropout_rate': 0.5
    }
    
    model = SimpleCNN(config)
    num_params = model.get_num_parameters()
    
    assert num_params > 0
    assert isinstance(num_params, int)


def test_model_checkpoint_save_load(tmp_path):
    """Test saving and loading model checkpoints."""
    config = {
        'version': 'v1',
        'input_channels': 3,
        'input_size': 224,
        'dropout_rate': 0.5
    }
    
    # Create and train model briefly
    model = SimpleCNN(config)
    dummy_input = torch.randn(2, 3, 224, 224)
    _ = model(dummy_input)
    
    # Save checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pth"
    model.save_checkpoint(
        str(checkpoint_path),
        epoch=1,
        metrics={'train_loss': 0.5, 'val_loss': 0.6}
    )
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    new_model = SimpleCNN(config)
    checkpoint = new_model.load_checkpoint(str(checkpoint_path))
    
    assert checkpoint['epoch'] == 1
    assert 'metrics' in checkpoint
    assert checkpoint['metrics']['train_loss'] == 0.5


def test_different_input_sizes():
    """Test SimpleCNN with different input sizes."""
    for input_size in [128, 224, 256]:
        config = {
            'version': 'v1',
            'input_channels': 3,
            'input_size': input_size,
            'dropout_rate': 0.5
        }
        
        model = SimpleCNN(config)
        dummy_input = torch.randn(2, 3, input_size, input_size)
        output = model(dummy_input)
        
        assert output.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
