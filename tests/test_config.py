"""
Tests for configuration management.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config import Config


def test_config_initialization():
    """Test Config initialization with default values."""
    config = Config()
    
    assert config is not None
    assert config.get('experiment.name') == 'lettuce_dry_weight_prediction'
    assert config.get('model.name') == 'SimpleCNN'


def test_config_get():
    """Test getting configuration values."""
    config = Config()
    
    assert config.get('model.input_size') == 224
    assert config.get('training.epochs') == 100
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_set():
    """Test setting configuration values."""
    config = Config()
    
    config.set('model.input_size', 256)
    assert config.get('model.input_size') == 256
    
    config.set('new.nested.key', 'value')
    assert config.get('new.nested.key') == 'value'


def test_config_get_section():
    """Test getting entire configuration sections."""
    config = Config()
    
    model_config = config.get_section('model')
    assert isinstance(model_config, dict)
    assert 'name' in model_config
    assert 'input_size' in model_config


def test_config_update():
    """Test updating configuration."""
    config = Config()
    
    updates = {
        'model': {
            'dropout_rate': 0.3
        },
        'training': {
            'epochs': 50
        }
    }
    
    config.update(updates)
    
    assert config.get('model.dropout_rate') == 0.3
    assert config.get('training.epochs') == 50
    assert config.get('model.name') == 'SimpleCNN'  # Unchanged


def test_config_save_load_yaml(tmp_path):
    """Test saving and loading configuration from YAML."""
    config = Config()
    config.set('model.input_size', 256)
    config.set('training.epochs', 50)
    
    # Save to YAML
    yaml_path = tmp_path / "test_config.yaml"
    config.to_yaml(str(yaml_path))
    
    assert yaml_path.exists()
    
    # Load from YAML
    loaded_config = Config.from_yaml(str(yaml_path))
    
    assert loaded_config.get('model.input_size') == 256
    assert loaded_config.get('training.epochs') == 50


def test_config_custom_initialization():
    """Test Config initialization with custom dictionary."""
    custom_dict = {
        'experiment': {
            'name': 'test_experiment',
            'version': 'v2'
        },
        'model': {
            'name': 'CustomModel'
        }
    }
    
    config = Config(custom_dict)
    
    assert config.get('experiment.name') == 'test_experiment'
    assert config.get('experiment.version') == 'v2'
    assert config.get('model.name') == 'CustomModel'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
