"""
Configuration management for experiments and model training.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for organizing experiments and model iterations.
    
    This class handles loading, saving, and managing configurations for
    different model versions and training runs.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional dictionary with configuration values
        """
        self.config = config_dict or self._get_default_config()
        
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Dictionary with default configuration values
        """
        return {
            'experiment': {
                'name': 'lettuce_dry_weight_prediction',
                'version': 'v1',
                'description': 'Baseline CNN model',
            },
            'model': {
                'name': 'SimpleCNN',
                'version': 'v1',
                'input_channels': 3,
                'input_size': 224,
                'dropout_rate': 0.5,
            },
            'data': {
                'train_path': 'data/train',
                'val_path': 'data/val',
                'test_path': 'data/test',
                'batch_size': 32,
                'num_workers': 4,
            },
            'augmentation': {
                'pipeline': 'generic',
                'image_size': 224,
                'augmentation_probability': 0.5,
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'reduce_on_plateau',
                'early_stopping_patience': 15,
            },
            'output': {
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs',
                'save_best_only': True,
            }
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.input_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.input_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'model', 'training')
            
        Returns:
            Dictionary with section configuration
        """
        return self.config.get(section, {})
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with updates to apply
        """
        self._deep_update(self.config, updates)
    
    @staticmethod
    def _deep_update(base_dict: dict, update_dict: dict):
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
