"""
Setup script for Model_v6.
Creates directory structure, initializes configs, and prepares for training.
"""

import os
from pathlib import Path
import shutil
import json

from config import Config


def create_version_directories(base_dir: str = "./experiments", version: str = "6.1"):
    """Create version-specific directories."""
    version_dir = Path(base_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "checkpoints",
        "logs",
        "predictions",
        "visualizations",
        "augmentations"
    ]
    
    for subdir in subdirs:
        (version_dir / subdir).mkdir(exist_ok=True)
    
    print(f"Created version directory structure at {version_dir}")
    
    return version_dir


def initialize_config_file(version_dir: Path, config: Config):
    """Save config to version directory."""
    config_path = version_dir / "config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    print(f"Config saved to {config_path}")


def create_preprocessing_output_dirs(config: Config):
    """Create directories for preprocessed data."""
    aug_output_dir = Path(config.AUGMENTED_OUTPUT_DIR)
    
    # Create subdirectories
    (aug_output_dir / "RGBImages").mkdir(parents=True, exist_ok=True)
    (aug_output_dir / "RGBDImages").mkdir(parents=True, exist_ok=True)
    (aug_output_dir / "Logs").mkdir(parents=True, exist_ok=True)
    
    print(f"Created preprocessing output directories at {aug_output_dir}")


def setup_model_v6(version: str = "6.1"):
    """Main setup function."""
    print("=" * 60)
    print("Setting up Model_v6")
    print("=" * 60)
    
    # Update config with version
    config = Config()
    config.VERSION = version
    config.EXPERIMENT_DIR = f"./experiments/{version}"
    config.AUGMENTED_OUTPUT_DIR = f"../../datasets/Training/Augmented/{version}"
    
    # Create directories
    version_dir = create_version_directories("./experiments", version)
    initialize_config_file(version_dir, config)
    create_preprocessing_output_dirs(config)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review CONFIG.md for preprocessing/augmentation settings")
    print(f"2. Run preprocessing: python preprocess.py")
    print(f"3. Run training: python train.py")
    print(f"4. Run predictions: python predict.py")
    print(f"5. Generate visualizations: python visualize.py")
    print(f"\nExperiment directory: {version_dir}")


if __name__ == "__main__":
    setup_model_v6("6.1")
