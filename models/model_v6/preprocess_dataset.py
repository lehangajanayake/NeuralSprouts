"""
Preprocessing dataset script for Model_v6.
Runs the full preprocessing pipeline for the training dataset.
"""

import os
import sys
from pathlib import Path

from config import Config
from preprocess import BatchPreprocessor


def main():
    """Main preprocessing function."""
    config = Config()
    config.print_config()
    
    print("\n" + "=" * 60)
    print("Starting Preprocessing Pipeline")
    print("=" * 60)
    
    # Create preprocessor
    preprocessor = BatchPreprocessor(config)
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    preprocessor.preprocess_dataset(
        rgb_dir=config.TRAIN_RGB_DIR,
        depth_dir=config.TRAIN_DEPTH_DIR,
        output_rgb_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBImages",
        output_depth_dir=f"{config.AUGMENTED_OUTPUT_DIR}/DepthImages",
        output_rgbd_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBDImages",
        csv_file=config.TRAIN_CSV,
        apply_augmentation=True
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nPreprocessed images saved to:")
    print(f"  - RGB: {config.AUGMENTED_OUTPUT_DIR}/RGBImages")
    print(f"  - RGBD: {config.AUGMENTED_OUTPUT_DIR}/RGBDImages")
    print(f"  - Logs: {config.LOG_DIR}/{config.AUGMENTATION_LOG_FILE}")
    print("\nNext: Run training script (python train.py)")


if __name__ == "__main__":
    main()
