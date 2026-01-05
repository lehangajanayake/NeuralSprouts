"""
Data verification and setup script.

Checks if data is properly structured and provides diagnostic information.
"""

import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

from config import Config


def check_directory(path, description):
    """Check if directory exists and count files."""
    if path.exists():
        files = list(path.glob('*.*'))
        print(f"✓ {description}: {path}")
        print(f"  Files: {len(files)}")
        return True
    else:
        print(f"✗ {description}: {path} [NOT FOUND]")
        return False


def check_file(path, description):
    """Check if file exists."""
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} [NOT FOUND]")
        return False


def verify_labels_csv(path):
    """Verify labels CSV format."""
    if not path.exists():
        return False
    
    try:
        df = pd.read_csv(path)
        print(f"\n  Labels CSV Info:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        if 'id' not in df.columns:
            print(f"  ✗ Missing 'id' column")
            return False
        
        if 'dry_weight' not in df.columns:
            print(f"  ✗ Missing 'dry_weight' column")
            return False
        
        print(f"  - Dry weight range: [{df['dry_weight'].min():.4f}, {df['dry_weight'].max():.4f}]")
        print(f"  - Dry weight mean: {df['dry_weight'].mean():.4f}")
        print(f"  - Sample IDs: {df['id'].head(3).tolist()}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return False


def test_load_sample(rgb_dir, depth_dir, sample_id):
    """Test loading a sample RGB and depth image."""
    print(f"\nTesting sample load: ID={sample_id}")
    
    # Try to load RGB
    rgb_found = False
    for ext in ['.png', '.jpg', '.jpeg']:
        rgb_path = rgb_dir / f"{sample_id}{ext}"
        if rgb_path.exists():
            try:
                img = cv2.imread(str(rgb_path))
                print(f"  ✓ RGB: {rgb_path.name} - Shape: {img.shape}")
                rgb_found = True
                break
            except Exception as e:
                print(f"  ✗ Error loading RGB: {e}")
    
    if not rgb_found:
        print(f"  ✗ RGB not found for ID {sample_id}")
    
    # Try to load Depth
    depth_found = False
    
    # Convert RGB_xxx to Depth_xxx for depth images
    depth_id = sample_id.replace('RGB_', 'Depth_')
    
    # Try .npy
    depth_path = depth_dir / f"{depth_id}.npy"
    if depth_path.exists():
        try:
            depth = np.load(str(depth_path))
            print(f"  ✓ Depth: {depth_path.name} - Shape: {depth.shape}, Type: numpy array")
            print(f"    Range: [{depth.min():.2f}, {depth.max():.2f}]")
            depth_found = True
        except Exception as e:
            print(f"  ✗ Error loading depth .npy: {e}")
    
    # Try image formats
    if not depth_found:
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            depth_path = depth_dir / f"{depth_id}{ext}"
            if depth_path.exists():
                try:
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    print(f"  ✓ Depth: {depth_path.name} - Shape: {depth.shape}")
                    print(f"    Range: [{depth.min():.2f}, {depth.max():.2f}]")
                    depth_found = True
                    break
                except Exception as e:
                    print(f"  ✗ Error loading depth image: {e}")
    
    if not depth_found:
        print(f"  ✗ Depth not found for ID {sample_id}")
    
    return rgb_found and depth_found


def main():
    """Main verification function."""
    print("="*60)
    print("Data Structure Verification")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Image size: {Config.IMAGE_SIZE}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Num folds: {Config.NUM_FOLDS}")
    print(f"  Use phenotype features: {Config.USE_PHENOTYPE_FEATURES}")
    
    print(f"\n{'='*60}")
    print("Checking Training Data")
    print("="*60)
    
    train_ok = True
    
    # Check directories
    if not check_directory(Config.DATA_DIR, "Data directory"):
        print("\n✗ Data directory not found!")
        print(f"  Please create: {Config.DATA_DIR}")
        return
    
    if not check_directory(Config.TRAIN_DIR, "Training directory"):
        train_ok = False
    
    if not check_directory(Config.RGB_TRAIN_DIR, "Training RGB images"):
        train_ok = False
    
    if not check_directory(Config.DEPTH_TRAIN_DIR, "Training depth images"):
        train_ok = False
    
    if Config.USE_PHENOTYPE_FEATURES:
        if not check_directory(Config.MASK_TRAIN_DIR, "Training masks"):
            print("  Warning: Masks not found. Set USE_PHENOTYPE_FEATURES=False in config.py")
    
    # Check labels
    if not check_file(Config.LABELS_PATH, "Training labels CSV"):
        train_ok = False
    else:
        if not verify_labels_csv(Config.LABELS_PATH):
            train_ok = False
    
    # Test loading a sample
    if train_ok and Config.LABELS_PATH.exists():
        df = pd.read_csv(Config.LABELS_PATH)
        if len(df) > 0:
            sample_id = str(df.iloc[0]['id'])
            test_load_sample(Config.RGB_TRAIN_DIR, Config.DEPTH_TRAIN_DIR, sample_id)
    
    print(f"\n{'='*60}")
    print("Checking Test Data")
    print("="*60)
    
    test_ok = True
    
    if not check_directory(Config.TEST_DIR, "Test directory"):
        print("  Note: Test data not required for training")
        test_ok = False
    
    if test_ok:
        if not check_directory(Config.RGB_TEST_DIR, "Test RGB images"):
            test_ok = False
        
        if not check_directory(Config.DEPTH_TEST_DIR, "Test depth images"):
            test_ok = False
        
        check_file(Config.SUBMISSION_PATH, "Sample submission (optional)")
    
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    
    if train_ok:
        print("✓ Training data looks good!")
        print("\nYou can now run:")
        print("  python train.py")
    else:
        print("✗ Training data has issues. Please fix them before training.")
        print("\nExpected structure:")
        print("""
data/
  train/
    rgb/           # RGB images
    depth/         # Depth images
    masks/         # Binary masks (optional)
    labels.csv     # Columns: id, dry_weight
  test/
    rgb/           # Test RGB images
    depth/         # Test depth images
        """)
    
    if test_ok:
        print("\n✓ Test data looks good!")
        print("  After training, run: python predict.py")
    elif Config.RGB_TEST_DIR.exists():
        print("\n! Test data incomplete (this is OK for training)")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
