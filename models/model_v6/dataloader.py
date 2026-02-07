"""
Dataloader module for Model_v6.
Handles loading RGB and RGBD images with proper batching.
"""

import os
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class DualBranchDataset(Dataset):
    """Dataset for dual-branch CNN (RGB and RGBD images)."""
    
    def __init__(self, 
                 csv_file: str,
                 rgb_dir: str,
                 rgbd_dir: str,
                 image_id_col: str = "ID",
                 target_col: str = "DryWeight",
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            csv_file: Path to CSV file with image IDs and targets
            rgb_dir: Directory containing RGB images
            rgbd_dir: Directory containing RGBD images
            image_id_col: Column name for image IDs in CSV
            target_col: Column name for target (dry weight) in CSV
            transform: Optional transforms to apply
        """
        self.data = pd.read_csv(csv_file)
        self.rgb_dir = Path(rgb_dir)
        self.rgbd_dir = Path(rgbd_dir)
        self.image_id_col = image_id_col
        self.target_col = target_col
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get item by index.
        
        Returns:
            Tuple of (rgb_image, rgbd_image, dry_weight, image_id)
        """
        row = self.data.iloc[idx]
        image_id = str(row[self.image_id_col])
        dry_weight = float(row[self.target_col])
        
        # Load RGB image using PIL
        rgb_path = self.rgb_dir / f"{image_id}.png"
        rgb_image = Image.open(str(rgb_path))
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if rgb_image.mode != 'RGB':
            rgb_image = rgb_image.convert('RGB')
        rgb_image = np.array(rgb_image)
        
        # Load RGBD image
        rgbd_path = self.rgbd_dir / f"{image_id}.npy"
        if rgbd_path.exists():
            rgbd_image = np.load(str(rgbd_path))
        else:
            # Fallback: load RGB and create dummy depth
            rgbd_image = np.concatenate([rgb_image, np.zeros((rgb_image.shape[0], rgb_image.shape[1], 1), dtype=np.uint8)], axis=-1)
        
        # Ensure 4-channel RGBD
        if rgbd_image.shape[2] == 3:
            rgbd_image = np.concatenate([rgbd_image, np.zeros((rgbd_image.shape[0], rgbd_image.shape[1], 1), dtype=np.uint8)], axis=-1)
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        rgbd_tensor = torch.from_numpy(rgbd_image).permute(2, 0, 1).float() / 255.0
        dry_weight_tensor = torch.tensor(dry_weight, dtype=torch.float32)
        
        if self.transform:
            # Apply transforms if provided
            pass
        
        return rgb_tensor, rgbd_tensor, dry_weight_tensor, image_id


class PreprocessedDataset(Dataset):
    """Dataset for preprocessed images (already resized and augmented)."""
    
    def __init__(self,
                 csv_file: str,
                 rgb_dir: str,
                 rgbd_dir: str,
                 image_id_col: str = "ID",
                 target_col: str = "DryWeight"):
        """
        Initialize dataset.
        
        Args:
            csv_file: Path to CSV file with image IDs and targets
            rgb_dir: Directory containing preprocessed RGB images
            rgbd_dir: Directory containing preprocessed RGBD images
            image_id_col: Column name for image IDs
            target_col: Column name for target (dry weight)
        """
        self.data = pd.read_csv(csv_file)
        self.rgb_dir = Path(rgb_dir)
        self.rgbd_dir = Path(rgbd_dir)
        self.image_id_col = image_id_col
        self.target_col = target_col
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get preprocessed images.
        
        Returns:
            Tuple of (rgb_image, rgbd_image, dry_weight, image_id)
        """
        row = self.data.iloc[idx]
        image_id = str(row[self.image_id_col])
        dry_weight = float(row[self.target_col])
        
        # Load RGB image using PIL
        rgb_path = self.rgb_dir / f"{image_id}.png"
        rgb_image = Image.open(str(rgb_path))
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if rgb_image.mode != 'RGB':
            rgb_image = rgb_image.convert('RGB')
        rgb_image = np.array(rgb_image)
        
        # Load RGBD image
        rgbd_path = self.rgbd_dir / f"{image_id}.npy"
        if not rgbd_path.exists():
            raise FileNotFoundError(f"RGBD image not found: {rgbd_path}")
        rgbd_image = np.load(str(rgbd_path))
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        rgbd_tensor = torch.from_numpy(rgbd_image).permute(2, 0, 1).float() / 255.0
        dry_weight_tensor = torch.tensor(dry_weight, dtype=torch.float32)
        
        return rgb_tensor, rgbd_tensor, dry_weight_tensor, image_id


def create_dataloader(csv_file: str,
                      rgb_dir: str,
                      rgbd_dir: str,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      preprocessed: bool = True) -> DataLoader:
    """
    Create a dataloader for the dataset.
    
    Args:
        csv_file: Path to CSV file
        rgb_dir: Directory containing RGB images
        rgbd_dir: Directory containing RGBD images
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        preprocessed: Whether to use preprocessed dataset
    
    Returns:
        DataLoader instance
    """
    if preprocessed:
        dataset = PreprocessedDataset(csv_file, rgb_dir, rgbd_dir)
    else:
        dataset = DualBranchDataset(csv_file, rgb_dir, rgbd_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate function to handle variable-length batches."""
    rgb_images, rgbd_images, dry_weights, image_ids = zip(*batch)
    
    rgb_batch = torch.stack(rgb_images, dim=0)
    rgbd_batch = torch.stack(rgbd_images, dim=0)
    dry_weight_batch = torch.stack(dry_weights, dim=0)
    
    return rgb_batch, rgbd_batch, dry_weight_batch, list(image_ids)


if __name__ == "__main__":
    # Test dataloader
    config = Config()
    
    # Test with preprocessed data
    try:
        dataloader = create_dataloader(
            csv_file=config.TRAIN_CSV,
            rgb_dir=config.TRAIN_RGB_DIR,
            rgbd_dir=config.TRAIN_DEPTH_DIR,
            batch_size=4,
            shuffle=False,
            preprocessed=False
        )
        
        for rgb, rgbd, weight, image_ids in dataloader:
            print(f"RGB shape: {rgb.shape}")
            print(f"RGBD shape: {rgbd.shape}")
            print(f"Dry weight shape: {weight.shape}")
            print(f"Image IDs: {image_ids}")
            break
    except Exception as e:
        print(f"Error loading dataset: {e}")
