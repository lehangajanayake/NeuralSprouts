"""
PyTorch Dataset for RGB + Depth multimodal data.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from utils import normalize_depth


class LettuceDataset(Dataset):
    """
    Dataset for RGB + Depth images with optional segmentation masks.
    """
    
    def __init__(self, data_df, rgb_dir, depth_dir, mask_dir=None, 
                 transform=None, is_train=True, config=None):
        """
        Args:
            data_df: DataFrame with columns ['id', 'dry_weight'] (dry_weight optional for test)
            rgb_dir: Path to RGB images directory
            depth_dir: Path to depth images directory
            mask_dir: Path to mask images directory (optional)
            transform: Albumentations transform
            is_train: Whether this is training data
            config: Configuration object
        """
        self.data_df = data_df.reset_index(drop=True)
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.is_train = is_train
        self.config = config or Config()
        
        # Check if masks are available
        self.has_masks = self.mask_dir is not None and self.mask_dir.exists()
        
    def __len__(self):
        return len(self.data_df)
    
    def _load_rgb(self, image_id):
        """Load RGB image."""
        # Try common extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            rgb_path = self.rgb_dir / f"{image_id}{ext}"
            if rgb_path.exists():
                img = cv2.imread(str(rgb_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        raise FileNotFoundError(f"RGB image not found for ID: {image_id}")
    
    def _load_depth(self, image_id):
        """Load depth image (supports .png, .npy, etc.)."""
        # Convert RGB_xxx to Depth_xxx for depth image naming
        depth_id = image_id.replace('RGB_', 'Depth_')
        
        # Try .npy first
        depth_path = self.depth_dir / f"{depth_id}.npy"
        if depth_path.exists():
            depth = np.load(str(depth_path))
            return depth.astype(np.float32)
        
        # Try image formats
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            depth_path = self.depth_dir / f"{depth_id}{ext}"
            if depth_path.exists():
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                return depth.astype(np.float32)
        
        raise FileNotFoundError(f"Depth image not found for ID: {depth_id}")
    
    def _load_mask(self, image_id):
        """Load segmentation mask."""
        if not self.has_masks:
            return None
        
        # Convert RGB_xxx to Mask_xxx for mask naming convention
        mask_id = image_id.replace('RGB_', 'Mask_')
        
        for ext in ['.png', '.jpg']:
            mask_path = self.mask_dir / f"{mask_id}{ext}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                # Normalize to [0, 1]
                mask = (mask > 127).astype(np.float32)
                return mask
        
        # If mask not found, return None (will be handled gracefully)
        return None
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.data_df.iloc[idx]
        image_id = str(row['id'])
        
        # Load images
        rgb = self._load_rgb(image_id)
        depth = self._load_depth(image_id)
        mask = self._load_mask(image_id) if self.has_masks else None
        
        # Ensure depth is 2D
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # Resize if needed
        if rgb.shape[:2] != (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE):
            rgb = cv2.resize(rgb, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            depth = cv2.resize(depth, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            if mask is not None:
                mask = cv2.resize(mask, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=rgb, depth=depth, mask=mask)
                rgb = transformed['image']
                depth = transformed['depth']
                mask = transformed['mask']
            else:
                # Transform without mask
                transformed = self.transform(image=rgb, depth=depth)
                rgb = transformed['image']
                depth = transformed['depth']
        else:
            # Default: convert to tensor
            rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            if mask is not None:
                mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Normalize RGB
        rgb = self._normalize_rgb(rgb)
        
        # Normalize depth
        depth = normalize_depth(
            depth, 
            strategy=self.config.DEPTH_NORM_STRATEGY,
            global_mean=self.config.DEPTH_GLOBAL_MEAN,
            global_std=self.config.DEPTH_GLOBAL_STD
        )
        
        # Prepare output
        sample = {
            'rgb': rgb,
            'depth': depth,
            'id': image_id
        }
        
        if mask is not None:
            sample['mask'] = mask
        
        if self.is_train and 'dry_weight' in row:
            sample['dry_weight'] = torch.tensor(row['dry_weight'], dtype=torch.float32)
        
        return sample
    
    def _normalize_rgb(self, rgb):
        """Normalize RGB with ImageNet stats."""
        mean = torch.tensor(self.config.RGB_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.RGB_STD).view(3, 1, 1)
        return (rgb - mean) / std


def get_train_transform(config):
    """Get training augmentation transform."""
    if not config.USE_AUGMENTATION:
        return A.Compose([
            A.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD, max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={'depth': 'image', 'mask': 'mask'})
    
    return A.Compose([
        A.HorizontalFlip(p=config.AUG_FLIP_PROB),
        A.VerticalFlip(p=config.AUG_FLIP_PROB),
        A.Rotate(limit=config.AUG_ROTATE_LIMIT, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=0.5
        ),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ], additional_targets={'depth': 'image', 'mask': 'mask'})


def get_val_transform(config):
    """Get validation transform (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ], additional_targets={'depth': 'image', 'mask': 'mask'})


def create_dataloaders(train_df, val_df, config):
    """
    Create train and validation dataloaders.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        config: Configuration object
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = LettuceDataset(
        data_df=train_df,
        rgb_dir=config.RGB_TRAIN_DIR,
        depth_dir=config.DEPTH_TRAIN_DIR,
        mask_dir=config.MASK_TRAIN_DIR if config.USE_PHENOTYPE_FEATURES else None,
        transform=get_train_transform(config),
        is_train=True,
        config=config
    )
    
    val_dataset = LettuceDataset(
        data_df=val_df,
        rgb_dir=config.RGB_TRAIN_DIR,
        depth_dir=config.DEPTH_TRAIN_DIR,
        mask_dir=config.MASK_TRAIN_DIR if config.USE_PHENOTYPE_FEATURES else None,
        transform=get_val_transform(config),
        is_train=True,
        config=config
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader
