import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


def group_aware_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Group-aware split: all variants of an original plant stay in same split.
    
    Groups by 'orig_id' if available; otherwise falls back to row-wise split.
    Returns train_indices, val_indices.
    """
    if 'orig_id' not in df.columns:
        # Fallback: row-wise split
        n = len(df)
        indices = np.arange(n)
        np.random.RandomState(seed).shuffle(indices)
        split_point = int(n * (1 - val_ratio))
        return indices[:split_point].tolist(), indices[split_point:].tolist()
    
    # Group by orig_id
    orig_groups = df.groupby('orig_id').groups
    orig_ids = sorted(orig_groups.keys())
    
    # Split orig_ids
    n_orig = len(orig_ids)
    n_val_orig = max(1, int(n_orig * val_ratio))
    
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(orig_ids)
    val_orig_ids = set(shuffled[:n_val_orig])
    
    train_indices = []
    val_indices = []
    for orig_id, group_indices in orig_groups.items():
        if orig_id in val_orig_ids:
            val_indices.extend(group_indices.tolist())
        else:
            train_indices.extend(group_indices.tolist())
    
    return train_indices, val_indices


class PlantDatasetV5(Dataset):
    """Loads RGB and Depth separately for 3-branch fusion (RGB, RGBD, Depth).
    
    CSV expectations:
    - image_id or id
    - DryWeightShoot (float)
    
    Files expected:
    - RGBImages/RGB_<id>.png
    - DepthImages/Depth_<id>.png
    
    Preprocessing: center crop 900x900 then resize 128x128.
    Images are loaded and cached in VRAM at initialization for maximum speed.
    """

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        labels_csv: str,
        *,
        image_size: int = 128,
        seed: int = 42,
        device: str = 'cuda',
    ):
        """Initialize dataset with VRAM caching.
        
        Args:
            rgb_dir: Directory containing RGB_<id>.png files
            depth_dir: Directory containing Depth_<id>.png files
            labels_csv: CSV with 'image_id'/'id' and 'DryWeightShoot'
            image_size: Expected image size (128x128)
            seed: Random seed
            device: Device to load images to ('cuda' or 'cpu')
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.device = device
        
        # Load CSV
        self.df = pd.read_csv(labels_csv)
        if 'image_id' not in self.df.columns and 'id' not in self.df.columns:
            raise ValueError(f"CSV must have 'image_id' or 'id' column")
        
        self.id_col = 'image_id' if 'image_id' in self.df.columns else 'id'
        
        # Required label column
        if 'DryWeightShoot' not in self.df.columns:
            raise ValueError("CSV must have 'DryWeightShoot' column")
        
        self.label_col = 'DryWeightShoot'
        
        # Load all images into memory
        print(f"Loading {len(self.df)} images into {device}...")
        self.rgb_cache = {}
        self.depth_cache = {}
        self.labels_cache = {}
        self.valid_indices = []
        
        for idx, row in self.df.iterrows():
            image_id = row[self.id_col]
            rgb_path = os.path.join(self.rgb_dir, f"RGB_{image_id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Warning: Missing files for {image_id}")
                continue
            
            try:
                # Load images
                rgb = Image.open(rgb_path).convert('RGB')
                depth = Image.open(depth_path).convert('L')
                
                # Normalize to [0, 1]
                rgb = np.array(rgb, dtype=np.float32) / 255.0
                depth = np.array(depth, dtype=np.float32) / 255.0
                
                # Convert to torch tensors on device
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)  # (3, H, W)
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)  # (1, H, W)
                
                # Create RGBD by stacking
                rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
                
                # Cache
                self.rgb_cache[idx] = rgb_tensor
                self.depth_cache[idx] = depth_tensor
                self.labels_cache[idx] = torch.tensor(row[self.label_col], dtype=torch.float32).to(device)
                self.valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error loading {image_id}: {e}")
        
        print(f"Successfully loaded {len(self.valid_indices)}/{len(self.df)} images")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rgb, rgbd, depth, label) all as torch tensors on device.
        
        Note: idx is a position in valid_indices, not the original dataframe index.
        """
        # Map position to actual dataframe index
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.valid_indices)}")
        
        df_idx = self.valid_indices[idx]
        
        rgb = self.rgb_cache[df_idx]  # (3, H, W)
        depth = self.depth_cache[df_idx]  # (1, H, W)
        rgbd = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        label = self.labels_cache[df_idx]  # scalar
        
        return rgb, rgbd, depth, label
