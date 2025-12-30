import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from typing import Optional, Dict

try:
    from torchvision import transforms
except Exception:
    transforms = None
    import numpy as np

class SimplePlantDataset(Dataset):
    def __init__(self, RGB_dir, labels_file, image_size=224, return_debug: bool = False):
        self.RGB_dir = RGB_dir
        self.labels_file = labels_file
        self.image_size = image_size
        self.return_debug = bool(return_debug)
        self.df = pd.read_csv(labels_file)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)
        if 'DryWeightShoot' not in self.df.columns:
            raise ValueError("CSV must contain a 'DryWeightShoot' column for regression")

        keep_rows = []
        for _, row in self.df.iterrows():
            image_id = row['id']
            rgb_path = os.path.join(self.RGB_dir, f"RGB_{image_id}.png")
            if not os.path.exists(rgb_path):
                print(f"Image not found: RGB: {rgb_path}")
                continue
            row = row.copy()
            row['rgb_path'] = rgb_path
            keep_rows.append(row)
        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)

        # Optional CPU-RAM cache to avoid repeating expensive PIL decode/resize
        # and normalization on every epoch.
        #
        # Two modes:
        # 1) enable_cache=True, num_views=1: cache the single preprocessed tensor per item.
        # 2) enable_cache=True, num_views>1: cache K random augmentations ("views") per item.
        #
        # NOTE: Default is off to avoid large RAM usage.
        self.enable_cache = False
        # Option B: when num_views > 1, the dataset length becomes N * num_views.
        # Each (base_idx, view_idx) is treated as a separate sample.
        self.num_views = 1
        self.cache_seed = 42
        self._cache = {}  # type: Dict[object, torch.Tensor]

        if transforms is None:
            # Fallback if torchvision isn't importable.
            self.image_size = (self.image_size, self.image_size)
            self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            self.transform = None
            self.aug_transform = None
        else:
            # Base preprocessing for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # Random augmentations (applied before resize/ToTensor/Normalize)
            # These are used only when building multi-view caches.
            self.aug_transform = transforms.Compose([
                transforms.CenterCrop(900),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ])

    def __len__(self):
        base_len = len(self.df)
        if self.num_views and self.num_views > 1:
            return base_len * int(self.num_views)
        return base_len

    def _map_index(self, global_idx: int):
        """Map a global sample index into (base_idx, view_idx)."""
        base_len = len(self.df)
        if self.num_views and self.num_views > 1:
            base_idx = global_idx % base_len
            view_idx = global_idx // base_len
            return int(base_idx), int(view_idx)
        return int(global_idx), 0

    def __getitem__(self, idx):
        base_idx, view_idx = self._map_index(idx)
        if self.enable_cache:
            key = (base_idx, view_idx)
            if key in self._cache:
                rgb = self._cache[key]
                dry_weight = float(self.df.iloc[base_idx]['DryWeightShoot'])
                y = torch.tensor(dry_weight, dtype=torch.float32)
                if self.return_debug:
                    meta = {
                        'global_idx': int(idx),
                        'base_idx': int(base_idx),
                        'view_idx': int(view_idx),
                        'id': int(self.df.iloc[base_idx]['id']),
                        'rgb_path': str(self.df.iloc[base_idx]['rgb_path']),
                        'cached': True,
                    }
                    # When serving from cache, we can still provide an unaugmented/original view
                    # by re-loading the image (debug only).
                    rgb_img = Image.open(meta['rgb_path']).convert('RGB')
                    if self.transform is not None:
                        rgb_orig = self.transform(rgb_img)
                    else:
                        rgb_img = rgb_img.resize(self.image_size, resample=Image.BILINEAR)
                        rgb_np = np.asarray(rgb_img, dtype=np.float32) / 255.0
                        rgb_orig = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
                        rgb_orig = (rgb_orig - self._mean) / self._std
                    return rgb, rgb_orig, y, meta
                return rgb, y

        rgb_path = self.df.iloc[base_idx]['rgb_path']

        rgb_img = Image.open(rgb_path).convert('RGB')

        # Deterministic augmentation for Option B (num_views > 1)
        # This makes the debugger reflect what each (base_idx, view_idx) view actually is,
        # even before you build the cache.
        aug_applied = False
        view_seed = None
        if self.transform is not None and self.num_views and self.num_views > 1:
            view_seed = int(self.cache_seed + base_idx * 1000 + view_idx)
            torch.manual_seed(view_seed)
            rgb_img_aug = self.aug_transform(rgb_img)
            aug_applied = True
        else:
            rgb_img_aug = rgb_img

        # Original (no random augmentation), used for debug views.
        if self.transform is not None:
            rgb_orig = self.transform(rgb_img)
        else:
            rgb_img_fallback = rgb_img.resize(self.image_size, resample=Image.BILINEAR)
            rgb_np = np.asarray(rgb_img_fallback, dtype=np.float32) / 255.0  # (H, W, C)
            rgb_orig = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()  # (C, H, W)
            rgb_orig = (rgb_orig - self._mean) / self._std

        # The view returned for training/debug.
        if self.transform is not None:
            rgb = self.transform(rgb_img_aug)
        else:
            # Fallback: no torchvision => no random aug implemented here.
            rgb = rgb_orig

        dry_weight = float(self.df.iloc[base_idx]['DryWeightShoot'])
        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)

        if self.enable_cache:
            # Keep tensors on CPU; DataLoader can pin memory and transfer async.
            self._cache[(base_idx, view_idx)] = rgb

        if self.return_debug:
            meta = {
                'global_idx': int(idx),
                'base_idx': int(base_idx),
                'view_idx': int(view_idx),
                'id': int(self.df.iloc[base_idx]['id']),
                'rgb_path': str(rgb_path),
                'cached': False,
                'aug_applied': bool(aug_applied),
                'view_seed': int(view_seed) if view_seed is not None else None,
            }
            return rgb, rgb_orig, dry_weight, meta

        return rgb, dry_weight

    def build_cache(self, max_items: Optional[int] = None):
        """Precompute and store preprocessed tensors in CPU RAM.

        This can significantly speed up training when the disk/CPU pipeline is
        the bottleneck. It does NOT move data to GPU (VRAM) because that rarely
        fits for real datasets and prevents multi-worker loading.
        """
        self.enable_cache = True

        # Cache always builds from the *base* dataset (before view expansion).
        base_len = len(self.df)
        n = base_len if max_items is None else min(base_len, max_items)

        # Deterministic cache generation
        torch.manual_seed(self.cache_seed)

        for i in range(n):
            if self.num_views <= 1:
                # Populate (i,0)
                _ = self._build_one(i, view_idx=0)
            else:
                for v in range(self.num_views):
                    _ = self._build_one(i, view_idx=v)

    def _build_one(self, idx: int, view_idx: int = 0):
        """Build one cached view deterministically."""
        rgb_path = self.df.iloc[idx]['rgb_path']
        rgb_img = Image.open(rgb_path).convert('RGB')

        if self.transform is None:
            # Fallback path (no torchvision). No random augmentation here.
            rgb_img = rgb_img.resize(self.image_size, resample=Image.BILINEAR)
            rgb_np = np.asarray(rgb_img, dtype=np.float32) / 255.0
            rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
            rgb = (rgb - self._mean) / self._std
        else:
            # For caching, apply deterministic random augmentation by reseeding per (idx, view_idx)
            if self.num_views > 1:
                torch.manual_seed(self.cache_seed + idx * 1000 + view_idx)
                rgb_img = self.aug_transform(rgb_img)
            rgb = self.transform(rgb_img)

        self._cache[(idx, view_idx)] = rgb
        return rgb
