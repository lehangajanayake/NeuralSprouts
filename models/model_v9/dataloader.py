"""Data loaders for model_v9 (RGB + depth + surface normals)."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from normal_utils import depth_image_to_normal_image, normal_image_to_tensor

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


MANDATORY_CROP = 1000


def _center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < MANDATORY_CROP or h < MANDATORY_CROP:
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))

    left = (w - MANDATORY_CROP) / 2
    top = (h - MANDATORY_CROP) / 2
    return img.crop((left, top, left + MANDATORY_CROP, top + MANDATORY_CROP))


def _load_or_build_normal(depth_img: Image.Image, normal_path: str) -> torch.Tensor:
    if os.path.exists(normal_path):
        normal_img = Image.open(normal_path).convert('RGB')
    else:
        os.makedirs(os.path.dirname(normal_path), exist_ok=True)
        normal_img = depth_image_to_normal_image(depth_img)
        normal_img.save(normal_path)
    return normal_image_to_tensor(normal_img)


class PlantDatasetV9(Dataset):
    """Returns RGB tensors, RGB+normal tensors, RGBD tensors, and targets."""

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        normal_dir: str,
        labels_csv: str,
        *,
        image_size: int = 96,
        augment: bool = False,
        seed: int = 42,
        enable_cache: bool = False,
        num_views: int = 1,
    ):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.normal_dir = normal_dir
        self.labels_csv = labels_csv
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.enable_cache = bool(enable_cache)
        self.num_views = max(1, int(num_views))

        self._cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        self.df = pd.read_csv(labels_csv)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)

        required = {'id', 'DryWeightShoot'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        keep_rows = []
        for _, row in self.df.iterrows():
            image_id = int(row['id'])
            rgb_path = os.path.join(self.rgb_dir, f"RGB_{image_id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")
            normal_path = os.path.join(self.normal_dir, f"Normal_{image_id}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
            row = row.copy()
            row['rgb_path'] = rgb_path
            row['depth_path'] = depth_path
            row['normal_path'] = normal_path
            keep_rows.append(row)
        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)

        if T is None:
            self.resize = None
            self.color_jitter = None
        else:
            self.resize = T.Resize((self.image_size, self.image_size))
            self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self) -> int:
        base_len = len(self.df)
        return base_len * self.num_views if self.num_views > 1 else base_len

    def _map_index(self, global_idx: int) -> Tuple[int, int]:
        base_len = len(self.df)
        if self.num_views > 1:
            base_idx = int(global_idx) % base_len
            view_idx = int(global_idx) // base_len
            return base_idx, view_idx
        return int(global_idx), 0

    def _maybe_aug(
        self,
        rgb: Image.Image,
        depth: Image.Image,
        base_idx: int,
        view_idx: int,
    ) -> Tuple[Image.Image, Image.Image]:
        if not self.augment or T is None:
            return rgb, depth

        rng = np.random.RandomState(self.seed + int(base_idx) * 9973 + int(view_idx) * 101)

        if rng.rand() < 0.5:
            rgb = T.functional.hflip(rgb)
            depth = T.functional.hflip(depth)
        if rng.rand() < 0.5:
            rgb = T.functional.vflip(rgb)
            depth = T.functional.vflip(depth)

        k = int(rng.randint(0, 4))
        if k:
            angle = 90 * k
            rgb = T.functional.rotate(rgb, angle)
            depth = T.functional.rotate(depth, angle)

        rgb = self.color_jitter(rgb)
        return rgb, depth

    def __getitem__(self, idx: int):
        base_idx, view_idx = self._map_index(idx)

        if self.enable_cache:
            key = (base_idx, view_idx)
            if key in self._cache:
                return self._cache[key]

        row = self.df.iloc[int(base_idx)]
        rgb = Image.open(row['rgb_path']).convert('RGB')
        depth = Image.open(row['depth_path']).convert('L')

        rgb = _center_crop(rgb)
        depth = _center_crop(depth)
        rgb, depth = self._maybe_aug(rgb, depth, base_idx, view_idx)

        if self.resize is not None:
            rgb = self.resize(rgb)
            depth = self.resize(depth)
        else:
            rgb = rgb.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            depth = depth.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]

        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)

        normal_t = _load_or_build_normal(depth, row['normal_path'])
        rgbn_t = torch.cat([rgb_t, normal_t], dim=0)

        sample = {
            'id': torch.tensor(int(row['id']), dtype=torch.long),
            'rgb': rgb_t,
            'rgbd': rgbd_t,
            'rgbn': rgbn_t,
            'normal': normal_t,
            'dry_weight': torch.tensor(float(row['DryWeightShoot']), dtype=torch.float32),
        }

        if self.enable_cache:
            self._cache[(base_idx, view_idx)] = sample

        return sample

    def build_cache(self, max_base_items: Optional[int] = None) -> None:
        self.enable_cache = True

        base_len = len(self.df)
        n = base_len if max_base_items is None else min(base_len, int(max_base_items))

        for base_idx in range(n):
            views = self.num_views if self.num_views > 1 else 1
            for view_idx in range(views):
                global_idx = base_idx + view_idx * base_len
                _ = self.__getitem__(global_idx)


class TestPlantDatasetV9(Dataset):
    """Loads RGB + Depth + normal maps for inference."""

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        normal_dir: str,
        csv_file: str,
        *,
        image_size: int = 96,
    ):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.normal_dir = normal_dir
        self.csv_file = csv_file
        self.image_size = int(image_size)

        df = pd.read_csv(csv_file)
        if 'image_id' in df.columns:
            df.rename(columns={'image_id': 'id'}, inplace=True)
        if 'id' not in df.columns:
            raise ValueError("Test CSV must contain 'image_id' or 'id'")

        keep = []
        for _, row in df.iterrows():
            image_id = int(row['id'])
            rgb_path = os.path.join(self.rgb_dir, f"RGB_{image_id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")
            normal_path = os.path.join(self.normal_dir, f"Normal_{image_id}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
            keep.append({'id': image_id, 'rgb_path': rgb_path, 'depth_path': depth_path, 'normal_path': normal_path})
        self.df = pd.DataFrame(keep)

        if T is None:
            self.resize = None
        else:
            self.resize = T.Resize((self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        image_id = int(row['id'])

        rgb = Image.open(row['rgb_path']).convert('RGB')
        depth = Image.open(row['depth_path']).convert('L')

        rgb = _center_crop(rgb)
        depth = _center_crop(depth)

        if self.resize is not None:
            rgb = self.resize(rgb)
            depth = self.resize(depth)
        else:
            rgb = rgb.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            depth = depth.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]

        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)

        normal_t = _load_or_build_normal(depth, row['normal_path'])
        rgbn_t = torch.cat([rgb_t, normal_t], dim=0)

        return {
            'id': torch.tensor(image_id, dtype=torch.long),
            'rgb': rgb_t,
            'rgbd': rgbd_t,
            'rgbn': rgbn_t,
        }
