import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


def _center_crop_900(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < 900 or h < 900:
        # fallback: crop to min side to avoid negative coords
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))

    left = (w - 900) / 2
    top = (h - 900) / 2
    return img.crop((left, top, left + 900, top + 900))


class PlantDatasetV4(Dataset):
    """Loads RGB + Depth and returns separate tensors for the two branches.

    CSV expectations (from repo Train.csv):
    - image_id or id
    - Variety (string)
    - DryWeightShoot (float)

    Files expected:
    - RGBImages/RGB_<id>.png
    - DepthImages/Depth_<id>.png

    Mandatory preprocessing (train + eval): center crop 900x900 then resize 64x64.
    """

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        labels_csv: str,
        *,
        image_size: int = 64,
        augment: bool = False,
        seed: int = 42,
    ):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.labels_csv = labels_csv
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)

        self.df = pd.read_csv(labels_csv)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)

        if 'DryWeightShoot' not in self.df.columns:
            raise ValueError("CSV must contain 'DryWeightShoot'")
        if 'Variety' not in self.df.columns:
            raise ValueError("CSV must contain 'Variety' for classification")

        varieties = sorted(self.df['Variety'].unique())
        self.variety2idx = {v: i for i, v in enumerate(varieties)}
        self.df['VarietyClass'] = self.df['Variety'].map(self.variety2idx)

        keep_rows = []
        for _, row in self.df.iterrows():
            image_id = int(row['id'])
            rgb_path = os.path.join(self.rgb_dir, f"RGB_{image_id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            row = row.copy()
            row['rgb_path'] = rgb_path
            row['depth_path'] = depth_path
            keep_rows.append(row)
        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)

        if T is None:
            self.resize = None
            self.color_jitter = None
        else:
            self.resize = T.Resize((self.image_size, self.image_size))
            self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self) -> int:
        return len(self.df)

    def _maybe_aug(self, rgb: Image.Image, depth: Image.Image, idx: int) -> Tuple[Image.Image, Image.Image]:
        if not self.augment or T is None:
            return rgb, depth

        # deterministic-ish per index
        rng = np.random.RandomState(self.seed + int(idx) * 9973)

        # flips
        if rng.rand() < 0.5:
            rgb = T.functional.hflip(rgb)
            depth = T.functional.hflip(depth)
        if rng.rand() < 0.5:
            rgb = T.functional.vflip(rgb)
            depth = T.functional.vflip(depth)

        # rotation multiples of 90 keeps it stable and avoids introducing depth artifacts
        k = int(rng.randint(0, 4))
        if k:
            angle = 90 * k
            rgb = T.functional.rotate(rgb, angle)
            depth = T.functional.rotate(depth, angle)

        # rgb-only photometric aug
        rgb = self.color_jitter(rgb)
        return rgb, depth

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        rgb_path = row['rgb_path']
        depth_path = row['depth_path']

        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        rgb = _center_crop_900(rgb)
        depth = _center_crop_900(depth)

        rgb, depth = self._maybe_aug(rgb, depth, idx)

        if self.resize is not None:
            rgb = self.resize(rgb)
            depth = self.resize(depth)
        else:
            rgb = rgb.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            depth = depth.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0  # (H,W,3)
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0  # (H,W)
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]

        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()

        # Depth normalization "separately" => keep as its own channel in [0,1].
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)

        variety_class = torch.tensor(int(row['VarietyClass']), dtype=torch.long)
        dry_weight = torch.tensor(float(row['DryWeightShoot']), dtype=torch.float32)

        # Return plain tensors in a dict so PyTorch's default_collate can batch them.
        return {
            'rgb': rgb_t,
            'rgbd': rgbd_t,
            'variety_class': variety_class,
            'dry_weight': dry_weight,
        }
