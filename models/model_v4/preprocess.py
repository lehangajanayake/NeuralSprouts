import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


@dataclass
class PreprocessConfig:
    train_rgb_dir: str = '../../datasets/Training/RGBImages'
    train_depth_dir: str = '../../datasets/Training/DepthImages'
    labels_csv: str = '../../datasets/Training/Train.csv'

    out_rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    out_depth_dir: str = '../../datasets/Training/Augmented/DepthImages'
    out_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'

    image_size: int = 64
    crop_size: int = 900

    # how many augmented variants per original (not counting the original)
    num_aug_per_image: int = 3
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    w, h = img.size
    if w < crop_size or h < crop_size:
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))

    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    return img.crop((left, top, left + crop_size, top + crop_size))


def apply_aug(rgb: Image.Image, depth: Image.Image, rng: np.random.RandomState) -> Tuple[Image.Image, Image.Image]:
    """Apply random augmentations, keeping RGB and Depth geometrically aligned."""

    if T is None:
        return rgb, depth

    # flips
    if rng.rand() < 0.5:
        rgb = T.functional.hflip(rgb)
        depth = T.functional.hflip(depth)
    if rng.rand() < 0.5:
        rgb = T.functional.vflip(rgb)
        depth = T.functional.vflip(depth)

    # rotation (0/90/180/270) keeps depth realistic
    k = int(rng.randint(0, 4))
    if k:
        angle = 90 * k
        rgb = T.functional.rotate(rgb, angle)
        depth = T.functional.rotate(depth, angle)

    # RGB-only color jitter
    cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    rgb = cj(rgb)
    return rgb, depth


def preprocess_one(rgb: Image.Image, depth: Image.Image, cfg: PreprocessConfig) -> Tuple[Image.Image, Image.Image]:
    rgb = center_crop(rgb, cfg.crop_size)
    depth = center_crop(depth, cfg.crop_size)

    rgb = rgb.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    depth = depth.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    return rgb, depth


def main(cfg: Optional[PreprocessConfig] = None) -> None:
    cfg = cfg or PreprocessConfig()
    seed_everything(cfg.seed)

    os.makedirs(cfg.out_rgb_dir, exist_ok=True)
    os.makedirs(cfg.out_depth_dir, exist_ok=True)

    df = pd.read_csv(cfg.labels_csv)
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'id'}, inplace=True)

    required = {'id', 'Variety', 'DryWeightShoot'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")

    aug_rows = []
    next_id = 1

    for i, row in df.iterrows():
        orig_id = int(row['id'])
        rgb_path = os.path.join(cfg.train_rgb_dir, f'RGB_{orig_id}.png')
        depth_path = os.path.join(cfg.train_depth_dir, f'Depth_{orig_id}.png')
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        rgb0 = Image.open(rgb_path).convert('RGB')
        depth0 = Image.open(depth_path).convert('L')

        # original (no aug)
        rgb, depth = preprocess_one(rgb0, depth0, cfg)
        rgb.save(os.path.join(cfg.out_rgb_dir, f'RGB_{next_id}.png'))
        depth.save(os.path.join(cfg.out_depth_dir, f'Depth_{next_id}.png'))
        aug_rows.append({**row.to_dict(), 'id': next_id})
        next_id += 1

        # augmented variants
        for k in range(cfg.num_aug_per_image):
            rng = np.random.RandomState(cfg.seed + orig_id * 100 + k)
            rgb_aug, depth_aug = apply_aug(rgb0, depth0, rng)
            rgb_aug, depth_aug = preprocess_one(rgb_aug, depth_aug, cfg)

            rgb_aug.save(os.path.join(cfg.out_rgb_dir, f'RGB_{next_id}.png'))
            depth_aug.save(os.path.join(cfg.out_depth_dir, f'Depth_{next_id}.png'))
            aug_rows.append({**row.to_dict(), 'id': next_id})
            next_id += 1

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(df)} originals...")

    out_df = pd.DataFrame(aug_rows)
    out_df.to_csv(cfg.out_csv, index=False)
    print(f"Augmented images saved to: {cfg.out_rgb_dir} and {cfg.out_depth_dir}")
    print(f"Augmented CSV saved to: {cfg.out_csv} (rows={len(out_df)})")


if __name__ == '__main__':
    main()
