import os
import random
import concurrent.futures as cf
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    image_size: int = 128  # Changed from 64 to 128
    crop_size: int = 900

    # how many augmented variants per original (not counting the original)
    num_aug_per_image: int = 30
    seed: int = 42

    # Parallelism / speed knobs
    num_workers: Optional[int] = None
    max_items: Optional[int] = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    """Center crop image to crop_size x crop_size."""
    w, h = img.size
    if w < crop_size or h < crop_size:
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))

    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    return img.crop((left, top, left + crop_size, top + crop_size))


def preprocess_single_image(
    rgb_path: str,
    depth_path: str,
    image_size: int,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(rgb_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')
    rgb = center_crop(rgb, crop_size)
    depth = center_crop(depth, crop_size)
    rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    depth = depth.resize((image_size, image_size), Image.Resampling.BILINEAR)
    rgb_array = np.array(rgb, dtype=np.uint8)
    depth_array = np.array(depth, dtype=np.uint8)
    return rgb_array, depth_array


def apply_augmentation(rgb: np.ndarray, depth: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations (rotation, brightness, contrast, etc.)."""
    rng = np.random.RandomState(seed)
    
    # Random rotation
    angle = rng.uniform(-30, 30)
    rgb = Image.fromarray(rgb).rotate(angle, resample=Image.Resampling.BILINEAR)
    depth = Image.fromarray(depth).rotate(angle, resample=Image.Resampling.BILINEAR)
    
    rgb = np.array(rgb, dtype=np.uint8)
    depth = np.array(depth, dtype=np.uint8)
    
    # Random horizontal flip
    if rng.rand() > 0.5:
        rgb = np.fliplr(rgb)
        depth = np.fliplr(depth)
    
    # Random brightness adjustment to RGB
    brightness_factor = rng.uniform(0.8, 1.2)
    rgb = np.clip(rgb.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment to RGB
    contrast_factor = rng.uniform(0.8, 1.2)
    mean_rgb = rgb.mean()
    rgb = np.clip((rgb.astype(np.float32) - mean_rgb) * contrast_factor + mean_rgb, 0, 255).astype(np.uint8)
    
    return rgb, depth


def _process_original(args):
    image_id, row, cfg = args
    rgb_path = os.path.join(cfg.train_rgb_dir, f"RGB_{image_id}.png")
    depth_path = os.path.join(cfg.train_depth_dir, f"Depth_{image_id}.png")
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        return None, image_id, "missing"
    try:
        rgb_arr, depth_arr = preprocess_single_image(rgb_path, depth_path, cfg.image_size, cfg.crop_size)
        Image.fromarray(rgb_arr).save(os.path.join(cfg.out_rgb_dir, f"RGB_{image_id}.png"))
        Image.fromarray(depth_arr).save(os.path.join(cfg.out_depth_dir, f"Depth_{image_id}.png"))
        return row, image_id, None
    except Exception as e:
        return None, image_id, str(e)


def _process_aug(args):
    row, cfg, aug_idx, seed_offset = args
    id_col = 'image_id' if 'image_id' in row.index else 'id'
    image_id = row[id_col]
    rgb_path = os.path.join(cfg.out_rgb_dir, f"RGB_{image_id}.png")
    depth_path = os.path.join(cfg.out_depth_dir, f"Depth_{image_id}.png")
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        return None, "missing"
    try:
        rgb_orig = np.array(Image.open(rgb_path))
        depth_orig = np.array(Image.open(depth_path))
        rgb_aug, depth_aug = apply_augmentation(rgb_orig, depth_orig, seed_offset)
        aug_id_str = f"{image_id}_aug_{aug_idx}"
        Image.fromarray(rgb_aug).save(os.path.join(cfg.out_rgb_dir, f"RGB_{aug_id_str}.png"))
        Image.fromarray(depth_aug).save(os.path.join(cfg.out_depth_dir, f"Depth_{aug_id_str}.png"))
        aug_row = row.copy()
        aug_row[id_col] = aug_id_str
        aug_row['orig_id'] = row['orig_id']
        return aug_row, None
    except Exception as e:
        return None, str(e)


def main(cfg: Optional[PreprocessConfig] = None) -> None:
    """Preprocess training data: center crop 900x900 and resize to 128x128.
    
    Also creates augmented variants with random transformations.
    Adds 'orig_id' column to track which original image each augmentation came from
    (prevents data leakage in train/val split).
    """
    if cfg is None:
        cfg = PreprocessConfig()
    
    seed_everything(cfg.seed)
    
    # Load CSV
    df = pd.read_csv(cfg.labels_csv)
    if 'image_id' not in df.columns and 'id' not in df.columns:
        raise ValueError(f"CSV must have 'image_id' or 'id' column. Got: {df.columns.tolist()}")
    
    id_col = 'image_id' if 'image_id' in df.columns else 'id'
    
    # Add orig_id column to original images to track them
    if 'orig_id' not in df.columns:
        df['orig_id'] = df[id_col]
    
    # Create output directories
    os.makedirs(cfg.out_rgb_dir, exist_ok=True)
    os.makedirs(cfg.out_depth_dir, exist_ok=True)
    
    # Process original images in parallel
    print(f"Processing {len(df)} original images...")
    n_workers = cfg.num_workers or os.cpu_count() or 4
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_process_original, [(row[id_col], row, cfg) for _, row in df.iterrows()]))

    kept_rows = []
    failed_ids = []
    for out_row, image_id, err in results:
        if err is None and out_row is not None:
            kept_rows.append(out_row)
        else:
            failed_ids.append(image_id)

    print(f"Successfully processed {len(kept_rows)}/{len(df)} originals")
    if failed_ids:
        print(f"Failed IDs: {failed_ids[:10]}")
    
    # Create augmented variants
    print(f"\nCreating {cfg.num_aug_per_image} augmented variants per original...")
    tasks = []
    max_items = cfg.max_items if cfg.max_items else len(kept_rows)
    for idx, row in enumerate(kept_rows[:max_items]):
        for aug_idx in range(cfg.num_aug_per_image):
            seed_offset = cfg.seed + (idx * cfg.num_aug_per_image) + aug_idx
            tasks.append((row, cfg, aug_idx, seed_offset))

    aug_rows = []
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_process_aug, tasks))

    for aug_row, err in results:
        if err is None and aug_row is not None:
            aug_rows.append(aug_row)
        elif err is not None:
            # keep short message for debug
            print(f"Aug error: {err}")
    
    # Create augmented CSV
    df_aug = pd.concat([df.copy(), pd.DataFrame(aug_rows)], ignore_index=True)
    df_aug.to_csv(cfg.out_csv, index=False)
    print(f"\nAugmented CSV saved: {cfg.out_csv} ({len(df_aug)} total rows)")
    print(f"✓ Data leakage prevention: All augmentations of the same original image will stay together in train/val split")


if __name__ == '__main__':
    main()
