"""Offline preprocessing for Model_v6 with paired RGB/Depth outputs."""

import os
import random
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from config import Config


@dataclass
class PreprocessConfig:
    """Configuration for the offline preprocessing pipeline."""

    train_rgb_dir: str = Config.TRAIN_RGB_DIR
    train_depth_dir: str = Config.TRAIN_DEPTH_DIR
    labels_csv: str = Config.TRAIN_CSV

    out_rgb_dir: str = Config.AUGMENTED_RGB_DIR
    out_depth_dir: str = Config.AUGMENTED_DEPTH_DIR
    out_csv: str = Config.AUGMENTED_CSV

    crop_size: int = Config.CENTER_CROP_SIZE
    image_size: int = Config.RESIZE_SIZE

    num_aug_per_image: int = Config.PREPROCESS_NUM_AUG
    seed: int = Config.PREPROCESS_SEED
    num_workers: Optional[int] = Config.PREPROCESS_NUM_WORKERS
    max_items: Optional[int] = Config.PREPROCESS_MAX_ITEMS

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.out_rgb_dir, exist_ok=True)
        os.makedirs(self.out_depth_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)


COLOR_JITTER = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    w, h = img.size
    side = min(max(crop_size, 1), min(w, h))
    if w < crop_size or h < crop_size:
        side = min(w, h)
    left = (w - side) / 2
    top = (h - side) / 2
    return img.crop((left, top, left + side, top + side))


def preprocess_pair(rgb: Image.Image, depth: Image.Image, cfg: PreprocessConfig) -> Tuple[Image.Image, Image.Image]:
    rgb = center_crop(rgb, cfg.crop_size)
    depth = center_crop(depth, cfg.crop_size)

    rgb = rgb.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
    depth = depth.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
    return rgb, depth


def apply_aligned_augmentations(rgb: Image.Image, depth: Image.Image, rng: np.random.RandomState) -> Tuple[Image.Image, Image.Image]:
    """Apply paired geometric augments so RGB and depth stay aligned."""

    if rng.rand() < 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)
    if rng.rand() < 0.5:
        rgb = TF.vflip(rgb)
        depth = TF.vflip(depth)

    k = int(rng.randint(0, 4))
    if k:
        angle = 90 * k
        rgb = TF.rotate(rgb, angle, fill=0)
        depth = TF.rotate(depth, angle, fill=0)

    rgb = COLOR_JITTER(rgb)
    return rgb, depth


def _resolve_path(root: str, prefix: str, image_id: int) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(root, f"{prefix}{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _process_one_row(args) -> List[Dict]:
    row_index, row_dict, cfg_dict = args
    cfg = PreprocessConfig(**cfg_dict)

    image_id = row_dict.get('id', row_dict.get('image_id'))
    if image_id is None or (isinstance(image_id, float) and np.isnan(image_id)):
        return []
    original_id = row_dict.get('original_id', image_id)

    rgb_path = _resolve_path(cfg.train_rgb_dir, 'RGB_', int(image_id))
    depth_path = _resolve_path(cfg.train_depth_dir, 'Depth_', int(image_id))
    if rgb_path is None or depth_path is None:
        return []

    try:
        rgb0 = Image.open(rgb_path).convert('RGB')
        depth0 = Image.open(depth_path).convert('L')
    except Exception:
        return []

    per = 1 + max(0, int(cfg.num_aug_per_image))
    base_out_id = int(row_index) * per + 1

    outputs: List[Dict] = []

    # Save canonical sample (no aug)
    rgb_base, depth_base = preprocess_pair(rgb0, depth0, cfg)
    rgb_base.save(os.path.join(cfg.out_rgb_dir, f"RGB_{base_out_id}.png"))
    depth_base.save(os.path.join(cfg.out_depth_dir, f"Depth_{base_out_id}.png"))

    canonical_row = dict(row_dict)
    canonical_row['id'] = base_out_id
    canonical_row['original_id'] = original_id
    outputs.append(canonical_row)

    for aug_idx in range(cfg.num_aug_per_image):
        rng = np.random.RandomState(cfg.seed + int(image_id) * 101 + aug_idx)
        rgb_aug, depth_aug = apply_aligned_augmentations(rgb0, depth0, rng)
        rgb_aug, depth_aug = preprocess_pair(rgb_aug, depth_aug, cfg)

        out_id = base_out_id + aug_idx + 1
        rgb_aug.save(os.path.join(cfg.out_rgb_dir, f"RGB_{out_id}.png"))
        depth_aug.save(os.path.join(cfg.out_depth_dir, f"Depth_{out_id}.png"))

        augmented_row = dict(row_dict)
        augmented_row['id'] = out_id
        augmented_row['original_id'] = original_id
        outputs.append(augmented_row)

    return outputs


def run_preprocessing(cfg: Optional[PreprocessConfig] = None) -> None:
    cfg = cfg or PreprocessConfig()
    seed_everything(cfg.seed)
    cfg.ensure_output_dirs()

    df = pd.read_csv(cfg.labels_csv)
    if 'image_id' in df.columns:
        df = df.rename(columns={'image_id': 'id'})

    if 'original_id' not in df.columns:
        df['original_id'] = df['id']

    required_cols = {'id', 'DryWeightShoot'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")

    if cfg.max_items is not None:
        df = df.iloc[: int(cfg.max_items)].reset_index(drop=True)

    worker_count = cfg.num_workers
    if worker_count is None:
        cpu_count = os.cpu_count() or 1
        worker_count = max(1, cpu_count - 1)
    worker_count = max(1, int(worker_count))

    cfg_dict = {
        'train_rgb_dir': cfg.train_rgb_dir,
        'train_depth_dir': cfg.train_depth_dir,
        'labels_csv': cfg.labels_csv,
        'out_rgb_dir': cfg.out_rgb_dir,
        'out_depth_dir': cfg.out_depth_dir,
        'out_csv': cfg.out_csv,
        'crop_size': cfg.crop_size,
        'image_size': cfg.image_size,
        'num_aug_per_image': cfg.num_aug_per_image,
        'seed': cfg.seed,
        'num_workers': worker_count,
        'max_items': cfg.max_items,
    }

    tasks = [(idx, row.to_dict(), cfg_dict) for idx, (_, row) in enumerate(df.iterrows())]
    total = len(tasks)
    if not tasks:
        print("No rows to process.")
        return

    per = 1 + max(0, int(cfg.num_aug_per_image))
    print(f"Preprocessing {total} originals -> up to {total * per} samples (workers={worker_count})")

    all_rows: List[Dict] = []
    processed = 0

    def _iter_results(iterator):
        nonlocal processed
        for rows in iterator:
            processed += 1
            if rows:
                all_rows.extend(rows)
            if processed % 25 == 0 or processed == total:
                print(f"Processed {processed}/{total} originals...")

    if worker_count == 1:
        _iter_results(map(_process_one_row, tasks))
    else:
        with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
            _iter_results(executor.map(_process_one_row, tasks, chunksize=8))

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(cfg.out_csv, index=False)
    print(f"Saved RGB images to {cfg.out_rgb_dir}")
    print(f"Saved Depth images to {cfg.out_depth_dir}")
    print(f"Wrote augmented CSV with {len(out_df)} rows to {cfg.out_csv}")


if __name__ == "__main__":
    run_preprocessing()
