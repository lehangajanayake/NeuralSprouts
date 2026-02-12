import os
import random
import concurrent.futures as cf
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

    out_rgb_dir: str = '../../datasets/Training/Augmented_v8/RGBImages'
    out_depth_dir: str = '../../datasets/Training/Augmented_v8/DepthImages'
    out_csv: str = '../../datasets/Training/Augmented_v8/Train_aug.csv'

    image_size: int = 96
    crop_size: int = 1000  # minimum side length for random crops
    randomize_crop: bool = False  # allow disabling random crop-size sampling

    # how many augmented variants per original (not counting the original)
    num_aug_per_image: int = 40
    max_center_shift: int = 20 # max pixel shift for random pre-crop translations
    seed: int = 42
    depth_noise_std: float = 0.03  # Gaussian noise std (0-1 range) applied to depth channel
    depth_noise_prob: float = 0.7  # Probability of applying depth noise to a sample
    color_jitter_prob: float = 0.8  # Probability of applying color jitter to RGB

    # Parallelism / speed knobs
    num_workers: Optional[int] = None  # default computed in main()
    max_items: Optional[int] = None  # optionally limit number of originals processed


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


def shifted_center_crop(img: Image.Image, crop_size: int, shift: Tuple[int, int]) -> Image.Image:
    w, h = img.size
    if w < crop_size or h < crop_size:
        return center_crop(img, crop_size)

    dx, dy = shift
    left = (w - crop_size) / 2 + dx
    top = (h - crop_size) / 2 + dy
    left = min(max(0, left), max(0, w - crop_size))
    top = min(max(0, top), max(0, h - crop_size))
    return img.crop((left, top, left + crop_size, top + crop_size))


def random_center_shift(rng: np.random.RandomState, max_shift: int) -> Tuple[int, int]:
    """Sample a deterministic shift whose magnitude is in [0, max_shift]."""

    if max_shift <= 0:
        return 0, 0

    def _sample_axis() -> int:
        magnitude = int(rng.randint(0, max_shift + 1))
        if magnitude == 0:
            return 0
        sign = -1 if rng.rand() < 0.5 else 1
        return sign * magnitude

    return _sample_axis(), _sample_axis()


def random_crop_size(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    rng: np.random.RandomState,
) -> int:
    """Return a crop side sampled between the full frame and cfg.crop_size."""

    max_crop = min(min(rgb.size), min(depth.size))
    min_crop = min(int(cfg.crop_size), max_crop)
    if not cfg.randomize_crop:
        return min_crop
    if min_crop >= max_crop:
        return max_crop
    return int(rng.randint(min_crop, max_crop + 1))


def _apply_depth_noise(depth: Image.Image, rng: np.random.RandomState, std: float) -> Image.Image:
    if std <= 0.0:
        return depth
    depth_np = np.asarray(depth, dtype=np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=float(std), size=depth_np.shape).astype(np.float32)
    depth_np = np.clip(depth_np + noise, 0.0, 1.0)
    depth_uint8 = (depth_np * 255.0).astype(np.uint8)
    return Image.fromarray(depth_uint8, mode='L')


def _maybe_apply_depth_noise(
    depth: Image.Image,
    rng: np.random.RandomState,
    std: float,
    prob: float,
) -> Image.Image:
    prob = float(prob)
    if std <= 0.0 or prob <= 0.0:
        return depth
    prob = max(0.0, min(1.0, prob))
    if rng.rand() > prob:
        return depth
    return _apply_depth_noise(depth, rng, std)


def apply_aug(
    rgb: Image.Image,
    depth: Image.Image,
    rng: np.random.RandomState,
    cfg: PreprocessConfig,
) -> Tuple[Image.Image, Image.Image]:
    """Apply random augmentations, keeping RGB and Depth geometrically aligned."""

    if T is None:
        depth = _maybe_apply_depth_noise(depth, rng, cfg.depth_noise_std, cfg.depth_noise_prob)
        return rgb, depth

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

    cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    if rng.rand() < max(0.0, min(1.0, float(cfg.color_jitter_prob))):
        rgb = cj(rgb)
    depth = _maybe_apply_depth_noise(depth, rng, cfg.depth_noise_std, cfg.depth_noise_prob)
    return rgb, depth


def preprocess_one(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    rng: np.random.RandomState,
    shift: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Image.Image]:
    crop_side = random_crop_size(rgb, depth, cfg, rng)
    if shift is None:
        shift = random_center_shift(rng, int(cfg.max_center_shift))

    if shift != (0, 0):
        rgb = shifted_center_crop(rgb, crop_side, shift)
        depth = shifted_center_crop(depth, crop_side, shift)
    else:
        rgb = center_crop(rgb, crop_side)
        depth = center_crop(depth, crop_side)

    rgb = rgb.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    depth = depth.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    return rgb, depth


def _process_one_row(args) -> List[Dict]:
    row_index, row_dict, cfg_dict = args
    cfg = PreprocessConfig(**cfg_dict)

    row_dict = dict(row_dict)
    orig_id = int(row_dict['id'])
    row_dict['original_id'] = orig_id
    rgb_path = os.path.join(cfg.train_rgb_dir, f'RGB_{orig_id}.png')
    depth_path = os.path.join(cfg.train_depth_dir, f'Depth_{orig_id}.png')
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        return []

    try:
        rgb0 = Image.open(rgb_path).convert('RGB')
        depth0 = Image.open(depth_path).convert('L')
    except Exception:
        return []

    per = 1 + int(cfg.num_aug_per_image)
    base_out_id = int(row_index) * per + 1

    out_rows: List[Dict] = []

    base_rng = np.random.RandomState(int(cfg.seed) + orig_id)
    shift0 = random_center_shift(base_rng, int(cfg.max_center_shift))
    rgb, depth = preprocess_one(rgb0, depth0, cfg, base_rng, shift=shift0)
    rgb.save(os.path.join(cfg.out_rgb_dir, f'RGB_{base_out_id}.png'))
    depth.save(os.path.join(cfg.out_depth_dir, f'Depth_{base_out_id}.png'))
    r0 = dict(row_dict)
    r0['id'] = base_out_id
    out_rows.append(r0)

    for k in range(int(cfg.num_aug_per_image)):
        out_id = base_out_id + 1 + k

        rng = np.random.RandomState(int(cfg.seed) + orig_id * 100 + k)
        rgb_aug, depth_aug = apply_aug(rgb0, depth0, rng, cfg)
        shift = random_center_shift(rng, int(cfg.max_center_shift))
        rgb_aug, depth_aug = preprocess_one(rgb_aug, depth_aug, cfg, rng, shift=shift)

        rgb_aug.save(os.path.join(cfg.out_rgb_dir, f'RGB_{out_id}.png'))
        depth_aug.save(os.path.join(cfg.out_depth_dir, f'Depth_{out_id}.png'))

        rk = dict(row_dict)
        rk['id'] = out_id
        out_rows.append(rk)

    return out_rows


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

    if cfg.max_items is not None:
        df = df.iloc[: int(cfg.max_items)].reset_index(drop=True)

    if cfg.num_workers is None:
        cpu = os.cpu_count() or 1
        cfg.num_workers = max(1, cpu - 1)
    cfg.num_workers = max(1, int(cfg.num_workers))

    cfg_dict = {
        'train_rgb_dir': cfg.train_rgb_dir,
        'train_depth_dir': cfg.train_depth_dir,
        'labels_csv': cfg.labels_csv,
        'out_rgb_dir': cfg.out_rgb_dir,
        'out_depth_dir': cfg.out_depth_dir,
        'out_csv': cfg.out_csv,
        'image_size': cfg.image_size,
        'crop_size': cfg.crop_size,
        'randomize_crop': cfg.randomize_crop,
        'num_aug_per_image': cfg.num_aug_per_image,
        'max_center_shift': cfg.max_center_shift,
        'seed': cfg.seed,
        'depth_noise_std': cfg.depth_noise_std,
        'depth_noise_prob': cfg.depth_noise_prob,
        'color_jitter_prob': cfg.color_jitter_prob,
        'num_workers': cfg.num_workers,
        'max_items': cfg.max_items,
    }

    tasks = [(i, row.to_dict(), cfg_dict) for i, (_, row) in enumerate(df.iterrows())]
    total = len(tasks)
    per = 1 + int(cfg.num_aug_per_image)
    print(f"Parallel preprocessing: originals={total}, outputs per original={per}, workers={cfg.num_workers}")

    all_rows: List[Dict] = []
    done = 0

    with cf.ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        for out_rows in ex.map(_process_one_row, tasks, chunksize=8):
            if out_rows:
                all_rows.extend(out_rows)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"Processed {done}/{total} originals...")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(cfg.out_csv, index=False)
    print(f"Augmented images saved to: {cfg.out_rgb_dir} and {cfg.out_depth_dir}")
    print(f"Augmented CSV saved to: {cfg.out_csv} (rows={len(out_df)})")


if __name__ == '__main__':
    main()
