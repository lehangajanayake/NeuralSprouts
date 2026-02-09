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

    out_rgb_dir: str = '../../datasets/Training/AugmentedMixup/RGBImages'
    out_depth_dir: str = '../../datasets/Training/AugmentedMixup/DepthImages'
    out_csv: str = '../../datasets/Training/AugmentedMixup/Train_mixup.csv'

    image_size: int = 96
    crop_size: int = 1000

    # how many augmented variants per original (not counting the original)
    num_aug_per_image: int = 30
    max_center_shift: int = 100  # max pixel shift for random pre-crop translations
    mixup_prob: float = 0.4
    mixup_alpha: float = 0.4
    seed: int = 42

    # Parallelism / speed knobs
    # Use processes (not threads) to bypass the GIL for CPU-heavy image decode/resize.
    num_workers: Optional[int] = None  # default computed in main()
    max_items: Optional[int] = None  # optionally limit number of originals processed


_ROW_CACHE: Dict[str, Tuple[Dict[int, Dict], np.ndarray]] = {}


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
    if max_shift <= 0:
        return 0, 0
    pixels = int(rng.randint(0, max_shift + 1))
    if pixels == 0:
        return 0, 0
    angle = rng.uniform(0.0, 2.0 * math.pi)
    dx = int(round(pixels * math.cos(angle)))
    dy = int(round(pixels * math.sin(angle)))
    return dx, dy


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


def _get_row_cache(labels_csv: str) -> Tuple[Dict[int, Dict], np.ndarray]:
    cache = _ROW_CACHE.get(labels_csv)
    if cache is not None:
        return cache

    df = pd.read_csv(labels_csv)
    if 'image_id' in df.columns and 'id' not in df.columns:
        df = df.rename(columns={'image_id': 'id'})
    row_map: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        try:
            row_map[int(row['id'])] = row.to_dict()
        except Exception:
            continue
    ids = np.array(list(row_map.keys()), dtype=np.int64)
    cache = (row_map, ids)
    _ROW_CACHE[labels_csv] = cache
    return cache


def _mix_images(img_a: Image.Image, img_b: Image.Image, lam: float) -> Image.Image:
    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    arr_mix = lam * arr_a + (1.0 - lam) * arr_b
    arr_mix = np.clip(arr_mix, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr_mix)


def maybe_apply_mixup(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    rng: np.random.RandomState,
    orig_row: Dict,
    orig_id: int,
) -> Tuple[Image.Image, Image.Image, Optional[float], Optional[int], Optional[float]]:
    if cfg.mixup_prob <= 0.0 or cfg.mixup_alpha <= 0.0:
        return rgb, depth, None, None, None
    if rng.rand() >= cfg.mixup_prob:
        return rgb, depth, None, None, None

    row_map, id_pool = _get_row_cache(cfg.labels_csv)
    if len(id_pool) < 2:
        return rgb, depth, None, None, None

    partner_id = orig_id
    for _ in range(5):
        candidate = int(id_pool[rng.randint(0, len(id_pool))])
        if candidate != orig_id:
            partner_id = candidate
            break
    if partner_id == orig_id:
        return rgb, depth, None, None, None

    partner_rgb_path = os.path.join(cfg.train_rgb_dir, f'RGB_{partner_id}.png')
    partner_depth_path = os.path.join(cfg.train_depth_dir, f'Depth_{partner_id}.png')
    if not os.path.exists(partner_rgb_path) or not os.path.exists(partner_depth_path):
        return rgb, depth, None, None, None

    try:
        partner_rgb = Image.open(partner_rgb_path).convert('RGB')
        partner_depth = Image.open(partner_depth_path).convert('L')
    except Exception:
        return rgb, depth, None, None, None

    partner_rgb, partner_depth = apply_aug(partner_rgb, partner_depth, rng)
    partner_shift = random_center_shift(rng, int(cfg.max_center_shift))
    partner_rgb, partner_depth = preprocess_one(partner_rgb, partner_depth, cfg, shift=partner_shift)

    lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
    rgb_mix = _mix_images(rgb, partner_rgb, lam)
    depth_mix = _mix_images(depth, partner_depth, lam)

    try:
        label_a = float(orig_row['DryWeightShoot'])
        partner_row = row_map.get(partner_id)
        if partner_row is None:
            return rgb, depth, None, None, None
        label_b = float(partner_row['DryWeightShoot'])
    except Exception:
        return rgb, depth, None, None, None

    mixed_label = lam * label_a + (1.0 - lam) * label_b
    return rgb_mix, depth_mix, mixed_label, partner_id, lam


def preprocess_one(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    shift: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Image.Image]:
    if shift is not None and shift != (0, 0):
        rgb = shifted_center_crop(rgb, cfg.crop_size, shift)
        depth = shifted_center_crop(depth, cfg.crop_size, shift)
    else:
        rgb = center_crop(rgb, cfg.crop_size)
        depth = center_crop(depth, cfg.crop_size)

    rgb = rgb.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    depth = depth.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
    return rgb, depth


def _process_one_row(args) -> List[Dict]:
    """Worker function: process one original row and return CSV rows for (original + aug variants).

    Notes:
    - Must be top-level for Windows multiprocessing.
    - Uses deterministic output ids computed from the row_index, so results are stable
      regardless of worker completion order.
    """

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

    # Deterministic id allocation: each original produces (1 + num_aug_per_image) outputs
    per = 1 + int(cfg.num_aug_per_image)
    base_out_id = int(row_index) * per + 1

    out_rows: List[Dict] = []

    # original (no aug)
    rgb, depth = preprocess_one(rgb0, depth0, cfg)
    rgb.save(os.path.join(cfg.out_rgb_dir, f'RGB_{base_out_id}.png'))
    depth.save(os.path.join(cfg.out_depth_dir, f'Depth_{base_out_id}.png'))
    r0 = dict(row_dict)
    r0['id'] = base_out_id
    r0.pop('mixup_partner_id', None)
    r0.pop('mixup_lambda', None)
    out_rows.append(r0)

    # augmented variants
    for k in range(int(cfg.num_aug_per_image)):
        out_id = base_out_id + 1 + k

        rng = np.random.RandomState(int(cfg.seed) + orig_id * 100 + k)
        rgb_aug, depth_aug = apply_aug(rgb0, depth0, rng)
        shift = random_center_shift(rng, int(cfg.max_center_shift))
        rgb_aug, depth_aug = preprocess_one(rgb_aug, depth_aug, cfg, shift=shift)

        rgb_aug, depth_aug, mix_label, partner_id, lam = maybe_apply_mixup(
            rgb_aug,
            depth_aug,
            cfg,
            rng,
            row_dict,
            orig_id,
        )

        rgb_aug.save(os.path.join(cfg.out_rgb_dir, f'RGB_{out_id}.png'))
        depth_aug.save(os.path.join(cfg.out_depth_dir, f'Depth_{out_id}.png'))

        rk = dict(row_dict)
        rk['id'] = out_id
        if mix_label is not None:
            rk['DryWeightShoot'] = mix_label
            rk['mixup_partner_id'] = partner_id
            rk['mixup_lambda'] = lam
        else:
            rk.pop('mixup_partner_id', None)
            rk.pop('mixup_lambda', None)
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

    # Optionally limit number of originals
    if cfg.max_items is not None:
        df = df.iloc[: int(cfg.max_items)].reset_index(drop=True)

    # Compute default worker count
    if cfg.num_workers is None:
        cpu = os.cpu_count() or 1
        # leave 1 core for OS/UI; minimum 1 worker
        cfg.num_workers = max(1, cpu - 1)
    cfg.num_workers = max(1, int(cfg.num_workers))

    # Send a plain dict to workers (dataclasses aren't always pickle-friendly between processes)
    cfg_dict = {
        'train_rgb_dir': cfg.train_rgb_dir,
        'train_depth_dir': cfg.train_depth_dir,
        'labels_csv': cfg.labels_csv,
        'out_rgb_dir': cfg.out_rgb_dir,
        'out_depth_dir': cfg.out_depth_dir,
        'out_csv': cfg.out_csv,
        'image_size': cfg.image_size,
        'crop_size': cfg.crop_size,
        'num_aug_per_image': cfg.num_aug_per_image,
        'max_center_shift': cfg.max_center_shift,
        'seed': cfg.seed,
        'num_workers': cfg.num_workers,
        'max_items': cfg.max_items,
        'mixup_prob': cfg.mixup_prob,
        'mixup_alpha': cfg.mixup_alpha,
    }

    tasks = [(i, row.to_dict(), cfg_dict) for i, (_, row) in enumerate(df.iterrows())]
    total = len(tasks)
    per = 1 + int(cfg.num_aug_per_image)
    print(f"Parallel preprocessing: originals={total}, outputs per original={per}, workers={cfg.num_workers}")

    all_rows: List[Dict] = []
    done = 0

    # Use processes: fastest for PIL decode/resize
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
