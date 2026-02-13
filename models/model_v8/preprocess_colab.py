"""Google Colab-friendly GPU preprocessor for the lettuce dataset.

This script mirrors preprocess.py but keeps all image tensors on a chosen
Torch device (cuda if available) so augmentations such as flips, rotations,
color jitter, and depth noise run on the GPU. It works well inside Colab: put
this file in your repo, upload the datasets folder to /content, and run
`python preprocess_colab.py --help` for options.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

Tensor = torch.Tensor


@dataclass
class ColabPreprocessConfig:
    train_rgb_dir: str = '../../datasets/Training/RGBImages'
    train_depth_dir: str = '../../datasets/Training/DepthImages'
    labels_csv: str = '../../datasets/Training/Train.csv'

    out_rgb_dir: str = '../../datasets/Training/Augmented_v8/RGBImages'
    out_depth_dir: str = '../../datasets/Training/Augmented_v8/DepthImages'
    out_csv: str = '../../datasets/Training/Augmented_v8/Train_aug.csv'

    image_size: int = 96
    crop_size: int = 1000
    randomize_crop: bool = False
    num_aug_per_image: int = 45
    max_center_shift: int = 75
    seed: int = 42

    depth_noise_std: float = 0.03
    depth_noise_prob: float = 0.7
    color_jitter_prob: float = 0.8

    num_workers: int = 16  # GPU loop, so keep single process
    max_items: Optional[int] = None
    device: str = 'cuda'
    flush_size: int = 64
    save_workers: int = 2


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _load_image(path: str, mode: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert(mode)


def _pil_to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def _tensor_to_pil(tensor: Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr, mode='RGB')


class AsyncImageWriter:
    """Batches GPU tensors and writes PNGs on background threads."""

    def __init__(self, cfg: ColabPreprocessConfig) -> None:
        self.rgb_dir = cfg.out_rgb_dir
        self.depth_dir = cfg.out_depth_dir
        self.flush_size = max(1, int(cfg.flush_size))
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(cfg.save_workers)))
        self._rgb_buf: List[Tensor] = []
        self._depth_buf: List[Tensor] = []
        self._ids: List[int] = []
        self._futures: List = []
        self._closed = False

    def add(self, rgb: Tensor, depth: Tensor, out_id: int) -> None:
        if self._closed:
            raise RuntimeError('AsyncImageWriter already closed')
        rgb_u8 = (rgb.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        depth_u8 = (depth.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        self._rgb_buf.append(rgb_u8)
        self._depth_buf.append(depth_u8)
        self._ids.append(out_id)
        if len(self._ids) >= self.flush_size:
            self._flush()

    def _flush(self) -> None:
        if not self._ids:
            return
        rgb_tensor = torch.stack(self._rgb_buf, dim=0).to('cpu', non_blocking=True)
        depth_tensor = torch.stack(self._depth_buf, dim=0).to('cpu', non_blocking=True)
        rgb_np = rgb_tensor.permute(0, 2, 3, 1).numpy()
        depth_np = depth_tensor.squeeze(1).numpy()
        ids = self._ids[:]
        future = self.executor.submit(self._write_batch, rgb_np, depth_np, ids, self.rgb_dir, self.depth_dir)
        self._futures.append(future)
        self._rgb_buf.clear()
        self._depth_buf.clear()
        self._ids.clear()

    @staticmethod
    def _write_batch(rgb_np: np.ndarray, depth_np: np.ndarray, ids: List[int], rgb_dir: str, depth_dir: str) -> None:
        for rgb_arr, depth_arr, out_id in zip(rgb_np, depth_np, ids):
            rgb_img = Image.fromarray(rgb_arr.astype(np.uint8), mode='RGB')
            depth_img = Image.fromarray(depth_arr.astype(np.uint8), mode='L')
            rgb_img.save(os.path.join(rgb_dir, f'RGB_{out_id}.png'))
            depth_img.save(os.path.join(depth_dir, f'Depth_{out_id}.png'))

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        for fut in self._futures:
            fut.result()
        self.executor.shutdown(wait=True)
        self._closed = True


def _resize_tensor(img: Tensor, size: int) -> Tensor:
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
    return img.squeeze(0)


def _crop_tensor(img: Tensor, crop_size: int, shift: Tuple[int, int]) -> Tensor:
    _, h, w = img.shape
    if h < crop_size or w < crop_size:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        return img[:, top : top + side, left : left + side]
    dx, dy = shift
    top = int((h - crop_size) / 2 + dy)
    left = int((w - crop_size) / 2 + dx)
    top = max(0, min(top, h - crop_size))
    left = max(0, min(left, w - crop_size))
    return img[:, top : top + crop_size, left : left + crop_size]


def _random_center_shift(rng: np.random.RandomState, max_shift: int) -> Tuple[int, int]:
    if max_shift <= 0:
        return 0, 0

    def _axis() -> int:
        mag = int(rng.randint(0, max_shift + 1))
        if mag == 0:
            return 0
        return mag if rng.rand() < 0.5 else -mag

    return _axis(), _axis()


def _random_crop_size(rgb_hw: Tuple[int, int], depth_hw: Tuple[int, int], cfg: ColabPreprocessConfig, rng: np.random.RandomState) -> int:
    max_crop = min(rgb_hw + depth_hw)
    min_crop = min(int(cfg.crop_size), max_crop)
    if not cfg.randomize_crop or min_crop >= max_crop:
        return min_crop
    return int(rng.randint(min_crop, max_crop + 1))


def _maybe_depth_noise(depth: Tensor, rng: np.random.RandomState, std: float, prob: float, device: torch.device) -> Tensor:
    if std <= 0.0 or prob <= 0.0 or rng.rand() > prob:
        return depth
    noise = torch.from_numpy(rng.normal(0.0, float(std), size=depth.shape)).to(device=device, dtype=depth.dtype)
    depth = (depth + noise).clamp(0.0, 1.0)
    return depth


def _color_jitter(rgb: Tensor, rng: np.random.RandomState) -> Tensor:
    def _rand_factor(strength: float) -> float:
        return float(rng.uniform(1.0 - strength, 1.0 + strength))

    brightness = _rand_factor(0.3)
    contrast = _rand_factor(0.3)
    saturation = _rand_factor(0.3)
    hue = float(rng.uniform(-0.03, 0.03))

    rgb = rgb * brightness
    mean = rgb.mean(dim=(1, 2), keepdim=True)
    rgb = (rgb - mean) * contrast + mean

    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    rgb = torch.stack([
        (r - gray) * saturation + gray,
        (g - gray) * saturation + gray,
        (b - gray) * saturation + gray,
    ])

    hue_tensor = torch.tensor(hue, dtype=rgb.dtype, device=rgb.device)
    cos_h = torch.cos(hue_tensor)
    sin_h = torch.sin(hue_tensor)
    rot = torch.eye(3, dtype=rgb.dtype, device=rgb.device)
    rot[0, 0] = cos_h
    rot[0, 1] = -sin_h
    rot[1, 0] = sin_h
    rot[1, 1] = cos_h
    rgb = torch.matmul(rot, rgb.view(3, -1)).view_as(rgb)
    return rgb.clamp(0.0, 1.0)


def _apply_geom_augs(rgb: Tensor, depth: Tensor, rng: np.random.RandomState) -> Tuple[Tensor, Tensor]:
    if rng.rand() < 0.5:
        rgb = torch.flip(rgb, dims=[2])
        depth = torch.flip(depth, dims=[2])
    if rng.rand() < 0.5:
        rgb = torch.flip(rgb, dims=[1])
        depth = torch.flip(depth, dims=[1])
    k = int(rng.randint(0, 4))
    if k:
        rgb = torch.rot90(rgb, k, dims=(1, 2))
        depth = torch.rot90(depth, k, dims=(1, 2))
    return rgb, depth


def build_augmented_samples(row: Dict, idx: int, cfg: ColabPreprocessConfig, device: torch.device, writer: AsyncImageWriter) -> List[Dict]:
    orig_id = int(row['id'])
    rgb_path = os.path.join(cfg.train_rgb_dir, f'RGB_{orig_id}.png')
    depth_path = os.path.join(cfg.train_depth_dir, f'Depth_{orig_id}.png')
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        return []

    rgb0 = _pil_to_tensor(_load_image(rgb_path, 'RGB')).to(device)
    depth0 = _pil_to_tensor(_load_image(depth_path, 'L')).to(device)

    rng = _rng(cfg.seed + orig_id)
    per = 1 + int(cfg.num_aug_per_image)
    base_out_id = idx * per + 1
    outputs: List[Dict] = []

    def _prep(sample_rgb: Tensor, sample_depth: Tensor, local_rng: np.random.RandomState, out_id: int) -> None:
        shift = _random_center_shift(local_rng, cfg.max_center_shift)
        crop_side = _random_crop_size((rgb0.shape[1], rgb0.shape[2]), (depth0.shape[1], depth0.shape[2]), cfg, local_rng)
        rgb_c = _resize_tensor(_crop_tensor(sample_rgb, crop_side, shift), cfg.image_size)
        depth_c = _resize_tensor(_crop_tensor(sample_depth, crop_side, shift), cfg.image_size)
        writer.add(rgb_c, depth_c, out_id)
        out_row = dict(row)
        out_row['id'] = out_id
        out_row['original_id'] = orig_id
        outputs.append(out_row)

    _prep(rgb0, depth0, rng, base_out_id)

    for aug_idx in range(cfg.num_aug_per_image):
        aug_rng = _rng(cfg.seed + orig_id * 100 + aug_idx)
        rgb_aug, depth_aug = _apply_geom_augs(rgb0.clone(), depth0.clone(), aug_rng)
        if aug_rng.rand() < cfg.color_jitter_prob:
            rgb_aug = _color_jitter(rgb_aug, aug_rng)
        depth_aug = _maybe_depth_noise(depth_aug, aug_rng, cfg.depth_noise_std, cfg.depth_noise_prob, device)
        out_id = base_out_id + aug_idx + 1
        _prep(rgb_aug, depth_aug, aug_rng, out_id)

    return outputs


def preprocess(cfg: ColabPreprocessConfig) -> None:
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

    device = torch.device(cfg.device if torch.cuda.is_available() or 'cuda' not in cfg.device else 'cpu')
    print(f"Using device: {device}")

    writer = AsyncImageWriter(cfg)
    all_rows: List[Dict] = []
    try:
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc='Augmenting', unit='img')):
            rows = build_augmented_samples(row.to_dict(), idx, cfg, device, writer)
            if rows:
                all_rows.extend(rows)
    finally:
        writer.close()

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(cfg.out_csv, index=False)
    print(f"Augmented set written to {cfg.out_csv} (rows={len(out_df)})")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GPU-friendly preprocessing pipeline for Colab')
    parser.add_argument('--train-rgb-dir', default=ColabPreprocessConfig.train_rgb_dir)
    parser.add_argument('--train-depth-dir', default=ColabPreprocessConfig.train_depth_dir)
    parser.add_argument('--labels-csv', default=ColabPreprocessConfig.labels_csv)
    parser.add_argument('--out-rgb-dir', default=ColabPreprocessConfig.out_rgb_dir)
    parser.add_argument('--out-depth-dir', default=ColabPreprocessConfig.out_depth_dir)
    parser.add_argument('--out-csv', default=ColabPreprocessConfig.out_csv)
    parser.add_argument('--image-size', type=int, default=ColabPreprocessConfig.image_size)
    parser.add_argument('--crop-size', type=int, default=ColabPreprocessConfig.crop_size)
    parser.add_argument('--randomize-crop', action='store_true', default=ColabPreprocessConfig.randomize_crop)
    parser.add_argument('--num-aug-per-image', type=int, default=ColabPreprocessConfig.num_aug_per_image)
    parser.add_argument('--max-center-shift', type=int, default=ColabPreprocessConfig.max_center_shift)
    parser.add_argument('--seed', type=int, default=ColabPreprocessConfig.seed)
    parser.add_argument('--depth-noise-std', type=float, default=ColabPreprocessConfig.depth_noise_std)
    parser.add_argument('--depth-noise-prob', type=float, default=ColabPreprocessConfig.depth_noise_prob)
    parser.add_argument('--color-jitter-prob', type=float, default=ColabPreprocessConfig.color_jitter_prob)
    parser.add_argument('--max-items', type=int, default=None)
    parser.add_argument('--device', default=ColabPreprocessConfig.device)
    parser.add_argument('--flush-size', type=int, default=ColabPreprocessConfig.flush_size)
    parser.add_argument('--save-workers', type=int, default=ColabPreprocessConfig.save_workers)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    cfg = ColabPreprocessConfig(
        train_rgb_dir=args.train_rgb_dir,
        train_depth_dir=args.train_depth_dir,
        labels_csv=args.labels_csv,
        out_rgb_dir=args.out_rgb_dir,
        out_depth_dir=args.out_depth_dir,
        out_csv=args.out_csv,
        image_size=args.image_size,
        crop_size=args.crop_size,
        randomize_crop=args.randomize_crop,
        num_aug_per_image=args.num_aug_per_image,
        max_center_shift=args.max_center_shift,
        seed=args.seed,
        depth_noise_std=args.depth_noise_std,
        depth_noise_prob=args.depth_noise_prob,
        color_jitter_prob=args.color_jitter_prob,
        max_items=args.max_items,
        device=args.device,
        flush_size=args.flush_size,
        save_workers=args.save_workers,
    )
    preprocess(cfg)
