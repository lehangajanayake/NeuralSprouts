"""Preprocessing pipeline for model_v10 — writes tensor shards to disk.

Instead of saving augmented images as individual PNGs (slow I/O during
training), this pipeline:

1. Reads the original RGB + Depth PNGs.
2. Applies augmentations (flip, rotate, colour jitter, depth noise,
   random centre-shift cropping) — identically to v8.
3. Packs every ``shard_size`` samples into a single ``.pt`` file containing
   a dict of stacked tensors:  ``{"rgb": ..., "rgbd": ..., "target": ...,
   "id": ..., "original_id": ...}``.

During training the ``ShardDataset`` (see ``dataloader.py``) reads entire
shards at once, which is **dramatically** faster than opening thousands of
small PNG files, especially on spinning disks or network mounts.

Reproducibility
---------------
All random state is derived deterministically from the master ``seed`` plus
the original image id and augmentation index.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except Exception:  # pragma: no cover
    T = None  # type: ignore[assignment]
    TF = None  # type: ignore[assignment]

from _reproducibility import seed_everything

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """All knobs for the preprocessing / augmentation pipeline."""

    # ---- input paths ------------------------------------------------------
    train_rgb_dir: str = "../../datasets/Training/RGBImages"
    train_depth_dir: str = "../../datasets/Training/DepthImages"
    labels_csv: str = "../../datasets/Training/Train.csv"

    # ---- output paths (tensor shards) -------------------------------------
    out_dir: str = "../../datasets/Training/Shards_v10"
    out_csv: str = "../../datasets/Training/Shards_v10/manifest.csv"

    # ---- geometry ---------------------------------------------------------
    image_size: int = 96
    crop_size: int = 1000
    randomize_crop: bool = False
    max_center_shift: int = 0  # disabled — affine translate handles positional jitter

    # ---- augmentation -----------------------------------------------------
    num_aug_per_image: int = 45
    depth_noise_std: float = 0.05
    depth_noise_prob: float = 0.8
    color_jitter_prob: float = 0.9
    gaussian_blur_prob: float = 0.3
    random_affine_prob: float = 0.4
    random_erasing_prob: float = 0.3

    # ---- sharding ---------------------------------------------------------
    shard_size: int = 256  # samples per .pt file

    # ---- reproducibility / performance ------------------------------------
    seed: int = 42
    num_workers: Optional[int] = None
    max_items: Optional[int] = None
    use_gpu: bool = True  # run augmentations on GPU (much faster); set False for CPU-only


# ---------------------------------------------------------------------------
# Deterministic augmentation helpers
# ---------------------------------------------------------------------------


def _center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h, crop_size)
    left = (w - side) / 2
    top = (h - side) / 2
    return img.crop((left, top, left + side, top + side))


def _shifted_center_crop(
    img: Image.Image, crop_size: int, shift: Tuple[int, int]
) -> Image.Image:
    w, h = img.size
    if w < crop_size or h < crop_size:
        return _center_crop(img, crop_size)
    dx, dy = shift
    left = max(0.0, min((w - crop_size) / 2 + dx, w - crop_size))
    top = max(0.0, min((h - crop_size) / 2 + dy, h - crop_size))
    return img.crop((left, top, left + crop_size, top + crop_size))


def _random_center_shift(rng: np.random.RandomState, max_shift: int) -> Tuple[int, int]:
    if max_shift <= 0:
        return 0, 0

    def _axis() -> int:
        mag = int(rng.randint(0, max_shift + 1))
        return (-1 if rng.rand() < 0.5 else 1) * mag if mag else 0

    return _axis(), _axis()


def _random_crop_size(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    rng: np.random.RandomState,
) -> int:
    max_crop = min(min(rgb.size), min(depth.size))
    min_crop = min(cfg.crop_size, max_crop)
    if not cfg.randomize_crop or min_crop >= max_crop:
        return min_crop
    return int(rng.randint(min_crop, max_crop + 1))


def _apply_depth_noise(
    depth: Image.Image, rng: np.random.RandomState, std: float
) -> Image.Image:
    if std <= 0.0:
        return depth
    arr = np.asarray(depth, dtype=np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0.0, std, arr.shape).astype(np.float32), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def _apply_augmentations(
    rgb: Image.Image,
    depth: Image.Image,
    rng: np.random.RandomState,
    cfg: PreprocessConfig,
) -> Tuple[Image.Image, Image.Image]:
    """Geometric + photometric augmentations (deterministic given *rng*)."""
    if T is None or TF is None:
        return rgb, depth

    # --- geometric (applied to both rgb and depth) ---
    if rng.rand() < 0.5:
        rgb, depth = TF.hflip(rgb), TF.hflip(depth)
    if rng.rand() < 0.5:
        rgb, depth = TF.vflip(rgb), TF.vflip(depth)
    k = int(rng.randint(0, 4))
    if k:
        rgb, depth = TF.rotate(rgb, 90 * k), TF.rotate(depth, 90 * k)

    # Random affine (small shear + translate) for pose diversity
    if rng.rand() < cfg.random_affine_prob:
        angle = float(rng.uniform(-15, 15))
        shear = float(rng.uniform(-10, 10))
        tx = float(rng.uniform(-0.05, 0.05))
        ty = float(rng.uniform(-0.05, 0.05))
        w, h = rgb.size
        translate = (int(tx * w), int(ty * h))
        rgb = TF.affine(rgb, angle=angle, translate=translate, scale=1.0, shear=shear)
        depth = TF.affine(depth, angle=angle, translate=translate, scale=1.0, shear=shear)

    # --- photometric (rgb only) ---
    if rng.rand() < max(0.0, min(1.0, cfg.color_jitter_prob)):
        rgb = T.ColorJitter(0.4, 0.4, 0.3, 0.05)(rgb)

    # Gaussian blur
    if rng.rand() < cfg.gaussian_blur_prob:
        radius = float(rng.uniform(0.5, 1.5))
        from PIL import ImageFilter
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=radius))

    # --- depth noise ---
    if cfg.depth_noise_std > 0 and rng.rand() < cfg.depth_noise_prob:
        depth = _apply_depth_noise(depth, rng, cfg.depth_noise_std)
    return rgb, depth


def _crop_and_resize(
    rgb: Image.Image,
    depth: Image.Image,
    cfg: PreprocessConfig,
    rng: np.random.RandomState,
    shift: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Image.Image]:
    crop_side = _random_crop_size(rgb, depth, cfg, rng)
    if shift is None:
        shift = _random_center_shift(rng, cfg.max_center_shift)
    crop_fn = _shifted_center_crop if shift != (0, 0) else lambda img, cs, _s: _center_crop(img, cs)
    rgb = crop_fn(rgb, crop_side, shift)
    depth = crop_fn(depth, crop_side, shift)
    sz = (cfg.image_size, cfg.image_size)
    return rgb.resize(sz, Image.BILINEAR), depth.resize(sz, Image.BILINEAR)


# ---------------------------------------------------------------------------
# Image → tensor conversion
# ---------------------------------------------------------------------------


def _to_tensors(
    rgb: Image.Image, depth: Image.Image
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert PIL images to float32 [C,H,W] tensors in [0,1]."""
    rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
    depth_np = np.asarray(depth, dtype=np.float32) / 255.0
    if depth_np.ndim == 2:
        depth_np = depth_np[..., np.newaxis]
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
    depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
    rgbd_t = torch.cat([rgb_t, depth_t], dim=0)
    return rgb_t, rgbd_t


# ---------------------------------------------------------------------------
# GPU-accelerated augmentation pipeline
# ---------------------------------------------------------------------------


def _pil_to_tensor(img: Image.Image, channels: int = 3) -> torch.Tensor:
    """PIL Image → float32 [C, H, W] tensor in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _center_crop_tensor(t: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Centre-crop a [C, H, W] tensor."""
    _, h, w = t.shape
    side = min(h, w, crop_size)
    top = (h - side) // 2
    left = (w - side) // 2
    return t[:, top : top + side, left : left + side]


def _shifted_crop_tensor(
    t: torch.Tensor, crop_size: int, shift: Tuple[int, int]
) -> torch.Tensor:
    _, h, w = t.shape
    if w < crop_size or h < crop_size:
        return _center_crop_tensor(t, crop_size)
    dx, dy = shift
    left = max(0, min((w - crop_size) // 2 + dx, w - crop_size))
    top = max(0, min((h - crop_size) // 2 + dy, h - crop_size))
    return t[:, top : top + crop_size, left : left + crop_size]


def _gpu_augment_one_original(
    rgb_pil: Image.Image,
    depth_pil: Image.Image,
    orig_id: int,
    target: float,
    cfg: PreprocessConfig,
    device: torch.device,
) -> List[Dict]:
    """Generate original + all augmented copies on the GPU."""
    results: List[Dict] = []

    # Convert to tensors on GPU once
    rgb_full = _pil_to_tensor(rgb_pil, 3).to(device)
    depth_full = _pil_to_tensor(depth_pil, 1).to(device)
    sz = (cfg.image_size, cfg.image_size)

    def _crop_resize(rgb_t: torch.Tensor, depth_t: torch.Tensor,
                     rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor]:
        shift = _random_center_shift(rng, cfg.max_center_shift)
        if shift != (0, 0):
            rgb_t = _shifted_crop_tensor(rgb_t, cfg.crop_size, shift)
            depth_t = _shifted_crop_tensor(depth_t, cfg.crop_size, shift)
        else:
            rgb_t = _center_crop_tensor(rgb_t, cfg.crop_size)
            depth_t = _center_crop_tensor(depth_t, cfg.crop_size)
        rgb_t = torch.nn.functional.interpolate(
            rgb_t.unsqueeze(0), size=sz, mode="bilinear", align_corners=False
        ).squeeze(0)
        depth_t = torch.nn.functional.interpolate(
            depth_t.unsqueeze(0), size=sz, mode="bilinear", align_corners=False
        ).squeeze(0)
        return rgb_t, depth_t

    def _apply_aug_gpu(rgb_t: torch.Tensor, depth_t: torch.Tensor,
                       rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- geometric (both) ---
        if rng.rand() < 0.5:
            rgb_t = rgb_t.flip(-1)
            depth_t = depth_t.flip(-1)
        if rng.rand() < 0.5:
            rgb_t = rgb_t.flip(-2)
            depth_t = depth_t.flip(-2)
        k = int(rng.randint(0, 4))
        if k:
            rgb_t = torch.rot90(rgb_t, k, dims=(-2, -1))
            depth_t = torch.rot90(depth_t, k, dims=(-2, -1))

        # Random affine
        if rng.rand() < cfg.random_affine_prob:
            angle = float(rng.uniform(-15, 15))
            shear = [float(rng.uniform(-10, 10))]
            _, h, w = rgb_t.shape
            tx = float(rng.uniform(-0.05, 0.05))
            ty = float(rng.uniform(-0.05, 0.05))
            translate = [int(tx * w), int(ty * h)]
            rgb_t = TF.affine(rgb_t, angle=angle, translate=translate,
                              scale=1.0, shear=shear)
            depth_t = TF.affine(depth_t, angle=angle, translate=translate,
                                scale=1.0, shear=shear)

        # --- photometric (rgb only) ---
        if rng.rand() < cfg.color_jitter_prob:
            # Apply jitter params deterministically from rng
            brightness = float(rng.uniform(max(0, 1 - 0.4), 1 + 0.4))
            contrast = float(rng.uniform(max(0, 1 - 0.4), 1 + 0.4))
            saturation = float(rng.uniform(max(0, 1 - 0.3), 1 + 0.3))
            hue = float(rng.uniform(-0.05, 0.05))
            rgb_t = TF.adjust_brightness(rgb_t, brightness)
            rgb_t = TF.adjust_contrast(rgb_t, contrast)
            rgb_t = TF.adjust_saturation(rgb_t, saturation)
            rgb_t = TF.adjust_hue(rgb_t, hue)
            rgb_t = rgb_t.clamp(0.0, 1.0)

        # Gaussian blur
        if rng.rand() < cfg.gaussian_blur_prob:
            ks = 3  # kernel size must be odd
            sigma = float(rng.uniform(0.5, 1.5))
            rgb_t = TF.gaussian_blur(rgb_t, kernel_size=ks, sigma=sigma)

        # Depth noise
        if cfg.depth_noise_std > 0 and rng.rand() < cfg.depth_noise_prob:
            noise = torch.randn_like(depth_t) * cfg.depth_noise_std
            depth_t = (depth_t + noise).clamp(0.0, 1.0)

        return rgb_t, depth_t

    # --- original (no photometric aug) ---
    rng = np.random.RandomState(cfg.seed + orig_id)
    rgb_c, depth_c = _crop_resize(rgb_full, depth_full, rng)
    rgbd_c = torch.cat([rgb_c, depth_c], dim=0)
    results.append({
        "rgb": rgb_c.cpu(), "rgbd": rgbd_c.cpu(),
        "target": target, "id": orig_id, "original_id": orig_id,
    })

    # --- augmented copies ---
    for k in range(cfg.num_aug_per_image):
        rng_k = np.random.RandomState(cfg.seed + orig_id * 100 + k)
        rgb_a, depth_a = _apply_aug_gpu(rgb_full.clone(), depth_full.clone(), rng_k)
        rgb_a, depth_a = _crop_resize(rgb_a, depth_a, rng_k)
        # Random erasing
        if cfg.random_erasing_prob > 0 and rng_k.rand() < cfg.random_erasing_prob:
            _, h, w = rgb_a.shape
            eh = int(rng_k.randint(h // 8, h // 3 + 1))
            ew = int(rng_k.randint(w // 8, w // 3 + 1))
            ey = int(rng_k.randint(0, h - eh + 1))
            ex = int(rng_k.randint(0, w - ew + 1))
            rgb_a[:, ey:ey + eh, ex:ex + ew] = 0.0
            depth_a[:, ey:ey + eh, ex:ex + ew] = 0.0
        rgbd_a = torch.cat([rgb_a, depth_a], dim=0)
        results.append({
            "rgb": rgb_a.cpu(), "rgbd": rgbd_a.cpu(),
            "target": target, "id": orig_id, "original_id": orig_id,
        })

    return results


# ---------------------------------------------------------------------------
# Per-original worker (runs in a process-pool — CPU path)
# ---------------------------------------------------------------------------


def _process_one_original(args) -> List[Dict]:
    """Generate original + augmented tensor dicts for one image id."""
    row_dict, cfg_dict = args
    cfg = PreprocessConfig(**cfg_dict)
    orig_id = int(row_dict["id"])
    target = float(row_dict["DryWeightShoot"])

    rgb_path = os.path.join(cfg.train_rgb_dir, f"RGB_{orig_id}.png")
    depth_path = os.path.join(cfg.train_depth_dir, f"Depth_{orig_id}.png")
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        return []

    try:
        rgb0 = Image.open(rgb_path).convert("RGB")
        depth0 = Image.open(depth_path).convert("L")
    except Exception:
        return []

    results: List[Dict] = []

    # --- original (no photometric aug) ---
    rng = np.random.RandomState(cfg.seed + orig_id)
    rgb_c, depth_c = _crop_and_resize(rgb0, depth0, cfg, rng)
    rgb_t, rgbd_t = _to_tensors(rgb_c, depth_c)
    results.append(
        {"rgb": rgb_t, "rgbd": rgbd_t, "target": target, "id": orig_id, "original_id": orig_id}
    )

    # --- augmented copies ---
    for k in range(cfg.num_aug_per_image):
        rng_k = np.random.RandomState(cfg.seed + orig_id * 100 + k)
        rgb_a, depth_a = _apply_augmentations(rgb0, depth0, rng_k, cfg)
        rgb_a, depth_a = _crop_and_resize(rgb_a, depth_a, cfg, rng_k)
        rgb_t, rgbd_t = _to_tensors(rgb_a, depth_a)
        # Random erasing (cutout) — masks a random rectangle with zeros
        if cfg.random_erasing_prob > 0 and rng_k.rand() < cfg.random_erasing_prob:
            _, h, w = rgb_t.shape
            eh = int(rng_k.randint(h // 8, h // 3 + 1))
            ew = int(rng_k.randint(w // 8, w // 3 + 1))
            ey = int(rng_k.randint(0, h - eh + 1))
            ex = int(rng_k.randint(0, w - ew + 1))
            rgb_t[:, ey:ey + eh, ex:ex + ew] = 0.0
            rgbd_t[:, ey:ey + eh, ex:ex + ew] = 0.0
        results.append(
            {"rgb": rgb_t, "rgbd": rgbd_t, "target": target, "id": orig_id, "original_id": orig_id}
        )

    return results


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


def _write_shard(samples: List[Dict], shard_path: str) -> None:
    """Stack a list of sample dicts into a single .pt shard file."""
    shard = {
        "rgb": torch.stack([s["rgb"] for s in samples]),
        "rgbd": torch.stack([s["rgbd"] for s in samples]),
        "target": torch.tensor([s["target"] for s in samples], dtype=torch.float32),
        "id": torch.tensor([s["id"] for s in samples], dtype=torch.long),
        "original_id": torch.tensor([s["original_id"] for s in samples], dtype=torch.long),
    }
    torch.save(shard, shard_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(cfg: Optional[PreprocessConfig] = None) -> None:  # noqa: C901
    cfg = cfg or PreprocessConfig()
    seed_everything(cfg.seed, deterministic=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.labels_csv)
    if "image_id" in df.columns:
        df.rename(columns={"image_id": "id"}, inplace=True)

    required = {"id", "Variety", "DryWeightShoot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")

    if cfg.max_items is not None:
        df = df.iloc[: int(cfg.max_items)].reset_index(drop=True)

    total = len(df)
    per = 1 + cfg.num_aug_per_image

    # Decide GPU vs CPU path
    gpu_available = cfg.use_gpu and torch.cuda.is_available()
    if gpu_available:
        device = torch.device("cuda")
        print(
            f"[preprocess] GPU mode on {torch.cuda.get_device_name(0)}  |  "
            f"originals={total}, outputs/original={per}, shard_size={cfg.shard_size}"
        )
        all_samples: List[Dict] = []
        for done, (_, row) in enumerate(df.iterrows(), 1):
            orig_id = int(row["id"])
            target_val = float(row["DryWeightShoot"])
            rgb_path = os.path.join(cfg.train_rgb_dir, f"RGB_{orig_id}.png")
            depth_path = os.path.join(cfg.train_depth_dir, f"Depth_{orig_id}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
            try:
                rgb0 = Image.open(rgb_path).convert("RGB")
                depth0 = Image.open(depth_path).convert("L")
            except Exception:
                continue
            samples = _gpu_augment_one_original(
                rgb0, depth0, orig_id, target_val, cfg, device,
            )
            all_samples.extend(samples)
            if done % 25 == 0 or done == total:
                print(f"  processed {done}/{total} originals ({len(all_samples)} samples)")
    else:
        # CPU multiprocessing path (original)
        if cfg.num_workers is None:
            cfg.num_workers = max(1, (os.cpu_count() or 1) - 1)
        cfg.num_workers = max(1, int(cfg.num_workers))

        cfg_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
        tasks = [(row.to_dict(), cfg_dict) for _, row in df.iterrows()]
        print(
            f"[preprocess] CPU mode ({cfg.num_workers} workers)  |  "
            f"originals={total}, outputs/original={per}, shard_size={cfg.shard_size}"
        )
        all_samples = []
        done = 0
        with cf.ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            for sample_list in pool.map(_process_one_original, tasks, chunksize=8):
                all_samples.extend(sample_list)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  processed {done}/{total} originals ({len(all_samples)} samples)")

    # Deterministic shuffle then write shards
    rng = np.random.RandomState(cfg.seed)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)

    shard_paths: List[str] = []
    shard_idx = 0
    for start in range(0, len(indices), cfg.shard_size):
        chunk_idx = indices[start : start + cfg.shard_size]
        chunk = [all_samples[int(i)] for i in chunk_idx]
        shard_name = f"shard_{shard_idx:05d}.pt"
        shard_path = os.path.join(cfg.out_dir, shard_name)
        _write_shard(chunk, shard_path)
        shard_paths.append(shard_name)
        shard_idx += 1

    # Write manifest CSV (one row per shard, with counts)
    manifest_rows = []
    for sp in shard_paths:
        full = os.path.join(cfg.out_dir, sp)
        s = torch.load(full, weights_only=True)
        manifest_rows.append({"shard": sp, "num_samples": int(s["target"].shape[0])})
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(cfg.out_csv, index=False)

    print(
        f"[preprocess] wrote {shard_idx} shards ({len(all_samples)} samples) → {cfg.out_dir}"
    )
    print(f"[preprocess] manifest → {cfg.out_csv}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
