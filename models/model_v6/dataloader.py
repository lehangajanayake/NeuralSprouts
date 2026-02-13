"""Dual-branch dataloader that loads paired RGB and Depth PNGs."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


def _resolve_column(columns, preferred: Optional[str], fallbacks: List[str]) -> str:
    normalized = {col.lower(): col for col in columns}
    candidates = [preferred] if preferred else []
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in columns:
            return candidate
        lowered = candidate.lower()
        if lowered in normalized:
            return normalized[lowered]
    raise KeyError(f"Could not find any of the columns: {candidates}")


def _candidate_paths(root: Path, prefixes: List[str], image_id: str) -> List[Path]:
    candidates: List[Path] = []
    extensions = [".png", ".jpg", ".jpeg"]
    for prefix in prefixes:
        for ext in extensions:
            candidates.append(root / f"{prefix}{image_id}{ext}")
    for ext in extensions:
        candidates.append(root / f"{image_id}{ext}")
    return candidates


def _center_crop_and_resize(img: Image.Image,
                            crop_size: Optional[int],
                            resize_size: Optional[int],
                            is_depth: bool = False) -> Image.Image:
    if crop_size:
        w, h = img.size
        side = min(max(crop_size, 1), w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        img = img.crop((left, top, left + side, top + side))
    if resize_size:
        resample = Image.NEAREST if is_depth else Image.BILINEAR
        img = img.resize((resize_size, resize_size), resample)
    return img


def _load_rgb_depth(rgb_path: Path,
                    depth_path: Path,
                    crop_size: Optional[int],
                    resize_size: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(str(rgb_path)).convert("RGB")
    depth = Image.open(str(depth_path)).convert("L")
    rgb = _center_crop_and_resize(rgb, crop_size, resize_size)
    depth = _center_crop_and_resize(depth, crop_size, resize_size, is_depth=True)
    return np.array(rgb), np.array(depth)


class RGBDepthDataset(Dataset):
    """Dataset that returns RGB tensors, RGBD tensors, targets, and image ids."""

    def __init__(self,
                 csv_file: str,
                 rgb_dir: str,
                 depth_dir: str,
                 image_id_col: str = "id",
                 target_col: str = "DryWeightShoot",
                 include_target: bool = True,
                 crop_size: Optional[int] = None,
                 resize_size: Optional[int] = None,
                 enable_cache: Optional[bool] = None):
        self.data = pd.read_csv(csv_file)
        self.data.columns = [col.strip() for col in self.data.columns]
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.image_id_col = _resolve_column(self.data.columns, image_id_col, ["image_id", "ID"])

        self.include_target = include_target and target_col in self.data.columns
        self.target_col = None
        if self.include_target:
            self.target_col = _resolve_column(
                self.data.columns,
                target_col,
                ["DryWeightShoot", "dry_weight", "DryWeight", "target"]
            )

        self.crop_size = crop_size if crop_size is not None else Config.CENTER_CROP_SIZE
        self.resize_size = resize_size if resize_size is not None else Config.RESIZE_SIZE
        self.enable_cache = Config.ENABLE_DATASET_CACHE if enable_cache is None else enable_cache

        self.data[self.image_id_col] = self.data[self.image_id_col].astype(str)
        self._path_cache: Dict[str, Tuple[Path, Path]] = {}
        self._tensor_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._filter_available_rows()

    def _filter_available_rows(self) -> None:
        mask: List[bool] = []
        missing = 0
        for _, row in self.data.iterrows():
            image_id = row[self.image_id_col]
            rgb_path = self._resolve_rgb_path(image_id)
            depth_path = self._resolve_depth_path(image_id)
            if rgb_path and depth_path:
                mask.append(True)
                self._path_cache[image_id] = (rgb_path, depth_path)
            else:
                mask.append(False)
                missing += 1
        if missing:
            print(f"Warning: skipped {missing} samples without paired RGB/Depth files.")
        self.data = self.data[mask].reset_index(drop=True)

    def _resolve_rgb_path(self, image_id: str) -> Optional[Path]:
        for path in _candidate_paths(self.rgb_dir, ["RGB_"], image_id):
            if path.exists():
                return path
        return None

    def _resolve_depth_path(self, image_id: str) -> Optional[Path]:
        for path in _candidate_paths(self.depth_dir, ["Depth_"], image_id):
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.data.iloc[idx]
        image_id = row[self.image_id_col]

        rgb_path, depth_path = self._path_cache.get(image_id, (None, None))
        if rgb_path is None or depth_path is None:
            rgb_path = self._resolve_rgb_path(image_id)
            depth_path = self._resolve_depth_path(image_id)
        if rgb_path is None or depth_path is None:
            raise FileNotFoundError(f"Missing RGB/Depth files for image id {image_id}")

        if self.enable_cache and image_id in self._tensor_cache:
            rgb_tensor, rgbd_tensor = self._tensor_cache[image_id]
        else:
            rgb_np, depth_np = _load_rgb_depth(
                rgb_path,
                depth_path,
                crop_size=self.crop_size,
                resize_size=self.resize_size
            )
            depth_np = np.expand_dims(depth_np, axis=-1)
            rgbd_np = np.concatenate([rgb_np, depth_np], axis=-1)

            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
            rgbd_tensor = torch.from_numpy(rgbd_np).permute(2, 0, 1).float() / 255.0

            if self.enable_cache:
                self._tensor_cache[image_id] = (rgb_tensor, rgbd_tensor)

        if self.include_target and self.target_col is not None:
            target_value = float(row[self.target_col])
        else:
            target_value = 0.0
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        return rgb_tensor, rgbd_tensor, target_tensor, str(image_id)


def create_dataloader(csv_file: str,
                      rgb_dir: str,
                      depth_dir: str,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      persistent_workers: Optional[bool] = None,
                      include_target: bool = True,
                      crop_size: Optional[int] = None,
                      resize_size: Optional[int] = None,
                      enable_cache: Optional[bool] = None) -> DataLoader:
    dataset = RGBDepthDataset(
        csv_file=csv_file,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        include_target=include_target,
        crop_size=crop_size,
        resize_size=resize_size,
        enable_cache=enable_cache
    )
    if persistent_workers is None:
        persistent_workers = Config.PERSISTENT_WORKERS if num_workers > 0 else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )
