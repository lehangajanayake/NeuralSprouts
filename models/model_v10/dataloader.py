"""Data loaders for model_v10 — shard-based training, PNG-based inference.

Training
--------
``ShardDataset`` loads pre-computed ``.pt`` shard files (written by
``preprocess.py``).  Each shard is a dict of stacked tensors so a single
``torch.load`` fetches hundreds of ready-to-use samples — **far** faster
than opening individual PNG files.

Inference / Evaluation
----------------------
``TestPlantDataset`` reads raw PNG files at test time (same as v8) since
the test set is small and does not benefit from sharding.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Training dataset — reads tensor shards
# ---------------------------------------------------------------------------

class ShardDataset(Dataset):
    """Concatenates pre-built ``.pt`` shards into a single flat dataset.

    Parameters
    ----------
    shard_dir:
        Directory containing ``shard_XXXXX.pt`` files (output of
        ``preprocess.py``).
    manifest_csv:
        Optional path to the manifest CSV.  If *None*, all ``.pt`` files in
        *shard_dir* are used (sorted by name).
    blacklist_ids:
        Original image ids to exclude from training.
    preload:
        If *True* (default) every shard is loaded into RAM on construction.
        This uses more memory but avoids repeated disk reads during training.
        For very large datasets, set to *False* to load shards lazily.
    """

    def __init__(
        self,
        shard_dir: str,
        *,
        manifest_csv: Optional[str] = None,
        blacklist_ids: Optional[Sequence[int]] = None,
        preload: bool = True,
    ) -> None:
        self.shard_dir = shard_dir
        self.blacklist = {int(x) for x in blacklist_ids} if blacklist_ids else set()

        # Discover shards
        if manifest_csv and os.path.isfile(manifest_csv):
            mf = pd.read_csv(manifest_csv)
            shard_names: List[str] = mf["shard"].tolist()
        else:
            shard_names = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))

        # Load & concatenate
        rgb_parts: List[torch.Tensor] = []
        rgbd_parts: List[torch.Tensor] = []
        target_parts: List[torch.Tensor] = []
        id_parts: List[torch.Tensor] = []
        orig_id_parts: List[torch.Tensor] = []

        for name in shard_names:
            path = os.path.join(shard_dir, name)
            shard: Dict[str, torch.Tensor] = torch.load(path, weights_only=True)
            rgb_parts.append(shard["rgb"])
            rgbd_parts.append(shard["rgbd"])
            target_parts.append(shard["target"])
            id_parts.append(shard["id"])
            orig_id_parts.append(shard["original_id"])

        self.rgb = torch.cat(rgb_parts, dim=0)
        self.rgbd = torch.cat(rgbd_parts, dim=0)
        self.targets = torch.cat(target_parts, dim=0)
        self.ids = torch.cat(id_parts, dim=0)
        self.original_ids = torch.cat(orig_id_parts, dim=0)

        # Apply blacklist
        if self.blacklist:
            keep = torch.ones(len(self.ids), dtype=torch.bool)
            for bid in self.blacklist:
                keep &= self.original_ids != bid
            idx = torch.where(keep)[0]
            self.rgb = self.rgb[idx]
            self.rgbd = self.rgbd[idx]
            self.targets = self.targets[idx]
            self.ids = self.ids[idx]
            self.original_ids = self.original_ids[idx]

        if not preload:
            # Move to shared memory so DataLoader workers don't duplicate
            self.rgb = self.rgb.share_memory_()
            self.rgbd = self.rgbd.share_memory_()
            self.targets = self.targets.share_memory_()

    def __len__(self) -> int:
        return self.rgb.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rgb": self.rgb[idx],
            "rgbd": self.rgbd[idx],
            "dry_weight": self.targets[idx],
            "id": self.ids[idx],
            "original_id": self.original_ids[idx],
        }

    def get_original_ids_array(self) -> np.ndarray:
        """Return original_id for every sample as a NumPy array (for splitting)."""
        return self.original_ids.numpy().astype(int)


# ---------------------------------------------------------------------------
# Test / eval dataset — reads PNGs directly (test set is small)
# ---------------------------------------------------------------------------

MANDATORY_CROP = 1000


def _center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h, MANDATORY_CROP)
    left = (w - side) / 2
    top = (h - side) / 2
    return img.crop((left, top, left + side, top + side))


class PlantDatasetV10(Dataset):
    """PNG-based dataset for evaluation on the *original* (un-augmented) data.

    Identical interface to v8's ``PlantDatasetV8`` so that ``eval.py`` can use
    the same loop.
    """

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        labels_csv: str,
        *,
        image_size: int = 96,
        center_crop: bool = True,
        blacklist_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        bl = {int(x) for x in blacklist_ids} if blacklist_ids else set()

        df = pd.read_csv(labels_csv)
        if "image_id" in df.columns:
            df.rename(columns={"image_id": "id"}, inplace=True)

        keep: List[Dict] = []
        for _, row in df.iterrows():
            iid = int(row["id"])
            if iid in bl:
                continue
            rp = os.path.join(rgb_dir, f"RGB_{iid}.png")
            dp = os.path.join(depth_dir, f"Depth_{iid}.png")
            if os.path.exists(rp) and os.path.exists(dp):
                keep.append(
                    {"id": iid, "rgb_path": rp, "depth_path": dp, "DryWeightShoot": float(row["DryWeightShoot"])}
                )
        self.df = pd.DataFrame(keep)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        depth = Image.open(row["depth_path"]).convert("L")
        if self.center_crop:
            rgb, depth = _center_crop(rgb), _center_crop(depth)
        sz = (self.image_size, self.image_size)
        rgb = rgb.resize(sz, Image.BILINEAR)
        depth = depth.resize(sz, Image.BILINEAR)
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0
        if depth_np.ndim == 2:
            depth_np = depth_np[..., np.newaxis]
        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)
        return {
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
            "rgb": rgb_t,
            "rgbd": rgbd_t,
            "dry_weight": torch.tensor(float(row["DryWeightShoot"]), dtype=torch.float32),
        }


class TestPlantDataset(Dataset):
    """Loads RGB + Depth pairs for inference (no labels)."""

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        csv_file: str,
        *,
        image_size: int = 96,
        center_crop: bool = True,
        blacklist_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        bl = {int(x) for x in blacklist_ids} if blacklist_ids else set()

        df = pd.read_csv(csv_file)
        if "image_id" in df.columns:
            df.rename(columns={"image_id": "id"}, inplace=True)
        if "id" not in df.columns:
            raise ValueError("Test CSV must contain 'image_id' or 'id'.")

        keep: List[Dict] = []
        for _, row in df.iterrows():
            iid = int(row["id"])
            if iid in bl:
                continue
            rp = os.path.join(rgb_dir, f"RGB_{iid}.png")
            dp = os.path.join(depth_dir, f"Depth_{iid}.png")
            if os.path.exists(rp) and os.path.exists(dp):
                keep.append({"id": iid, "rgb_path": rp, "depth_path": dp})
        self.df = pd.DataFrame(keep)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        depth = Image.open(row["depth_path"]).convert("L")
        if self.center_crop:
            rgb, depth = _center_crop(rgb), _center_crop(depth)
        sz = (self.image_size, self.image_size)
        rgb = rgb.resize(sz, Image.BILINEAR)
        depth = depth.resize(sz, Image.BILINEAR)
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0
        if depth_np.ndim == 2:
            depth_np = depth_np[..., np.newaxis]
        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)
        return {
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
            "rgb": rgb_t,
            "rgbd": rgbd_t,
        }
