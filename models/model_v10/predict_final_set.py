"""Predict dry-weight for the *Final* image set (no ground-truth CSV needed).

Scans ``datasets/Final/RGBImages`` and ``datasets/Final/DepthImages`` for
paired ``RGB_<id>.png`` / ``Depth_<id>.png`` files, runs them through the
trained LettuceSAMFusionNet checkpoint, and writes a two-column CSV:

    image_id, DryWeightShoot
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import LettuceSAMFusionNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANDATORY_CROP = 1000
RGB_PATTERN = re.compile(r"^RGB_(\d+)\.png$", re.IGNORECASE)
DEPTH_PATTERN = re.compile(r"^Depth_(\d+)\.png$", re.IGNORECASE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors dataloader.py exactly)
# ---------------------------------------------------------------------------

def _center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h, MANDATORY_CROP)
    left = (w - side) / 2
    top = (h - side) / 2
    return img.crop((left, top, left + side, top + side))


# ---------------------------------------------------------------------------
# Dataset — scans folders, no CSV required
# ---------------------------------------------------------------------------

class FinalFolderDataset(Dataset):
    """Loads RGB + Depth pairs for inference from raw image directories."""

    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        ids: Sequence[int],
        *,
        image_size: int = 128,
        center_crop: bool = True,
    ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.ids = sorted(ids)
        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_id = int(self.ids[index])
        rgb = Image.open(self.rgb_dir / f"RGB_{image_id}.png").convert("RGB")
        depth = Image.open(self.depth_dir / f"Depth_{image_id}.png").convert("L")

        if self.center_crop:
            rgb = _center_crop(rgb)
            depth = _center_crop(depth)

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
            "id": torch.tensor(image_id, dtype=torch.long),
            "rgb": rgb_t,
            "rgbd": rgbd_t,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_ids(directory: Path, pattern: re.Pattern[str]) -> List[int]:
    """Extract integer IDs from filenames matching *pattern*."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    ids: List[int] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise ValueError(f"No files matching {pattern.pattern} in {directory}")
    return ids


# ---------------------------------------------------------------------------
# Main prediction routine
# ---------------------------------------------------------------------------

def predict_final_set(
    rgb_dir: Path,
    depth_dir: Path,
    model_path: Path,
    output_csv: Path,
    *,
    expected_pairs: int = 81,
    batch_size: int = 64,
    image_size: int = 128,
) -> Path:
    # ---- discover paired IDs -------------------------------------------
    rgb_ids = set(_collect_ids(rgb_dir, RGB_PATTERN))
    depth_ids = set(_collect_ids(depth_dir, DEPTH_PATTERN))
    paired_ids = sorted(rgb_ids & depth_ids)

    if not paired_ids:
        raise ValueError(
            f"No overlapping RGB/Depth IDs.\n"
            f"  RGB dir  ({rgb_dir}): {len(rgb_ids)} files\n"
            f"  Depth dir ({depth_dir}): {len(depth_ids)} files"
        )

    if expected_pairs > 0 and len(paired_ids) != expected_pairs:
        print(
            f"[warn] Expected {expected_pairs} pairs but found {len(paired_ids)}. "
            "Continuing with what's available."
        )

    print(f"Found {len(paired_ids)} paired images  (IDs {paired_ids[0]}–{paired_ids[-1]})")

    # ---- dataset / loader ----------------------------------------------
    dataset = FinalFolderDataset(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        ids=paired_ids,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- load model ----------------------------------------------------
    model = LettuceSAMFusionNet.from_checkpoint(str(model_path), device=DEVICE)
    model.eval()
    print(f"Loaded checkpoint: {model_path}")

    # ---- inference -----------------------------------------------------
    all_ids: List[int] = []
    all_preds: List[float] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(DEVICE, non_blocking=True)
            rgbd = batch["rgbd"].to(DEVICE, non_blocking=True)
            ids = batch["id"].cpu().numpy().tolist()
            preds = model.predict_dry_weight(rgb, rgbd).cpu().numpy().flatten().tolist()
            all_ids.extend(ids)
            all_preds.extend(preds)

    # ---- save CSV (2 columns only) -------------------------------------
    df = pd.DataFrame({"image_id": all_ids, "DryWeightShoot": all_preds})
    df.sort_values("image_id", inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions → {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-dir", type=Path, default=Path("../../datasets/Final/RGBImages"))
    p.add_argument("--depth-dir", type=Path, default=Path("../../datasets/Final/DepthImages"))
    p.add_argument("--model-path", type=Path, default=Path("best_model_v10.pth"))
    p.add_argument("--output-csv", type=Path, default=Path("Final_Submission_v10.csv"))
    p.add_argument("--expected-count", type=int, default=81)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    predict_final_set(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        expected_pairs=args.expected_count if args.expected_count > 0 else 0,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
