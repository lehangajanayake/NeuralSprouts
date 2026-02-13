"""End-to-end script for producing final predictions without a pre-existing CSV.

It scans the provided RGB/Depth directories, ensures paired IDs (default 81),
feeds them through the trained LettuceSAMFusionNet, and saves the submission
CSV containing the required columns: image_id, DryWeightShoot.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader import MANDATORY_CROP
from model import LettuceSAMFusionNet

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RGB_PATTERN = re.compile(r"^RGB_(\d+)\.png$", re.IGNORECASE)
DEPTH_PATTERN = re.compile(r"^Depth_(\d+)\.png$", re.IGNORECASE)


def _center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < MANDATORY_CROP or h < MANDATORY_CROP:
        side = min(w, h)
        left = (w - side) / 2
        top = (h - side) / 2
        return img.crop((left, top, left + side, top + side))
    left = (w - MANDATORY_CROP) / 2
    top = (h - MANDATORY_CROP) / 2
    return img.crop((left, top, left + MANDATORY_CROP, top + MANDATORY_CROP))


class FolderPairDataset(Dataset):
    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        ids: Sequence[int],
        *,
        image_size: int = 96,
        center_crop: bool = True,
    ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.ids = list(ids)
        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        if T is None:
            self.resize = None
        else:
            self.resize = T.Resize((self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_pair(self, image_id: int) -> Tuple[Image.Image, Image.Image]:
        rgb_path = self.rgb_dir / f"RGB_{image_id}.png"
        depth_path = self.depth_dir / f"Depth_{image_id}.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing Depth image: {depth_path}")
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        return rgb, depth

    def __getitem__(self, index: int):
        image_id = int(self.ids[index])
        rgb, depth = self._load_pair(image_id)

        if self.center_crop:
            rgb = _center_crop(rgb)
            depth = _center_crop(depth)
        if self.resize is not None:
            rgb = self.resize(rgb)
            depth = self.resize(depth)
        else:
            rgb = rgb.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            depth = depth.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_np = np.asarray(depth, dtype=np.float32) / 255.0
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]

        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)

        return {
            'id': torch.tensor(image_id, dtype=torch.long),
            'rgb': rgb_t,
            'rgbd': rgbd_t,
        }


def _collect_ids(directory: Path, pattern: re.Pattern[str]) -> List[int]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    ids = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        raise ValueError(f"No files matching {pattern.pattern} in {directory}")
    return ids


def _summarize_ids(ids: Iterable[int], *, max_items: int = 8) -> str:
    sample = sorted(ids)[:max(0, max_items)]
    if not sample:
        return "[]"
    suffix = "" if len(sample) < max_items else " …"
    return f"{sample}{suffix}"


def _raise_no_overlap(rgb_ids: Iterable[int], depth_ids: Iterable[int], rgb_dir: Path, depth_dir: Path) -> None:
    msg = (
        "No overlapping RGB/Depth IDs found.\n"
        f"RGB dir ({rgb_dir}) has {len(set(rgb_ids))} ids, sample {_summarize_ids(rgb_ids)}\n"
        f"Depth dir ({depth_dir}) has {len(set(depth_ids))} ids, sample {_summarize_ids(depth_ids)}\n"
        "Ensure you are pointing both arguments at the same final dataset bundle."
    )
    raise ValueError(msg)


def _infer_branch_widths(state_dict, prefix: str) -> Tuple[int, ...]:
    widths = []
    idx = 0
    while True:
        key = f"{prefix}.features.{idx}.conv3.1.weight"
        tensor = state_dict.get(key)
        if tensor is None:
            break
        widths.append(int(tensor.shape[0]))
        idx += 1
    if not widths:
        raise ValueError(f"Unable to infer widths for {prefix} from checkpoint")
    return tuple(widths)


def predict_final_set(
    rgb_dir: Path,
    depth_dir: Path,
    model_path: Path,
    output_csv: Path,
    *,
    expected_pairs: int = 81,
    batch_size: int = 64,
    image_size: int = 96,
    drop_path_prob: float = 0.1,
) -> Path:
    rgb_ids = set(_collect_ids(rgb_dir, RGB_PATTERN))
    depth_ids = set(_collect_ids(depth_dir, DEPTH_PATTERN))
    paired_ids = sorted(rgb_ids & depth_ids)
    if not paired_ids:
        _raise_no_overlap(rgb_ids, depth_ids, rgb_dir, depth_dir)
    if expected_pairs > 0 and len(paired_ids) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} pairs but found {len(paired_ids)}."
        )

    dataset = FolderPairDataset(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        ids=paired_ids,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=torch.cuda.is_available(),
    )

    state = torch.load(model_path, map_location=DEVICE)
    try:
        rgb_widths = _infer_branch_widths(state, 'rgb_branch')
        rgbd_widths = _infer_branch_widths(state, 'rgbd_branch')
    except ValueError:
        print('[predict-final] fallback to default widths (32, 64, 96, 128).')
        rgb_widths = (32, 64, 96, 128)
        rgbd_widths = (32, 64, 96, 128)

    model = LettuceSAMFusionNet(
        drop_path_prob=drop_path_prob,
        rgb_widths=rgb_widths,
        rgbd_widths=rgbd_widths,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(DEVICE, non_blocking=True)
            rgbd = batch['rgbd'].to(DEVICE, non_blocking=True)
            batch_ids = batch['id'].cpu().numpy().tolist()
            outputs = model.predict_dry_weight(rgb, rgbd)
            preds.extend(outputs.cpu().numpy().tolist())
            ids.extend(batch_ids)

    df = pd.DataFrame({
        'image_id': ids,
        'DryWeightShoot': preds,
    })
    df.sort_values('image_id', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions for {len(df)} images to {output_csv}")
    return output_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-dir', type=Path, required=True, help='Path to RGB images directory (RGB_<id>.png).')
    parser.add_argument('--depth-dir', type=Path, required=True, help='Path to Depth images directory (Depth_<id>.png).')
    parser.add_argument('--model-path', type=Path, default=Path('best_model_v8.pth'), help='Checkpoint to load.')
    parser.add_argument('--output-csv', type=Path, default=Path('Final_Submission.csv'), help='Where to save predictions.')
    parser.add_argument('--expected-count', type=int, default=81, help='Number of paired images expected; set <=0 to disable.')
    parser.add_argument('--batch-size', type=int, default=64, help='Inference batch size.')
    parser.add_argument('--image-size', type=int, default=96, help='Resize resolution for RGB/RGBD inputs.')
    parser.add_argument('--drop-path-prob', type=float, default=0.1, help='Drop path prob (must match training config).')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    expected = args.expected_count if args.expected_count > 0 else 0
    predict_final_set(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        expected_pairs=expected,
        batch_size=args.batch_size,
        image_size=args.image_size,
        drop_path_prob=args.drop_path_prob,
    )


if __name__ == '__main__':
    main()
