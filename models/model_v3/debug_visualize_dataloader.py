import argparse
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataloader import SimplePlantDataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(img_chw: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW torch tensor into a displayable HWC uint8 image."""
    x = img_chw.detach().cpu().float().numpy()
    if x.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got {x.shape}")
    x = np.transpose(x, (1, 2, 0))
    x = (x * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def build_dataset(rgb_dir: str, csv_path: str, image_size: int, num_views: int, cache_seed: int, build_cache: bool, max_items: Optional[int]):
    ds = SimplePlantDataset(rgb_dir, csv_path, image_size=image_size)
    ds.num_views = num_views
    ds.cache_seed = cache_seed

    if build_cache:
        ds.enable_cache = True
        ds.build_cache(max_items=max_items)
    else:
        ds.enable_cache = False

    return ds


def main():
    parser = argparse.ArgumentParser(description='Debug/visualize model_v3 dataloader output (including cached views).')
    parser.add_argument('--csv', default='../../datasets/Training/Train.csv', help='CSV file (Train.csv or Test.csv).')
    parser.add_argument('--rgb', default='../../datasets/Training/RGBImages/', help='RGB images directory.')
    parser.add_argument('--image-size', type=int, default=224)

    parser.add_argument('--num-views', type=int, default=4, help='K views per image (Option B expands dataset to N*K).')
    parser.add_argument('--cache-seed', type=int, default=42)
    parser.add_argument('--build-cache', action='store_true', help='Build the cached views before visualizing.')
    parser.add_argument('--max-items', type=int, default=None, help='Only cache first N base items (saves RAM).')

    parser.add_argument('--index', type=int, default=0, help='Global dataset index to visualize (0..len(ds)-1).')
    parser.add_argument('--grid', type=int, default=12, help='If set >1, visualize a grid of the first N samples starting at --index.')

    parser.add_argument('--out', default='v3_dataloader_debug.png', help='Output PNG path.')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f'CSV not found: {args.csv}')
    if not os.path.isdir(args.rgb):
        raise NotADirectoryError(f'RGB directory not found: {args.rgb}')

    ds = build_dataset(
        rgb_dir=args.rgb,
        csv_path=args.csv,
        image_size=args.image_size,
        num_views=args.num_views,
        cache_seed=args.cache_seed,
        build_cache=args.build_cache,
        max_items=args.max_items,
    )

    n = len(ds)
    if n == 0:
        raise RuntimeError('Dataset is empty after filtering for available images.')

    start = max(0, min(args.index, n - 1))
    count = max(1, min(args.grid, n - start))

    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(count):
        idx = start + i
        x, y = ds[idx]
        img = denormalize(x)

        base_len = len(ds.df)
        if ds.num_views > 1:
            base_idx = idx % base_len
            view_idx = idx // base_len
        else:
            base_idx, view_idx = idx, 0

        img_id = int(ds.df.iloc[base_idx]['id'])
        cached = bool(ds.enable_cache and (base_idx, view_idx) in ds._cache)

        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f'global={idx} base={base_idx} view={view_idx}\n'
            f'id={img_id} y={float(y):.3f} cached={cached}'
        )

    for j in range(count, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'model_v3 dataloader debug | views={ds.num_views} | cache={ds.enable_cache} | seed={ds.cache_seed} | size={args.image_size}',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
