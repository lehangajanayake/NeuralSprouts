"""Fast side-by-side viewer for RGB + Depth images.

Goal
----
A lightweight, low-lag viewer to inspect paired RGB/Depth samples quickly.

- Loads matching pairs by plant id: RGB_{id}.png and Depth_{id}.png
- Keyboard navigation (prev/next/jump/random)
- Small in-memory LRU cache to keep navigation snappy
- Depth visualization uses contrast stretching + colormap

Works well on Windows.

Example
-------
python view_rgb_depth_side_by_side.py --split Training

"""

from __future__ import annotations

import argparse
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This viewer requires opencv-python. Install it with: pip install opencv-python"
        ) from e


@dataclass(frozen=True)
class Pair:
    plant_id: int
    rgb_path: str
    depth_path: str


class LRUCache:
    def __init__(self, max_items: int = 256):
        self.max_items = int(max_items)
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        v = self._data.get(key)
        if v is None:
            return None
        self._data.move_to_end(key)
        return v

    def put(self, key: str, value: np.ndarray) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


def _resolve_default_paths(split: str) -> Tuple[str, str, str]:
    # Script sits at models/model_v4; datasets are at ../../datasets
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
    if split.lower() == "training":
        return (
            os.path.join(base, "Training", "Train.csv"),
            os.path.join(base, "Training", "RGBImages"),
            os.path.join(base, "Training", "DepthImages"),
        )
    if split.lower() == "test":
        return (
            os.path.join(base, "Test", "Test.csv"),
            os.path.join(base, "Test", "RGBImages"),
            os.path.join(base, "Test", "DepthImages"),
        )
    raise ValueError("split must be Training or Test")


def _build_pairs(csv_path: str, rgb_dir: str, depth_dir: str) -> List[Pair]:
    df = pd.read_csv(csv_path)
    if "image_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"image_id": "id"})
    if "id" not in df.columns:
        raise ValueError(f"CSV must contain an 'id' or 'image_id' column: {csv_path}")

    pairs: List[Pair] = []
    missing_rgb = 0
    missing_depth = 0

    for pid in df["id"].tolist():
        try:
            pid_int = int(pid)
        except Exception:
            continue

        rgb_path = os.path.join(rgb_dir, f"RGB_{pid_int}.png")
        depth_path = os.path.join(depth_dir, f"Depth_{pid_int}.png")

        ok_rgb = os.path.exists(rgb_path)
        ok_depth = os.path.exists(depth_path)

        if not ok_rgb:
            missing_rgb += 1
        if not ok_depth:
            missing_depth += 1
        if ok_rgb and ok_depth:
            pairs.append(Pair(pid_int, rgb_path, depth_path))

    if not pairs:
        raise FileNotFoundError(
            "No RGB/Depth pairs found. Check paths and filename pattern RGB_{id}.png / Depth_{id}.png"
        )

    print(
        f"Loaded {len(pairs)} pairs | missing RGB: {missing_rgb} | missing Depth: {missing_depth}"
    )
    return pairs


def _read_rgb_bgr(cv2, path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img  # BGR


def _read_depth_u16_or_u8(cv2, path: str) -> np.ndarray:
    # Try to keep original bit depth.
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        # Some depth PNGs might be stored as RGB; convert to gray.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _depth_to_viz(cv2, depth: np.ndarray, colormap: int, clip_percent: float) -> np.ndarray:
    # Contrast stretch using percentiles for better visibility.
    d = depth.astype(np.float32)
    if d.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    lo = np.percentile(d, clip_percent)
    hi = np.percentile(d, 100.0 - clip_percent)
    if hi <= lo:
        lo, hi = float(d.min()), float(d.max())
        if hi <= lo:
            hi = lo + 1.0

    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d8 = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, colormap)


def _fit_to_height(cv2, img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _apply_aug_preview(
    cv2,
    rgb_bgr: np.ndarray,
    depth_raw: np.ndarray,
    seed: int,
    enable_color_jitter: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply *aligned* geometric aug to RGB and Depth for preview.

    - Same flip/rotation for both so they stay registered.
    - Optional light color jitter for RGB only.

    This is a visualization tool (not training-time perfect fidelity).
    """

    rng = random.Random(int(seed))

    # 0/90/180/270 keeps depth looking clean and is fast.
    rot_k = rng.choice([0, 1, 2, 3])
    do_flip = rng.random() < 0.5

    rgb = rgb_bgr
    depth = depth_raw

    if do_flip:
        rgb = cv2.flip(rgb, 1)
        depth = cv2.flip(depth, 1)

    if rot_k:
        # np.rot90 works for HxW and HxWxC
        rgb = np.ascontiguousarray(np.rot90(rgb, k=rot_k))
        depth = np.ascontiguousarray(np.rot90(depth, k=rot_k))

    if enable_color_jitter:
        # Simple, fast HSV jitter.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.int32)
        dh = rng.randint(-5, 5)
        ds = rng.randint(-25, 25)
        dv = rng.randint(-25, 25)
        hsv[..., 0] = (hsv[..., 0] + dh) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return rgb, depth


def _make_canvas(cv2, rgb_bgr: np.ndarray, depth_bgr: np.ndarray, target_height: int) -> np.ndarray:
    rgb = _fit_to_height(cv2, rgb_bgr, target_height)
    dep = _fit_to_height(cv2, depth_bgr, target_height)

    gap = 16
    h = target_height
    w = rgb.shape[1] + gap + dep.shape[1]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, : rgb.shape[1]] = rgb
    canvas[:, rgb.shape[1] + gap : rgb.shape[1] + gap + dep.shape[1]] = dep
    return canvas


def _put_hud(cv2, canvas: np.ndarray, text: str) -> np.ndarray:
    out = canvas.copy()
    cv2.putText(
        out,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "[A/D] prev/next | [W/S] depth clip | [C] colormap | [T] aug | [K] new aug | [J] jump | [R] random | [Q/ESC] quit",
        (10, out.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main():
    cv2 = _require_cv2()

    parser = argparse.ArgumentParser(description="Fast RGB+Depth side-by-side viewer")
    parser.add_argument("--split", choices=["Training", "Test"], default="Training")
    parser.add_argument("--csv", default=None, help="CSV path (overrides --split)")
    parser.add_argument("--rgb", default=None, help="RGBImages directory (overrides --split)")
    parser.add_argument("--depth", default=None, help="DepthImages directory (overrides --split)")
    parser.add_argument("--height", type=int, default=540, help="Display height in pixels")
    parser.add_argument("--cache", type=int, default=256, help="LRU cache size (images)")
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Depth visualization percentile clip (e.g., 1.0 means 1st..99th percentiles)",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Start with augmented preview enabled (toggle with T).",
    )
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=123,
        help="Base seed for augmentation preview. Use K to re-roll per image.",
    )
    parser.add_argument(
        "--aug-color",
        action="store_true",
        help="Add light RGB-only color jitter in augmented preview.",
    )
    args = parser.parse_args()

    default_csv, default_rgb, default_depth = _resolve_default_paths(args.split)
    csv_path = args.csv or default_csv
    rgb_dir = args.rgb or default_rgb
    depth_dir = args.depth or default_depth

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(rgb_dir)
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(depth_dir)

    pairs = _build_pairs(csv_path, rgb_dir, depth_dir)

    rgb_cache = LRUCache(max_items=args.cache)
    depth_cache = LRUCache(max_items=args.cache)

    colormaps = [
        cv2.COLORMAP_VIRIDIS,
        cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_TURBO,
        cv2.COLORMAP_JET,
    ]
    cmap_idx = 0

    i = 0
    depth_clip = float(args.clip)
    aug_enabled = bool(args.aug)
    aug_base_seed = int(args.aug_seed)
    aug_reroll = 0

    win = "RGB | Depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(args.height * 2.2), int(args.height * 1.1))

    while True:
        pair = pairs[i]

        rgb_bgr = rgb_cache.get(pair.rgb_path)
        if rgb_bgr is None:
            rgb_bgr = _read_rgb_bgr(cv2, pair.rgb_path)
            rgb_cache.put(pair.rgb_path, rgb_bgr)

        depth_raw = depth_cache.get(pair.depth_path)
        if depth_raw is None:
            depth_raw = _read_depth_u16_or_u8(cv2, pair.depth_path)
            depth_cache.put(pair.depth_path, depth_raw)

        depth_viz = _depth_to_viz(cv2, depth_raw, colormaps[cmap_idx], clip_percent=depth_clip)

        if aug_enabled:
            # Deterministic per-image unless re-rolled.
            seed = aug_base_seed + pair.plant_id * 10_000 + aug_reroll
            rgb_bgr, depth_aug = _apply_aug_preview(
                cv2,
                rgb_bgr,
                depth_raw,
                seed=seed,
                enable_color_jitter=bool(args.aug_color),
            )
            depth_viz = _depth_to_viz(cv2, depth_aug, colormaps[cmap_idx], clip_percent=depth_clip)

        canvas = _make_canvas(cv2, rgb_bgr, depth_viz, target_height=int(args.height))
        hud = _put_hud(
            cv2,
            canvas,
            text=(
                f"idx {i+1}/{len(pairs)} | id={pair.plant_id} | clip={depth_clip:.1f}% | "
                f"cmap={cmap_idx+1}/{len(colormaps)} | aug={'ON' if aug_enabled else 'OFF'}"
            ),
        )

        cv2.imshow(win, hud)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("d"), ord("D")):
            i = (i + 1) % len(pairs)
            continue
        if key in (ord("a"), ord("A")):
            i = (i - 1) % len(pairs)
            continue
        if key in (ord("r"), ord("R")):
            i = random.randrange(0, len(pairs))
            continue
        if key in (ord("c"), ord("C")):
            cmap_idx = (cmap_idx + 1) % len(colormaps)
            continue
        if key in (ord("t"), ord("T")):
            aug_enabled = not aug_enabled
            continue
        if key in (ord("k"), ord("K")):
            # Change augmentation for the current ID (same ID, different preview params)
            aug_reroll = random.randint(0, 1_000_000)
            continue
        if key in (ord("w"), ord("W")):
            depth_clip = min(10.0, depth_clip + 0.5)
            continue
        if key in (ord("s"), ord("S")):
            depth_clip = max(0.0, depth_clip - 0.5)
            continue
        if key in (ord("j"), ord("J")):
            # Jump by plant id (fast lookup)
            target = input("Enter plant id to jump to: ").strip()
            try:
                pid = int(target)
            except Exception:
                print("Not a valid int")
                continue
            idx_map: Dict[int, int] = {p.plant_id: ix for ix, p in enumerate(pairs)}
            if pid in idx_map:
                i = idx_map[pid]
            else:
                print("ID not found")
            continue

        # Unhandled key -> ignore

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
