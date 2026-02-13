"""Scan augmented RGB/Depth folders for duplicate crops.

The script pairs files named like RGB_<id>.png and Depth_<id>.png, computes a
hash of the combined image contents, and reports IDs that share the same hash.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def _hash_image(path: Path) -> bytes:
    """Return a stable hash of the image pixel data (ignores metadata)."""

    with Image.open(path) as img:
        # Convert to raw bytes so identical pixels map to same hash even if
        # metadata or compression differs.
        data = img.tobytes()
        digest = hashlib.sha256(data).digest()
    return digest


def _hash_pair(rgb_path: Path, depth_path: Path | None) -> str:
    rgb_hash = _hash_image(rgb_path)
    if depth_path is not None and depth_path.exists():
        depth_hash = _hash_image(depth_path)
    else:
        depth_hash = b""
    combined = hashlib.sha256(rgb_hash + depth_hash).hexdigest()
    return combined


def _collect_pairs(rgb_dir: Path, depth_dir: Path | None) -> Dict[int, Tuple[Path, Path | None]]:
    pairs: Dict[int, Tuple[Path, Path | None]] = {}
    for rgb_file in rgb_dir.glob("RGB_*.png"):
        try:
            sample_id = int(rgb_file.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        depth_path = None
        if depth_dir is not None:
            depth_candidate = depth_dir / f"Depth_{sample_id}.png"
            if depth_candidate.exists():
                depth_path = depth_candidate
        pairs[sample_id] = (rgb_file, depth_path)
    return pairs


@dataclass
class DuplicateReport:
    total_pairs: int
    duplicate_groups: Dict[str, List[int]]

    def print(self) -> None:
        print(f"Checked {self.total_pairs} RGB/Depth pairs.")
        if not self.duplicate_groups:
            print("No duplicate crops detected.")
            return
        print(f"Found {len(self.duplicate_groups)} groups of duplicates:")
        for digest, ids in sorted(self.duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True):
            if len(ids) < 2:
                continue
            print(f"  hash={digest[:12]}… count={len(ids)} ids={sorted(ids)}")


def find_duplicates(rgb_dir: Path, depth_dir: Path | None) -> DuplicateReport:
    pairs = _collect_pairs(rgb_dir, depth_dir)
    digest_map: Dict[str, List[int]] = {}
    for sample_id, (rgb_path, depth_path) in pairs.items():
        digest = _hash_pair(rgb_path, depth_path)
        digest_map.setdefault(digest, []).append(sample_id)
    dupes = {digest: ids for digest, ids in digest_map.items() if len(ids) > 1}
    return DuplicateReport(total_pairs=len(pairs), duplicate_groups=dupes)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect duplicate augmented crops by hashing pixel data.")
    parser.add_argument(
        "--rgb-dir",
        default="../../datasets/Training/Augmented_v8/RGBImages",
        type=Path,
        help="Directory containing RGB_*.png files",
    )
    parser.add_argument(
        "--depth-dir",
        default="../../datasets/Training/Augmented_v8/DepthImages",
        type=Path,
        help="Directory containing Depth_*.png files (optional)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Ignore depth images even if depth directory exists.",
    )
    args = parser.parse_args(argv)

    rgb_dir = args.rgb_dir.resolve()
    depth_dir = None if args.rgb_only else args.depth_dir.resolve()

    if not rgb_dir.exists():
        print(f"RGB directory '{rgb_dir}' not found.", file=sys.stderr)
        return 1
    if depth_dir is not None and not depth_dir.exists():
        print(f"Depth directory '{depth_dir}' not found; continuing with RGB only.")
        depth_dir = None

    report = find_duplicates(rgb_dir, depth_dir)
    report.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
