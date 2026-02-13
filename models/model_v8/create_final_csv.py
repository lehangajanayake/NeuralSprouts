"""Utility to build an inference CSV for the final test set.

The final bundle only ships RGB and Depth folders, so we enumerate the IDs
from the filenames and emit a CSV with the required submission columns.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd

RGB_PATTERN = re.compile(r"^RGB_(\d+)\.png$", re.IGNORECASE)
DEPTH_PATTERN = re.compile(r"^Depth_(\d+)\.png$", re.IGNORECASE)


def _collect_ids(directory: Path, pattern: re.Pattern[str]) -> Set[int]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    ids: Set[int] = set()
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if match:
            ids.add(int(match.group(1)))
    if not ids:
        raise ValueError(f"No matching images found in {directory}")
    return ids


def _format_id_list(values: Sequence[int], limit: int = 10) -> str:
    if not values:
        return "[]"
    head = list(values[:limit])
    suffix = "" if len(values) <= limit else " …"
    return f"{head}{suffix}"


def build_final_csv(
    rgb_dir: Path,
    depth_dir: Path,
    output_csv: Path,
    *,
    expected_count: int | None = 81,
    placeholder: str = "",
) -> Path:
    rgb_ids = _collect_ids(rgb_dir, RGB_PATTERN)
    depth_ids = _collect_ids(depth_dir, DEPTH_PATTERN)

    overlapping = sorted(rgb_ids & depth_ids)
    if not overlapping:
        raise ValueError("No overlapping image IDs between RGB and Depth folders.")

    missing_rgb = sorted(depth_ids - rgb_ids)
    missing_depth = sorted(rgb_ids - depth_ids)
    if missing_rgb:
        print(
            f"[warn] {len(missing_rgb)} depth images lack RGB counterparts; sample { _format_id_list(missing_rgb) }"
        )
    if missing_depth:
        print(
            f"[warn] {len(missing_depth)} RGB images lack depth counterparts; sample { _format_id_list(missing_depth) }"
        )

    if expected_count is not None and expected_count > 0 and len(overlapping) != expected_count:
        raise ValueError(
            f"Expected {expected_count} paired images but found {len(overlapping)}."
        )

    df = pd.DataFrame({
        "image_id": overlapping,
        "DryWeightShoot": [placeholder] * len(overlapping),
    })
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")
    return output_csv


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        required=True,
        help="Directory containing RGB_{id}.png files.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Directory containing Depth_{id}.png files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path (will be overwritten).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=81,
        help="Number of paired images expected (set <=0 to disable the check).",
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default="",
        help="Value to prefill DryWeightShoot column with before prediction.",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    expected = args.expected_count if args.expected_count > 0 else None
    build_final_csv(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        output_csv=args.output,
        expected_count=expected,
        placeholder=args.placeholder,
    )


if __name__ == "__main__":
    main()
