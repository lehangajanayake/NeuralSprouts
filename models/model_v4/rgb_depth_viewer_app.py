"""
Streamlit app: Browse RGB + Depth image pairs side-by-side (Training / Augmented / Test).

- Select dataset split (Training, Augmented, Test)
- Auto-resolves CSV and image folders (overridable via sidebar)
- Navigate by id with prev/next/random controls
- Adjustable depth visualization (percentile clip + colormap)
- Shows basic row metadata when available (Variety, DryWeightShoot)

Run:
    streamlit run models/model_v4/rgb_depth_viewer_app.py

Requirements:
    pip install streamlit pandas pillow numpy matplotlib
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --------------------
# Config + path helpers
# --------------------

@dataclass(frozen=True)
class DatasetPaths:
    csv_path: str
    rgb_dir: str
    depth_dir: str


def resolve_paths(split: str) -> DatasetPaths:
    base = Path(__file__).resolve().parents[2] / "datasets"
    s = split.lower()
    if s == "training":
        return DatasetPaths(
            csv_path=str(base / "Training" / "Train.csv"),
            rgb_dir=str(base / "Training" / "RGBImages"),
            depth_dir=str(base / "Training" / "DepthImages"),
        )
    if s == "augmented":
        return DatasetPaths(
            csv_path=str(base / "Training" / "Augmented" / "Train_aug.csv"),
            rgb_dir=str(base / "Training" / "Augmented" / "RGBImages"),
            depth_dir=str(base / "Training" / "Augmented" / "DepthImages"),
        )
    if s == "test":
        return DatasetPaths(
            csv_path=str(base / "Test" / "Test.csv"),
            rgb_dir=str(base / "Test" / "RGBImages"),
            depth_dir=str(base / "Test" / "DepthImages"),
        )
    raise ValueError("split must be one of: Training, Augmented, Test")


# --------------
# Data loading
# --------------

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"image_id": "id"})
    return df


@st.cache_data(show_spinner=False)
def read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


@st.cache_data(show_spinner=False)
def read_depth(path: str) -> np.ndarray:
    img = Image.open(path)
    # Convert to grayscale consistently (handles L, I; preserves values as numpy array)
    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")
    return np.array(img)


def build_pairs(df: pd.DataFrame, rgb_dir: str, depth_dir: str) -> List[int]:
    ids: List[int] = []
    seen = set()
    for pid in df["id"].tolist():
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid in seen:
            continue
        rgb_path = os.path.join(rgb_dir, f"RGB_{pid}.png")
        depth_path = os.path.join(depth_dir, f"Depth_{pid}.png")
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            ids.append(pid)
            seen.add(pid)
    return sorted(ids)


# -------------------
# Depth visualization
# -------------------

@st.cache_data(show_spinner=False)
def list_cmaps() -> List[str]:
    # Matplotlib is optional; provide a small curated list
    return [
        "viridis",
        "magma",
        "inferno",
        "plasma",
        "cividis",
        "turbo",
        "jet",
        "gray",
    ]


def depth_to_rgb(depth: np.ndarray, clip_percent: float, cmap_name: str) -> np.ndarray:
    d = depth.astype(np.float32)
    if d.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    lo = np.percentile(d, clip_percent)
    hi = np.percentile(d, 100.0 - clip_percent)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(d)), float(np.max(d))
        if hi <= lo:
            hi = lo + 1.0

    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)

    try:
        import matplotlib.cm as cm
    except Exception:
        # Fallback: simple grayscale
        d8 = (d * 255.0).astype(np.uint8)
        return np.stack([d8, d8, d8], axis=-1)

    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(d)[..., :3]  # drop alpha
    return (rgb * 255.0).astype(np.uint8)


# --------------
# UI + app logic
# --------------

def main():
    st.set_page_config(page_title="RGB + Depth Viewer", layout="wide")
    st.title("RGB + Depth Side-by-Side Viewer")

    with st.sidebar:
        st.header("Dataset")
        split = st.selectbox("Choose dataset", ["Training", "Augmented", "Test"], index=0)
        defaults = resolve_paths(split)

        with st.expander("Paths (override if needed)"):
            csv_path = st.text_input("CSV path", value=defaults.csv_path)
            rgb_dir = st.text_input("RGBImages dir", value=defaults.rgb_dir)
            depth_dir = st.text_input("DepthImages dir", value=defaults.depth_dir)

        st.header("Visualization")
        colormap = st.selectbox("Colormap", list_cmaps(), index=0)
        clip = st.slider("Depth clip percentile", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        disp_height = st.slider("Display height (px)", min_value=200, max_value=900, value=500, step=50)

    # Load CSV
    try:
        df = load_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    if "id" not in df.columns:
        st.error("CSV must contain 'id' or 'image_id' column")
        return

    # Build available ids that have both images
    ids = build_pairs(df, rgb_dir, depth_dir)
    if not ids:
        st.warning("No matching RGB_*.png and Depth_*.png pairs found at the given paths.")
        st.stop()

    # Create lookup to row metadata
    rows_by_id: Dict[int, Dict] = {int(r["id"]): r for r in df.to_dict(orient="records") if "id" in r}

    # Navigation controls
    st.subheader("Navigation")
    nav_cols = st.columns([2, 2, 2, 6])
    with nav_cols[0]:
        if st.button("◀ Prev"):
            st.session_state.setdefault("idx", 0)
            st.session_state["idx"] = (st.session_state["idx"] - 1) % len(ids)
    with nav_cols[1]:
        if st.button("Next ▶"):
            st.session_state.setdefault("idx", 0)
            st.session_state["idx"] = (st.session_state["idx"] + 1) % len(ids)
    with nav_cols[2]:
        if st.button("🎲 Random"):
            import random
            st.session_state.setdefault("idx", 0)
            st.session_state["idx"] = random.randrange(0, len(ids))
    with nav_cols[3]:
        # Direct id selection
        selected_id = st.selectbox("Select id", ids, index=st.session_state.get("idx", 0))
        st.session_state["idx"] = ids.index(selected_id)

    cur_id = ids[st.session_state.get("idx", 0)]

    # Display metadata if present
    meta_cols = st.columns(3)
    meta = rows_by_id.get(cur_id, {})
    with meta_cols[0]:
        st.markdown(f"**id**: {cur_id}")
    with meta_cols[1]:
        if "Variety" in meta:
            st.markdown(f"**Variety**: {meta['Variety']}")
    with meta_cols[2]:
        if "DryWeightShoot" in meta:
            st.markdown(f"**DryWeightShoot**: {meta['DryWeightShoot']}")

    # Load images
    rgb_path = os.path.join(rgb_dir, f"RGB_{cur_id}.png")
    depth_path = os.path.join(depth_dir, f"Depth_{cur_id}.png")

    try:
        rgb = read_rgb(rgb_path)  # HxWx3, RGB
    except Exception as e:
        st.error(f"Failed to read RGB image: {rgb_path} ({e})")
        return

    try:
        depth_raw = read_depth(depth_path)  # HxW, gray or 16-bit
    except Exception as e:
        st.error(f"Failed to read Depth image: {depth_path} ({e})")
        return

    depth_rgb = depth_to_rgb(depth_raw, clip_percent=clip, cmap_name=colormap)

    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb, caption=f"RGB_{cur_id}.png", use_column_width=True)
    with c2:
        st.image(depth_rgb, caption=f"Depth_{cur_id}.png [{colormap}, clip={clip}%]", use_column_width=True)

    st.caption(
        f"Showing split: {split} | Total pairs: {len(ids)} | CSV: {os.path.relpath(csv_path)}"
    )


if __name__ == "__main__":
    main()
