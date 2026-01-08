"""Streamlit app for visualizing RGB and Depth images side-by-side."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import glob


# Page config
st.set_page_config(page_title="Plant Dataset Viewer", layout="wide")

st.title("🌱 Plant Dataset Viewer")
st.markdown("View RGB and Depth images side-by-side with dataset statistics")

# Sidebar - Configuration
st.sidebar.header("Configuration")

dataset_type = st.sidebar.radio(
    "Select Dataset",
    ["Original", "Augmented"],
    help="Original: raw preprocessed images | Augmented: with random transformations"
)

# Set paths based on selection
if dataset_type == "Original":
    rgb_dir = "../../datasets/Training/Augmented/RGBImages"
    depth_dir = "../../datasets/Training/Augmented/DepthImages"
    csv_path = "../../datasets/Training/Train.csv"
    title_prefix = "Original"
else:
    rgb_dir = "../../datasets/Training/Augmented/RGBImages"
    depth_dir = "../../datasets/Training/Augmented/DepthImages"
    csv_path = "../../datasets/Training/Augmented/Train_aug.csv"
    title_prefix = "Augmented"

# Make paths absolute
rgb_dir = os.path.abspath(rgb_dir)
depth_dir = os.path.abspath(depth_dir)
csv_path = os.path.abspath(csv_path)

# Load CSV
try:
    df = pd.read_csv(csv_path)
    id_col = 'image_id' if 'image_id' in df.columns else 'id'
    available_ids = sorted(df[id_col].unique().tolist())
except FileNotFoundError:
    st.error(f"❌ CSV not found: {csv_path}")
    st.stop()

# Filter by selection
if dataset_type == "Original":
    # Only original images (no _aug_ in name)
    available_ids = [i for i in available_ids if '_aug_' not in str(i)]
else:
    # Only augmented images (_aug_ in name)
    available_ids = [i for i in available_ids if '_aug_' in str(i)]

if not available_ids:
    st.warning(f"⚠️ No {dataset_type.lower()} images found. Make sure to run preprocess.py first.")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(available_ids)} {dataset_type.lower()} images")

# Selection widget
selected_id = st.sidebar.selectbox(
    "Select Image ID",
    available_ids,
    help="Choose which image to display"
)

# Navigation buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("⬅️ Previous"):
        current_idx = available_ids.index(selected_id)
        new_idx = (current_idx - 1) % len(available_ids)
        st.session_state.selected_id = available_ids[new_idx]
        st.rerun()

with col2:
    st.write(f"{available_ids.index(selected_id) + 1}/{len(available_ids)}")

with col3:
    if st.button("Next ➡️"):
        current_idx = available_ids.index(selected_id)
        new_idx = (current_idx + 1) % len(available_ids)
        st.session_state.selected_id = available_ids[new_idx]
        st.rerun()

# Load and display images
rgb_path = os.path.join(rgb_dir, f"RGB_{selected_id}.png")
depth_path = os.path.join(depth_dir, f"Depth_{selected_id}.png")

if not os.path.exists(rgb_path):
    st.error(f"❌ RGB image not found: {rgb_path}")
    st.stop()

if not os.path.exists(depth_path):
    st.error(f"❌ Depth image not found: {depth_path}")
    st.stop()

# Load images
rgb_img = Image.open(rgb_path).convert('RGB')
depth_img = Image.open(depth_path).convert('L')

rgb_arr = np.array(rgb_img, dtype=np.uint8)
depth_arr = np.array(depth_img, dtype=np.uint8)

# Main display
st.header(f"{title_prefix} Image: {selected_id}")

# Side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎨 RGB Image")
    st.image(rgb_img, use_column_width=True)
    st.caption(f"Size: {rgb_img.size[0]}×{rgb_img.size[1]}")
    st.caption(f"Min: {rgb_arr.min()}, Max: {rgb_arr.max()}, Mean: {rgb_arr.mean():.1f}")

with col2:
    st.subheader("📊 Depth Image")
    st.image(depth_img, use_column_width=True, clamp=True)
    st.caption(f"Size: {depth_img.size[0]}×{depth_img.size[1]}")
    st.caption(f"Min: {depth_arr.min()}, Max: {depth_arr.max()}, Mean: {depth_arr.mean():.1f}")

# Show metadata
st.divider()
st.subheader("📋 Image Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.metric("Image ID", selected_id)
    
    # Find metadata from CSV
    row = df[df[id_col] == selected_id]
    if not row.empty:
        row = row.iloc[0]
        if 'DryWeightShoot' in df.columns:
            st.metric("Dry Weight (g)", f"{row['DryWeightShoot']:.3f}")
        if 'Variety' in df.columns:
            st.metric("Variety", row['Variety'])

with info_col2:
    st.metric("Position in Dataset", f"{available_ids.index(selected_id) + 1}/{len(available_ids)}")
    st.metric("RGB File Size", f"{os.path.getsize(rgb_path) / 1024:.1f} KB")
    st.metric("Depth File Size", f"{os.path.getsize(depth_path) / 1024:.1f} KB")

# Dataset Statistics
st.divider()
st.subheader("📈 Dataset Statistics")

stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.metric("Total Images", len(available_ids))

with stat_col2:
    if 'DryWeightShoot' in df.columns:
        dry_weights = df['DryWeightShoot'].dropna()
        st.metric("Avg Dry Weight (g)", f"{dry_weights.mean():.3f}")

with stat_col3:
    if 'Variety' in df.columns:
        st.metric("Unique Varieties", df['Variety'].nunique())

# Histogram
if 'DryWeightShoot' in df.columns:
    st.subheader("Dry Weight Distribution")
    
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        # All dry weights
        dry_weights_all = df['DryWeightShoot'].dropna()
        st.bar_chart(
            dry_weights_all.value_counts(bins=20).sort_index(),
            use_container_width=True
        )
        st.caption("All Dry Weights")
    
    with hist_col2:
        # Show stats table
        stats_data = {
            'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max'],
            'Value': [
                len(dry_weights_all),
                f"{dry_weights_all.mean():.3f}",
                f"{dry_weights_all.std():.3f}",
                f"{dry_weights_all.min():.3f}",
                f"{dry_weights_all.quantile(0.25):.3f}",
                f"{dry_weights_all.median():.3f}",
                f"{dry_weights_all.quantile(0.75):.3f}",
                f"{dry_weights_all.max():.3f}",
            ]
        }
        st.dataframe(stats_data, use_container_width=True, hide_index=True)

# Advanced options
with st.expander("🔧 Advanced Options"):
    st.subheader("Augmentation Filter")
    
    show_originals_only = st.checkbox("Show only originals (no _aug_)", value=False)
    show_augmented_only = st.checkbox("Show only augmented (_aug_)", value=False)
    
    if show_originals_only and show_augmented_only:
        st.warning("Cannot select both filters")
    else:
        filtered_ids = available_ids.copy()
        
        if show_originals_only:
            filtered_ids = [i for i in filtered_ids if '_aug_' not in str(i)]
        elif show_augmented_only:
            filtered_ids = [i for i in filtered_ids if '_aug_' in str(i)]
        
        st.info(f"Filtered to {len(filtered_ids)} images")
    
    # Raw data viewer
    st.subheader("Raw Image Data")
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        if st.checkbox("Show RGB array"):
            st.write(f"Shape: {rgb_arr.shape}, dtype: {rgb_arr.dtype}")
            st.write(f"RGB stats: min={rgb_arr.min()}, max={rgb_arr.max()}, mean={rgb_arr.mean():.1f}")
            if st.checkbox("Show RGB values"):
                st.dataframe(rgb_arr[:10, :10, 0], use_container_width=True)
    
    with data_col2:
        if st.checkbox("Show Depth array"):
            st.write(f"Shape: {depth_arr.shape}, dtype: {depth_arr.dtype}")
            st.write(f"Depth stats: min={depth_arr.min()}, max={depth_arr.max()}, mean={depth_arr.mean():.1f}")
            if st.checkbox("Show Depth values"):
                st.dataframe(depth_arr[:10, :10], use_container_width=True)

# Footer
st.divider()
st.markdown("""
---
**Model V5 Dataset Viewer** | Built with Streamlit  
RGB and Depth images are normalized to [0, 1] during training
""")
