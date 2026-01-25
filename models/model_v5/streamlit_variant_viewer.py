"""Streamlit app to visualize all 25 variants of each image in model_v5.

Variants breakdown:
- 1 original (center crop, no augmentation)
- 20 random augmented (flips, rotations, color jitter)
- 4 directional shifts (10% offset: up, down, left, right - no color changes)

Run with:
  streamlit run streamlit_variant_viewer.py
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Page config
st.set_page_config(
    page_title="model_v5 Variant Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 Model V5: Variant Visualization & Debugging")
st.markdown(
    """
    Explore all **25 variants** per image:
    - **1 Original**: Center crop, no augmentation
    - **20 Random**: Flips, 90° rotations, color jitter
    - **4 Directional Shifts**: 10% offset (up/down/left/right) - no color changes
    """
)


@st.cache_resource
def load_csv_data():
    """Load the Training CSV to get original plant IDs."""
    csv_path = "../../datasets/Training/Train.csv"
    if not os.path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if "image_id" in df.columns and "id" not in df.columns:
        df.rename(columns={"image_id": "id"}, inplace=True)
    return df


@st.cache_resource
def find_available_originals():
    """Find which original IDs have RGB images in the Training directory."""
    rgb_dir = "../../datasets/Training/RGBImages"
    originals = set()
    if os.path.isdir(rgb_dir):
        for fname in os.listdir(rgb_dir):
            if fname.startswith("RGB_") and fname.endswith(".png"):
                try:
                    pid = int(fname[4:-4])  # Extract ID from "RGB_{id}.png"
                    originals.add(pid)
                except Exception:
                    pass
    return sorted(list(originals))


def load_image_safe(path: str) -> Optional[np.ndarray]:
    """Load image and return as numpy array, or None if missing."""
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        if img.mode == "L":  # Depth (grayscale)
            return np.array(img)
        else:  # RGB
            return np.array(img)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None


def plot_variant_grid(
    plant_id: int,
    variant_type: str = "all"  # "all", "original", "random", "shifted"
):
    """Plot grid of 25 variants for a given plant ID.
    
    Variants layout:
    - Row 0: Original (1 image)
    - Rows 1-4: Random augmented (20 images)
    - Row 5: Directional shifts (4 images)
    """
    
    # Determine which variants to load
    variant_ids = []
    if variant_type in ("all", "original"):
        variant_ids.extend([1])  # Original
    if variant_type in ("all", "random"):
        variant_ids.extend(range(2, 22))  # Random: 2-21
    if variant_type in ("all", "shifted"):
        variant_ids.extend(range(22, 26))  # Shifts: 22-25 (up, down, left, right)
    
    # Try both Training and Augmented directories
    rgb_dir_orig = "../../datasets/Training/RGBImages"
    rgb_dir_aug = "../../datasets/Training/Augmented/RGBImages"
    depth_dir_orig = "../../datasets/Training/DepthImages"
    depth_dir_aug = "../../datasets/Training/Augmented/DepthImages"
    
    # For original, use original directory
    # For variants, use Augmented directory
    
    # Create grid
    n_cols = 5
    n_rows = 5  # 1 + 20 + 4 = 25, so 5x5 grid
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    fig.suptitle(f"Plant ID {plant_id}: All 25 Variants (1 Original + 20 Random + 4 Shifts)", fontsize=16, weight='bold')
    
    axes = axes.flatten()  # Flatten for easier iteration
    
    for idx, ax in enumerate(axes):
        variant_num = idx + 1  # 1-indexed
        
        # Determine which directory to load from and label
        if variant_num == 1:
            # Original: load from Training directory
            rgb_path = os.path.join(rgb_dir_orig, f"RGB_{plant_id}.png")
            depth_path = os.path.join(depth_dir_orig, f"Depth_{plant_id}.png")
            label = "Original\n(Center Crop)"
            variant_label = "ORIG"
            color_frame = "green"
        elif variant_num <= 21:
            # Random augmented: load from Augmented directory
            # Map: variant_num 2-21 → augmented variant_num
            per_original = 25  # 1 original + 24 variants
            base_aug_id = (plant_id - 1) * per_original + 1
            aug_variant_id = base_aug_id + variant_num - 1
            rgb_path = os.path.join(rgb_dir_aug, f"RGB_{aug_variant_id}.png")
            depth_path = os.path.join(depth_dir_aug, f"Depth_{aug_variant_id}.png")
            aug_num = variant_num - 1
            label = f"Random {aug_num}\n(Flip/Rot/Color)"
            variant_label = f"RND{aug_num}"
            color_frame = "blue"
        else:
            # Directional shifts: load from Augmented directory
            shift_index = variant_num - 22  # 0-3
            directions = ["Up", "Down", "Left", "Right"]
            direction = directions[shift_index]
            
            per_original = 25
            base_aug_id = (plant_id - 1) * per_original + 1
            aug_variant_id = base_aug_id + 20 + shift_index
            rgb_path = os.path.join(rgb_dir_aug, f"RGB_{aug_variant_id}.png")
            depth_path = os.path.join(depth_dir_aug, f"Depth_{aug_variant_id}.png")
            label = f"Shift {direction}\n(10% Offset)"
            variant_label = direction.upper()
            color_frame = "red"
        
        # Try to load RGB first
        rgb = load_image_safe(rgb_path)
        depth = load_image_safe(depth_path)
        
        if rgb is not None:
            ax.imshow(rgb)
        elif depth is not None:
            ax.imshow(depth, cmap='viridis')
        else:
            ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', fontsize=10, color='red')
        
        # Add colored frame based on variant type
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor=color_frame, 
                                  facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
        
        ax.set_title(label, fontsize=10, weight='bold', color=color_frame)
        ax.axis('off')
        ax.text(0.05, 0.95, variant_label, transform=ax.transAxes, 
               fontsize=8, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor=color_frame, alpha=0.7),
               verticalalignment='top')
    
    plt.tight_layout()
    return fig


def main():
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Get available originals
    available_originals = find_available_originals()
    
    if not available_originals:
        st.error("❌ No original RGB images found in Training/RGBImages/")
        return
    
    # Plant ID selector
    selected_plant = st.sidebar.selectbox(
        "Select Plant ID",
        available_originals,
        index=0,
    )
    
    # Variant type filter
    variant_type = st.sidebar.radio(
        "Show Variants",
        ["all", "original", "random", "shifted"],
        format_func=lambda x: {
            "all": "All 25 Variants",
            "original": "Original Only",
            "random": "20 Random Augmented",
            "shifted": "4 Directional Shifts"
        }[x]
    )
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Variant Breakdown")
    st.sidebar.markdown("""
    **25 total variants per image:**
    - **1 Green**: Original (center crop)
    - **20 Blue**: Random augmented (flips, rotations, color jitter)
    - **4 Red**: Directional shifts (10% offset, no color changes)
    
    **Shift Directions:**
    - **Up**: Crop window sees upper part
    - **Down**: Crop window sees lower part
    - **Left**: Crop window sees left part
    - **Right**: Crop window sees right part
    """)
    
    # Check if augmented dataset exists
    aug_rgb_dir = "../../datasets/Training/Augmented/RGBImages"
    if not os.path.isdir(aug_rgb_dir):
        st.warning(
            "⚠️ Augmented images not found. Run preprocessing first:\n"
            "```bash\npython preprocess.py\n```"
        )
        return
    
    # Generate and display grid
    st.subheader(f"🖼️ Plant {selected_plant}: Variant Visualization")
    
    try:
        fig = plot_variant_grid(selected_plant, variant_type)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error generating visualization: {e}")
        import traceback
        st.text(traceback.format_exc())
    
    # Statistics section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Dataset Stats")
    
    try:
        train_csv = "../../datasets/Training/Train.csv"
        if os.path.exists(train_csv):
            df_train = pd.read_csv(train_csv)
            n_originals = len(df_train)
            st.sidebar.metric("Original Images", n_originals)
            st.sidebar.metric("Total Variants", n_originals * 25)
        
        # Check augmented output
        if os.path.isdir(aug_rgb_dir):
            n_augmented = len([f for f in os.listdir(aug_rgb_dir) if f.endswith('.png')])
            st.sidebar.metric("Augmented RGB Files", n_augmented)
        
        aug_csv = "../../datasets/Training/Augmented/Train_aug.csv"
        if os.path.exists(aug_csv):
            df_aug = pd.read_csv(aug_csv)
            st.sidebar.metric("Augmented CSV Rows", len(df_aug))
    except Exception as e:
        st.sidebar.warning(f"Stats error: {e}")


if __name__ == "__main__":
    main()
