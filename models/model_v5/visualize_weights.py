"""Streamlit app to visualize learned convolutional filters from Model V5.

Run:
    streamlit run visualize_weights.py

Loads a trained checkpoint and displays the learned conv filters from each branch,
helping you understand what features the model has learned to detect.

STANDALONE: Works directly with .pth files - no model.py needed!
"""

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

st.set_page_config(page_title="Model V5 Conv Filters", layout="wide")
st.title("Model V5 Convolutional Filter Visualization")
st.caption("Visualize what the model has learned to detect in each branch")


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load checkpoint and extract state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract state dict (handle different checkpoint formats)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume the checkpoint IS the state dict
        state_dict = checkpoint
    
    return state_dict, checkpoint


def normalize_filters(filters):
    """Normalize filter weights to [0, 1] for visualization."""
    filters = filters.detach().cpu().numpy()
    # Normalize each filter independently
    f_min = filters.min(axis=(2, 3), keepdims=True)
    f_max = filters.max(axis=(2, 3), keepdims=True)
    filters_norm = (filters - f_min) / (f_max - f_min + 1e-8)
    return filters_norm


def plot_filters(filters, title, max_filters=64):
    """Plot conv filters in a grid."""
    filters = normalize_filters(filters)
    n_filters = min(filters.shape[0], max_filters)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(grid_size * grid_size):
        ax = axes[idx]
        ax.axis("off")
        
        if idx < n_filters:
            # For multi-channel filters, show first channel or RGB composite
            filter_img = filters[idx]
            
            if filter_img.shape[0] == 1:
                # Single channel (grayscale)
                ax.imshow(filter_img[0], cmap="viridis")
            elif filter_img.shape[0] == 3:
                # RGB channels
                filter_rgb = np.transpose(filter_img, (1, 2, 0))
                ax.imshow(filter_rgb)
            elif filter_img.shape[0] == 4:
                # RGBD - show first 3 channels as RGB
                filter_rgb = np.transpose(filter_img[:3], (1, 2, 0))
                ax.imshow(filter_rgb)
            else:
                # Multiple channels - show first channel
                ax.imshow(filter_img[0], cmap="viridis")
            
            ax.set_title(f"F{idx}", fontsize=8)
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    return fig


def extract_conv_layers(state_dict):
    """Extract all conv layer weights directly from state dict."""
    conv_layers = {}
    
    for key, value in state_dict.items():
        # Look for conv weight keys (e.g., "rgb_branch.features.0.net.0.weight")
        if "features" in key and "net.0.weight" in key:
            # Parse branch name and layer number
            parts = key.split(".")
            branch = parts[0].replace("_branch", "").upper()  # e.g., "rgb_branch" -> "RGB"
            layer_idx = int(parts[2]) + 1  # e.g., "features.0" -> Conv1
            
            layer_name = f"{branch}_Conv{layer_idx}"
            conv_layers[layer_name] = value
    
    return conv_layers


def get_layer_input_channels(layer_name, all_layers):
    """Get the expected input channels for a layer."""
    if layer_name in all_layers:
        weight = all_layers[layer_name]
        # Weight shape is (out_channels, in_channels, H, W)
        return weight.shape[1]
    return None


def apply_conv_layer(image, weight, kernel_size=9, padding=4, validate_channels=True):
    """Apply a conv layer to an image and return feature maps.
    
    Note: Standalone mode - we only have conv weights, not BatchNorm params.
    So we do Conv → ReLU → MaxPool without BatchNorm normalization.
    """
    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Validate channel compatibility (only for first layer in sequence)
    if validate_channels:
        expected_channels = weight.shape[1]  # in_channels from weight
        actual_channels = image.shape[1]     # channels from input
        
        if expected_channels != actual_channels:
            raise ValueError(
                f"Channel mismatch: Filter expects {expected_channels} channels, "
                f"but input has {actual_channels} channels. "
                f"Make sure you've uploaded the correct image type for this branch."
            )
    
    # Apply convolution
    feature_maps = F.conv2d(image, weight, padding=padding)
    
    # Apply ReLU (no BatchNorm since we don't have those weights in standalone mode)
    feature_maps = F.relu(feature_maps)
    
    # Apply MaxPool
    feature_maps = F.max_pool2d(feature_maps, kernel_size=2)
    
    return feature_maps


def visualize_feature_maps(feature_maps, title, max_maps=16):
    """Visualize feature maps in a grid."""
    feature_maps = feature_maps.squeeze(0).detach().cpu().numpy()  # Remove batch dim
    n_maps = min(feature_maps.shape[0], max_maps)
    
    grid_size = int(np.ceil(np.sqrt(n_maps)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(grid_size * grid_size):
        ax = axes[idx]
        ax.axis("off")
        
        if idx < n_maps:
            fmap = feature_maps[idx]
            # Normalize for visualization
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Map {idx}", fontsize=8)
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    return fig


# Sidebar controls
st.sidebar.header("Model Selection")

# Find available checkpoints
checkpoint_dir = Path("./5.5")
checkpoint_files = list(checkpoint_dir.glob("best_*.pth")) + list(checkpoint_dir.glob("*.pth"))
checkpoint_files = [f for f in checkpoint_files if f.is_file()]
    
if not checkpoint_files:
    st.error("No checkpoint files found in current directory. Train a model first!")
    st.stop()

checkpoint_names = [f.name for f in checkpoint_files]
selected_checkpoint = st.sidebar.selectbox("Select checkpoint", checkpoint_names)
checkpoint_path = str(checkpoint_dir / selected_checkpoint)

# Load model
try:
    with st.spinner("Loading checkpoint..."):
        state_dict, checkpoint = load_checkpoint(checkpoint_path)
    
    st.sidebar.success(f"✓ Loaded: {selected_checkpoint}")
    
    # Show checkpoint info
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            st.sidebar.info(f"Epoch: {checkpoint['epoch']}")
        if "best_val_mae" in checkpoint:
            st.sidebar.info(f"Best Val MAE: {checkpoint['best_val_mae']:.4f}")
    
except Exception as e:
    st.error(f"Failed to load checkpoint: {e}")
    st.stop()

# Extract all conv layers directly from state dict
st.sidebar.header("Filter Selection")

all_layers = extract_conv_layers(state_dict)
layer_names = sorted(all_layers.keys())

# Debug info
if not all_layers:
    st.sidebar.warning("⚠️ No conv layers found in checkpoint")
    st.sidebar.write("State dict keys:", list(state_dict.keys())[:10])
else:
    st.sidebar.success(f"✓ Found {len(all_layers)} conv layers")

selected_layer = st.sidebar.selectbox("Select layer", layer_names)
max_filters = st.sidebar.slider("Max filters to display", 16, 256, 64, step=16)

# Display filters
st.subheader(f"Filters: {selected_layer}")

filters = all_layers[selected_layer]
st.write(f"**Shape:** {tuple(filters.shape)} (out_channels, in_channels, height, width)")

fig = plot_filters(filters, f"{selected_layer} Filters", max_filters=max_filters)
st.pyplot(fig)
plt.close(fig)

# Statistics
st.subheader("Filter Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean", f"{filters.mean().item():.4f}")
with col2:
    st.metric("Std Dev", f"{filters.std().item():.4f}")
with col3:
    st.metric("Min", f"{filters.min().item():.4f}")
with col4:
    st.metric("Max", f"{filters.max().item():.4f}")

# Show all layers overview
with st.expander("📊 All Layers Overview"):
    st.markdown("### Conv Layer Shapes")
    
    layer_info = []
    for name, weights in all_layers.items():
        layer_info.append({
            "Layer": name,
            "Shape": str(tuple(weights.shape)),
            "Parameters": weights.numel(),
            "Mean": f"{weights.mean().item():.4f}",
            "Std": f"{weights.std().item():.4f}",
        })
    
    st.table(layer_info)

# Tips
st.info("""
💡 **Tips:**
- **Early layers** (Conv1, Conv2) typically detect low-level features (edges, colors, textures)
- **Deeper layers** (Conv3, Conv4) detect more complex patterns
- **Bright spots** in filters indicate strong activation for certain features
- Compare filters across branches to see how each specializes

🎨 **Color Guide (Viridis colormap):**
- **Purple/Dark Blue**: Low/negative weights - filters respond weakly or inhibit features
- **Green/Yellow**: Medium weights - moderate filter response
- **Bright Yellow/White**: High weights - strong filter response to detected features

**Standalone mode:** This tool reads weights directly from .pth files - no model.py needed!
""")

# ============================================================================
# FEATURE MAP VISUALIZATION
# ============================================================================

st.markdown("---")
st.header("🔍 Feature Map Visualization")
st.caption("See what the convolutional filters detect in a sample image")

# Image upload - separate RGB and Depth
col1, col2 = st.columns(2)
with col1:
    uploaded_rgb = st.file_uploader("Upload RGB Image", type=["png", "jpg", "jpeg"], key="rgb_upload")
with col2:
    uploaded_depth = st.file_uploader("Upload Depth Image", type=["png", "jpg", "jpeg"], key="depth_upload")

# Prepare tensors
rgb_tensor = None
depth_tensor = None
rgbd_tensor = None

if uploaded_rgb is not None or uploaded_depth is not None:
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])
    
    # Display uploaded images
    display_cols = st.columns(2)
    
    if uploaded_rgb is not None:
        rgb_image = Image.open(uploaded_rgb).convert("RGB")
        with display_cols[0]:
            st.subheader("RGB Image")
            st.image(rgb_image, use_container_width=True)
        rgb_tensor = transform(rgb_image)
        st.write(f"**RGB Shape:** {tuple(rgb_tensor.shape)}")
    
    if uploaded_depth is not None:
        depth_image = Image.open(uploaded_depth).convert("L")  # Grayscale
        with display_cols[1]:
            st.subheader("Depth Image")
            st.image(depth_image, use_container_width=True)
        depth_tensor = transform(depth_image)
        st.write(f"**Depth Shape:** {tuple(depth_tensor.shape)}")
    
    # Create RGBD if both available
    if rgb_tensor is not None and depth_tensor is not None:
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        st.success(f"✓ Combined RGBD tensor created: {tuple(rgbd_tensor.shape)}")
    
    # Determine available branches based on filters and uploaded images
    st.subheader("Select Branch & Layer")
    
    # Check what branches exist in the model
    model_branches = set()
    for layer_name in layer_names:
        branch = layer_name.split("_")[0]
        model_branches.add(branch)
    
    # Map branches to their expected channels (first conv layer)
    branch_channel_requirements = {}
    for branch in model_branches:
        first_layer = f"{branch}_Conv1"
        if first_layer in all_layers:
            branch_channel_requirements[branch] = get_layer_input_channels(first_layer, all_layers)
    
    # Determine which branches we can actually use
    available_branches = []
    branch_hints = []
    
    for branch, required_channels in branch_channel_requirements.items():
        if required_channels == 3 and rgb_tensor is not None:
            available_branches.append(branch)
            branch_hints.append(f"{branch} (3ch) ✓")
        elif required_channels == 1 and depth_tensor is not None:
            available_branches.append(branch)
            branch_hints.append(f"{branch} (1ch) ✓")
        elif required_channels == 4 and rgbd_tensor is not None:
            available_branches.append(branch)
            branch_hints.append(f"{branch} (4ch) ✓")
        else:
            missing = "RGB" if required_channels == 3 else "Depth" if required_channels == 1 else "RGB+Depth"
            branch_hints.append(f"{branch} ({required_channels}ch) - need {missing}")
    
    if available_branches:
        st.info(f"✓ Available branches: {', '.join(branch_hints)}")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_branch = st.selectbox("Branch", available_branches, key="feature_branch")
        with col2:
            # Get layers for selected branch (use exact prefix with underscore to avoid RGBD matching RGB)
            branch_prefix = f"{selected_branch}_"
            branch_layers = [name for name in layer_names if name.startswith(branch_prefix)]
            if not branch_layers:
                st.error(f"No layers found for {selected_branch} branch!")
                st.write(f"Available layers: {layer_names}")
                st.stop()
            selected_feature_layer = st.selectbox("Layer", branch_layers, key="feature_layer")
        
        max_feature_maps = st.slider("Max feature maps to display", 8, 64, 16, step=8)
        
        if st.button("🚀 Generate Feature Maps", type="primary"):
            with st.spinner("Computing feature maps..."):
                try:
                    # Determine required channels from the filter
                    conv_weight = all_layers[selected_feature_layer]
                    required_channels = conv_weight.shape[1]
                    
                    # Select appropriate input tensor based on required channels (only for first layer)
                    if required_channels == 3:
                        if rgb_tensor is None:
                            st.error("RGB image (3 channels) required for this branch")
                            st.stop()
                        input_tensor = rgb_tensor
                    elif required_channels == 1:
                        if depth_tensor is None:
                            st.error("Depth image (1 channel) required for this branch")
                            st.stop()
                        input_tensor = depth_tensor
                    elif required_channels == 4:
                        if rgbd_tensor is None:
                            st.error("Both RGB and Depth images (4 channels) required for this branch")
                            st.stop()
                        input_tensor = rgbd_tensor
                    else:
                        # Layer expects intermediate features - need to build them up
                        # Extract branch name and layer number from selected layer
                        layer_parts = selected_feature_layer.split("_Conv")
                        branch = layer_parts[0]
                        current_layer_num = int(layer_parts[1])
                        
                        # Determine initial image type
                        first_layer_name = f"{branch}_Conv1"
                        first_layer_channels = get_layer_input_channels(first_layer_name, all_layers)
                        
                        if first_layer_channels == 3:
                            if rgb_tensor is None:
                                st.error("RGB image (3 channels) required for this branch")
                                st.stop()
                            current_tensor = rgb_tensor
                        elif first_layer_channels == 1:
                            if depth_tensor is None:
                                st.error("Depth image (1 channel) required for this branch")
                                st.stop()
                            current_tensor = depth_tensor
                        elif first_layer_channels == 4:
                            if rgbd_tensor is None:
                                st.error("Both RGB and Depth images (4 channels) required for this branch")
                                st.stop()
                            current_tensor = rgbd_tensor
                        
                        # Feed through all previous layers
                        st.info(f"Building intermediate features through {branch}_Conv1 → {selected_feature_layer}...")
                        for layer_num in range(1, current_layer_num):
                            prev_layer = f"{branch}_Conv{layer_num}"
                            prev_weight = all_layers[prev_layer]
                            current_tensor = apply_conv_layer(current_tensor, prev_weight, validate_channels=False)
                        
                        input_tensor = current_tensor
                    
                    # Apply the selected layer
                    feature_maps = apply_conv_layer(input_tensor, conv_weight, validate_channels=False)
                    
                    # Visualize
                    st.subheader(f"Feature Maps: {selected_feature_layer}")
                    st.write(f"**Input shape to {selected_feature_layer}:** {tuple(input_tensor.shape)}")
                    st.write(f"**Output shape:** {tuple(feature_maps.shape)} (batch, channels, H, W)")
                    
                    fig = visualize_feature_maps(feature_maps, 
                                                f"{selected_feature_layer} Feature Maps",
                                                max_maps=max_feature_maps)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.success("✓ Feature maps generated! Each map shows what different filters detect in your image.")
                    
                    st.info("""
                    🎨 **Feature Map Color Guide:**
                    - **Dark purple/blue**: No activation - filter doesn't detect its pattern here
                    - **Green/Yellow**: Medium activation - filter partially detects its pattern
                    - **Bright yellow**: Strong activation - filter strongly detects its learned pattern
                    
                    Each feature map shows WHERE in the image a specific filter responds. Bright regions 
                    indicate the filter found what it's looking for (edges, textures, shapes, etc.).
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating feature maps: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning(f"No compatible branches found. Upload images to match: {', '.join(branch_hints)}")
    
    # Sequential visualization through all layers
    if st.checkbox("🎬 Show Sequential Processing (all layers in branch)", key="sequential"):
        st.subheader("Sequential Feature Map Evolution")
        st.caption("Watch how the image transforms through each layer")
        
        if available_branches:
            seq_branch = st.selectbox("Branch for sequential view", available_branches, key="seq_branch")
            # Use exact prefix with underscore to avoid RGBD matching RGB
            seq_branch_prefix = f"{seq_branch}_"
            branch_layers_seq = [name for name in sorted(layer_names) if name.startswith(seq_branch_prefix)]
            
            if st.button("▶️ Process Through All Layers", type="primary", key="process_sequential"):
                with st.spinner("Processing through all layers..."):
                    # Determine required channels from first layer of selected branch
                    first_layer = f"{seq_branch}_Conv1"
                    if first_layer not in all_layers:
                        st.error(f"Cannot find first layer for {seq_branch} branch")
                        st.stop()
                    
                    required_channels = get_layer_input_channels(first_layer, all_layers)
                    
                    # Select appropriate input tensor based on required channels
                    if required_channels == 3:
                        if rgb_tensor is None:
                            st.error("RGB image (3 channels) required")
                            st.stop()
                        current_tensor = rgb_tensor.clone()
                    elif required_channels == 1:
                        if depth_tensor is None:
                            st.error("Depth image (1 channel) required")
                            st.stop()
                        current_tensor = depth_tensor.clone()
                    elif required_channels == 4:
                        if rgbd_tensor is None:
                            st.error("Both RGB and Depth images (4 channels) required")
                            st.stop()
                        current_tensor = rgbd_tensor.clone()
                    else:
                        st.error(f"Unexpected channel requirement: {required_channels}")
                        st.stop()
                    
                    for idx, layer_name in enumerate(branch_layers_seq):
                        st.markdown(f"### {layer_name}")
                        
                        conv_weight = all_layers[layer_name]
                        
                        # Validate only on first layer - subsequent layers get validated automatically
                        # because they receive output from previous layer which has correct channels
                        if idx == 0:
                            current_tensor = apply_conv_layer(current_tensor, conv_weight, validate_channels=True)
                        else:
                            # Subsequent layers: don't validate, just process
                            # (channels are guaranteed to match because we're feeding layer output to layer input)
                            current_tensor = apply_conv_layer(current_tensor, conv_weight, validate_channels=False)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = visualize_feature_maps(current_tensor,
                                                        f"{layer_name} Output",
                                                        max_maps=16)
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            st.metric("Channels", current_tensor.shape[1])
                            st.metric("Spatial Size", f"{current_tensor.shape[2]}x{current_tensor.shape[3]}")
                            st.metric("Mean Activation", f"{current_tensor.mean().item():.4f}")
                            st.metric("Max Activation", f"{current_tensor.max().item():.4f}")
                        
                        st.markdown("---")
                    
                    st.success("✓ Complete sequential processing shown!")
                    
                    st.info("""
                    📊 **What you're seeing:**
                    - Each layer progressively abstracts the image
                    - **Conv1**: Detects basic edges, colors, simple textures
                    - **Conv2-3**: Combines low-level features into patterns
                    - **Conv4**: High-level features (shapes, object parts)
                    - **Spatial size shrinks** due to pooling (128→64→32→16→8)
                    - **Channels increase** as features become more complex
                    """)

else:
    st.info("👆 Upload RGB and/or Depth images to visualize feature maps and see what the model sees!")
