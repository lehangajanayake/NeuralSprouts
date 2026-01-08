"""Streamlit app to visualize conv filters and feature maps using full model definition (with BatchNorm).

Run:
    streamlit run visualize_weights_model.py

Requires model.py (PlantV5TripleBranch) so BatchNorm + pooling behavior matches training.
"""
import io
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from model import PlantV5TripleBranch

st.set_page_config(page_title="Model V5 Filters (full model)", layout="wide")
st.title("Model V5 Convolutional Filters & Feature Maps")
st.caption("Uses model.py to keep BatchNorm/ReLU/Pool behavior consistent with training")


def infer_cfg_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int | float]:
    """Infer branch_dim and fc_hidden from weight shapes when checkpoint lacks them."""
    cfg: Dict[str, int | float] = {"branch_dim": 256, "fc_hidden": 256, "dropout": 0.2}

    # branch_dim from head linear layer (out_features)
    head_key = "rgb_branch.head.1.weight"
    if head_key in state_dict:
        cfg["branch_dim"] = state_dict[head_key].shape[0]

    # fc_hidden from fusion_fc first linear (out_features)
    fusion_key = "fusion_fc.net.0.weight"
    if fusion_key in state_dict:
        cfg["fc_hidden"] = state_dict[fusion_key].shape[0]

    return cfg


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[PlantV5TripleBranch, Dict[str, List[str]]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Infer cfg from weights if not present
    cfg = {
        "branch_dim": ckpt.get("branch_dim"),
        "fc_hidden": ckpt.get("fc_hidden"),
        "dropout": ckpt.get("dropout", 0.2),
    }
    inferred = infer_cfg_from_state_dict(state_dict)
    if cfg["branch_dim"] is None:
        cfg["branch_dim"] = inferred["branch_dim"]
    if cfg["fc_hidden"] is None:
        cfg["fc_hidden"] = inferred["fc_hidden"]

    model = PlantV5TripleBranch(**cfg).to(device)

    # Load with strict=False to allow shape mismatches to be reported but not fatal
    missing_unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, {
        "missing": missing_unexpected.missing_keys,
        "unexpected": missing_unexpected.unexpected_keys,
    }


def collect_conv_layers(model: PlantV5TripleBranch) -> Dict[str, nn.Conv2d]:
    layers = {}
    branches = {
        "RGB": model.rgb_branch,
        "RGBD": model.rgbd_branch,
        "DEPTH": model.depth_branch,
    }
    for bname, branch in branches.items():
        for idx, block in enumerate(branch.features):
            conv = block.net[0]  # Conv2d
            layers[f"{bname}_Conv{idx+1}"] = conv
    return layers


def normalize_filters(filters: torch.Tensor) -> np.ndarray:
    f = filters.detach().cpu().numpy()
    f_min = f.min(axis=(2, 3), keepdims=True)
    f_max = f.max(axis=(2, 3), keepdims=True)
    return (f - f_min) / (f_max - f_min + 1e-8)


def plot_filters(filters: torch.Tensor, title: str, max_filters: int = 64):
    f = normalize_filters(filters)
    n = min(f.shape[0], max_filters)
    grid = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(grid * grid):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            fi = f[i]
            if fi.shape[0] == 1:
                ax.imshow(fi[0], cmap="viridis")
            elif fi.shape[0] >= 3:
                ax.imshow(np.transpose(fi[:3], (1, 2, 0)))
            else:
                ax.imshow(fi[0], cmap="viridis")
            ax.set_title(f"F{i}", fontsize=8)
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    return fig


def make_rgbd(rgb_t: torch.Tensor | None, depth_t: torch.Tensor | None) -> torch.Tensor | None:
    if rgb_t is None or depth_t is None:
        return None
    return torch.cat([rgb_t, depth_t], dim=0)


def run_branch_until(layer_name: str, model: PlantV5TripleBranch, rgb: torch.Tensor | None, rgbd: torch.Tensor | None, depth: torch.Tensor | None, device: str = "cpu") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run the specific branch up to and including the requested conv layer, capturing activations."""
    activations: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def hook_fn(name):
        def _fn(_, __, output):
            activations[name] = output.detach().cpu()
        return _fn

    # Decide branch and input
    if layer_name.startswith("RGBD_"):
        branch = model.rgbd_branch
        x = rgbd
    elif layer_name.startswith("DEPTH_"):
        branch = model.depth_branch
        x = depth
    else:
        branch = model.rgb_branch
        x = rgb

    if x is None:
        raise ValueError("Missing required input for this branch")

    # Attach hooks to relu inside each ConvBlock
    target_idx = int(layer_name.split("Conv")[1]) - 1
    for idx, block in enumerate(branch.features):
        handle = block.net[2].register_forward_hook(hook_fn(f"Conv{idx+1}"))
        hooks.append(handle)

    with torch.no_grad():
        _ = branch.features(x.unsqueeze(0).to(device))  # only features part needed

    # Clean hooks
    for h in hooks:
        h.remove()

    # Return requested activation
    key = f"Conv{target_idx+1}"
    if key not in activations:
        raise ValueError(f"Activation {key} not captured")
    return activations[key], activations


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sidebar: checkpoint selection
    st.sidebar.header("Checkpoint")
    # Look in current directory; fall back to ./5.8 if empty
    ckpt_dirs = [Path("."), Path("./5.8/")]
    ckpts: List[Path] = []
    for d in ckpt_dirs:
        ckpts.extend([p for p in d.glob("*.pth") if p.is_file()])
    ckpts = sorted(ckpts)
    if not ckpts:
        st.error("No .pth checkpoints found (checked . and ./5.8).")
        return
    ckpt_name = st.sidebar.selectbox("Select checkpoint", [f"{p.parent}/" + p.name for p in ckpts])
    ckpt_path = ckpts[[f"{p.parent}/" + p.name for p in ckpts].index(ckpt_name)]

    # Load model
    with st.spinner("Loading model and weights..."):
        model, state_info = load_model(str(ckpt_path), device=device)
    st.sidebar.success("Model loaded with BatchNorm & full graph")
    if state_info["missing"]:
        st.sidebar.warning(f"Missing keys: {state_info['missing']}")
    if state_info["unexpected"]:
        st.sidebar.warning(f"Unexpected keys: {state_info['unexpected']}")

    # Collect layers
    conv_layers = collect_conv_layers(model)
    layer_names = sorted(conv_layers.keys())

    # Upload images
    st.markdown("---")
    st.header("Feature Maps")
    st.caption("Upload RGB and/or Depth; select branch/layer; see activations")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        rgb_file = st.file_uploader("RGB image", type=["png", "jpg", "jpeg"], key="rgb_u")
    with col_up2:
        depth_file = st.file_uploader("Depth image", type=["png", "jpg", "jpeg"], key="depth_u")

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])
    rgb_t = depth_t = rgbd_t = None
    if rgb_file:
        rgb_img = Image.open(rgb_file).convert("RGB")
        rgb_t = transform(rgb_img)
        st.image(rgb_img, caption="RGB", use_container_width=True)
    if depth_file:
        depth_img = Image.open(depth_file).convert("L")
        depth_t = transform(depth_img)
        st.image(depth_img, caption="Depth", use_container_width=True)
    if rgb_t is not None and depth_t is not None:
        rgbd_t = make_rgbd(rgb_t, depth_t)
        st.info(f"RGBD tensor: {tuple(rgbd_t.shape)}")

    # Branch/layer selection
    branch_choices = sorted({name.split("_")[0] for name in layer_names})
    branch = st.selectbox("Branch", branch_choices)
    branch_layers = [n for n in layer_names if n.startswith(branch)]
    layer = st.selectbox("Layer", branch_layers)
    max_maps = st.slider("Max feature maps to display", 8, 64, 32, step=8)

    if st.button("Generate Feature Maps", type="primary"):
        try:
            activation, _ = run_branch_until(layer, model, rgb_t, rgbd_t, depth_t, device=device)
            st.write(f"Input shape → {layer}: {activation.shape}")
            fig = visualize_feature_maps(activation, f"{layer} Feature Maps", max_maps=max_maps)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Failed to compute feature maps: {e}")

    # Filter view
    st.markdown("---")
    st.header("Conv Filters (weights)")
    layer_f = st.selectbox("Layer (filters)", layer_names, key="filters_layer")
    max_f = st.slider("Max filters to display", 16, 256, 64, step=16, key="filters_max")
    if st.button("Show Filters"):
        filt = conv_layers[layer_f].weight.detach().cpu()
        st.write(f"Shape: {tuple(filt.shape)} (out, in, H, W)")
        fig = plot_filters(filt, f"{layer_f} Filters", max_filters=max_f)
        st.pyplot(fig)
        plt.close(fig)

    st.info("Use this view for accurate activations with BatchNorm (since model.py is used).")


def visualize_feature_maps(feature_maps: torch.Tensor, title: str, max_maps: int = 16):
    fm = feature_maps.squeeze(0).cpu().numpy()
    n = min(fm.shape[0], max_maps)
    grid = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(grid * grid):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            fmap = fm[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Map {i}", fontsize=8)
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
