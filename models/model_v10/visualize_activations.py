"""Gradio app to visualise SE / SAM attention maps and embeddings for model_v10.

Launch
------
    cd models/model_v10
    python visualize_activations.py

Then open the URL printed in the terminal (usually http://127.0.0.1:7860).

What it shows
-------------
For **each branch** (RGB and RGBD) the app renders:

1. **Input image** — the RGB channels (or RGB+Depth side-by-side for RGBD).
2. **Feature maps** — output of each BottleneckBlock (mean over channels,
   shown as a heatmap and as an overlay on the input).
3. **SE channel weights** — bar chart of per-channel scaling factors learned
   by Squeeze-and-Excitation.
4. **Spatial Attention map (SAM)** — the CBAM sigmoid gate, shown as a
   heatmap and overlaid on the input.
5. **Embedding vector** — the 1-D embedding produced after GeM pooling,
   visualised as a bar chart.

The **Fusion** section shows the concatenated embedding and the final
predicted dry-weight.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---- local imports ---------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from model import LettuceSAMFusionNet  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT = "best_model_v10.pth"
TRAIN_RGB_DIR = "../../datasets/Training/RGBImages"
TRAIN_DEPTH_DIR = "../../datasets/Training/DepthImages"
LABELS_CSV = "../../datasets/Training/Train.csv"
IMAGE_SIZE = 128
CENTER_CROP = 1000
BLACKLIST_IDS = {163}

FIGURE_DPI = 120  # higher → sharper images in the UI


# ---------------------------------------------------------------------------
# Hook-based activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Register forward-hooks to capture intermediate activations."""

    def __init__(self) -> None:
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str):
        def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
        return hook_fn

    def register(self, model: LettuceSAMFusionNet) -> None:
        """Attach hooks to every layer we want to visualise."""
        # ---------- RGB branch ----------
        for i, block in enumerate(model.rgb_branch.features):
            h = block.register_forward_hook(self._make_hook(f"rgb_block_{i}"))
            self._handles.append(h)
        self._handles.append(
            model.rgb_branch.se.fc.register_forward_hook(self._make_hook("rgb_se_weights"))
        )
        self._handles.append(
            model.rgb_branch.se.register_forward_hook(self._make_hook("rgb_se_output"))
        )
        self._handles.append(
            model.rgb_branch.spatial_attn.conv.register_forward_hook(
                self._make_hook("rgb_sam_conv")
            )
        )
        self._handles.append(
            model.rgb_branch.spatial_attn.register_forward_hook(self._make_hook("rgb_sam_output"))
        )
        self._handles.append(
            model.rgb_branch.global_pool.register_forward_hook(self._make_hook("rgb_gem_output"))
        )
        self._handles.append(
            model.rgb_branch.embedding.register_forward_hook(self._make_hook("rgb_embedding"))
        )
        # ---------- RGBD branch ----------
        for i, block in enumerate(model.rgbd_branch.features):
            h = block.register_forward_hook(self._make_hook(f"rgbd_block_{i}"))
            self._handles.append(h)
        self._handles.append(
            model.rgbd_branch.se.fc.register_forward_hook(self._make_hook("rgbd_se_weights"))
        )
        self._handles.append(
            model.rgbd_branch.se.register_forward_hook(self._make_hook("rgbd_se_output"))
        )
        self._handles.append(
            model.rgbd_branch.spatial_attn.conv.register_forward_hook(
                self._make_hook("rgbd_sam_conv")
            )
        )
        self._handles.append(
            model.rgbd_branch.spatial_attn.register_forward_hook(
                self._make_hook("rgbd_sam_output")
            )
        )
        self._handles.append(
            model.rgbd_branch.global_pool.register_forward_hook(
                self._make_hook("rgbd_gem_output")
            )
        )
        self._handles.append(
            model.rgbd_branch.embedding.register_forward_hook(
                self._make_hook("rgbd_embedding")
            )
        )

    def clear(self) -> None:
        self.activations.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Image loading (same logic as dataloader.py)
# ---------------------------------------------------------------------------

def _center_crop(img: Image.Image, size: int = CENTER_CROP) -> Image.Image:
    w, h = img.size
    side = min(w, h, size)
    left = (w - side) / 2
    top = (h - side) / 2
    return img.crop((left, top, left + side, top + side))


def load_sample(image_id: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Return ``(rgb_tensor, rgbd_tensor, rgb_np_HWC, depth_np_HW)``."""
    rgb_path = os.path.join(TRAIN_RGB_DIR, f"RGB_{image_id}.png")
    depth_path = os.path.join(TRAIN_DEPTH_DIR, f"Depth_{image_id}.png")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    rgb = Image.open(rgb_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")
    rgb = _center_crop(rgb)
    depth = _center_crop(depth)
    sz = (IMAGE_SIZE, IMAGE_SIZE)
    rgb = rgb.resize(sz, Image.BILINEAR)
    depth = depth.resize(sz, Image.BILINEAR)

    rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
    depth_np = np.asarray(depth, dtype=np.float32) / 255.0
    if depth_np.ndim == 2:
        depth_hw = depth_np
        depth_np = depth_np[..., np.newaxis]
    else:
        depth_hw = depth_np[:, :, 0]

    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    depth_t = torch.from_numpy(depth_np).permute(2, 0, 1)            # (1,H,W)
    rgbd_t = torch.cat([rgb_t.squeeze(0), depth_t], dim=0).unsqueeze(0)  # (1,4,H,W)

    return rgb_t, rgbd_t, rgb_np, depth_hw


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _to_heatmap(arr_2d: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Convert a 2-D array to an RGBA heatmap image (H, W, 4), float [0,1]."""
    vmin, vmax = arr_2d.min(), arr_2d.max()
    if vmax - vmin > 1e-8:
        normed = (arr_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(arr_2d)
    cmap_fn = cm.get_cmap(cmap)
    return cmap_fn(normed)  # (H, W, 4)


def _overlay(rgb_hw3: np.ndarray, heat_hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap over an RGB image and return (H, W, 3) uint8."""
    from skimage.transform import resize as sk_resize
    # Resize heatmap to match the input image size
    h, w = rgb_hw3.shape[:2]
    if heat_hw.shape != (h, w):
        heat_hw = sk_resize(heat_hw, (h, w), order=1, preserve_range=True).astype(np.float32)
    heatmap_rgba = _to_heatmap(heat_hw, cmap="jet")[:, :, :3]  # drop alpha
    blended = (1 - alpha) * rgb_hw3 + alpha * heatmap_rgba
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def plot_feature_maps(
    activations: Dict[str, torch.Tensor],
    branch: str,
    rgb_np: np.ndarray,
    n_blocks: int,
) -> plt.Figure:
    """Grid of feature-map mean activations + overlays for one branch."""
    from skimage.transform import resize as sk_resize

    fig, axes = plt.subplots(n_blocks, 3, figsize=(14, 4.5 * n_blocks), dpi=FIGURE_DPI)
    if n_blocks == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D

    for i in range(n_blocks):
        key = f"{branch}_block_{i}"
        feat = activations[key][0]  # (C, H', W')
        mean_map = feat.mean(dim=0).numpy()  # (H', W')
        n_channels = feat.shape[0]

        # Column 0: raw heatmap
        axes[i, 0].imshow(_to_heatmap(mean_map, "inferno"), interpolation="nearest")
        axes[i, 0].set_title(f"Block {i}  (mean of {n_channels} ch)   [{mean_map.shape[0]}×{mean_map.shape[1]}]",
                             fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Column 1: overlay on input
        overlay = _overlay(rgb_np, mean_map, alpha=0.55)
        axes[i, 1].imshow(overlay, interpolation="nearest")
        axes[i, 1].set_title(f"Block {i}  overlay", fontsize=11)
        axes[i, 1].axis("off")

        # Column 2: top-4 individual channels
        topk = min(4, n_channels)
        channel_energy = feat.pow(2).sum(dim=(1, 2))
        top_idx = channel_energy.topk(topk).indices.tolist()
        grid = np.zeros((mean_map.shape[0] * 1, mean_map.shape[1] * topk))
        for j, ci in enumerate(top_idx):
            ch_map = feat[ci].numpy()
            grid[:, j * mean_map.shape[1] : (j + 1) * mean_map.shape[1]] = ch_map
        axes[i, 2].imshow(_to_heatmap(grid, "viridis"), interpolation="nearest")
        axes[i, 2].set_title(f"Top-{topk} channels by energy  (idx {top_idx})", fontsize=10)
        axes[i, 2].axis("off")

    fig.suptitle(f"{branch.upper()} Branch — Feature Maps", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_se_weights(
    activations: Dict[str, torch.Tensor],
    branch: str,
) -> plt.Figure:
    """Bar chart of Squeeze-and-Excitation channel scaling factors."""
    key = f"{branch}_se_weights"
    # se.fc output shape is (B, C) after the sigmoid
    weights = activations[key][0].numpy()  # (C,)
    n = len(weights)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.18), 4), dpi=FIGURE_DPI)
    colors = plt.cm.RdYlGn(weights)  # low = red, high = green
    bars = ax.bar(range(n), weights, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Channel index", fontsize=11)
    ax.set_ylabel("SE gate value (0–1)", fontsize=11)
    ax.set_title(f"{branch.upper()} — SE Channel Attention Weights", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=weights.mean(), color="blue", linestyle="--", linewidth=1, label=f"mean = {weights.mean():.3f}")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, n, max(1, n // 20)))
    fig.tight_layout()
    return fig


def plot_sam_attention(
    activations: Dict[str, torch.Tensor],
    branch: str,
    rgb_np: np.ndarray,
) -> plt.Figure:
    """Spatial attention gate map + overlay on input."""
    # The SAM conv output is pre-sigmoid; let's use the full SAM module output
    # which has the attention already applied. We can reconstruct the gate
    # from sam_conv (sigmoid of the conv output).
    sam_conv = activations[f"{branch}_sam_conv"][0]  # (1, H', W')
    gate = torch.sigmoid(sam_conv).squeeze(0).numpy()  # (H', W')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)

    # 1) Raw gate as heatmap
    im = axes[0].imshow(_to_heatmap(gate, "hot"), interpolation="nearest")
    axes[0].set_title(f"SAM Gate  [{gate.shape[0]}×{gate.shape[1]}]", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2) Overlay on input
    overlay = _overlay(rgb_np, gate, alpha=0.6)
    axes[1].imshow(overlay, interpolation="nearest")
    axes[1].set_title("SAM Gate overlay on input", fontsize=12)
    axes[1].axis("off")

    # 3) Gate histogram
    axes[2].hist(gate.ravel(), bins=50, color="coral", edgecolor="black", linewidth=0.4)
    axes[2].set_xlabel("Gate value", fontsize=11)
    axes[2].set_ylabel("Pixel count", fontsize=11)
    axes[2].set_title("SAM Gate distribution", fontsize=12, fontweight="bold")
    axes[2].axvline(gate.mean(), color="blue", linestyle="--", label=f"mean={gate.mean():.3f}")
    axes[2].legend()

    fig.suptitle(f"{branch.upper()} — Spatial Attention (CBAM)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_embedding(
    activations: Dict[str, torch.Tensor],
    branch: str,
) -> plt.Figure:
    """Visualise the embedding vector as a coloured bar chart."""
    emb = activations[f"{branch}_embedding"][0].numpy()  # (embed_dim,)
    n = len(emb)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 0.12), 4), dpi=FIGURE_DPI,
                              gridspec_kw={"width_ratios": [3, 1]})

    # Bar chart
    norm_emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-8)
    colors = plt.cm.coolwarm(norm_emb)
    axes[0].bar(range(n), emb, color=colors, edgecolor="none", width=1.0)
    axes[0].set_xlabel("Embedding dimension", fontsize=11)
    axes[0].set_ylabel("Activation", fontsize=11)
    axes[0].set_title(f"{branch.upper()} Embedding  (dim={n})", fontsize=13, fontweight="bold")
    axes[0].set_xticks(range(0, n, max(1, n // 10)))

    # Distribution histogram
    axes[1].hist(emb, bins=30, orientation="horizontal", color="steelblue", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].set_title("Distribution", fontsize=11)

    fig.tight_layout()
    return fig


def plot_input_images(rgb_np: np.ndarray, depth_np: np.ndarray) -> plt.Figure:
    """Side-by-side RGB + Depth input visualisation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=FIGURE_DPI)
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB Input", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(depth_np, cmap="magma")
    axes[1].set_title("Depth Input", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_fusion_summary(
    activations: Dict[str, torch.Tensor],
    predictions: Dict[str, float],
    true_weight: Optional[float],
) -> plt.Figure:
    """Concatenated embedding + prediction summary."""
    rgb_emb = activations["rgb_embedding"][0].numpy()
    rgbd_emb = activations["rgbd_embedding"][0].numpy()
    fused = np.concatenate([rgb_emb, rgbd_emb])

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=FIGURE_DPI,
                              gridspec_kw={"width_ratios": [3, 1]})

    # Fused embedding
    n = len(fused)
    mid = len(rgb_emb)
    colors = ["#4C72B0"] * mid + ["#DD8452"] * (n - mid)
    axes[0].bar(range(n), fused, color=colors, edgecolor="none", width=1.0)
    axes[0].axvline(mid - 0.5, color="red", linestyle="--", linewidth=1.5, label="RGB | RGBD boundary")
    axes[0].set_xlabel("Dimension", fontsize=11)
    axes[0].set_ylabel("Value", fontsize=11)
    axes[0].set_title(f"Concatenated Fusion Embedding  (dim={n})", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    # Prediction summary as text
    axes[1].axis("off")
    text_lines = [
        f"RGB branch:   {predictions['rgb']:.3f} g",
        f"RGBD branch:  {predictions['rgbd']:.3f} g",
        f"Fused pred:   {predictions['fused']:.3f} g",
    ]
    if true_weight is not None:
        text_lines.append(f"\nTrue weight:  {true_weight:.3f} g")
        text_lines.append(f"Error:        {abs(predictions['fused'] - true_weight):.3f} g")
    axes[1].text(
        0.1, 0.6, "\n".join(text_lines),
        fontsize=14, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"),
        transform=axes[1].transAxes,
    )
    axes[1].set_title("Predictions", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core inference + visualisation
# ---------------------------------------------------------------------------

def run_visualisation(
    image_id: int,
    model: LettuceSAMFusionNet,
    capture: ActivationCapture,
    labels: Dict[int, float],
) -> Tuple[plt.Figure, ...]:
    """Run forward pass, capture activations, and build all plots."""
    capture.clear()

    rgb_t, rgbd_t, rgb_np, depth_np = load_sample(image_id)

    device = next(model.parameters()).device
    rgb_t = rgb_t.to(device)
    rgbd_t = rgbd_t.to(device)

    model.eval()
    with torch.no_grad():
        rgb_pred, rgbd_pred, fused_pred = model(rgb_t, rgbd_t)

    preds = {
        "rgb": rgb_pred.item(),
        "rgbd": rgbd_pred.item(),
        "fused": fused_pred.item(),
    }
    true_w = labels.get(image_id)

    acts = capture.activations

    n_rgb_blocks = sum(1 for k in acts if k.startswith("rgb_block_"))
    n_rgbd_blocks = sum(1 for k in acts if k.startswith("rgbd_block_"))

    fig_input = plot_input_images(rgb_np, depth_np)
    fig_rgb_feat = plot_feature_maps(acts, "rgb", rgb_np, n_rgb_blocks)
    fig_rgb_se = plot_se_weights(acts, "rgb")
    fig_rgb_sam = plot_sam_attention(acts, "rgb", rgb_np)
    fig_rgb_emb = plot_embedding(acts, "rgb")
    fig_rgbd_feat = plot_feature_maps(acts, "rgbd", rgb_np, n_rgbd_blocks)
    fig_rgbd_se = plot_se_weights(acts, "rgbd")
    fig_rgbd_sam = plot_sam_attention(acts, "rgbd", rgb_np)
    fig_rgbd_emb = plot_embedding(acts, "rgbd")
    fig_fusion = plot_fusion_summary(acts, preds, true_w)

    return (
        fig_input,
        fig_rgb_feat, fig_rgb_se, fig_rgb_sam, fig_rgb_emb,
        fig_rgbd_feat, fig_rgbd_se, fig_rgbd_sam, fig_rgbd_emb,
        fig_fusion,
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr
    import pandas as pd

    # ---- Load model + labels ----
    print("Loading model …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Train the model first or update the CHECKPOINT path at the top of this script."
        )

    model = LettuceSAMFusionNet.from_checkpoint(CHECKPOINT, device=device, log_targets=True)
    model.eval()

    capture = ActivationCapture()
    capture.register(model)

    # ---- Read labels ----
    labels: Dict[int, float] = {}
    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        id_col = "image_id" if "image_id" in df.columns else "id"
        for _, row in df.iterrows():
            iid = int(row[id_col])
            if iid not in BLACKLIST_IDS:
                labels[iid] = float(row["DryWeightShoot"])

    # ---- Discover available image IDs ----
    available_ids = sorted(labels.keys())
    if not available_ids:
        # fallback: scan RGB dir
        for fname in sorted(os.listdir(TRAIN_RGB_DIR)):
            if fname.startswith("RGB_") and fname.endswith(".png"):
                iid = int(fname.replace("RGB_", "").replace(".png", ""))
                if iid not in BLACKLIST_IDS:
                    available_ids.append(iid)

    print(f"Found {len(available_ids)} images.  Model on {device}.")

    # ---- Gradio callback ----
    def on_submit(image_id: int):
        try:
            figs = run_visualisation(image_id, model, capture, labels)
        except FileNotFoundError as e:
            raise gr.Error(str(e))
        return figs

    # ---- Build interface ----
    with gr.Blocks(
        title="🥬 Model v10 — Activation Visualiser",
        theme=gr.themes.Soft(),
        css="""
            .plot-container { min-height: 400px !important; }
            .gradio-container { max-width: 1400px !important; margin: auto; }
        """,
    ) as app:
        gr.Markdown(
            "# 🥬 Model v10 — Activation Map Visualiser\n"
            "Select a training image ID and see how **SE**, **Spatial Attention (SAM)**, "
            "feature maps, and embeddings respond.  All plots are large and zoomable."
        )

        with gr.Row():
            id_input = gr.Dropdown(
                choices=available_ids,
                value=available_ids[0] if available_ids else None,
                label="Image ID",
                info="Pick a training image to visualise",
            )
            run_btn = gr.Button("🔍  Visualise", variant="primary", scale=0)

        # ---------- Input ----------
        gr.Markdown("---\n## 📷 Input Images")
        fig_input = gr.Plot(label="RGB + Depth Input", elem_classes=["plot-container"])

        # ---------- RGB Branch ----------
        gr.Markdown("---\n## 🔴 RGB Branch")
        with gr.Accordion("Feature Maps (per block)", open=True):
            fig_rgb_feat = gr.Plot(label="RGB Feature Maps", elem_classes=["plot-container"])
        with gr.Row():
            fig_rgb_se = gr.Plot(label="RGB SE Channel Weights", elem_classes=["plot-container"])
        with gr.Accordion("Spatial Attention (SAM)", open=True):
            fig_rgb_sam = gr.Plot(label="RGB Spatial Attention", elem_classes=["plot-container"])
        fig_rgb_emb = gr.Plot(label="RGB Embedding", elem_classes=["plot-container"])

        # ---------- RGBD Branch ----------
        gr.Markdown("---\n## 🟢 RGBD Branch")
        with gr.Accordion("Feature Maps (per block)", open=True):
            fig_rgbd_feat = gr.Plot(label="RGBD Feature Maps", elem_classes=["plot-container"])
        with gr.Row():
            fig_rgbd_se = gr.Plot(label="RGBD SE Channel Weights", elem_classes=["plot-container"])
        with gr.Accordion("Spatial Attention (SAM)", open=True):
            fig_rgbd_sam = gr.Plot(label="RGBD Spatial Attention", elem_classes=["plot-container"])
        fig_rgbd_emb = gr.Plot(label="RGBD Embedding", elem_classes=["plot-container"])

        # ---------- Fusion ----------
        gr.Markdown("---\n## 🔗 Fusion & Prediction")
        fig_fusion = gr.Plot(label="Fused Embedding + Predictions", elem_classes=["plot-container"])

        # ---- Wire events ----
        outputs = [
            fig_input,
            fig_rgb_feat, fig_rgb_se, fig_rgb_sam, fig_rgb_emb,
            fig_rgbd_feat, fig_rgbd_se, fig_rgbd_sam, fig_rgbd_emb,
            fig_fusion,
        ]
        run_btn.click(fn=on_submit, inputs=[id_input], outputs=outputs)
        id_input.change(fn=on_submit, inputs=[id_input], outputs=outputs)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, inbrowser=True)
