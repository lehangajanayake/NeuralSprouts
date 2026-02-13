"""Interactive Gradio explorer for model_v8 feature maps and kernels."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.hooks import RemovableHandle

from dataloader import PlantDatasetV8
from model import LettuceSAMFusionNet

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


@dataclass
class ExplorerConfig:
    train_csv: str = '../../datasets/Training/Train.csv'
    rgb_dir: str = '../../datasets/Training/RGBImages'
    depth_dir: str = '../../datasets/Training/DepthImages'
    checkpoint: str = 'best_model_v8.pth'
    batch_size: int = 1
    seed: int = 42
    blacklist_ids: Tuple[int, ...] = (163,)
    drop_path_prob: float = 0.1
    rgb_widths: Tuple[int, ...] = (32, 64, 96, 128)
    rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128)
    embed_dim: int = 256


ACTIVATION_PATHS: Dict[str, str] = {
    'RGB Block 1': 'rgb_branch.features.0',
    'RGB Block 4': 'rgb_branch.features.3',
    'RGB Spatial Attention': 'rgb_branch.spatial_attn',
    'RGB Spatial Attention Map': 'rgb_branch.spatial_attn.activation',
    'RGB Embedding': 'rgb_branch.embedding',
    'RGBD Block 1': 'rgbd_branch.features.0',
    'RGBD Block 4': 'rgbd_branch.features.3',
    'RGBD Spatial Attention': 'rgbd_branch.spatial_attn',
    'RGBD Spatial Attention Map': 'rgbd_branch.spatial_attn.activation',
    'RGBD Embedding': 'rgbd_branch.embedding',
    'Fusion Input': 'fusion_in_dropout',
    'Fusion Output': 'fusion',
}

KERNEL_PATHS: Dict[str, str] = {
    'RGB Conv1': 'rgb_branch.features.0.conv1.0',
    'RGBD Conv1': 'rgbd_branch.features.0.conv1.0',
}

ACTIVATION_SEQUENCE: List[str] = list(ACTIVATION_PATHS.keys())

FEATURE_MAPS: Dict[str, torch.Tensor] = {}
HOOK_HANDLES: List[RemovableHandle] = []

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    obj: torch.nn.Module = model
    for part in path.split('.'):
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj


def _register_activation_hooks(model: torch.nn.Module) -> None:
    for name, path in ACTIVATION_PATHS.items():
        module = _resolve_module(model, path)

        def capture(layer_name):
            def _hook(_, __, output):
                FEATURE_MAPS[layer_name] = output.detach().cpu()

            return _hook

        HOOK_HANDLES.append(module.register_forward_hook(capture(name)))


def _tensor_to_image(tensor: torch.Tensor, channel: int | None = None, min_render_size: int = 256) -> Image.Image:
    arr = tensor.squeeze(0).detach().cpu().numpy()
    if arr.ndim == 0:
        plane = np.array([[float(arr)]], dtype=np.float32)
    elif arr.ndim == 3:
        c, h, w = arr.shape
        use_channel = channel if channel is not None and 0 <= channel < c else None
        plane = arr[use_channel] if use_channel is not None else arr.mean(axis=0)
    elif arr.ndim == 2:
        plane = arr
    elif arr.ndim == 1:
        size = int(math.ceil(math.sqrt(arr.shape[0])))
        padded = np.zeros(size * size, dtype=arr.dtype)
        padded[: arr.shape[0]] = arr
        plane = padded.reshape(size, size)
    else:
        raise ValueError(f'Unsupported activation shape: {arr.shape}')

    plane = plane - plane.min()
    denom = plane.max() if plane.max() > 1e-8 else 1.0
    norm = (plane / denom * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(norm, mode='L').convert('RGB')
    height, width = plane.shape
    min_dim = max(1, min(height, width))
    scale = max(1, min_render_size // min_dim)
    if scale > 1:
        img = img.resize((width * scale, height * scale), resample=Image.NEAREST)
    return img


def _tensor_to_depth_image(tensor: torch.Tensor) -> Image.Image:
    depth = tensor.squeeze().detach().cpu().numpy()
    depth = np.clip(depth, 0.0, 1.0)
    img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
    return ImageOps.autocontrast(img).convert('RGB')


def _tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')
    return img


def _kernel_image(module: torch.nn.Conv2d, index: int) -> Tuple[Image.Image, Dict[str, float]]:
    weights = module.weight.detach().cpu().numpy()
    idx = max(0, min(index, weights.shape[0] - 1))
    kernel = weights[idx]
    plane = kernel.mean(axis=0)
    plane = plane - plane.min()
    denom = plane.max() if plane.max() > 1e-8 else 1.0
    norm = (plane / denom * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(norm, mode='L').convert('RGB')
    stats = {
        'kernel_index': idx,
        'min': float(kernel.min()),
        'max': float(kernel.max()),
        'shape': list(kernel.shape),
    }
    return img, stats


def _attention_overlay(
    rgb: torch.Tensor,
    attn: torch.Tensor | None,
    *,
    min_render_size: int = 384,
) -> Tuple[Image.Image, Dict[str, float]]:
    base = _tensor_to_rgb_image(rgb)
    bw, bh = base.size
    min_dim = max(1, min(bw, bh))
    scale = max(1, min_render_size // min_dim)
    if scale > 1:
        base = base.resize((bw * scale, bh * scale), Image.NEAREST)
    bw, bh = base.size
    if attn is None:
        return ImageOps.autocontrast(base), {'error': 'attention map unavailable'}
    heat = attn.squeeze().detach().cpu().numpy()
    if heat.ndim == 3:
        heat = heat[0]
    heat = np.clip(heat, 0.0, 1.0)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode='L').resize((bw, bh), Image.BILINEAR)
    heat_rgb = ImageOps.colorize(heat_img, black=(0, 0, 0), white=(255, 0, 0))
    blended = Image.blend(base, heat_rgb, alpha=0.45)
    stats = {
        'min': float(heat.min()),
        'max': float(heat.max()),
        'mean': float(heat.mean()),
        'shape': list(attn.squeeze(0).shape),
        'high_activation_pct': float((heat > 0.7).mean()),
    }
    return blended, stats


def _attention_heatmap(attn: torch.Tensor | None, *, min_render_size: int = 384) -> Image.Image:
    if attn is None:
        return Image.new('RGB', (min_render_size, min_render_size), color='black')
    heat = attn.squeeze().detach().cpu().numpy()
    if heat.ndim == 3:
        heat = heat[0]
    heat = np.clip(heat, 0.0, 1.0)
    h, w = heat.shape
    new_w = max(min_render_size, w)
    new_h = max(min_render_size, h)
    img = Image.fromarray((heat * 255).astype(np.uint8), mode='L')
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return ImageOps.colorize(img, black=(6, 34, 64), white=(255, 234, 132))


def _channel_journey_images(channel_idx: int) -> List[Tuple[Image.Image, str]]:
    channel = None if channel_idx < 0 else channel_idx
    gallery: List[Tuple[Image.Image, str]] = []
    for name in ACTIVATION_SEQUENCE:
        activation = FEATURE_MAPS.get(name)
        if activation is None:
            img = Image.new('RGB', (64, 64), color='gray')
            caption = f'{name}\n(missing)'
        else:
            img = _tensor_to_image(activation, channel)
            stats_tensor = activation.squeeze(0).detach().cpu()
            channel_cap = stats_tensor.shape[0] if stats_tensor.ndim >= 3 else 1
            caption = f'{name}\nshape={list(stats_tensor.shape)}'
            if channel is not None and channel >= channel_cap:
                caption += ' (mean shown)'
        gallery.append((img, caption))
    return gallery


# -----------------------------------------------------------------------------
# Core analysis functions
# -----------------------------------------------------------------------------


def build_dataset(cfg: ExplorerConfig) -> Tuple[PlantDatasetV8, Dict[int, int]]:
    dataset = PlantDatasetV8(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=False,
        seed=cfg.seed,
        enable_cache=False,
        num_views=1,
        blacklist_ids=cfg.blacklist_ids,
    )
    id_to_idx = {int(row['id']): idx for idx, row in dataset.df.iterrows()}
    return dataset, id_to_idx


def load_model(cfg: ExplorerConfig, device: torch.device) -> LettuceSAMFusionNet:
    model = LettuceSAMFusionNet(
        drop_path_prob=cfg.drop_path_prob,
        rgb_widths=cfg.rgb_widths,
        rgbd_widths=cfg.rgbd_widths,
        embed_dim=cfg.embed_dim,
    ).to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _register_activation_hooks(model)
    return model


def run_forward(sample, model, device):
    FEATURE_MAPS.clear()
    rgb = sample['rgb'].unsqueeze(0).to(device)
    rgbd = sample['rgbd'].unsqueeze(0).to(device)
    with torch.no_grad():
        rgb_pred, rgbd_pred, fusion_pred = model(rgb, rgbd)
    return {
        'rgb_pred': float(rgb_pred.item()),
        'rgbd_pred': float(rgbd_pred.item()),
        'fusion_pred': float(fusion_pred.item()),
    }


def analyze_sample(
    sample_idx: int,
    dataset: PlantDatasetV8,
    model: LettuceSAMFusionNet,
    device: torch.device,
    layer_name: str,
    channel_idx: int,
    kernel_branch: str,
    kernel_idx: int,
):
    sample_idx = max(0, min(sample_idx, len(dataset) - 1))
    sample = dataset[sample_idx]
    preds = run_forward(sample, model, device)

    rgb_img = _tensor_to_rgb_image(sample['rgb'])
    depth_img = _tensor_to_depth_image(sample['rgbd'][3:4])

    activation = FEATURE_MAPS.get(layer_name)
    if activation is None:
        act_img = Image.new('RGB', (64, 64), color='gray')
        act_stats = {'error': f'Layer "{layer_name}" not captured.'}
    else:
        ch = None if channel_idx < 0 else channel_idx
        act_img = _tensor_to_image(activation, ch)
        stats_tensor = activation.squeeze(0).detach().cpu()
        act_stats = {
            'layer': layer_name,
            'shape': list(stats_tensor.shape),
            'min': float(stats_tensor.min()),
            'max': float(stats_tensor.max()),
            'mean': float(stats_tensor.mean()),
            'channel_used': ch,
        }

    kernel_module = _resolve_module(model, KERNEL_PATHS[kernel_branch])
    kernel_img, kernel_stats = _kernel_image(kernel_module, kernel_idx)
    journey_images = _channel_journey_images(channel_idx)

    rgb_attn = FEATURE_MAPS.get('RGB Spatial Attention Map')
    rgbd_attn = FEATURE_MAPS.get('RGBD Spatial Attention Map')
    rgb_overlay, rgb_attn_stats = _attention_overlay(sample['rgb'], rgb_attn)
    rgbd_overlay, rgbd_attn_stats = _attention_overlay(sample['rgb'], rgbd_attn)
    rgb_heatmap = _attention_heatmap(rgb_attn)
    rgbd_heatmap = _attention_heatmap(rgbd_attn)

    info = {
        'dataset_index': int(sample_idx),
        'sample_id': int(sample['id'].item()),
        'target': float(sample['dry_weight'].item()),
        'pred_rgb': preds['rgb_pred'],
        'pred_rgbd': preds['rgbd_pred'],
        'pred_fusion': preds['fusion_pred'],
        'abs_error': abs(preds['fusion_pred'] - float(sample['dry_weight'].item())),
        'activation': act_stats,
        'kernel': kernel_stats,
        'rgb_attention': rgb_attn_stats,
        'rgbd_attention': rgbd_attn_stats,
    }

    return (
        sample_idx,
        rgb_img,
        depth_img,
        act_img,
        kernel_img,
        journey_images,
        rgb_overlay,
        rgb_heatmap,
        rgbd_overlay,
        rgbd_heatmap,
        info,
    )


def main(cfg: ExplorerConfig):
    seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, id_lookup = build_dataset(cfg)
    model = load_model(cfg, device)

    def by_index(idx, layer, channel, kernel_branch, kernel_idx):
        return analyze_sample(int(idx), dataset, model, device, layer, int(channel), kernel_branch, int(kernel_idx))

    def by_id(sample_id, layer, channel, kernel_branch, kernel_idx):
        try:
            sid = int(float(sample_id))
        except Exception:
            sid = None
        idx = id_lookup.get(sid, 0) if sid is not None else 0
        results = analyze_sample(idx, dataset, model, device, layer, int(channel), kernel_branch, int(kernel_idx))
        return results

    idx_slider = gr.Slider(0, max(0, len(dataset) - 1), value=0, step=1, label='Dataset index')
    layer_dropdown = gr.Dropdown(choices=list(ACTIVATION_PATHS.keys()), value='Fusion Input', label='Activation Layer')
    channel_input = gr.Number(value=-1, precision=0, label='Channel (use -1 for mean)')
    kernel_branch = gr.Dropdown(choices=list(KERNEL_PATHS.keys()), value='RGB Conv1', label='Kernel Group')
    kernel_input = gr.Number(value=0, precision=0, label='Kernel index')
    id_box = gr.Textbox(label='Jump to sample ID', placeholder='Enter image ID')

    with gr.Blocks(title='model_v8 Feature Explorer') as demo:
        gr.Markdown(
            '## model_v8 Feature Explorer\n'
            'Inspect intermediate activations and first-layer kernels to debug what the model attends to.'
        )
        with gr.Row():
            idx_slider.render()
            layer_dropdown.render()
            channel_input.render()
        with gr.Row():
            kernel_branch.render()
            kernel_input.render()
            id_box.render()
        run_btn = gr.Button('Analyze current index')
        id_btn = gr.Button('Load by ID')
        random_btn = gr.Button('Random sample')

        rgb_view = gr.Image(label='RGB input', type='pil')
        depth_view = gr.Image(label='Depth input', type='pil')
        act_view = gr.Image(label='Activation map', type='pil')
        kernel_view = gr.Image(label='Kernel heatmap', type='pil')
        channel_gallery = gr.Gallery(label='Channel journey', columns=3)
        rgb_attn_view = gr.Image(label='RGB Spatial Attention Overlay', type='pil')
        rgb_attn_heat = gr.Image(label='RGB Attention Heatmap', type='pil')
        rgbd_attn_view = gr.Image(label='RGBD Spatial Attention Overlay', type='pil')
        rgbd_attn_heat = gr.Image(label='RGBD Attention Heatmap', type='pil')
        meta_view = gr.JSON(label='Details')

        run_btn.click(
            by_index,
            inputs=[idx_slider, layer_dropdown, channel_input, kernel_branch, kernel_input],
            outputs=[
                idx_slider,
                rgb_view,
                depth_view,
                act_view,
                kernel_view,
                channel_gallery,
                rgb_attn_view,
                rgb_attn_heat,
                rgbd_attn_view,
                rgbd_attn_heat,
                meta_view,
            ],
        )
        id_btn.click(
            by_id,
            inputs=[id_box, layer_dropdown, channel_input, kernel_branch, kernel_input],
            outputs=[
                idx_slider,
                rgb_view,
                depth_view,
                act_view,
                kernel_view,
                channel_gallery,
                rgb_attn_view,
                rgb_attn_heat,
                rgbd_attn_view,
                rgbd_attn_heat,
                meta_view,
            ],
        )

        def random_index(_, layer, channel, branch, kernel_idx):
            ridx = int(np.random.randint(0, max(1, len(dataset))))
            return analyze_sample(ridx, dataset, model, device, layer, int(channel), branch, int(kernel_idx))

        random_btn.click(
            random_index,
            inputs=[idx_slider, layer_dropdown, channel_input, kernel_branch, kernel_input],
            outputs=[
                idx_slider,
                rgb_view,
                depth_view,
                act_view,
                kernel_view,
                channel_gallery,
                rgb_attn_view,
                rgb_attn_heat,
                rgbd_attn_view,
                rgbd_attn_heat,
                meta_view,
            ],
        )

        # Auto-load first sample
        demo.load(
            by_index,
            inputs=[idx_slider, layer_dropdown, channel_input, kernel_branch, kernel_input],
            outputs=[
                idx_slider,
                rgb_view,
                depth_view,
                act_view,
                kernel_view,
                channel_gallery,
                rgb_attn_view,
                rgb_attn_heat,
                rgbd_attn_view,
                rgbd_attn_heat,
                meta_view,
            ],
        )

    demo.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradio feature explorer for model_v8')
    parser.add_argument('--train-csv', default=ExplorerConfig.train_csv)
    parser.add_argument('--rgb-dir', default=ExplorerConfig.rgb_dir)
    parser.add_argument('--depth-dir', default=ExplorerConfig.depth_dir)
    parser.add_argument('--checkpoint', default=ExplorerConfig.checkpoint)
    parser.add_argument('--drop-path', type=float, default=ExplorerConfig.drop_path_prob)
    parser.add_argument('--blacklist-ids', type=int, nargs='*', default=list(ExplorerConfig.blacklist_ids))
    args = parser.parse_args()

    cfg = ExplorerConfig(
        train_csv=args.train_csv,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        checkpoint=args.checkpoint,
        blacklist_ids=tuple(sorted(set(args.blacklist_ids))),
        drop_path_prob=args.drop_path,
    )
    main(cfg)
