"""Gradio viewer for the worst-performing training samples.

The script runs inference on the training CSV, sorts examples by absolute error,
then launches a lightweight web UI to inspect RGB/Depth pairs with the highest
residuals.
"""

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None
from torch.utils.data import DataLoader

from dataloader import PlantDatasetV8
from model import LettuceSAMFusionNet


@dataclass
class ViewerConfig:
    train_csv: str
    rgb_dir: str
    depth_dir: str
    checkpoint: str
    batch_size: int
    top_k: int
    host: str
    port: int
    share: bool
    blacklist_ids: Tuple[int, ...]


def parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser(description='Launch a viewer for the worst MAE samples.')
    parser.add_argument('--train-csv', default='../../datasets/Training/Train.csv')
    parser.add_argument('--rgb-dir', default='../../datasets/Training/RGBImages')
    parser.add_argument('--depth-dir', default='../../datasets/Training/DepthImages')
    parser.add_argument('--checkpoint', default='best_model_v8.pth')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=100, help='How many worst samples to expose in the UI.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7870)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--blacklist-ids', type=int, nargs='*', default=[163], help='IDs to exclude entirely.')
    args = parser.parse_args()
    return ViewerConfig(
        train_csv=args.train_csv,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
        share=args.share,
        blacklist_ids=tuple(sorted(set(args.blacklist_ids))),
    )


def compute_worst_samples(cfg: ViewerConfig, device: torch.device):
    dataset = PlantDatasetV8(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=False,
        seed=42,
        enable_cache=False,
        blacklist_ids=cfg.blacklist_ids,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = LettuceSAMFusionNet().to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    entries: List[dict] = []
    mae = torch.nn.L1Loss(reduction='none')

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device)
            rgbd = batch['rgbd'].to(device)
            y = batch['dry_weight'].to(device)
            ids = batch['id'].cpu().numpy().astype(int)

            _, _, fusion_pred = model(rgb, rgbd)
            abs_err = mae(fusion_pred, y).detach().cpu().numpy()
            preds = fusion_pred.detach().cpu().numpy()
            targets = y.detach().cpu().numpy()

            for sample_id, target, pred, err in zip(ids, targets, preds, abs_err):
                entries.append(
                    {
                        'id': int(sample_id),
                        'target': float(target),
                        'prediction': float(pred),
                        'abs_error': float(abs(err)),
                    }
                )

    if not entries:
        raise ValueError('No samples evaluated; confirm CSV paths and preprocessing assets.')

    entries.sort(key=lambda x: x['abs_error'], reverse=True)
    return entries[: min(cfg.top_k, len(entries))]


def build_viewer(entries: List[dict], cfg: ViewerConfig):
    total = len(entries)
    if total == 0:
        raise ValueError('No entries to visualize.')
    id_to_idx = {int(e['id']): idx for idx, e in enumerate(entries)}

    def make_scatter():
        if go is None:
            return None
        targets = [e['target'] for e in entries]
        preds = [e['prediction'] for e in entries]
        errors = [e['abs_error'] for e in entries]
        line_min = float(min(targets + preds))
        line_max = float(max(targets + preds))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=targets,
                y=preds,
                mode='markers',
                marker={
                    'color': errors,
                    'colorscale': 'Turbo',
                    'showscale': True,
                    'size': 8,
                },
                text=[f"ID {e['id']}" for e in entries],
                customdata=list(range(total)),
                hovertemplate='ID %{text}<br>Target=%{x:.2f}<br>Pred=%{y:.2f}<br>Abs err=%{marker.color:.2f}<extra></extra>',
                name='Samples',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[line_min, line_max],
                y=[line_min, line_max],
                mode='lines',
                line={'color': 'red', 'dash': 'dash'},
                name='Ideal',
            )
        )
        fig.update_layout(
            title='Prediction Scatter (click point to inspect)',
            xaxis_title='Ground truth DryWeightShoot',
            yaxis_title='Predicted DryWeightShoot',
            template='plotly_white',
        )
        return fig

    def _clamp_index(idx: int) -> int:
        return max(0, min(int(idx), total - 1))

    def load_depth_rgb(sample_id: int):
        rgb_path = os.path.join(cfg.rgb_dir, f'RGB_{sample_id}.png')
        depth_path = os.path.join(cfg.depth_dir, f'Depth_{sample_id}.png')
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        depth = ImageOps.autocontrast(depth).convert('RGB')
        return rgb, depth

    def fetch(idx: float):
        idx = _clamp_index(idx)
        entry = entries[idx]
        rgb_img, depth_img = load_depth_rgb(entry['id'])
        meta = {
            'rank': idx + 1,
            'id': entry['id'],
            'target': entry['target'],
            'prediction': entry['prediction'],
            'abs_error': entry['abs_error'],
        }
        return idx, rgb_img, depth_img, meta

    def random_idx():
        return random.randint(0, total - 1)

    def fetch_by_id(sample_id: str):
        if sample_id is None or str(sample_id).strip() == '':
            return fetch(0)
        try:
            sid = int(float(sample_id))
        except ValueError:
            idx, rgb_img, depth_img, meta = fetch(0)
            meta['note'] = f'Invalid ID "{sample_id}"; showing worst sample.'
            return idx, rgb_img, depth_img, meta
        idx_from_id = id_to_idx.get(sid)
        if idx_from_id is None:
            idx, rgb_img, depth_img, meta = fetch(0)
            meta['note'] = f'ID {sid} not in top {total}; showing worst sample.'
            return idx, rgb_img, depth_img, meta
        return fetch(idx_from_id)

    with gr.Blocks(title='Worst Performing Samples — model_v8') as demo:
        gr.Markdown(
            '## model_v8 — Worst Performing Samples\n'
            'Use the slider to inspect samples with the largest absolute errors.'
        )
        scatter_fig = make_scatter()
        if scatter_fig is not None:
            plot = gr.Plot(value=scatter_fig, label='Predictions vs Targets')
        slider = gr.Slider(0, total - 1, step=1, value=0, label='Rank (0 = worst)')
        rand_btn = gr.Button('Jump to random')
        with gr.Row():
            id_box = gr.Textbox(label='Jump to ID', placeholder='Enter image ID (e.g., 1234)')
            go_btn = gr.Button('Go')
        with gr.Row():
            rgb_view = gr.Image(label='RGB', type='pil')
            depth_view = gr.Image(label='Depth', type='pil')
        meta_view = gr.JSON(label='Details')

        slider.change(fetch, inputs=slider, outputs=[slider, rgb_view, depth_view, meta_view])
        rand_btn.click(random_idx, outputs=slider)
        go_btn.click(fetch_by_id, inputs=id_box, outputs=[slider, rgb_view, depth_view, meta_view])
        demo.load(fetch, inputs=slider, outputs=[slider, rgb_view, depth_view, meta_view])
        if scatter_fig is not None and hasattr(plot, 'select'):
            def on_plot_select(evt: gr.SelectData | None):
                if evt is None or evt.index is None:
                    return fetch(0)
                idx = evt.index
                if isinstance(idx, (list, tuple)):
                    idx = idx[0]
                custom = None
                if hasattr(evt, 'data') and isinstance(evt.data, dict):
                    custom = evt.data.get('customdata')
                if custom is None and hasattr(evt, 'value') and isinstance(evt.value, dict):
                    custom = evt.value.get('customdata')
                if isinstance(custom, (list, tuple)) and custom:
                    idx = custom[0]
                return fetch(idx)

            plot.select(on_plot_select, outputs=[slider, rgb_view, depth_view, meta_view])
        elif scatter_fig is not None:
            gr.Markdown('**Note:** Gradio version does not support clicking the scatter plot; use the slider or random button to inspect samples.')

    return demo


def main():
    cfg = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing residuals ...')
    entries = compute_worst_samples(cfg, device)
    print(f'Top {len(entries)} samples ready. Launching viewer...')
    demo = build_viewer(entries, cfg)
    demo.launch(server_name=cfg.host, server_port=cfg.port, share=cfg.share)


if __name__ == '__main__':
    main()
