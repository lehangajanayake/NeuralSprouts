"""Web-based viewer for inspecting augmented RGB/Depth pairs.

Run:
    python view_augmented.py --rgb-dir <path> --depth-dir <path> --csv <path>
Requires gradio>=4.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    from preprocess import PreprocessConfig
except Exception:  # pragma: no cover
    PreprocessConfig = None


@dataclass
class ViewerPaths:
    rgb_dir: str
    depth_dir: str
    csv_path: str


class AugmentedDataset:
    """Helper class that keeps metadata/indexing fast for the viewer."""

    def __init__(self, paths: ViewerPaths) -> None:
        self.paths = paths
        self.df = self._load_dataframe(paths.csv_path)
        self.ids: List[int] = self.df['id'].tolist()
        self.rows: Dict[int, Dict] = {
            int(row['id']): row for row in self.df.to_dict(orient='records')
        }
        if not self.ids:
            raise ValueError('No entries found in augmented CSV')

    @staticmethod
    def _load_dataframe(csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'CSV not found: {csv_path}')
        df = pd.read_csv(csv_path)
        if 'image_id' in df.columns and 'id' not in df.columns:
            df = df.rename(columns={'image_id': 'id'})
        if 'id' not in df.columns:
            raise ValueError('CSV must contain an "id" column')
        df['id'] = df['id'].astype(int)
        df = df.sort_values('id').reset_index(drop=True)
        return df

    def ensure_index(self, index: int) -> int:
        return max(0, min(index, len(self.ids) - 1))

    def index_of(self, image_id: int) -> Optional[int]:
        try:
            return self.ids.index(image_id)
        except ValueError:
            return None

    def metadata(self, image_id: int) -> Dict:
        row = self.rows.get(image_id, {})
        return {k: v for k, v in row.items() if k not in {'id'}}

    def rgb_path(self, image_id: int) -> str:
        return os.path.join(self.paths.rgb_dir, f'RGB_{image_id}.png')

    def depth_path(self, image_id: int) -> str:
        return os.path.join(self.paths.depth_dir, f'Depth_{image_id}.png')

    def load_rgb(self, image_id: int) -> Image.Image:
        path = self.rgb_path(image_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f'RGB image missing: {path}')
        return Image.open(path).convert('RGB')

    def load_depth(self, image_id: int) -> Image.Image:
        path = self.depth_path(image_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Depth image missing: {path}')
        depth = Image.open(path).convert('L')
        depth = ImageOps.autocontrast(depth)
        return depth.convert('RGB')


def parse_args() -> argparse.Namespace:
    default_rgb = PreprocessConfig().out_rgb_dir if PreprocessConfig else '../../datasets/Training/Augmented/RGBImages'
    default_depth = PreprocessConfig().out_depth_dir if PreprocessConfig else '../../datasets/Training/Augmented/DepthImages'
    default_csv = PreprocessConfig().out_csv if PreprocessConfig else '../../datasets/Training/Augmented/Train_aug.csv'

    parser = argparse.ArgumentParser(description='Launch a Gradio viewer for augmented RGB/Depth pairs.')
    parser.add_argument('--rgb-dir', default=default_rgb, help='Directory containing RGB_*.png files.')
    parser.add_argument('--depth-dir', default=default_depth, help='Directory containing Depth_*.png files.')
    parser.add_argument('--csv', default=default_csv, help='CSV with augmented metadata (must include id column).')
    parser.add_argument('--host', default='127.0.0.1', help='Listener host for the Gradio server.')
    parser.add_argument('--port', type=int, default=7860, help='Listener port for the Gradio server.')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing link.')
    return parser.parse_args()


def build_interface(dataset: AugmentedDataset) -> gr.Blocks:
    total = len(dataset.ids)

    def fetch_by_index(index: float) -> Tuple[int, Image.Image, Image.Image, Dict]:
        index = dataset.ensure_index(int(index))
        image_id = dataset.ids[index]
        rgb = dataset.load_rgb(image_id)
        depth = dataset.load_depth(image_id)
        meta = dataset.metadata(image_id)
        meta.update({'id': image_id})
        return image_id, rgb, depth, meta

    def go_to_id(image_id_str: str) -> int:
        try:
            image_id = int(image_id_str)
        except Exception:
            return 0
        idx = dataset.index_of(image_id)
        return idx if idx is not None else 0

    def random_index(_: Optional[str] = None) -> int:
        return random.randint(0, total - 1)

    with gr.Blocks(title='Augmented RGB/Depth Viewer') as demo:
        gr.Markdown('## Augmented RGB/Depth Viewer\nUse the controls below to scrub through generated samples and verify preprocessing quality.')
        with gr.Row():
            slider = gr.Slider(minimum=0, maximum=total - 1, step=1, value=0, label='Sample index')
            id_display = gr.Number(label='Augmented id', interactive=False)
        with gr.Row():
            id_input = gr.Textbox(label='Jump to augmented id (e.g., 1523)', placeholder='Enter exact augmented id...')
            go_btn = gr.Button('Go')
            rand_btn = gr.Button('Random sample')
        with gr.Row():
            rgb_view = gr.Image(label='RGB view', type='pil')
            depth_view = gr.Image(label='Depth view', type='pil')
        meta_view = gr.JSON(label='Metadata / mixup info')

        slider.change(fn=fetch_by_index, inputs=slider, outputs=[id_display, rgb_view, depth_view, meta_view])
        go_btn.click(fn=go_to_id, inputs=id_input, outputs=slider)
        rand_btn.click(fn=random_index, outputs=slider)

        # Trigger initial render
        demo.load(fn=fetch_by_index, inputs=slider, outputs=[id_display, rgb_view, depth_view, meta_view])

    return demo


def main() -> None:
    args = parse_args()
    paths = ViewerPaths(rgb_dir=args.rgb_dir, depth_dir=args.depth_dir, csv_path=args.csv)
    dataset = AugmentedDataset(paths)
    interface = build_interface(dataset)
    interface.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
