import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataloader import SimplePlantDataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = (x * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def map_index(base_len: int, num_views: int, global_idx: int) -> Tuple[int, int]:
    if num_views > 1:
        base_idx = global_idx % base_len
        view_idx = global_idx // base_len
        return int(base_idx), int(view_idx)
    return int(global_idx), 0


@st.cache_data(show_spinner=False)
def _read_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def _build_dataset(rgb_dir: str, csv_path: str, image_size: int) -> SimplePlantDataset:
    # Enable debug returns so the app can show original vs transformed views.
    return SimplePlantDataset(rgb_dir, csv_path, image_size=image_size, return_debug=True)


def build_cache(ds: SimplePlantDataset, num_views: int, seed: int, max_items: int | None):
    ds.num_views = int(num_views)
    ds.cache_seed = int(seed)
    ds.enable_cache = True
    ds.build_cache(max_items=max_items)


def main():
    st.set_page_config(page_title='NeuralSprouts v3 — Dataloader Debugger', layout='wide')

    st.title('NeuralSprouts — model_v3 dataloader debugger')
    st.caption('Grid + single view visualizer for cached multi-view augmentation (Option B).')

    # Navigation state (separate from widget-bound keys).
    if '__nav_idx_state' not in st.session_state:
        st.session_state['__nav_idx_state'] = 0
    if '__page_start_state' not in st.session_state:
        st.session_state['__page_start_state'] = 0

    def _nav_prev(total_len: int):
        st.session_state['__nav_idx_state'] = max(0, int(st.session_state['__nav_idx_state']) - 1)

    def _nav_next(total_len: int):
        st.session_state['__nav_idx_state'] = min(total_len - 1, int(st.session_state['__nav_idx_state']) + 1)

    def _nav_random(total_len: int):
        st.session_state['__nav_idx_state'] = int(torch.randint(0, total_len, (1,)).item())

    def _page_prev(total_len: int, count: int):
        st.session_state['__page_start_state'] = max(0, int(st.session_state['__page_start_state']) - int(count))

    def _page_next(total_len: int, count: int):
        st.session_state['__page_start_state'] = min(total_len - 1, int(st.session_state['__page_start_state']) + int(count))

    def _page_random(total_len: int):
        st.session_state['__page_start_state'] = int(torch.randint(0, total_len, (1,)).item())

    with st.sidebar:
        st.header('Dataset')
        csv_path = st.text_input('CSV path', value='../../datasets/Training/Train.csv')
        rgb_dir = st.text_input('RGB dir', value='../../datasets/Training/RGBImages/')
        image_size = st.number_input('Image size', min_value=32, max_value=1024, value=224, step=32)

        st.divider()
        st.header('Option B (views)')
        num_views = st.number_input('num_views (K)', min_value=1, max_value=32, value=4, step=1)
        cache_seed = st.number_input('cache_seed', min_value=0, max_value=2_000_000_000, value=42, step=1)
        max_items = st.number_input('cache max_items (base images)', min_value=0, value=0, step=50,
                                   help='0 means: cache ALL base images.')

        st.divider()
        st.header('View')
        mode = st.radio('Mode', options=['Grid', 'Single'], index=0)
        if mode == 'Grid':
            grid_count = st.number_input('Grid count', min_value=1, max_value=64, value=12, step=1)
            start_index = st.number_input(
                'Start global index',
                min_value=0,
                value=int(st.session_state.get('__page_start_state', 0)),
                step=1,
                key='__page_start',
            )
        else:
            grid_count = 1
            start_index = st.number_input(
                'Global index',
                min_value=0,
                value=int(st.session_state.get('__nav_idx_state', 0)),
                step=1,
                key='__nav_idx',
            )

        st.divider()
        build_cache_btn = st.button('Build/Refresh cache')
        clear_cache_btn = st.button('Clear cache')

    if not os.path.exists(csv_path):
        st.error(f'CSV not found: {csv_path}')
        st.stop()
    if not os.path.isdir(rgb_dir):
        st.error(f'RGB dir not found: {rgb_dir}')
        st.stop()

    # Load dataset (resource cached, fast to re-open)
    ds = _build_dataset(rgb_dir, csv_path, int(image_size))

    # Apply view settings
    ds.num_views = int(num_views)
    ds.cache_seed = int(cache_seed)

    # Cache controls
    if clear_cache_btn:
        ds.enable_cache = False
        ds._cache.clear()
        st.success('Cleared cache (in-memory).')

    if build_cache_btn:
        ds._cache.clear()
        mi = None if int(max_items) == 0 else int(max_items)
        with st.spinner('Building cache...'):
            build_cache(ds, num_views=int(num_views), seed=int(cache_seed), max_items=mi)
        st.success(f'Cache built: {len(ds._cache)} tensors')

    base_len = len(ds.df)
    total_len = len(ds)

    col1, col2, col3 = st.columns(3)
    col1.metric('Base images (N)', base_len)
    col2.metric('num_views (K)', int(num_views))
    col3.metric('Expanded samples (N*K)', total_len)
    st.write(f'Cache enabled: **{ds.enable_cache}** | Cached tensors: **{len(ds._cache)}**')

    # Bounds (do NOT write back into st.session_state keys that are bound to widgets;
    # Streamlit forbids mutating those after instantiation.)
    start = int(start_index)
    if start < 0:
        start = 0
    if start >= total_len:
        st.warning(f'Index out of range; clamping to {total_len - 1}')
        start = max(0, total_len - 1)

    # Keep our internal navigation state in sync with widget values.
    if mode == 'Single':
        st.session_state['__nav_idx_state'] = start
    else:
        st.session_state['__page_start_state'] = start

    count = int(grid_count)
    count = max(1, min(count, total_len - start))

    # Render
    if mode == 'Single':
        idx = int(st.session_state.get('__nav_idx_state', start))
        if idx < 0:
            idx = 0
        if idx >= total_len:
            idx = total_len - 1
        st.session_state['__nav_idx_state'] = idx
        base_idx, view_idx = map_index(base_len, int(num_views), idx)
        x_aug, x_orig, y, meta = ds[idx]
        img_id = int(meta['id'])
        cached = bool(meta.get('cached', False))
        st.subheader(
            f"global={idx} → base={base_idx}, view={view_idx} | id={img_id} | y={float(y):.4f} | cached={cached}"
        )

        left, right = st.columns(2)
        with left:
            st.markdown('**Original (no random aug)**')
            st.image(denormalize(x_orig), use_container_width=True)
        with right:
            st.markdown('**Transformed (view)**')
            st.image(denormalize(x_aug), use_container_width=True)

        with st.expander('Metadata', expanded=False):
            st.json({
                **meta,
                'cache_seed': int(cache_seed),
            })

        nav1, nav2, nav3, nav4 = st.columns(4)
        # IMPORTANT: don't write to __nav_idx (widget key) directly; update our state and rerun.
        nav1.button('◀ Prev', on_click=_nav_prev, args=(total_len,))
        nav2.button('Next ▶', on_click=_nav_next, args=(total_len,))
        nav3.button('Random 🎲', on_click=_nav_random, args=(total_len,))
        if nav4.button('Sync to typed index'):
            # Start index widget already updated `start`; we synced it into __nav_idx_state above.
            st.rerun()

    else:
        start = int(st.session_state.get('__page_start_state', start))
        if start < 0:
            start = 0
        if start >= total_len:
            start = max(0, total_len - 1)
        st.session_state['__page_start_state'] = start

        count = int(grid_count)
        count = max(1, min(count, total_len - start))

        st.subheader(f'Grid view: global indices [{start}, {start + count - 1}]')
        cols = 4
        rows = int(np.ceil(count / cols))

        k = 0
        for r in range(rows):
            row_cols = st.columns(cols)
            for c in range(cols):
                if k >= count:
                    break
                idx = start + k
                base_idx, view_idx = map_index(base_len, int(num_views), idx)
                x_aug, x_orig, y, meta = ds[idx]
                img_id = int(meta['id'])
                cached = bool(meta.get('cached', False))

                with row_cols[c]:
                    # Show transformed view; hover/caption includes that original exists.
                    st.image(denormalize(x_aug), use_container_width=True)
                    st.caption(f'g={idx} b={base_idx} v={view_idx}\nid={img_id} y={float(y):.3f} cache={cached}')
                k += 1

        nav1, nav2, nav3 = st.columns(3)
        # IMPORTANT: don't write to __page_start (widget key) directly; update our state and rerun.
        nav1.button('◀ Prev page', on_click=_page_prev, args=(total_len, count))
        nav2.button('Next page ▶', on_click=_page_next, args=(total_len, count))
        nav3.button('Random page 🎲', on_click=_page_random, args=(total_len,))


if __name__ == '__main__':
    main()
