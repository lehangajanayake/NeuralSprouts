# model_v9 — RGB + depth + surface normals

Model v9 extends the v8 SAM fusion backbone by concatenating a 3-channel normal
map (Nx, Ny, Nz) with the RGB inputs. Surface normals are derived from the depth
map using a pinhole camera assumption and Sobel gradients to estimate partial
3D derivatives. The RGB+normal branch therefore ingests six channels, while the
RGBD branch stays unchanged. Both branches share the DropPath-enabled bottleneck
stacks introduced earlier.

## Highlights

- **Depth → Normal preprocessing** (`normal_utils.py`, `preprocess.py`): produces
  `Normal_XXXX.png` files normalized to [-1, 1] (stored as 0–255 in PNG form).
- **RGBN branch**: identical to the RGB branch in v8 but configured for 6 input
  channels. Outputs an auxiliary scalar prediction for deep supervision.
- **RGBD branch + fusion MLP**: unchanged architecture, supervised alongside
  the RGBN branch with the same loss weights (0.2 / 0.2 / 0.6 by default).
- **Fallback normal generation**: loaders automatically regenerate and cache
  normals if a PNG is missing (useful for the Test set or legacy assets).

## File map

- `normal_utils.py` — shared helpers for computing normals via cross products.
- `preprocess.py` — augmentation pipeline that now saves RGB, depth, and normal
  PNGs plus the augmented CSV.
- `dataloader.py` — `PlantDatasetV9` / `TestPlantDatasetV9`, each returning
  `rgb`, `rgbn`, `rgbd`, and ids.
- `model.py` — `LettuceNormalFusionNet` (DropPath bottlenecks + SAM + fusion).
- `train.py` — grouped training loop with deep supervision.
- `eval.py` — evaluation + scatter plot + per-sample CSV export.
- `predict.py` — inference helper for the Test split (auto-builds normals if
  `datasets/Test/NormalMaps` is empty).
- `tests/test_model_v9_shapes.py` — lightweight forward-shape regression test.

## Usage

```cmd
cd models/model_v9
python preprocess.py        # builds Augmented_v9/{RGB,Depth,Normal}
python train.py             # trains LettuceNormalFusionNet (saves best_model_v9.pth)
python eval.py --checkpoint best_model_v9.pth
python predict.py --model-path best_model_v9.pth --output-csv submissions/v9.csv
```

To evaluate on the original (non-augmented) training data, point `eval.py` to
`datasets/Training/{RGBImages,DepthImages,NormalMaps}`. The normal directory can
start empty — the loader will populate it on demand from the depth maps.
