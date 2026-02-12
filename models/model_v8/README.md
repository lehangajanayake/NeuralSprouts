# model_v8 — spatial-attention dual branch with deep supervision

Model v8 is the first upgrade over the v4.2 baseline. It keeps the dual RGB/RGBD
design but swaps the plain ConvBlocks for bottleneck residual stacks with
**stochastic depth (DropPath)** and adds a CBAM-style **Spatial Attention Module
(SAM)** after each branch. Training now applies **deep supervision** (MAE on both
branch heads + the fusion head), improving gradient flow and regularisation.

## Highlights

- Bottleneck blocks (1×1→3×3→1×1) with per-layer DropPath schedules for better
  optimization at depth.
- SAM (7×7 convolution on concatenated avg/max channel projections) focuses each
  branch on the most salient plant regions without extra parameters.
- Deep supervision with configurable weights (`RGB_LOSS_WEIGHT = 0.2`,
  `RGBD_LOSS_WEIGHT = 0.2`, `FUSION_LOSS_WEIGHT = 0.6`).
- Updated evaluation tooling: scatter-plot + per-sample CSV exporter and a
  Gradio-based "worst sample" browser.

## File map

- `preprocess.py` — same random crop + ±100 px shift augmentation as v4.2, but
  outputs to `datasets/Training/Augmented_v8` (separate from earlier versions).
- `dataloader.py` — `PlantDatasetV8` / `TestPlantDatasetV8`, returning RGB
  tensors, RGBD tensors, ids, and targets.
- `model.py` — `LettuceSAMFusionNet` with DropPath-enabled bottlenecks, SAM,
  adaptive pooling, and a fusion MLP.
- `train.py` — grouped single-stage MAE training with deep supervision and
  optional GPU preloading; checkpoints are saved as `best_model_v8*.pth`.
- `eval.py` — evaluates on the base Training CSV, computes MAE, exports scatter
  plots and CSVs sorted by absolute error.
- `predict.py` — inference helper writing `Test_with_predictions_v8.csv`.
- `view_worst_samples.py` — ranks predictions by error and serves a Gradio UI
  for qualitative inspection (RGB vs RGBD side-by-side).
- `tests/test_model_v8_shapes.py` — fast forward-shape regression test.

## Quick start

```cmd
cd models/model_v8
python preprocess.py        # optional; reuse existing Augmented_v8 assets
python train.py             # saves best_model_v8.pth by default
python eval.py --checkpoint best_model_v8.pth
python predict.py --model-path best_model_v8.pth --output-csv submissions/v8.csv
python view_worst_samples.py --checkpoint best_model_v8.pth
```

Tune the dataclass knobs (`num_aug_per_image`, folds, loss weights, etc.) to fit
your compute budget. Despite the new attention + residual layers, footprint and
throughput stay close to v4.2 thanks to the lightweight 96×96 resolution.
