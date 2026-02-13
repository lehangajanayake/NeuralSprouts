# model_v8 — dual-branch CNN + spatial attention

## What is new?

Model v8 keeps the light dual-regression backbone from v4.2 (parallel RGB and RGBD
branches feeding a fusion MLP) but injects a **Spatial Attention Module (SAM)** after
the final convolutional stack of each branch. The SAM is the CBAM-style block that
combines channel-wise average/max projections, passes them through a 7×7 conv, and
reweights the feature map. This helps each branch focus on salient leaf regions
without increasing parameters dramatically.

## File map

- `preprocess.py` — random crop (full image → 1000 px) + random center shift (≤100 px)
  pipeline that exports aligned RGB/Depth PNG pairs to `datasets/Training/Augmented_v8`.
- `dataloader.py` — `PlantDatasetV8` / `TestPlantDatasetV8` (identical API to v4.2 but
  scoped to this version).
- `model.py` — `LettuceSAMFusionNet` (RGB branch + RGBD branch + SAM + fusion MLP).
- `train.py` — single-stage MAE training (grouped by original ids, optional folds) that
  saves `best_model_v8*.pth` checkpoints.
- `eval.py` — MAE on the original training CSV + scatter plot + per-sample CSV for
  downstream inspection.
- `predict.py` — helper that writes `Test_with_predictions_v8.csv`.
- `tests/test_model_v8_shapes.py` — sanity check for forward shapes.
- `view_worst_samples.py` — runs inference, ranks samples by MAE, and serves a
  Gradio UI for browsing the hardest cases.

## Quick start

```cmd
cd models/model_v8
python preprocess.py        # optional; reuse previous augmented assets if desired
python train.py             # saves best_model_v8.pth by default
python eval.py --checkpoint best_model_v8.pth
python predict.py --model-path best_model_v8.pth --output-csv submissions/v8.csv
```

Tune the usual knobs (`num_aug_per_image`, learning rate, folds) inside the respective
config dataclasses. The SAM block is lightweight, so GPU footprint stays close to v4.2.
