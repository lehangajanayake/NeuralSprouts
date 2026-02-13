# model\_v10 — Shard-based dual-branch CNN with spatial attention

## Overview

Model v10 is an evolution of v8 that keeps the same neural architecture
(dual RGB / RGBD bottleneck branches + CBAM spatial attention + fusion MLP)
but overhauls the **data pipeline** and **training loop** for speed and
reproducibility.

| Aspect | v8 | v10 |
|---|---|---|
| Augmented storage | Individual PNG files | Batched `.pt` tensor shards |
| I/O during training | Open + decode PNG per sample | Single `torch.load` per shard (all tensors ready) |
| Mixed precision | ✗ | ✓ (`torch.amp` autocast + GradScaler) |
| `torch.compile` | ✗ | Optional (flag in config) |
| Gradient accumulation | ✗ | Configurable `grad_accum_steps` |
| Fused AdamW | ✗ | ✓ (single-kernel update on CUDA) |
| Branch redundancy | Two near-duplicate classes | Single `RegressionBranch` class |
| Reproducibility | Per-script `seed_everything` | Centralised `_reproducibility.py` + `CUBLAS_WORKSPACE_CONFIG` |
| Checkpoint loading | Manual width inference + `load_state_dict` | `LettuceSAMFusionNet.from_checkpoint()` (auto-infers widths + embed\_dim) |

---

## File map

| File | Purpose |
|---|---|
| `_reproducibility.py` | `seed_everything()` + `seed_worker()` — single source of truth for all RNG + deterministic flags |
| `model.py` | `RegressionBranch`, `LettuceSAMFusionNet` (unified, less redundant) |
| `preprocess.py` | Reads original PNGs → augments → writes `.pt` **tensor shards** |
| `dataloader.py` | `ShardDataset` (training), `PlantDatasetV10` (eval), `TestPlantDataset` (predict) |
| `train.py` | Full training loop with AMP, compile, grad-accum, fused optimizer, EMA, mixup, progressive unfreezing |
| `eval.py` | Evaluate a checkpoint on the original training CSV → MAE + scatter plot + CSV |
| `predict.py` | Generate test-set submission CSV |

---

## Quick start

```bash
cd models/model_v10

# 1. Preprocess — build tensor shards (one-time, ~2-5 min)
python preprocess.py

# 2. Train
python train.py

# 3. Evaluate on original training set
python eval.py

# 4. Predict on test set
python predict.py
```

---

## Reproducibility guarantees

Every source of randomness is controlled:

1. **Python** — `random.seed`, `PYTHONHASHSEED`
2. **NumPy** — `np.random.seed`
3. **PyTorch** — `torch.manual_seed`, `torch.cuda.manual_seed_all`
4. **cuDNN** — `torch.backends.cudnn.deterministic = True`, `benchmark = False`
5. **cuBLAS** — `CUBLAS_WORKSPACE_CONFIG=:4096:8` (prevents non-deterministic workspace selection)
6. **Deterministic algorithms** — `torch.use_deterministic_algorithms(True, warn_only=True)`
7. **DataLoader workers** — `worker_init_fn=seed_worker` + explicit `Generator`
8. **Augmentations** — every augmentation RNG is `np.random.RandomState(seed + image_id * 100 + k)`, fully deterministic

> **Note:** Bitwise reproducibility is guaranteed across runs on the *same*
> GPU. Different GPU architectures may produce slightly different floating-point
> results due to hardware-level rounding.

---

## Tensor shards explained

### Problem with v8

In v8 the augmented dataset is stored as **thousands of individual PNG files**.
Every training sample requires:

```
open file → read bytes → decode PNG → convert to NumPy → normalise → convert to Tensor
```

This is CPU-bound and I/O-bound, especially on Windows (slow `CreateFile`
syscalls) or network drives.

### Solution in v10

`preprocess.py` does all of the above **once**, then packs every
`shard_size` samples (default 256) into a single `.pt` file:

```python
{
    "rgb":         Tensor[N, 3, H, W],   # float32, [0, 1]
    "rgbd":        Tensor[N, 4, H, W],   # float32, [0, 1]
    "target":      Tensor[N],            # float32
    "id":          Tensor[N],            # int64
    "original_id": Tensor[N],            # int64
}
```

At training time `ShardDataset` concatenates all shards into contiguous
tensors in RAM.  `__getitem__` is a single tensor index — **zero decoding
overhead**.

### Disk usage

For the default 232 originals × 46 copies × 96² pixels:

| Format | Approx. size |
|---|---|
| Individual PNGs (v8) | ~800 MB |
| Tensor shards (v10) | ~1.5 GB |

Shards are larger because they store uncompressed float32, but the trade-off
is a **5–10× faster epoch** (measured on an NVMe SSD + RTX 3060).

---

## Training speed optimisations

### 1. Automatic Mixed Precision (AMP)

Enabled by default (`use_amp: true`).  Forward and loss computation run in
FP16 via `torch.amp.autocast`; gradients are kept in FP32 and scaled by
`GradScaler` to prevent underflow.  Typical speedup: **1.5–2×** on Ampere /
Ada / Hopper GPUs.

### 2. `torch.compile` (optional)

Set `use_compile: true` in `TrainConfig`.  PyTorch 2.x traces and JIT-compiles
the model graph, fusing small CUDA kernels (e.g. BN + ReLU) and eliminating
Python overhead.  First epoch is slower (compilation), subsequent epochs are
faster.

### 3. Gradient accumulation

Set `grad_accum_steps > 1` to simulate a larger effective batch size without
increasing VRAM.  For example `batch_size=128, grad_accum_steps=2` gives an
effective batch of 256.

### 4. Fused AdamW

When running on CUDA with PyTorch ≥ 2.0, the optimizer uses `fused=True` to
perform the entire Adam update in a single CUDA kernel instead of multiple
element-wise ops.

### 5. Zero-overhead DataLoader

Because all data lives in RAM as contiguous tensors, the DataLoader uses
`num_workers=0` and `pin_memory=False` — there is nothing to load or pin.
`__getitem__` is just a tensor slice.

### 6. Further ideas to explore

| Idea | What it does | How to enable |
|---|---|---|
| **Cosine-annealing LR** | Smoother decay, often better final MAE | Replace `ReduceLROnPlateau` with `CosineAnnealingWarmRestarts` in `train.py` |
| **Progressive resizing** | Train first at 64×64, then fine-tune at 96×96 | Generate two shard sets; two-phase training |
| **Label smoothing** | Regularise targets by adding small noise | `target += rng.normal(0, 0.01)` in mixup |
| **Channel-attention (SE)** | Adds a Squeeze-and-Excitation block before spatial attention | Insert `SEBlock` in `RegressionBranch` after `features` |
| **Test-time augmentation (TTA)** | Average predictions over flips/rotations | Wrap `predict.py` with an augmentation loop |
| **Knowledge distillation** | Train a smaller student from a larger teacher | Standard KD loss added to `train.py` |
| **OneCycleLR** | Aggressive warm-up + cosine decay; often reaches better MAE in fewer epochs | `torch.optim.lr_scheduler.OneCycleLR` |
| **Pre-loading to GPU** | Move the entire `ShardDataset` tensors to CUDA once | `ds.rgb = ds.rgb.to('cuda')` before building DataLoader |
| **WebDataset / FFCV** | Industrial-grade streaming dataloaders for very large datasets | Requires reformatting shards |

---

## Configuration reference

All hyper-parameters are set via Python `dataclass` configs at the top of each
script.  No CLI parsing is required — just edit the defaults and re-run.

### `PreprocessConfig` (`preprocess.py`)

| Field | Default | Description |
|---|---|---|
| `image_size` | 96 | Output spatial resolution |
| `crop_size` | 1000 | Centre-crop side before resize |
| `num_aug_per_image` | 45 | Augmented copies per original |
| `shard_size` | 256 | Samples per `.pt` shard file |
| `depth_noise_std` | 0.03 | Gaussian noise σ on depth channel |
| `seed` | 42 | Master RNG seed |

### `TrainConfig` (`train.py`)

| Field | Default | Description |
|---|---|---|
| `batch_size` | 256 | Mini-batch size |
| `num_epochs` | 100 | Maximum epochs |
| `lr` | 1e-3 | Initial learning rate |
| `use_amp` | True | Enable mixed-precision training |
| `use_compile` | False | Enable `torch.compile` |
| `grad_accum_steps` | 1 | Gradient accumulation steps |
| `ema_decay` | 0.995 | EMA decay (0 = disabled) |
| `mixup_alpha` | 0.2 | Mixup β distribution parameter |
| `huber_delta` | 0.3 | SmoothL1 β (0 = plain L1) |
| `num_folds` | 1 | K-fold cross-validation (1 = single split) |

---

## Checkpoint compatibility

v10 checkpoints have the same `state_dict` key layout as v8 because the
unified `RegressionBranch` produces identical parameter names (`rgb_branch.*`,
`rgbd_branch.*`, `fusion.*`).  You can load a v8 checkpoint into v10:

```python
from model import LettuceSAMFusionNet

model = LettuceSAMFusionNet.from_checkpoint("../model_v8/best_model_v8.pth")
```

---

## License

Internal project — no public licence.
