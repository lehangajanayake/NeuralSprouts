"""Training script for model_v10.

Key improvements over v8
-------------------------
* **Tensor-shard dataloader** — no more per-image PNG reads during training.
* **Automatic Mixed Precision (AMP)** — ``torch.amp.autocast`` + ``GradScaler``
  for ~1.5–2× throughput on GPUs with Tensor Cores (Ampere, Ada, …).
* **``torch.compile``** — optional (PyTorch ≥ 2.0) graph-level optimisation that
  can fuse kernels and eliminate overhead.
* **Gradient accumulation** — train with an effective batch size larger than what
  fits in VRAM.
* **Fused AdamW** — uses ``fused=True`` when CUDA is available (PyTorch ≥ 2.0)
  for a single-kernel parameter update.
* **Centralised reproducibility** via ``_reproducibility.seed_everything``.
* All v8 training features are preserved: EMA, Huber loss, mixup, progressive
  block unfreezing with warmup scaling, cross-validation folds, early stopping.
"""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from _reproducibility import seed_everything, seed_worker
from dataloader import ShardDataset
from model import LettuceSAMFusionNet

# Branch loss weights (unchanged from v8)
RGB_LOSS_WEIGHT = 0.2
RGBD_LOSS_WEIGHT = 0.3
FUSION_LOSS_WEIGHT = 0.5

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyper-parameters and paths for a training run."""

    # ---- data paths -------------------------------------------------------
    shard_dir: str = "../../datasets/Training/Shards_v10"
    manifest_csv: str = "../../datasets/Training/Shards_v10/manifest.csv"

    # ---- training ---------------------------------------------------------
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_min_lr: float = 1e-6

    # ---- split / fold -----------------------------------------------------
    val_ratio: float = 0.2
    seed: int = 43
    patience: int = 100
    num_folds: int = 1
    group_by_original: bool = True

    # ---- architecture -----------------------------------------------------
    drop_path_prob: float = 0.1
    rgb_widths: Tuple[int, ...] = (32, 64, 96, 128)
    rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128)
    embed_dim: int = 256

    # ---- regularisation ---------------------------------------------------
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    huber_delta: float = 0.3
    ema_decay: float = 0.995

    # ---- progressive unfreezing -------------------------------------------
    initial_frozen_rgb_blocks: int = 3
    initial_frozen_rgbd_blocks: int = 3
    unfreeze_start_epoch: int = 7
    rgb_unfreeze_interval: int = 5
    rgbd_unfreeze_interval: int = 7
    branch_warmup_epochs: int = 2
    branch_warmup_scale: float = 0.3

    # ---- speed / hardware -------------------------------------------------
    use_amp: bool = True
    use_compile: bool = False
    grad_accum_steps: int = 1

    # ---- output -----------------------------------------------------------
    out_dir: str = "."
    blacklist_ids: Tuple[int, ...] = (163,)
    best_mae_window: int = 5


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def _maybe_mixup(
    rgb: torch.Tensor,
    rgbd: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or prob <= 0.0:
        return rgb, rgbd, y
    if torch.rand(1, device=rgb.device).item() > prob:
        return rgb, rgbd, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(rgb.size(0), device=rgb.device)
    return (
        lam * rgb + (1 - lam) * rgb[perm],
        lam * rgbd + (1 - lam) * rgbd[perm],
        lam * y + (1 - lam) * y[perm],
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(mp, alpha=1.0 - decay)
        for eb, mb in zip(ema.buffers(), model.buffers()):
            eb.copy_(mb)


# ---------------------------------------------------------------------------
# Block freezing / unfreezing / warmup (identical logic to v8)
# ---------------------------------------------------------------------------

def _apply_freezing(blocks: Sequence[nn.Module], frozen: int) -> None:
    frozen = max(0, min(int(frozen), len(blocks)))
    for i, b in enumerate(blocks):
        req = i >= frozen
        for p in b.parameters():
            p.requires_grad = req


def _maybe_unfreeze(
    epoch: int,
    start: int,
    interval: int,
    frozen: int,
    blocks: Sequence[nn.Module],
    name: str,
    on_unfreeze: Optional[Callable[[int], None]] = None,
) -> int:
    if frozen <= 0:
        return frozen
    if (epoch + 1) < max(1, start):
        return frozen
    elapsed = (epoch + 1) - max(1, start)
    trigger = elapsed >= 0 and (elapsed % max(1, interval) == 0)
    if not trigger:
        return frozen
    new = max(0, frozen - 1)
    if new == frozen:
        return frozen
    _apply_freezing(blocks, new)
    print(f"  ↳ unfreezing {name} block {new}")
    if on_unfreeze:
        on_unfreeze(new)
    return new


_WarmupTracker = Dict[Tuple[str, int], Dict[str, int]]


def _start_warmup(tracker: _WarmupTracker, branch: str, idx: int, epochs: int) -> None:
    if epochs > 0:
        tracker[(branch, idx)] = {"remaining": epochs, "total": epochs}


def _scale_warmup_grads(
    tracker: _WarmupTracker,
    blocks_map: Dict[str, Sequence[nn.Module]],
    min_scale: float,
) -> None:
    for (branch, idx), state in tracker.items():
        rem = state.get("remaining", 0)
        if rem <= 0:
            continue
        seq = blocks_map.get(branch)
        if seq is None or idx < 0 or idx >= len(seq):
            continue
        progress = 1.0 - rem / max(1, state.get("total", 1))
        scale = min_scale + (1.0 - min_scale) * progress
        for p in seq[idx].parameters():
            if p.grad is not None:
                p.grad.mul_(scale)


def _tick_warmups(tracker: _WarmupTracker) -> None:
    to_del = []
    for key, state in tracker.items():
        state["remaining"] -= 1
        if state["remaining"] <= 0:
            to_del.append(key)
    for k in to_del:
        tracker.pop(k, None)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def _split_by_group(
    original_ids: np.ndarray,
    cfg: TrainConfig,
) -> List[Tuple[List[int], List[int]]]:
    """Group-aware train/val split (or K-fold) preserving original_id boundaries."""
    unique = np.unique(original_ids)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(unique)
    n_folds = max(1, min(cfg.num_folds, len(unique)))

    splits: List[Tuple[List[int], List[int]]] = []
    if n_folds > 1:
        fold_groups = np.array_split(unique, n_folds)
        for val_groups in fold_groups:
            val_set = set(int(g) for g in val_groups)
            train_idx = [i for i, g in enumerate(original_ids) if g not in val_set]
            val_idx = [i for i, g in enumerate(original_ids) if g in val_set]
            if train_idx and val_idx:
                splits.append((train_idx, val_idx))
    else:
        n_val = max(1, int(round(len(unique) * cfg.val_ratio)))
        val_set = set(int(g) for g in unique[:n_val])
        train_idx = [i for i, g in enumerate(original_ids) if g not in val_set]
        val_idx = [i for i, g in enumerate(original_ids) if g in val_set]
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError("Could not create any train/val split. Check your data.")
    return splits


# ---------------------------------------------------------------------------
# Early stopper / checkpoint helpers
# ---------------------------------------------------------------------------

class _EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = float("inf")
        self.bad = 0

    def step(self, metric: float) -> bool:
        if metric < self.best - 1e-9:
            self.best = float(metric)
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def _save(path: str, model: nn.Module) -> None:
    torch.save(model.state_dict(), path)


def _save_curves(
    train_h: List[float], val_h: List[float], out_dir: str, suffix: str = "", best_ep: Optional[int] = None
) -> None:
    if plt is None or not train_h:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_h) + 1), train_h, label="Train MAE")
    ax.plot(range(1, len(val_h) + 1), val_h, label="Val MAE")
    if best_ep and 1 <= best_ep <= len(val_h):
        ax.scatter([best_ep], [val_h[best_ep - 1]], c="red", s=40, zorder=5, label=f"Best ep {best_ep}")
    ax.set(xlabel="Epoch", ylabel="MAE", title="Training vs Validation MAE")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend()
    name = f"training_curves{suffix}.png" if suffix else "training_curves.png"
    fig.savefig(Path(out_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _train_one_fold(  # noqa: C901
    cfg: TrainConfig,
    model: LettuceSAMFusionNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fold_suffix: str = "",
    fold_label: str = "",
) -> Tuple[str, List[float], List[float], float, Optional[int]]:
    # Loss
    criterion = (
        nn.SmoothL1Loss(beta=cfg.huber_delta) if cfg.huber_delta > 0 else nn.L1Loss()
    )

    # Optimizer — fused kernel on CUDA
    fused = device.type == "cuda"
    try:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=fused
        )
    except TypeError:
        # PyTorch < 2.0 doesn't support fused
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience, min_lr=cfg.scheduler_min_lr,
    )
    stopper = _EarlyStopper(cfg.patience)

    # AMP scaler
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    amp_dtype = torch.float16

    # EMA
    use_ema = cfg.ema_decay > 0
    ema_model = copy.deepcopy(model).to(device) if use_ema else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # Block freezing
    rgb_blocks = list(model.rgb_branch.features)
    rgbd_blocks = list(model.rgbd_branch.features)
    blocks_map: Dict[str, Sequence[nn.Module]] = {"RGB": rgb_blocks, "RGBD": rgbd_blocks}
    warmups: _WarmupTracker = {}
    frozen_rgb = min(cfg.initial_frozen_rgb_blocks, len(rgb_blocks))
    frozen_rgbd = min(cfg.initial_frozen_rgbd_blocks, len(rgbd_blocks))
    if frozen_rgb > 0:
        _apply_freezing(rgb_blocks, frozen_rgb)
    if frozen_rgbd > 0:
        _apply_freezing(rgbd_blocks, frozen_rgbd)

    best_name = f"best_model_v10{fold_suffix}.pth"
    best_path = str(Path(cfg.out_dir) / best_name)
    train_hist: List[float] = []
    val_hist: List[float] = []
    best_epoch: Optional[int] = None
    accum = max(1, cfg.grad_accum_steps)

    def _on_rgb_unfreeze(idx: int) -> None:
        _start_warmup(warmups, "RGB", idx, cfg.branch_warmup_epochs)

    def _on_rgbd_unfreeze(idx: int) -> None:
        _start_warmup(warmups, "RGBD", idx, cfg.branch_warmup_epochs)

    try:
        for epoch in range(cfg.num_epochs):
            frozen_rgb = _maybe_unfreeze(
                epoch, cfg.unfreeze_start_epoch, cfg.rgb_unfreeze_interval,
                frozen_rgb, rgb_blocks, "RGB", _on_rgb_unfreeze,
            )
            frozen_rgbd = _maybe_unfreeze(
                epoch, cfg.unfreeze_start_epoch, cfg.rgbd_unfreeze_interval,
                frozen_rgbd, rgbd_blocks, "RGBD", _on_rgbd_unfreeze,
            )

            # ---- train ------------------------------------------------
            model.train()
            t_mae_sum, t_loss_sum, n_t = 0.0, 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                rgb = batch["rgb"].to(device, non_blocking=True)
                rgbd = batch["rgbd"].to(device, non_blocking=True)
                y = batch["dry_weight"].to(device, non_blocking=True)
                rgb, rgbd, y = _maybe_mixup(rgb, rgbd, y, cfg.mixup_alpha, cfg.mixup_prob)

                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    rp, dp, fp = model(rgb, rgbd)
                    loss = (
                        RGB_LOSS_WEIGHT * criterion(rp, y)
                        + RGBD_LOSS_WEIGHT * criterion(dp, y)
                        + FUSION_LOSS_WEIGHT * criterion(fp, y)
                    ) / accum

                scaler.scale(loss).backward()
                _scale_warmup_grads(warmups, blocks_map, cfg.branch_warmup_scale)

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if use_ema and ema_model is not None:
                        _update_ema(ema_model, model, cfg.ema_decay)

                bs = y.size(0)
                t_loss_sum += loss.item() * accum * bs
                t_mae_sum += torch.mean(torch.abs(fp.detach() - y)).item() * bs
                n_t += bs

            # ---- validate ----------------------------------------------
            model.eval()
            v_mae_sum, v_loss_sum, n_v = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    rgb = batch["rgb"].to(device, non_blocking=True)
                    rgbd = batch["rgbd"].to(device, non_blocking=True)
                    y = batch["dry_weight"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                        rp, dp, fp = model(rgb, rgbd)
                        vloss = (
                            RGB_LOSS_WEIGHT * criterion(rp, y)
                            + RGBD_LOSS_WEIGHT * criterion(dp, y)
                            + FUSION_LOSS_WEIGHT * criterion(fp, y)
                        )
                    bs = y.size(0)
                    v_loss_sum += vloss.item() * bs
                    v_mae_sum += torch.mean(torch.abs(fp - y)).item() * bs
                    n_v += bs

            t_mae = t_mae_sum / max(1, n_t)
            v_mae = v_mae_sum / max(1, n_v)
            train_hist.append(t_mae)
            val_hist.append(v_mae)
            span = min(cfg.best_mae_window, len(val_hist))
            smooth = float(np.mean(val_hist[-span:]))
            lr_now = optimizer.param_groups[0]["lr"]
            tag = f"[{fold_label}] " if fold_label else ""
            print(
                f"  {tag}epoch {epoch+1:>3}/{cfg.num_epochs}  lr={lr_now:.2e}  "
                f"train_mae={t_mae:.4f}  val_mae={v_mae:.4f}  "
                f"smooth({span})={smooth:.4f}"
            )

            if smooth <= stopper.best:
                target = ema_model if use_ema and ema_model is not None else model
                _save(best_path, target)
                best_epoch = epoch + 1

            if stopper.step(smooth):
                print(f"  early stop at epoch {epoch+1} (best={stopper.best:.4f})")
                break

            scheduler.step(v_mae)
            _tick_warmups(warmups)

    except KeyboardInterrupt:
        print("\n  interrupted — saving checkpoint…")
        target = ema_model if use_ema and ema_model is not None else model
        _save(best_path, target)

    # Ensure at least one checkpoint exists
    if not os.path.exists(best_path):
        target = ema_model if use_ema and ema_model is not None else model
        _save(best_path, target)

    return best_path, train_hist, val_hist, stopper.best, best_epoch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  AMP={cfg.use_amp and device.type == 'cuda'}  "
          f"compile={cfg.use_compile}  grad_accum={cfg.grad_accum_steps}")

    # ---- Load shard dataset ------------------------------------------------
    ds = ShardDataset(
        cfg.shard_dir,
        manifest_csv=cfg.manifest_csv,
        blacklist_ids=cfg.blacklist_ids,
    )
    print(f"[train] loaded {len(ds)} samples from shards")

    orig_ids = ds.get_original_ids_array()
    splits = _split_by_group(orig_ids, cfg)
    total_folds = len(splits)
    results: List[Tuple[str, str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        label = f"fold{fold_idx}" if total_folds > 1 else "split"
        suffix = f"_{label}" if total_folds > 1 else ""

        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        # Shards are already in RAM → no workers needed, skip pin_memory
        g = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False, worker_init_fn=seed_worker, generator=g,
        )

        model = LettuceSAMFusionNet(
            drop_path_prob=cfg.drop_path_prob,
            rgb_widths=cfg.rgb_widths,
            rgbd_widths=cfg.rgbd_widths,
            embed_dim=cfg.embed_dim,
        ).to(device)

        if cfg.use_compile:
            try:
                model = torch.compile(model)  # type: ignore[assignment]
                print(f"  [{label}] torch.compile enabled")
            except Exception as exc:
                print(f"  [{label}] torch.compile unavailable: {exc}")

        print(f"[train] {label}: train={len(train_idx)} val={len(val_idx)}")
        best_path, t_hist, v_hist, best_val, best_ep = _train_one_fold(
            cfg, model, train_loader, val_loader, device,
            fold_suffix=suffix, fold_label=label,
        )
        _save_curves(t_hist, v_hist, cfg.out_dir, suffix=suffix, best_ep=best_ep)
        results.append((label, best_path, best_val))
        print(f"  {label} done → {best_path}  best_val_mae={best_val:.4f}")

    # ---- Cross-validation summary ------------------------------------------
    if total_folds > 1:
        print("\nCross-validation summary:")
        for lbl, p, v in results:
            print(f"  {lbl}: val_mae={v:.4f}  ({p})")
        avg = float(np.mean([v for *_, v in results]))
        print(f"  average val MAE = {avg:.4f}")
        best_lbl, best_p, _ = min(results, key=lambda x: x[2])
        canon = str(Path(cfg.out_dir) / "best_model_v10.pth")
        if os.path.abspath(best_p) != os.path.abspath(canon):
            shutil.copy2(best_p, canon)
            print(f"  copied best fold ({best_lbl}) → {canon}")
    else:
        lbl, p, v = results[0]
        print(f"\nDone. Best model → {p}  (val_mae={v:.4f})")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
