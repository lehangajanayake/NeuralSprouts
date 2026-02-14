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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    batch_size: int = 128
    weight_decay: float = 5e-4

    # OneCycleLR schedule (used inside each phase)
    onecycle_pct_start: float = 0.3        # fraction of phase spent warming up
    onecycle_div_factor: float = 25.0      # initial_lr = max_lr / div_factor
    onecycle_final_div_factor: float = 1e4 # final_lr  = initial_lr / final_div_factor

    # ---- split / fold -----------------------------------------------------
    labels_csv: str = "../../datasets/Training/Train.csv"  # for variety + weight info
    val_per_cell: int = 2  # originals per (variety × regime) cell in val
    val_ratio: float = 0.2
    seed: int = 43
    patience: int = 150
    num_folds: int = 1
    group_by_original: bool = True

    # ---- architecture -----------------------------------------------------
    drop_path_prob: float = 0.05
    rgb_widths: Tuple[int, ...] = (16, 32, 64, 128)
    rgbd_widths: Tuple[int, ...] = (16, 32, 64, 144)
    embed_dim: int = 96

    # ---- regularisation ---------------------------------------------------
    mixup_alpha: float = 0.1
    mixup_prob: float = 0.0
    huber_delta: float = 0.5   # in log-space; ~0.5 is a good default
    ema_decay: float = 0.995
    log_targets: bool = True   # train in log1p(y) space to fix heavy-plant under-prediction

    # ---- phased branch-wise pretraining -----------------------------------
    # Phase 1: train RGB branch alone (RGBD + fusion frozen)
    # Phase 2: train RGBD branch alone (RGB + fusion frozen)
    # Phase 3: fine-tune everything together (branches + fusion)
    phase1_epochs: int = 25      # RGB branch pretraining  (converges by ~12, stop before memorising)
    phase2_epochs: int = 25      # RGBD branch pretraining
    phase3_epochs: int = 120     # joint fine-tuning — this is where real gains happen
    phase1_lr: float = 1e-3      # calmer LR for branch pretraining
    phase2_lr: float = 1e-3
    phase3_lr: float = 5e-4      # lower LR for fine-tuning
    phase3_branch_lr_scale: float = 0.2  # branches get lr * this in phase 3

    # ---- speed / hardware -------------------------------------------------
    use_amp: bool = True
    use_compile: bool = False  # requires Triton (Linux only); leave False on Windows
    grad_accum_steps: int = 1
    preload_to_gpu: bool = False  # move entire dataset to CUDA once (eliminates CPU→GPU transfers)

    # ---- output -----------------------------------------------------------
    out_dir: str = "."
    blacklist_ids: Tuple[int, ...] = (163,)
    best_mae_window: int = 1


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
# Freeze / unfreeze helpers for phased training
# ---------------------------------------------------------------------------

def _freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in *module*."""
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in *module*."""
    for p in module.parameters():
        p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def _split_by_group(
    original_ids: np.ndarray,
    cfg: TrainConfig,
) -> List[Tuple[List[int], List[int]]]:
    """Stratified train/val split ensuring every (variety × weight-regime)
    cell is represented in the validation set.

    Steps
    -----
    1. Read *labels_csv* to obtain ``Variety`` and ``DryWeightShoot`` for
       each original ``image_id``.
    2. Split originals into three weight regimes (low / mid / high) using
       the 33rd and 67th percentile of ``DryWeightShoot``.
    3. For each of the 4 varieties × 3 regimes = 12 cells, randomly select
       ``val_per_cell`` originals for validation.
    4. Every augmented copy of a selected original goes to val; the rest
       go to train.

    This guarantees that the validation set always covers all varieties
    and the full range of dry-weight values.
    """
    # --- load metadata ---------------------------------------------------
    csv_path = Path(cfg.labels_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"labels_csv not found: {csv_path}  (needed for stratified split)"
        )
    meta = pd.read_csv(csv_path)

    # Remove blacklisted
    if cfg.blacklist_ids:
        meta = meta[~meta["image_id"].isin(cfg.blacklist_ids)]

    # Restrict to originals actually present in the shard dataset
    unique_in_ds = set(int(x) for x in np.unique(original_ids))
    meta = meta[meta["image_id"].isin(unique_in_ds)].copy()

    # --- assign weight regime --------------------------------------------
    q33 = float(meta["DryWeightShoot"].quantile(0.333))
    q67 = float(meta["DryWeightShoot"].quantile(0.667))

    def _regime(w: float) -> str:
        if w <= q33:
            return "low"
        elif w <= q67:
            return "mid"
        return "high"

    meta["regime"] = meta["DryWeightShoot"].apply(_regime)

    # --- stratified selection --------------------------------------------
    rng = np.random.RandomState(cfg.seed)
    val_originals: set[int] = set()

    for _variety in sorted(meta["Variety"].unique()):
        for _regime in ["low", "mid", "high"]:
            cell = meta[(meta["Variety"] == _variety) & (meta["regime"] == _regime)]
            ids_in_cell = cell["image_id"].values.copy()
            rng.shuffle(ids_in_cell)
            n_pick = min(cfg.val_per_cell, len(ids_in_cell))
            if n_pick == 0:
                print(
                    f"  [split] WARNING: no originals for {_variety}/{_regime}"
                )
                continue
            for oid in ids_in_cell[:n_pick]:
                val_originals.add(int(oid))

    # --- map back to shard indices ---------------------------------------
    train_idx = [i for i, g in enumerate(original_ids) if int(g) not in val_originals]
    val_idx = [i for i, g in enumerate(original_ids) if int(g) in val_originals]

    n_val_orig = len(val_originals)
    n_train_orig = len(unique_in_ds) - n_val_orig
    print(
        f"  [split] stratified: {n_val_orig} val originals "
        f"({len(val_idx)} samples) / {n_train_orig} train originals "
        f"({len(train_idx)} samples)"
    )
    print(
        f"  [split] weight terciles: low <= {q33:.2f} | mid <= {q67:.2f} | high > {q67:.2f}"
    )

    if not train_idx or not val_idx:
        raise ValueError(
            "Stratified split produced an empty train or val set. "
            "Lower val_per_cell or check your data."
        )
    return [(train_idx, val_idx)]


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
    train_h: List[float], val_h: List[float], out_dir: str, suffix: str = "",
    best_ep: Optional[int] = None,
    phase_boundaries: Optional[List[int]] = None,
) -> None:
    if plt is None or not train_h:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(train_h) + 1), train_h, label="Train MAE")
    ax.plot(range(1, len(val_h) + 1), val_h, label="Val MAE")
    if best_ep and 1 <= best_ep <= len(val_h):
        ax.scatter([best_ep], [val_h[best_ep - 1]], c="red", s=40, zorder=5, label=f"Best ep {best_ep}")
    # Draw vertical lines at phase boundaries
    if phase_boundaries:
        phase_labels = ["RGB→RGBD", "RGBD→Joint"]
        for i, bnd in enumerate(phase_boundaries):
            lbl = phase_labels[i] if i < len(phase_labels) else f"Phase {i+2}"
            ax.axvline(bnd + 0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label=lbl)
    ax.set(xlabel="Epoch", ylabel="MAE", title="Training vs Validation MAE (3-phase)")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    name = f"training_curves{suffix}.png" if suffix else "training_curves.png"
    fig.savefig(Path(out_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _make_optimizer(
    params,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> optim.Optimizer:
    """Create AdamW with fused kernel when on CUDA."""
    fused = device.type == "cuda"
    try:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except TypeError:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _run_phase(  # noqa: C901
    *,
    phase_name: str,
    cfg: TrainConfig,
    model: LettuceSAMFusionNet,
    ema_model: Optional[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_lr: float,
    num_epochs: int,
    loss_weights: Tuple[float, float, float],
    train_hist: List[float],
    val_hist: List[float],
    best_smooth: float,
    best_epoch: Optional[int],
    best_path: str,
    curves_suffix: str,
    fold_label: str,
    use_early_stop: bool = False,
    eval_metric: str = "fused",
    branch_lr_scale: float = 1.0,
    phase_boundaries: Optional[List[int]] = None,
) -> Tuple[float, Optional[int]]:
    """Run one training phase (branch pretrain or joint fine-tune).

    Parameters
    ----------
    loss_weights : (rgb_w, rgbd_w, fusion_w)
        Weights for each branch's loss.  Set a weight to 0 to ignore that head.
    eval_metric : 'fused' | 'rgb' | 'rgbd'
        Which prediction to use for val MAE tracking.
    branch_lr_scale : float
        Multiply the LR for branch parameters by this factor.  Use < 1.0
        in Phase 3 so pretrained branches don't drift while fusion trains.
    phase_boundaries : list of epoch indices where phases change (for plotting).
    """
    criterion = (
        nn.SmoothL1Loss(beta=cfg.huber_delta) if cfg.huber_delta > 0 else nn.L1Loss()
    )
    rgb_w, rgbd_w, fusion_w = loss_weights

    # Build parameter groups with differential LR for branches vs fusion
    branch_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and ("rgb_branch" in n or "rgbd_branch" in n)]
    fusion_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and "rgb_branch" not in n and "rgbd_branch" not in n]
    branch_lr = max_lr * branch_lr_scale
    param_groups = []
    if branch_params:
        param_groups.append({"params": branch_params, "lr": branch_lr})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": max_lr})
    if not param_groups:
        print(f"  WARNING: no trainable parameters in {phase_name}!")
        return best_smooth, best_epoch

    n_trainable = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name}")
    print(f"  Epochs: {num_epochs}  |  trainable params: {n_trainable:,}")
    if branch_lr_scale < 1.0:
        print(f"  LR: branches={branch_lr:.1e}  fusion/head={max_lr:.1e}  (scale={branch_lr_scale})")
    else:
        print(f"  LR: {max_lr:.1e}")
    print(f"  Loss weights: RGB={rgb_w:.1f}  RGBD={rgbd_w:.1f}  Fusion={fusion_w:.1f}")
    print(f"  Eval metric: {eval_metric}")
    print(f"{'='*60}")

    fused = device.type == "cuda"
    try:
        optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay, fused=fused)
    except TypeError:
        optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    max_lrs = [pg["lr"] for pg in param_groups]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=cfg.onecycle_pct_start,
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
        anneal_strategy="cos",
    )

    stopper = _EarlyStopper(cfg.patience) if use_early_stop else None

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    amp_dtype = torch.float16
    use_ema = ema_model is not None
    accum = max(1, cfg.grad_accum_steps)
    global_epoch_offset = len(train_hist)  # for display numbering
    best_raw_val = float("inf")  # track lowest single-epoch val MAE for saving

    try:
        for ep in range(num_epochs):
            global_ep = global_epoch_offset + ep + 1

            # ---- train ------------------------------------------------
            model.train()
            t_mae_sum, n_t = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                rgb = batch["rgb"].to(device, non_blocking=True)
                rgbd = batch["rgbd"].to(device, non_blocking=True)
                y_raw = batch["dry_weight"].to(device, non_blocking=True)
                # Log-transform targets: model predicts in log1p space
                y = torch.log1p(y_raw) if cfg.log_targets else y_raw
                rgb, rgbd, y = _maybe_mixup(rgb, rgbd, y, cfg.mixup_alpha, cfg.mixup_prob)

                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    rp, dp, fp = model(rgb, rgbd)
                    loss = 0.0
                    if rgb_w > 0:
                        loss = loss + rgb_w * criterion(rp, y)
                    if rgbd_w > 0:
                        loss = loss + rgbd_w * criterion(dp, y)
                    if fusion_w > 0:
                        loss = loss + fusion_w * criterion(fp, y)
                    loss = loss / accum

                scaler.scale(loss).backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if use_ema and ema_model is not None:
                        _update_ema(ema_model, model, cfg.ema_decay)

                bs = y.size(0)
                # Track MAE in *original* grams (invert log-space predictions)
                if cfg.log_targets:
                    rp_g = torch.expm1(rp.detach())
                    dp_g = torch.expm1(dp.detach())
                    fp_g = torch.expm1(fp.detach())
                    y_g = y_raw
                else:
                    rp_g, dp_g, fp_g, y_g = rp.detach(), dp.detach(), fp.detach(), y
                if eval_metric == "rgb":
                    t_mae_sum += torch.mean(torch.abs(rp_g - y_g)).item() * bs
                elif eval_metric == "rgbd":
                    t_mae_sum += torch.mean(torch.abs(dp_g - y_g)).item() * bs
                else:
                    t_mae_sum += torch.mean(torch.abs(fp_g - y_g)).item() * bs
                n_t += bs

            # ---- validate ----------------------------------------------
            model.eval()
            v_mae_sum, n_v = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    rgb_b = batch["rgb"].to(device, non_blocking=True)
                    rgbd_b = batch["rgbd"].to(device, non_blocking=True)
                    y_raw_b = batch["dry_weight"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                        rp, dp, fp = model(rgb_b, rgbd_b)
                    bs = y_raw_b.size(0)
                    # Val MAE always in original grams
                    if cfg.log_targets:
                        rp_g = torch.expm1(rp)
                        dp_g = torch.expm1(dp)
                        fp_g = torch.expm1(fp)
                        y_g = y_raw_b
                    else:
                        rp_g, dp_g, fp_g, y_g = rp, dp, fp, y_raw_b
                    if eval_metric == "rgb":
                        v_mae_sum += torch.mean(torch.abs(rp_g - y_g)).item() * bs
                    elif eval_metric == "rgbd":
                        v_mae_sum += torch.mean(torch.abs(dp_g - y_g)).item() * bs
                    else:
                        v_mae_sum += torch.mean(torch.abs(fp_g - y_g)).item() * bs
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
                f"  {tag}{phase_name} ep {ep+1:>3}/{num_epochs} (global {global_ep})  "
                f"lr={lr_now:.2e}  train={t_mae:.4f}  val={v_mae:.4f}  "
                f"smooth({span})={smooth:.4f}"
            )

            # Save checkpoint when raw val MAE improves (not smooth)
            # This ensures the saved model is truly the best single epoch.
            if v_mae < best_raw_val:
                best_raw_val = v_mae
                target = ema_model if use_ema and ema_model is not None else model
                _save(best_path, target)
                best_epoch = global_ep
                print(f"    ★ saved best model (val_mae={v_mae:.4f})")

            # Track smooth for early stopping only
            if smooth < best_smooth:
                best_smooth = smooth

            if stopper is not None and stopper.step(smooth):
                print(f"  early stop at epoch {ep+1} (best_val={best_raw_val:.4f}  best_smooth={best_smooth:.4f})")
                break

            if (ep + 1) % 5 == 0 or (ep + 1) == num_epochs:
                _save_curves(train_hist, val_hist, cfg.out_dir,
                             suffix=curves_suffix, best_ep=best_epoch,
                             phase_boundaries=phase_boundaries)

    except KeyboardInterrupt:
        # Save current state to a SEPARATE file — never overwrite the best checkpoint
        last_path = best_path.replace("best_model", "last_model")
        target = ema_model if use_ema and ema_model is not None else model
        _save(last_path, target)
        print(f"\n  interrupted during {phase_name} — saved current state → {last_path}")
        print(f"  (best checkpoint untouched at {best_path}, val_mae={best_raw_val:.4f})")

    # If no checkpoint was ever saved (e.g. val never improved), save current state
    if not os.path.exists(best_path):
        target = ema_model if use_ema and ema_model is not None else model
        _save(best_path, target)
        print(f"  (no improvement recorded — saved current state as fallback)")

    _save_curves(train_hist, val_hist, cfg.out_dir,
                 suffix=curves_suffix, best_ep=best_epoch,
                 phase_boundaries=phase_boundaries)

    return best_smooth, best_epoch


def _train_one_fold(  # noqa: C901
    cfg: TrainConfig,
    model: LettuceSAMFusionNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fold_suffix: str = "",
    fold_label: str = "",
) -> Tuple[str, List[float], List[float], float, Optional[int]]:
    best_name = f"best_model_v10{fold_suffix}.pth"
    best_path = str(Path(cfg.out_dir) / best_name)
    train_hist: List[float] = []
    val_hist: List[float] = []
    best_epoch: Optional[int] = None
    best_smooth: float = float("inf")
    curves_suffix = fold_suffix
    phase_boundaries: List[int] = []  # global epoch indices where phases change

    # EMA
    use_ema = cfg.ema_decay > 0
    ema_model = copy.deepcopy(model).to(device) if use_ema else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # ==================================================================
    # Phase 1 — RGB branch pretraining
    # ==================================================================
    _freeze_module(model.rgbd_branch)          # freeze RGBD
    _freeze_module(model.fusion)               # freeze fusion
    _freeze_module(model.fusion_in_dropout)    # freeze fusion dropout
    _unfreeze_module(model.rgb_branch)         # only RGB trains

    best_smooth, best_epoch = _run_phase(
        phase_name="Phase 1: RGB branch",
        cfg=cfg, model=model, ema_model=ema_model,
        train_loader=train_loader, val_loader=val_loader, device=device,
        max_lr=cfg.phase1_lr, num_epochs=cfg.phase1_epochs,
        loss_weights=(1.0, 0.0, 0.0),  # only RGB loss
        train_hist=train_hist, val_hist=val_hist,
        best_smooth=best_smooth, best_epoch=best_epoch,
        best_path=best_path, curves_suffix=curves_suffix,
        fold_label=fold_label,
        use_early_stop=False,
        eval_metric="rgb",
        phase_boundaries=phase_boundaries,
    )
    phase_boundaries.append(len(train_hist))  # mark Phase 1 → 2 boundary

    # ==================================================================
    # Phase 2 — RGBD branch pretraining
    # ==================================================================
    _freeze_module(model.rgb_branch)           # freeze RGB
    _freeze_module(model.fusion)               # keep fusion frozen
    _unfreeze_module(model.rgbd_branch)        # only RGBD trains

    best_smooth, best_epoch = _run_phase(
        phase_name="Phase 2: RGBD branch",
        cfg=cfg, model=model, ema_model=ema_model,
        train_loader=train_loader, val_loader=val_loader, device=device,
        max_lr=cfg.phase2_lr, num_epochs=cfg.phase2_epochs,
        loss_weights=(0.0, 1.0, 0.0),  # only RGBD loss
        train_hist=train_hist, val_hist=val_hist,
        best_smooth=best_smooth, best_epoch=best_epoch,
        best_path=best_path, curves_suffix=curves_suffix,
        fold_label=fold_label,
        use_early_stop=False,
        eval_metric="rgbd",
        phase_boundaries=phase_boundaries,
    )
    phase_boundaries.append(len(train_hist))  # mark Phase 2 → 3 boundary

    # ==================================================================
    # Phase 3 — Joint fine-tuning (everything unfrozen)
    # ==================================================================
    # Reset val tracking for the joint phase — branch-level MAE is not
    # comparable to fused MAE so we start best_smooth fresh.
    best_smooth = float("inf")
    best_epoch = None

    _unfreeze_module(model.rgb_branch)
    _unfreeze_module(model.rgbd_branch)
    _unfreeze_module(model.fusion)
    _unfreeze_module(model.fusion_in_dropout)

    best_smooth, best_epoch = _run_phase(
        phase_name="Phase 3: Joint fine-tune",
        cfg=cfg, model=model, ema_model=ema_model,
        train_loader=train_loader, val_loader=val_loader, device=device,
        max_lr=cfg.phase3_lr, num_epochs=cfg.phase3_epochs,
        loss_weights=(RGB_LOSS_WEIGHT, RGBD_LOSS_WEIGHT, FUSION_LOSS_WEIGHT),
        train_hist=train_hist, val_hist=val_hist,
        best_smooth=best_smooth, best_epoch=best_epoch,
        best_path=best_path, curves_suffix=curves_suffix,
        fold_label=fold_label,
        use_early_stop=True,
        eval_metric="fused",
        branch_lr_scale=cfg.phase3_branch_lr_scale,
        phase_boundaries=phase_boundaries,
    )

    # Ensure at least one checkpoint exists
    if not os.path.exists(best_path):
        target = ema_model if use_ema and ema_model is not None else model
        _save(best_path, target)

    _save_curves(train_hist, val_hist, cfg.out_dir,
                 suffix=curves_suffix, best_ep=best_epoch,
                 phase_boundaries=phase_boundaries)

    return best_path, train_hist, val_hist, best_smooth, best_epoch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
    print(f"[train] device={device}  AMP={cfg.use_amp and device.type == 'cuda'}  "
          f"compile={cfg.use_compile}  grad_accum={cfg.grad_accum_steps}  "
          f"gpu_preload={cfg.preload_to_gpu and device.type == 'cuda'}  "
          f"log_targets={cfg.log_targets}")
    print(f"[train] 3-phase training: {cfg.phase1_epochs} (RGB) + {cfg.phase2_epochs} (RGBD) "
          f"+ {cfg.phase3_epochs} (joint) = {total_epochs} total epochs")

    # ---- Load shard dataset ------------------------------------------------
    ds = ShardDataset(
        cfg.shard_dir,
        manifest_csv=cfg.manifest_csv,
        blacklist_ids=cfg.blacklist_ids,
    )
    print(f"[train] loaded {len(ds)} samples from shards")

    # Must extract original_ids (CPU numpy) *before* moving tensors to GPU
    orig_ids = ds.get_original_ids_array()

    # Pre-load entire dataset to GPU — eliminates all CPU→GPU transfers
    if cfg.preload_to_gpu and device.type == "cuda":
        ds.to_device(device)
        print(f"[train] pre-loaded dataset to {device}")

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
            log_targets=cfg.log_targets,
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
        # Curves are already saved inside _train_one_fold (periodically + at end)
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
