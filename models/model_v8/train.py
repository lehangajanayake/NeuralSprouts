import copy
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from dataloader import PlantDatasetV8
from model import LettuceSAMFusionNet

RGB_LOSS_WEIGHT = 0.2
RGBD_LOSS_WEIGHT = 0.3
FUSION_LOSS_WEIGHT = .5

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class TrainConfig:
    train_csv: str = '../../datasets/Training/Augmented_v8/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented_v8/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented_v8/DepthImages'

    batch_size: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    seed: int = 43
    patience: int = 100
    outputs_per_original: int = 41
    num_folds: int = 1
    group_by_original: bool = True
    val_only_originals: bool = False

    preload_to_gpu: bool = True
    preload_device: str = 'cuda'

    out_dir: str = '.'
    blacklist_ids: Tuple[int, ...] = (163,)
    best_mae_window: int = 5
    ema_decay: float = 0.995
    drop_path_prob: float = 0.1
    rgb_widths: Tuple[int, ...] = (32, 64, 96, 128)
    rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128)
    embed_dim: int = 256
    spatial_kernel: Tuple[int, ...] = (5, 7)
    spatial_layers: int = 1
    spatial_dropout: float = 0.1
    share_spatial_attn: bool = False

    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3

    cutmix_alpha: float = 0.4
    cutmix_prob: float = 0.0

    huber_delta: float = 1.0

    # ---- 3-phase staged training ------------------------------------------
    phase1_epochs: int = 20       # RGB branch pretraining
    phase2_epochs: int = 20       # RGBD branch pretraining
    phase3_epochs: int = 80       # Joint fine-tuning
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-3
    phase3_lr: float = 5e-4       # lower for fine-tuning
    phase3_branch_lr_scale: float = 0.2  # branches get lr * this in phase 3

    # OneCycleLR schedule
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4


def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    decay = float(decay)
    if decay <= 0.0:
        return
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_param = ema_params.get(name)
            if ema_param is None:
                continue
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        # Keep buffers (e.g., BatchNorm stats) in sync
        for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
            ema_buf.copy_(buf)


def maybe_mixup_batch(
    rgb: torch.Tensor,
    rgbd: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or prob <= 0.0:
        return rgb, rgbd, targets
    if torch.rand(1, device=rgb.device).item() > prob:
        return rgb, rgbd, targets
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(rgb.size(0), device=rgb.device)
    rgb_mix = lam * rgb + (1.0 - lam) * rgb[perm]
    rgbd_mix = lam * rgbd + (1.0 - lam) * rgbd[perm]
    targets_mix = lam * targets + (1.0 - lam) * targets[perm]
    return rgb_mix, rgbd_mix, targets_mix


def _rand_bbox(width: int, height: int, lam: float, device: torch.device) -> Tuple[int, int, int, int]:
    cut_rat = torch.sqrt(torch.tensor(1.0 - lam, device=device))
    cut_w = int(width * cut_rat.item())
    cut_h = int(height * cut_rat.item())
    cx = torch.randint(0, width, (1,), device=device).item()
    cy = torch.randint(0, height, (1,), device=device).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    if x1 == x2:
        x2 = min(x1 + 1, width)
        x1 = max(0, x2 - 1)
    if y1 == y2:
        y2 = min(y1 + 1, height)
        y1 = max(0, y2 - 1)
    return x1, y1, x2, y2


def maybe_cutmix_batch(
    rgb: torch.Tensor,
    rgbd: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or prob <= 0.0:
        return rgb, rgbd, targets
    if torch.rand(1, device=rgb.device).item() > prob:
        return rgb, rgbd, targets
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(rgb.size(0), device=rgb.device)
    _, _, h, w = rgb.shape
    x1, y1, x2, y2 = _rand_bbox(w, h, lam, rgb.device)
    rgb[:, :, y1:y2, x1:x2] = rgb[perm, :, y1:y2, x1:x2]
    rgbd[:, :, y1:y2, x1:x2] = rgbd[perm, :, y1:y2, x1:x2]
    box_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - (box_area / float(w * h))
    targets_mix = lam_adjusted * targets + (1.0 - lam_adjusted) * targets[perm]
    return rgb, rgbd, targets_mix


def apply_block_freezing(blocks: Sequence[nn.Module], frozen_count: int) -> None:
    frozen = max(0, min(int(frozen_count), len(blocks)))
    for idx, block in enumerate(blocks):
        requires = idx >= frozen
        for param in block.parameters():
            param.requires_grad = requires


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


class TensorDictDataset(Dataset):
    """Simple dataset wrapper around preloaded tensors on any device."""

    def __init__(self, rgb: torch.Tensor, rgbd: torch.Tensor, dry_weight: torch.Tensor):
        self.rgb = rgb
        self.rgbd = rgbd
        self.dry_weight = dry_weight

    def __len__(self) -> int:
        return self.rgb.size(0)

    def __getitem__(self, idx: int):
        return {
            'rgb': self.rgb[idx],
            'rgbd': self.rgbd[idx],
            'dry_weight': self.dry_weight[idx],
        }


def _unwrap_subset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, dataset.indices
    return dataset, range(len(dataset))


def preload_subset_to_device(dataset, device: torch.device, label: str) -> TensorDictDataset:
    base_ds, indices = _unwrap_subset(dataset)

    rgb_list: List[torch.Tensor] = []
    rgbd_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []

    for idx in indices:
        sample = base_ds[int(idx)]
        rgb_list.append(sample['rgb'])
        rgbd_list.append(sample['rgbd'])
        target_list.append(sample['dry_weight'])

    if not rgb_list:
        raise ValueError(f'Cannot preload empty {label} dataset.')

    rgb = torch.stack(rgb_list, dim=0).to(device, non_blocking=True)
    rgbd = torch.stack(rgbd_list, dim=0).to(device, non_blocking=True)
    targets = torch.stack(target_list, dim=0).to(device, non_blocking=True)

    print(f"[preload] moved {rgb.shape[0]} {label} samples to {device}")
    return TensorDictDataset(rgb, rgbd, targets)


def _build_full_dataset(cfg: TrainConfig) -> PlantDatasetV8:
    dataset = PlantDatasetV8(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=True,
        seed=cfg.seed,
        enable_cache=True,
        num_views=3,
        blacklist_ids=cfg.blacklist_ids,
    )
    if len(dataset.df) == 0:
        raise ValueError('No samples found in augmented CSV; check preprocessing paths.')
    return dataset


def _compute_group_ids(full_df, cfg: TrainConfig) -> Tuple[np.ndarray, bool, np.ndarray | None]:
    has_original_ids = 'original_id' in full_df.columns
    original_mask: np.ndarray | None = None

    if cfg.val_only_originals:
        if not has_original_ids:
            raise ValueError('val_only_originals requires original_id column in the CSV.')
        if 'is_original' in full_df.columns:
            original_mask = full_df['is_original'].astype(bool).to_numpy()
        elif 'shard_index' in full_df.columns:
            original_mask = (full_df['shard_index'].astype(int) == 0).to_numpy()
        else:
            originals = ~full_df['original_id'].astype(int).duplicated(keep='first')
            original_mask = originals.to_numpy()

    if not cfg.group_by_original:
        ids = np.arange(len(full_df), dtype=int)
        return ids, has_original_ids, original_mask

    if has_original_ids:
        group_ids = full_df['original_id'].astype(int).to_numpy()
    else:
        outputs_per_original = max(1, int(cfg.outputs_per_original))
        group_ids = ((full_df['id'].astype(int) - 1) // outputs_per_original).to_numpy()
    return group_ids, has_original_ids, original_mask


def _split_group_indices(
    group_ids: np.ndarray,
    cfg: TrainConfig,
    has_original_ids: bool,
    original_mask: np.ndarray | None,
) -> List[Tuple[List[int], List[int]]]:
    unique_groups = np.unique(group_ids)
    total_groups = len(unique_groups)
    if total_groups < 2:
        raise ValueError('Need at least two unique originals to create a split. Reduce folds or gather more data.')

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(unique_groups)

    num_folds = max(1, int(cfg.num_folds))
    if num_folds > total_groups:
        print(f"[kfold] Requested {num_folds} folds but only {total_groups} unique originals; capping folds at {total_groups}.")
        num_folds = total_groups

    splits: List[Tuple[List[int], List[int]]] = []
    if num_folds > 1:
        fold_groups = np.array_split(unique_groups, num_folds)
        for fold_id, val_groups in enumerate(fold_groups, 1):
            val_set = set(int(g) for g in val_groups)
            train_idx = [int(i) for i, g in enumerate(group_ids) if g not in val_set]
            val_idx = [int(i) for i, g in enumerate(group_ids) if g in val_set]
            if cfg.val_only_originals:
                if not has_original_ids or original_mask is None:
                    raise ValueError('val_only_originals requires original_id column in the CSV.')
                val_idx = [idx for idx in val_idx if bool(original_mask[idx])]
            if not train_idx or not val_idx:
                raise ValueError(f'Fold {fold_id} is empty. Reduce num_folds or ensure more originals are available.')
            splits.append((train_idx, val_idx))
    else:
        val_group_count = max(1, int(round(total_groups * float(cfg.val_ratio))))
        if val_group_count >= total_groups:
            val_group_count = total_groups - 1
        if val_group_count < 1:
            raise ValueError('Validation split is empty. Increase val_ratio or collect more data.')
        val_group_ids = set(unique_groups[:val_group_count])
        train_idx = [int(i) for i, g in enumerate(group_ids) if g not in val_group_ids]
        val_idx = [int(i) for i, g in enumerate(group_ids) if g in val_group_ids]
        if cfg.val_only_originals:
            if not has_original_ids or original_mask is None:
                raise ValueError('val_only_originals requires original_id column in the CSV.')
            val_idx = [idx for idx in val_idx if bool(original_mask[idx])]
        if not train_idx or not val_idx:
            hint = 'Ensure original_id exists in the CSV' if has_original_ids else 'Adjust val_ratio or outputs_per_original'
            raise ValueError(f'Group-based split resulted in empty train/val set. {hint}.')
        splits.append((train_idx, val_idx))

    return splits


def _build_dataloaders_from_indices(
    full_ds: PlantDatasetV8,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    cfg: TrainConfig,
    label: str,
) -> Tuple[DataLoader, DataLoader]:
    train_ds: Dataset = Subset(full_ds, list(train_idx))
    val_ds: Dataset = Subset(full_ds, list(val_idx))

    preload_device = None
    if cfg.preload_to_gpu:
        try:
            requested_device = torch.device(cfg.preload_device)
        except Exception as exc:
            print(f"[preload] invalid device '{cfg.preload_device}': {exc}. Falling back to CPU.")
            requested_device = None

        if requested_device is not None:
            if requested_device.type == 'cuda' and not torch.cuda.is_available():
                print('[preload] CUDA not available; skipping GPU preload.')
            else:
                preload_device = requested_device
                train_ds = preload_subset_to_device(train_ds, preload_device, f'{label}-train')
                val_ds = preload_subset_to_device(val_ds, preload_device, f'{label}-val')

    num_workers = 0 if (os.name == 'nt' or preload_device is not None) else 2
    pin_memory = torch.cuda.is_available() and preload_device is None
    g = torch.Generator().manual_seed(cfg.seed)

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        **loader_kwargs,
    )

    return train_loader, val_loader


def make_fold_loaders(cfg: TrainConfig) -> List[Tuple[DataLoader, DataLoader, str]]:
    full = _build_full_dataset(cfg)
    group_ids, has_original_ids, original_mask = _compute_group_ids(full.df, cfg)
    splits = _split_group_indices(group_ids, cfg, has_original_ids, original_mask)

    full.build_cache(max_base_items=None)

    loaders: List[Tuple[DataLoader, DataLoader, str]] = []
    multiple = len(splits) > 1
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        label = f'fold{fold_idx}' if multiple else 'split'
        train_loader, val_loader = _build_dataloaders_from_indices(full, train_idx, val_idx, cfg, label)
        loaders.append((train_loader, val_loader, label))

    return loaders


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best = float('inf')
        self.bad = 0

    def step(self, metric: float) -> bool:
        if metric < self.best - 1e-9:
            self.best = float(metric)
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def save_checkpoint(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)


def save_training_curves(
    train_history: List[float],
    val_history: List[float],
    out_dir: str,
    suffix: str = '',
    best_epoch: int | None = None,
    phase_boundaries: List[int] | None = None,
) -> None:
    if not train_history or not val_history:
        return
    if plt is None:
        print('matplotlib not available; skipping training curve plot.')
        return

    epochs = range(1, len(train_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_history, label='Train MAE')
    ax.plot(epochs, val_history, label='Val MAE')
    if best_epoch is not None and 1 <= best_epoch <= len(train_history):
        best_val = val_history[best_epoch - 1]
        ax.scatter([best_epoch], [best_val], color='red', s=40, zorder=5, label=f'Best ep {best_epoch}')
    if phase_boundaries:
        phase_labels = ['RGB→RGBD', 'RGBD→Joint']
        for i, bnd in enumerate(phase_boundaries):
            lbl = phase_labels[i] if i < len(phase_labels) else f'Phase {i+2}'
            ax.axvline(bnd + 0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=lbl)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Training vs Validation MAE (3-phase)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)

    out_name = f'training_curves{suffix}.png' if suffix else 'training_curves.png'
    out_path = Path(out_dir) / out_name
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved training curves to: {out_path}')


def train_fusion_regressor(
    cfg: TrainConfig,
    model: LettuceSAMFusionNet,
    train_loader,
    val_loader,
    device,
    fold_suffix: str = '',
    fold_label: str = '',
):
    criterion = (
        nn.SmoothL1Loss(beta=float(cfg.huber_delta)) if cfg.huber_delta and float(cfg.huber_delta) > 0.0
        else nn.L1Loss()
    )

    best_name = f'best_model_v8{fold_suffix}.pth' if fold_suffix else 'best_model_v8.pth'
    best_path = str(Path(cfg.out_dir) / best_name)
    train_history: List[float] = []
    val_history: List[float] = []
    best_epoch_index: int | None = None
    best_val_mae: float = float('inf')
    phase_boundaries: List[int] = []

    use_ema = cfg.ema_decay is not None and float(cfg.ema_decay) > 0.0
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for param in ema_model.parameters():
            param.requires_grad_(False)

    def _run_phase(
        phase_name: str,
        num_epochs: int,
        max_lr: float,
        loss_weights: Tuple[float, float, float],
        eval_metric: str = 'fused',
        branch_lr_scale: float = 1.0,
        use_early_stop: bool = False,
    ) -> None:
        nonlocal best_val_mae, best_epoch_index
        rgb_w, rgbd_w, fusion_w = loss_weights

        # Build param groups with differential LR
        branch_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and ('rgb_branch' in n or 'rgbd_branch' in n)]
        fusion_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and 'rgb_branch' not in n and 'rgbd_branch' not in n]
        branch_lr = max_lr * branch_lr_scale
        param_groups = []
        if branch_params:
            param_groups.append({'params': branch_params, 'lr': branch_lr})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': max_lr})
        if not param_groups:
            print(f'  WARNING: no trainable params in {phase_name}!')
            return

        n_trainable = sum(p.numel() for pg in param_groups for p in pg['params'])
        print(f"\n{'='*60}")
        print(f"  Phase: {phase_name}")
        print(f"  Epochs: {num_epochs}  |  trainable params: {n_trainable:,}")
        if branch_lr_scale < 1.0:
            print(f"  LR: branches={branch_lr:.1e}  fusion/head={max_lr:.1e}")
        else:
            print(f"  LR: {max_lr:.1e}")
        print(f"  Loss weights: RGB={rgb_w:.1f}  RGBD={rgbd_w:.1f}  Fusion={fusion_w:.1f}")
        print(f"{'='*60}")

        optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        max_lrs = [pg['lr'] for pg in param_groups]
        total_steps = len(train_loader) * num_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=cfg.onecycle_pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor,
            anneal_strategy='cos',
        )
        stopper = EarlyStopper(cfg.patience) if use_early_stop else None
        global_offset = len(train_history)

        try:
            for ep in range(num_epochs):
                global_ep = global_offset + ep + 1

                # ---- train ------------------------------------------------
                model.train()
                t_mae_sum, n_t = 0.0, 0

                for batch in train_loader:
                    rgb = batch['rgb'].to(device, non_blocking=True)
                    rgbd = batch['rgbd'].to(device, non_blocking=True)
                    y = batch['dry_weight'].to(device, non_blocking=True)

                    rgb, rgbd, y = maybe_mixup_batch(rgb, rgbd, y, cfg.mixup_alpha, cfg.mixup_prob)
                    rgb, rgbd, y = maybe_cutmix_batch(rgb, rgbd, y, cfg.cutmix_alpha, cfg.cutmix_prob)

                    optimizer.zero_grad(set_to_none=True)
                    rp, dp, fp = model(rgb, rgbd)
                    loss = 0.0
                    if rgb_w > 0:
                        loss = loss + rgb_w * criterion(rp, y)
                    if rgbd_w > 0:
                        loss = loss + rgbd_w * criterion(dp, y)
                    if fusion_w > 0:
                        loss = loss + fusion_w * criterion(fp, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if use_ema and ema_model is not None:
                        update_ema_model(ema_model, model, cfg.ema_decay)

                    bs = y.size(0)
                    if eval_metric == 'rgb':
                        t_mae_sum += torch.mean(torch.abs(rp.detach() - y)).item() * bs
                    elif eval_metric == 'rgbd':
                        t_mae_sum += torch.mean(torch.abs(dp.detach() - y)).item() * bs
                    else:
                        t_mae_sum += torch.mean(torch.abs(fp.detach() - y)).item() * bs
                    n_t += bs

                # ---- validate ----------------------------------------------
                model.eval()
                v_mae_sum, n_v = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        rgb_b = batch['rgb'].to(device, non_blocking=True)
                        rgbd_b = batch['rgbd'].to(device, non_blocking=True)
                        y_b = batch['dry_weight'].to(device, non_blocking=True)
                        rp, dp, fp = model(rgb_b, rgbd_b)
                        bs = y_b.size(0)
                        if eval_metric == 'rgb':
                            v_mae_sum += torch.mean(torch.abs(rp - y_b)).item() * bs
                        elif eval_metric == 'rgbd':
                            v_mae_sum += torch.mean(torch.abs(dp - y_b)).item() * bs
                        else:
                            v_mae_sum += torch.mean(torch.abs(fp - y_b)).item() * bs
                        n_v += bs

                t_mae = t_mae_sum / max(1, n_t)
                v_mae = v_mae_sum / max(1, n_v)
                train_history.append(t_mae)
                val_history.append(v_mae)
                span = min(cfg.best_mae_window, len(val_history))
                smooth = float(np.mean(val_history[-span:]))
                lr_now = optimizer.param_groups[0]['lr']
                tag = f'[{fold_label}] ' if fold_label else ''
                print(
                    f"  {tag}{phase_name} ep {ep+1:>3}/{num_epochs} (global {global_ep})  "
                    f"lr={lr_now:.2e}  train={t_mae:.4f}  val={v_mae:.4f}  "
                    f"smooth({span})={smooth:.4f}"
                )

                if v_mae < best_val_mae:
                    best_val_mae = v_mae
                    target = ema_model if use_ema and ema_model is not None else model
                    save_checkpoint(best_path, target)
                    best_epoch_index = global_ep
                    print(f"    ★ saved best model (val_mae={v_mae:.4f})")

                if stopper is not None and stopper.step(smooth):
                    print(f"  early stop at ep {ep+1} (best_val={best_val_mae:.4f})")
                    break

                if (ep + 1) % 5 == 0 or (ep + 1) == num_epochs:
                    save_training_curves(train_history, val_history, cfg.out_dir,
                                         suffix=fold_suffix, best_epoch=best_epoch_index,
                                         phase_boundaries=phase_boundaries)
        except KeyboardInterrupt:
            last_path = best_path.replace('best_model', 'last_model')
            target = ema_model if use_ema and ema_model is not None else model
            save_checkpoint(last_path, target)
            print(f"\n  interrupted during {phase_name} — saved current → {last_path}")
            print(f"  (best checkpoint untouched at {best_path}, val_mae={best_val_mae:.4f})")

    # ==================================================================
    # Phase 1 — RGB branch only
    # ==================================================================
    _freeze_module(model.rgbd_branch)
    _freeze_module(model.fusion)
    _freeze_module(model.fusion_in_dropout)
    # Also freeze shared spatial attention for RGBD if not shared
    _unfreeze_module(model.rgb_branch)

    _run_phase(
        phase_name='Phase 1: RGB branch',
        num_epochs=cfg.phase1_epochs,
        max_lr=cfg.phase1_lr,
        loss_weights=(1.0, 0.0, 0.0),
        eval_metric='rgb',
    )
    phase_boundaries.append(len(train_history))

    # ==================================================================
    # Phase 2 — RGBD branch only
    # ==================================================================
    _freeze_module(model.rgb_branch)
    _freeze_module(model.fusion)
    _unfreeze_module(model.rgbd_branch)

    _run_phase(
        phase_name='Phase 2: RGBD branch',
        num_epochs=cfg.phase2_epochs,
        max_lr=cfg.phase2_lr,
        loss_weights=(0.0, 1.0, 0.0),
        eval_metric='rgbd',
    )
    phase_boundaries.append(len(train_history))

    # ==================================================================
    # Phase 3 — Joint fine-tuning (everything)
    # ==================================================================
    best_val_mae = float('inf')  # reset for phase 3
    best_epoch_index = None

    _unfreeze_module(model.rgb_branch)
    _unfreeze_module(model.rgbd_branch)
    _unfreeze_module(model.fusion)
    _unfreeze_module(model.fusion_in_dropout)

    _run_phase(
        phase_name='Phase 3: Joint fine-tune',
        num_epochs=cfg.phase3_epochs,
        max_lr=cfg.phase3_lr,
        loss_weights=(RGB_LOSS_WEIGHT, RGBD_LOSS_WEIGHT, FUSION_LOSS_WEIGHT),
        eval_metric='fused',
        branch_lr_scale=cfg.phase3_branch_lr_scale,
        use_early_stop=True,
    )

    # Ensure checkpoint exists
    if not os.path.exists(best_path):
        target = ema_model if use_ema and ema_model is not None else model
        save_checkpoint(best_path, target)

    save_training_curves(train_history, val_history, cfg.out_dir,
                         suffix=fold_suffix, best_epoch=best_epoch_index,
                         phase_boundaries=phase_boundaries)

    return best_path, train_history, val_history, best_val_mae, best_epoch_index


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
    print(f'[train] device={device}  3-phase: {cfg.phase1_epochs} (RGB) + '
          f'{cfg.phase2_epochs} (RGBD) + {cfg.phase3_epochs} (joint) = {total_epochs} total')

    fold_loaders = make_fold_loaders(cfg)
    total_folds = len(fold_loaders)
    fold_results = []

    for fold_idx, (train_loader, val_loader, label) in enumerate(fold_loaders, 1):
        fold_suffix = f'_{label}' if total_folds > 1 else ''
        model = LettuceSAMFusionNet(
            drop_path_prob=cfg.drop_path_prob,
            rgb_widths=cfg.rgb_widths,
            rgbd_widths=cfg.rgbd_widths,
            embed_dim=cfg.embed_dim,
            spatial_kernel=cfg.spatial_kernel,
            spatial_layers=cfg.spatial_layers,
            spatial_dropout=cfg.spatial_dropout,
            share_spatial_attn=cfg.share_spatial_attn,
        ).to(device)
        print(f"[train] training fusion regressor ({label})...")
        best_full, train_hist, val_hist, best_val, best_epoch = train_fusion_regressor(
            cfg,
            model,
            train_loader,
            val_loader,
            device,
            fold_suffix=fold_suffix,
            fold_label=label,
        )
        save_training_curves(train_hist, val_hist, cfg.out_dir, suffix=fold_suffix, best_epoch=best_epoch)
        fold_results.append((label, best_full, best_val))
        print(f"[train] fold {label} best checkpoint: {best_full} (val_mae={best_val:.4f})")

    if total_folds > 1:
        print('\nCross-validation summary:')
        for label, path, best_val in fold_results:
            print(f"  - {label}: best_val_mae={best_val:.4f} ({path})")
        avg_val = float(np.mean([val for _, _, val in fold_results]))
        print(f"Average best val MAE: {avg_val:.4f}")

        best_label, best_path, best_val = min(fold_results, key=lambda x: x[2])
        canonical_path = Path(cfg.out_dir) / 'best_model_v8.pth'
        dest_path = str(canonical_path)
        if os.path.abspath(best_path) != os.path.abspath(dest_path):
            shutil.copy2(best_path, dest_path)
            print(f"Copied best fold {best_label} checkpoint to {dest_path} (val_mae={best_val:.4f})")
    else:
        label, path, best_val = fold_results[0]
        print(f"Done. Best full model saved to: {path} (val_mae={best_val:.4f})")


if __name__ == '__main__':
    import multiprocessing as mp

    mp.freeze_support()
    main()
