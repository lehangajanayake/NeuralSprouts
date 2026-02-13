import copy
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

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

    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_min_lr: float = 1e-6

    val_ratio: float = 0.2
    seed: int = 43
    patience: int = 100
    outputs_per_original: int = 41
    num_folds: int = 1
    group_by_original: bool = True

    preload_to_gpu: bool = False
    preload_device: str = 'cuda'

    out_dir: str = '.'
    blacklist_ids: Tuple[int, ...] = (163,)
    best_mae_window: int = 5
    ema_decay: float = 0.995
    drop_path_prob: float = 0.1
    rgb_widths: Tuple[int, ...] = (32, 64, 96, 128)
    rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128)
    embed_dim: int = 256
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    initial_frozen_rgb_blocks: int = 3
    initial_frozen_rgbd_blocks: int = 3
    unfreeze_interval: int = 5
    rgb_unfreeze_interval: int = 5
    rgbd_unfreeze_interval: int = 7
    unfreeze_start_epoch: int =7
    branch_warmup_epochs: int = 2
    branch_warmup_scale: float = 0.3
    huber_delta: float = 0.3


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


def apply_block_freezing(blocks: Sequence[nn.Module], frozen_count: int) -> None:
    frozen = max(0, min(int(frozen_count), len(blocks)))
    for idx, block in enumerate(blocks):
        requires = idx >= frozen
        for param in block.parameters():
            param.requires_grad = requires


def maybe_unfreeze_blocks(
    epoch: int,
    start_epoch: int,
    interval: int,
    current_frozen: int,
    blocks: Sequence[nn.Module],
    branch_name: str,
    on_unfreeze: Callable[[int], None] | None = None,
) -> Tuple[int, int | None]:
    if current_frozen <= 0:
        return current_frozen, None
    if (epoch + 1) < max(1, start_epoch):
        return current_frozen, None
    if interval <= 0:
        interval_trigger = (epoch + 1) == max(1, start_epoch)
    else:
        elapsed = (epoch + 1) - max(1, start_epoch)
        interval_trigger = elapsed >= 0 and (elapsed % interval == 0)
    if not interval_trigger:
        return current_frozen, None
    new_frozen = max(0, current_frozen - 1)
    if new_frozen == current_frozen:
        return current_frozen, None
    apply_block_freezing(blocks, new_frozen)
    print(f"[train] unfreezing {branch_name} block index {new_frozen} (reason=schedule)")
    if on_unfreeze is not None:
        on_unfreeze(new_frozen)
    return new_frozen, new_frozen


def start_block_warmup(
    tracker: Dict[Tuple[str, int], Dict[str, int]],
    branch: str,
    block_idx: int,
    epochs: int,
) -> None:
    epochs = max(0, int(epochs))
    if epochs <= 0:
        return
    tracker[(branch, block_idx)] = {
        'remaining': epochs,
        'total': epochs,
    }


def apply_block_warmup_scaling(
    tracker: Dict[Tuple[str, int], Dict[str, int]],
    blocks_map: Dict[str, Sequence[nn.Module]],
    min_scale: float,
) -> None:
    if not tracker:
        return
    min_scale = float(min_scale)
    if min_scale >= 1.0:
        return
    clamped = max(0.0, min(1.0, min_scale))
    for (branch, idx), state in tracker.items():
        remaining = int(state.get('remaining', 0))
        total = max(1, int(state.get('total', 1)))
        if remaining <= 0:
            continue
        seq = blocks_map.get(branch)
        if seq is None or idx < 0 or idx >= len(seq):
            continue
        progress = 1.0 - (remaining / total)
        scale = clamped + (1.0 - clamped) * progress
        block = seq[idx]
        for param in block.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)


def decay_block_warmups(tracker: Dict[Tuple[str, int], Dict[str, int]]) -> None:
    if not tracker:
        return
    to_remove: List[Tuple[str, int]] = []
    for key, state in tracker.items():
        remaining = int(state.get('remaining', 0)) - 1
        state['remaining'] = remaining
        if remaining <= 0:
            to_remove.append(key)
    for key in to_remove:
        tracker.pop(key, None)


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


def _compute_group_ids(full_df, cfg: TrainConfig) -> Tuple[np.ndarray, bool]:
    if not cfg.group_by_original:
        ids = np.arange(len(full_df), dtype=int)
        return ids, False
    has_original_ids = 'original_id' in full_df.columns
    if has_original_ids:
        group_ids = full_df['original_id'].astype(int).to_numpy()
    else:
        outputs_per_original = max(1, int(cfg.outputs_per_original))
        group_ids = ((full_df['id'].astype(int) - 1) // outputs_per_original).to_numpy()
    return group_ids, has_original_ids


def _split_group_indices(
    group_ids: np.ndarray,
    cfg: TrainConfig,
    has_original_ids: bool,
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
    group_ids, has_original_ids = _compute_group_ids(full.df, cfg)
    splits = _split_group_indices(group_ids, cfg, has_original_ids)

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


def save_training_curves(train_history: List[float], val_history: List[float], out_dir: str, suffix: str = '', best_epoch: int | None = None) -> None:
    if not train_history or not val_history:
        return
    if plt is None:
        print('matplotlib not available; skipping training curve plot.')
        return

    epochs = range(1, len(train_history) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_history, label='Train MAE')
    ax.plot(epochs, val_history, label='Val MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Training vs Validation MAE')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend()

    out_name = f'training_curves{suffix}.png' if suffix else 'training_curves.png'
    out_path = Path(out_dir) / out_name
    if best_epoch is not None and 1 <= best_epoch <= len(train_history):
        best_val = val_history[best_epoch - 1]
        ax.scatter([best_epoch], [best_val], color='red', s=40, label=f'Best epoch {best_epoch}')
        ax.legend()
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
    if cfg.huber_delta is not None and float(cfg.huber_delta) > 0.0:
        criterion = nn.SmoothL1Loss(beta=float(cfg.huber_delta))
    else:
        criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )
    stopper = EarlyStopper(cfg.patience)

    best_name = f'best_model_v8{fold_suffix}.pth' if fold_suffix else 'best_model_v8.pth'
    best_path = str(Path(cfg.out_dir) / best_name)
    train_history: List[float] = []
    val_history: List[float] = []
    mae_window = max(1, int(cfg.best_mae_window))
    use_ema = cfg.ema_decay is not None and float(cfg.ema_decay) > 0.0
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for param in ema_model.parameters():
            param.requires_grad_(False)
    best_saved = False
    best_epoch_index: int | None = None
    interrupted = False
    rgb_blocks = list(model.rgb_branch.features)
    rgbd_blocks = list(model.rgbd_branch.features)
    blocks_map: Dict[str, Sequence[nn.Module]] = {'RGB': rgb_blocks, 'RGBD': rgbd_blocks}
    block_warmups: Dict[Tuple[str, int], Dict[str, int]] = {}
    frozen_rgb = min(cfg.initial_frozen_rgb_blocks, len(rgb_blocks))
    frozen_rgbd = min(cfg.initial_frozen_rgbd_blocks, len(rgbd_blocks))
    if frozen_rgb > 0:
        apply_block_freezing(rgb_blocks, frozen_rgb)
        print(f"[train] freezing first {frozen_rgb} RGB blocks")
    if frozen_rgbd > 0:
        apply_block_freezing(rgbd_blocks, frozen_rgbd)
        print(f"[train] freezing first {frozen_rgbd} RGBD blocks")

    rgb_interval = cfg.rgb_unfreeze_interval if cfg.rgb_unfreeze_interval > 0 else cfg.unfreeze_interval
    rgbd_interval = cfg.rgbd_unfreeze_interval if cfg.rgbd_unfreeze_interval > 0 else cfg.unfreeze_interval

    def on_rgb_unfreeze(block_idx: int) -> None:
        start_block_warmup(block_warmups, 'RGB', block_idx, cfg.branch_warmup_epochs)

    def on_rgbd_unfreeze(block_idx: int) -> None:
        start_block_warmup(block_warmups, 'RGBD', block_idx, cfg.branch_warmup_epochs)

    try:
        for epoch in range(cfg.num_epochs):
            # Recompute freezing schedule before training so newly unfrozen blocks
            # participate in the current epoch instead of waiting one more.
            frozen_rgb, _ = maybe_unfreeze_blocks(
                epoch,
                cfg.unfreeze_start_epoch,
                rgb_interval,
                frozen_rgb,
                rgb_blocks,
                'RGB',
                on_rgb_unfreeze,
            )
            frozen_rgbd, _ = maybe_unfreeze_blocks(
                epoch,
                cfg.unfreeze_start_epoch,
                rgbd_interval,
                frozen_rgbd,
                rgbd_blocks,
                'RGBD',
                on_rgbd_unfreeze,
            )
            model.train()
            train_mae_sum, n_train = 0.0, 0
            train_sup_loss_sum = 0.0

            for batch in train_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                rgbd = batch['rgbd'].to(device, non_blocking=True)
                y = batch['dry_weight'].to(device, non_blocking=True)

                rgb, rgbd, y = maybe_mixup_batch(rgb, rgbd, y, cfg.mixup_alpha, cfg.mixup_prob)

                optimizer.zero_grad(set_to_none=True)
                rgb_pred, rgbd_pred, fusion_pred = model(rgb, rgbd)
                loss_rgb = criterion(rgb_pred, y)
                loss_rgbd = criterion(rgbd_pred, y)
                loss_fusion = criterion(fusion_pred, y)
                fusion_mae = torch.mean(torch.abs(fusion_pred - y))
                loss = (
                    RGB_LOSS_WEIGHT * loss_rgb
                    + RGBD_LOSS_WEIGHT * loss_rgbd
                    + FUSION_LOSS_WEIGHT * loss_fusion
                )
                loss.backward()
                apply_block_warmup_scaling(
                    block_warmups,
                    blocks_map,
                    cfg.branch_warmup_scale,
                )
                optimizer.step()
                if use_ema and ema_model is not None:
                    update_ema_model(ema_model, model, cfg.ema_decay)

                bs = y.size(0)
                train_sup_loss_sum += loss.item() * bs
                train_mae_sum += fusion_mae.item() * bs
                n_train += bs

            model.eval()
            val_mae_sum, n_val = 0.0, 0
            val_sup_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    rgb = batch['rgb'].to(device, non_blocking=True)
                    rgbd = batch['rgbd'].to(device, non_blocking=True)
                    y = batch['dry_weight'].to(device, non_blocking=True)

                    rgb_pred, rgbd_pred, fusion_pred = model(rgb, rgbd)
                    loss_rgb = criterion(rgb_pred, y)
                    loss_rgbd = criterion(rgbd_pred, y)
                    loss_fusion = criterion(fusion_pred, y)
                    fusion_mae = torch.mean(torch.abs(fusion_pred - y))
                    val_loss = (
                        RGB_LOSS_WEIGHT * loss_rgb
                        + RGBD_LOSS_WEIGHT * loss_rgbd
                        + FUSION_LOSS_WEIGHT * loss_fusion
                    )
                    bs = y.size(0)
                    val_sup_loss_sum += val_loss.item() * bs
                    val_mae_sum += fusion_mae.item() * bs
                    n_val += bs

            train_mae = train_mae_sum / max(1, n_train)
            val_mae = val_mae_sum / max(1, n_val)
            train_sup_loss = train_sup_loss_sum / max(1, n_train)
            val_sup_loss = val_sup_loss_sum / max(1, n_val)
            train_history.append(train_mae)
            val_history.append(val_mae)
            smooth_span = min(mae_window, len(val_history))
            smooth_val_mae = float(np.mean(val_history[-smooth_span:]))
            current_lr = optimizer.param_groups[0]['lr']
            prefix = f"[train][{fold_label}]" if fold_label else '[train]'
            print(
                f"{prefix} epoch {epoch+1}/{cfg.num_epochs} lr={current_lr:.3e} "
                f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} "
                f"smoothed_val_mae({smooth_span})={smooth_val_mae:.4f} "
                f"train_loss={train_sup_loss:.4f} val_loss={val_sup_loss:.4f}"
            )

            if smooth_val_mae <= stopper.best:
                target = ema_model if use_ema and ema_model is not None else model
                save_checkpoint(best_path, target)
                best_saved = True
                best_epoch_index = epoch + 1
            if stopper.step(smooth_val_mae):
                print(f"[train] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
                break

            scheduler.step(val_mae)
            decay_block_warmups(block_warmups)
    except KeyboardInterrupt:
        interrupted = True
        print('\n[train] KeyboardInterrupt received; saving current progress...')
        target = ema_model if use_ema and ema_model is not None else model
        save_checkpoint(best_path, target)
        best_saved = True

    target = ema_model if use_ema and ema_model is not None else model
    if not best_saved or not os.path.exists(best_path):
        save_checkpoint(best_path, target)
        best_saved = True
    if interrupted:
        print(f"[train] interrupted — best checkpoint saved to {best_path}")
    return best_path, train_history, val_history, stopper.best, best_epoch_index


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
