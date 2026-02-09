import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class TrainConfig:
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'

    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3    
    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    seed: int = 42
    patience: int = 50
    # Matches 1 + num_aug_per_image from preprocess.py
    outputs_per_original: int = 31

    out_dir: str = '.'


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


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    full = PlantDatasetV4(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=True,
        seed=cfg.seed,
        enable_cache=True,
        num_views=1,
    )

    if len(full.df) == 0:
        raise ValueError('No samples found in augmented CSV; check preprocessing paths.')

    has_original_ids = 'original_id' in full.df.columns
    if has_original_ids:
        group_ids = full.df['original_id'].astype(int).to_numpy()
    else:
        outputs_per_original = max(1, int(cfg.outputs_per_original))
        group_ids = ((full.df['id'].astype(int) - 1) // outputs_per_original).to_numpy()
    unique_groups = np.unique(group_ids)
    total_groups = len(unique_groups)
    if total_groups < 2:
        raise ValueError('Need at least two unique originals to create a train/val split. Reduce val_ratio or add more data.')

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(unique_groups)

    val_group_count = max(1, int(round(total_groups * float(cfg.val_ratio))))
    if val_group_count >= total_groups:
        val_group_count = total_groups - 1
    val_group_ids = set(unique_groups[:val_group_count])

    base_indices = np.arange(len(full.df))
    train_indices = [int(i) for i, g in zip(base_indices, group_ids) if g not in val_group_ids]
    val_indices = [int(i) for i, g in zip(base_indices, group_ids) if g in val_group_ids]

    if not train_indices or not val_indices:
        hint = 'Ensure original_id exists in the CSV' if has_original_ids else 'Adjust val_ratio or outputs_per_original'
        raise ValueError(f'Group-based split resulted in empty train/val set. {hint}.')

    train_ds = Subset(full, train_indices)
    val_ds = Subset(full, val_indices)

    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed)

    full.build_cache(max_base_items=None)

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


def save_training_curves(train_history: List[float], val_history: List[float], out_dir: str) -> None:
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

    out_path = Path(out_dir) / 'training_curves.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved training curves to: {out_path}')


def train_fusion_regressor(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device):
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience)

    best_path = str(Path(cfg.out_dir) / 'best_model_v4.pth')
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(cfg.num_epochs):
        model.train()
        train_mae_sum, n_train = 0.0, 0

        for batch in train_loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            rgbd = batch['rgbd'].to(device, non_blocking=True)
            y = batch['dry_weight'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            _, _, fusion_pred = model(rgb, rgbd)
            loss = criterion(fusion_pred, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_mae_sum += loss.item() * bs
            n_train += bs

        model.eval()
        val_mae_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                rgbd = batch['rgbd'].to(device, non_blocking=True)
                y = batch['dry_weight'].to(device, non_blocking=True)

                _, _, fusion_pred = model(rgb, rgbd)
                loss = criterion(fusion_pred, y)
                bs = y.size(0)
                val_mae_sum += loss.item() * bs
                n_val += bs

        train_mae = train_mae_sum / max(1, n_train)
        val_mae = val_mae_sum / max(1, n_val)
        train_history.append(train_mae)
        val_history.append(val_mae)
        print(f"[train] epoch {epoch+1}/{cfg.num_epochs} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
        if stopper.step(val_mae):
            print(f"[train] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path, train_history, val_history


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = make_loaders(cfg)
    model = LettuceMultiBranchCNN().to(device)

    print('[train] training fusion regressor...')
    best_full, train_hist, val_hist = train_fusion_regressor(cfg, model, train_loader, val_loader, device)
    save_training_curves(train_hist, val_hist, cfg.out_dir)

    print(f"Done. Best full model saved to: {best_full}")


if __name__ == '__main__':
    import multiprocessing as mp

    mp.freeze_support()
    main()
