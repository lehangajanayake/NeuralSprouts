import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN


@dataclass
class TrainConfig:
    # use augmented RGB-only dataset by default
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'

    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    seed: int = 42
    patience: int = 50

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
    # Cache preprocessed tensors in CPU RAM to reduce per-epoch PIL decode/resize overhead.
    # This typically improves GPU utilization more reliably than trying to preload everything into VRAM.
    full = PlantDatasetV4(
        cfg.rgb_dir,
        cfg.train_csv,
        augment=True,
        seed=cfg.seed,
        enable_cache=True,
        num_views=1,
    )

    val_size = int(len(full) * float(cfg.val_ratio))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed)

    # Optional: build cache once up-front to avoid lazy cache misses during epoch 1.
    # Set max_base_items to an int to limit RAM usage.
    full.build_cache(max_base_items=None)

    loader_kwargs = {}
    # persistent_workers only applies when num_workers > 0
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
        """Returns True if should stop."""
        if metric < self.best - 1e-9:
            self.best = float(metric)
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def save_checkpoint(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)


def train_regressor(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device: torch.device) -> str:
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience)

    best_path = str(Path(cfg.out_dir) / 'best_model_v4.pth')

    for epoch in range(cfg.num_epochs):
        model.train()
        train_mae_sum, n_train = 0.0, 0

        for batch in train_loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            y = batch['dry_weight'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(rgb)
            loss = criterion(preds, y)
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
                y = batch['dry_weight'].to(device, non_blocking=True)

                preds = model(rgb)
                loss = criterion(preds, y)
                bs = y.size(0)
                val_mae_sum += loss.item() * bs
                n_val += bs

        train_mae = train_mae_sum / max(1, n_train)
        val_mae = val_mae_sum / max(1, n_val)
        print(f"[regressor] epoch {epoch+1}/{cfg.num_epochs} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
        if stopper.step(val_mae):
            print(f"[regressor] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = make_loaders(cfg)
    model = LettuceMultiBranchCNN().to(device)

    print('[regressor] training RGB-only model...')
    best_full = train_regressor(cfg, model, train_loader, val_loader, device)

    print(f"Done. Best model saved to: {best_full}")


if __name__ == '__main__':
    # Windows multi-worker safety (even though we default to 0 workers)
    import multiprocessing as mp

    mp.freeze_support()
    main()
