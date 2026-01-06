import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN, set_requires_grad


@dataclass
class TrainConfig:
    # use augmented by default
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'

    batch_size: int = 64
    num_epochs_stage1: int = 30
    num_epochs_stage2: int = 40
    num_epochs_stage3: int = 60

    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-3
    lr_stage3: float = 5e-4

    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    seed: int = 42

    patience_stage1: int = 7
    patience_stage2: int = 10
    patience_stage3: int = 12

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
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    # Cache preprocessed tensors in CPU RAM to reduce per-epoch PIL decode/resize overhead.
    # This typically improves GPU utilization more reliably than trying to preload everything into VRAM.
    full = PlantDatasetV4(
        cfg.rgb_dir,
        cfg.depth_dir,
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

    meta = {
        'num_classes': len(full.variety2idx),
    }
    return train_loader, val_loader, meta


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


def load_checkpoint(path: str, model: nn.Module, device: torch.device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def stage1_train_rgb_classifier(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device):
    # Freeze RGBD and fusion
    set_requires_grad(model.rgb_branch, True)
    set_requires_grad(model.rgbd_branch, False)
    set_requires_grad(model.fusion, False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage1)

    best_path = str(Path(cfg.out_dir) / 'best_rgb_branch_v4.pth')

    for epoch in range(cfg.num_epochs_stage1):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        for batch in train_loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            rgbd = batch['rgbd'].to(device, non_blocking=True)
            y_cls = batch['variety_class'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(rgb, rgbd)
            loss = criterion(logits, y_cls)
            loss.backward()
            optimizer.step()

            bs = y_cls.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                rgbd = batch['rgbd'].to(device, non_blocking=True)
                y_cls = batch['variety_class'].to(device, non_blocking=True)

                logits, _, _ = model(rgb, rgbd)
                loss = criterion(logits, y_cls)
                bs = y_cls.size(0)
                val_loss_sum += loss.item() * bs
                n_val += bs

        val_loss = val_loss_sum / max(1, n_val)
        print(f"[stage1] epoch {epoch+1}/{cfg.num_epochs_stage1} train_ce={train_loss_sum/max(1,n_train):.4f} val_ce={val_loss:.4f}")

        if val_loss <= stopper.best:
            save_checkpoint(best_path, model)
        if stopper.step(val_loss):
            print(f"[stage1] early stop at epoch {epoch+1} (best val_ce={stopper.best:.4f})")
            break

    return best_path


def stage2_train_rgbd_regressor(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device):
    # Freeze RGB and fusion
    set_requires_grad(model.rgb_branch, False)
    set_requires_grad(model.rgbd_branch, True)
    set_requires_grad(model.fusion, False)

    criterion = nn.L1Loss()  # MAE
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage2)

    best_path = str(Path(cfg.out_dir) / 'best_rgbd_branch_v4.pth')

    for epoch in range(cfg.num_epochs_stage2):
        model.train()
        train_mae_sum, n_train = 0.0, 0

        for batch in train_loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            rgbd = batch['rgbd'].to(device, non_blocking=True)
            y = batch['dry_weight'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            _, rgbd_pred, _ = model(rgb, rgbd)
            loss = criterion(rgbd_pred, y)
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

                _, rgbd_pred, _ = model(rgb, rgbd)
                loss = criterion(rgbd_pred, y)
                bs = y.size(0)
                val_mae_sum += loss.item() * bs
                n_val += bs

        val_mae = val_mae_sum / max(1, n_val)
        print(f"[stage2] epoch {epoch+1}/{cfg.num_epochs_stage2} train_mae={train_mae_sum/max(1,n_train):.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
        if stopper.step(val_mae):
            print(f"[stage2] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path


def stage3_train_fusion(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device):
    # Unfreeze all
    set_requires_grad(model.rgb_branch, True)
    set_requires_grad(model.rgbd_branch, True)
    set_requires_grad(model.fusion, True)

    criterion = nn.L1Loss()  # competition metric: MAE on fusion output
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage3)

    best_path = str(Path(cfg.out_dir) / 'best_model_v4.pth')

    for epoch in range(cfg.num_epochs_stage3):
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

        val_mae = val_mae_sum / max(1, n_val)
        print(f"[stage3] epoch {epoch+1}/{cfg.num_epochs_stage3} train_mae={train_mae_sum/max(1,n_train):.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
        if stopper.step(val_mae):
            print(f"[stage3] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, meta = make_loaders(cfg)
    num_classes = int(meta['num_classes'])
    if num_classes != 4:
        print(f"[warn] expected 4 classes, but found {num_classes} unique Variety values")

    model = LettuceMultiBranchCNN(num_classes=num_classes).to(device)

    # Stage 1
    print('[stage1] training RGB classifier...')
    rgb_ckpt = stage1_train_rgb_classifier(cfg, model, train_loader, val_loader, device)
    load_checkpoint(rgb_ckpt, model, device)

    # Stage 2
    print('[stage2] training RGBD regressor...')
    rgbd_ckpt = stage2_train_rgbd_regressor(cfg, model, train_loader, val_loader, device)
    load_checkpoint(rgbd_ckpt, model, device)

    # Stage 3
    print('[stage3] training fusion model...')
    best_full = stage3_train_fusion(cfg, model, train_loader, val_loader, device)

    print(f"Done. Best full model saved to: {best_full}")


if __name__ == '__main__':
    # Windows multi-worker safety (even though we default to 0 workers)
    import multiprocessing as mp

    mp.freeze_support()
    main()
