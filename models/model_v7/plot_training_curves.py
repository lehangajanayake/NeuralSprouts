"""
Quick script to re-run training with logging and plot the curves.
This will show train vs val loss to check for overfitting.
"""
import os
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataloader import PlantDatasetV4
from model import LettuceMultiBranchCNN, set_requires_grad


@dataclass
class TrainConfig:
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'

    batch_size: int = 64
    num_epochs_stage1: int = 100
    num_epochs_stage2: int = 100
    num_epochs_stage3: int = 200

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
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage_name: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        rgb = batch['rgb'].to(device)
        rgbd = batch['rgbd'].to(device)
        y = batch['dry_weight'].to(device)
        y_cls = batch['variety_class'].to(device)

        optimizer.zero_grad()
        logits, rgbd_pred, fusion_pred = model(rgb, rgbd)

        if stage_name == "stage1":
            loss = loss_fn(logits, y_cls)
        elif stage_name == "stage2":
            loss = loss_fn(rgbd_pred, y)
        else:  # stage3
            loss = loss_fn(fusion_pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        count += y.size(0)

    return total_loss / max(1, count)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    stage_name: str = "val",
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device)
            rgbd = batch['rgbd'].to(device)
            y = batch['dry_weight'].to(device)
            y_cls = batch['variety_class'].to(device)

            logits, rgbd_pred, fusion_pred = model(rgb, rgbd)

            if stage_name == "stage1":
                loss = loss_fn(logits, y_cls)
            elif stage_name == "stage2":
                loss = loss_fn(rgbd_pred, y)
            else:  # stage3
                loss = loss_fn(fusion_pred, y)

            total_loss += loss.item() * y.size(0)
            count += y.size(0)

    return total_loss / max(1, count)


def main(cfg: TrainConfig = None):
    if cfg is None:
        cfg = TrainConfig()
    seed_everything(cfg.seed, deterministic=True)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "Metal (Apple GPU)"
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = "CUDA (NVIDIA GPU)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    print(f"Device: {device_name} ({device})")

    # Load data
    ds = PlantDatasetV4(cfg.rgb_dir, cfg.depth_dir, cfg.train_csv, augment=True, seed=cfg.seed)
    train_size = int(len(ds) * (1 - cfg.val_ratio))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = LettuceMultiBranchCNN(num_classes=len(ds.variety2idx)).to(device)

    # Storage for metrics
    all_metrics = {
        'stage1': {'train': [], 'val': []},
        'stage2': {'train': [], 'val': []},
        'stage3': {'train': [], 'val': []},
    }

    print(f"\n{'='*80}")
    print(f"Stage 1: RGB Classification (Variety Prediction)")
    print(f"{'='*80}")

    set_requires_grad(model, False)
    model.rgb_branch.requires_grad_(True)
    model.fusion_mlp.requires_grad_(True)

    optimizer = optim.Adam(
        list(model.rgb_branch.parameters()) + list(model.fusion_mlp.parameters()),
        lr=cfg.lr_stage1,
        weight_decay=cfg.weight_decay,
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs_stage1 + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, "stage1")
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device, "stage1")

        all_metrics['stage1']['train'].append(train_loss)
        all_metrics['stage1']['val'].append(val_loss)

        if epoch <= 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = str(Path(cfg.out_dir) / 'best_rgb_branch_v7.pth')
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[stage1] epoch {epoch}/{cfg.num_epochs_stage1} train_ce={train_loss:.6f} val_ce={val_loss:.6f}")

        if patience_counter >= cfg.patience_stage1:
            print(f"[stage1] epoch {epoch} early stop (best val_ce={best_val_loss:.6f})")
            break

    state = torch.load(str(Path(cfg.out_dir) / 'best_rgb_branch_v7.pth'), map_location=device)
    model.load_state_dict(state)

    print(f"\n{'='*80}")
    print(f"Stage 2: RGBD Regression (Dry Weight)")
    print(f"{'='*80}")

    set_requires_grad(model, False)
    model.rgbd_branch.requires_grad_(True)

    optimizer = optim.Adam(
        model.rgbd_branch.parameters(),
        lr=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
    loss_fn = nn.L1Loss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs_stage2 + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, "stage2")
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device, "stage2")

        all_metrics['stage2']['train'].append(train_loss)
        all_metrics['stage2']['val'].append(val_loss)

        if epoch <= 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = str(Path(cfg.out_dir) / 'best_rgbd_branch_v7.pth')
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[stage2] epoch {epoch}/{cfg.num_epochs_stage2} train_mae={train_loss:.6f} val_mae={val_loss:.6f}")

        if patience_counter >= cfg.patience_stage2:
            print(f"[stage2] epoch {epoch} early stop (best val_mae={best_val_loss:.6f})")
            break

    state = torch.load(str(Path(cfg.out_dir) / 'best_rgbd_branch_v7.pth'), map_location=device)
    model.load_state_dict(state)

    print(f"\n{'='*80}")
    print(f"Stage 3: Fusion Network (Final Prediction)")
    print(f"{'='*80}")

    set_requires_grad(model, True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr_stage3,
        weight_decay=cfg.weight_decay,
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=195, eta_min=1e-5)
    loss_fn = nn.L1Loss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs_stage3 + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, "stage3")
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device, "stage3")

        all_metrics['stage3']['train'].append(train_loss)
        all_metrics['stage3']['val'].append(val_loss)

        if epoch <= 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = str(Path(cfg.out_dir) / 'best_model_v7.pth')
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[stage3] epoch {epoch}/{cfg.num_epochs_stage3} train_mae={train_loss:.6f} val_mae={val_loss:.6f}")

        if patience_counter >= cfg.patience_stage3:
            print(f"[stage3] epoch {epoch} early stop (best val_mae={best_val_loss:.6f})")
            break

    print(f"\nDone. Best full model saved to: best_model_v7.pth")

    # Save metrics
    metrics_path = 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Plot curves
    plot_training_curves(all_metrics)


def plot_training_curves(metrics: Dict):
    """Plot train vs val loss for all stages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Model v7: Training Curves (Train vs Validation)', fontsize=14, fontweight='bold')

    stages = ['stage1', 'stage2', 'stage3']
    titles = ['Stage 1: RGB Classification', 'Stage 2: RGBD Regression', 'Stage 3: Fusion Network']
    ylabels = ['Cross-Entropy Loss', 'MAE (L1)', 'MAE (L1)']

    for idx, (stage, title, ylabel) in enumerate(zip(stages, titles, ylabels)):
        ax = axes[idx]
        train_losses = metrics[stage]['train']
        val_losses = metrics[stage]['val']
        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=2, alpha=0.7)
        ax.fill_between(epochs, train_losses, val_losses, alpha=0.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add best val loss annotation
        best_val_idx = np.argmin(val_losses)
        best_val = val_losses[best_val_idx]
        ax.axvline(best_val_idx + 1, color='green', linestyle='--', alpha=0.5)
        ax.text(best_val_idx + 1, best_val, f'  Best: {best_val:.4f}', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig('training_curves_v7.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: training_curves_v7.png")
    plt.show()


if __name__ == '__main__':
    main()
