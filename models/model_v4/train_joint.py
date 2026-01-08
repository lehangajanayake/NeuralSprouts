"""
Joint (end-to-end) training for model_v4: All branches trained simultaneously from scratch.
Optimizes directly for dry_weight MAE, optionally with auxiliary classification loss.
No staged training—simpler, potentially better generalization for the regression task.
"""
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import logging

from dataloader import PlantDatasetV4, group_aware_train_val_split
from model import LettuceMultiBranchCNN


@dataclass
class JointTrainConfig:
    # Data paths
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'

    # Training params
    batch_size: int = 64
    num_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loss weights - pure regression mode (no classification)
    mae_weight: float = 1.0      # Primary: dry_weight MAE (competition metric)
    cls_weight: float = 0.0       # Set to 0 for pure regression (classification ignored)

    val_ratio: float = 0.2
    seed: int = 42
    patience: int = 100

    out_dir: str = '.'
    debug: bool = True
    log_name: str = 'train_joint'


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


def make_loaders(cfg: JointTrainConfig, logger: logging.Logger) -> Tuple[DataLoader, DataLoader, Dict]:
    df = pd.read_csv(cfg.train_csv)
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'id'}, inplace=True)
    
    train_indices, val_indices = group_aware_train_val_split(df, val_ratio=cfg.val_ratio, seed=cfg.seed)
    logger.info(f"[data] Group-aware split: train={len(train_indices)}, val={len(val_indices)}")
    
    train_ds = PlantDatasetV4(
        cfg.rgb_dir, cfg.depth_dir, cfg.train_csv,
        augment=False, seed=cfg.seed, enable_cache=True, num_views=1
    )
    val_ds = PlantDatasetV4(
        cfg.rgb_dir, cfg.depth_dir, cfg.train_csv,
        augment=False, seed=cfg.seed, enable_cache=True, num_views=1
    )
    
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    
    num_classes = len(train_ds.variety2idx)
    inv_idx = {idx: name for name, idx in train_ds.variety2idx.items()}
    class_names = [inv_idx[i] for i in range(num_classes)]
    
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed)
    
    loader_kwargs = {'generator': g, 'pin_memory': pin_memory, 'num_workers': num_workers}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

    meta = {
        'num_classes': num_classes,
        'class_names': class_names,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
    }
    return train_loader, val_loader, meta


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float('inf')

    def step(self, val_loss: float) -> bool:
        if val_loss < (self.best - self.min_delta):
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)


def load_checkpoint(path: str, model: nn.Module, device: torch.device):
    model.load_state_dict(torch.load(path, map_location=device))


def setup_logger(out_dir: str, name: str = 'train_joint') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh = logging.FileHandler(str(Path(out_dir) / 'debug_joint.log'), encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def train_joint(cfg: JointTrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader,
                device, logger: logging.Logger, num_classes: int):
    """Train all branches jointly, optimizing for dry_weight MAE."""
    
    criterion_mae = nn.L1Loss()
    criterion_cls = nn.CrossEntropyLoss() if cfg.cls_weight > 0 else None
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience)

    best_path = str(Path(cfg.out_dir) / 'best_joint_v4.pth')
    train_mae_hist, val_mae_hist = [], []
    train_cls_hist, val_cls_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    best_epoch = -1

    mode = "pure regression" if cfg.cls_weight == 0 else f"multi-task (MAE={cfg.mae_weight}, cls={cfg.cls_weight})"
    logger.info(f"[train] Starting joint training in {mode} mode")
    logger.info(f"[train] Epochs={cfg.num_epochs}, LR={cfg.lr}, patience={cfg.patience}")

    for epoch in range(cfg.num_epochs):
        model.train()
        train_mae_sum, train_cls_sum = 0.0, 0.0
        n_train, correct_train = 0, 0

        for batch in train_loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            rgbd = batch['rgbd'].to(device, non_blocking=True)
            y = batch['dry_weight'].to(device, non_blocking=True)
            y_cls = batch['variety_class'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, rgbd_pred, fusion_pred = model(rgb, rgbd)
            
            # Pure regression loss (or multi-task if cls_weight > 0)
            loss_mae = criterion_mae(fusion_pred, y)
            if cfg.cls_weight > 0 and criterion_cls is not None:
                loss_cls = criterion_cls(logits, y_cls)
                loss = cfg.mae_weight * loss_mae + cfg.cls_weight * loss_cls
            else:
                loss_cls = torch.tensor(0.0, device=device)
                loss = loss_mae
            
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_mae_sum += loss_mae.item() * bs
            train_cls_sum += loss_cls.item() * bs
            n_train += bs
            
            if cfg.cls_weight > 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    correct_train += (preds == y_cls).sum().item()

        model.eval()
        val_mae_sum, val_cls_sum = 0.0, 0.0
        n_val, correct_val = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                rgbd = batch['rgbd'].to(device, non_blocking=True)
                y = batch['dry_weight'].to(device, non_blocking=True)
                y_cls = batch['variety_class'].to(device, non_blocking=True)

                logits, rgbd_pred, fusion_pred = model(rgb, rgbd)
                
                loss_mae = criterion_mae(fusion_pred, y)
                if cfg.cls_weight > 0 and criterion_cls is not None:
                    loss_cls = criterion_cls(logits, y_cls)
                else:
                    loss_cls = torch.tensor(0.0, device=device)
                
                bs = y.size(0)
                val_mae_sum += loss_mae.item() * bs
                val_cls_sum += loss_cls.item() * bs
                n_val += bs
                
                if cfg.cls_weight > 0:
                    preds = logits.argmax(dim=1)
                    correct_val += (preds == y_cls).sum().item()

        train_mae = train_mae_sum / max(1, n_train)
        val_mae = val_mae_sum / max(1, n_val)
        train_cls = train_cls_sum / max(1, n_train)
        val_cls = val_cls_sum / max(1, n_val)
        train_acc = correct_train / max(1, n_train)
        val_acc = correct_val / max(1, n_val)

        train_mae_hist.append(train_mae)
        val_mae_hist.append(val_mae)
        train_cls_hist.append(train_cls)
        val_cls_hist.append(val_cls)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Logging
        log_msg = f"[train] epoch {epoch+1}/{cfg.num_epochs} train_mae={train_mae:.4f} val_mae={val_mae:.4f}"
        if cfg.cls_weight > 0:
            log_msg += f" | train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        logger.info(log_msg)

        # Early stopping on validation MAE
        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
            best_epoch = epoch + 1
        
        if stopper.step(val_mae):
            logger.info(f"[train] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    # Load best model
    load_checkpoint(best_path, model, device)
    
    # Compute final confusion matrix if classification is enabled
    cm = None
    if cfg.cls_weight > 0 and cfg.debug:
        cm = _confusion_matrix(model, val_loader, device, num_classes)
        logger.info('[debug] Final validation confusion matrix:')
        logger.info(f"\n{cm}")
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums>0)
        logger.info('[debug] Row-normalized CM:')
        logger.info(f"\n{cm_norm}")
        cm_acc = float(cm.diagonal().sum()) / float(cm.sum()) if cm.sum() > 0 else 0.0
        logger.info(f"[debug] Final val accuracy (from CM): {cm_acc:.3f}")

    hist = {
        'train_mae': train_mae_hist,
        'val_mae': val_mae_hist,
        'train_cls': train_cls_hist,
        'val_cls': val_cls_hist,
        'train_acc': train_acc_hist,
        'val_acc': val_acc_hist,
        'best_epoch': best_epoch,
    }
    return best_path, hist


@torch.no_grad()
def _confusion_matrix(model: LettuceMultiBranchCNN, loader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for batch in loader:
        rgb = batch['rgb'].to(device, non_blocking=True)
        rgbd = batch['rgbd'].to(device, non_blocking=True)
        y = batch['variety_class'].to(device, non_blocking=True)
        logits, _, _ = model(rgb, rgbd)
        preds = logits.argmax(dim=1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[int(t.item()), int(p.item())] += 1
    return cm


def plot_histories(hist: Dict, out_dir: str, logger: logging.Logger):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"[plot] matplotlib not available: {e}")
        return

    epochs = range(1, len(hist['train_mae']) + 1)
    best_epoch = hist.get('best_epoch', None)

    # MAE plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist['train_mae'], label='train MAE', color='#1f77b4')
    plt.plot(epochs, hist['val_mae'], label='val MAE', color='#ff7f0e')
    if best_epoch and best_epoch > 0:
        plt.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.2, label=f'best@{best_epoch}')
    plt.title('Joint Training: MAE (dry_weight)')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_dir) / 'joint_mae_curve.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"[plot] Saved MAE curve to: {out_path}")

    # Classification plot (if enabled)
    if max(hist['train_cls']) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.plot(epochs, hist['train_cls'], label='train CE', color='#1f77b4')
        ax1.plot(epochs, hist['val_cls'], label='val CE', color='#ff7f0e')
        if best_epoch and best_epoch > 0:
            ax1.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.2, label=f'best@{best_epoch}')
        ax1.set_title('Joint Training: Classification Loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('CrossEntropy')
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        ax2.plot(epochs, hist['train_acc'], label='train acc', color='#1f77b4')
        ax2.plot(epochs, hist['val_acc'], label='val acc', color='#ff7f0e')
        if best_epoch and best_epoch > 0:
            ax2.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.2, label=f'best@{best_epoch}')
        ax2.set_title('Joint Training: Classification Accuracy')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.25)
        ax2.legend()

        plt.tight_layout()
        out_path = Path(out_dir) / 'joint_cls_curve.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"[plot] Saved classification curves to: {out_path}")


def main():
    cfg = JointTrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info('[init] Starting joint (end-to-end) training for model_v4')
    logger.info(f'[config] Pure regression mode (MAE only, ignoring classification)')

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'[device] {device}')

    train_loader, val_loader, meta = make_loaders(cfg, logger)
    num_classes = meta['num_classes']
    logger.info(f"[data] {num_classes} classes: {meta['class_names']}")

    model = LettuceMultiBranchCNN(num_classes=num_classes).to(device)
    logger.info(f"[model] LettuceMultiBranchCNN initialized (params={sum(p.numel() for p in model.parameters())})")

    try:
        best_path, hist = train_joint(cfg, model, train_loader, val_loader, device, logger, num_classes)
        logger.info(f"[done] Best model saved to: {best_path}")
        
        plot_histories(hist, cfg.out_dir, logger)
        
        logger.info("[done] Joint training complete")
    except KeyboardInterrupt:
        logger.warning("[interrupt] Training interrupted by user (KeyboardInterrupt)")
        logger.info("[interrupt] Attempting to save plots with partial history...")
        # No plot since we don't have hist yet in this scope if interrupted early
        logger.info("[interrupt] Exiting gracefully")


if __name__ == '__main__':
    main()
