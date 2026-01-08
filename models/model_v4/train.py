import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import logging

from dataloader import PlantDatasetV4, group_aware_train_val_split
from model import LettuceMultiBranchCNN, set_requires_grad


@dataclass
class TrainConfig:
    # use augmented by default
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
    # Debug controls
    debug: bool = True
    # Logging
    log_name: str = 'train_v4'


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


def make_loaders(cfg: TrainConfig, logger: logging.Logger) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    # Load CSV and perform group-aware split (keeps all variants of an original in same split)
    df = pd.read_csv(cfg.train_csv)
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'id'}, inplace=True)
    
    train_indices, val_indices = group_aware_train_val_split(df, val_ratio=cfg.val_ratio, seed=cfg.seed)
    logger.info(f"[data] Group-aware split: train={len(train_indices)}, val={len(val_indices)}")
    
    # Build train dataset WITH augmentation
    train_ds = PlantDatasetV4(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=True,
        seed=cfg.seed,
        enable_cache=True,
        num_views=1,
    )
    
    # Build val dataset WITHOUT augmentation (use same indices for variety2idx consistency)
    val_ds = PlantDatasetV4(
        cfg.rgb_dir,
        cfg.depth_dir,
        cfg.train_csv,
        augment=False,  # No augmentation for validation
        seed=cfg.seed,
        enable_cache=True,
        num_views=1,
    )
    
    # Filter datasets by group-aware indices
    from torch.utils.data import Subset
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    
    # Precompute variety2idx from train_ds (shared for both)
    num_classes = len(train_ds.variety2idx)
    inv_idx = {idx: name for name, idx in train_ds.variety2idx.items()}
    class_names = [inv_idx[i] for i in range(num_classes)]
    
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed)
    
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        **loader_kwargs,
    )

    # Optional debug: class distribution in train/val splits
    if cfg.debug:
        try:
            train_counts = train_ds.df.iloc[train_indices]['VarietyClass'].value_counts().to_dict()
            val_counts = val_ds.df.iloc[val_indices]['VarietyClass'].value_counts().to_dict()
            logger.info(f"[debug] class mapping: {train_ds.variety2idx}")
            logger.info(f"[debug] train class counts: {[ (class_names[k], v) for k, v in train_counts.items() ]}")
            logger.info(f"[debug] val class counts: {[ (class_names[k], v) for k, v in val_counts.items() ]}")
        except Exception as e:
            logger.warning(f"[debug] failed to compute class distributions: {e}")

    meta = {
        'num_classes': num_classes,
        'class_names': class_names,
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


def stage1_train_rgb_classifier(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device, logger: logging.Logger):
    # Freeze RGBD and fusion
    set_requires_grad(model.rgb_branch, True)
    set_requires_grad(model.rgbd_branch, False)
    set_requires_grad(model.fusion, False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage1)

    best_path = str(Path(cfg.out_dir) / 'best_rgb_branch_v4.pth')

    # Track history for plotting
    train_hist, val_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    best_epoch = -1

    for epoch in range(cfg.num_epochs_stage1):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        correct_train = 0
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
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct_train += int((preds == y_cls).sum().item())

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        correct_val = 0
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
                preds = logits.argmax(dim=1)
                correct_val += int((preds == y_cls).sum().item())

        train_ce = train_loss_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        train_hist.append(train_ce)
        val_hist.append(val_loss)
        train_acc = correct_train / max(1, n_train)
        val_acc = correct_val / max(1, n_val)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        logger.info(f"[stage1] epoch {epoch+1}/{cfg.num_epochs_stage1} train_ce={train_ce:.4f} val_ce={val_loss:.4f} | train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_loss <= stopper.best:
            save_checkpoint(best_path, model)
            best_epoch = epoch + 1
        if stopper.step(val_loss):
            logger.info(f"[stage1] early stop at epoch {epoch+1} (best val_ce={stopper.best:.4f})")
            break

    return best_path, {"train": train_hist, "val": val_hist, "best_epoch": best_epoch, "train_acc": train_acc_hist, "val_acc": val_acc_hist}


def stage2_train_rgbd_regressor(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device, logger: logging.Logger):
    # Freeze RGB and fusion
    set_requires_grad(model.rgb_branch, False)
    set_requires_grad(model.rgbd_branch, True)
    set_requires_grad(model.fusion, False)

    criterion = nn.L1Loss()  # MAE
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage2)

    best_path = str(Path(cfg.out_dir) / 'best_rgbd_branch_v4.pth')

    # Track history for plotting
    train_hist, val_hist = [], []
    best_epoch = -1

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

        train_mae = train_mae_sum / max(1, n_train)
        val_mae = val_mae_sum / max(1, n_val)
        train_hist.append(train_mae)
        val_hist.append(val_mae)
        logger.info(f"[stage2] epoch {epoch+1}/{cfg.num_epochs_stage2} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
            best_epoch = epoch + 1
        if stopper.step(val_mae):
            logger.info(f"[stage2] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path, {"train": train_hist, "val": val_hist, "best_epoch": best_epoch}


def stage3_train_fusion(cfg: TrainConfig, model: LettuceMultiBranchCNN, train_loader, val_loader, device, logger: logging.Logger):
    # Unfreeze all
    set_requires_grad(model.rgb_branch, True)
    set_requires_grad(model.rgbd_branch, True)
    set_requires_grad(model.fusion, True)

    criterion = nn.L1Loss()  # competition metric: MAE on fusion output
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(cfg.patience_stage3)

    best_path = str(Path(cfg.out_dir) / 'best_model_v4.pth')

    # Track history for plotting
    train_hist, val_hist = [], []
    best_epoch = -1

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

        train_mae = train_mae_sum / max(1, n_train)
        val_mae = val_mae_sum / max(1, n_val)
        train_hist.append(train_mae)
        val_hist.append(val_mae)
        logger.info(f"[stage3] epoch {epoch+1}/{cfg.num_epochs_stage3} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")

        if val_mae <= stopper.best:
            save_checkpoint(best_path, model)
            best_epoch = epoch + 1
        if stopper.step(val_mae):
            logger.info(f"[stage3] early stop at epoch {epoch+1} (best val_mae={stopper.best:.4f})")
            break

    return best_path, {"train": train_hist, "val": val_hist, "best_epoch": best_epoch}


@torch.no_grad()
def _confusion_matrix(model: LettuceMultiBranchCNN, loader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    import numpy as np
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


def _plot_histories(stage_histories, out_dir: str, logger: logging.Logger):
    """Plot histories for all stages and save PNGs.

    stage_histories: List of tuples (title, y_label, hist_dict)
    where hist_dict has keys 'train', 'val', 'best_epoch'.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("[plot] matplotlib not available. Install with: pip install matplotlib")
        logger.warning(f"[plot] Skipping plots. Reason: {e}")
        return

    # Composite figure
    n = len(stage_histories)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.0 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (title, y_label, hist) in zip(axes, stage_histories):
        epochs = range(1, len(hist["train"]) + 1)
        ax.plot(epochs, hist["train"], label="train", color="#1f77b4")
        ax.plot(epochs, hist["val"], label="val", color="#ff7f0e")
        be = hist.get("best_epoch", None)
        if be is not None and be > 0 and be <= len(epochs):
            ax.axvline(be, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"best@{be}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend()

    out_path = Path(out_dir) / "training_curves_v4.png"
    fig.savefig(out_path, dpi=150)
    logger.info(f"[plot] Saved composite curves to: {out_path}")

    # Individual stage plots
    for idx, (title, y_label, hist) in enumerate(stage_histories, start=1):
        plt.figure(figsize=(7, 4))
        epochs = range(1, len(hist["train"]) + 1)
        plt.plot(epochs, hist["train"], label="train", color="#1f77b4")
        plt.plot(epochs, hist["val"], label="val", color="#ff7f0e")
        be = hist.get("best_epoch", None)
        if be is not None and be > 0 and be <= len(epochs):
            plt.axvline(be, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"best@{be}")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.25)
        plt.legend()
        stage_path = Path(out_dir) / f"stage{idx}_curve.png"
        plt.savefig(stage_path, dpi=150)
        plt.close()
        logger.info(f"[plot] Saved stage {idx} curve to: {stage_path}")


def setup_logger(out_dir: str, name: str = 'train_v4') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers if re-run in the same process
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh = logging.FileHandler(str(Path(out_dir) / 'debug.log'), encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main():
    cfg = TrainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info('[init] starting training run for model_v4')

    seed_everything(cfg.seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, meta = make_loaders(cfg, logger)
    num_classes = int(meta['num_classes'])
    if num_classes != 4:
        logger.warning(f"[warn] expected 4 classes, but found {num_classes} unique Variety values")

    model = LettuceMultiBranchCNN(num_classes=num_classes).to(device)

    stage1_hist = None
    stage2_hist = None
    stage3_hist = None
    best_full = None

    try:
        # Stage 1
        logger.info('[stage1] training RGB classifier...')
        rgb_ckpt, stage1_hist = stage1_train_rgb_classifier(cfg, model, train_loader, val_loader, device, logger)
        load_checkpoint(rgb_ckpt, model, device)

        # Debug: print confusion matrix for val after stage 1
        if cfg.debug:
            try:
                import numpy as np
                class_names = meta.get('class_names', [f'class{i}' for i in range(meta['num_classes'])])
                cm = _confusion_matrix(model, val_loader, device, num_classes=int(meta['num_classes']))
                logger.info('[debug] Stage 1 validation confusion matrix:')
                logger.info(f"\n{cm}")
                # Optional: normalized per true class
                with np.errstate(divide='ignore', invalid='ignore'):
                    row_sums = cm.sum(axis=1, keepdims=True)
                    cm_norm = np.divide(cm, row_sums, where=row_sums>0)
                logger.info('[debug] Stage 1 validation confusion matrix (row-normalized):')
                logger.info(f"\n{cm_norm}")
                # Overall accuracy from confusion matrix
                total = cm.sum()
                correct = np.trace(cm)
                acc_cm = (float(correct) / float(total)) if total > 0 else 0.0
                logger.info(f"[debug] Stage 1 validation accuracy (from CM): {acc_cm:.3f}")
            except Exception as e:
                logger.warning(f"[debug] failed to compute confusion matrix: {e}")

        # Stage 2
        logger.info('[stage2] training RGBD regressor...')
        rgbd_ckpt, stage2_hist = stage2_train_rgbd_regressor(cfg, model, train_loader, val_loader, device, logger)
        load_checkpoint(rgbd_ckpt, model, device)

        # Stage 3
        logger.info('[stage3] training fusion model...')
        best_full, stage3_hist = stage3_train_fusion(cfg, model, train_loader, val_loader, device, logger)

    except KeyboardInterrupt:
        logger.warning('[interrupt] Training interrupted by user (KeyboardInterrupt). Attempting to save plots and exit gracefully.')
    finally:
        # Plot curves for any completed stages
        histories = []
        if stage1_hist is not None:
            histories.append(("Stage 1: RGB classifier", "Cross-Entropy", stage1_hist))
        if stage2_hist is not None:
            histories.append(("Stage 2: RGBD regressor", "MAE", stage2_hist))
        if stage3_hist is not None:
            histories.append(("Stage 3: Fusion (final MAE)", "MAE", stage3_hist))
        if histories:
            _plot_histories(histories, cfg.out_dir, logger)

        if best_full is not None:
            logger.info(f"Done. Best full model saved to: {best_full}")
        else:
            logger.info("Run finished without completing Stage 3.")


if __name__ == '__main__':
    # Windows multi-worker safety (even though we default to 0 workers)
    import multiprocessing as mp

    mp.freeze_support()
    main()
