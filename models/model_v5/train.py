import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataloader import PlantDatasetV5, group_aware_train_val_split
from model import PlantV5TripleBranch


@dataclass
class TrainConfig:
    """Training configuration for Model V5."""
    # Dataset
    train_csv: str = '../../datasets/Training/Augmented/Train_aug.csv'
    rgb_dir: str = '../../datasets/Training/Augmented/RGBImages'
    depth_dir: str = '../../datasets/Training/Augmented/DepthImages'
    
    # Training
    batch_size: int = 8  # 6GB VRAM, 128x128 images
    num_epochs: int = 200
    lr: float = 1e-3  # Adam learning rate
    weight_decay: float = 1e-3
    
    # Learning rate scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'exponential'
    scheduler_patience: int = 10  # for ReduceLROnPlateau
    scheduler_step_size: int = 30  # for StepLR
    scheduler_gamma: float = 0.1  # step/exponential decay
    
    # Model
    branch_dim: int = 64
    fc_hidden: int = 128
    dropout: float = 0.2
    
    # Data
    val_ratio: float = 0.2
    seed: int = 42
    device: str = 'cuda'
    
    # Output
    out_dir: str = './5.6/'
    log_name: str = 'train_v5.6'
    save_interval: int = 50  # Save checkpoint every N epochs
    
    # Debug
    debug: bool = True


def setup_logging(cfg: TrainConfig) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = Path(cfg.out_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{cfg.log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def create_scheduler(optimizer, cfg: TrainConfig):
    """Create learning rate scheduler."""
    if cfg.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    elif cfg.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.scheduler_gamma,
            patience=cfg.scheduler_patience, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    logger: logging.Logger,
    cfg: TrainConfig,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Returns:
        (train_loss, train_mae)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0
    
    try:
        for batch_idx, (rgb, rgbd, depth, labels) in enumerate(dataloader):
            # Move to device (already on device from dataloader, but ensure)
            rgb = rgb.to(device)
            rgbd = rgbd.to(device)
            depth = depth.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(rgb, rgbd, depth)
            
            # RMSE loss for training
            loss = criterion(predictions, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                mae = torch.abs(predictions - labels).mean().item()
            
            total_loss += loss.item() * labels.size(0)
            total_mae += mae * labels.size(0)
            num_samples += labels.size(0)
            
            if cfg.debug and batch_idx % 10 == 0:
                logger.debug(f"  Batch {batch_idx}: Loss={loss.item():.4f}, MAE={mae:.4f}")
    
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt in train epoch")
        raise
    
    return total_loss / num_samples, total_mae / num_samples


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    logger: logging.Logger,
    cfg: TrainConfig,
) -> Tuple[float, float]:
    """Validate for one epoch.
    
    Returns:
        (val_loss, val_mae)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        try:
            for rgb, rgbd, depth, labels in dataloader:
                rgb = rgb.to(device)
                rgbd = rgbd.to(device)
                depth = depth.to(device)
                labels = labels.to(device)
                
                predictions = model(rgb, rgbd, depth)
                loss = criterion(predictions, labels)
                mae = torch.abs(predictions - labels).mean()
                
                total_loss += loss.item() * labels.size(0)
                total_mae += mae.item() * labels.size(0)
                num_samples += labels.size(0)
        
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt in validation")
            raise
    
    return total_loss / num_samples, total_mae / num_samples


def plot_metrics(train_losses, val_losses, train_maes, val_maes, out_dir: str, logger: logging.Logger):
    """Plot and save training metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train RMSE Loss', marker='o', markersize=3)
    ax1.plot(epochs, val_losses, 'r-', label='Val RMSE Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(epochs, train_maes, 'b-', label='Train MAE', marker='o', markersize=3)
    ax2.plot(epochs, val_maes, 'r-', label='Val MAE (Competition Metric)', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=100)
    logger.info(f"Saved metrics plot to {save_path}")
    plt.close()


def main(cfg: Optional[TrainConfig] = None):
    """Main training loop."""
    if cfg is None:
        cfg = TrainConfig()
    
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(cfg)
    logger.info("=" * 80)
    logger.info("Model V5: Triple-Branch Fusion (RGB, RGBD, Depth)")
    logger.info("=" * 80)
    logger.info(f"Config: {cfg}")
    
    # Device
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    # Dataset
    logger.info("\n" + "=" * 80)
    logger.info("Loading dataset...")
    dataset = PlantDatasetV5(
        rgb_dir=cfg.rgb_dir,
        depth_dir=cfg.depth_dir,
        labels_csv=cfg.train_csv,
        image_size=128,
        device=device,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Train/val split
    # Create a temporary df with only valid indices for splitting
    valid_df = dataset.df.iloc[dataset.valid_indices].reset_index(drop=True)
    train_indices_in_valid, val_indices_in_valid = group_aware_train_val_split(
        valid_df, val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    train_dataset = Subset(dataset, train_indices_in_valid)
    val_dataset = Subset(dataset, val_indices_in_valid)
    
    logger.info(f"Train split: {len(train_dataset)} samples")
    logger.info(f"Val split: {len(val_dataset)} samples")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Data already in VRAM
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Model
    logger.info("\n" + "=" * 80)
    logger.info("Initializing model...")
    model = PlantV5TripleBranch(
        branch_dim=cfg.branch_dim,
        fc_hidden=cfg.fc_hidden,
        dropout=cfg.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # RMSE is sqrt(MSE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = create_scheduler(optimizer, cfg)
    
    logger.info(f"Optimizer: Adam (lr={cfg.lr}, weight_decay={cfg.weight_decay})")
    logger.info(f"Scheduler: {cfg.scheduler_type}")
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    best_val_mae = float('inf')
    best_epoch = -1
    patience_counter = 0
    
    try:
        for epoch in range(cfg.num_epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_mae = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, cfg
            )
            train_losses.append(train_loss)
            train_maes.append(train_mae)
            
            # Validate
            val_loss, val_mae = validate_epoch(
                model, val_loader, criterion, device, logger, cfg
            )
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            
            # Log
            logger.info(
                f"Epoch {epoch+1:3d}/{cfg.num_epochs} | "
                f"LR={current_lr:.2e} | "
                f"Train RMSE={train_loss**0.5:.4f} | Train MAE={train_mae:.4f} | "
                f"Val RMSE={val_loss**0.5:.4f} | Val MAE={val_mae:.4f}"
            )
            
            # Scheduler step
            if cfg.scheduler_type == 'plateau':
                scheduler.step(val_mae)
            else:
                scheduler.step()
            
            # Save best model (based on validation MAE)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                patience_counter = 0
                
                best_path = out_dir / 'best_model_v5.pth'
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved best model (Val MAE={val_mae:.4f})")
            else:
                patience_counter += 1
            
            # Periodic checkpoint
            if (epoch + 1) % cfg.save_interval == 0:
                ckpt_path = out_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
            
            # Early stopping (optional - not enforced, just patience info)
            if patience_counter > 0 and patience_counter % 20 == 0:
                logger.info(f"No improvement for {patience_counter} epochs")
    
    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("Training interrupted by user (Ctrl+C)")
        logger.warning("=" * 80)
        
        # Save current state
        ckpt_path = out_dir / 'checkpoint_interrupted.pth'
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved interrupted checkpoint to {ckpt_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch + 1}/{cfg.num_epochs}")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Final training MAE: {train_maes[-1]:.4f}")
    logger.info(f"Final validation MAE: {val_maes[-1]:.4f}")
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_maes, val_maes, str(out_dir), logger)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'best_epoch': best_epoch + 1,
        'best_val_mae': best_val_mae,
    }
    history_path = out_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info("=" * 80)
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
