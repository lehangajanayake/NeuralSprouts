"""
K-Fold Cross-Validation Training Script

USAGE:
    python train.py

Before running:
1. Ensure data is in the correct structure:
   data/
     train/
       rgb/
       depth/
       masks/ (optional)
       labels.csv
2. Install dependencies:
   pip install torch torchvision timm numpy pandas opencv-python albumentations scikit-learn pyyaml tqdm
3. Adjust config.py if needed (paths, hyperparameters, etc.)

This script will:
- Perform 5-fold cross-validation
- Train a model for each fold
- Save best checkpoint per fold
- Log metrics to console and files
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path

# Change to the script's directory to ensure relative paths work
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

from config import Config
from dataset import create_dataloaders
from model import build_model
from losses import MultiTaskLoss
from utils import (
    set_seed, calculate_metrics, save_checkpoint, 
    AverageMeter, EarlyStopping, get_lr
)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    deep_loss_meter = AverageMeter()
    phen_loss_meter = AverageMeter()
    final_loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        dry_weight = batch['dry_weight'].to(device)
        
        targets = {'dry_weight': dry_weight}
        if 'mask' in batch:
            targets['masks'] = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_AMP:
            with autocast():
                outputs = model(rgb, depth)
                loss, loss_dict = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb, depth)
            loss, loss_dict = criterion(outputs, targets)
            
            loss.backward()
            
            if config.GRADIENT_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            optimizer.step()
        
        # Update meters
        batch_size = rgb.size(0)
        loss_meter.update(loss_dict['total_loss'], batch_size)
        deep_loss_meter.update(loss_dict['deep_loss'], batch_size)
        final_loss_meter.update(loss_dict['final_loss'], batch_size)
        
        if 'seg_loss' in loss_dict:
            seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        if 'phen_loss' in loss_dict:
            phen_loss_meter.update(loss_dict['phen_loss'], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'deep': f'{deep_loss_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })
    
    return {
        'loss': loss_meter.avg,
        'seg_loss': seg_loss_meter.avg,
        'deep_loss': deep_loss_meter.avg,
        'phen_loss': phen_loss_meter.avg,
        'final_loss': final_loss_meter.avg
    }


def validate(model, val_loader, criterion, device, config, epoch):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            dry_weight = batch['dry_weight'].to(device)
            
            targets = {'dry_weight': dry_weight}
            if 'mask' in batch:
                targets['masks'] = batch['mask'].to(device)
            
            outputs = model(rgb, depth)
            loss, loss_dict = criterion(outputs, targets)
            
            # Collect predictions
            predictions = outputs['final_pred'].cpu().numpy()
            targets_np = dry_weight.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets_np)
            
            loss_meter.update(loss_dict['total_loss'], rgb.size(0))
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    metrics['loss'] = loss_meter.avg
    
    return metrics


def train_fold(fold, train_df, val_df, config, device):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}/{config.NUM_FOLDS}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_df, val_df, config)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Build model
    model = build_model(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = MultiTaskLoss(
        lambda_seg=config.LAMBDA_SEG,
        lambda_deep=config.LAMBDA_DEEP,
        lambda_phen=config.LAMBDA_PHEN,
        lambda_final=config.LAMBDA_FINAL,
        huber_delta=config.HUBER_DELTA,
        use_segmentation=config.USE_PHENOTYPE_FEATURES
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    if config.SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.MIN_LR
        )
    elif config.SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, mode='min')
    
    # Training loop
    best_mae = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, epoch)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Save best model
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            checkpoint_path = config.CHECKPOINT_DIR / f'fold_{fold}_best.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, fold, val_metrics, checkpoint_path
            )
            print(f"✓ Best model saved (MAE: {best_mae:.4f})")
        
        # Early stopping check
        if early_stopping(val_metrics['mae'], epoch):
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best epoch was {early_stopping.best_epoch} with MAE: {best_mae:.4f}")
                break
    
    return best_mae


def main():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Create output directories
    Config.create_dirs()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load data
    print("\nLoading dataset...")
    if not Config.LABELS_PATH.exists():
        print(f"Error: Labels file not found at {Config.LABELS_PATH}")
        print("Please ensure your data is structured correctly.")
        return
    
    df = pd.read_csv(Config.LABELS_PATH)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    if 'id' not in df.columns or 'dry_weight' not in df.columns:
        print("Error: Labels CSV must contain 'id' and 'dry_weight' columns")
        return
    
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    fold_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        best_mae = train_fold(fold, train_df, val_df, Config, device)
        fold_maes.append(best_mae)
    
    # Summary
    print("\n" + "="*60)
    print("K-Fold Cross-Validation Summary")
    print("="*60)
    for fold, mae in enumerate(fold_maes):
        print(f"Fold {fold + 1}: MAE = {mae:.4f}")
    print(f"\nMean MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print("="*60)
    
    print("\nTraining complete! Checkpoints saved in:", Config.CHECKPOINT_DIR)
    print("\nTo generate predictions, run: python predict.py")


if __name__ == '__main__':
    main()
