"""
Utility functions for training, evaluation, and checkpointing.
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions, targets):
    """
    Calculate regression metrics.
    
    Args:
        predictions: numpy array of predictions
        targets: numpy array of ground truth values
    
    Returns:
        dict: Dictionary containing MAE, RMSE, and R2
    """
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def save_checkpoint(model, optimizer, scheduler, epoch, fold, metrics, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        fold: Current fold number
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load model on
    
    Returns:
        dict: Checkpoint information (epoch, metrics, etc.)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience=7, mode='min', delta=0.0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            mode: 'min' for metrics like loss, 'max' for metrics like accuracy
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, metric, epoch):
        """
        Check if should stop training.
        
        Args:
            metric: Current metric value
            epoch: Current epoch
        
        Returns:
            bool: True if metric improved
        """
        score = -metric if self.mode == 'min' else metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        if score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_depth(depth, strategy='per_image', global_mean=0.5, global_std=0.25):
    """
    Normalize depth images.
    
    Args:
        depth: Depth image tensor (H, W) or (C, H, W)
        strategy: Normalization strategy ('per_image', 'global', 'percentile')
        global_mean: Mean for global normalization
        global_std: Std for global normalization
    
    Returns:
        Normalized depth tensor
    """
    if strategy == 'per_image':
        # Handle NaN/invalid values
        valid_mask = torch.isfinite(depth) & (depth > 0)
        if valid_mask.sum() == 0:
            return torch.zeros_like(depth)
        
        valid_depth = depth[valid_mask]
        depth_min = valid_depth.min()
        depth_max = valid_depth.max()
        
        if depth_max - depth_min < 1e-6:
            normalized = torch.zeros_like(depth)
        else:
            normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Set invalid pixels to 0
        normalized[~valid_mask] = 0
        return normalized
    
    elif strategy == 'global':
        normalized = (depth - global_mean) / (global_std + 1e-8)
        normalized[~torch.isfinite(normalized)] = 0
        return normalized
    
    elif strategy == 'percentile':
        valid_mask = torch.isfinite(depth) & (depth > 0)
        if valid_mask.sum() == 0:
            return torch.zeros_like(depth)
        
        valid_depth = depth[valid_mask]
        p2 = torch.quantile(valid_depth, 0.02)
        p98 = torch.quantile(valid_depth, 0.98)
        
        if p98 - p2 < 1e-6:
            normalized = torch.zeros_like(depth)
        else:
            normalized = (depth - p2) / (p98 - p2 + 1e-8)
            normalized = torch.clamp(normalized, 0, 1)
        
        normalized[~valid_mask] = 0
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")
