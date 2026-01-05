"""
Loss functions for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""
    
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (B,)
            targets: Ground truth values (B,)
        
        Returns:
            Huber loss
        """
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Huber loss: quadratic for small errors, linear for large errors
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        loss = torch.where(abs_error <= self.delta, quadratic, linear)
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) in [0, 1]
        
        Returns:
            Dice loss
        """
        # Apply sigmoid to logits
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedSegmentationLoss(nn.Module):
    """Combined BCE and Dice loss for segmentation."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) in [0, 1]
        
        Returns:
            Combined loss
        """
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    - Segmentation loss (BCE + Dice)
    - Deep regression loss (Huber)
    - Phenotype regression loss (Huber)
    - Final blended regression loss (Huber)
    """
    
    def __init__(self, lambda_seg=0.5, lambda_deep=1.0, lambda_phen=1.0, 
                 lambda_final=2.0, huber_delta=1.0, use_segmentation=True):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_deep = lambda_deep
        self.lambda_phen = lambda_phen
        self.lambda_final = lambda_final
        self.use_segmentation = use_segmentation
        
        self.seg_loss = CombinedSegmentationLoss()
        self.huber_loss = HuberLoss(delta=huber_delta)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing:
                - 'mask_logits': (B, 1, H, W) if segmentation enabled
                - 'deep_pred': (B,) deep regression prediction
                - 'phen_pred': (B,) phenotype regression prediction
                - 'final_pred': (B,) final blended prediction
            targets: Dictionary containing:
                - 'masks': (B, 1, H, W) if segmentation enabled
                - 'dry_weight': (B,) regression target
        
        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Segmentation loss
        if self.use_segmentation and 'mask_logits' in outputs and 'masks' in targets:
            seg_loss = self.seg_loss(outputs['mask_logits'], targets['masks'])
            loss_dict['seg_loss'] = seg_loss.item()
            total_loss += self.lambda_seg * seg_loss
        
        # Regression losses
        dry_weight = targets['dry_weight']
        
        # Deep regression loss
        deep_loss = self.huber_loss(outputs['deep_pred'], dry_weight)
        loss_dict['deep_loss'] = deep_loss.item()
        total_loss += self.lambda_deep * deep_loss
        
        # Phenotype regression loss
        if 'phen_pred' in outputs:
            phen_loss = self.huber_loss(outputs['phen_pred'], dry_weight)
            loss_dict['phen_loss'] = phen_loss.item()
            total_loss += self.lambda_phen * phen_loss
        
        # Final blended loss
        final_loss = self.huber_loss(outputs['final_pred'], dry_weight)
        loss_dict['final_loss'] = final_loss.item()
        total_loss += self.lambda_final * final_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
