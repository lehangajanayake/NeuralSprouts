"""
Training module for Model_v6.
Handles training loop with logging, checkpointing, and versioning.
"""

import os
from pathlib import Path
from datetime import datetime
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from model import create_model
from dataloader import create_dataloader


class Trainer:
    """Trainer class for Model_v6."""
    
    def __init__(self, config: Config = None):
        """Initialize trainer."""
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Create experiment directory
        self.experiment_dir = Path(self.config.EXPERIMENT_DIR)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Model, optimizer, loss
        self.model = create_model(self.config).to(self.device)
        self.optimizer = self.setup_optimizer()
        self.loss_fn = self.setup_loss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging."""
        log_file = self.experiment_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizer(self):
        """Setup optimizer."""
        if self.config.OPTIMIZER.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
    
    def setup_loss(self):
        """Setup loss function."""
        if self.config.LOSS_FUNCTION.lower() == "mse":
            return nn.MSELoss()
        elif self.config.LOSS_FUNCTION.lower() == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.LOSS_FUNCTION}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (rgb_images, rgbd_images, dry_weights, image_ids) in enumerate(train_loader):
            rgb_images = rgb_images.to(self.device)
            rgbd_images = rgbd_images.to(self.device)
            dry_weights = dry_weights.to(self.device).view(-1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(rgb_images, rgbd_images)
            loss = self.loss_fn(predictions, dry_weights)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        ground_truth_list = []
        image_ids_list = []
        
        with torch.no_grad():
            for rgb_images, rgbd_images, dry_weights, image_ids in val_loader:
                rgb_images = rgb_images.to(self.device)
                rgbd_images = rgbd_images.to(self.device)
                dry_weights = dry_weights.to(self.device).view(-1, 1)
                
                predictions = self.model(rgb_images, rgbd_images)
                loss = self.loss_fn(predictions, dry_weights)
                
                total_loss += loss.item()
                
                predictions_list.append(predictions.cpu().numpy())
                ground_truth_list.append(dry_weights.cpu().numpy())
                image_ids_list.extend(image_ids)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list), image_ids_list
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.experiment_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.EPOCHS} epochs...")
        self.logger.info(f"Config: {self.config.to_dict()}")
        
        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)
        
        for epoch in range(self.config.EPOCHS):
            self.logger.info(f"\n=== Epoch {epoch + 1}/{self.config.EPOCHS} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss, predictions, ground_truth, image_ids = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"Best model updated! (Loss: {val_loss:.4f})")
            else:
                self.save_checkpoint(epoch)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        self.logger.info(f"\nTraining complete! Best model at epoch {self.best_epoch + 1}")


def train_model(config: Config = None, 
                train_loader: DataLoader = None,
                val_loader: DataLoader = None):
    """Convenience function to train the model."""
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    return trainer


if __name__ == "__main__":
    # Example usage
    config = Config()
    config.print_config()
    
    # Create dataloaders
    try:
        train_loader = create_dataloader(
            csv_file=config.TRAIN_CSV,
            rgb_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBImages",
            rgbd_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBDImages",
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            preprocessed=True
        )
        
        val_loader = create_dataloader(
            csv_file=config.TEST_CSV,
            rgb_dir=config.TEST_RGB_DIR,
            rgbd_dir=config.TEST_DEPTH_DIR,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            preprocessed=False
        )
        
        # Train
        trainer = train_model(config, train_loader, val_loader)
    except Exception as e:
        print(f"Error during training: {e}")
