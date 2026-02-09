"""
Training module for Model_v6.
Handles training loop with logging, checkpointing, and versioning.
"""

from pathlib import Path
from typing import Tuple
import json
import logging

import numpy as np
import pandas as pd
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
    
    @staticmethod
    def _shutdown_loader(loader: DataLoader):
        """Attempt to shut down dataloader workers cleanly."""
        try:
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                shutdown = getattr(iterator, '_shutdown_workers', None)
                if callable(shutdown):
                    shutdown()
        except Exception:
            pass
    
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

    def _resolve_original_id_column(self, columns):
        normalized = {col.lower(): col for col in columns}
        for candidate in ("original_id", "source_id", "parent_id"):
            if candidate in normalized:
                return normalized[candidate]
            lowered = candidate.lower()
            if lowered in normalized:
                return normalized[lowered]
        return None

    def _prepare_split_csvs(self) -> Tuple[Path, Path]:
        augmented_csv = Path(self.config.AUGMENTED_CSV)
        if not augmented_csv.exists():
            raise FileNotFoundError(f"Augmented CSV not found at {augmented_csv}")

        split_dir = self.experiment_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_split = split_dir / "train_split.csv"
        val_split = split_dir / "val_split.csv"

        df = pd.read_csv(augmented_csv)
        if df.empty:
            raise ValueError("Augmented CSV is empty; run preprocessing first.")

        original_col = self._resolve_original_id_column(df.columns)
        if original_col is None:
            per = 1 + max(0, int(self.config.PREPROCESS_NUM_AUG))
            df['_original_id'] = ((df['id'].astype(int) - 1) // per).astype(int)
            original_col = '_original_id'
        else:
            df['_original_id'] = df[original_col]

        unique_originals = df['_original_id'].astype(str).unique()
        if len(unique_originals) < 2:
            raise ValueError("Need at least two unique originals to create a validation split.")

        rng = np.random.RandomState(self.config.VAL_SPLIT_SEED)
        rng.shuffle(unique_originals)

        val_count = max(1, int(len(unique_originals) * self.config.VAL_SPLIT_RATIO))
        val_count = min(len(unique_originals) - 1, val_count)
        val_ids = set(unique_originals[:val_count])

        train_df = df[~df['_original_id'].astype(str).isin(val_ids)].reset_index(drop=True)
        val_df = df[df['_original_id'].astype(str).isin(val_ids)].reset_index(drop=True)

        train_df.to_csv(train_split, index=False)
        val_df.to_csv(val_split, index=False)

        self.logger.info(
            "Created train/val splits with %d/%d originals (train rows=%d, val rows=%d)",
            len(unique_originals) - val_count,
            val_count,
            len(train_df),
            len(val_df)
        )

        return train_split, val_split

    def _build_default_loader(self,
                              csv_file: Path,
                              shuffle: bool,
                              include_target: bool = True) -> DataLoader:
        return create_dataloader(
            csv_file=str(csv_file),
            rgb_dir=self.config.AUGMENTED_RGB_DIR,
            depth_dir=self.config.AUGMENTED_DEPTH_DIR,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.config.NUM_WORKERS,
            persistent_workers=self.config.PERSISTENT_WORKERS,
            include_target=include_target
        )
    
    def train(self, train_loader: DataLoader = None, val_loader: DataLoader = None):
        """Main training loop with automatic train/validation split generation."""
        self.logger.info(f"Starting training for {self.config.EPOCHS} epochs...")
        self.logger.info(f"Config: {self.config.to_dict()}")
        
        split_train_csv = split_val_csv = None
        if train_loader is None or val_loader is None:
            split_train_csv, split_val_csv = self._prepare_split_csvs()

        if train_loader is None and split_train_csv is not None:
            train_loader = self._build_default_loader(split_train_csv, shuffle=True, include_target=True)
        if val_loader is None and split_val_csv is not None:
            val_loader = self._build_default_loader(split_val_csv, shuffle=False, include_target=True)
        
        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)
        
        try:
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
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Cleaning up dataloaders...")
            self._shutdown_loader(train_loader)
            if val_loader is not None:
                self._shutdown_loader(val_loader)
            raise
        
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
    config = Config()
    config.print_config()

    try:
        trainer = train_model(config)
    except Exception as e:
        print(f"Error during training: {e}")
