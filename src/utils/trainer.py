"""
Generic trainer for model training with experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm
from .experiment_tracker import ExperimentTracker


class Trainer:
    """
    Generic trainer class for training models with proper experiment tracking.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 experiment_tracker: ExperimentTracker,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            experiment_tracker: Experiment tracker instance
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = experiment_tracker
        self.device = device
        
        # Setup training components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.run_id = None
        
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion."""
        # MSE for regression task
        return nn.MSELoss()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, 
                           momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name is None or scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, run_name: str, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            run_name: Name for this training run
            epochs: Number of epochs (overrides config if provided)
            
        Returns:
            Dictionary with final training results
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        # Start tracking this run
        full_config = {
            'model': self.model.get_version() if hasattr(self.model, 'get_version') else 'unknown',
            'training': self.config
        }
        self.run_id = self.tracker.start_run(run_name, full_config)
        
        print(f"\nStarting training run: {self.run_id}")
        print(f"Training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # Train and validate
                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Log metrics
                self.tracker.log_metrics(self.run_id, metrics, epoch)
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = self.tracker.get_checkpoint_path(self.run_id)
                    if hasattr(self.model, 'save_checkpoint'):
                        self.model.save_checkpoint(
                            str(checkpoint_path),
                            epoch,
                            self.optimizer.state_dict(),
                            metrics
                        )
                    else:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'metrics': metrics
                        }, checkpoint_path)
                    
                    print(f"\n✓ New best model saved (val_loss: {val_loss:.6f})")
                else:
                    self.patience_counter += 1
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"  Best Val Loss: {self.best_val_loss:.6f}")
                print(f"  Patience: {self.patience_counter}/{early_stopping_patience}")
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Training completed
            results = {
                'final_train_loss': train_metrics['train_loss'],
                'final_val_loss': val_metrics['val_loss'],
                'best_val_loss': self.best_val_loss,
                'epochs_trained': epoch + 1
            }
            
            self.tracker.end_run(self.run_id, results, 'completed')
            print(f"\n✓ Training completed successfully")
            return results
            
        except Exception as e:
            # Training failed
            self.tracker.end_run(self.run_id, {'error': str(e)}, 'failed')
            print(f"\n✗ Training failed: {str(e)}")
            raise
