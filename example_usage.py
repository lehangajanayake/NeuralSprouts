"""
Example usage of the lettuce dry weight prediction framework.

This script demonstrates how to:
1. Load configuration
2. Create a model (SimpleCNN v1)
3. Set up data augmentation pipeline
4. Initialize experiment tracking
5. Train the model

This is a template - you'll need to implement the actual dataset loading
based on your data format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.models import SimpleCNN
from src.data_augmentation import GenericAugmentationPipeline
from src.config import Config
from src.utils import ExperimentTracker
from src.utils.trainer import Trainer
import numpy as np
from PIL import Image


# ============================================================================
# STEP 1: Define your custom dataset (you'll need to implement this)
# ============================================================================

class LettuceDataset(Dataset):
    """
    Custom dataset for lettuce images and dry weight labels.
    
    You'll need to implement this based on your data format.
    This is just a template showing the expected interface.
    """
    
    def __init__(self, data_path, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory
            transform: Augmentation pipeline to apply
        """
        self.data_path = data_path
        self.transform = transform
        
        # TODO: Load your image paths and labels here
        # Example:
        # self.image_paths = [list of image file paths]
        # self.labels = [list of dry weight values]
        
        # For demonstration purposes, we'll just create dummy data
        self.image_paths = []
        self.labels = []
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # TODO: Implement actual image loading
        # Example:
        # image = Image.open(self.image_paths[idx]).convert('RGB')
        # image = np.array(image)
        
        # Dummy image for demonstration
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        label = 0.0  # Dummy label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# STEP 2: Main training function
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("Lettuce Dry Weight Prediction - Training Pipeline")
    print("=" * 80)
    
    # ========================================================================
    # Load configuration
    # ========================================================================
    print("\n[1] Loading configuration...")
    config = Config.from_yaml('configs/cnn_v1_config.yaml')
    
    # You can also create config programmatically:
    # config = Config()
    # config.set('model.dropout_rate', 0.3)
    
    print(f"Experiment: {config.get('experiment.name')}")
    print(f"Model: {config.get('model.name')} {config.get('model.version')}")
    
    # ========================================================================
    # Setup data augmentation pipelines
    # ========================================================================
    print("\n[2] Setting up data augmentation pipelines...")
    
    # Generic pipeline for training
    train_augmentation = GenericAugmentationPipeline(
        name="generic_train",
        image_size=config.get('augmentation.image_size', 224),
        is_training=True,
        augmentation_probability=config.get('augmentation.augmentation_probability', 0.5)
    )
    
    # Generic pipeline for validation (no augmentation)
    val_augmentation = GenericAugmentationPipeline(
        name="generic_val",
        image_size=config.get('augmentation.image_size', 224),
        is_training=False
    )
    
    print(f"Training pipeline: {train_augmentation.get_name()}")
    print(f"Validation pipeline: {val_augmentation.get_name()}")
    
    # ========================================================================
    # Create datasets and dataloaders
    # ========================================================================
    print("\n[3] Creating datasets and dataloaders...")
    
    # TODO: Replace with actual data paths
    train_dataset = LettuceDataset(
        data_path=config.get('data.train_path'),
        transform=train_augmentation
    )
    
    val_dataset = LettuceDataset(
        data_path=config.get('data.val_path'),
        transform=val_augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('data.batch_size', 32),
        shuffle=True,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('data.batch_size', 32),
        shuffle=False,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.get('data.batch_size')}")
    
    # ========================================================================
    # Create model
    # ========================================================================
    print("\n[4] Creating model...")
    
    model_config = config.get_section('model')
    model = SimpleCNN(model_config)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Version: {model.get_version()}")
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # ========================================================================
    # Setup experiment tracking
    # ========================================================================
    print("\n[5] Setting up experiment tracking...")
    
    experiment_tracker = ExperimentTracker(
        experiment_name=config.get('experiment.name'),
        base_dir="experiments"
    )
    
    print(f"Experiment directory: {experiment_tracker.experiment_dir}")
    
    # ========================================================================
    # Create trainer and start training
    # ========================================================================
    print("\n[6] Initializing trainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.get_section('training'),
        experiment_tracker=experiment_tracker,
        device=device
    )
    
    # ========================================================================
    # Train the model
    # ========================================================================
    print("\n[7] Starting training...")
    
    run_name = f"cnn_{config.get('model.version')}_run1"
    results = trainer.train(run_name=run_name)
    
    # ========================================================================
    # Print results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Results")
    print("=" * 80)
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Final train loss: {results['final_train_loss']:.6f}")
    print(f"Final validation loss: {results['final_val_loss']:.6f}")
    
    # ========================================================================
    # Show best runs
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Run Summary")
    print("=" * 80)
    
    best_run = experiment_tracker.get_best_run(metric='val_loss', minimize=True)
    if best_run:
        print(f"Run ID: {best_run['run_id']}")
        print(f"Started: {best_run['started_at']}")
        print(f"Validation Loss: {best_run['results']['best_val_loss']:.6f}")
    
    print("\n✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
