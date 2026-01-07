"""
Configuration file for multimodal fusion model training and inference.
"""

import os
from pathlib import Path

class Config:
    """Configuration class with all hyperparameters and paths."""
    
    # Data paths
    DATA_DIR = Path("data")
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    RGB_TRAIN_DIR = TRAIN_DIR / "rgb"
    DEPTH_TRAIN_DIR = TRAIN_DIR / "depth"
    MASK_TRAIN_DIR = TRAIN_DIR / "masks"
    LABELS_PATH = TRAIN_DIR / "labels.csv"
    RGB_TEST_DIR = TEST_DIR / "rgb"
    DEPTH_TEST_DIR = TEST_DIR / "depth"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    
    # Output paths
    OUTPUT_DIR = Path("output")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    
    # Model architecture
    RGB_BACKBONE = "convnext_tiny"  # Options: convnext_tiny, efficientnetv2_s, etc.
    DEPTH_BACKBONE = "convnext_tiny"
    PRETRAINED = True
    IMAGE_SIZE = 384  # Input image size (height, width)
    
    # Fusion configuration
    FUSION_CHANNELS = 256  # Channels after fusion
    
    # Segmentation decoder
    DECODER_CHANNELS = [256, 128, 64, 32]  # UNet-style decoder
    
    # Regression head
    REGRESSION_HIDDEN = [512, 256, 128]  # MLP hidden dimensions
    
    # Phenotype feature extractor
    USE_PHENOTYPE_FEATURES = False  # Set to False if masks not available
    PHENOTYPE_THRESHOLD = 0.5  # Threshold for binary mask
    PHENOTYPE_HIDDEN = [64, 32]  # Phenotype MLP hidden dims
    LEARNABLE_ALPHA = True  # Learnable blending weight vs fixed
    FIXED_ALPHA = 0.7  # Used if LEARNABLE_ALPHA=False
    
    # Training hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    
    # K-fold cross-validation
    NUM_FOLDS = 5
    RANDOM_STATE = 42
    
    # Loss weights
    LAMBDA_SEG = 0.5  # Weight for segmentation loss
    LAMBDA_DEEP = 1.0  # Weight for deep regression loss
    LAMBDA_PHEN = 1.0  # Weight for phenotype regression loss
    LAMBDA_FINAL = 2.0  # Weight for final blended loss
    HUBER_DELTA = 1.0  # Delta parameter for Huber loss
    
    # Training strategies
    USE_AMP = True  # Automatic mixed precision
    GRADIENT_CLIP = 1.0  # Gradient clipping value
    
    # Learning rate scheduler
    SCHEDULER = "cosine"  # Options: cosine, step, plateau
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    
    # Early stopping
    PATIENCE = 15  # Epochs to wait for improvement
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUG_FLIP_PROB = 0.5
    AUG_ROTATE_LIMIT = 15  # degrees
    AUG_BRIGHTNESS_LIMIT = 0.2
    AUG_CONTRAST_LIMIT = 0.2
    
    # Normalization (ImageNet stats for RGB)
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]
    
    # Depth normalization strategy
    DEPTH_NORM_STRATEGY = "per_image"  # Options: per_image, global, percentile
    DEPTH_GLOBAL_MEAN = 0.5  # Used if global normalization
    DEPTH_GLOBAL_STD = 0.25
    
    # Misc
    SEED = 42
    DEVICE = "cuda"  # Will auto-detect in code
    LOG_INTERVAL = 10  # Print every N batches
    SAVE_BEST_ONLY = True
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
