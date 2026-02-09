"""
Configuration module for Model_v6.
All preprocessing, augmentation, training, and model parameters are configurable here.
"""

class Config:
    """Configuration class for Model_v6."""
    
    # ======================== Preprocessing ========================
    # Center crop size (pixels)
    CENTER_CROP_SIZE = 1000
    
    # Resize size (pixels)
    RESIZE_SIZE = 96
    
    # ======================== Augmentations ========================
    # Enable/disable augmentations
    AUGMENTATIONS_ENABLED = True
    
    # Horizontal flip
    HORIZONTAL_FLIP_ENABLED = True
    HORIZONTAL_FLIP_PROB = 0.5
    
    # Vertical flip
    VERTICAL_FLIP_ENABLED = True
    VERTICAL_FLIP_PROB = 0.5
    
    # Rotation
    ROTATION_ENABLED = True
    ROTATION_ANGLE_RANGE = (-15, 15)  # degrees
    ROTATION_PROB = 0.5
    
    # Horizontal shift (pixels)
    HORIZONTAL_SHIFT_ENABLED = False
    HORIZONTAL_SHIFT_MAX = 0.1  # percentage of image width
    HORIZONTAL_SHIFT_PROB = 0.5
    
    # Vertical shift (pixels)
    VERTICAL_SHIFT_ENABLED = False
    VERTICAL_SHIFT_MAX = 0.1  # percentage of image height
    VERTICAL_SHIFT_PROB = 0.5
    
    # ======================== Logging ========================
    # Directory to store logs
    LOG_DIR = "./logs"
    
    # Augmentation log file name
    AUGMENTATION_LOG_FILE = "augmentations.csv"
    
    # Keep original image ID
    KEEP_IMAGE_ID = True
    
    # ======================== Training ========================
    # Batch size
    BATCH_SIZE = 16
    
    # Learning rate
    LEARNING_RATE = 0.001
    
    # Number of epochs
    EPOCHS = 100
    
    # Optimizer (adam, sgd)
    OPTIMIZER = "adam"
    
    # Loss function (mse, mae)
    LOSS_FUNCTION = "mse"
    
    # Device (cuda, cpu)
    DEVICE = "cuda"
    
    # Number of workers for dataloader
    NUM_WORKERS = 16
    PERSISTENT_WORKERS = True

    # Dataset caching (store tensors in memory after first load)
    ENABLE_DATASET_CACHE = True

    # Validation split parameters
    VAL_SPLIT_RATIO = 0.2
    VAL_SPLIT_SEED = 42
    
    # ======================== Model Architecture ========================
    # Input channels
    RGB_CHANNELS = 3
    RGBD_CHANNELS = 4
    
    # Number of output classes (1 for dry weight prediction)
    OUTPUT_DIM = 1
    
    # CNN architecture parameters
    NUM_CONV_LAYERS = 4
    INITIAL_FILTERS =24
    FILTER_MULTIPLIER = 2
    BRANCH_EMBED_DIM = 256
    
    # Fusion layer hidden dim
    FUSION_HIDDEN_DIM = 256
    
    # Dropout rate
    DROPOUT_RATE = 0.5
    
    # ======================== Versioning ========================
    # Version folder prefix (e.g., "6.1", "6.2")
    VERSION = "6.1"
    
    # Base directory for versioned experiments
    EXPERIMENT_DIR = f"./experiments/{VERSION}"
    
    # ======================== Data Paths ========================
    # Dataset paths
    TRAIN_RGB_DIR = "../../datasets/Training/RGBImages" 
    TRAIN_DEPTH_DIR = "../../datasets/Training/DepthImages"
    TRAIN_CSV = "../../datasets/Training/Train.csv"
    
    TEST_RGB_DIR = "../../datasets/Test/RGBImages"
    TEST_DEPTH_DIR = "../../datasets/Test/DepthImages"
    TEST_CSV = "../../datasets/Test/Test.csv"
    
    # Augmented data output directories
    AUGMENTED_OUTPUT_DIR = f"../../datasets/Training/Augmented/{VERSION}"
    AUGMENTED_RGB_DIR = f"{AUGMENTED_OUTPUT_DIR}/RGBImages"
    AUGMENTED_DEPTH_DIR = f"{AUGMENTED_OUTPUT_DIR}/DepthImages"
    AUGMENTED_CSV = f"{AUGMENTED_OUTPUT_DIR}/Train_aug.csv"

    # Preprocessing pipeline parameters
    PREPROCESS_NUM_AUG = 5  # number of augmented variants per original image
    PREPROCESS_SEED = 42
    PREPROCESS_NUM_WORKERS = None  # auto-detect if None
    PREPROCESS_MAX_ITEMS = None  # limit for debugging
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary, excluding non-serializable objects."""
        import json
        result = {}
        for k, v in vars(cls).items():
            if k.startswith('_') or callable(v):
                continue
            try:
                json.dumps(v)
                result[k] = v
            except (TypeError, OverflowError):
                continue
        return result
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("=" * 60)
        print("Model_v6 Configuration")
        print("=" * 60)
        for key, value in cls.to_dict().items():
            print(f"{key}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()
