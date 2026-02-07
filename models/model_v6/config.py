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
    HORIZONTAL_SHIFT_ENABLED = True
    HORIZONTAL_SHIFT_MAX = 0.1  # percentage of image width
    HORIZONTAL_SHIFT_PROB = 0.5
    
    # Vertical shift (pixels)
    VERTICAL_SHIFT_ENABLED = True
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
    BATCH_SIZE = 32
    
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
    NUM_WORKERS = 4
    
    # ======================== Model Architecture ========================
    # Input channels
    RGB_CHANNELS = 3
    RGBD_CHANNELS = 4
    
    # Number of output classes (1 for dry weight prediction)
    OUTPUT_DIM = 1
    
    # CNN architecture parameters
    NUM_CONV_LAYERS = 3
    INITIAL_FILTERS = 32
    FILTER_MULTIPLIER = 2
    
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
    
    # Augmented data output directory
    AUGMENTED_OUTPUT_DIR = f"../../datasets/Training/Augmented/{VERSION}"
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {k: v for k, v in vars(cls).items() if not k.startswith('_') and not callable(v)}
    
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
