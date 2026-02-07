# QUICK_START.md

## Quick Start Guide for Model_v6

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Experiment Directory**
   ```bash
   python setup.py
   ```
   This creates the directory structure for experiment version 6.1 with logs, checkpoints, and visualization folders.

### Workflow

#### Step 1: Configure Preprocessing
Edit `config.py` to set your desired preprocessing and augmentation parameters:
- Center crop size (default: 1000x1000)
- Resize size (default: 96x96)
- Augmentation probabilities and ranges

#### Step 2: Preprocess Dataset
```bash
python preprocess_dataset.py
```
This will:
- Load RGB and Depth images
- Apply center cropping and resizing
- Apply configured augmentations
- Log all augmentations per image ID
- Save preprocessed images to `../../datasets/Training/Augmented/6.1/`

#### Step 3: Train Model
```bash
python train.py
```
This will:
- Load preprocessed training data
- Initialize the dual-branch CNN (RGB + RGBD + Fusion)
- Train for configured epochs
- Save checkpoints and best model
- Log training metrics to `experiments/6.1/logs/`

#### Step 4: Make Predictions
```bash
python predict.py
```
This will:
- Load the best trained model
- Make predictions on test set
- Save predictions to CSV with error analysis
- Generate attention maps for visualization

#### Step 5: Visualize Results
```bash
python visualize.py
```
This will:
- Create prediction vs ground truth plots
- Visualize error distribution
- Show attention maps from both branches
- Highlight top error cases
- Save visualizations to `experiments/6.1/visualizations/`

### File Structure

```
model_v6/
├── config.py                 # Configuration class (edit this!)
├── preprocess.py             # Preprocessing and augmentation
├── model.py                  # Dual-branch CNN architecture
├── dataloader.py             # Data loading utilities
├── train.py                  # Training loop
├── predict.py                # Inference script
├── visualize.py              # Visualization tools
├── setup.py                  # Experiment setup
├── requirements.txt          # Dependencies
└── experiments/
    └── 6.1/                  # Version 6.1 results
        ├── logs/             # Training logs
        ├── checkpoints/      # Model checkpoints
        ├── predictions/      # Prediction results
        └── visualizations/   # Generated visualizations
```

### Key Features

- **Dual-Branch CNN:** Separate RGB and RGBD branches with fusion layer
- **Configurable Preprocessing:** All augmentation parameters are customizable
- **Augmentation Logging:** Track which augmentations were applied to each image
- **Versioning:** Organize experiments in version folders (6.1, 6.2, etc.)
- **Visualization:** Attention maps, error analysis, and augmentation effects
- **GPU Optimized:** Pre-processes data before training for fast GPU utilization

### Troubleshooting

- **Out of Memory:** Reduce `BATCH_SIZE` in config.py
- **Slow Training:** Ensure preprocessing is complete and data is on disk
- **Missing Images:** Check dataset paths in config.py
- **CUDA Not Available:** Falls back to CPU automatically

### Next Steps

1. Try different augmentation parameters and track results in different versions (6.2, 6.3, etc.)
2. Analyze augmentation logs to understand which transformations help predictions
3. Use visualization tools to identify outliers and failure cases
4. Experiment with model architecture by editing `config.py` (e.g., `NUM_CONV_LAYERS`, `FUSION_HIDDEN_DIM`)
