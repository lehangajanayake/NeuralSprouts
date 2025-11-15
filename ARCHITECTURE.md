# Architecture Overview 🏛️

This document explains the architecture and design decisions of the NeuralSprouts framework.

## Design Philosophy

The framework is built around three core principles:

1. **Modularity**: Each component (models, augmentation, config) is independent and reusable
2. **Extensibility**: Easy to add new models and pipelines without modifying existing code
3. **Organization**: Built-in experiment tracking to manage multiple iterations

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       NeuralSprouts Framework                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Models     │  │ Augmentation │  │    Config    │         │
│  │              │  │              │  │              │         │
│  │ - BaseModel  │  │ - BasePipe   │  │ - Config     │         │
│  │ - SimpleCNN  │  │ - Generic    │  │ - YAML I/O   │         │
│  │ - CNN v2...  │  │ - Private    │  │              │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│                   ┌────────▼─────────┐                          │
│                   │     Trainer      │                          │
│                   │  - Training loop │                          │
│                   │  - Validation    │                          │
│                   │  - Checkpointing │                          │
│                   └────────┬─────────┘                          │
│                            │                                     │
│                   ┌────────▼──────────┐                         │
│                   │ ExperimentTracker │                         │
│                   │  - Run management │                         │
│                   │  - Metric logging │                         │
│                   │  - Result storage │                         │
│                   └───────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Models (`src/models/`)

**BaseModel** - Abstract base class for all models
- Provides common interface: `forward()`, checkpointing, parameter counting
- Ensures consistency across model iterations
- Handles model versioning

**SimpleCNN (v1)** - First iteration baseline model
- 3 convolutional blocks with batch normalization
- 2 fully connected layers with dropout
- Designed for 224x224 RGB images
- ~2-5M parameters (depends on input size)

**Future Models**
- You can add CNN v2, v3, ResNet-based, EfficientNet-based, etc.
- All inherit from BaseModel for consistent interface

### 2. Data Augmentation (`src/data_augmentation/`)

**AugmentationPipeline** - Abstract base class
- Defines interface for all augmentation strategies
- Ensures pipelines are callable and describable

**GenericAugmentationPipeline** - Shared across models
- Uses Albumentations library
- Training mode: Full augmentation suite
- Validation mode: Only resize + normalize
- Configurable augmentation probability

**PrivatePipelineTemplate** - Model-specific pipelines
- Template for creating custom augmentation strategies
- Examples: Aggressive, Conservative pipelines
- Use for fine-tuning specific model versions

### 3. Configuration (`src/config/`)

**Config** - Centralized configuration management
- YAML-based configuration files
- Dot notation for nested access: `config.get('model.input_size')`
- Default values with easy overrides
- Sections: experiment, model, data, augmentation, training, output

### 4. Utilities (`src/utils/`)

**ExperimentTracker** - Organize experiments
- Creates structured directory layout
- Tracks multiple runs with unique IDs
- Logs metrics per epoch
- Saves run metadata and results
- Enables run comparison

**Trainer** - Generic training pipeline
- Handles training loop, validation, and checkpointing
- Automatic learning rate scheduling
- Early stopping support
- Integration with ExperimentTracker
- Loss function: MSE (regression task)

## Data Flow

```
┌─────────────┐
│  Raw Image  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Augmentation   │ ◄── GenericPipeline or PrivatePipeline
│   Pipeline      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Tensor (CHW)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│     Model       │ ◄── SimpleCNN, CNN_v2, etc.
│   (forward)     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Prediction     │
│  (Dry Weight)   │
└─────────────────┘
```

## Training Flow

```
1. Load Config
   ├── Model params (architecture, dropout, etc.)
   ├── Data params (paths, batch size, etc.)
   ├── Training params (lr, epochs, etc.)
   └── Augmentation params (pipeline type, probability)

2. Initialize Components
   ├── Create Model (e.g., SimpleCNN)
   ├── Create Augmentation Pipelines (train + val)
   ├── Create Datasets with pipelines
   └── Create DataLoaders

3. Setup Tracking
   ├── Initialize ExperimentTracker
   ├── Create experiment directory structure
   └── Start new run with unique ID

4. Training Loop (Trainer)
   ├── For each epoch:
   │   ├── Train on training set
   │   ├── Validate on validation set
   │   ├── Log metrics
   │   ├── Update learning rate
   │   ├── Save checkpoint if best
   │   └── Check early stopping
   └── Return final results

5. Save Results
   ├── Best model checkpoint
   ├── Training metrics log
   ├── Final results JSON
   └── Run metadata
```

## Extension Points

### Adding a New Model

1. Create `src/models/your_model.py`
2. Inherit from `BaseModel`
3. Implement `__init__()` and `forward()`
4. Register in `src/models/__init__.py`
5. Create config file in `configs/`

### Adding a New Augmentation Pipeline

1. Create file in `src/data_augmentation/`
2. Inherit from `AugmentationPipeline`
3. Implement `__call__()` and `get_description()`
4. Use in your training script

### Customizing Training

1. Subclass `Trainer` for custom training logic
2. Override `train_epoch()`, `validate()`, or other methods
3. Add custom metrics or callbacks

## File Organization

```
experiments/
└── {experiment_name}/
    ├── checkpoints/
    │   ├── {run_id}_best.pth       # Best model weights
    │   └── {run_id}_epoch_50.pth   # Epoch checkpoints (optional)
    ├── configs/
    │   └── {run_id}_config.json    # Run configuration
    ├── logs/
    │   └── {run_id}_metrics.jsonl  # Per-epoch metrics
    ├── results/
    │   └── {run_id}_results.json   # Final results
    └── metadata.json               # All runs metadata
```

## Design Decisions

### Why Abstract Base Classes?
- Ensures consistency across iterations
- Makes it easy to swap models/pipelines
- Provides type hints and documentation

### Why Separate Generic and Private Pipelines?
- Generic: Test different augmentations quickly
- Private: Fine-tune for specific models later
- Flexibility without duplication

### Why YAML Configs?
- Human-readable and editable
- Easy version control
- Standard in ML projects

### Why ExperimentTracker?
- Prevents confusion with multiple runs
- Easy comparison between iterations
- Reproducibility through config versioning

## Performance Considerations

### GPU Optimization
- Uses `pin_memory=True` in DataLoaders
- Automatic device detection (CUDA/CPU)
- Batch processing in training loop

### Memory Efficiency
- Configurable batch size and num_workers
- Gradient accumulation possible (extend Trainer)
- Optional mixed precision training (can be added)

### Training Speed
- DataLoader multiprocessing
- Efficient augmentation with Albumentations
- Early stopping to avoid unnecessary epochs

## Testing Strategy

### Unit Tests
- `test_models.py`: Model initialization, forward pass, checkpointing
- `test_augmentation.py`: Pipeline creation, transforms, output shapes
- `test_config.py`: Config management, YAML I/O, updates

### Integration Testing (Manual)
- Run `example_usage.py` with dummy data
- Verify experiment tracking works
- Check checkpoint saving/loading

## Future Enhancements

Possible extensions to consider:

1. **Advanced Models**: ResNet, EfficientNet, Vision Transformers
2. **Ensemble Methods**: Combine multiple models
3. **TensorBoard Integration**: Enhanced visualization
4. **Hyperparameter Tuning**: Optuna/Ray Tune integration
5. **Data Validation**: Automated data quality checks
6. **Model Export**: ONNX export for deployment
7. **Distributed Training**: Multi-GPU support
8. **Mixed Precision**: AMP for faster training

## Summary

The NeuralSprouts framework provides:
- ✅ Clean separation of concerns
- ✅ Easy model iteration and comparison
- ✅ Flexible augmentation strategies
- ✅ Comprehensive experiment tracking
- ✅ Production-ready training pipeline
- ✅ Extensible architecture

Start with the SimpleCNN baseline, iterate on your architecture, experiment with augmentation, and use the experiment tracker to find your best model!
