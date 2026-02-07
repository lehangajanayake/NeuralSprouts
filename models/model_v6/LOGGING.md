# LOGGING.md

## Logging & Versioning Strategy for Model_v6

### Overview
Model_v6 implements a comprehensive logging and versioning system to track all experiments, hyperparameters, augmentations, and results. This enables reproducibility and easy comparison across different versions.

## Versioning Structure

Each experiment is organized in a versioned folder with the following structure:

```
experiments/
├── 6.1/
│   ├── config.json              # Configuration snapshot
│   ├── logs/
│   │   └── training.log         # Training log file
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_1.pth
│   │   ├── checkpoint_epoch_2.pth
│   │   └── best_model.pth       # Best model
│   ├── predictions/
│   │   └── predictions.csv      # Predictions on test set
│   ├── training_history.json    # Training metrics
│   └── visualizations/
│       ├── predictions_analysis.png
│       └── top_errors/
├── 6.2/
│   └── ...
└── 6.3/
    └── ...
```

## Augmentation Logging

Each preprocessed image has its augmentations logged in CSV format for full traceability and outlier analysis.

## Best Practices

1. **Document Changes**: When creating a new version, note what you changed
2. **Keep Versions**: Never delete old versions for reproducibility
3. **Track Hyperparameters**: Always update config.py before running
4. **Analyze Augmentations**: Use augmentation logs to understand model behavior
5. **Compare Systematically**: Create structured experiments

---
For more details, see VISUALIZATION.md and QUICK_START.md.