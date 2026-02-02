# LOGGING.md

## Logging & Versioning Strategy

- All experiments, logs, and checkpoints are organized in versioned subfolders (e.g., `6.1/`, `6.2/`).
- Each run is tracked for reproducibility and analysis.
- Augmentation logs are linked to image IDs for performance tracking and outlier detection.
- Log files record:
  - Augmentation parameters applied to each image
  - Training/validation metrics per run
  - Model checkpoints and configuration

---
Use this file to document your logging and versioning practices for model_v6.