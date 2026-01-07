# Model V3: ResNet18 (regression)

## Model Description

**Model v3** represents a strategic shift from custom CNNs to transfer learning, utilizing a pretrained ResNet18 backbone for simplified single-task dry weight regression. This model serves as a strong baseline to evaluate the benefits of transfer learning versus custom architectures.

### Architecture Overview
- **Backbone**: Pretrained ResNet18 (ImageNet weights)
- **Task**: Regression only - predicting dry shoot weight
- **Input**: 3×224×224 RGB images (standard ImageNet preprocessing)
- **Output**: Single scalar value for dry weight

### Key Features
- **Transfer Learning**: Leverages ImageNet pretrained features
- **Simplified Architecture**: No multi-branch design, focuses on regression only
- **Standard Preprocessing**: ResNet-compatible normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Regression Head**: ResNet features → Linear layer → Dry weight prediction
- **Multi-view Augmentation Support**: Option B cached augmentation (N×K samples)

### Training Strategy
- Fine-tuning pretrained ResNet18 weights
- MAE (Mean Absolute Error) as primary metric
- MSE loss for optimization
- Reproducible seeding for consistency
- Early stopping and checkpointing

### Debugging Tools
- **Streamlit Web UI** (`debug_dataloader_app.py`): Interactive dataloader inspection
- Visual debugging for augmentation and preprocessing validation
- Grid and single-view visualization modes

This is a baseline model using a pretrained ResNet18 for **regression only** (predicting `DryWeightShoot`).

## Files

- `dataloader.py`: Loads RGB images and labels, applies normalization for ResNet. Supports cached multi-view augmentation (Option B: N\*K samples).
- `model.py`: ResNet18 feature extractor + regression head.
- `train.py`: Training script (MAE + MSE plots), reproducible seeding, optional cached multi-view loading.
- `eval.py`: Evaluate a checkpoint (MAE + plots + worst samples).
- `predict.py`: Generate `Test_with_predictions_v3.csv` from `best_model_v3.pth`.
- `debug_dataloader_app.py`: Streamlit web UI to inspect what the dataloader is producing (grid + single view).

## Usage

- Update the paths in `train.py` if needed.
- Run training: `python train.py`

### Dataloader debugger (web UI)

Run from this folder (`models/model_v3`) with Streamlit:

```cmd
streamlit run debug_dataloader_app.py
```

If `streamlit` isn’t recognized (common on Windows when the venv Scripts folder isn’t on PATH), run it via Python instead:

```cmd
python -m streamlit run debug_dataloader_app.py
```

Or explicitly using this repo’s venv Python:

```cmd
C:\Users\gajan\Documents\Projects\NeuralSprouts\.venv\Scripts\python.exe -m streamlit run debug_dataloader_app.py
```

## Requirements

- torch
- torchvision
- pandas
- matplotlib
- streamlit

## Notes

- Only RGB images are used (no depth or leaf area).
- Images are resized and normalized for ResNet.
- Cached multi-view augmentation can use a lot of RAM: roughly $N \times K \times 3 \times 224 \times 224 \times 4$ bytes.
