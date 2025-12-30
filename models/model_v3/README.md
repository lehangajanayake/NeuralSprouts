# Model V3: ResNet18 (regression)

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
