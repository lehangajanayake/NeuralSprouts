# model_v1: Lettuce Weight & Variety Prediction

This model predicts lettuce dry weight (regression) and variety (classification) from RGB and depth images using a deep multi-branch CNN in PyTorch.

## Quick Start

### 1. Install Requirements

```
pip install -r ../../requirements.txt
```

### 2. Preprocess the Data (Required!)

Before training, you **must** run the preprocessing script to generate augmented images and the new CSV:

```
python preprocess.py
```

This will create an `Augmented` folder in `datasets/Training/` and a new `Train_aug.csv` file. These are required for training.

### 3. Train the Model

Run the training script:

```
python -m models.model_v1.train
```

This will train the model using the augmented dataset and save the best model to `models/best_model.pth`.

### 4. Evaluate the Model

To evaluate the model on the training set and visualize predictions:

```
python eval.py
```

This will print sample predictions, error statistics, and show a scatter plot of actual vs predicted dry weight.

## File Structure

- `preprocess.py` — Preprocesses and augments the training data. **Run this first!**
- `train.py` — Trains the model on the augmented dataset.
- `eval.py` — Evaluates the model and visualizes predictions.
- `model.py` — Model architecture.
- `dataloader.py` — Data loading utilities.
- `augmented_dataloader.py` — Loads the augmented dataset.
- `../../datasets/Training/Augmented/` — Folder created by preprocessing, contains augmented images.
- `../../datasets/Training/Augmented/Train_aug.csv` — CSV created by preprocessing, used for training.

## Notes

- Make sure your images and CSVs are in the correct folders as expected by the scripts.
- The model expects 64x64 images. If your images are a different size, adjust `IMAGE_SIZE` in the scripts.
- Training and evaluation will use GPU if available, otherwise CPU.

## Troubleshooting

- If you get file not found errors, check that you have run the preprocessing step and that the paths in the scripts match your folder structure.
- For custom datasets, update the paths in the scripts accordingly.

---

For questions or issues, please open an issue or contact the project maintainer.
