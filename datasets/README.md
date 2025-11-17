# Datasets

Place your lettuce image datasets here.

## Expected Structure

```
datasets/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── val/
│   ├── image_001.jpg
│   └── ...
├── test/
│   └── ...
├── train_labels.csv
├── val_labels.csv
└── test_labels.csv
```

## Label Format

CSV files should have at least two columns:
- `image_name`: Filename of the image
- `dry_weight`: The dry weight value (target variable)

Example `train_labels.csv`:
```csv
image_name,dry_weight
image_001.jpg,12.5
image_002.jpg,15.3
...
```
