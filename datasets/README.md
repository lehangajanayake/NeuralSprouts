# Datasets

Lettuce phenotyping dataset with RGB and depth images for dry weight prediction.

## Dataset Structure

```
datasets/
├── Training/
│   ├── Train.csv
│   ├── RGBImages/
│   │   ├── RGB_001.png
│   │   ├── RGB_002.png
│   │   └── ...
│   └── DepthImages/
│       ├── Depth_001.png
│       ├── Depth_002.png
│       └── ...
└── Test/
    ├── Test.csv
    ├── RGBImages/
    │   ├── RGB_001.png
    │   └── ...
    └── DepthImages/
        ├── Depth_001.png
        └── ...
```

## Data Format

### CSV Files

**Train.csv** and **Test.csv** contain the following columns:
- `image_id` (or `id`): Unique identifier for each sample
- `Variety`: Lettuce variety/type (string)
- `DryWeightShoot`: Target variable - dry shoot weight in grams (float)

Example:
```csv
image_id,Variety,DryWeightShoot
001,TypeA,12.5
002,TypeB,15.3
003,TypeA,11.8
...
```

### Image Files

- **RGB Images**: Color images of lettuce plants (PNG format)
  - Naming convention: `RGB_{image_id}.png`
  - Typical size: Variable (preprocessed to 900×900 center crop)

- **Depth Images**: Depth sensor images (PNG format, grayscale)
  - Naming convention: `Depth_{image_id}.png`
  - Typical size: Same as RGB images
  - Values represent distance/depth information

### Preprocessing

Most models apply the following preprocessing:
1. Center crop: 900×900 pixels
2. Resize: 64×64 or 224×224 (model-dependent)
3. Normalization: Model-specific (e.g., ImageNet stats for ResNet)
