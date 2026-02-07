# VISUALIZATION.md

## Visualization & Debugging Tools for Model_v6

Model_v6 provides comprehensive visualization tools to understand model behavior, identify failure cases, and debug predictions.

## Built-in Visualization Functions

### 1. Attention Map Visualization

Visualizes what each branch of the model "sees" in the image:

```python
from visualize import ModelVisualizer

visualizer = ModelVisualizer("experiments/6.1/best_model.pth")
visualizer.visualize_attention_maps(
    rgb_image=rgb_array,
    rgbd_image=rgbd_array,
    image_id="img_001",
    output_path="attention_maps.png"
)
```

**Output:**
- RGB image and its attention map
- RGBD image and its attention map  
- Overlaid attention maps on original images
- Helps identify if model focuses on correct plant regions

### 2. Prediction Analysis

Comprehensive prediction visualization:

```python
visualizer.visualize_predictions(
    predictions_df=results_df,
    output_path="predictions_analysis.png"
)
```

**Output:**
- Scatter plot: Predicted vs Actual values
- Error distribution histogram
- Absolute error by actual value
- Perfect prediction reference line

### 3. Top Errors Visualization

Identifies and visualizes predictions with largest errors:

```python
visualizer.visualize_top_errors(
    predictions_df=results_df,
    rgb_dir="datasets/Test/RGBImages",
    rgbd_dir="datasets/Test/DepthImages",
    num_samples=5,
    output_dir="top_errors"
)
```

**Output:**
- Images with top 5 largest prediction errors
- Actual vs predicted values for each
- Error magnitudes
- Helps identify systematic failure patterns

### 4. Augmentation Effect Visualization

Shows what augmentations were applied to each image:

```python
visualizer.visualize_augmentation_effects(
    original_image_path="datasets/Training/RGBImages/img_001.jpg",
    augmentation_log_path="logs/augmentations.csv",
    image_id="img_001",
    output_path="augmentation_effects.png"
)
```

**Output:**
- Original image
- List of applied augmentations (flips, rotations, shifts)
- Useful for understanding data augmentation pipeline

## Command-Line Usage

```bash
python main.py visualize --model experiments/6.1/best_model.pth \
                         --predictions experiments/6.1/predictions.csv
```

## Debugging Checklist

1. **Attention Maps**
   - [ ] Do RGB attention maps focus on plant regions?
   - [ ] Do RGBD attention maps show depth features?
   - [ ] Are both branches learning complementary features?

2. **Error Patterns**
   - [ ] Are errors uniformly distributed or clustered?
   - [ ] Do errors correlate with specific augmentations?
   - [ ] Do certain image types have consistently high errors?

3. **Model Behavior**
   - [ ] Is train/val loss converging?
   - [ ] Is there overfitting?
   - [ ] Are both branches contributing?

---
For more details, see QUICK_START.md and CONFIG.md.