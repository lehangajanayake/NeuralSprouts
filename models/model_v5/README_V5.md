# Model V5 - Enhanced Data Augmentation

## What's New in model_v5

model_v5 is a **copy of model_v4** with an enhanced preprocessing pipeline that includes **4 directional shift variants** in addition to the original random augmentations.

## Augmentation Strategy

For each original image (e.g., ID 15), model_v5 creates **25 variants**:

### Variants 1-20: Random Augmentations (Same as model_v4)
Each variant randomly applies:
- ✅ Horizontal flip (50% chance)
- ✅ Vertical flip (50% chance)
- ✅ 90° rotation (0°, 90°, 180°, 270°)
- ✅ Color jitter (brightness, contrast, saturation, hue)
- ✅ Center crop 900×900
- ✅ Resize to 64×64

Example: Original ID 15 → RGB_1.png to RGB_20.png (20 random variants)

### Variants 21-24: Directional Shifts (NEW!)
Each variant shifts the crop center by **10% of image dimensions**, then crops:

1. **Shift UP** (10% up)
   - Center crop window moves down
   - Captures upper portion of lettuce
   - No color changes, pure spatial shift

2. **Shift DOWN** (10% down)
   - Center crop window moves up
   - Captures lower portion of lettuce
   - No color changes, pure spatial shift

3. **Shift LEFT** (10% left)
   - Center crop window moves right
   - Captures left portion of lettuce
   - No color changes, pure spatial shift

4. **Shift RIGHT** (10% right)
   - Center crop window moves left
   - Captures right portion of lettuce
   - No color changes, pure spatial shift

All shifts are followed by standard cropping (900×900) and resizing (64×64).

## Output Structure

```
Original Images:
  - RGB_15.png (1920×1080)
  - Depth_15.png (1920×1080)

Generated Variants (25 per original):
  - RGB_1 to RGB_20: Random augmented variants
  - RGB_21: Shift up
  - RGB_22: Shift down
  - RGB_23: Shift left
  - RGB_24: Shift right

Plus same for next original (ID 16):
  - RGB_26 to RGB_50 (variants)
  - RGB_51: Shift up
  - ...etc
```

**Total dataset:**
- 230 originals × 25 variants = **5,750 training images**

## How to Run

```bash
cd /Users/hansikodikara/NeuralSprouts/models/model_v5

# Run preprocessing
python preprocess.py
```

**Expected output:**
```
Parallel preprocessing: originals=230, outputs per original=25, workers=7
Processed 25/230 originals...
Processed 50/230 originals...
...
Processed 230/230 originals...
Augmented images saved to: ../../datasets/Training/Augmented/RGBImages
Augmented CSV saved to: ../../datasets/Training/Augmented/Train_aug.csv (rows=5750)
```

## Configuration

Edit `PreprocessConfig` in `preprocess.py`:

```python
num_aug_per_image: int = 24    # 24 variants per original (was 20)
num_random_aug: int = 20       # First 20 are random
```

To adjust:
- **More random variants**: Increase `num_random_aug` and `num_aug_per_image`
- **Keep only directional shifts**: Set `num_random_aug = 0` and `num_aug_per_image = 4`
- **Fewer total variants**: Decrease `num_aug_per_image`

## Key Differences from model_v4

| Aspect | model_v4 | model_v5 |
|--------|----------|----------|
| **Variants per image** | 20 (1 original + 20 random) | 25 (1 original + 20 random + 4 shifts) |
| **Random augmentations** | ✅ All 20 | ✅ First 20 |
| **Directional shifts** | ❌ None | ✅ 4 (up/down/left/right) |
| **Color changes in shifts** | N/A | ❌ No, pure spatial |
| **Total training images** | 4,830 | 5,750 |
| **Preprocess.py** | 20 random variants | 20 random + 4 directional shifts |

## Why This Helps

1. **Robustness**: Model sees lettuce from multiple viewpoints (up, down, left, right)
2. **Deterministic**: Shifts are always in same directions (reproducible)
3. **No color distortion**: Shifts don't change appearance, pure spatial variation
4. **Comprehensive training**: Both random and systematic augmentation

## Files Modified

Only `preprocess.py` was changed in model_v5:
- Added `num_random_aug` config parameter
- Added `apply_directional_shift()` function
- Updated `_process_one_row()` to apply shifts for variants 21-24
- Updated `cfg_dict` to pass `num_random_aug` to workers

All other files (dataloader.py, model.py, train.py, etc.) are **identical to model_v4**.

## Next Steps

1. **Run preprocessing:**
   ```bash
   python preprocess.py
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Evaluate:**
   ```bash
   python eval.py
   ```

## Troubleshooting

**Q: Why 10% shift (not 5% or 20%)?**
A: 10% is a good balance between variety and safety (doesn't cut off lettuce)

**Q: Can I adjust shift percentage?**
A: Yes! In `apply_directional_shift()` function, change `shift_percent=0.1` to your desired value (0.05 = 5%, 0.15 = 15%, etc.)

**Q: Still want only 20 variants like model_v4?**
A: Set `num_aug_per_image = 20` in config

**Q: Want more total variants?**
A: Increase `num_aug_per_image` and/or `num_random_aug`

---

**Ready to preprocess!** 🚀
