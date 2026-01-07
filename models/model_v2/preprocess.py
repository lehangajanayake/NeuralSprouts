import os
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T

# Paths
TRAIN_RGB = '../../datasets/Training/RGBImages'
TRAIN_DEPTH = '../../datasets/Training/DepthImages'
LABELS_CSV = '../../datasets/Training/Train.csv'
AUG_RGB = '../../datasets/Training/Augmented/RGBImages'
AUG_DEPTH = '../../datasets/Training/Augmented/DepthImages'
AUG_CSV = '../../datasets/Training/Augmented/Train_aug.csv'

# Create output dirs
os.makedirs(AUG_RGB, exist_ok=True)
os.makedirs(AUG_DEPTH, exist_ok=True)

# Augmentation transforms
flip = T.RandomHorizontalFlip(p=1.0)
brightness = T.ColorJitter(brightness=0.5)
rotate = T.RandomRotation(30)
resize = T.Resize((64, 64))

# Read CSV
df = pd.read_csv(LABELS_CSV)
if 'image_id' in df.columns:
    df.rename(columns={'image_id': 'id'}, inplace=True)

aug_rows = []


# Generate unique integer IDs for each augmented image
next_id = 1
id_map = {}

for idx, row in df.iterrows():
    orig_id = row['id']
    rgb_path = os.path.join(TRAIN_RGB, f'RGB_{orig_id}.png')
    depth_path = os.path.join(TRAIN_DEPTH, f'Depth_{orig_id}.png')
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Missing: {rgb_path} or {depth_path}")
        continue
    # Load images
    rgb = Image.open(rgb_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')

    # --- Save original ---
    w, h = rgb.size
    left = (w - 900) / 2
    top = (h - 900) / 2
    right = (w + 900) / 2
    bottom = (h + 900) / 2
    rgb_cropped = rgb.crop((left, top, right, bottom))
    depth_cropped = depth.crop((left, top, right, bottom))
    rgb_resized = resize(rgb_cropped)
    depth_resized = resize(depth_cropped)
    new_id = next_id
    rgb_resized.save(os.path.join(AUG_RGB, f'RGB_{new_id}.png'))
    depth_resized.save(os.path.join(AUG_DEPTH, f'Depth_{new_id}.png'))
    aug_rows.append({**row, 'id': new_id})
    next_id += 1

    # --- Augment: flip (augment, then crop/resize/save) ---
    rgb_flip = flip(rgb)
    depth_flip = flip(depth)
    rgb_flip_cropped = rgb_flip.crop((left, top, right, bottom))
    depth_flip_cropped = depth_flip.crop((left, top, right, bottom))
    rgb_flip_resized = resize(rgb_flip_cropped)
    depth_flip_resized = resize(depth_flip_cropped)
    new_id = next_id
    rgb_flip_resized.save(os.path.join(AUG_RGB, f'RGB_{new_id}.png'))
    depth_flip_resized.save(os.path.join(AUG_DEPTH, f'Depth_{new_id}.png'))
    aug_rows.append({**row, 'id': new_id})
    next_id += 1

    # --- Augment: rotate (augment, then crop/resize/save) ---
    rgb_rot = rotate(rgb)
    depth_rot = rotate(depth)
    rgb_rot_cropped = rgb_rot.crop((left, top, right, bottom))
    depth_rot_cropped = depth_rot.crop((left, top, right, bottom))
    rgb_rot_resized = resize(rgb_rot_cropped)
    depth_rot_resized = resize(depth_rot_cropped)
    new_id = next_id
    rgb_rot_resized.save(os.path.join(AUG_RGB, f'RGB_{new_id}.png'))
    depth_rot_resized.save(os.path.join(AUG_DEPTH, f'Depth_{new_id}.png'))
    aug_rows.append({**row, 'id': new_id})
    next_id += 1

    # --- Augment: brightness (augment, then crop/resize/save) ---
    rgb_bright = brightness(rgb)
    depth_bright = brightness(depth)
    rgb_bright_cropped = rgb_bright.crop((left, top, right, bottom))
    depth_bright_cropped = depth_bright.crop((left, top, right, bottom))
    rgb_bright_resized = resize(rgb_bright_cropped)
    depth_bright_resized = resize(depth_bright_cropped)
    new_id = next_id
    rgb_bright_resized.save(os.path.join(AUG_RGB, f'RGB_{new_id}.png'))
    depth_bright_resized.save(os.path.join(AUG_DEPTH, f'Depth_{new_id}.png'))
    aug_rows.append({**row, 'id': new_id})
    next_id += 1

# Write new CSV
aug_df = pd.DataFrame(aug_rows)
aug_df.to_csv(AUG_CSV, index=False)
print(f"Augmented images and CSV saved to {AUG_CSV}")
