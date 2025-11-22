import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class AugmentedLettuceDataset(Dataset):
    """
    Dataset for augmented lettuce images and dry weight labels.
    """
    def __init__(self, RGB_dir, depth_dir, labels_file, image_size=64):
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.labels_file = labels_file
        self.image_size = image_size
        self.df = pd.read_csv(labels_file)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)
        # Encode 'Variety' as integer class
        self.variety2idx = {v: i for i, v in enumerate(sorted(self.df['Variety'].unique()))}
        self.df['VarietyClass'] = self.df['Variety'].map(self.variety2idx)
        print(f"Variety to class mapping: {self.variety2idx}")
        for index, row in self.df.iterrows():
            id = row['id']
            rgb_path = os.path.join(self.RGB_dir, f"RGB_{id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{id}.png")
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Image not found: RGB: {rgb_path}, Depth: {depth_path}")
                self.df.drop(index, inplace=True)
                continue
            self.df.at[index, 'rgb_path'] = rgb_path
            self.df.at[index, 'depth_path'] = depth_path
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        rgb_path = self.df.iloc[idx]['rgb_path']
        depth_path = self.df.iloc[idx]['depth_path']
        image_id = self.df.iloc[idx]['id']
        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')
        rgb_np = np.array(image_rgb)
        depth_np = np.array(image_depth)
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]
        combined_np = np.concatenate([rgb_np, depth_np], axis=2)
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0
        dry_weight = float(self.df.iloc[idx]['DryWeightShoot'])
        variety_class = int(self.df.iloc[idx]['VarietyClass'])
        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)
        variety_class = torch.tensor(variety_class, dtype=torch.long)
        return image, dry_weight, variety_class, image_id

class AugmentedTestLettuceDataset(Dataset):
    """
    Dataset for augmented lettuce test images without labels.
    """
    def __init__(self, RGB_dir, depth_dir, image_size=64):
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.image_ids = []
        for filename in os.listdir(RGB_dir):
            if filename.startswith("RGB_") and filename.endswith(".png"):
                id = filename[len("RGB_"):-len(".png")]
                rgb_path = os.path.join(self.RGB_dir, filename)
                depth_path = os.path.join(self.depth_dir, f"Depth_{id}.png")
                if os.path.exists(depth_path):
                    self.image_ids.append(id)
                else:
                    print(f"Depth image not found for ID {id}, skipping.")
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rgb_path = os.path.join(self.RGB_dir, f"RGB_{image_id}.png")
        depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")
        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')
        rgb_np = np.array(image_rgb)
        depth_np = np.array(image_depth)
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]
        combined_np = np.concatenate([rgb_np, depth_np], axis=2)
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0
        return image, image_id
