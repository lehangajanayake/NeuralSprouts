
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class PlantDatasetV2(Dataset):
    def __init__(self, RGB_dir, depth_dir, labels_file, image_size=64):
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.labels_file = labels_file
        self.image_size = image_size

        self.df = pd.read_csv(labels_file)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)

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

        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')

        # Convert images to numpy arrays
        rgb_np = np.array(image_rgb)  # Shape: (H, W, 3)
        depth_np = np.array(image_depth)  # Shape: (H, W) or (H, W, 1)
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]  # Shape: (H, W, 1)
        combined_np = np.concatenate([rgb_np, depth_np], axis=2)  # Shape: (H, W, 4)
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0  # Shape: (4, H, W)

        dry_weight = float(self.df.iloc[idx]['DryWeightShoot'])
        variety_class = int(self.df.iloc[idx]['VarietyClass'])
        leaf_area = float(self.df.iloc[idx]['LeafArea']) if 'LeafArea' in self.df.columns else 0.0

        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)
        variety_class = torch.tensor(variety_class, dtype=torch.long)
        leaf_area = torch.tensor(leaf_area, dtype=torch.float32)

        return image, dry_weight, variety_class, leaf_area



class TestPlantDatasetV2(Dataset):
    """
    Dataset for test/inference: expects only 'id' column in CSV, loads RGB and depth images, returns image tensor and id.
    """
    def __init__(self, RGB_dir, depth_dir, csv_file, image_size=64):
        import pandas as pd
        import os
        from PIL import Image
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        for index, row in self.df.iterrows():
            id = row['image_id'] if 'image_id' in self.df.columns else row['id']
            rgb_path = os.path.join(self.RGB_dir, f"RGB_{id}.png")
            depth_path = os.path.join(self.depth_dir, f"Depth_{id}.png")
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Image not found: RGB: {rgb_path}, Depth: {depth_path}")
                self.df.drop(index, inplace=True)
                continue

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import numpy as np
        import torch
        from PIL import Image
        rgb_path = self.df.iloc[idx]['rgb_path']
        depth_path = self.df.iloc[idx]['depth_path']
        image_id = self.df.iloc[idx]['image_id']
        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')
        dryweight = self.df.iloc[idx].get('DryWeightShoot')
        # crop to square from center to a size of 900x900
        width, height = image_rgb.size
        left = (width - 900) / 2
        top = (height - 900) / 2
        right = (width + 900) / 2
        bottom = (height + 900) / 2
        image_rgb = image_rgb.crop((left, top, right, bottom))
        image_depth = image_depth.crop((left, top, right, bottom))
        # Resize images
        image_rgb = image_rgb.resize((self.image_size, self.image_size))
        image_depth = image_depth.resize((self.image_size, self.image_size))
        # Convert images to numpy arrays
        rgb_np = np.array(image_rgb)
        depth_np = np.array(image_depth)
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]
        combined_np = np.concatenate([rgb_np, depth_np], axis=2)
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0
        return image, dryweight,  image_id
