import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

try:
    from torchvision import transforms
except Exception:
    transforms = None
    import numpy as np

class SimplePlantDataset(Dataset):
    def __init__(self, RGB_dir, labels_file, image_size=224):
        self.RGB_dir = RGB_dir
        self.labels_file = labels_file
        self.image_size = image_size
        self.df = pd.read_csv(labels_file)
        if 'image_id' in self.df.columns:
            self.df.rename(columns={'image_id': 'id'}, inplace=True)
        if 'Variety' not in self.df.columns:
            raise ValueError("CSV must contain a 'Variety' column for classification")
        if 'DryWeightShoot' not in self.df.columns:
            raise ValueError("CSV must contain a 'DryWeightShoot' column for regression")

        self.variety2idx = {v: i for i, v in enumerate(sorted(self.df['Variety'].unique()))}
        self.df['VarietyClass'] = self.df['Variety'].map(self.variety2idx)
        print(f"Variety to class mapping: {self.variety2idx}")

        keep_rows = []
        for _, row in self.df.iterrows():
            image_id = row['id']
            rgb_path = os.path.join(self.RGB_dir, f"RGB_{image_id}.png")
            if not os.path.exists(rgb_path):
                print(f"Image not found: RGB: {rgb_path}")
                continue
            row = row.copy()
            row['rgb_path'] = rgb_path
            keep_rows.append(row)
        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)

        if transforms is None:
            # Fallback if torchvision isn't importable.
            self.image_size = (self.image_size, self.image_size)
            self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rgb_path = self.df.iloc[idx]['rgb_path']

        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb = self.transform(rgb_img)
        else:
            rgb_img = rgb_img.resize(self.image_size, resample=Image.BILINEAR)
            rgb_np = np.asarray(rgb_img, dtype=np.float32) / 255.0  # (H, W, C)
            rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()  # (C, H, W)
            rgb = (rgb - self._mean) / self._std

        dry_weight = float(self.df.iloc[idx]['DryWeightShoot'])
        variety_class = int(self.df.iloc[idx]['VarietyClass'])

        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)
        variety_class = torch.tensor(variety_class, dtype=torch.long)
        return rgb, dry_weight, variety_class
