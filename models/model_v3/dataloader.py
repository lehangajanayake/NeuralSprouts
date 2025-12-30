import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

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

        self.transform = transforms.Compose([
            transforms.CenterCrop(900),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        rgb_path = self.df.iloc[idx]['rgb_path']
        image_rgb = Image.open(rgb_path).convert('RGB')
        image = self.transform(image_rgb)
        dry_weight = float(self.df.iloc[idx]['DryWeightShoot'])
        variety_class = int(self.df.iloc[idx]['VarietyClass'])
        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)
        variety_class = torch.tensor(variety_class, dtype=torch.long)
        return image, dry_weight, variety_class
