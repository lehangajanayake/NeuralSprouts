import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class PlantDatasetV2(Dataset):
    def __init__(self, csv_file, rgb_dir, depth_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        # Persistent mapping from variety to class index
        self.varieties = sorted(self.data['Variety'].unique())
        self.variety_map = {v: i for i, v in enumerate(self.varieties)}
        print('Variety to class mapping:', self.variety_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # You may need to adjust these column names to match your CSV
        # e.g., row['image_id'] for image filename, etc.
        rgb_path = os.path.join(self.rgb_dir, f"RGB_{row['id']}.png")
        depth_path = os.path.join(self.depth_dir, f"Depth_{row['id']}.png")
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')
        rgb_tensor = self.transform(rgb_image)
        depth_tensor = self.transform(depth_image)

        # Stack RGB and depth: [4, H, W]
        x = torch.cat([rgb_tensor, depth_tensor], dim=0)
        y_reg = torch.tensor(row['DryWeightShoot'], dtype=torch.float32)
        # Map Variety to class index using persistent mapping
        y_class = torch.tensor(self.variety_map[row['Variety']], dtype=torch.long)
        leaf_area = torch.tensor(row['LeafArea'], dtype=torch.float32)
        return x, y_reg, y_class, leaf_area
