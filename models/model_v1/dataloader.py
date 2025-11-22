import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class LettuceDataset(Dataset):
    """
    Dataset for lettuce images and dry weight labels.

    Customize the __init__ and __getitem__ methods based on your data format.
    """

    def __init__(self, RGB_dir , depth_dir, labels_file, image_size=64, augment=False):
        """
        Args:
            RGB_dir: Directory with RGB images
            depth_dir: Directory with depth images
            labels_file: CSV or file with labels
            image_size: Size to resize images to
        """
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.labels_file = labels_file
        self.image_size = image_size
        self.augment = augment
        import torchvision.transforms as T
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
        ]) if augment else None

        # Load image paths and labels from CSV and add to the dataframe

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

            #check if files exist
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Image not found: RGB: {rgb_path}, Depth: {depth_path}")
                self.df.drop(index, inplace=True)
                continue

            self.df.at[index, 'rgb_path'] = rgb_path
            self.df.at[index, 'depth_path'] = depth_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image paths and label
        rgb_path = self.df.iloc[idx]['rgb_path']
        depth_path = self.df.iloc[idx]['depth_path']
        image_id = self.df.iloc[idx]['id']

        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')

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
        rgb_np = np.array(image_rgb)  # Shape: (H, W, 3)
        depth_np = np.array(image_depth)  # Shape: (H, W) or (H, W, 1)

        # Ensure depth has a channel dimension
        if depth_np.ndim == 2:
            depth_np = depth_np[..., None]  # Shape: (H, W, 1)

        # Stack RGB and depth to get a 4-channel image
        combined_np = np.concatenate([rgb_np, depth_np], axis=2)  # Shape: (H, W, 4)

        # Convert to torch tensor and normalize to [0, 1]
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0  # Shape: (4, H, W)

        # Data augmentation (random flips and rotations) for training only
        if self.augment:
            # Convert to PIL for transforms
            import torchvision.transforms.functional as TF
            image_rgb = TF.to_pil_image(image[:3])
            image_depth = TF.to_pil_image(image[3].unsqueeze(0))
            # Use same random seed for both
            import random
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            random.seed(seed)
            image_rgb = self.aug_transform(image_rgb)
            torch.manual_seed(seed)
            random.seed(seed)
            image_depth = self.aug_transform(image_depth)
            # Convert back to numpy and stack
            rgb_np = np.array(image_rgb)
            depth_np = np.array(image_depth)[..., None]
            combined_np = np.concatenate([rgb_np, depth_np], axis=2)
            image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0

        # Get labels
        dry_weight = float(self.df.iloc[idx]['DryWeightShoot'])
        variety_class = int(self.df.iloc[idx]['VarietyClass'])

        dry_weight = torch.tensor(dry_weight, dtype=torch.float32)
        variety_class = torch.tensor(variety_class, dtype=torch.long)

        return image, dry_weight, variety_class, image_id



class TestLettuceDataset(Dataset):
    """
    Dataset for lettuce images without labels (for testing/inference).
    """

    def __init__(self, RGB_dir , depth_dir, image_size=64):
        """
        Args:
            RGB_dir: Directory with RGB images
            depth_dir: Directory with depth images
            image_size: Size to resize images to
        """
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.image_size = image_size

        # List all RGB images in the directory
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
        # Load image paths
        image_id = self.image_ids[idx]
        rgb_path = os.path.join(self.RGB_dir, f"RGB_{image_id}.png")
        depth_path = os.path.join(self.depth_dir, f"Depth_{image_id}.png")

        image_rgb = Image.open(rgb_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('L')

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
        rgb_np = np.array(image_rgb)  # Shape: (H, W, 3)
        depth_np = np.array(image_depth)  # Shape: (H, W) or (H, W, 1)


        combined_np = np.concatenate([rgb_np, depth_np[..., None]], axis=2)  # Shape: (H, W, 4)
        # Convert to torch tensor and normalize to [0, 1]
        image = torch.from_numpy(combined_np).permute(2, 0, 1).float() / 255.0  # Shape: (4, H, W)
        return image, image_id
