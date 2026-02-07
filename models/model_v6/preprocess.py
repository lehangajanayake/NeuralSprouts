"""
Preprocessing module for Model_v6.
Handles center cropping, resizing, augmentations, and logging.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

from config import Config


class PreprocessingLogger:
    """Logs augmentations applied to each image."""
    
    def __init__(self, log_dir: str, log_file: str = "augmentations.csv"):
        """Initialize preprocessing logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file
        
        # Create CSV header if file doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'image_id', 'horizontal_flip', 'vertical_flip', 'rotation_angle',
                    'horizontal_shift', 'vertical_shift', 'augmentation_applied'
                ])
                writer.writeheader()
    
    def log_augmentation(self, image_id: str, augmentation_params: Dict):
        """Log augmentations applied to an image."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image_id', 'horizontal_flip', 'vertical_flip', 'rotation_angle',
                'horizontal_shift', 'vertical_shift', 'augmentation_applied'
            ])
            writer.writerow({
                'image_id': image_id,
                'horizontal_flip': augmentation_params.get('horizontal_flip', False),
                'vertical_flip': augmentation_params.get('vertical_flip', False),
                'rotation_angle': augmentation_params.get('rotation_angle', 0),
                'horizontal_shift': augmentation_params.get('horizontal_shift', 0),
                'vertical_shift': augmentation_params.get('vertical_shift', 0),
                'augmentation_applied': augmentation_params.get('augmentation_applied', False)
            })


class ImagePreprocessor:
    """Handles image preprocessing: center crop, resize, and augmentations."""
    
    def __init__(self, config: Config = None):
        """Initialize preprocessor with config."""
        self.config = config or Config()
        self.logger = PreprocessingLogger(self.config.LOG_DIR, self.config.AUGMENTATION_LOG_FILE)
        
        # Setup augmentation pipeline
        self.transform = self._setup_augmentations()
    
    def _setup_augmentations(self):
        """Setup augmentation pipeline using torchvision transforms."""
        aug_list = []
        
        if self.config.AUGMENTATIONS_ENABLED:
            if self.config.HORIZONTAL_FLIP_ENABLED:
                aug_list.append(transforms.RandomHorizontalFlip(p=self.config.HORIZONTAL_FLIP_PROB))
            
            if self.config.VERTICAL_FLIP_ENABLED:
                aug_list.append(transforms.RandomVerticalFlip(p=self.config.VERTICAL_FLIP_PROB))
            
            if self.config.ROTATION_ENABLED:
                aug_list.append(transforms.RandomRotation(
                    degrees=self.config.ROTATION_ANGLE_RANGE,
                    fill=0
                ))
            
            if self.config.HORIZONTAL_SHIFT_ENABLED or self.config.VERTICAL_SHIFT_ENABLED:
                # Use RandomAffine for translation
                translate = None
                if self.config.HORIZONTAL_SHIFT_ENABLED or self.config.VERTICAL_SHIFT_ENABLED:
                    translate = (self.config.HORIZONTAL_SHIFT_MAX, self.config.VERTICAL_SHIFT_MAX)
                aug_list.append(transforms.RandomAffine(
                    degrees=0,
                    translate=translate,
                    fill=0
                ))
        
        # Always resize at the end
        aug_list.append(transforms.Resize((self.config.RESIZE_SIZE, self.config.RESIZE_SIZE), 
                                          interpolation=transforms.InterpolationMode.LANCZOS))
        
        return transforms.Compose(aug_list)
    
    def center_crop(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        """Center crop an image."""
        h, w = image.shape[:2]
        if h < crop_size or w < crop_size:
            # Pad if image is smaller than crop size
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)), mode='constant')
            h, w = image.shape[:2]
        
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        return image[start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    def preprocess_image(self, image_path: str, apply_augmentation: bool = True) -> Tuple[np.ndarray, Dict]:
        """Preprocess a single image."""
        # Load image using PIL
        image = Image.open(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(image)
        
        # Center crop
        image = self.center_crop(image, self.config.CENTER_CROP_SIZE)
        
        # Track augmentations
        augmentation_params = {
            'horizontal_flip': False,
            'vertical_flip': False,
            'rotation_angle': 0,
            'horizontal_shift': 0,
            'vertical_shift': 0,
            'augmentation_applied': False
        }
        
        # Apply augmentations using torch transforms
        if apply_augmentation and self.config.AUGMENTATIONS_ENABLED:
            augmentation_params['augmentation_applied'] = True
            # Convert to PIL for torchvision transforms
            pil_image = Image.fromarray(image)
            # Apply transforms (returns tensor, convert back to numpy)
            augmented_tensor = self.transform(pil_image)
            image = (augmented_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        else:
            # Just resize using PIL
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((self.config.RESIZE_SIZE, self.config.RESIZE_SIZE), Image.LANCZOS)
            image = np.array(pil_image)
        
        return image, augmentation_params
    
    def preprocess_rgb_depth_pair(self, rgb_path: str, depth_path: str, apply_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Preprocess RGB and Depth image pair."""
        # Load RGB image
        rgb_image, _ = self.preprocess_image(rgb_path, apply_augmentation=False)
        
        # Load depth image (usually grayscale) using PIL
        depth_pil = Image.open(str(depth_path))
        if depth_pil is None:
            raise ValueError(f"Could not load depth image: {depth_path}")
        
        # Convert to grayscale numpy array
        depth_image = np.array(depth_pil.convert('L'))
        
        # Center crop depth
        depth_image = self.center_crop(depth_image, self.config.CENTER_CROP_SIZE)
        
        # Resize depth using PIL
        depth_pil = Image.fromarray(depth_image)
        depth_pil = depth_pil.resize((self.config.RESIZE_SIZE, self.config.RESIZE_SIZE), Image.LANCZOS)
        depth_image = np.array(depth_pil)
        
        # Convert depth to 3-channel for consistency
        if len(depth_image.shape) == 2:
            depth_image = np.stack([depth_image] * 3, axis=-1)
        
        # Stack RGB and depth (RGBD)
        rgbd_image = np.concatenate([rgb_image, depth_image[:, :, 0:1]], axis=-1)
        
        # Apply augmentations to both
        augmentation_params = {
            'augmentation_applied': False
        }
        
        if apply_augmentation and self.config.AUGMENTATIONS_ENABLED:
            augmentation_params['augmentation_applied'] = True
            # Use only the resize transform for RGBD (other transforms handled by rgb_image)
            # Resize RGBD
            rgbd_pil = Image.fromarray(rgbd_image)
            rgbd_pil = rgbd_pil.resize((self.config.RESIZE_SIZE, self.config.RESIZE_SIZE), Image.LANCZOS)
            rgbd_image = np.array(rgbd_pil)
        else:
            # Resize using PIL
            rgbd_pil = Image.fromarray(rgbd_image)
            rgbd_pil = rgbd_pil.resize((self.config.RESIZE_SIZE, self.config.RESIZE_SIZE), Image.LANCZOS)
            rgbd_image = np.array(rgbd_pil)
        
        return rgb_image, rgbd_image, augmentation_params
    
    def log_preprocessing(self, image_id: str, augmentation_params: Dict):
        """Log preprocessing/augmentation for an image."""
        self.logger.log_augmentation(image_id, augmentation_params)


class BatchPreprocessor:
    """Preprocesses entire batches of images."""
    
    def __init__(self, config: Config = None):
        """Initialize batch preprocessor."""
        self.config = config or Config()
        self.preprocessor = ImagePreprocessor(self.config)
    
    def preprocess_dataset(self, 
                          rgb_dir: str, 
                          depth_dir: str, 
                          output_rgb_dir: str,
                          output_depth_dir: str,
                          output_rgbd_dir: str,
                          csv_file: str = None,
                          apply_augmentation: bool = True):
        """Preprocess entire dataset and save to disk."""
        
        # Create output directories
        Path(output_rgb_dir).mkdir(parents=True, exist_ok=True)
        Path(output_depth_dir).mkdir(parents=True, exist_ok=True)
        Path(output_rgbd_dir).mkdir(parents=True, exist_ok=True)
        
        rgb_files = sorted(Path(rgb_dir).glob("*.jpg")) + sorted(Path(rgb_dir).glob("*.png"))
        
        print(f"Preprocessing {len(rgb_files)} images...")
        
        for idx, rgb_path in enumerate(rgb_files):
            image_id = rgb_path.stem
            depth_path = Path(depth_dir) / f"{image_id}.jpg"
            
            if not depth_path.exists():
                depth_path = Path(depth_dir) / f"{image_id}.png"
            
            if not depth_path.exists():
                print(f"Warning: Depth image not found for {image_id}, skipping...")
                continue
            
            try:
                rgb_img, rgbd_img, aug_params = self.preprocessor.preprocess_rgb_depth_pair(
                    str(rgb_path), str(depth_path), apply_augmentation=apply_augmentation
                )
                
                # Save preprocessed images using PIL
                rgb_pil = Image.fromarray(rgb_img)
                rgb_pil.save(str(Path(output_rgb_dir) / f"{image_id}.png"))
                
                np.save(str(Path(output_rgbd_dir) / f"{image_id}.npy"), rgbd_img.astype(np.uint8))
                
                # Log augmentations
                self.preprocessor.log_preprocessing(image_id, aug_params)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(rgb_files)} images...")
            
            except Exception as e:
                print(f"Error processing {image_id}: {str(e)}")
                continue
        
        print("Preprocessing complete!")


if __name__ == "__main__":
    # Example usage
    config = Config()
    config.print_config()
    
    preprocessor = ImagePreprocessor(config)
    print("Preprocessor initialized successfully!")
