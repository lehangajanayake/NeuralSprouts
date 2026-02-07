"""
Visualization module for Model_v6.
Handles visualization of model behavior, attention maps, and predictions.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns

from config import Config
from model import create_model
from predict import Predictor
from dataloader import create_dataloader


class ModelVisualizer:
    """Visualizer for Model_v6."""
    
    def __init__(self, model_path: str, config: Config = None):
        """Initialize visualizer."""
        self.config = config or Config()
        self.predictor = Predictor(model_path, self.config)
        self.output_dir = Path(self.config.EXPERIMENT_DIR) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_augmentation_effects(self, 
                                       original_image_path: str,
                                       augmentation_log_path: str,
                                       image_id: str,
                                       output_path: str = None):
        """
        Visualize augmentation effects on a single image.
        
        Args:
            original_image_path: Path to original image
            augmentation_log_path: Path to augmentation log CSV
            image_id: ID of image to visualize
            output_path: Path to save visualization
        """
        # Load original image using PIL
        original = Image.open(original_image_path)
        if original.mode != 'RGB':
            original = original.convert('RGB')
        original = np.array(original)
        
        # Load augmentation info
        aug_log = pd.read_csv(augmentation_log_path)
        aug_info = aug_log[aug_log['image_id'] == image_id]
        
        if aug_info.empty:
            print(f"No augmentation log found for {image_id}")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original
        axes[0].imshow(original)
        axes[0].set_title(f"Original Image: {image_id}")
        axes[0].axis('off')
        
        # Plot info
        axes[1].axis('off')
        info_text = f"Augmentations Applied:\n"
        info_text += f"- Horizontal Flip: {aug_info['horizontal_flip'].values[0]}\n"
        info_text += f"- Vertical Flip: {aug_info['vertical_flip'].values[0]}\n"
        info_text += f"- Rotation: {aug_info['rotation_angle'].values[0]}°\n"
        info_text += f"- Horizontal Shift: {aug_info['horizontal_shift'].values[0]}\n"
        info_text += f"- Vertical Shift: {aug_info['vertical_shift'].values[0]}\n"
        info_text += f"- Augmentation Applied: {aug_info['augmentation_applied'].values[0]}"
        
        axes[1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                     family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.close()
    
    def visualize_attention_maps(self,
                                 rgb_image: np.ndarray,
                                 rgbd_image: np.ndarray,
                                 image_id: str,
                                 output_path: str = None):
        """
        Visualize attention maps for RGB and RGBD branches.
        
        Args:
            rgb_image: RGB image as numpy array
            rgbd_image: RGBD image as numpy array
            image_id: Image ID for labeling
            output_path: Path to save visualization
        """
        import torch
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_image).unsqueeze(0).float()
        rgbd_tensor = torch.from_numpy(rgbd_image).unsqueeze(0).float()
        
        # Get attention maps
        rgb_attention, rgbd_attention = self.predictor.model.get_attention_maps(rgb_tensor, rgbd_tensor)
        rgb_attention = rgb_attention[0].cpu().numpy()
        rgbd_attention = rgbd_attention[0].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # RGB branch
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis('off')
        
        im = axes[0, 1].imshow(rgb_attention, cmap='hot')
        axes[0, 1].set_title("RGB Attention Map")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Overlay attention on RGB
        overlay = self._overlay_attention(rgb_image, rgb_attention)
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title("RGB with Attention Overlay")
        axes[0, 2].axis('off')
        
        # RGBD branch
        rgbd_display = rgbd_image[:, :, :3]  # Display only RGB channels
        axes[1, 0].imshow(rgbd_display)
        axes[1, 0].set_title("RGBD Image (RGB channels)")
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(rgbd_attention, cmap='hot')
        axes[1, 1].set_title("RGBD Attention Map")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Overlay attention on RGBD
        overlay = self._overlay_attention(rgbd_display, rgbd_attention)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("RGBD with Attention Overlay")
        axes[1, 2].axis('off')
        
        fig.suptitle(f"Attention Maps: {image_id}", fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Attention visualization saved to {output_path}")
        
        plt.close()
    
    def _overlay_attention(self, image: np.ndarray, attention: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Overlay attention map on image."""
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Resize attention to match image using PIL
        attention_pil = Image.fromarray((attention * 255).astype(np.uint8))
        attention_pil = attention_pil.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
        attention_resized = np.array(attention_pil) / 255.0
        
        # Create heatmap
        heatmap = cm.get_cmap('jet')(attention_resized)[:, :, :3]
        
        # Blend
        overlay = (alpha * image + (1 - alpha) * heatmap * 255).astype(np.uint8)
        return overlay
    
    def visualize_predictions(self, 
                             predictions_df: pd.DataFrame,
                             output_path: str = None):
        """
        Visualize predictions vs ground truth.
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(predictions_df['actual_dry_weight'], predictions_df['predicted_dry_weight'], alpha=0.6)
        axes[0].plot([predictions_df['actual_dry_weight'].min(), predictions_df['actual_dry_weight'].max()],
                     [predictions_df['actual_dry_weight'].min(), predictions_df['actual_dry_weight'].max()],
                     'r--', label='Perfect Prediction')
        axes[0].set_xlabel('Actual Dry Weight')
        axes[0].set_ylabel('Predicted Dry Weight')
        axes[0].set_title('Predictions vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1].hist(predictions_df['error'], bins=30, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Absolute error by prediction
        axes[2].scatter(predictions_df['actual_dry_weight'], predictions_df['abs_error'], alpha=0.6)
        axes[2].set_xlabel('Actual Dry Weight')
        axes[2].set_ylabel('Absolute Error')
        axes[2].set_title('Absolute Error by Actual Value')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Prediction visualization saved to {output_path}")
        
        plt.close()
    
    def visualize_top_errors(self, 
                            predictions_df: pd.DataFrame,
                            rgb_dir: str,
                            rgbd_dir: str,
                            num_samples: int = 5,
                            output_dir: str = None):
        """
        Visualize predictions with largest errors.
        
        Args:
            predictions_df: DataFrame with predictions
            rgb_dir: Directory with RGB images
            rgbd_dir: Directory with RGBD images
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = str(self.output_dir / "top_errors")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get top error samples
        top_errors = predictions_df.nlargest(num_samples, 'abs_error')
        
        for idx, (_, row) in enumerate(top_errors.iterrows()):
            image_id = row['image_id']
            
            try:
                rgb_path = Path(rgb_dir) / f"{image_id}.png"
                rgb_image = Image.open(str(rgb_path))
                if rgb_image.mode != 'RGB':
                    rgb_image = rgb_image.convert('RGB')
                rgb_image = np.array(rgb_image)
                
                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Image
                axes[0].imshow(rgb_image)
                axes[0].set_title(f"Image: {image_id}")
                axes[0].axis('off')
                
                # Stats
                axes[1].axis('off')
                stats_text = f"Prediction Analysis:\n"
                stats_text += f"Actual: {row['actual_dry_weight']:.4f}\n"
                stats_text += f"Predicted: {row['predicted_dry_weight']:.4f}\n"
                stats_text += f"Error: {row['error']:.4f}\n"
                stats_text += f"Abs Error: {row['abs_error']:.4f}"
                
                axes[1].text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center',
                           family='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                plt.tight_layout()
                
                output_path = Path(output_dir) / f"error_{idx + 1}_{image_id}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            except Exception as e:
                print(f"Error visualizing {image_id}: {e}")


def create_comprehensive_visualization(model_path: str,
                                      predictions_csv: str,
                                      config: Config = None,
                                      output_dir: str = None):
    """
    Create comprehensive visualization suite.
    
    Args:
        model_path: Path to model checkpoint
        predictions_csv: Path to predictions CSV
        config: Configuration object
        output_dir: Directory to save visualizations
    """
    config = config or Config()
    if output_dir is None:
        output_dir = str(Path(config.EXPERIMENT_DIR) / "visualizations")
    
    visualizer = ModelVisualizer(model_path, config)
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_csv)
    
    # Visualize predictions
    viz_path = Path(output_dir) / "predictions_analysis.png"
    visualizer.visualize_predictions(predictions_df, str(viz_path))
    
    # Visualize top errors
    visualizer.visualize_top_errors(predictions_df, config.TEST_RGB_DIR, 
                                    config.TEST_DEPTH_DIR, num_samples=5,
                                    output_dir=str(Path(output_dir) / "top_errors"))
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    config = Config()
    
    model_path = str(Path(config.EXPERIMENT_DIR) / "best_model.pth")
    predictions_csv = str(Path(config.EXPERIMENT_DIR) / "predictions.csv")
    
    try:
        create_comprehensive_visualization(model_path, predictions_csv, config)
    except Exception as e:
        print(f"Error during visualization: {e}")
