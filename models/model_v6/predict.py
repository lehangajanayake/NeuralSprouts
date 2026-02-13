"""
Prediction module for Model_v6.
Handles inference and prediction on test set.
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import torch

from config import Config
from model import create_model
from dataloader import create_dataloader


class Predictor:
    """Predictor class for Model_v6."""
    
    def __init__(self, model_path: str, config: Config = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            config: Configuration object
        """
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = create_model(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict_single(self, rgb_image: torch.Tensor, rgbd_image: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Make prediction on single image pair.
        
        Returns:
            Tuple of (predicted_value, rgb_features, rgbd_features)
        """
        with torch.no_grad():
            rgb_image = rgb_image.unsqueeze(0).to(self.device)
            rgbd_image = rgbd_image.unsqueeze(0).to(self.device)
            
            # Get features for visualization
            rgb_features, rgbd_features = self.model.get_branch_features(rgb_image, rgbd_image)
            
            # Get prediction
            prediction = self.model(rgb_image, rgbd_image)
            
        return prediction.item(), rgb_features.cpu().numpy(), rgbd_features.cpu().numpy()
    
    def predict_batch(self, test_loader) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Make predictions on batch of data.
        
        Returns:
            Tuple of (predictions, ground_truth, image_ids, errors)
        """
        predictions_list = []
        ground_truth_list = []
        image_ids_list = []
        errors_list = []
        
        with torch.no_grad():
            for rgb_images, rgbd_images, dry_weights, image_ids in test_loader:
                rgb_images = rgb_images.to(self.device)
                rgbd_images = rgbd_images.to(self.device)
                dry_weights = dry_weights.to(self.device)
                
                predictions = self.model(rgb_images, rgbd_images)
                
                predictions_list.append(predictions.cpu().numpy())
                ground_truth_list.append(dry_weights.cpu().numpy())
                image_ids_list.extend(image_ids)
                
                # Calculate errors
                errors = (predictions.cpu().numpy() - dry_weights.cpu().numpy()).flatten()
                errors_list.extend(errors)
        
        predictions = np.concatenate(predictions_list)
        ground_truth = np.concatenate(ground_truth_list)
        
        return predictions, ground_truth, image_ids_list, errors_list
    
    def predict_with_attention(self, test_loader):
        """
        Make predictions with attention maps.
        
        Returns:
            Dictionary with predictions, ground truth, image IDs, and attention maps
        """
        results = {
            'image_ids': [],
            'predictions': [],
            'ground_truth': [],
            'errors': [],
            'rgb_attention': [],
            'rgbd_attention': []
        }
        
        with torch.no_grad():
            for rgb_images, rgbd_images, dry_weights, image_ids in test_loader:
                rgb_images = rgb_images.to(self.device)
                rgbd_images = rgbd_images.to(self.device)
                dry_weights = dry_weights.to(self.device)
                
                # Get predictions
                predictions = self.model(rgb_images, rgbd_images)
                
                # Get attention maps
                rgb_attention, rgbd_attention = self.model.get_attention_maps(rgb_images, rgbd_images)
                
                # Store results
                results['image_ids'].extend(image_ids)
                results['predictions'].extend(predictions.cpu().numpy().flatten())
                results['ground_truth'].extend(dry_weights.cpu().numpy().flatten())
                results['errors'].extend((predictions.cpu().numpy() - dry_weights.cpu().numpy()).flatten())
                results['rgb_attention'].append(rgb_attention.cpu().numpy())
                results['rgbd_attention'].append(rgbd_attention.cpu().numpy())
        
        # Concatenate attention maps
        results['rgb_attention'] = np.concatenate(results['rgb_attention'], axis=0)
        results['rgbd_attention'] = np.concatenate(results['rgbd_attention'], axis=0)
        
        return results


def predict_on_test_set(model_path: str, 
                        config: Config = None,
                        output_csv: str = None) -> pd.DataFrame:
    """
    Make predictions on entire test set and save to CSV.
    
    Args:
        model_path: Path to saved model
        config: Configuration object
        output_csv: Path to save predictions CSV
    
    Returns:
        DataFrame with predictions and ground truth
    """
    config = config or Config()
    
    # Create predictor
    predictor = Predictor(model_path, config)
    
    # Create test dataloader
    test_loader = create_dataloader(
        csv_file=config.TEST_CSV,
        rgb_dir=config.TEST_RGB_DIR,
        depth_dir=config.TEST_DEPTH_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        include_target=False
    )
    
    # Make predictions
    predictions, ground_truth, image_ids, errors = predictor.predict_batch(test_loader)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_dry_weight': predictions.flatten(),
        'actual_dry_weight': ground_truth.flatten(),
        'error': errors,
        'abs_error': np.abs(errors)
    })
    
    # Save to CSV if specified
    if output_csv:
        output_dir = Path(output_csv).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    
    return results_df


if __name__ == "__main__":
    # Example usage
    config = Config()
    
    model_path = str(Path(config.EXPERIMENT_DIR) / "best_model.pth")
    output_csv = str(Path(config.EXPERIMENT_DIR) / "predictions.csv")
    
    try:
        results_df = predict_on_test_set(model_path, config, output_csv)
        print(f"\nPrediction Summary:")
        print(f"Mean Absolute Error: {results_df['abs_error'].mean():.4f}")
        print(f"Mean Squared Error: {(results_df['error'] ** 2).mean():.4f}")
        print(f"Root Mean Squared Error: {np.sqrt((results_df['error'] ** 2).mean()):.4f}")
        print(f"\nFirst 10 predictions:")
        print(results_df.head(10))
    except Exception as e:
        print(f"Error during prediction: {e}")
