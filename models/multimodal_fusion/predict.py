"""
Ensemble Inference Script

Loads all fold checkpoints and generates averaged predictions for test set.

USAGE:
    python predict.py

Output:
    - submission.csv with columns [id, dry_weight]
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from config import Config
from dataset import LettuceDataset, get_val_transform
from model import build_model
from utils import load_checkpoint


def predict_with_model(model, test_loader, device):
    """
    Generate predictions with a single model.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run on
    
    Returns:
        predictions: numpy array of predictions
        ids: list of sample IDs
    """
    model.eval()
    
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            
            outputs = model(rgb, depth)
            predictions = outputs['final_pred'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_ids.extend(batch['id'])
    
    return np.array(all_predictions), all_ids


def ensemble_predict(checkpoint_paths, test_loader, config, device):
    """
    Generate ensemble predictions from multiple fold checkpoints.
    
    Args:
        checkpoint_paths: List of paths to fold checkpoints
        test_loader: DataLoader for test set
        config: Configuration object
        device: Device to run on
    
    Returns:
        final_predictions: numpy array of averaged predictions
        ids: list of sample IDs
    """
    all_fold_predictions = []
    ids = None
    
    for fold, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\nLoading fold {fold + 1} checkpoint: {checkpoint_path}")
        
        # Build model
        model = build_model(config).to(device)
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        print(f"Loaded from epoch {checkpoint['epoch']}, MAE: {checkpoint['metrics']['mae']:.4f}")
        
        # Predict
        predictions, ids = predict_with_model(model, test_loader, device)
        all_fold_predictions.append(predictions)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Average predictions across folds
    all_fold_predictions = np.array(all_fold_predictions)  # (n_folds, n_samples)
    final_predictions = all_fold_predictions.mean(axis=0)
    
    # Also compute std for uncertainty estimation
    prediction_std = all_fold_predictions.std(axis=0)
    
    print(f"\nEnsemble predictions generated for {len(final_predictions)} samples")
    print(f"Mean prediction std: {prediction_std.mean():.4f}")
    
    return final_predictions, ids, prediction_std


def main():
    """Main prediction function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if checkpoints exist
    checkpoint_dir = Config.CHECKPOINT_DIR
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        print("Please train the model first using train.py")
        return
    
    # Find all fold checkpoints
    checkpoint_paths = []
    for fold in range(Config.NUM_FOLDS):
        checkpoint_path = checkpoint_dir / f'fold_{fold}_best.pth'
        if checkpoint_path.exists():
            checkpoint_paths.append(checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found for fold {fold}: {checkpoint_path}")
    
    if len(checkpoint_paths) == 0:
        print("Error: No checkpoints found. Please train the model first.")
        return
    
    print(f"\nFound {len(checkpoint_paths)} fold checkpoints")
    
    # Check if test data exists
    if not Config.RGB_TEST_DIR.exists():
        print(f"Error: Test RGB directory not found: {Config.RGB_TEST_DIR}")
        return
    
    if not Config.DEPTH_TEST_DIR.exists():
        print(f"Error: Test depth directory not found: {Config.DEPTH_TEST_DIR}")
        return
    
    # Create test dataset
    # Get test IDs from sample submission or RGB directory
    if Config.SUBMISSION_PATH.exists():
        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        test_ids = submission_df['id'].tolist()
    else:
        # Get IDs from RGB images
        import glob
        rgb_files = glob.glob(str(Config.RGB_TEST_DIR / '*.*'))
        test_ids = [Path(f).stem.replace('RGB_', '').replace('rgb_', '') for f in rgb_files]
    
    # Create DataFrame for test set
    test_df = pd.DataFrame({'id': test_ids})
    
    print(f"\nTest samples: {len(test_df)}")
    
    # Create test dataset
    test_dataset = LettuceDataset(
        data_df=test_df,
        rgb_dir=Config.RGB_TEST_DIR,
        depth_dir=Config.DEPTH_TEST_DIR,
        mask_dir=None,  # No masks for test set
        transform=get_val_transform(Config),
        is_train=False,
        config=Config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Generate ensemble predictions
    print("\nGenerating ensemble predictions...")
    predictions, ids, prediction_std = ensemble_predict(
        checkpoint_paths, test_loader, Config, device
    )
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': ids,
        'dry_weight': predictions
    })
    
    # Optionally add prediction uncertainty
    submission_df['prediction_std'] = prediction_std
    
    # Save submission
    output_path = Config.OUTPUT_DIR / 'submission.csv'
    submission_df[['id', 'dry_weight']].to_csv(output_path, index=False)
    
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"\nSample predictions:")
    print(submission_df.head(10))
    
    # Save full results with uncertainty
    full_output_path = Config.OUTPUT_DIR / 'submission_with_uncertainty.csv'
    submission_df.to_csv(full_output_path, index=False)
    print(f"\n✓ Full results with uncertainty saved to: {full_output_path}")
    
    print("\nPrediction statistics:")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std: {predictions.std():.4f}")
    print(f"Min: {predictions.min():.4f}")
    print(f"Max: {predictions.max():.4f}")


if __name__ == '__main__':
    main()
