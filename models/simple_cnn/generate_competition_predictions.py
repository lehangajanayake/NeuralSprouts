"""
Generate Competition Predictions with Test-Time Augmentation
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import os
from competition_train import CompetitionLettuceModel
import torchvision.transforms as transforms

def predict_test_set(model_path, scaler_path, label_encoder_path,
                     test_csv, rgb_dir, depth_dir, output_csv,
                     device='cpu', n_tta=10):
    """
    Generate predictions for test set with Test-Time Augmentation
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to saved scaler
        label_encoder_path: Path to saved label encoder
        test_csv: Path to test CSV
        rgb_dir: Path to test RGB images
        depth_dir: Path to test depth images
        output_csv: Path to save predictions
        device: Device to run on
        n_tta: Number of test-time augmentation iterations
    """
    
    print("="*70)
    print("🎯 GENERATING COMPETITION PREDICTIONS")
    print("="*70)
    
    # Load model
    print("\n📦 Loading model...")
    model = CompetitionLettuceModel(num_tabular_features=5, dropout=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (trained MAE: {checkpoint['train_mae']:.4f})")
    
    # Load scaler and encoder
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Scaler and encoder loaded")
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    image_ids = test_df['image_id'].values
    print(f"✓ Test set loaded: {len(image_ids)} samples")
    
    print(f"\n🔮 Generating predictions with {n_tta}x Test-Time Augmentation...")
    print("   (This improves robustness by averaging multiple augmented predictions)")
    
    all_predictions = []
    
    for idx, img_id in enumerate(image_ids):
        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx + 1}/{len(image_ids)} samples")
        
        img_predictions = []
        
        rgb_path = os.path.join(rgb_dir, f"RGB_{img_id}.png")
        depth_path = os.path.join(depth_dir, f"Depth_{img_id}.png")
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"   Warning: Missing images for ID {img_id}, using mean prediction")
            all_predictions.append(5.5)  # Use approximate mean
            continue
        
        # Multiple predictions with TTA
        for _ in range(n_tta):
            # Random augmentations for TTA
            rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            depth_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb_img = rgb_transform(rgb_img).unsqueeze(0).to(device)
            
            depth_img = Image.open(depth_path).convert('L')
            depth_img = depth_transform(depth_img).unsqueeze(0).to(device)
            
            # Dummy tabular features (zeros since test set doesn't have them)
            tabular = torch.zeros(1, 5).to(device)
            
            with torch.no_grad():
                output = model(rgb_img, depth_img, tabular)
                img_predictions.append(output.item())
        
        # Average all TTA predictions
        final_pred = np.mean(img_predictions)
        all_predictions.append(final_pred)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'image_id': image_ids,
        'DryWeightShoot': all_predictions
    })
    
    # Save predictions
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Predictions saved to: {output_csv}")
    print(f"\n📊 Prediction Statistics:")
    print(f"   Min:  {output_df['DryWeightShoot'].min():.4f}g")
    print(f"   Max:  {output_df['DryWeightShoot'].max():.4f}g")
    print(f"   Mean: {output_df['DryWeightShoot'].mean():.4f}g")
    print(f"   Std:  {output_df['DryWeightShoot'].std():.4f}g")
    
    print("\n" + "="*70)
    print("🎉 PREDICTIONS COMPLETE!")
    print("="*70)
    print(f"📤 Submit '{output_csv}' to the competition!")
    print("="*70)
    
    return output_df


if __name__ == "__main__":
    DEVICE = (
        'mps' if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    predictions = predict_test_set(
        model_path='best_competition_model.pth',
        scaler_path='competition_scaler.pkl',
        label_encoder_path='competition_label_encoder.pkl',
        test_csv='../../datasets/Test/Test.csv',
        rgb_dir='../../datasets/Test/RGBImages',
        depth_dir='../../datasets/Test/DepthImages',
        output_csv='competition_predictions.csv',
        device=DEVICE,
        n_tta=10  # 10x TTA for robust predictions
    )
    
    print("\n💡 Tip: You can also create an ensemble by training multiple models")
    print("   and averaging their predictions for even better results!")
