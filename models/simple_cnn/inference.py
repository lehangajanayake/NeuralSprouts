"""
Inference Script for Advanced Lettuce Dry Weight Prediction
Load a trained model and make predictions on new data
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import os
import argparse


# Import model class from training script
from advanced_train import MultiModalLettuceModel


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    model = MultiModalLettuceModel(num_tabular_features=5, dropout=0.3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"  Trained for {checkpoint['epoch']+1} epochs")
    print(f"  Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"  Best validation MAE: {checkpoint['val_mae']:.4f}")
    
    return model


def load_preprocessors(scaler_path='scaler.pkl', encoder_path='label_encoder.pkl'):
    """Load scaler and label encoder"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"✓ Loaded preprocessors")
    print(f"  Varieties: {label_encoder.classes_}")
    
    return scaler, label_encoder


def preprocess_image(image_path, image_size=224, is_rgb=True):
    """Preprocess a single image"""
    img = Image.open(image_path)
    
    if is_rgb:
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        img = img.convert('L')  # Grayscale for depth
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    return transform(img).unsqueeze(0)  # Add batch dimension


def prepare_tabular_features(height, diameter, leaf_area, fresh_weight, variety, 
                            scaler, label_encoder):
    """Prepare and scale tabular features"""
    variety_encoded = label_encoder.transform([variety])[0]
    features = np.array([[height, diameter, leaf_area, fresh_weight, variety_encoded]])
    scaled_features = scaler.transform(features)
    return torch.FloatTensor(scaled_features)


def predict_single(model, rgb_path, depth_path, height, diameter, leaf_area, 
                   fresh_weight, variety, scaler, label_encoder, device='cpu'):
    """Make prediction for a single sample"""
    
    # Preprocess images
    rgb_img = preprocess_image(rgb_path, is_rgb=True).to(device)
    depth_img = preprocess_image(depth_path, is_rgb=False).to(device)
    
    # Prepare tabular features
    tabular = prepare_tabular_features(
        height, diameter, leaf_area, fresh_weight, variety,
        scaler, label_encoder
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(rgb_img, depth_img, tabular)
    
    return prediction.item()


def predict_batch(model, csv_path, rgb_dir, depth_dir, scaler, label_encoder, 
                 device='cpu', output_path='predictions.csv'):
    """Make predictions for a batch of samples from CSV"""
    
    df = pd.read_csv(csv_path)
    predictions = []
    
    print(f"\nProcessing {len(df)} samples...")
    
    for idx, row in df.iterrows():
        img_id = row['image_id']
        rgb_path = os.path.join(rgb_dir, f"RGB_{img_id}.png")
        depth_path = os.path.join(depth_dir, f"Depth_{img_id}.png")
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"Warning: Missing images for ID {img_id}, skipping...")
            predictions.append(None)
            continue
        
        pred = predict_single(
            model, rgb_path, depth_path,
            row['Height'], row['Diameter'], row['LeafArea'],
            row['FreshWeightShoot'], row['Variety'],
            scaler, label_encoder, device
        )
        predictions.append(pred)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    # Add predictions to dataframe
    df['PredictedDryWeight'] = predictions
    
    # If actual values exist, calculate metrics
    if 'DryWeightShoot' in df.columns:
        df['AbsoluteError'] = abs(df['DryWeightShoot'] - df['PredictedDryWeight'])
        mae = df['AbsoluteError'].mean()
        rmse = np.sqrt(((df['DryWeightShoot'] - df['PredictedDryWeight']) ** 2).mean())
        
        print(f"\n✓ Predictions complete!")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # Save predictions
    df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, default='best_advanced_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with samples')
    parser.add_argument('--rgb_dir', type=str, required=True,
                       help='Directory containing RGB images')
    parser.add_argument('--depth_dir', type=str, required=True,
                       help='Directory containing Depth images')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--scaler', type=str, default='scaler.pkl',
                       help='Path to scaler pickle file')
    parser.add_argument('--encoder', type=str, default='label_encoder.pkl',
                       help='Path to label encoder pickle file')
    
    args = parser.parse_args()
    
    # Device selection
    DEVICE = (
        'mps' if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print("="*60)
    print("ADVANCED LETTUCE DRY WEIGHT PREDICTION - INFERENCE")
    print("="*60)
    print(f"Device: {DEVICE}\n")
    
    # Load model and preprocessors
    model = load_model(args.model, device=DEVICE)
    scaler, label_encoder = load_preprocessors(args.scaler, args.encoder)
    
    # Make predictions
    predictions_df = predict_batch(
        model, args.csv, args.rgb_dir, args.depth_dir,
        scaler, label_encoder, DEVICE, args.output
    )
    
    print("="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    # Example usage without command line:
    # Uncomment and modify the following for quick testing
    
    # DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # model = load_model('best_advanced_model.pth', device=DEVICE)
    # scaler, label_encoder = load_preprocessors()
    # 
    # # Single prediction example
    # prediction = predict_single(
    #     model,
    #     rgb_path='../../datasets/Test/RGBImages/RGB_100.png',
    #     depth_path='../../datasets/Test/DepthImages/Depth_100.png',
    #     height=5.1, diameter=16.1, leaf_area=87.6,
    #     fresh_weight=3.2, variety='Aphylion',
    #     scaler=scaler, label_encoder=label_encoder,
    #     device=DEVICE
    # )
    # print(f"Predicted dry weight: {prediction:.4f}")
    
    # Command line interface
    main()
