"""
Competition-Optimized Multi-Modal Model for Lettuce Dry Weight Prediction
Maximized for lowest MAE on competition leaderboard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

# Fix SSL certificate verification issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================================
# Competition-Optimized Model (More Capacity)
# ============================================================================

class CompetitionLettuceModel(nn.Module):
    """
    Enhanced model with more trainable parameters for competition
    """
    
    def __init__(self, num_tabular_features=5, dropout=0.2):
        super(CompetitionLettuceModel, self).__init__()
        
        # RGB Image Branch - ResNet50 pretrained (unfreeze more layers)
        self.rgb_backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Only freeze first few layers (unfreeze more for better fine-tuning)
        for param in list(self.rgb_backbone.parameters())[:-40]:  # Unfreeze more layers
            param.requires_grad = False
        rgb_features = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()
        
        # Depth Image Branch - ResNet18
        self.depth_backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Unfreeze more depth layers
        for param in list(self.depth_backbone.parameters())[:-25]:
            param.requires_grad = False
        depth_features = self.depth_backbone.fc.in_features
        self.depth_backbone.fc = nn.Identity()
        
        # Enhanced Tabular Feature Branch
        self.tabular_fc1 = nn.Linear(num_tabular_features, 128)
        self.tabular_fc2 = nn.Linear(128, 256)
        self.tabular_fc3 = nn.Linear(256, 256)
        self.tabular_bn1 = nn.BatchNorm1d(128)
        self.tabular_bn2 = nn.BatchNorm1d(256)
        self.tabular_bn3 = nn.BatchNorm1d(256)
        
        # Enhanced Fusion layers with residual connections
        total_features = rgb_features + depth_features + 256
        self.fusion_fc1 = nn.Linear(total_features, 1024)
        self.fusion_bn1 = nn.BatchNorm1d(1024)
        self.fusion_fc2 = nn.Linear(1024, 512)
        self.fusion_bn2 = nn.BatchNorm1d(512)
        self.fusion_fc3 = nn.Linear(512, 256)
        self.fusion_bn3 = nn.BatchNorm1d(256)
        self.fusion_fc4 = nn.Linear(256, 128)
        self.fusion_bn4 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rgb_img, depth_img, tabular):
        # RGB features
        rgb_feat = self.rgb_backbone(rgb_img)
        
        # Depth features
        depth_feat = self.depth_backbone(depth_img)
        
        # Enhanced tabular features
        tab_feat = self.dropout(self.relu(self.tabular_bn1(self.tabular_fc1(tabular))))
        tab_feat = self.dropout(self.relu(self.tabular_bn2(self.tabular_fc2(tab_feat))))
        tab_feat = self.relu(self.tabular_bn3(self.tabular_fc3(tab_feat)))
        
        # Concatenate all features
        combined = torch.cat([rgb_feat, depth_feat, tab_feat], dim=1)
        
        # Deep fusion network
        x = self.dropout(self.relu(self.fusion_bn1(self.fusion_fc1(combined))))
        x = self.dropout(self.relu(self.fusion_bn2(self.fusion_fc2(x))))
        x = self.dropout(self.relu(self.fusion_bn3(self.fusion_fc3(x))))
        x = self.dropout(self.relu(self.fusion_bn4(self.fusion_fc4(x))))
        x = self.output(x)
        
        return x


# ============================================================================
# Dataset with Enhanced Augmentation
# ============================================================================

class CompetitionLettuceDataset(Dataset):
    """Enhanced dataset with better augmentation"""
    
    def __init__(self, rgb_dir, depth_dir, labels_file, image_size=224, 
                 augment=False, scaler=None, label_encoder=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.augment = augment
        
        # Load data
        self.df = pd.read_csv(labels_file)
        
        # Filter out samples with missing images
        valid_indices = []
        for idx, img_id in enumerate(self.df['image_id'].values):
            rgb_path = os.path.join(rgb_dir, f"RGB_{img_id}.png")
            depth_path = os.path.join(depth_dir, f"Depth_{img_id}.png")
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # Extract features
        self.image_ids = self.df['image_id'].values
        self.dry_weights = self.df['DryWeightShoot'].values
        
        # Check if tabular features are available
        self.has_tabular = all(col in self.df.columns for col in ['Height', 'Diameter', 'LeafArea', 'FreshWeightShoot', 'Variety'])
        
        if self.has_tabular:
            self.heights = self.df['Height'].values
            self.diameters = self.df['Diameter'].values
            self.leaf_areas = self.df['LeafArea'].values
            self.fresh_weights = self.df['FreshWeightShoot'].values
            self.varieties = self.df['Variety'].values
        else:
            n_samples = len(self.image_ids)
            self.heights = np.zeros(n_samples)
            self.diameters = np.zeros(n_samples)
            self.leaf_areas = np.zeros(n_samples)
            self.fresh_weights = np.zeros(n_samples)
            self.varieties = np.array(['Unknown'] * n_samples)
        
        # Encode categorical variable
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            unique_varieties = np.unique(self.varieties)
            if 'Unknown' in unique_varieties and len(unique_varieties) == 1:
                self.varieties_encoded = np.zeros(len(self.varieties), dtype=int)
            else:
                self.varieties_encoded = self.label_encoder.fit_transform(self.varieties)
        else:
            self.label_encoder = label_encoder
            try:
                self.varieties_encoded = self.label_encoder.transform(self.varieties)
            except ValueError:
                self.varieties_encoded = np.zeros(len(self.varieties), dtype=int)
        
        # Standardize tabular features
        tabular_features = np.column_stack([
            self.heights, self.diameters, self.leaf_areas, 
            self.fresh_weights, self.varieties_encoded
        ])
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.tabular_features = self.scaler.fit_transform(tabular_features)
        else:
            self.scaler = scaler
            self.tabular_features = self.scaler.transform(tabular_features)
        
        # Enhanced augmentation for training
        if augment:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        rgb_path = os.path.join(self.rgb_dir, f"RGB_{img_id}.png")
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.rgb_transform(rgb_img)
        
        depth_path = os.path.join(self.depth_dir, f"Depth_{img_id}.png")
        depth_img = Image.open(depth_path).convert('L')
        depth_img = self.depth_transform(depth_img)
        
        tabular = torch.FloatTensor(self.tabular_features[idx])
        label = torch.FloatTensor([self.dry_weights[idx]])
        
        return rgb_img, depth_img, tabular, label


# ============================================================================
# MAE Loss Function (optimize what competition measures)
# ============================================================================

class MAELoss(nn.Module):
    """Mean Absolute Error Loss - directly optimize competition metric"""
    def __init__(self):
        super(MAELoss, self).__init__()
    
    def forward(self, predictions, targets):
        return torch.mean(torch.abs(predictions - targets))


# ============================================================================
# Training Function with Advanced Features
# ============================================================================

def train_competition_model(model, train_loader, epochs=100, lr=0.001, device='cpu', save_dir='./'):
    """Train model optimized for competition"""
    
    model = model.to(device)
    
    # Use MAE loss (what competition measures)
    criterion = MAELoss()
    
    # Separate learning rates
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': lr * 0.1},
        {'params': new_params, 'lr': lr}
    ], weight_decay=0.01)
    
    # OneCycleLR for better convergence
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[lr * 0.1, lr],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_train_mae = float('inf')
    train_losses = []
    train_maes = []
    
    print(f"\n🚀 COMPETITION MODE: Training on 100% of data")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for rgb_imgs, depth_imgs, tabular, labels in train_loader:
            rgb_imgs = rgb_imgs.to(device)
            depth_imgs = depth_imgs.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_imgs, depth_imgs, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model based on training MAE
        if train_mae < best_train_mae:
            best_train_mae = train_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_mae': train_mae
            }, os.path.join(save_dir, 'best_competition_model.pth'))
            print(f"  ✓ Saved best model (Train MAE: {train_mae:.4f})")
        
        print()
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_maes, label='Train MAE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training MAE over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'competition_training_curve.png'))
    print(f"Training curve saved to {os.path.join(save_dir, 'competition_training_curve.png')}")
    
    return model


# ============================================================================
# Test-Time Augmentation for Better Predictions
# ============================================================================

def predict_with_tta(model, rgb_dir, depth_dir, image_ids, scaler, label_encoder, 
                     device='cpu', n_tta=5):
    """Predict with Test-Time Augmentation for more robust predictions"""
    
    model.eval()
    all_predictions = []
    
    for img_id in image_ids:
        img_predictions = []
        
        for _ in range(n_tta):
            # Load images
            rgb_path = os.path.join(rgb_dir, f"RGB_{img_id}.png")
            depth_path = os.path.join(depth_dir, f"Depth_{img_id}.png")
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            
            # Apply slight augmentation for TTA
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
            
            # Dummy tabular features (zeros for test set)
            tabular = torch.zeros(1, 5).to(device)
            
            with torch.no_grad():
                output = model(rgb_img, depth_img, tabular)
                img_predictions.append(output.item())
        
        # Average predictions
        if img_predictions:
            all_predictions.append(np.mean(img_predictions))
        else:
            all_predictions.append(0.0)
    
    return np.array(all_predictions)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Competition Configuration
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50  # Fewer epochs but all data
    LEARNING_RATE = 0.001
    DEVICE = (
        'mps' if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print("="*70)
    print("🏆 COMPETITION-OPTIMIZED LETTUCE DRY WEIGHT PREDICTION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Strategy: Train on 100% data, optimize MAE directly")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Initial Learning Rate: {LEARNING_RATE}")
    print("="*70)
    
    # Create training dataset with ALL data
    train_dataset = CompetitionLettuceDataset(
        rgb_dir='../../datasets/Training/RGBImages',
        depth_dir='../../datasets/Training/DepthImages',
        labels_file='../../datasets/Training/Train.csv',
        image_size=IMAGE_SIZE,
        augment=True  # Strong augmentation
    )
    
    # Save scaler and encoder
    with open('competition_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    with open('competition_label_encoder.pkl', 'wb') as f:
        pickle.dump(train_dataset.label_encoder, f)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training samples: {len(train_dataset)} (100% of data)")
    print(f"  Varieties: {train_dataset.label_encoder.classes_}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create enhanced model
    model = CompetitionLettuceModel(num_tabular_features=5, dropout=0.2)
    
    # Load weights from previous best model if available
    if os.path.exists('best_advanced_model.pth'):
        print("\n🔥 Loading weights from previous best model...")
        try:
            prev_checkpoint = torch.load('best_advanced_model.pth', map_location=DEVICE)
            prev_state_dict = prev_checkpoint['model_state_dict']
            
            # Load compatible weights (some layers might not match due to architecture changes)
            model_state_dict = model.state_dict()
            compatible_weights = {}
            
            for name, param in prev_state_dict.items():
                if name in model_state_dict and model_state_dict[name].shape == param.shape:
                    compatible_weights[name] = param
            
            model.load_state_dict(compatible_weights, strict=False)
            print(f"✓ Loaded {len(compatible_weights)}/{len(model_state_dict)} compatible layers")
            print(f"✓ Previous model MAE: {prev_checkpoint.get('val_mae', 'N/A')}")
            print("✓ Starting from warm-start (better than random initialization!)")
        except Exception as e:
            print(f"⚠️  Could not load previous weights: {e}")
            print("Starting from scratch with pretrained ImageNet weights...")
    else:
        print("\n💡 No previous model found, starting from pretrained ImageNet weights...")
    
    # Train model
    print("\n" + "="*70)
    model = train_competition_model(
        model, 
        train_loader,
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=DEVICE,
        save_dir='./'
    )
    
    # Load best model
    checkpoint = torch.load('best_competition_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"✓ Best Training MAE: {checkpoint['train_mae']:.4f}")
    
    print("\n" + "="*70)
    print("🎯 COMPETITION MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: best_competition_model.pth")
    print(f"Scaler saved to: competition_scaler.pkl")
    print(f"Label encoder saved to: competition_label_encoder.pkl")
    print("\n💡 Next step: Generate predictions for test set!")
    print("   Run: python3 generate_competition_predictions.py")
    print("="*70)
