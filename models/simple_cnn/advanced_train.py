"""
Advanced Multi-Modal Model for Lettuce Dry Weight Prediction
Combines RGB images, Depth images, and tabular features for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Workaround for missing _lzma module: directly import only what we need
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Fix SSL certificate verification issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================================
# Advanced Multi-Modal Model
# ============================================================================

class MultiModalLettuceModel(nn.Module):
    """
    Advanced model combining:
    1. RGB images (ResNet50 backbone)
    2. Depth images (ResNet18 backbone)
    3. Tabular features (Height, Diameter, LeafArea, FreshWeightShoot, Variety)
    """
    
    def __init__(self, num_tabular_features=5, dropout=0.3):
        super(MultiModalLettuceModel, self).__init__()
        
        # RGB Image Branch - ResNet50 pretrained
        self.rgb_backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze early layers for transfer learning
        for param in list(self.rgb_backbone.parameters())[:-20]:
            param.requires_grad = False
        rgb_features = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()
        
        # Depth Image Branch - ResNet18
        self.depth_backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify first conv to accept 1 channel (grayscale)
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in list(self.depth_backbone.parameters())[:-15]:
            param.requires_grad = False
        depth_features = self.depth_backbone.fc.in_features
        self.depth_backbone.fc = nn.Identity()
        
        # Tabular Feature Branch
        self.tabular_fc1 = nn.Linear(num_tabular_features, 64)
        self.tabular_fc2 = nn.Linear(64, 128)
        self.tabular_bn1 = nn.BatchNorm1d(64)
        self.tabular_bn2 = nn.BatchNorm1d(128)
        
        # Fusion layers
        total_features = rgb_features + depth_features + 128
        self.fusion_fc1 = nn.Linear(total_features, 512)
        self.fusion_bn1 = nn.BatchNorm1d(512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.fusion_bn2 = nn.BatchNorm1d(256)
        self.fusion_fc3 = nn.Linear(256, 64)
        self.fusion_bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rgb_img, depth_img, tabular):
        # RGB features
        rgb_feat = self.rgb_backbone(rgb_img)
        
        # Depth features
        depth_feat = self.depth_backbone(depth_img)
        
        # Tabular features
        tab_feat = self.relu(self.tabular_bn1(self.tabular_fc1(tabular)))
        tab_feat = self.dropout(tab_feat)
        tab_feat = self.relu(self.tabular_bn2(self.tabular_fc2(tab_feat)))
        
        # Concatenate all features
        combined = torch.cat([rgb_feat, depth_feat, tab_feat], dim=1)
        
        # Fusion network
        x = self.dropout(self.relu(self.fusion_bn1(self.fusion_fc1(combined))))
        x = self.dropout(self.relu(self.fusion_bn2(self.fusion_fc2(x))))
        x = self.dropout(self.relu(self.fusion_bn3(self.fusion_fc3(x))))
        x = self.output(x)
        
        return x


# ============================================================================
# Advanced Dataset with Augmentation
# ============================================================================

class AdvancedLettuceDataset(Dataset):
    """
    Multi-modal dataset loading RGB images, Depth images, and tabular features
    """
    
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
            else:
                print(f"  Warning: Missing images for ID {img_id}, skipping...")
        
        # Keep only valid samples
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
            # Use dummy values for test set without tabular features
            print(f"Warning: Tabular features not found in {labels_file}. Using dummy values.")
            n_samples = len(self.image_ids)
            self.heights = np.zeros(n_samples)
            self.diameters = np.zeros(n_samples)
            self.leaf_areas = np.zeros(n_samples)
            self.fresh_weights = np.zeros(n_samples)
            self.varieties = np.array(['Unknown'] * n_samples)
        
        # Encode categorical variable
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            # Handle unknown varieties
            unique_varieties = np.unique(self.varieties)
            if 'Unknown' in unique_varieties and len(unique_varieties) == 1:
                # All unknown, use single class
                self.varieties_encoded = np.zeros(len(self.varieties), dtype=int)
            else:
                self.varieties_encoded = self.label_encoder.fit_transform(self.varieties)
        else:
            self.label_encoder = label_encoder
            # Handle unknown varieties in test set
            try:
                self.varieties_encoded = self.label_encoder.transform(self.varieties)
            except ValueError:
                # If unknown variety, use the first class
                print(f"Warning: Unknown varieties found. Using default encoding.")
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
        
        # Data augmentation transforms
        if augment:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, f"RGB_{img_id}.png")
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.rgb_transform(rgb_img)
        
        # Load Depth image
        depth_path = os.path.join(self.depth_dir, f"Depth_{img_id}.png")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_img = Image.open(depth_path).convert('L')  # Grayscale
        depth_img = self.depth_transform(depth_img)
        
        # Get tabular features
        tabular = torch.FloatTensor(self.tabular_features[idx])
        
        # Get label
        label = torch.FloatTensor([self.dry_weights[idx]])
        
        return rgb_img, depth_img, tabular, label


# ============================================================================
# Training with Advanced Features
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_advanced_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                        device='cpu', save_dir='./'):
    """Train the advanced multi-modal model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Separate learning rates for pretrained and new layers
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': lr}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training on {device}...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    for epoch in range(epochs):
        # Training
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
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for rgb_imgs, depth_imgs, tabular, labels in val_loader:
                rgb_imgs = rgb_imgs.to(device)
                depth_imgs = depth_imgs.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)
                
                outputs = model(rgb_imgs, depth_imgs, tabular)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, os.path.join(save_dir, 'best_advanced_model.pth'))
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.6f})")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    print(f"\nTraining curves saved to {os.path.join(save_dir, 'training_curves.png')}")
    
    return model


# ============================================================================
# Evaluation and Prediction
# ============================================================================

def evaluate_model(model, test_loader, device='cpu', save_dir='./'):
    """Evaluate the model and generate predictions"""
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for rgb_imgs, depth_imgs, tabular, labels in test_loader:
            rgb_imgs = rgb_imgs.to(device)
            depth_imgs = depth_imgs.to(device)
            tabular = tabular.to(device)
            
            outputs = model(rgb_imgs, depth_imgs, tabular)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(labels.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    print("="*50)
    
    # Plot predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Dry Weight')
    plt.ylabel('Predicted Dry Weight')
    plt.title(f'Predictions vs Actuals (R² = {r2:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'predictions_vs_actuals.png'))
    print(f"\nPredictions plot saved to {os.path.join(save_dir, 'predictions_vs_actuals.png')}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # Smaller batch size for complex model
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2  # 20% for validation
    DEVICE = (
        'mps' if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print("="*60)
    print("ADVANCED MULTI-MODAL LETTUCE DRY WEIGHT PREDICTION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Initial Learning Rate: {LEARNING_RATE}")
    print(f"Validation Split: {VAL_SPLIT*100}%")
    print("="*60)
    
    # Create full training dataset (with augmentation)
    full_train_dataset = AdvancedLettuceDataset(
        rgb_dir='../../datasets/Training/RGBImages',
        depth_dir='../../datasets/Training/DepthImages',
        labels_file='../../datasets/Training/Train.csv',
        image_size=IMAGE_SIZE,
        augment=True
    )
    
    # Split into train and validation
    from torch.utils.data import random_split
    train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset_temp = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset without augmentation using same indices
    val_dataset_no_aug = AdvancedLettuceDataset(
        rgb_dir='../../datasets/Training/RGBImages',
        depth_dir='../../datasets/Training/DepthImages',
        labels_file='../../datasets/Training/Train.csv',
        image_size=IMAGE_SIZE,
        augment=False,  # No augmentation for validation
        scaler=full_train_dataset.scaler,
        label_encoder=full_train_dataset.label_encoder
    )
    
    # Use the same validation indices
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_dataset_temp.indices)
    
    # Save scaler and encoder for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(full_train_dataset.scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(full_train_dataset.label_encoder, f)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(full_train_dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Varieties: {full_train_dataset.label_encoder.classes_}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=False  # MPS doesn't support pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=False  # MPS doesn't support pin_memory
    )
    
    # Create model
    model = MultiModalLettuceModel(num_tabular_features=5, dropout=0.3)
    
    # Train model
    print("\n" + "="*60)
    model = train_advanced_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=DEVICE,
        save_dir='./'
    )
    
    # Load best model for evaluation
    checkpoint = torch.load('best_advanced_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on validation set
    results = evaluate_model(model, val_loader, device=DEVICE, save_dir='./')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: best_advanced_model.pth")
    print(f"Scaler saved to: scaler.pkl")
    print(f"Label encoder saved to: label_encoder.pkl")
    print("="*60)
