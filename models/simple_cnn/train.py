"""
Simple CNN for Lettuce Dry Weight Prediction
A standalone PyTorch implementation with minimal boilerplate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os


# ============================================================================
# Model Definition
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for regression"""
    
    def __init__(self, input_channels=3, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # FC layers (assuming 224x224 input -> 28x28 after 3 poolings)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: dry weight
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================================
# Dataset (customize for your data)
# ============================================================================

class LettuceDataset(Dataset):
    """
    Dataset for lettuce images and dry weight labels.
    
    Customize the __init__ and __getitem__ methods based on your data format.
    """
    
    def __init__(self, image_dir, labels_file, image_size=224):
        """
        Args:
            image_dir: Directory with images
            labels_file: CSV or file with labels (format: image_name, dry_weight)
            image_size: Size to resize images to
        """
        self.image_dir = image_dir
        self.image_size = image_size
        
        # TODO: Load your image paths and labels
        # Example:
        # import pandas as pd
        # df = pd.read_csv(labels_file)
        # self.image_paths = df['image_name'].tolist()
        # self.labels = df['dry_weight'].values
        
        self.image_paths = []
        self.labels = []
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Get label
        label = torch.FloatTensor([self.labels[idx]])
        
        return image, label


# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> Saved best model")
    
    return model


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Create datasets
    # TODO: Update paths to your actual data
    # train_dataset = LettuceDataset('../../datasets/train', '../../datasets/train_labels.csv')
    # val_dataset = LettuceDataset('../../datasets/val', '../../datasets/val_labels.csv')
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = SimpleCNN(input_channels=3, dropout=0.5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    # model = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE)
    
    print("\nTo use this script:")
    print("1. Add your dataset to ../../datasets/")
    print("2. Update the LettuceDataset class to load your data")
    print("3. Uncomment the dataset and training code")
    print("4. Run: python train.py")
