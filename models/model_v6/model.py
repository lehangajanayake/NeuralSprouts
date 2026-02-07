"""
Model architecture for Model_v6.
Dual-branch CNN with RGB and RGBD branches, fusion layer, and dry weight prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNBranch(nn.Module):
    """Single CNN branch for feature extraction."""
    
    def __init__(self, in_channels: int, num_layers: int = 3, initial_filters: int = 32, 
                 filter_multiplier: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        
        layers = []
        in_ch = in_channels
        out_ch = initial_filters
        
        # Build convolutional layers
        for i in range(num_layers):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch * filter_multiplier
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        # For 96x96 input with 3 maxpool layers: 96 -> 48 -> 24 -> 12
        self.flattened_size = (out_ch // filter_multiplier) * 12 * 12
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps before flattening (for visualization)."""
        return self.conv_layers(x)


class FusionLayer(nn.Module):
    """Fusion layer to combine RGB and RGBD branch features."""
    
    def __init__(self, rgb_feature_dim: int, rgbd_feature_dim: int, hidden_dim: int = 256, 
                 dropout_rate: float = 0.5):
        super().__init__()
        
        # Concatenate RGB and RGBD features
        combined_dim = rgb_feature_dim + rgbd_feature_dim
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.output_dim = hidden_dim // 2
    
    def forward(self, rgb_features, rgbd_features):
        combined = torch.cat([rgb_features, rgbd_features], dim=1)
        fused = self.fusion_layers(combined)
        return fused


class DualBranchCNN(nn.Module):
    """Dual-branch CNN for dry weight prediction."""
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        self.config = config or Config()
        
        # RGB branch
        self.rgb_branch = CNNBranch(
            in_channels=self.config.RGB_CHANNELS,
            num_layers=self.config.NUM_CONV_LAYERS,
            initial_filters=self.config.INITIAL_FILTERS,
            filter_multiplier=self.config.FILTER_MULTIPLIER,
            dropout_rate=self.config.DROPOUT_RATE
        )
        
        # RGBD branch
        self.rgbd_branch = CNNBranch(
            in_channels=self.config.RGBD_CHANNELS,
            num_layers=self.config.NUM_CONV_LAYERS,
            initial_filters=self.config.INITIAL_FILTERS,
            filter_multiplier=self.config.FILTER_MULTIPLIER,
            dropout_rate=self.config.DROPOUT_RATE
        )
        
        # Fusion layer
        self.fusion = FusionLayer(
            rgb_feature_dim=self.rgb_branch.flattened_size,
            rgbd_feature_dim=self.rgbd_branch.flattened_size,
            hidden_dim=self.config.FUSION_HIDDEN_DIM,
            dropout_rate=self.config.DROPOUT_RATE
        )
        
        # Output layer for dry weight prediction
        self.output_layer = nn.Linear(self.fusion.output_dim, self.config.OUTPUT_DIM)
    
    def forward(self, rgb_image: torch.Tensor, rgbd_image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Extract features from both branches
        rgb_features = self.rgb_branch(rgb_image)
        rgbd_features = self.rgbd_branch(rgbd_image)
        
        # Fuse features
        fused_features = self.fusion(rgb_features, rgbd_features)
        
        # Predict dry weight
        output = self.output_layer(fused_features)
        
        return output
    
    def get_branch_features(self, rgb_image: torch.Tensor, rgbd_image: torch.Tensor):
        """Get feature maps from both branches (for visualization)."""
        rgb_feature_maps = self.rgb_branch.get_feature_maps(rgb_image)
        rgbd_feature_maps = self.rgbd_branch.get_feature_maps(rgbd_image)
        
        return rgb_feature_maps, rgbd_feature_maps
    
    def get_attention_maps(self, rgb_image: torch.Tensor, rgbd_image: torch.Tensor):
        """Get attention maps for visualization."""
        rgb_maps, rgbd_maps = self.get_branch_features(rgb_image, rgbd_image)
        
        # Average across channels to get attention map
        rgb_attention = rgb_maps.mean(dim=1)  # (batch_size, height, width)
        rgbd_attention = rgbd_maps.mean(dim=1)  # (batch_size, height, width)
        
        return rgb_attention, rgbd_attention


def create_model(config: Config = None) -> DualBranchCNN:
    """Create and return the model."""
    return DualBranchCNN(config)


if __name__ == "__main__":
    # Test model initialization
    config = Config()
    model = create_model(config)
    
    # Print model architecture
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 4
    rgb_input = torch.randn(batch_size, 3, 96, 96)
    rgbd_input = torch.randn(batch_size, 4, 96, 96)
    
    output = model(rgb_input, rgbd_input)
    print(f"\nInput shapes: RGB {rgb_input.shape}, RGBD {rgbd_input.shape}")
    print(f"Output shape: {output.shape}")
