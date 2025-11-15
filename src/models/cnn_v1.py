"""
Simple CNN model (v1) for lettuce dry weight prediction.

This is the first iteration of the model - a straightforward CNN architecture
to establish a baseline and test the pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    Simple Convolutional Neural Network for dry weight prediction.
    
    Architecture:
    - 3 convolutional blocks with max pooling
    - 2 fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SimpleCNN model.
        
        Args:
            config: Configuration dictionary with keys:
                - version: Model version (default: 'v1')
                - input_channels: Number of input channels (default: 3 for RGB)
                - input_size: Input image size (default: 224)
                - dropout_rate: Dropout probability (default: 0.5)
        """
        super(SimpleCNN, self).__init__(config)
        
        # Configuration parameters
        input_channels = config.get('input_channels', 3)
        input_size = config.get('input_size', 224)
        dropout_rate = config.get('dropout_rate', 0.5)
        
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened size after convolutions
        # Input size reduces by factor of 8 (3 max pools with stride 2)
        self.flattened_size = 128 * (input_size // 8) * (input_size // 8)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Single output for regression (dry weight)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted dry weight tensor of shape (batch_size, 1)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_layers(x)
        return x
