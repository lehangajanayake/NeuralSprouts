"""
Base model interface for all lettuce dry weight prediction models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all prediction models.
    
    This class provides a common interface for model iterations,
    ensuring consistency across different architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.version = config.get('version', 'unknown')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted dry weight
        """
        pass
    
    def get_version(self) -> str:
        """Get the model version."""
        return self.version
    
    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None, 
                       metrics: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'version': self.version,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str, device: str = 'cpu'):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Checkpoint dictionary containing metadata
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
