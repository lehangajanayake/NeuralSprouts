import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard conv block: Conv2d -> BatchNorm -> ReLU -> MaxPool2d."""
    
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBBranch(nn.Module):
    """RGB branch (3 channels) -> shared feature representation."""
    
    def __init__(self, in_channels: int = 3, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # H/2
            ConvBlock(32, 64),             # H/4
            ConvBlock(64, 128),            # H/8
            ConvBlock(128, 256),           # H/16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, branch_dim),  # 128x128 -> 8x8 after 4 poolings
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (N, 3, 128, 128). Returns: (N, branch_dim)."""
        return self.head(self.features(x))


class RGBDBranch(nn.Module):
    """RGBD branch (4 channels: RGB + Depth) -> shared feature representation."""
    
    def __init__(self, in_channels: int = 4, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, branch_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (N, 4, 128, 128). Returns: (N, branch_dim)."""
        return self.head(self.features(x))


class DepthBranch(nn.Module):
    """Depth branch (1 channel) -> shared feature representation."""
    
    def __init__(self, in_channels: int = 1, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, branch_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (N, 1, 128, 128). Returns: (N, branch_dim)."""
        return self.head(self.features(x))


class FusionFC(nn.Module):
    """Fusion layer: concatenate features from 3 branches -> final prediction."""
    
    def __init__(self, branch_dim: int = 256, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        in_dim = branch_dim * 3  # 3 branches concatenated
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, 1),  # Single output: dry weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (N, in_dim). Returns: (N,)."""
        return self.net(x).squeeze(-1)


class PlantV5TripleBranch(nn.Module):
    """Model V5: Triple-branch fusion for plant dry weight prediction.
    
    Architecture:
    - RGB Branch: 3 channels
    - RGBD Branch: 4 channels (R, G, B, D)
    - Depth Branch: 1 channel
    
    Each branch has independent conv blocks (NO weight sharing).
    Features concatenated and passed through final FC for prediction.
    
    Input:
        rgb: (N, 3, 128, 128)
        rgbd: (N, 4, 128, 128)
        depth: (N, 1, 128, 128)
    
    Output:
        pred: (N,) - dry weight predictions
    """
    
    def __init__(
        self,
        branch_dim: int = 256,
        fc_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Three independent branches
        self.rgb_branch = RGBBranch(in_channels=3, branch_dim=branch_dim, dropout=dropout)
        self.rgbd_branch = RGBDBranch(in_channels=4, branch_dim=branch_dim, dropout=dropout)
        self.depth_branch = DepthBranch(in_channels=1, branch_dim=branch_dim, dropout=dropout)
        
        # Fusion layer
        self.fusion_fc = FusionFC(branch_dim=branch_dim, hidden=fc_hidden, dropout=dropout)

    def forward(
        self,
        rgb: torch.Tensor,
        rgbd: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all branches and fusion.
        
        Args:
            rgb: (N, 3, 128, 128)
            rgbd: (N, 4, 128, 128)
            depth: (N, 1, 128, 128)
        
        Returns:
            pred: (N,) - dry weight predictions
        """
        rgb_features = self.rgb_branch(rgb)      # (N, branch_dim)
        rgbd_features = self.rgbd_branch(rgbd)   # (N, branch_dim)
        depth_features = self.depth_branch(depth)  # (N, branch_dim)
        
        # Concatenate all features
        fused = torch.cat([rgb_features, rgbd_features, depth_features], dim=1)  # (N, branch_dim*3)
        
        # Final prediction
        pred = self.fusion_fc(fused)  # (N,)
        
        return pred

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, rgbd: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        return self.forward(rgb, rgbd, depth)
