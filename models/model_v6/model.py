"""Dual-branch fusion model for direct dry-weight regression."""

from typing import Tuple

import torch
import torch.nn as nn

from config import Config


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> 2x2 max pool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BranchEncoder(nn.Module):
    """Shared encoder logic for RGB and RGBD branches."""

    def __init__(self,
                 in_channels: int,
                 input_size: int,
                 num_layers: int,
                 initial_filters: int,
                 filter_multiplier: int,
                 embed_dim: int,
                 dropout: float):
        super().__init__()
        layers = []
        channels = in_channels
        filters = initial_filters
        for _ in range(num_layers):
            layers.append(ConvBlock(channels, filters))
            channels = filters
            filters = max(filters * filter_multiplier, filters)
        self.features = nn.Sequential(*layers)

        dummy = torch.zeros(1, in_channels, input_size, input_size)
        with torch.no_grad():
            flattened = self.features(dummy).view(1, -1).shape[1]
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.features(x)
        embedding = self.projection(feature_maps)
        return embedding, feature_maps


class DualBranchFusion(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config or Config()

        self.rgb_branch = BranchEncoder(
            in_channels=self.config.RGB_CHANNELS,
            input_size=self.config.RESIZE_SIZE,
            num_layers=self.config.NUM_CONV_LAYERS,
            initial_filters=self.config.INITIAL_FILTERS,
            filter_multiplier=self.config.FILTER_MULTIPLIER,
            embed_dim=self.config.BRANCH_EMBED_DIM,
            dropout=self.config.DROPOUT_RATE
        )

        self.rgbd_branch = BranchEncoder(
            in_channels=self.config.RGBD_CHANNELS,
            input_size=self.config.RESIZE_SIZE,
            num_layers=self.config.NUM_CONV_LAYERS,
            initial_filters=self.config.INITIAL_FILTERS,
            filter_multiplier=self.config.FILTER_MULTIPLIER,
            embed_dim=self.config.BRANCH_EMBED_DIM,
            dropout=self.config.DROPOUT_RATE
        )

        fusion_input_dim = self.config.BRANCH_EMBED_DIM * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, self.config.FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.DROPOUT_RATE),
            nn.Linear(self.config.FUSION_HIDDEN_DIM, self.config.FUSION_HIDDEN_DIM // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.DROPOUT_RATE),
            nn.Linear(self.config.FUSION_HIDDEN_DIM // 2, self.config.OUTPUT_DIM)
        )

    def forward(self, rgb: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        rgb_embed, _ = self.rgb_branch(rgb)
        rgbd_embed, _ = self.rgbd_branch(rgbd)
        fused = torch.cat([rgb_embed, rgbd_embed], dim=1)
        return self.fusion_head(fused)

    def get_branch_features(self, rgb: torch.Tensor, rgbd: torch.Tensor):
        _, rgb_maps = self.rgb_branch(rgb)
        _, rgbd_maps = self.rgbd_branch(rgbd)
        return rgb_maps, rgbd_maps

    def get_attention_maps(self, rgb: torch.Tensor, rgbd: torch.Tensor):
        rgb_maps, rgbd_maps = self.get_branch_features(rgb, rgbd)
        rgb_attention = rgb_maps.mean(dim=1)
        rgbd_attention = rgbd_maps.mean(dim=1)
        return rgb_attention, rgbd_attention


def create_model(config: Config = None) -> DualBranchFusion:
    return DualBranchFusion(config)


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
