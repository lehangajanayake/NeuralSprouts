import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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


class RGBRegressionBackbone(nn.Module):
    """RGB -> dry weight regression head."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 96 -> 48
            ConvBlock(32, 64),            # 48 -> 24
            ConvBlock(64, 128),           # 24 -> 12
            ConvBlock(128, 256),          # 12 -> 6
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.features(rgb)
        return self.head(x).squeeze(-1)


class LettuceMultiBranchCNN(nn.Module):
    """Single-branch RGB regressor (legacy name for compatibility)."""

    def __init__(self):
        super().__init__()
        self.regressor = RGBRegressionBackbone()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.regressor(rgb)

    @torch.no_grad()
    def predict_dry_weight(self, rgb: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(rgb)
