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


class RGBRegressionBranch(nn.Module):
    """Processes RGB inputs and predicts dry weight."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.features(rgb)
        return self.head(x).squeeze(-1)


class RGBDRegressionBranch(nn.Module):
    """Processes RGBD inputs and predicts dry weight."""

    def __init__(self, in_channels: int = 4, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        x = self.features(rgbd)
        return self.head(x).squeeze(-1)


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LettuceMultiBranchCNN(nn.Module):
    """Two regression branches + fusion for dry-weight prediction."""

    def __init__(self):
        super().__init__()
        self.rgb_branch = RGBRegressionBranch()
        self.rgbd_branch = RGBDRegressionBranch()
        self.fusion = FusionMLP(in_dim=2)

    def forward(self, rgb: torch.Tensor, rgbd: torch.Tensor):
        rgb_pred = self.rgb_branch(rgb)
        rgbd_pred = self.rgbd_branch(rgbd)
        fusion_in = torch.stack([rgb_pred, rgbd_pred], dim=1)
        fusion_pred = self.fusion(fusion_in)
        return rgb_pred, rgbd_pred, fusion_pred

    @torch.no_grad()
    def predict_dry_weight(self, rgb: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, _, fusion_pred = self.forward(rgb, rgbd)
        return fusion_pred


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)
