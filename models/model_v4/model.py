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


class RGBClassificationBranch(nn.Module):
    """RGB -> 4 logits."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),  # 64->32
            ConvBlock(32, 64),           # 32->16
            ConvBlock(64, 128),          # 16->8
            ConvBlock(128, 256),         # 8->4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.features(rgb)
        return self.head(x)


class RGBDRegressionBranch(nn.Module):
    """RGBD (4ch) -> 1 scalar dry weight."""

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
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        x = self.features(rgbd)
        return self.head(x).squeeze(-1)


class FusionMLP(nn.Module):
    """(4 logits + 1 reg) -> final dry weight."""

    def __init__(self, in_dim: int = 5, hidden: int = 64, dropout: float = 0.3):
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
    """Model v4.

    Contract:
    - Input:
        rgb:  (N,3,64,64)
        rgbd: (N,4,64,64)
    - Outputs:
        rgb_logits: (N,4)
        rgbd_pred:  (N,)
        fusion_pred:(N,)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.rgb_branch = RGBClassificationBranch(num_classes=num_classes)
        self.rgbd_branch = RGBDRegressionBranch()
        self.fusion = FusionMLP(in_dim=num_classes + 1)

    def forward(self, rgb: torch.Tensor, rgbd: torch.Tensor):
        rgb_logits = self.rgb_branch(rgb)
        rgbd_pred = self.rgbd_branch(rgbd)  # (N,)
        fusion_in = torch.cat([rgb_logits, rgbd_pred.unsqueeze(1)], dim=1)
        fusion_pred = self.fusion(fusion_in)
        return rgb_logits, rgbd_pred, fusion_pred

    @torch.no_grad()
    def predict_dry_weight(self, rgb: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, _, fusion_pred = self.forward(rgb, rgbd)
        return fusion_pred


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)
