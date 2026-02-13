from typing import Tuple

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth (a.k.a. DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class BottleneckBlock(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck with residual + pooling."""

    def __init__(self, in_ch: int, out_ch: int, *, reduction: int = 4, drop_prob: float = 0.0):
        super().__init__()
        mid_ch = max(1, out_ch // max(1, reduction))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = identity + self.drop_path(out)
        out = self.activation(out)
        out = self.pool(out)
        return out


class SpatialAttentionModule(nn.Module):
    """Spatial attention block from CBAM (avg + max pooling along channels)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        scale = self.activation(attn)
        return x * scale


def _drop_rates(num_blocks: int, drop_path_prob: float) -> Tuple[float, ...]:
    if num_blocks <= 0:
        return tuple()
    if drop_path_prob <= 0.0:
        return tuple(0.0 for _ in range(num_blocks))
    if num_blocks == 1:
        return (drop_path_prob,)
    return tuple(drop_path_prob * i / (num_blocks - 1) for i in range(num_blocks))


class RGBRegressionBranch(nn.Module):
    """Processes RGB inputs, outputs both scalar prediction and pooled features."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout: float = 0.2,
        drop_path_prob: float = 0.0,
        widths: Tuple[int, ...] = (24, 48, 64, 96),
        embed_dim: int = 256,
    ):
        super().__init__()
        if not widths:
            raise ValueError('RGBRegressionBranch requires at least one width value.')
        self.embedding_dim = int(embed_dim)
        drops = _drop_rates(len(widths), drop_path_prob)
        layers = []
        c_in = in_channels
        for width, drop in zip(widths, drops):
            layers.append(BottleneckBlock(c_in, width, drop_prob=drop))
            c_in = width
        self.features = nn.Sequential(*layers)
        self.spatial_attn = SpatialAttentionModule(kernel_size=7)
        self.post_attn_dropout = nn.Dropout(p=dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        last_width = widths[-1]
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_width, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.regressor = nn.Linear(self.embedding_dim, 1)

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(rgb)
        x = self.spatial_attn(x)
        x = self.post_attn_dropout(x)
        pooled = self.global_pool(x)
        features = self.embedding(pooled)
        pred = self.regressor(features).squeeze(-1)
        return pred, features


class RGBDRegressionBranch(nn.Module):
    """Processes RGBD inputs, outputs both scalar prediction and pooled features."""

    def __init__(
        self,
        in_channels: int = 4,
        dropout: float = 0.2,
        drop_path_prob: float = 0.1,
        widths: Tuple[int, ...] = (32, 64, 96, 128),
        embed_dim: int = 256,
    ):
        super().__init__()
        if not widths:
            raise ValueError('RGBDRegressionBranch requires at least one width value.')
        self.embedding_dim = int(embed_dim)
        drops = _drop_rates(len(widths), drop_path_prob)
        layers = []
        c_in = in_channels
        for width, drop in zip(widths, drops):
            layers.append(BottleneckBlock(c_in, width, drop_prob=drop))
            c_in = width
        self.features = nn.Sequential(*layers)
        self.spatial_attn = SpatialAttentionModule(kernel_size=7)
        self.post_attn_dropout = nn.Dropout(p=dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        last_width = widths[-1]
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_width, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.regressor = nn.Linear(self.embedding_dim, 1)

    def forward(self, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(rgbd)
        x = self.spatial_attn(x)
        x = self.post_attn_dropout(x)
        pooled = self.global_pool(x)
        features = self.embedding(pooled)
        pred = self.regressor(features).squeeze(-1)
        return pred, features


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256, dropout: float = 0.3):
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


class LettuceSAMFusionNet(nn.Module):
    """Dual-branch CNN with spatial attention + fusion head."""

    def __init__(
        self,
        drop_path_prob: float = 0.1,
        rgb_widths: Tuple[int, ...] = (32, 64, 96, 128),
        rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128),
        embed_dim: int = 256,
    ):
        super().__init__()
        self.rgb_branch = RGBRegressionBranch(drop_path_prob=drop_path_prob, widths=rgb_widths, embed_dim=embed_dim)
        self.rgbd_branch = RGBDRegressionBranch(drop_path_prob=drop_path_prob, widths=rgbd_widths, embed_dim=embed_dim)
        fusion_in_dim = self.rgb_branch.embedding_dim + self.rgbd_branch.embedding_dim
        self.fusion = FusionMLP(in_dim=fusion_in_dim)
        self.fusion_in_dropout = nn.Dropout(p=0.2)

    def forward(self, rgb: torch.Tensor, rgbd: torch.Tensor):
        rgb_pred, rgb_feat = self.rgb_branch(rgb)
        rgbd_pred, rgbd_feat = self.rgbd_branch(rgbd)
        fusion_in = torch.cat([rgb_feat, rgbd_feat], dim=1)
        fusion_in = self.fusion_in_dropout(fusion_in)
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
