"""model_v10 — Dual-branch CNN with SE + spatial attention, GeM pooling, and fusion MLP.

Architectural changes from v8
------------------------------
* **Unified ``RegressionBranch``** replaces the near-duplicate
  ``RGBRegressionBranch`` / ``RGBDRegressionBranch`` classes.
* **Squeeze-and-Excitation (SE) channel attention** inserted after the
  convolutional backbone and before spatial attention.  SE learns to
  re-weight channels dynamically — e.g. suppressing noisy depth when
  it is uninformative.
* **Generalised Mean (GeM) pooling** replaces ``AdaptiveAvgPool2d``.
  A learnable exponent *p* interpolates between average (*p* = 1) and
  max (*p* → ∞) pooling, consistently outperforming fixed average
  pooling for regression.
* ``_infer_branch_widths`` class-method on the top-level model makes
  checkpoint introspection reusable across eval / predict scripts.

Note: v10 checkpoints are **not** backward-compatible with v8.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth (drop-path) regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x.div(keep) * mask


class BottleneckBlock(nn.Module):
    """1×1 → 3×3 → 1×1 bottleneck with residual connection + 2× pooling."""

    def __init__(
        self, in_ch: int, out_ch: int, *, reduction: int = 4, drop_prob: float = 0.0
    ) -> None:
        super().__init__()
        mid = max(1, out_ch // max(1, reduction))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else None
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.pool(self.activation(identity + self.drop_path(out)))


class SpatialAttentionModule(nn.Module):
    """CBAM-style spatial attention (channel avg + max → conv → sigmoid)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class SqueezeExcitation(nn.Module):
    """SE channel-attention block.

    Learns per-channel scaling factors via global-pool → FC → ReLU → FC → Sigmoid.
    Particularly useful for RGBD inputs where depth-channel quality varies.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class GeM(nn.Module):
    """Generalised Mean pooling with a learnable exponent *p*.

    .. math::
        \text{GeM}(x) = \left(\frac{1}{HW}\sum_{h,w} x_{h,w}^{p}\right)^{1/p}

    When *p* = 1 this is average pooling; as *p* → ∞ it approaches max
    pooling.  The exponent is learned during training.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp to avoid negative bases (ReLU output is ≥ 0, but eps guards edge cases)
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(2, 3), keepdim=True).pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# Unified regression branch
# ---------------------------------------------------------------------------

def _linear_drop_rates(n: int, max_rate: float) -> Tuple[float, ...]:
    """Linearly increasing stochastic-depth rates from 0 to *max_rate*."""
    if n <= 0:
        return ()
    if max_rate <= 0.0:
        return (0.0,) * n
    if n == 1:
        return (max_rate,)
    return tuple(max_rate * i / (n - 1) for i in range(n))


class RegressionBranch(nn.Module):
    """Shared backbone for both the RGB and RGBD branches.

    This single class replaces the near-duplicate ``RGBRegressionBranch`` and
    ``RGBDRegressionBranch`` from v8.  The only things that differ between the
    two instantiations are ``in_channels`` and ``widths``.
    """

    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, ...],
        *,
        dropout: float = 0.2,
        drop_path_prob: float = 0.0,
        embed_dim: int = 256,
        se_reduction: int = 4,
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("RegressionBranch requires at least one width value.")
        self.embedding_dim = int(embed_dim)
        drops = _linear_drop_rates(len(widths), drop_path_prob)
        blocks: List[nn.Module] = []
        ch = in_channels
        for w, d in zip(widths, drops):
            blocks.append(BottleneckBlock(ch, w, drop_prob=d))
            ch = w
        self.features = nn.Sequential(*blocks)
        self.se = SqueezeExcitation(widths[-1], reduction=se_reduction)
        self.spatial_attn = SpatialAttentionModule(kernel_size=7)
        self.post_attn_dropout = nn.Dropout(p=dropout)
        self.global_pool = GeM(p=gem_p)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.regressor = nn.Linear(self.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.se(x)                          # channel attention (which channels matter)
        x = self.post_attn_dropout(self.spatial_attn(x))  # spatial attention (where to look)
        feat = self.embedding(self.global_pool(x))
        pred = self.regressor(feat).squeeze(-1)
        return pred, feat


# ---------------------------------------------------------------------------
# Fusion head
# ---------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """Two-hidden-layer MLP that fuses the two branch embeddings."""

    def __init__(self, in_dim: int = 512, hidden: int = 256, dropout: float = 0.3) -> None:
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


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class LettuceSAMFusionNet(nn.Module):
    """Dual-branch CNN with spatial attention + fusion head for dry-weight regression."""

    def __init__(
        self,
        *,
        drop_path_prob: float = 0.1,
        rgb_widths: Tuple[int, ...] = (32, 64, 96, 128),
        rgbd_widths: Tuple[int, ...] = (32, 64, 96, 128),
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.rgb_branch = RegressionBranch(
            in_channels=3,
            widths=rgb_widths,
            drop_path_prob=drop_path_prob,
            embed_dim=embed_dim,
        )
        self.rgbd_branch = RegressionBranch(
            in_channels=4,
            widths=rgbd_widths,
            drop_path_prob=drop_path_prob,
            embed_dim=embed_dim,
        )
        self.fusion = FusionMLP(
            in_dim=self.rgb_branch.embedding_dim + self.rgbd_branch.embedding_dim,
        )
        self.fusion_in_dropout = nn.Dropout(p=0.2)

    # ---- forward / inference -------------------------------------------

    def forward(
        self, rgb: torch.Tensor, rgbd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_pred, rgb_feat = self.rgb_branch(rgb)
        rgbd_pred, rgbd_feat = self.rgbd_branch(rgbd)
        fused = self.fusion_in_dropout(torch.cat([rgb_feat, rgbd_feat], dim=1))
        return rgb_pred, rgbd_pred, self.fusion(fused)

    @torch.no_grad()
    def predict_dry_weight(self, rgb: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, _, pred = self.forward(rgb, rgbd)
        return pred

    # ---- checkpoint helpers --------------------------------------------

    @staticmethod
    def infer_branch_widths(
        state_dict: Dict[str, torch.Tensor], branch_prefix: str
    ) -> Tuple[int, ...]:
        """Read per-block output widths from a saved ``state_dict``."""
        widths: List[int] = []
        idx = 0
        while True:
            key = f"{branch_prefix}.features.{idx}.conv3.1.weight"
            t = state_dict.get(key)
            if t is None:
                break
            widths.append(int(t.shape[0]))
            idx += 1
        if not widths:
            raise ValueError(
                f"Cannot infer widths for '{branch_prefix}'; checkpoint is missing expected keys."
            )
        return tuple(widths)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
        drop_path_prob: float = 0.1,
    ) -> "LettuceSAMFusionNet":
        """Instantiate a model and load weights from *path*.

        Branch widths and embed dim are inferred automatically so the caller
        never needs to hard-code architecture hyper-parameters.
        """
        state = torch.load(path, map_location=device, weights_only=True)
        rgb_widths = cls.infer_branch_widths(state, "rgb_branch")
        rgbd_widths = cls.infer_branch_widths(state, "rgbd_branch")
        # Infer embed_dim from the regressor weight shape
        embed_key = "rgb_branch.regressor.weight"
        embed_dim = int(state[embed_key].shape[1]) if embed_key in state else 256
        model = cls(
            drop_path_prob=drop_path_prob,
            rgb_widths=rgb_widths,
            rgbd_widths=rgbd_widths,
            embed_dim=embed_dim,
        )
        model.load_state_dict(state)
        return model.to(device)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Toggle ``requires_grad`` for every parameter in *module*."""
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)
