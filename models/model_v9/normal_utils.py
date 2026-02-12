"""Utilities for converting depth maps into surface-normal tensors.

The implementation follows a pinhole camera model under the assumption of a
square pixel grid and uses Sobel gradients to estimate the partial derivatives
of the reconstructed 3D points. Normals are normalized to the [-1, 1] cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class NormalComputationConfig:
    """Configuration for converting depth maps into normal maps."""

    fx: Optional[float] = None
    fy: Optional[float] = None
    normalize_output: bool = True
    eps: float = 1e-6


def _prepare_depth(depth: np.ndarray) -> torch.Tensor:
    if depth.ndim != 2:
        raise ValueError('Depth array must be H×W.')
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)


def compute_surface_normals(
    depth: np.ndarray,
    *,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return unit surface normals for the provided depth array.

    Args:
        depth: 2D numpy array containing metric depth (any linear scale).
        fx, fy: Optional focal lengths in pixels; defaults to max(width, height).
        eps: Numerical stability epsilon for normalization.
    """

    depth_t = _prepare_depth(depth)
    device = depth_t.device
    dtype = depth_t.dtype
    _, _, h, w = depth_t.shape

    fx = float(fx) if fx is not None else float(max(w, 1))
    fy = float(fy) if fy is not None else float(max(h, 1))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    u = torch.linspace(0, w - 1, w, dtype=dtype, device=device).view(1, 1, 1, w)
    v = torch.linspace(0, h - 1, h, dtype=dtype, device=device).view(1, 1, h, 1)

    X = (u - cx) / fx * depth_t
    Y = (v - cy) / fy * depth_t
    Z = depth_t.clone()

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0

    def grad(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(tensor, kernel, padding=1)

    dPdu = torch.cat([grad(X, sobel_x), grad(Y, sobel_x), grad(Z, sobel_x)], dim=1)
    dPdv = torch.cat([grad(X, sobel_y), grad(Y, sobel_y), grad(Z, sobel_y)], dim=1)

    normals = torch.cross(dPdu.permute(0, 2, 3, 1), dPdv.permute(0, 2, 3, 1), dim=-1)
    normals = F.normalize(normals, dim=-1, eps=eps)

    mask = (depth_t <= 0).permute(0, 2, 3, 1)
    normals = torch.where(mask, torch.zeros_like(normals), normals)

    normals_np = normals.squeeze(0).cpu().numpy()  # (H, W, 3)
    return normals_np.astype(np.float32, copy=False)


def normals_to_uint8(normals: np.ndarray) -> np.ndarray:
    normals = np.clip(normals, -1.0, 1.0, out=None)
    normal01 = (normals + 1.0) * 0.5
    return (normal01 * 255.0).round().astype(np.uint8)


def depth_image_to_normal_image(
    depth_img: Image.Image,
    *,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> Image.Image:
    depth = np.asarray(depth_img, dtype=np.float32)
    if depth.size and float(depth.max()) > 1.5:
        depth = depth / 255.0
    normals = compute_surface_normals(depth, fx=fx, fy=fy)
    normal_img = normals_to_uint8(normals)
    return Image.fromarray(normal_img, mode='RGB')


def normal_image_to_tensor(normal_img: Image.Image) -> torch.Tensor:
    arr = np.asarray(normal_img.convert('RGB'), dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def save_normal_image(normal_img: Image.Image, path: str) -> None:
    normal_img.save(path)
