# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions around images."""

import numpy as np
import torch
from tifffile import imread
from torch import Tensor


def read_tiff(filepath: str) -> Tensor:
    """Read a stack of images from a tiff file, and return a 3D Tensor [L, H, W]."""
    y = imread(filepath)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)
    return y


def estimate_background(y: Tensor) -> Tensor:
    """Return an estimate of the background of a 3D Tensor (min of each pixel)."""
    if y.ndim != 3:
        raise ValueError("estimate_background requires a 3D tensor")
    bg = y.min(dim=0).values
    return bg


def standardize_background(y: Tensor) -> Tensor:
    """Transform a 3D Tensor such that all pixels have a common minimum across time."""
    bg = estimate_background(y)
    bg_med = bg.median()
    y = y - bg + bg_med
    return y


def crop(y: Tensor, x0: int, x1: int, y0: int, y1: int) -> Tensor:
    """Crop a 3D Tensor of shape [L, H, W] to y0:y1 and x0:x1 along spatial dims."""
    if y.ndim != 3:
        raise ValueError("crop requires a 3D tensor")
    return y[:, y0:y1, x0:x1]


def bin(y: Tensor, bin_size: int, reduction: str = "mean") -> Tensor:
    """Temporally bin a 3D Tensor [L, H, W] to produce a [L/bin, H, W] tensor."""
    if y.ndim != 3:
        raise ValueError("bin requires a 3D tensor")
    bs, h, w = y.size()
    y = y[: bs - bs % bin_size]
    y = y.reshape(bs // bin_size, bin_size, h, w)
    if reduction == "mean":
        return y.mean(dim=1)
    if reduction == "max":
        return y.max(dim=1).values
    raise ValueError(f"Reduction {reduction} is not supported.")
