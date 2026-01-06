# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions that compute extents - i.e ranges - in 2D or 3D."""

import torch
from torch import Tensor

from .torch import to_pair


def get_img_extent(h: int, w: int, pixel_size: Tensor):
    """Return [[0, img_size_h], [0, img_size_w]] in nm."""
    pixel_size = to_pair(pixel_size)
    pixel_size = torch.as_tensor(pixel_size)
    img_size = torch.tensor((h, w), device=pixel_size.device) * pixel_size
    img_extent = [torch.zeros_like(img_size), img_size]
    img_extent = torch.stack(img_extent, dim=-1)
    return img_extent


def get_vol_extent(h: int, w: int, pixel_size: Tensor, z_extent: Tensor):
    """Return [[0, img_size_h], [0, img_size_w], [z_min, z_max]] in nm."""
    img_extent = get_img_extent(h=h, w=w, pixel_size=pixel_size)
    vol_extent = [img_extent, z_extent[None]]
    vol_extent = torch.cat(vol_extent, dim=0)
    return vol_extent
