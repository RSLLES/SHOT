# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def perform_hard_thresholding(x: Tensor, s: Tensor, threshold: float):
    """Select fluorophores if their detection score exceed a threshold."""
    x = [x_[s_ > threshold] for s_, x_ in zip(s, x)]
    return x


def perform_soft_thresholding(x: Tensor, s: Tensor):
    """Scale fluorophore intensities by their detection score."""
    xyz, n = x[..., :3], x[..., 3:]
    x = torch.cat([xyz, s[..., None] * n], dim=-1)
    return x
