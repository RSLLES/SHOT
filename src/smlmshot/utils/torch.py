# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""General utility functions for PyTorch."""

import logging

import torch
import torch.distributions.utils
from torch import Tensor
from torch.nn.modules.utils import _pair, _quadruple, _single, _triple

from .format import format_size


def initialize_torch(detect_anomaly: bool = False):
    """Set PyTorch to high precision and suppress debug logs."""
    if detect_anomaly:
        print("Warning: detect_anomaly is enabled.")
    torch.autograd.set_detect_anomaly(detect_anomaly)
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(linewidth=160)
    torch._logging.set_logs(all=logging.WARNING)


def hash_tensor(tensor: Tensor):
    """Hash a tensor."""
    return torch.hash_tensor(tensor).item()


def get_memory_size(tensor: Tensor, as_string: bool = True) -> int | str:
    """Return the size occupied by the tensor in memory in bytes."""
    size = tensor.element_size() * tensor.nelement()
    if not as_string:
        return size
    return format_size(size)


def are_broadcastable(shape1, shape2):
    """Determine whether two shapes are broadcastable."""
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True


def flatten_spatial(x: Tensor) -> Tensor:
    """Flatten and transpose a 4D input (B,C,H,W) to (B,N=HW,C)."""
    if x.ndim != 4:
        raise ValueError(f"Expect a 4D tensor, got {x.ndim} dimensions.")
    x = x.view(x.size(0), x.size(1), -1)  # b,c,n=hw
    x = x.transpose(1, 2)  # b,n,c
    x = x.contiguous()
    return x


def unflatten_spatial(x: Tensor, h: int, w: int) -> Tensor:
    """Reshape and transpose a 3D tensor (B, N=HW, C) back to (B, C, H, W)."""
    if x.ndim != 3:
        raise ValueError(f"Expect a 3D tensor, got {x.ndim} dimensions.")
    bs, n, c = x.shape
    if n != h * w:
        raise ValueError(f"N = {n} is not equal to H * W = {h} * {w} = {h * w}.")
    x = x.transpose(1, 2)  # b,c,n
    x = x.contiguous().view(bs, c, h, w)  # b,c,h,w
    return x


"""Helper functions to convert element to tuples"""
to_single = _single
to_pair = _pair
to_triple = _triple
to_quadruple = _quadruple

"""Helper functions to turn logits to prob and conversely."""
logits_to_probs = torch.distributions.utils.logits_to_probs
probs_to_logits = torch.distributions.utils.probs_to_logits
