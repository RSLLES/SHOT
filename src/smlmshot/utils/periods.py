# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions around periods.

A period is a time interval of the form [t0, t1] where t0 <= t1.
periods is a [L, 2] tensor containing L time interval.
"""

import torch
from torch import Tensor


def batch_discretize_periods(periods: Tensor, n_bins: int):
    """Discretize time intervals into fixed temporal bins.

    Given a batch of time intervals [B, L, 2], compute how much each periods overlaps
    with n_bins unit-length time interval [0, 1, ..., n_bins-1].
    Example: given periods [[0.5, 1.7], [1.9, 2.3]] and n_bins=4, it returns
    [0.5, 0.8, 0.3, 0.0]
    """
    frames = torch.arange(n_bins, device=periods.device)
    s = torch.maximum(periods[..., 0, None], frames)
    e = torch.minimum(periods[..., 1, None], frames + 1)
    intersection = torch.maximum(torch.zeros_like(e), e - s)
    intersection = intersection.sum(dim=1)
    return intersection
