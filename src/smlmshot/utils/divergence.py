# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility tools to detect when a training diverge."""

import torch
from torch import Tensor


def detect_divergence(loss_history: list[Tensor], rtol: float = 10.0, window: int = 7):
    """Detect divergence based on loss spikes over recent history."""
    if len(loss_history) == 0:
        return False

    last = loss_history[-1]
    if torch.isnan(last):
        return True

    if len(loss_history) < window:
        return False

    ref = torch.stack(loss_history[-window:-1])
    med = torch.quantile(ref, q=0.5)
    mad = torch.quantile(torch.abs(ref - med), q=0.5)

    diverge = last > med + rtol * mad
    return diverge.item()
