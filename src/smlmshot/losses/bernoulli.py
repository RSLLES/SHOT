# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn


class BernoulliLoss(nn.Module):
    """Original loss function for detection scores proposed by Decode."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, p: Tensor, n_gt: Tensor):  # noqa: D102
        p = p.clip(min=self.eps, max=1.0 - self.eps)
        mu = torch.sum(p, dim=-1)
        var = torch.sum(p * (1 - p), dim=-1)
        loss = torch.square(n_gt - mu) / var + torch.log(var)
        return 0.5 * loss.mean()
