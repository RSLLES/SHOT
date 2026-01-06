# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import log, pi

import torch
from torch import Tensor, nn


class GaussianMixtureModelLoss(nn.Module):
    """Original Gaussian Mixture loss function proposed by Decode."""

    LOG_2PI = log(2.0 * pi)

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, p: Tensor, x: Tensor, u: Tensor, x_gt: Tensor, mask_gt: Tensor):  # noqa: D102
        C = x[:, None] - x_gt[:, :, None]
        C = torch.square(C / u[:, None]) + torch.log(u[:, None]) + self.LOG_2PI
        C = -0.5 * torch.sum(C, dim=-1)
        p = torch.nn.functional.normalize(p, dim=-1, p=1.0, eps=self.eps)
        C = C + torch.log(p + self.eps)[:, None]
        C = torch.logsumexp(C, dim=-1)
        C = C.masked_fill(~mask_gt, torch.nan)
        C = C.nanmean(-1)
        loss = -1.0 * C.nanmean()
        return loss
