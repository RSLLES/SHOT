# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZNorm(nn.Module):
    """Applies (x - mu) / sigma with learnable mu and sigma > 0."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        mu = torch.tensor(float(mu))
        sigma = torch.tensor(float(sigma))
        if torch.any(sigma <= 0):
            raise ValueError("sigma must be > 0")
        b = -mu
        a = torch.log(torch.expm1(1 / sigma))  # inverse softplus
        self.a = nn.Parameter(a, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)

    def forward(self, x):  # noqa: D102
        return (x + self.b) * F.softplus(self.a)
