# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CoVWeighting as described in https://arxiv.org/abs/2009.01717."""

import torch
from torch import Tensor, nn


class CoVWeighting(nn.Module):
    """Compute dynamic loss weighting based on coefficient of variation."""

    def __init__(self, n_losses: int, eps: float = 1e-12, t_lim: int = 1000):
        super(CoVWeighting, self).__init__()
        self.eps = eps
        self.n_losses = n_losses
        self.t_lim = t_lim

        # Initialize moving averages for mean and variance estimation
        self.register_buffer("t", torch.tensor(-2.0))
        self.register_buffer("mu_L", torch.zeros(n_losses))
        self.register_buffer("mu_l", torch.zeros(n_losses))
        self.register_buffer("M", torch.zeros(n_losses))

    def forward(self, losses: Tensor):
        """Return the weighted loss."""
        if losses.ndim != 1 or losses.size(0) != self.n_losses:
            raise ValueError(
                f"Losses must have a shape of ({self.n_losses},); found {losses.size()}"
            )
        if torch.any(losses < 0):
            raise ValueError("CoVWeighting doesn't support negative losses.")

        losses = losses.detach()

        # warmup ?
        if self.t < 0.0:
            self.mu_l = losses / (self.mu_L + self.eps)
            self.mu_L = losses
            self.t += 1.0
            return torch.full_like(losses, 1 / self.n_losses)

        L = losses
        l = losses / self.mu_L.clamp_min(self.eps)  # noqa: E741

        if self.t < self.t_lim:
            self.t += 1.0
        lambd = 1.0 / self.t
        # lambd = 1.0 / torch.sqrt(self.t)

        mu_L = (1.0 - lambd) * self.mu_L + lambd * L
        mu_l = (1.0 - lambd) * self.mu_l + lambd * l
        M = (1.0 - lambd) * self.M + lambd * (l - self.mu_l) * (l - mu_l)

        std = torch.sqrt(M)
        cov = std / mu_l.clamp_min(self.eps) + self.eps
        w = cov / cov.sum().clamp_min(self.eps)

        self.mu_L = mu_L
        self.mu_l = mu_l
        self.M = M

        loss = (w * losses).sum()
        return loss
