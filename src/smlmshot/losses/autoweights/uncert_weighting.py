# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Uncertainty weighting for regression, as described in kendall2018multi."""

import torch
import torch.nn as nn


class RegressionUncertaintyWeighting(nn.Module):
    """Compute a weighted sum of losses based on learned uncertainties."""

    def __init__(self, n_losses: int):
        super().__init__()
        if n_losses < 1:
            raise ValueError(f"n_losses must be >=1, got {n_losses}")
        self.log_sigma = nn.Parameter(torch.full(n_losses, value=torch.nan))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """Return the weighted loss sum."""
        if losses.dim() != 1 or losses.size(0) != self.log_sigma.size(0):
            raise ValueError(
                f"Expected shape is {self.log_sigma.size(0)}, got {losses.shape}"
            )

        if torch.isnan(self.log_sigma).all():
            with torch.no_grad():
                self.log_sigma.copy_(0.5 * torch.log(losses.abs()))

        w = torch.exp(-2.0 * self.log_sigma)
        losses = 0.5 * w * losses + self.log_sigma
        loss = losses.sum()
        return loss
