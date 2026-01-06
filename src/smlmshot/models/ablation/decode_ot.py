# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt Decode to our optimal transport trainer.

It includes removing uncertainty prediction and providing a calibration pipeline.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from smlmshot import utils
from smlmshot.models.basics import UNet, ZNorm
from smlmshot.models.basics.unet_parts import Down, OutConv


class DECODEOT(nn.Module):
    """Adapt DECODE to our optimal transport training method."""

    _PHOTONS_CST = 1e3
    NEEDS_CALIBRATION = True

    ACT_LAYER = nn.SiLU
    NORM_LAYER = nn.BatchNorm2d

    def __init__(
        self, inner_dim: int, n_frames: int, pixel_size: Tensor, z_extent: Tensor
    ):
        super().__init__()
        assert n_frames % 2 == 1
        self.dim = inner_dim
        self.n_frames = n_frames
        self.register_buffer("_threshold", torch.tensor([0.5]))
        self.register_buffer("out_pixel_size", 2 * pixel_size)
        self.register_buffer("z_extent", z_extent)
        self.out_range_factor = 1.5
        self._enable_thresholding = True

        self.normalize = ZNorm(mu=0, sigma=self._PHOTONS_CST)
        self.frame_network = UNet(
            1,
            self.dim,
            depth=2,
            init_features=self.dim,
            act_layer=self.ACT_LAYER,
            norm_layer=self.NORM_LAYER,
        )
        self.core_network = UNet(
            n_frames * self.dim,
            self.dim,
            depth=2,
            init_features=self.dim,
            act_layer=self.ACT_LAYER,
            norm_layer=self.NORM_LAYER,
        )
        self.decoder_net = nn.Sequential(
            Down(self.dim, 2 * self.dim),
            OutConv(2 * self.dim, 5),
        )
        self.bg_net = OutConv(inner_dim, 1)

    @property
    def threshold(self) -> float:
        """Get the current threshold used during inference."""
        return self._threshold.item()

    @threshold.setter
    def threshold(self, value: float):
        """Set a new threshold for inference."""
        self._threshold.fill_(value)

    @property
    def enable_thresholding(self):
        """Return whether post processing is enabled."""
        return self._enable_thresholding

    @enable_thresholding.setter
    def enable_thresholding(self, value: bool):
        """Enable / disable the post processing step."""
        self._enable_thresholding = value

    def forward(self, y: Tensor) -> Tensor:  # noqa: D102
        bs, _, h, w = y.shape
        device = y.device

        x = self.normalize(y)
        x = x.view(bs * self.n_frames, 1, h, w)
        x = self.frame_network(x)
        x = x.view(bs, self.n_frames * self.dim, h, w)
        x = self.core_network(x)
        bg = self.bg_net(x)[:, 0]
        x = self.decoder_net(x)

        p = x[:, 0, None]
        xy = x[:, 1:3]
        z = x[:, 3, None]
        n = x[:, 4, None]

        xy_ref = utils.maths.generate_grid_centers(
            h=x.size(-2),
            w=x.size(-1),
            cell_width=self.out_pixel_size[0],
            cell_height=self.out_pixel_size[1],
            device=device,
        )
        xy = self.out_range_factor * self.out_pixel_size[:, None, None] * torch.tanh(xy)
        xy = xy_ref + xy

        z = (self.z_extent[1] - self.z_extent[0]) * torch.sigmoid(z) + self.z_extent[0]

        n = self._PHOTONS_CST * F.softplus(n)

        x = torch.cat([p, xy, z, n], dim=1)
        x = utils.torch.flatten_spatial(x)

        bg = self._PHOTONS_CST * F.softplus(bg)

        if self.training:
            return x, bg
        if self.enable_thresholding:
            x = self.perform_thresholding(x)
        return x

    def perform_thresholding(self, raw_x: Tensor):
        """Apply thresholding."""
        p = raw_x[..., 0].sigmoid()
        x = [x_[p_ > self.threshold, 1:] for p_, x_ in zip(p, raw_x)]
        return x

    def compute_valid_target_mask(self, x_gt: Tensor, h: int, w: int):
        """Return a [B, n_gt, n_pred] mask between predictions and ground-truths.

        It indicates for each ground truth which predictions are in range to predict it.
        """
        xy_grid_centers = utils.maths.generate_grid_centers(
            h=h // 2,
            w=w // 2,
            cell_width=self.out_pixel_size[0],
            cell_height=self.out_pixel_size[1],
            device=x_gt.device,
        )
        xy_grid_centers = utils.torch.flatten_spatial(xy_grid_centers[None])
        delta = xy_grid_centers[:, None, :] - x_gt[:, :, None, :2]
        delta = delta.square().sum(dim=-1)
        max_range = (0.5 * self.out_range_factor * self.out_pixel_size).square().sum()
        valid_target_mask = delta < max_range
        return valid_target_mask
