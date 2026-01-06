# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lite version of SHOT, that does not include our iterative architecture scheme."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from smlmshot import simulation, utils

from .basics import LayerNormChannelFirst, UNet, ZNorm
from .basics.unet_parts import Down, OutConv


class SHOTLite(nn.Module):
    """Version of SHOT without the iterative architecture."""

    _PHOTONS_CST = 1e3
    _RANGE_FACTOR = 1.5

    _ACT_LAYER = nn.SiLU
    _NORM_LAYER = LayerNormChannelFirst

    def __init__(
        self,
        inner_dim: int,
        n_frames: int,
        pixel_size: Tensor,
        psf: simulation.psfs.CSplinesPSF,
    ):
        super().__init__()
        if n_frames % 2 == 0:
            raise ValueError("n_frames must be odd.")

        self.dim = inner_dim
        self.n_frames = n_frames
        pixel_size = utils.torch.to_pair(pixel_size)
        pixel_size = torch.as_tensor(pixel_size)
        self.register_buffer("out_pixel_size", 2.0 * pixel_size)

        self.out_dim = 5
        self.center_frame_idx = n_frames // 2
        self.register_buffer("z_extent", psf.z_extent)
        self._enable_thresholding = True
        self.register_buffer("_threshold", torch.tensor([0.5]))

        self.normalize = ZNorm(mu=0, sigma=self._PHOTONS_CST)
        self.frame_network = UNet(
            1,
            self.dim,
            depth=2,
            init_features=self.dim,
            act_layer=self._ACT_LAYER,
            norm_layer=self._NORM_LAYER,
        )
        self.core_network = UNet(
            n_frames * self.dim,
            self.dim,
            depth=2,
            init_features=self.dim,
            act_layer=self._ACT_LAYER,
            norm_layer=self._NORM_LAYER,
        )
        self.decoder_net = nn.Sequential(
            Down(
                inner_dim,
                2 * inner_dim,
                act_layer=self._ACT_LAYER,
                norm_layer=self._NORM_LAYER,
            ),
            OutConv(2 * inner_dim, self.out_dim),
        )
        self.bg_net = OutConv(inner_dim, 1)
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

        s_logits = x[:, 0, None]
        xy = x[:, 1:3]
        z = x[:, 3, None]
        n_photons = x[:, 4, None]

        xy_grid_center = utils.maths.generate_grid_centers(
            h=x.size(-2),
            w=x.size(-1),
            cell_width=self.out_pixel_size[0],
            cell_height=self.out_pixel_size[1],
            device=device,
        )
        xy = self._RANGE_FACTOR * self.out_pixel_size[:, None, None] * torch.tanh(xy)
        xy = xy_grid_center + xy
        z = (self.z_extent[1] - self.z_extent[0]) * torch.sigmoid(z) + self.z_extent[0]
        xyz = torch.cat([xy, z], dim=1)

        n_photons = self._PHOTONS_CST * F.softplus(n_photons)

        s_logits = utils.torch.flatten_spatial(s_logits).squeeze(-1)
        xyz = utils.torch.flatten_spatial(xyz)
        n_photons = utils.torch.flatten_spatial(n_photons).squeeze(-1)

        if self.training or not self.enable_thresholding:
            return s_logits, xyz, n_photons, bg
        xyz, n_photons = self.perform_thresholding(s_logits, xyz, n_photons)
        return xyz, n_photons, bg

    def perform_thresholding(self, s_logits: Tensor, xyz: Tensor, n_photons: Tensor):
        """Apply thresholding."""
        mask = s_logits.sigmoid() > self.threshold
        xyz = [e[m] for e, m in zip(xyz, mask)]
        n_photons = [e[m] for e, m in zip(n_photons, mask)]
        return xyz, n_photons

    def compute_valid_target_mask(self, xyz_gt: Tensor, img_size):
        """Return a [B, n_gt, n_pred] mask between predictions and ground-truths.

        It indicates for each ground truth which predictions are in range to predict it.
        """
        xy_grid_centers = utils.maths.generate_grid_centers(
            h=img_size[0] // 2,
            w=img_size[1] // 2,
            cell_width=self.out_pixel_size[0],
            cell_height=self.out_pixel_size[1],
            device=xyz_gt.device,
        )
        xy_grid_centers = utils.torch.flatten_spatial(xy_grid_centers[None])
        delta = xy_grid_centers[:, None, :] - xyz_gt[:, :, None, :2]
        delta = delta.square()
        max_range = (0.5 * self._RANGE_FACTOR * self.out_pixel_size).square()
        valid_target_mask = (delta < max_range).all(dim=-1)
        return valid_target_mask
