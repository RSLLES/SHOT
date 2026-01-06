# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptation of our network to fit Decode's loss function."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from smlmshot import utils
from smlmshot.models.basics import LayerNormChannelFirst, UNet, ZNorm
from smlmshot.models.basics.unet_parts import OutConv
from smlmshot.models.decode import non_maximum_suppression
from smlmshot.models.simulator import Renderer


class SHOTDECODE(nn.Module):
    """SHOT adapted for DECODE's loss function."""

    NEEDS_CALIBRATION = False
    _PHOTONS_CST = 1e3

    ACT_LAYER = nn.SiLU
    NORM_LAYER = LayerNormChannelFirst

    def __init__(
        self,
        adu_baseline: float,
        camera_type: str,
        e_adu: float,
        em_gain: float,
        inner_dim: int,
        n_frames: int,
        n_iters: int,
        psf_center: Tensor,
        psf: Tensor,
        quantum_efficiency: float,
        voxel_size: Tensor,
        z_extent: Tensor,
    ):
        super().__init__()
        assert n_frames % 2 == 1
        self.x0 = None
        self.dim = inner_dim
        self.n_frames = n_frames
        self.center_frame_idx = n_frames // 2
        self.n_iters = n_iters
        self.out_dim = 9
        self.register_buffer("out_pixel_size", voxel_size[:2])
        self.register_buffer("z_extent", z_extent)
        self.out_range_factor = 1.5

        self.normalize = ZNorm(mu=0, sigma=self._PHOTONS_CST)
        self.encoder_net = UNet(
            1,
            inner_dim,
            depth=2,
            init_features=48,
            norm_layer=self.NORM_LAYER,
            act_layer=self.ACT_LAYER,
        )
        self.refinement_net = UNet(
            inner_dim * (n_frames + 2),
            inner_dim,
            depth=2,
            init_features=48,
            norm_layer=self.NORM_LAYER,
            act_layer=self.ACT_LAYER,
        )
        self.decoder_net = UNet(
            inner_dim,
            self.out_dim,
            depth=2,
            init_features=48,
            norm_layer=self.NORM_LAYER,
            act_layer=self.ACT_LAYER,
        )
        self.bg_net = OutConv(inner_dim, 1)

        self.psf = psf
        self.renderer = Renderer(
            psf=self.psf,
            psf_center=psf_center,
            voxel_size=voxel_size,
            quantum_efficiency=quantum_efficiency,
            em_gain=em_gain,
            adu_baseline=adu_baseline,
            e_adu=e_adu,
            camera_type=camera_type,
        )

    def forward(self, y0: Tensor) -> Tensor:  # noqa: D102
        z0 = self.encode(y0)
        idx = self.center_frame_idx
        z = z0[:, idx * self.dim : (idx + 1) * self.dim]
        x_hat, bg_hat = self.decode(z)
        if self.training:
            X = [(x_hat, bg_hat)]
        for i in range(self.n_iters):
            y_hat = self.render(x_hat, bg=bg_hat)
            z_hat = self.encode(y_hat)
            z = z + self.refinement(z0=z0, z_hat=z_hat, z=z)
            x_hat, bg_hat = self.decode(z)
            if self.training:
                X.append((x_hat, bg_hat))
        if self.training:
            return X
        x_hat = self.post_processing(x_hat, h=y0.size(-2), w=y0.size(-1))
        return x_hat

    def encode(self, y: Tensor) -> Tensor:
        """Encode an image to a latent representation."""
        assert y.ndim == 4
        bs, f, h, w = y.shape

        y = self.normalize(y)
        y = y.view(bs * f, 1, h, w)
        z = self.encoder_net(y)
        z = z.view(bs, self.dim * f, h, w)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode an image to a set of activations."""
        device = z.device

        bg = self.bg_net(z)
        bg = bg.squeeze(1)
        bg = self._PHOTONS_CST * F.softplus(bg)

        x = self.decoder_net(z)

        p = x[:, 0, None]
        xy = x[:, 1:3]
        z = x[:, 3, None]
        n = x[:, 4, None]
        uxy = x[:, 5:7]
        uz = x[:, 7, None]
        un = x[:, 8, None]

        p = torch.sigmoid(p)

        xy_ref = utils.maths.generate_grid_centers(
            h=x.size(-2),
            w=x.size(-1),
            cell_width=self.out_pixel_size[0],
            cell_height=self.out_pixel_size[1],
            device=device,
        )
        xy = self.out_range_factor * self.out_pixel_size[:, None, None] * torch.tanh(xy)
        xy = xy_ref + xy
        uxy = self.out_pixel_size[:, None, None] * F.softplus(uxy) + 5.0

        z = (self.z_extent[1] - self.z_extent[0]) * torch.sigmoid(z) + self.z_extent[0]
        uz = self.z_extent[1] * F.softplus(uz) + 5.0

        n = self._PHOTONS_CST * F.softplus(n)
        un = self._PHOTONS_CST * F.softplus(un) + 5.0

        x = torch.cat([p, xy, z, n, uxy, uz, un], dim=1)
        x = utils.torch.flatten_spatial(x)

        return x, bg

    def refinement(self, z0: Tensor, z_hat: Tensor, z: Tensor) -> Tensor:
        """Refinement network."""
        input = torch.cat([z0, z_hat, z], dim=1)
        dz = self.refinement_net(input)
        z = z + dz
        return z

    def render(self, x: Tensor, bg: Tensor) -> Tensor:
        """Render the resulting image given fluorophores and a background."""
        xyz = x[..., 1:4]
        n = x[..., 0, None] * x[..., 4, None]  # n <- p * n
        x = torch.cat([xyz, n], dim=-1)
        y = self.renderer(x, bg=bg)
        return y

    def post_processing(self, x: Tensor, h: int, w: int):
        """Apply non maximum suppression and thresholding."""
        p, x = x[..., 0], x[..., 1:]
        p = utils.torch.unflatten_spatial(p[..., None], h=h, w=w).squeeze(1)
        p = non_maximum_suppression(p, raw_th=0.1, split_th=0.6)
        p = utils.torch.flatten_spatial(p[:, None]).squeeze(-1)
        x = [x_[p_ >= 0.4] for p_, x_ in zip(p, x)]
        return x
