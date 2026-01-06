# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from smlmshot import simulation
from smlmshot.losses import OptimalTransportLossFunc
from smlmshot.models import Simulator
from smlmshot.utils import random


class OptimalTransportTrainer(nn.Module):
    """Train a model with our optimal transport loss function."""

    def __init__(
        self,
        camera: simulation.Camera,
        eps: float,
        jitter_std: float,
        model: nn.Module,
        n_frames: int,
        n_sinkhorn_iters: int,
        photon_flux_mean: float,
        pixel_size: Tensor,
        psf: simulation.psfs.CSplinesPSF,
        reg: float,
        seed: int,
    ):
        super().__init__()
        if n_frames % 2 != 1:
            raise ValueError("n_frames must be odd")
        self.n_frames = n_frames
        self.tg_frame_idx = n_frames // 2

        self.model = model
        self.simulator = Simulator(camera=camera, jitter_std=jitter_std, psf=psf)
        self.ot_loss = OptimalTransportLossFunc(
            photon_cst=photon_flux_mean,
            pixel_size=pixel_size,
            reg=reg,
            n_sinkhorn_iters=n_sinkhorn_iters,
        )
        self.register_buffer("seed", torch.tensor(seed, dtype=torch.int64))

    @classmethod
    def init_from_config(
        cls,
        camera: simulation.Camera,
        model: nn.Module,
        psf: simulation.psfs.CSplinesPSF,
        seed: int,
        cfg: DictConfig,
    ):
        """Initialize from a config."""
        return cls(
            camera=camera,
            eps=cfg.runtime.eps,
            jitter_std=cfg.ds_train.jitter_std,
            model=model,
            n_frames=cfg.runtime.n_frames,
            n_sinkhorn_iters=cfg.trainer.n_sinkhorn_iters,
            photon_flux_mean=cfg.fluorescence.photon_flux_mean,
            pixel_size=cfg.psf.pixel_size,
            psf=psf,
            reg=cfg.trainer.reg,
            seed=seed,
        )

    def forward(self, batch):  # noqa: D102
        xyz_all, _ = batch["xyz_all"]
        n_photons_all, _ = batch["n_photons_all"]
        bg_gt = batch["bg"]

        random.derive_new_seed_(self.seed)
        y = self.simulator(
            xyz=xyz_all, n_photons=n_photons_all, bg=bg_gt, seed=self.seed
        )

        xyz_gt, xyz_gt_length = batch["xyz_tg"]
        mask_gt = torch.arange(xyz_gt.size(1), device=xyz_gt.device)
        mask_gt = mask_gt < xyz_gt_length[:, None]
        n_photons_gt, _ = batch["n_photons_tg"]
        n_photons_gt = n_photons_gt[..., self.tg_frame_idx]

        x = self.model(y)
        valid_target_mask = self.model.compute_valid_target_mask(
            xyz_gt, img_size=(bg_gt.size(-2), bg_gt.size(-1))
        )

        losses_ot = []
        losses_bg = []
        X = x if isinstance(x, list) else [x]
        for s_logits, xyz, n_photons, bg in X:
            loss_ot = self.ot_loss(
                s_logits=s_logits,
                mask_gt=mask_gt,
                xyz=xyz,
                xyz_gt=xyz_gt,
                n_photons=n_photons,
                n_photons_gt=n_photons_gt,
                valid_target_mask=valid_target_mask,
            )
            losses_ot.append(loss_ot)
            loss_bg = torch.nn.functional.mse_loss(bg, bg_gt)
            losses_bg.append(loss_bg)
        loss_ot = torch.stack(losses_ot).mean()
        loss_bg = torch.stack(losses_bg).mean()
        loss = loss_ot + 1e-6 * loss_bg
        return {"loss": loss, "ot": loss_ot, "bg": loss_bg}
