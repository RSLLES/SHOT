# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

from smlmshot.losses import BernoulliLoss, GaussianMixtureModelLoss
from smlmshot.models import Simulator
from smlmshot.utils import random


class DecodeTrainer(nn.Module):
    """Re-implementation of Decode original training module."""

    def __init__(
        self,
        adu_baseline: float,
        camera_type: str,
        e_adu: float,
        em_gain: float,
        eps: float,
        jitter_std: float,
        model: nn.Module,
        n_frames: int,
        psf_center: Tensor,
        psf_voxel_size: Tensor,
        psf: Tensor,
        quantum_efficiency: float,
        readout_noise: float,
        seed: int,
        spurious_charge: float,
    ):
        super().__init__()
        if n_frames % 2 != 1:
            raise ValueError("n_frames must be odd")
        self.n_frames = n_frames
        self.tg_frame_idx = n_frames // 2

        self.model = model
        inv_voxel_size = psf_voxel_size.reciprocal()
        self.simulator = Simulator(
            adu_baseline=adu_baseline,
            e_adu=e_adu,
            em_gain=em_gain,
            inv_voxel_size=inv_voxel_size,
            psf_center=psf_center,
            psf=psf,
            quantum_efficiency=quantum_efficiency,
            readout_noise=readout_noise,
            spurious_charge=spurious_charge,
            camera_type=camera_type,
            jitter_std=jitter_std,
        )

        self.ber = BernoulliLoss(eps=eps)
        self.gmm = GaussianMixtureModelLoss(eps=eps)
        self.register_buffer("seed", torch.tensor(seed, dtype=torch.int64))

    def forward(self, batch):  # noqa: D102
        x_all_frames, _ = batch["x_all"]
        x_gt, x_gt_lengths = batch["x"]
        bg_gt = batch["bg"]
        device = x_gt.device

        random.derive_new_seed_(self.seed)
        y = self.simulator(x_all_frames, bg=bg_gt, seed=self.seed)

        mask_gt = torch.arange(x_gt.size(1), device=device)
        mask_gt = mask_gt < x_gt_lengths[:, None]
        x_gt = x_gt[..., [0, 1, 2, 3 + self.tg_frame_idx]]

        x = self.model(y)
        losses_gmm = []
        losses_p = []
        losses_bg = []
        X = x if isinstance(x, list) else [x]
        for x, bg in X:
            p, x = x[..., 0], x[..., 1:]
            d = x.size(-1) // 2
            x, u = x[..., :d], x[..., d:]
            loss_gmm = self.gmm(p=p, x=x, u=u, x_gt=x_gt, mask_gt=mask_gt)
            losses_gmm.append(loss_gmm)
            loss_p = self.ber(p=p, n_gt=x_gt_lengths)
            losses_p.append(loss_p)
            loss_bg = torch.nn.functional.mse_loss(bg, bg_gt)
            losses_bg.append(loss_bg)

        loss_gmm = torch.stack(losses_gmm).mean()
        loss_p = torch.stack(losses_p).mean()
        loss_bg = torch.stack(losses_bg).mean()
        loss = 1e2 * loss_gmm + loss_p + 1e-3 * loss_bg
        return {"loss": loss, "gmm": loss_gmm, "p": loss_p, "bg": loss_bg}
