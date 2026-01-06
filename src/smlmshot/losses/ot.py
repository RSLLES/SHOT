# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

from smlmshot import utils


def sinkhorn(a: Tensor, b: Tensor, M: Tensor, reg: float, n_iters: int):
    """Implement Sinkhorn algorithm in log space."""
    assert a.ndim == 2 and b.ndim == 2 and M.ndim == 3

    log_a = torch.log(a)
    log_b = torch.log(b)
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    M = M / (-reg)

    for _ in range(n_iters):
        v = log_b - torch.logsumexp(M + u[:, :, None], dim=-2)
        u = log_a - torch.logsumexp(M + v[:, None, :], dim=-1)

    logP = u[:, :, None] + M + v[:, None, :]
    return torch.exp(logP)


def optimal_transport_loss(
    s_logits: Tensor,
    mask_gt: Tensor,
    x: Tensor,
    x_gt: Tensor,
    log_sigma: Tensor,
    valid_target_mask: Tensor,
    reg: float,
    n_sinkhorn_iters: int,
):
    """Return our optimal transport loss as described in our paper."""
    B, N, _ = x.size()
    device, dtype = x.device, x.dtype

    atom_w = 1 / N
    b = torch.full((B, N), atom_w, device=device, dtype=dtype)
    a = torch.where(mask_gt, atom_w, 0.0)
    bg = 1.0 - a.sum(1, keepdim=True)
    a = torch.cat([a, bg], dim=1)

    # p loss function is a logit BCE
    s_logits = s_logits[:, None]
    lp1 = torch.nn.functional.softplus(-s_logits, beta=1.0)
    lp0 = s_logits + lp1

    # distance loss function is a normal distribution with learnt uncertainties
    delta = x[:, None, :] - x_gt[:, :, None]
    delta = delta / torch.exp(log_sigma)
    delta = torch.square(delta)
    delta = 0.5 * delta + log_sigma

    loss = torch.sum(delta, dim=-1) + lp1
    reg = reg * loss.median()
    inf = 10.0 * loss.max()
    cost = torch.where(valid_target_mask, loss, inf)
    loss = torch.cat([loss, lp0], dim=1)
    cost = torch.cat([cost, lp0], dim=1)

    # K = hungarian(M=cost, mask_a=a)
    K = sinkhorn(a=a, b=b, M=cost, reg=reg, n_iters=n_sinkhorn_iters)
    loss = (loss * K).view(B, -1).sum(-1)
    loss = loss.mean()
    return loss


class OptimalTransportLossFunc(nn.Module):
    """Torch module wrapping around our optimal transport loss."""

    def __init__(
        self,
        n_sinkhorn_iters: int,
        photon_cst: float,
        pixel_size: Tensor,
        reg: float,
    ):
        super().__init__()
        self.n_sinkhorn_iters = n_sinkhorn_iters
        photon_cst = torch.as_tensor(photon_cst)
        self.register_buffer("r_normalize_n_photons", 1.0 / photon_cst)
        pixel_size = utils.torch.to_pair(pixel_size)
        pixel_size = torch.as_tensor(pixel_size)
        z_chara_scale = 2 * pixel_size.max()
        normalize_xyz = torch.cat([pixel_size, z_chara_scale[None]], dim=0)
        self.register_buffer("r_normalize_xyz", 1.0 / normalize_xyz)
        self.reg = reg

        self.log_sigma = torch.nn.Parameter(torch.zeros((4,)))

    def forward(  # noqa: D102
        self,
        s_logits: Tensor,
        mask_gt: Tensor,
        xyz: Tensor,
        xyz_gt: Tensor,
        n_photons: Tensor,
        n_photons_gt: Tensor,
        valid_target_mask: Tensor,
    ):
        xyz = self.r_normalize_xyz * xyz
        xyz_gt = self.r_normalize_xyz * xyz_gt
        n_photons = self.r_normalize_n_photons * n_photons
        n_photons_gt = self.r_normalize_n_photons * n_photons_gt

        x = torch.cat([xyz, n_photons[..., None]], dim=-1)
        x_gt = torch.cat([xyz_gt, n_photons_gt[..., None]], dim=-1)

        return optimal_transport_loss(
            s_logits=s_logits,
            mask_gt=mask_gt,
            x=x,
            x_gt=x_gt,
            valid_target_mask=valid_target_mask,
            log_sigma=self.log_sigma,
            reg=self.reg,
            n_sinkhorn_iters=self.n_sinkhorn_iters,
        )
