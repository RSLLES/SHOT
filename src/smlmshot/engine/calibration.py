# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning_fabric import Fabric
from scipy.optimize import minimize_scalar
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from smlmshot.metrics import EPFLChallengeMetrics
from smlmshot.utils import nested


@torch.no_grad()
def calibrate(
    fabric: Fabric,
    dl: DataLoader,
    model: nn.Module,
    watched_metric: str,
    max_evaluations: int = 500,
):
    """Find a threshold (over detection scores) that maximize watched_metric."""
    model.eval()
    model.enable_thresholding = False

    all_s_logits, all_xyz, all_n_photons = [], [], []
    all_xyz_gt, all_significant_gt = [], []
    for batch in tqdm(
        dl, leave=False, desc="precompute", disable=not fabric.is_global_zero
    ):
        y = batch["y"]
        s_logits, xyz, n_photons, _ = model(y)
        all_s_logits.append(s_logits)
        all_xyz.append(xyz)
        all_n_photons.append(n_photons)

        xyz_gt, lengths = batch["xyz_tg"]
        significant_gt, _ = batch["significant_tg"]
        xyz_gt = nested.expand_to_list(xyz_gt, lengths=lengths)
        significant_gt = nested.expand_to_list(significant_gt, lengths=lengths)

        all_xyz_gt.extend(xyz_gt)
        all_significant_gt.extend(significant_gt)

    all_s_logits = torch.cat(all_s_logits, dim=0)  # stored as one large tensor
    all_xyz = torch.cat(all_xyz, dim=0)
    all_n_photons = torch.cat(all_n_photons, dim=0)

    threshold = None
    with tqdm(
        leave=False, disable=not fabric.is_global_zero, desc="search for threshold"
    ) as pbar:

        def fun(t):
            metric = EPFLChallengeMetrics().to(device=fabric.device)
            model.threshold = t.item()
            xyz, _ = model.perform_thresholding(
                s_logits=all_s_logits, xyz=all_xyz, n_photons=all_n_photons
            )
            metric.update(xyz, all_xyz_gt, s=all_significant_gt)
            r = metric.compute()
            pbar.update(1)
            return -r[watched_metric].item()

        res = minimize_scalar(
            fun,
            bounds=(0.0, 1.0),
            options={"maxiter": max_evaluations, "xatol": 1e-2},
        )
        threshold = res.x
        pbar.close()

    threshold = fabric.broadcast(threshold, src=0)  # make sure it propagates
    model.threshold = threshold
    model.enable_thresholding = True
