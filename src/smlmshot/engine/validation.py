# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning_fabric import Fabric
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from smlmshot.metrics import EPFLChallengeMetrics
from smlmshot.utils import nested


@torch.no_grad()
def validate(fabric: Fabric, dl: DataLoader, model: nn.Module):
    """Compute validation metrics after performing wobble correction."""
    metrics = _compute_metrics(
        fabric=fabric,
        dl=dl,
        model=model,
        offset=0.0,
        pbar="wobble correction",
    )
    offset = metrics["offset"]
    metrics = _compute_metrics(
        fabric,
        dl=dl,
        model=model,
        offset=offset,
        pbar="validation",
    )
    metrics["wobble_corr"] = offset
    return metrics


def _compute_metrics(
    fabric: Fabric, dl: DataLoader, model: nn.Module, offset: Tensor | float, pbar: str
):
    """Compute validation metrics with an added offset to spatial coordinates."""
    model.eval()
    metrics = EPFLChallengeMetrics().to(device=fabric.device)
    for batch in tqdm(dl, leave=False, desc=pbar, disable=not fabric.is_global_zero):
        y = batch["y"]
        xyz_gt, lengths = batch["xyz_tg"]
        significant_gt, _ = batch["significant_tg"]
        xyz_gt = nested.expand_to_list(xyz_gt, lengths=lengths)
        significant_gt = nested.expand_to_list(significant_gt, lengths=lengths)

        xyz, n_photons, _ = model(y)
        [e[..., :3].add_(offset) for e in xyz]

        metrics.update(xyz, xyz_gt, s=significant_gt)

    metrics = metrics.compute()
    return metrics
