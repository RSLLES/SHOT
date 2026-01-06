# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from smlmshot.writers import WriterInterface


@torch.no_grad()
def export(
    fabric: Fabric, dl: DataLoader, model: torch.nn.Module, writer: WriterInterface
):
    """Compute validation metrics with an added offset to spatial coordinates."""
    model.eval()
    if fabric.world_size > 1:
        raise ValueError("Export only works with one gpu.")

    frame_idx = 1

    with writer:
        for batch in tqdm(dl, desc="export", disable=not fabric.is_global_zero):
            y = batch["y"]
            xyz, n_photons, _ = model(y)

            for xyz_, n_photons_, y_ in zip(xyz, n_photons, y):
                frame = torch.full_like(n_photons_, frame_idx)
                data = torch.cat([frame[:, None], xyz_, n_photons_[:, None]], dim=-1)
                writer.write(data)
                frame_idx += 1
