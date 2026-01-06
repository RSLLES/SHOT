# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate an SMLM image from a scene."""

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset, default_collate

from smlmshot import simulation
from smlmshot.utils import extent, nested, random


class SimulateSceneDataset(Dataset):
    """Turn a scene with fluorophores and background into an SMLM image."""

    def __init__(
        self,
        ds: Dataset,
        camera: simulation.Camera,
        jitter_std: float,
        psf: simulation.psfs.CSplinesPSF,
        seed: int,
    ):
        super().__init__()
        self.ds = ds
        self.camera = camera
        self.jitter_std = jitter_std
        self.psf = psf
        self.seed = random.derive_new_seed(seed)

    def increment_seed(self):
        """Update internal seed, to produce different results for the same element."""
        self.seed = random.derive_new_seed(self.seed)
        self.ds.increment_seed()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise StopIteration()
        scene = self.ds[idx]

        seed = random.derive_new_seed(self.seed + idx + 1)
        gen = random.get_generator(seed)
        em_gain = self.jitter(self.camera.em_gain, gen=gen)
        quantum_efficiency = self.jitter(self.camera.quantum_efficiency, gen=gen)
        readout_noise = self.jitter(self.camera.readout_noise, gen=gen)
        spurious_charge = self.jitter(self.camera.spurious_charge, gen=gen)
        adu_baseline = self.jitter(self.camera.adu_baseline, gen=gen)
        inv_e_adu = self.jitter(self.camera.inv_e_adu, gen=gen)

        xyz = scene["xyz_all"]
        n_photons = scene["n_photons_all"]
        bg = scene["bg"]
        H, W = bg.shape

        y = self.psf.batched_render_fluorophores(
            xyz=xyz[None], n_photons=n_photons[None], img_size=(H, W)
        )
        y = y.squeeze(0)
        y = y + bg[None]
        y = simulation.camera.camera_noise(
            y,
            em_gain=em_gain,
            quantum_efficiency=quantum_efficiency,
            readout_noise=readout_noise,
            spurious_charge=spurious_charge,
            type=self.camera.type,
            gen=gen,
        )
        y = simulation.camera.digitalization(
            y, inv_e_adu=inv_e_adu, adu_baseline=adu_baseline
        )
        scene["y"] = y
        return scene

    def collate_fn(self, batch):
        """Pad elements of variable length."""
        collated_batch = self.ds.collate_fn(batch)
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
        y = default_collate(batch_dict["y"])
        collated_batch["y"] = y
        return collated_batch

    def jitter(self, value: float, gen: Generator, dtype=None):
        """Jitter value with multiplicative noise."""
        value = torch.as_tensor(value)
        value = value.to(dtype=dtype, device=gen.device)
        return random.multiplicative_noise(value, std=self.jitter_std, gen=gen)
