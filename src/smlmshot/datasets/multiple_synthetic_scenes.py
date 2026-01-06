# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset that returns one unique and independent synthetic scene per element.

A scene is made of fluorophores with dynamics and a background.
It is ready to be given to a simulator to generate a complete image.
"""

from math import ceil

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from smlmshot import simulation
from smlmshot.utils import extent, nested, random


class MultipleSyntheticScenesDataset(Dataset):
    """Generate one independent synthetic scene per requested sample."""

    def __init__(
        self,
        bg_photon_mean: float,
        bg_photon_std: float,
        length: int,
        n_acts_per_frame: float,
        n_frames: int,
        n_pixels: int,
        photon_flux_mean: float,
        photon_flux_std: float,
        pixel_size: Tensor,
        seed: int,
        time_bleach: float,
        time_off: float,
        time_on: float,
        z_extent: Tensor,
    ):
        super().__init__()
        if n_frames % 2 != 1:
            raise ValueError("n_frames must be odd")
        self.bg_photon_mean = bg_photon_mean
        self.bg_photon_std = bg_photon_std
        self.length = length
        self.n_frames = n_frames
        self.n_pixels = n_pixels
        self.photon_flux_mean = photon_flux_mean
        self.photon_flux_std = photon_flux_std
        self.seed = random.derive_new_seed(seed)
        self.time_bleach = time_bleach
        self.time_off = time_off
        self.time_on = time_on

        self.vol_extent = extent.get_vol_extent(
            h=n_pixels, w=n_pixels, pixel_size=pixel_size, z_extent=z_extent
        )

        self.tg_frame_idx = n_frames // 2
        self.n_periods_max = simulation.fluorophores.icdf_n_activations(
            1e-3, time_bleach=time_bleach, time_off=time_off
        )

        self.min_intensity = 0.01 * self.photon_flux_mean

        # We estimate a few constants based on some generated fluorophores
        gen = random.get_generator(seed=0)
        _, n_photons = simulation.fluorophores.sample_fluorophores(
            N=5000,
            vol_extent=self.vol_extent,
            time_on=self.time_on,
            time_off=self.time_off,
            time_bleach=self.time_bleach,
            n_periods_max=self.n_periods_max,
            n_frames=self.n_periods_max,
            photon_flux_mean=self.photon_flux_mean,
            photon_flux_std=self.photon_flux_std,
            gen=gen,
        )
        mean_n_frames_lifetime = (n_photons > 0.0).sum(dim=-1).float().mean(dim=0)
        self.n_warmup_frames = ceil(mean_n_frames_lifetime)
        self.total_n_frames = self.n_warmup_frames + self.n_frames
        self.n_fluos = ceil(
            n_acts_per_frame * self.total_n_frames / mean_n_frames_lifetime
        )

        n_photons = n_photons.flatten()
        n_photons = n_photons[n_photons > 0.0]
        self.significant_threshold = torch.quantile(n_photons, q=0.25)

    @classmethod
    def init_from_config(
        cls, length: int, seed: int, psf: simulation.psfs.CSplinesPSF, cfg: DictConfig
    ):
        """Initialize from a config, a psf and a seed."""
        return cls(
            bg_photon_mean=cfg.fluorescence.bg_photon_mean,
            bg_photon_std=cfg.fluorescence.bg_photon_std,
            length=length,
            n_acts_per_frame=cfg.fluorescence.n_acts_per_frame,
            n_frames=cfg.runtime.n_frames,
            n_pixels=cfg.runtime.n_pixels,
            photon_flux_mean=cfg.fluorescence.photon_flux_mean,
            photon_flux_std=cfg.fluorescence.photon_flux_std,
            pixel_size=cfg.psf.pixel_size,
            seed=seed,
            time_bleach=cfg.fluorescence.time_bleach,
            time_off=cfg.fluorescence.time_off,
            time_on=cfg.fluorescence.time_on,
            z_extent=psf.z_extent,
        )

    def increment_seed(self):
        """Update internal seed, so each epoch can produce different results."""
        self.seed = random.derive_new_seed(self.seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise StopIteration()

        seed = random.derive_new_seed(self.seed + idx + 1)
        gen = random.get_generator(seed)

        N = self.n_fluos
        if N > 10:
            N = torch.randint(low=10, high=N, size=(), generator=gen)

        xyz, n_photons = simulation.fluorophores.sample_fluorophores(
            N=N,
            vol_extent=self.vol_extent,
            time_on=self.time_on,
            time_off=self.time_off,
            time_bleach=self.time_bleach,
            n_periods_max=self.n_periods_max,
            n_frames=self.total_n_frames,
            photon_flux_mean=self.photon_flux_mean,
            photon_flux_std=self.photon_flux_std,
            gen=gen,
        )
        n_photons = n_photons[:, self.n_warmup_frames :]

        mask = n_photons.max(dim=-1).values > 0.0
        xyz, n_photons = xyz[mask], n_photons[mask]

        mask_tg = n_photons[:, self.tg_frame_idx] >= self.min_intensity
        xyz_tg, n_photons_tg = xyz[mask_tg], n_photons[mask_tg]
        significant_tg = n_photons[:, self.tg_frame_idx] >= self.significant_threshold

        bg = simulation.background.sample_perlin_background(
            h=self.n_pixels,
            w=self.n_pixels,
            mean=self.bg_photon_mean,
            std=self.bg_photon_std,
            gen=gen,
        )

        return {
            "xyz_all": xyz,
            "n_photons_all": n_photons,
            "xyz_tg": xyz_tg,
            "n_photons_tg": n_photons_tg,
            "significant_tg": significant_tg,
            "bg": bg,
        }

    def collate_fn(self, batch):
        """Pad elements of variable length."""
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        xyz_all = self._pad(batch_dict["xyz_all"])
        n_photons_all = self._pad(batch_dict["n_photons_all"])
        xyz_tg = self._pad(batch_dict["xyz_tg"])
        n_photons_tg = self._pad(batch_dict["n_photons_tg"])
        significant_tg = self._pad(batch_dict["significant_tg"])
        bg = torch.utils.data.default_collate(batch_dict["bg"])

        return {
            "xyz_all": xyz_all,
            "n_photons_all": n_photons_all,
            "xyz_tg": xyz_tg,
            "n_photons_tg": n_photons_tg,
            "significant_tg": significant_tg,
            "bg": bg,
        }

    def _pad(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Pad en element to n_fluos max length."""
        return nested.pad_sequence(x, target_len=self.n_fluos, returns_lengths=True)
