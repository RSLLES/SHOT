# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset that generates one scene of 'length' frames and iterates over it.

The key difference with MultipleSyntheticScene is that this dataset only contains
one unique scene (fluorophores and background) and is elements are frames over it.
"""

from math import ceil

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from smlmshot import simulation
from smlmshot.utils import extent, nested, periods, random


class OneSyntheticSceneDataset(Dataset):
    """Generate one unique synthetic scene for this dataset, and iterate over it."""

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
        self.n_fluos = ceil(n_acts_per_frame * self.length / mean_n_frames_lifetime)

        n_photons = n_photons.flatten()
        n_photons = n_photons[n_photons > 0.0]
        self.significant_threshold = torch.quantile(n_photons, q=0.25)

        self.sample_internal_scene()

    @classmethod
    def init_from_config(
        cls,
        length: int,
        seed: int,
        psf: simulation.psfs.CSplinesPSF,
        cfg: DictConfig,
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

    def get_max_n_acts(self):
        """Return the max number of activations in the sample."""
        return self.max_n_acts

    def sample_internal_scene(self):
        """Sample one scene given the current seed."""
        gen = random.get_generator(self.seed)

        self.xyz = simulation.fluorophores.sample_coordinates(
            self.n_fluos, vol_extent=self.vol_extent, gen=gen
        )
        self.d = simulation.fluorophores.sample_dynamics(
            self.n_fluos,
            time_bleach=self.time_bleach,
            time_off=self.time_off,
            time_on=self.time_on,
            n_periods_max=self.n_periods_max,
            gen=gen,
        )
        t0 = simulation.fluorophores.sample_init_time(
            self.n_fluos, length=self.length, gen=gen
        )
        self.d += t0[:, None, None]
        self.photon_flux = simulation.fluorophores.sample_photon_flux(
            self.n_fluos, mean=self.photon_flux_mean, std=self.photon_flux_std, gen=gen
        )

        self.bg = simulation.background.sample_perlin_background(
            h=self.n_pixels,
            w=self.n_pixels,
            mean=self.bg_photon_mean,
            std=self.bg_photon_std,
            gen=gen,
        )

        # not easy to get the max number of active fluorophores, so brute force...
        n_max_acts = [
            torch.sum(
                periods.batch_discretize_periods(self.d - i, n_bins=self.n_frames) > 0
            )
            for i in range(self.length)
        ]
        self.max_n_acts = max(n_max_acts)

    def increment_seed(self):
        """Update internal seed and sample a new scene."""
        self.seed = random.derive_new_seed(self.seed)
        self.sample_internal_scene()

    def __len__(self):
        return self.length + 1 - self.n_frames

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise StopIteration()

        dynamics = self.d - idx  # remove current time
        n_photons = periods.batch_discretize_periods(dynamics, n_bins=self.n_frames)
        n_photons = self.photon_flux[:, None] * n_photons

        mask = n_photons.max(dim=-1).values > self.min_intensity
        xyz = self.xyz[mask]
        n_photons = n_photons[mask]

        mask_tg = n_photons[:, self.tg_frame_idx] > self.min_intensity
        xyz_tg, n_photons_tg = xyz[mask_tg], n_photons[mask_tg]
        significant_tg = n_photons[:, self.tg_frame_idx] >= self.significant_threshold

        return {
            "xyz_all": xyz,
            "n_photons_all": n_photons,
            "xyz_tg": xyz_tg,
            "n_photons_tg": n_photons_tg,
            "significant_tg": significant_tg,
            "bg": self.bg,
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
        """Pad en element to n_fluos max number of activations."""
        return nested.pad_sequence(x, target_len=self.max_n_acts, returns_lengths=True)
