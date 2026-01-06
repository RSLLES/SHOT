# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wrapper-models around the simulation tools to generate SMLM acquisitions.

Simulation model simulate with sampled noise, while the Renderer model use
the expected value of the camera model.
"""

from copy import deepcopy

import torch
from torch import Generator, Tensor, nn

from smlmshot import simulation, utils


class Simulator(nn.Module):
    """Sample an SMLM acquisition given fluorophores.

    It can add jitter to the camera parameters.
    """

    def __init__(
        self,
        camera: simulation.Camera,
        jitter_std: float,
        psf: simulation.psfs.CSplinesPSF,
    ):
        super().__init__()
        self.camera = camera
        self.jitter_std = jitter_std
        self.psf = deepcopy(psf)

    def forward(  # noqa: D102
        self, xyz: Tensor, n_photons: Tensor, bg: Tensor, seed: int
    ) -> Tensor:
        dtype = xyz.dtype
        B, H, W = bg.size()
        gen = utils.random.get_generator(seed)

        # data augmentation
        em_gain = self.jitter(self.camera.em_gain, batch_size=B, gen=gen, dtype=dtype)
        quantum_efficiency = self.jitter(
            self.camera.quantum_efficiency, batch_size=B, gen=gen, dtype=dtype
        )
        readout_noise = self.jitter(
            self.camera.readout_noise, batch_size=B, gen=gen, dtype=dtype
        )
        spurious_charge = self.jitter(
            self.camera.spurious_charge, batch_size=B, gen=gen, dtype=dtype
        )
        adu_baseline = self.jitter(
            self.camera.adu_baseline, batch_size=B, gen=gen, dtype=dtype
        )
        inv_e_adu = self.jitter(
            self.camera.inv_e_adu, batch_size=B, gen=gen, dtype=dtype
        )

        y = self.psf.batched_render_fluorophores(
            xyz=xyz, n_photons=n_photons, img_size=(H, W)
        )
        y = y + bg[:, None]
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
        return y

    def jitter(self, value: float, batch_size: int, gen: Generator, dtype=None):
        """Jitter value with multiplicative noise."""
        value = torch.as_tensor(value)
        value = value.to(dtype=dtype, device=gen.device)
        value = value.expand(batch_size, 1, 1, 1)
        return utils.random.multiplicative_noise(value, std=self.jitter_std, gen=gen)


class Renderer(nn.Module):
    """Generate the SMLM acquisition using the expected value of the camera model."""

    def __init__(self, camera: simulation.Camera, psf: simulation.psfs.CSplinesPSF):
        super().__init__()
        self.camera = camera
        self.psf = deepcopy(psf)

    def forward(self, xyz: Tensor, n_photons: Tensor, bg: Tensor) -> Tensor:  # noqa: D102
        _, H, W = bg.size()
        y = self.psf.batched_render_fluorophores(
            xyz=xyz, n_photons=n_photons, img_size=(H, W)
        )
        y = y + bg[:, None]
        y = self.camera.apply_camera_gain(y)
        return y
