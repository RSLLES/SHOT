# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulate camera model."""

from dataclasses import dataclass

import torch
from torch import Generator, Tensor


@dataclass
class Camera:
    """Convenient wrapper around Camera related functions."""

    adu_baseline: int
    inv_e_adu: float
    em_gain: float
    quantum_efficiency: float
    readout_noise: float
    spurious_charge: float
    type: str

    def apply_camera(self, y: Tensor, relaxed: bool, gen: Generator):
        """Simulate complete camera model."""
        if relaxed:
            y = camera_noise_relaxed(
                y,
                quantum_efficiency=self.quantum_efficiency,
                spurious_charge=self.spurious_charge,
                em_gain=self.em_gain,
                readout_noise=self.readout_noise,
                type=self.type,
                gen=gen,
            )
            y = digitalization_relaxed(
                y, inv_e_adu=self.inv_e_adu, adu_baseline=self.adu_baseline
            )
            return y
        y = camera_noise(
            y,
            quantum_efficiency=self.quantum_efficiency,
            spurious_charge=self.spurious_charge,
            em_gain=self.em_gain,
            readout_noise=self.readout_noise,
            type=self.type,
            gen=gen,
        )
        y = digitalization(y, inv_e_adu=self.inv_e_adu, adu_baseline=self.adu_baseline)
        return y

    def apply_camera_gain(self, y: Tensor):
        """Apply noise-less/expected value of the camera model."""
        y = camera_noise_gain(
            y,
            quantum_efficiency=self.quantum_efficiency,
            em_gain=self.em_gain,
            type=self.type,
        )
        y = digitalization_gain(
            y, inv_e_adu=self.inv_e_adu, adu_baseline=self.adu_baseline
        )
        return y

    def apply_reciprocal_camera_gain(self, y: Tensor):
        """Apply the inverted noise-less camera model."""
        y = reciprocal_digitalization_gain(
            y, inv_e_adu=self.inv_e_adu, adu_baseline=self.adu_baseline
        )
        y = reciprocal_camera_noise_gain(
            y,
            quantum_efficiency=self.quantum_efficiency,
            em_gain=self.em_gain,
            type=self.type,
        )
        return y


def camera_noise(
    y: Tensor,
    quantum_efficiency: float,
    spurious_charge: float,
    em_gain: float,
    readout_noise: float,
    type: str,
    gen: Generator,
) -> Tensor:
    """Apply shot noise, amplification noise for EMCCD, and read noise."""
    # shot noise
    y = quantum_efficiency * y + spurious_charge
    y.clamp_(min=0.0)  # prevents weird bugs with negative 0.0 ...
    y = torch.poisson(y, generator=gen)

    # amplification noise for EMCCD camera
    if type == "EMCCD":
        y = em_gain * torch._standard_gamma(y, generator=gen)

    # read noise
    eps = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=gen)
    y = y + readout_noise * eps
    return y


def camera_noise_relaxed(
    y: Tensor,
    quantum_efficiency: float,
    spurious_charge: float,
    em_gain: float,
    readout_noise: float,
    type: str,
    gen: Generator,
    eps: float = 1e-6,
) -> Tensor:
    """Differentiable version of camera_noise.

    Reparameterization enables sampling amplification noise (Gamma distribution) and
    read noise (Normal distribution). Shot noise (Poisson distribution) is replaced by a
    Gamma distribution with rate 1.
    """
    # shot noise
    y = quantum_efficiency * y + spurious_charge
    # Poisson distribution is approximated by a Gamma distribution with rate 1
    y.clip_(min=eps)
    y = torch._standard_gamma(y, generator=gen)

    # amplification noise for EMCCD camera
    if type == "EMCCD":
        y = y.clip(min=eps)
        y = em_gain * torch._standard_gamma(y, generator=gen)

    # read noise
    eps = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=gen)
    y = y + readout_noise * eps
    return y


def camera_noise_gain(
    y: Tensor, quantum_efficiency: float, em_gain: float, type: str
) -> Tensor:
    """Apply camera gain."""
    if type == "EMCCD":
        return em_gain * quantum_efficiency * y
    if type == "sCMOS":
        return quantum_efficiency * y
    raise ValueError(f"Supported type are EMCCD or sCMOS; found '{type}'")


def reciprocal_camera_noise_gain(
    y: Tensor, quantum_efficiency: float, em_gain: float, type: str
) -> Tensor:
    """Apply reciprocal camera gain."""
    if type == "EMCCD":
        return y / em_gain / quantum_efficiency
    if type == "sCMOS":
        return y / quantum_efficiency
    raise ValueError(f"Supported type are EMCCD or sCMOS; found '{type}'")


def digitalization(y: Tensor, inv_e_adu: float, adu_baseline: float) -> Tensor:
    """Convert analog signal to digital units."""
    y = inv_e_adu * y + adu_baseline
    y = y.floor()
    return y


def digitalization_relaxed(y: Tensor, inv_e_adu: float, adu_baseline: float) -> Tensor:
    """Convert analog signal to quasi-digital units (without the floor operation)."""
    return digitalization_gain(y, inv_e_adu=inv_e_adu, adu_baseline=adu_baseline)


def digitalization_gain(y: Tensor, inv_e_adu: float, adu_baseline: float) -> Tensor:
    """Convert analog signal to quasi-digital units (without the floor operation)."""
    return inv_e_adu * y + adu_baseline - 0.5


def reciprocal_digitalization_gain(
    y: Tensor, inv_e_adu: float, adu_baseline: float
) -> Tensor:
    """Convert digital signal to analog signal."""
    return (y - adu_baseline + 0.5) / inv_e_adu
