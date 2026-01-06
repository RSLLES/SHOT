import torch
from torch import Generator, Tensor


def sample_photon_flux(N: int, mean: float, std: float, gen: Generator) -> Tensor:
    """Sample N photons flux normally distributed."""
    eps = torch.randn((N,), generator=gen, device=gen.device)
    photon_flux = mean + std * eps
    photon_flux.clamp_(min=1e-2 * mean)  # prevents negative
    return photon_flux
