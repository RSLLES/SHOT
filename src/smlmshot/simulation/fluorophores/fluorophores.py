from torch import Generator, Tensor

from .coordinates import sample_coordinates
from .dynamics import sample_emissions
from .photon_flux import sample_photon_flux


def sample_fluorophores(
    N: int | Tensor,
    time_on: float,
    time_off: float,
    time_bleach: float,
    n_periods_max: int,
    n_frames: int,
    photon_flux_mean: float,
    photon_flux_std: float,
    vol_extent: Tensor,
    gen: Generator,
) -> tuple[Tensor, Tensor]:
    """Sample N fluorophores, i.e N 3D coordinates and dynamics for n_frames."""
    xyz = sample_coordinates(N, vol_extent=vol_extent, gen=gen)
    n_photons_pecent = sample_emissions(
        N=N,
        time_on=time_on,
        time_off=time_off,
        time_bleach=time_bleach,
        n_periods_max=n_periods_max,
        n_frames=n_frames,
        gen=gen,
    )
    photon_flux = sample_photon_flux(
        N, mean=photon_flux_mean, std=photon_flux_std, gen=gen
    )
    n_photons = n_photons_pecent * photon_flux[:, None]
    return xyz, n_photons
