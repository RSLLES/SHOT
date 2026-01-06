# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tools to simulate fluorophore dynamic (their activation pattern over time).

We assume initial activation happens at a uniformly sampled time t0.
Once ON, fluorophores follow a three-state Markov chain process,
alternating between ON and OFF states until a final BLEACHED state is reached.
Fluorophore dynamics are implemented a sequences of time intervals (called periods)
corresponding to the intervals it spends on state ON (i.e emitting photons).

Relaxed version used the concrete distribution, see The Concrete Distribution: A
Continuous Relaxation of Discrete Random Variables (Maddison et al., 2017).
Implementation comes from
https://docs.pytorch.org/docs/stable/distributions.html#relaxedbernoulli
"""

from math import ceil, log

import torch
from torch import Generator, Tensor

from smlmshot.utils.periods import batch_discretize_periods


def sample_emissions(
    N: int,
    time_on: float,
    time_off: float,
    time_bleach: float,
    n_periods_max: int,
    n_frames: int,
    gen: Generator,
    relaxed: bool = False,
    temperature: float = 0.5,
    dtype=None,
    eps: float = 1e-9,
):
    """Return (N, n_frames) tensor of per-frame ON-time fractions."""
    device = gen.device
    dynamics = sample_dynamics(
        N,
        time_bleach=time_bleach,
        time_off=time_off,
        time_on=time_on,
        n_periods_max=n_periods_max,
        relaxed=relaxed,
        temperature=temperature,
        dtype=dtype,
        gen=gen,
        eps=eps,
    )
    t0 = sample_init_time(N, length=n_frames, gen=gen, device=device, dtype=dtype)
    dynamics = dynamics + t0[:, None, None]
    n_photons = batch_discretize_periods(dynamics, n_bins=n_frames)
    return n_photons


def sample_init_time(N: int, length: int, gen: Generator, dtype=None, device=None):
    """Sample the first activation time uniformly in [0, length]."""
    return length * torch.rand((N,), generator=gen, device=device, dtype=dtype)


def sample_dynamics(
    N: int,
    time_on: float,
    time_off: float,
    time_bleach: float,
    n_periods_max: int,
    gen: Generator,
    relaxed: bool = False,
    temperature: float = 0.5,
    dtype=None,
    eps: float = 1e-9,
) -> Tensor:
    """Sample ON periods until final BLEACHED state is reached for N fluorophores.

    Returns a tensor of periods, of shape [N, n_periods_max, 2].
    Unused (padded) periods are [inf, inf].
    """
    device = gen.device

    # sample ON times
    u = torch.rand((N, n_periods_max), generator=gen, dtype=dtype, device=device)
    u = u.clip(min=eps, max=1.0 - eps)
    # minimum of two exp distributions is an other exp distribution
    rate = 1.0 / (1.0 / time_off + 1.0 / time_bleach)
    t_ON = -rate * u.log()  # exponential inverse transformation

    # sample OFF times.
    u = torch.rand((2, N, n_periods_max - 1), generator=gen, dtype=dtype, device=device)
    t_OFF = -time_on * u[0].clip(min=eps).log()

    # sample Bleach or OFF states. Bleach are OFF states with inf time ###
    p_bleach = time_off / (time_off + time_bleach)
    if not isinstance(p_bleach, Tensor):
        p_bleach = torch.tensor(p_bleach, device=device, dtype=dtype)
    p_bleach = p_bleach.clip(min=eps, max=1.0 - eps)
    if not relaxed:
        bleach_mask = u[1] <= p_bleach
        t_OFF = torch.where(bleach_mask, torch.inf, t_OFF)
    else:
        # see (Maddison et al., 2017).
        bleach_mask = (
            u[1].log() - (-u[1]).log1p() + p_bleach.log() - (-p_bleach).log1p()
        ) / temperature
        bleach_mask = bleach_mask.sigmoid()
        t_OFF = bleach_mask * n_periods_max + (1.0 - bleach_mask) * t_OFF
    # always wake up from an off state
    t_OFF = torch.cat([torch.zeros((N, 1), dtype=dtype, device=device), t_OFF], dim=-1)

    # format as periods and return
    times = torch.stack([t_OFF, t_ON], dim=-1)
    times = times.reshape((N, n_periods_max * 2))
    periods = torch.cumsum(times, dim=-1)
    periods = periods.reshape((N, n_periods_max, 2))
    return periods


def icdf_n_activations(
    a: float, time_bleach: float, time_off: float, eps: float = 1e-12
):
    """Return the smallest number of activations n such that P(N>=n)<a."""
    if a > 1.0 or a <= 0.0:
        raise ValueError("Expect a to be in (0, 1].")
    p = time_bleach / (time_off + time_bleach + eps)
    n = log(a) / log(p)
    return 1 + int(ceil(n))


def expected_time_off(time_on: float) -> float:
    """Return the expected duration of OFF state."""
    return time_on


def expected_time_on(time_off: float, time_bleach: float) -> float:
    """Return the expected duration of ON state."""
    return time_off * time_bleach / (time_off + time_bleach)


def expected_num_activations(time_off: float, time_bleach: float):
    """Return the expected number of activations of a fluorophore."""
    return 1.0 + time_bleach / time_off


def expected_lifespan_on(time_off: float, time_bleach: float) -> float:
    """Return the total expected time spent in ON states."""
    return expected_time_on(time_off, time_bleach) * expected_num_activations(
        time_off, time_bleach
    )


def expected_frames_on(time_off: float, time_bleach: float) -> float:
    """Return the expected number of frames a fluorophore will spent in ON state."""
    return expected_time_on(time_off, time_bleach) * expected_num_activations(
        time_off, time_bleach
    )
