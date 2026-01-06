# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Comparison of the fast, accelerated new mod vs the old one."""

import torch

from smlmshot.a.fluorophore import batch_sample_dynamics, statistics
from smlmshot.utils.random import get_generator


def mean_on_time(x, inf=torch.inf):
    """Compute the average time spent in ON state."""
    x = torch.where(x >= inf, torch.nan, x)
    delta = x[..., 1] - x[..., 0]
    return torch.nanmean(delta)


def mean_off_time(x, inf=torch.inf):
    """Compute the average time spent in OFF state."""
    x = torch.where(x >= inf, torch.nan, x)
    delta = x[..., 1:, 0] - x[..., :-1, 1]
    return torch.nanmean(delta)


def mean_n_acts(x, inf=torch.inf):
    """Compute the average number of activations."""
    return (x[..., 0] < inf).sum(-1).float().mean()


def test_batch_sample_dynamics(
    N=100_000,
    time_bleach=1.5,
    time_off=3.0,
    time_on=2.5,
    n_periods_max=10,
    seed: int = 0,
):
    """Test batch_sample_dynamics function against theory and the legacy method."""
    gen = get_generator(seed=seed)
    x = batch_sample_dynamics(
        N=N,
        time_on=time_on,
        time_off=time_off,
        time_bleach=time_bleach,
        n_periods_max=n_periods_max,
        gen=gen,
    )
    expected_mean_on_time = statistics.expected_time_on(
        time_off=time_off, time_bleach=time_bleach
    )
    assert torch.allclose(
        mean_on_time(x),
        torch.tensor(expected_mean_on_time),
        rtol=1e-2,
    )
    expected_mean_off_time = statistics.expected_time_off(time_on)
    assert torch.allclose(
        mean_off_time(x), torch.tensor(expected_mean_off_time), rtol=1e-2
    )
    assert torch.allclose(mean_off_time(x), mean_off_time(x), rtol=1e-2)

    expected_n_acts = statistics.expected_num_activations(
        time_off=time_off, time_bleach=time_bleach
    )
    assert torch.allclose(mean_n_acts(x), torch.tensor(expected_n_acts), rtol=1e-2)
