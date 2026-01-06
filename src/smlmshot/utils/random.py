# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for reproducible random number generation."""

import torch
from torch import Generator, Tensor

_MAX_SEED = torch.iinfo(torch.int64).max


@torch.compiler.disable()
def get_generator(seed: int | Tensor, device=None) -> Generator:
    """Return a torch.Generator seeded with `seed` on `device`."""
    if isinstance(seed, Tensor):
        device = seed.device if device is None else device
        seed = seed.item()
    gen = torch.Generator(device).manual_seed(seed)
    return gen


@torch.compiler.disable()
def derive_new_seed(seed: int | Tensor) -> int | Tensor:
    """Derive a new seed from `seed` using a dedicated generator."""
    gen = get_generator(seed)
    new_seed = torch.randint(
        0, _MAX_SEED, size=(), device=gen.device, dtype=torch.int64, generator=gen
    )
    new_seed = new_seed.item() if isinstance(seed, int) else new_seed
    return new_seed


@torch.compiler.disable()
def derive_new_seed_(seed: Tensor):
    """In-place version of derive_new_seed."""
    gen = get_generator(seed)
    new_seed = torch.randint(
        0, _MAX_SEED, size=(), device=gen.device, dtype=torch.int64, generator=gen
    )
    seed.copy_(new_seed)


def multiplicative_noise(x: Tensor, std: float, gen: Generator):
    """Return a jittered version of x using log-normal noise."""
    if std <= 0.0:
        return x

    eps = torch.randn(x.size(), device=x.device, generator=gen)
    eps = torch.clip(eps, min=-5.0, max=5.0)
    x = x * torch.exp(std * eps)
    return x
