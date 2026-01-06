# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Provide functions to simulate backgrounds."""

import torch
from torch import Generator, Tensor


def sample_perlin_background(
    h: int,
    w: int,
    mean: float,
    std: float,
    gen: Generator,
    dtype=torch.get_default_dtype(),
    device=torch.get_default_device(),
) -> Tensor:
    """Sample a background, created with Perlin noise."""
    bg = torch.full((h, w), fill_value=mean, device=device, dtype=dtype)
    if std > 0.0:
        eps = generate_perlin_noise_2d(shape=(h, w), res=(1, 1), gen=gen)
        bg = mean + std * eps
        bg = bg.clip(min=0.0)
    return bg


def generate_perlin_noise_2d(shape: tuple, res: tuple, gen: Generator) -> Tensor:
    """Sample a 2D perlin noise."""

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    # Gradients
    gradients = torch.randn(res[0] + 1, res[1] + 1, 2, generator=gen)
    gradients = gradients / gradients.norm(dim=2, keepdim=True)

    # Coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, res[0], shape[0]),
        torch.linspace(0, res[1], shape[1]),
        indexing="ij",
    )

    grid = torch.stack([grid_y, grid_x], dim=-1)

    ij = grid.floor().long()
    fxy = grid - ij
    u, v = fade(fxy[..., 0]), fade(fxy[..., 1])

    def dot_grid(ix, iy, fx, fy):
        g = gradients[ix, iy]
        return (fx - ix) * g[..., 0] + (fy - iy) * g[..., 1]

    ix0 = ij[..., 0].clamp(0, res[0])
    iy0 = ij[..., 1].clamp(0, res[1])
    ix1 = (ix0 + 1).clamp(0, res[0])
    iy1 = (iy0 + 1).clamp(0, res[1])

    n00 = dot_grid(ix0, iy0, grid[..., 0], grid[..., 1])
    n10 = dot_grid(ix1, iy0, grid[..., 0], grid[..., 1])
    n01 = dot_grid(ix0, iy1, grid[..., 0], grid[..., 1])
    n11 = dot_grid(ix1, iy1, grid[..., 0], grid[..., 1])

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    return nxy
