import torch
from torch import Generator, Tensor


def sample_coordinates(N: int, vol_extent: Tensor, gen: Generator) -> Tensor:
    """Sample N 3D coordinates, uniformly on XY and normally distributed on Z."""
    xy = torch.rand((N, 2), generator=gen, device=gen.device)
    z = 0.5 + 0.2 * torch.randn((N, 1), generator=gen, device=gen.device)
    z = torch.clip(z, min=0.0, max=1.0)
    xyz = torch.cat([xy, z], dim=-1)
    xyz = (vol_extent[:, 1] - vol_extent[:, 0]) * xyz + vol_extent[:, 0]
    return xyz
