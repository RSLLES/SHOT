# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Everything related to cubic splines calibrated with SMAP."""

from functools import lru_cache

import scipy.io as sio
import torch
from torch import Tensor, nn

from smlmshot import utils


class CSplinesPSF(nn.Module):
    """Convenient wrapper around cubic splines PSF."""

    def __init__(
        self,
        center: tuple[float] | Tensor,
        coeffs: Tensor,
        delta_z: float | Tensor,
        pixel_size: float | tuple[float] | Tensor,
    ):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center).contiguous())
        self.register_buffer("coeffs", torch.as_tensor(coeffs).contiguous())
        self.register_buffer("delta_z", torch.as_tensor(delta_z).contiguous())
        pixel_size = utils.torch.to_pair(pixel_size)
        pixel_size = torch.as_tensor(pixel_size)
        voxel_size = torch.cat([pixel_size, self.delta_z[None]], dim=-1)
        self.register_buffer("inv_voxel_size", 1.0 / voxel_size)

    @classmethod
    @lru_cache(maxsize=1)
    def init_from_mat(cls, filepath: str, pixel_size: float | tuple[float] | Tensor):
        """Load a .mat cubic-spline PSF calibrated with SMAP."""
        data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)["SXY"]
        center = (data.cspline.x0 - 1, data.cspline.x0 - 1, data.cspline.z0 - 1)
        coeffs = data.cspline.coeff
        coeffs = coeffs.transpose([2, 0, 1, 3])  # axis convention(z, x, y)
        delta_z = data.cspline.dz
        return cls(center=center, coeffs=coeffs, delta_z=delta_z, pixel_size=pixel_size)

    @property
    def z_extent(self, shrink_factor: float = 1e-4) -> Tensor:
        """Return [zmin, zmax], shrunk by shrink_factor."""
        return compute_z_extent(
            coeffs=self.coeffs,
            center=self.center,
            delta_z=self.delta_z,
            shrink_factor=shrink_factor,
        )

    def batched_render_fluorophores(
        self,
        xyz: Tensor,
        n_photons: Tensor,
        img_size: int | tuple[int],
        chunk_size: int = -1,
    ) -> Tensor:
        """Render a SMLM acquisition (pre-camera) with fluorophores.

        It splits an intermediate heavy matrix into chunks to prevent OOM errors.
        """
        return batched_render_fluorophores(
            xyz=xyz,
            n_photons=n_photons,
            img_size=img_size,
            inv_voxel_size=self.inv_voxel_size,
            psf_center=self.center,
            psf=self.coeffs,
            chunk_size=chunk_size,
        )


def compute_z_extent(
    coeffs: Tensor, center: Tensor, delta_z: float, shrink_factor: float = 0.0
) -> Tensor:
    """Return [zmin, zmax], shrunk by shrink_factor."""
    z_min = -center[-1]
    z_max = coeffs.size(0) + z_min
    z_extent = torch.tensor([z_min, z_max], device=coeffs.device)
    z_extent = delta_z * z_extent
    z_extent = (1.0 - shrink_factor) * z_extent
    return z_extent


def batched_render_fluorophores(
    xyz: Tensor,
    n_photons: Tensor,
    img_size: int | tuple[int],
    inv_voxel_size: Tensor,
    psf_center: Tensor,
    psf: Tensor,
    chunk_size: int = -1,
) -> Tensor:
    """Apply the PSF to a set of fluorophores to produce an image.

    It takes fluorophores coordinates xyz: (B, N, 3) and their associated photon
    emissions per frame n: (B, N, F). It returns an image (B, F, H, W).
    Intermediate matrix multiplication is split into chunks to prevent OOM errors.
    """
    if xyz.ndim != 3 or xyz.size(-1) != 3:
        raise ValueError(f"Expect xyz to be a (B, N, 3) tensor, found {xyz.size()}.")
    if n_photons.ndim != 3 or n_photons.shape[:2] != xyz.shape[:2]:
        raise ValueError(
            f"Expect n_photons to be a (B, N, F) tensor, found {n_photons.size()}."
        )
    if torch.any(xyz.isnan()) or torch.any(n_photons.isnan()):
        raise ValueError("nan detected in xyz or n.")

    if psf.ndim != 4 or psf.shape[-1] != 64:
        raise ValueError("psf must be a 4D tensor of shape [D, H, W, 64].")
    if inv_voxel_size.ndim != 1 or inv_voxel_size.shape[0] != 3:
        raise ValueError("inv_voxel_size must be a 1D tensor of size 3.")
    if psf_center.ndim != 1 or psf_center.shape[0] != 3:
        raise ValueError("center must be a 1D tensor of size 3.")

    B, N = xyz.size(0), xyz.size(1)
    F = n_photons.size(-1)
    H, W = utils.torch.to_pair(img_size)
    PSF_D, PSF_H, PSF_W, _ = psf.size()
    n_photons = n_photons.transpose(1, 2).contiguous()  # (B, F, N)

    # Render each emitter with the PSF
    x_idx, y_idx, z_idx, u = _batched_preprocess_coordinates(
        xyz,
        inv_voxel_size=inv_voxel_size,
        psf_center=psf_center,
        psf_h=PSF_H,
        psf_w=PSF_W,
    )
    _check_index_range(x_idx, lb=0, ub=W + PSF_W, name="x")
    _check_index_range(y_idx, lb=0, ub=H + PSF_H, name="y")
    _check_index_range(z_idx, lb=0, ub=PSF_D, name="z")

    u = u.view(B * N, 3)
    u = _cubic_3d_power_series(u)  # (B*N, 64)
    z_idx = z_idx.view(B * N)
    if chunk_size is not None and chunk_size > 0:
        rendered_psfs = [
            torch.einsum(
                "bhwc,bc->bhw",
                psf[z_idx[i : i + chunk_size]],
                u[i : i + chunk_size],
            )
            for i in range(0, z_idx.size(0), chunk_size)
        ]
        rendered_psfs = torch.cat(rendered_psfs, dim=0)
    else:
        rendered_psfs = torch.einsum("bhwc,bc->bhw", psf[z_idx], u)

    # Construct frames
    y = torch.zeros(
        (B * F, H + 2 * PSF_H, W + 2 * PSF_W),
        dtype=rendered_psfs.dtype,
        device=rendered_psfs.device,
    )
    x_idx = x_idx.repeat_interleave(F, dim=0)  # (B*F, N)
    y_idx = y_idx.repeat_interleave(F, dim=0)  # (B*F, N)
    rendered_psfs = rendered_psfs.view(B, N, PSF_H, PSF_W)
    rendered_psfs = torch.einsum("bfn,bnhw->bfnhw", n_photons, rendered_psfs)
    rendered_psfs = rendered_psfs.reshape(B * F, N, PSF_H, PSF_W)
    _batched_accumulate_patches(patches=rendered_psfs, i_idx=x_idx, j_idx=y_idx, out=y)
    y = y[:, PSF_H:-PSF_H, PSF_W:-PSF_W]
    y = y.reshape(B, F, H, W)
    return y


def _batched_accumulate_patches(
    patches: Tensor, i_idx: Tensor, j_idx: Tensor, out: Tensor
):
    bs, N, p_h, p_w = patches.shape
    _, h, w = out.shape
    device = patches.device

    vals = patches.view(-1)
    out = out.view(-1)

    bs_offset = torch.arange(bs, device=device).view(bs, 1)
    base_idx = (bs_offset * h + j_idx) * w + i_idx
    base_idx = base_idx.view(bs * N, 1)
    j_offset = torch.arange(p_h, device=device).view(1, p_h, 1)
    i_offset = torch.arange(p_w, device=device).view(1, 1, p_w)
    offsets = j_offset * w + i_offset
    offsets = offsets.view(1, p_h * p_w)
    idx = offsets + base_idx
    idx = idx.view(bs * N * p_h * p_w)
    out = out.view(-1)
    out.index_add_(dim=0, index=idx, source=vals)


def _batched_preprocess_coordinates(
    x: Tensor, inv_voxel_size: Tensor, psf_center: Tensor, psf_h: int, psf_w: int
):
    """Split 3D coordinates into 3D indices and floating residuals."""
    u = x[..., :3] * inv_voxel_size
    u[..., :2] -= psf_center[:2]
    u[..., 2] += psf_center[2]
    u_idx = u.floor()
    u -= u_idx
    u_idx = u_idx.int()
    x_idx = u_idx[..., 0].add(psf_w)
    y_idx = u_idx[..., 1].add(psf_h)
    z_idx = u_idx[..., 2]
    u[..., :2] = 1.0 - u[..., :2]  # Z follows a different convention
    return x_idx, y_idx, z_idx, u


def _check_index_range(idx: Tensor, lb: int, ub: int, name: str):
    """Ensure index ranges are valid for one dimension 'name'."""
    if torch.any(idx < lb):
        x = idx[idx < lb][0]
        raise ValueError(f"Out of range {name}: Value {x} < {lb}.")

    if torch.any(idx >= ub):
        x = idx[idx >= ub][0]
        raise ValueError(f"Out of range {name}: Value {x} >= {ub}.")


def _cubic_3d_power_series(u: Tensor) -> Tensor:
    """Return the 64 elements of a 3D cubic power series."""
    # vander requires float32
    u = torch.linalg.vander(u.to(dtype=torch.float32), N=4).to(dtype=u.dtype)
    u = torch.einsum("bx,by,bz->bzxy", u[:, 0], u[:, 1], u[:, 2])  # z on front
    u = u.view(u.size(0), 64)
    return u
