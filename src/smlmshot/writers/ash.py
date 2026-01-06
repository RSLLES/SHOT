# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torchmetrics import Metric

from smlmshot import utils

from .writer import WriterInterface


class ASHWriter(WriterInterface):
    """Render a super-resolve image using Average Shifted Histogram."""

    def __init__(
        self,
        filepath: str,
        img_size: int | tuple[int, int],
        pixel_size: float | tuple[float] | Tensor,
        magnification: int,
        sharpening_factor: float,
        overwrite: bool = False,
    ):
        self.filepath = filepath
        self.overwrite = overwrite
        self.ash = AverageShiftedHistogram.from_magnification(
            img_size=img_size,
            pixel_size=pixel_size,
            magnification=magnification,
            sharpening_factor=sharpening_factor,
            export_as_figure=True,
        )

    def open(self):
        """Check if it does not overwrite an existing image."""
        if not self.overwrite and os.path.isfile(self.filepath):
            raise ValueError(f"Overwrite=false and '{self.filepath}' already exists.")

    def close(self):
        """Save the image."""
        fig = self.ash.compute()
        fig.savefig(self.filepath, dpi=300, bbox_inches="tight")

    def _write(self, data: Tensor):
        """Write one chunk of data."""
        xy = data[1:3].cpu()
        self.ash(xy=xy)


class AverageShiftedHistogram(Metric):
    """Render 2D visualizations of SMLM data by the average shifted histogram method."""

    full_state_update: bool = True

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        img_size: int | tuple[int, int],
        n_shifts: int = 2,
        kernel: str = "triweight",
        export_as_figure: bool = True,
    ):
        """Initialize the Average Shifted Histogram with physical coordinates.

        - x0, y0: The origin coordinates (top-left)
        - x1, y1: The end coordinates (bottom-right)
        - img_size: The desired output resolution (H, W)
        - smooth_factor: Determines the number of shifts.
        """
        super().__init__()
        self.register_buffer("origin", torch.tensor([x0, y0], dtype=torch.float32))
        self.H, self.W = utils.torch.to_pair(img_size)
        self.n_shifts = n_shifts
        kernel = utils.format.format_string(kernel)
        self.register_buffer("kernel", self.compute_kernel(kernel))
        self.export_as_figure = export_as_figure

        self.register_buffer(
            "bin_size", torch.tensor([abs(y1 - y0) / self.H, abs(x1 - x0) / self.W])
        )
        hist = torch.zeros((1, 1, self.H, self.W), dtype=torch.float)
        self.add_state("hist", default=hist, dist_reduce_fx="sum")

    @classmethod
    def from_magnification(
        cls,
        img_size: int | tuple[int, int],
        pixel_size: float | tuple[float] | Tensor,
        magnification: int = 2,
        sharpening_factor: float = 1.0,
        kernel: str = "triweight",
        export_as_figure: bool = True,
    ):
        """Initialize the Average Shifted Histogram.

        - img_size: the original image size
        - pixel_size: the original pixel size
        - magnification: controls the output image, that will be H * magn, W * magn
        - sharpening_factor: controls the resolvable distance of the output image.
        Resolvable distance will be pixel_size/sharpening_factor. Note that
        sharpening_factor must be <= than magnification.
        """
        if sharpening_factor > magnification:
            raise ValueError("sharpening_factor must be <= than magnification.")

        H_orig, W_orig = utils.torch.to_pair(img_size)
        pixel_size = utils.torch.to_pair(pixel_size)
        pixel_size = torch.as_tensor(pixel_size)

        H, W = H_orig * magnification, W_orig * magnification
        x0, y0 = 0.0, 0.0
        x1, y1 = W_orig * pixel_size[0], H_orig * pixel_size[1]
        n_shifts = int(round(magnification / sharpening_factor))
        return cls(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            img_size=(H, W),
            n_shifts=n_shifts,
            kernel=kernel,
            export_as_figure=export_as_figure,
        )

    @staticmethod
    def get_kernel(name: str):
        """Map kernel names to their implementation."""
        if name == "triangular":
            return lambda u: (1.0 - u.abs()).clip(min=0.0)
        if name == "triweight":
            return lambda u: (1.0 - u.square()).pow(3).clip(min=0.0)
        raise ValueError(
            f"Supported kernels are triangular and triweight, found {name}."
        )

    def compute_kernel(self, name: str):
        """Return the 2D convolution kernel needed to smooth the HD histogram."""
        kernel_func = self.get_kernel(name)
        u = torch.linspace(-1.0, 1.0, steps=2 * self.n_shifts + 1, device=self.device)
        kernel_1d = kernel_func(u[1:-1])  # edges are always 0
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d[None, None]  # pad for conv2d

    def get_output_img_size(self) -> [int, int]:
        """Return the output image size."""
        return self.H, self.W

    def get_output_pixel_size(self) -> Tensor:
        """Return the output image size."""
        return self.bin_size

    def update(self, xy: Tensor | list[Tensor]):
        """Update the high resolution histogram with the new values.

        Fastest implementation on GPU seems to rely on torch.bincount.
        """
        if isinstance(xy, list):
            xy = [e[..., :2] for e in xy]
            xy = torch.cat(xy, dim=0)
        xy = xy[..., :2].reshape(-1, 2)  # (N, 2), flatten potential batch size
        xy = xy.to(self.device)

        xy = xy - self.origin
        xy = xy[:, None]
        xy = xy / self.bin_size
        indices = xy.floor().long()

        mask = (
            (indices[..., 0] >= 0)
            & (indices[..., 0] < self.W)
            & (indices[..., 1] >= 0)
            & (indices[..., 1] < self.H)
        )
        indices = indices[mask]
        indices = indices[:, 1] * self.W + indices[:, 0]

        counts = torch.bincount(indices, minlength=self.hist.numel())
        self.hist += counts.view(*self.hist.shape).float()

    def compute(self):
        """Compute the ASH by convolving with a kernel the HR histogram."""
        ash = conv2d(self.hist, self.kernel, padding="same")[0, 0]
        if not self.export_as_figure:
            return ash

        img_extent = utils.extent.get_img_extent(
            h=self.H, w=self.W, pixel_size=self.bin_size
        )
        utils.plot.clf()
        utils.plot.imshow(ash, img_extent=img_extent)
        return utils.plot.plt.gcf()
