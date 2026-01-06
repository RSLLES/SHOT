# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to plot SMLM images."""

import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from torch import Tensor

DEFAULT_CMAP = "magma"

# those are identical to matplotlib's original functions
legend = plt.legend
title = plt.title
axis = plt.axis
show = plt.show
clf = plt.clf
tight_layout = plt.tight_layout


def _ensure_numpy(x: np.ndarray | Tensor | list) -> np.ndarray:
    """Make sure x is a numpy array."""
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError("ensure_numpy only accepts numpy arrays, list or PyTorch tensors.")


def set_dark_background():
    """Set matplotlib's background to dark."""
    plt.style.use("dark_background")


def imshow(
    image: np.ndarray,
    img_extent: np.ndarray = 2 * [[0, 6400]],
    **plt_kwargs,
):
    """Plot an image given the coordinates of it's 4 corners, in nanometers."""
    image = _ensure_numpy(image)
    img_extent = _ensure_numpy(img_extent)
    plt.imshow(
        image,
        cmap=DEFAULT_CMAP,
        extent=[img_extent[0, 0], img_extent[0, 1], img_extent[1, 1], img_extent[1, 0]],
        **plt_kwargs,
    )
    plt.axis("off")
    plt.tight_layout(pad=1.00)


def _rois_show(centers: np.ndarray, size: float, ax=plt.gca(), **kwargs_plt):
    if len(centers.shape) != 2 or centers.shape[-1] < 2:
        raise ValueError(
            f"ROI centers shape must be [N, d >= 2]; found {centers.shape}"
        )
    centers = _ensure_numpy(centers)
    XY = centers[:, :2]
    for x, y in XY:
        rect = patches.Rectangle(
            (x - size, y - size),
            2 * size,
            2 * size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            **kwargs_plt,
        )
        ax.add_patch(rect)


def _coordinates_show(coordinates: np.ndarray, **kwargs_plt):
    if len(coordinates.shape) != 2 or coordinates.shape[-1] < 2:
        raise ValueError(
            f"Plotted coordinates shape must be [N, d >= 2]; found {coordinates.shape}"
        )
    coordinates = _ensure_numpy(coordinates)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], **kwargs_plt)


def gt_coordinates_show(coordinates: np.ndarray, significant: np.ndarray = None):
    """Plot ground truth coordinates ac white circles.

    Optionally, a significant mask plots all insignificant elements with alpha=0.5.
    """
    if significant is None:
        significant = np.ones((coordinates.shape[0],), dtype=bool)
    _coordinates_show(
        coordinates=coordinates[significant],
        color="w",
        marker="+",
        label="Targets",
    )
    if significant.sum() < significant.shape[0]:
        _coordinates_show(
            coordinates=coordinates[~significant],
            color="#cccccc",
            marker="+",
            alpha=0.5,
        )


def pred_coordinates_show(coordinates: np.ndarray):
    """Plot predicted coordinates as yellow X."""
    _coordinates_show(
        coordinates=coordinates,
        color="yellow",
        marker="x",
        label="Guesses",
    )


def savefig(path: str = "/tmp/out.png", dpi: int = 300):
    """Save current figure, by default at '/tmp/out.png'."""
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def imsave(img: np.ndarray, path: str = "/tmp/out.png"):
    """Imsave the SMLM image with correct cmap."""
    img = _ensure_numpy(img)
    plt.imsave(path, img, cmap=DEFAULT_CMAP)


def gifsave(img: np.ndarray, path: str = "/tmp/out.gif", duration: int = 100):
    """Convert a NumPy array of shape [T, H, W] to a GIF."""
    frames = []
    buffers = []  # Store buffers to keep them open until after GIF creation

    for i in range(img.shape[0]):
        buf = io.BytesIO()
        imsave(img[i], path=buf)
        buf.seek(0)
        frames.append(Image.open(buf))
        buffers.append(buf)  # Keep buffer open by storing it
        plt.close(plt.gcf())

    # Save frames as a GIF
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )

    # Close all buffers after saving the GIF
    for buf in buffers:
        buf.close()


def histplot(data: np.ndarray, **kwargs):
    """Interface for seaborn's histplot function."""
    data = _ensure_numpy(data)
    sns.histplot(data=data, **kwargs)
