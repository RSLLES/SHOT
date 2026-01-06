# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions around activations.

Activations are implemented as pandas DataFrame with DEFAULT_COLUMNs names.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

DEFAULT_COLUMNS = ["frame", "x", "y", "z", "photons"]


def read_csv(filepath: str, columns: list) -> pd.DataFrame:
    """Read activations from a csv file with columns names for [frame, x, y, z, n]."""
    if len(columns) != 5:
        raise ValueError(
            "columns must contain the five column names for [frame, x, y, z, n]"
        )
    x = pd.read_csv(filepath)
    x.columns = x.columns.str.replace(" ", "", regex=False).str.lower()
    columns = [e.replace(" ", "").lower() for e in columns]
    x = x[columns]
    x.columns = DEFAULT_COLUMNS
    return x


def _is_above_quantile(series: pd.Series, q: float) -> np.array:
    """Return a mask with True for element above a specified quantile."""
    x = series.to_numpy()
    threshold = np.quantile(x, q=q)
    return series > threshold


def add_significant(x: pd.DataFrame, q: float = 0.25) -> pd.DataFrame:
    """Add a 'significant' boolean column, indicating if an activation is significant.

    The 2016 EPFL challenge consider an activation to be insignificant if its intensity
    is below the 25% quantile among all the activations.
    """
    x["significant"] = _is_above_quantile(x["photons"], q=q)
    return x


def subtract_frame_offset(x: pd.DataFrame, frame_offset: int) -> pd.DataFrame:
    """Subtract a frame offset to the 'frame' column."""
    x["frame"] -= frame_offset
    return x


def mirror(x: pd.DataFrame, axis: str, mirror_value: float) -> pd.DataFrame:
    """Mirror acquisition by performing x = mirror_value - x."""
    x[axis] = mirror_value - x[axis]
    return x


def flip_z(x: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates on the z axis."""
    return mirror(x, axis="z", mirror_value=0.0)


def bin(
    x: pd.DataFrame, bin_size: int, threshold: float = 1.0, reduction: str = "mean"
) -> pd.DataFrame:
    """Bin activations across time, speeding up the acquisitions by bin_size.

    Two activations gathered on a single frame that are specially close are considered
    to originate from the same fluorophore: their number of emitted photons is reduced.
    """
    x["frame"] = x["frame"] // bin_size

    x_merged = []
    for i in x["frame"].unique():
        xi = x[x["frame"] == i].copy()
        coords = xi[["x", "y", "z"]].values
        clusters = fclusterdata(coords, t=threshold, criterion="distance")
        xi["cluster"] = clusters
        xi = xi.groupby("cluster")
        xi = xi.agg(
            {"frame": "max", "x": "max", "y": "max", "z": "max", "photons": reduction}
        )
        x_merged.append(xi.reset_index(drop=True))
    x_merged = pd.concat(x_merged, ignore_index=True)
    return x_merged
