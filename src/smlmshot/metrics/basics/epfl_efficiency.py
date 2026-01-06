# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Functions around the EPFL efficiency metric.

See the original challenge paper for more information:
Sage, D., Pham, TA., Babcock, H. et al.
Super-resolution fight club: assessment of 2D and 3D single-molecule localization
microscopy software. https://doi.org/10.1038/s41592-019-0364-4
"""

from math import sqrt


def _compute_efficiency(jaccard: float, rmse: float, alpha: float) -> float:
    """Compute efficiency metric according to Sage et al."""
    return 1.0 - sqrt((1.0 - jaccard) ** 2 + (alpha * rmse) ** 2)


def compute_lat_efficiency(jaccard: float, rmse_lat: float) -> float:
    """Compute lateral efficiency (xy plane)."""
    return _compute_efficiency(jaccard=jaccard, rmse=rmse_lat, alpha=1e-2)


def compute_axial_efficiency(jaccard: float, rmse_axial: float) -> float:
    """Compute axial efficiency (z axis)."""
    return _compute_efficiency(jaccard=jaccard, rmse=rmse_axial, alpha=5e-3)


def compute_3D_efficiency(jaccard: float, rmse_lat: float, rmse_axial: float) -> float:
    """Compute 3D efficiency according to Sage et al."""
    e_lat = compute_lat_efficiency(jaccard=jaccard, rmse_lat=rmse_lat)
    e_axial = compute_axial_efficiency(jaccard=jaccard, rmse_axial=rmse_axial)
    return 0.5 * (e_lat + e_axial)
