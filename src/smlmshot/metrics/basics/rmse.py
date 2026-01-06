# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import sqrt

import numpy as np


def compute_rmse(matched_pred: np.ndarray, matched_gt: np.ndarray) -> float:
    """Compute the root mean square error between pairs of points."""
    if len(matched_gt.shape) != 2 or len(matched_pred.shape) != 2:
        raise ValueError("coordinates shape should be (N, d).")
    if len(matched_gt) != len(matched_pred):
        raise ValueError("arrays should contain the same number of elements.")
    if len(matched_gt) == 0:
        return 0.0
    diff = np.square(matched_pred - matched_gt)
    x = diff.sum(axis=-1).mean().item()
    return sqrt(x)
