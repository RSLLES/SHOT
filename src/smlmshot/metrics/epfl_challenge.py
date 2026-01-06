# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics of the EPFL 2016 SMLM challenge.

See https://srm.epfl.ch/srm/assessment/index.html or the complete challenge paper:
Sage, D., Pham, TA., Babcock, H. et al.
Super-resolution fight club: assessment of 2D and 3D single-molecule localization
microscopy software. https://doi.org/10.1038/s41592-019-0364-4
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import Tensor
from torchmetrics import Metric

from . import basics


def match_sets(
    set1: np.ndarray,
    set2: np.ndarray,
    xy_threshold=250,
    z_threshold=500,
):
    """Compute an assignment between two sets according to Sage et al.

    Thresholds are in nanometers and are defined in the challenge rules.
    """
    xy_sq_thresh, z_sq_thresh = xy_threshold**2, z_threshold**2
    numeric_inf = 10 * (xy_sq_thresh + z_sq_thresh)
    xy_sq_dist = cdist(set1[:, :2], set2[:, :2], metric="sqeuclidean")
    z_sq_dist = cdist(set1[:, 2:3], set2[:, 2:3], metric="sqeuclidean")
    valid_mask = (xy_sq_dist <= xy_sq_thresh) & (z_sq_dist <= z_sq_thresh)
    cost_matrix = np.where(valid_mask, xy_sq_dist + z_sq_dist, np.inf)
    cost_matrix_finite = np.where(valid_mask, cost_matrix, numeric_inf)
    matched_set1_idx, matched_set2_idx = linear_sum_assignment(cost_matrix_finite)
    valid = ~np.isinf(cost_matrix[matched_set1_idx, matched_set2_idx])
    return matched_set1_idx[valid], matched_set2_idx[valid]


def compute_metrics(x: Tensor, x_gt: Tensor, s: Tensor) -> dict:
    """Compute EPFL metrics for one image."""
    # Convert to numpy (matching and RMSE functions are numpy-based)
    device, dtype = x.device, x.dtype

    pred_np = x[:, :3].cpu().numpy()
    gt_np = x_gt[:, :3].cpu().numpy()
    is_significant_np = s.cpu().numpy()

    n_gts = is_significant_np.sum().item()
    # Perform matching between predicted and gt coordinates.
    matched_pred_idx, matched_gt_idx = match_sets(set1=pred_np, set2=gt_np)

    # Filter out matches that correspond to non-significant ground truths.
    significant_match = is_significant_np[matched_gt_idx]
    n_negligible_matches = len(significant_match) - significant_match.sum().item()
    n_matches = len(matched_gt_idx) - n_negligible_matches
    n_preds = len(pred_np) - n_negligible_matches

    # Compute TP, FP, FN from analytic helper.
    tp, fp, fn = basics.compute_tp_fp_fn(
        n_gts=n_gts, n_matches=n_matches, n_preds=n_preds
    )
    jaccard = basics.compute_jaccard(tp=tp, fp=fp, fn=fn)
    precision = basics.compute_precision(tp=tp, fp=fp)
    recall = basics.compute_recall(tp=tp, fn=fn)

    # Select only matches with significant ground truths.
    matched_gt = gt_np[matched_gt_idx][significant_match]
    matched_pred = pred_np[matched_pred_idx][significant_match]
    offset = (matched_gt - matched_pred).sum(axis=0)
    offset = torch.as_tensor(offset).to(device=device, dtype=dtype)

    # Compute RMSE values.
    rmse_lat = basics.compute_rmse(
        matched_pred=matched_pred[:, :2], matched_gt=matched_gt[:, :2]
    )
    rmse_axial = basics.compute_rmse(
        matched_pred=matched_pred[:, 2:], matched_gt=matched_gt[:, 2:]
    )
    rmse_vol = basics.compute_rmse(matched_pred=matched_pred, matched_gt=matched_gt)
    # Compute composite efficiency metric.
    E_lat = basics.compute_lat_efficiency(jaccard=jaccard, rmse_lat=rmse_lat)
    E_ax = basics.compute_axial_efficiency(jaccard=jaccard, rmse_axial=rmse_axial)
    E_3D = basics.compute_3D_efficiency(
        jaccard=jaccard, rmse_lat=rmse_lat, rmse_axial=rmse_axial
    )

    return {
        "n_matches": n_matches,
        "jac": jaccard,
        "prec": precision,
        "rec": recall,
        "n_detects": n_preds,
        "rmse_lat": rmse_lat,
        "rmse_axial": rmse_axial,
        "rmse_vol": rmse_vol,
        "E_lat": E_lat,
        "E_ax": E_ax,
        "E_3D": E_3D,
        "offset": offset,
    }


class EPFLChallengeMetrics(Metric):
    """Metric that returns a dict containing the EPFL 2016 challenge metrics."""

    def __init__(self):
        super().__init__()

        # counters
        self.add_state("n_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Running sums for metrics; these states will be summed across batches/updates.
        self.add_state("jac", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("prec", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_detects", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rmse_lat", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rmse_axial", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rmse_vol", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_lat", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_ax", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_3D", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "offset", default=torch.tensor([0.0, 0.0, 0.0]), dist_reduce_fx="sum"
        )

    def update(self, x: Tensor, x_gt: Tensor, s: Tensor):
        """Update state with a pairs of predictions / ground truths / significance."""
        B = len(x)
        for i in range(B):
            m = compute_metrics(x=x[i], x_gt=x_gt[i], s=s[i])
            self.n_batches += 1
            self.n_matches += m["n_matches"]
            self.jac += m["jac"]
            self.prec += m["prec"]
            self.rec += m["rec"]
            self.n_detects += m["n_detects"]
            self.rmse_lat += m["rmse_lat"]
            self.rmse_axial += m["rmse_axial"]
            self.rmse_vol += m["rmse_vol"]
            self.E_3D += m["E_3D"]
            self.E_lat += m["E_lat"]
            self.E_ax += m["E_ax"]
            self.offset += m["offset"]

    def compute(self):
        """Reduce and compute final metrics."""
        return {
            "jac": self.jac / self.n_batches,
            "prec": self.prec / self.n_batches,
            "rec": self.rec / self.n_batches,
            "n_detects": self.n_detects / self.n_batches,
            "rmse_lat": self.rmse_lat / self.n_batches,
            "rmse_axial": self.rmse_axial / self.n_batches,
            "rmse_vol": self.rmse_vol / self.n_batches,
            "E_lat": self.E_lat / self.n_batches,
            "E_ax": self.E_ax / self.n_batches,
            "E_3D": self.E_3D / self.n_batches,
            "offset": self.offset / self.n_matches,
        }
