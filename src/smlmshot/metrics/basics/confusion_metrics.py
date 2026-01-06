# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Helper functions to compute classic confusion metrics."""


def compute_tp_fp_fn(n_matches: int, n_preds: int, n_gts: int) -> (int, int, int):
    """Return num of true positives (tp), false positives (fp), false negatives (fn)."""
    tp = n_matches
    fp = n_preds - tp
    fn = n_gts - tp
    return tp, fp, fn


def compute_precision(tp: int, fp: int, eps: float = 1e-12):  # noqa: D103
    if fp == 0:
        return 1.0
    return tp / (tp + fp + eps)


def compute_recall(tp: int, fn: int, eps: float = 1e-12):  # noqa: D103
    if fn == 0:
        return 1.0
    return tp / (tp + fn + eps)


def compute_jaccard(tp: int, fp: int, fn: int, eps: float = 1e-12):
    """Compute jaccard index."""
    if fp == 0 and fn == 0:
        return 1.0
    return tp / (tp + fp + fn + eps)
