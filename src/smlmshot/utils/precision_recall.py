# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from .plot import _ensure_numpy


def _cvx(min, max, lambd):
    return (1 - lambd) * min + lambd * max


def decorate(ax):
    """Create a precision-recall plot figure."""
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_0 = _cvx(x_min, x_max, lambd=0.05)
    y_0 = _cvx(y_min, y_max, lambd=0.05)
    x_1 = _cvx(x_min, x_max, lambd=0.35)
    y_1 = _cvx(y_min, y_max, lambd=0.35)

    ax.annotate(
        "",
        xy=(x_1, y_0),
        xytext=(x_0, y_0),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    ax.annotate(
        "",
        xy=(x_0, y_1),
        xytext=(x_0, y_0),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    ax.text(
        _cvx(x_0, x_1, lambd=0.5),
        _cvx(y_min, y_0, lambd=0.5),
        "better",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        _cvx(x_min, x_0, lambd=0.5),
        _cvx(y_0, y_1, lambd=0.5),
        "better",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
    )


def plot(prec, rec, ax, **kwargs):
    """Add a precision-recall curve to the plot."""
    prec = _ensure_numpy(prec)
    rec = _ensure_numpy(rec)
    sorted_indices = np.argsort(rec)
    rec_sorted = rec[sorted_indices]
    prec_sorted = prec[sorted_indices]

    ax.plot(rec_sorted, prec_sorted, **kwargs)
    ax.legend()
