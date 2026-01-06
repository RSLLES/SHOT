# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ot.bregman import sinkhorn_log

from smlmshot.losses.ot import sinkhorn


def main():
    """Experiments with optimal transport."""
    d = torch.load("ot.tar")
    a, b, M, reg = d["a"], d["b"], d["M"], d["reg"]

    K = sinkhorn(a=a[None], b=b[None], K=M[None], reg=reg, n_iters=200)[0]
    K_ot = sinkhorn_log(a=a, b=b, M=M, reg=reg, numItermax=200)

    assert torch.all_close(K, K_ot)


if __name__ == "__main__":
    main()
