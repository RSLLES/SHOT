# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn

from smlmshot.utils.torch import hash_tensor


class Read(nn.Module):
    """Iterate over a predefined list of coordinates, disregarding the input."""

    def __init__(self, x: pd.Series, init_frame: int = 1):
        """x: Pandas Series [N, 6] with columns [frame, x, y, z, n, s]."""
        super().__init__()
        if dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError("Read module only supports 1 GPU.")
        self.x = x
        self.init_frame = init_frame
        self.first_y_hash = None
        self.frame_counter = None

    def forward(self, y: Tensor) -> Tensor:  # noqa: D102
        if y.ndim != 4:
            raise ValueError("Expect y to have 4 dimensions: (bs, n_frame, h, w)")
        device, dtype = y.device, y.dtype
        bs = y.size(0)

        y_hash = hash_tensor(y[0])
        if self.first_y_hash is None:
            self.first_y_hash = y_hash
        if self.first_y_hash == y_hash:
            self.frame_counter = self.init_frame

        x = []
        for i in range(bs):
            mask = self.x["frame"] == self.frame_counter
            xi = self.x[mask][["x", "y", "z", "photons"]]
            xi = torch.from_numpy(xi.to_numpy(dtype=np.float32))
            xi = xi.to(device=device, dtype=dtype)
            x.append(xi)
            self.frame_counter += 1
        return x
