# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset that returns frames along their associated activations."""

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from smlmshot.utils import nested


class ImagesAndActivationsDataset(Dataset):
    """Iterate over a stack of images and associated activations."""

    def __init__(self, y: Tensor, window: int, x: pd.DataFrame) -> None:
        """Construct the dataset.

        y: stack of image [N, W, H]
        window: sliding window size
        x: dataframe with columns [frame, x, y, z, n, s]
        """
        if y.ndim != 3:
            raise ValueError("y must be 3D tensor [N, W, H]")
        if window < 1:
            raise ValueError("window size must be >= 1")
        if window % 2 == 0:
            raise ValueError("window size must be odd")
        if x.shape[1] != 6:
            raise ValueError(
                "x must be a 2D tensor with columns [frame, x, y, z, n, s]"
            )
        self.y = y
        self.window = window
        self.pad = window // 2
        self.x = x
        self.max_n_acts = x.groupby("frame").size().max()

    def __len__(self) -> int:
        return self.y.size(0) - self.window + 1

    def __getitem__(self, index: int):
        if index < 0 or index >= self.__len__():
            raise StopIteration()

        y = self.y[index : index + self.window]

        mask = self.x["frame"] == index + self.pad
        xyz = self.x[mask][["x", "y", "z"]]
        xyz = torch.from_numpy(xyz.to_numpy().astype(np.float32))

        significant = self.x[mask][["significant"]]
        significant = torch.from_numpy(significant.to_numpy().astype(np.bool))
        significant.squeeze_(-1)

        return {"y": y, "xyz_tg": xyz, "significant_tg": significant}

    def collate_fn(self, batch):
        """Pad elements of variable length."""
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
        xyz_tg = self._pad(batch_dict["xyz_tg"])
        significant_tg = self._pad(batch_dict["significant_tg"])
        y = default_collate(batch_dict["y"])
        return {"y": y, "xyz_tg": xyz_tg, "significant_tg": significant_tg}

    def _pad(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Pad en element to n_fluos max number of activations."""
        return nested.pad_sequence(x, target_len=self.max_n_acts, returns_lengths=True)
