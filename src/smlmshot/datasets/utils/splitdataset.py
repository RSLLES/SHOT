# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset

from smlmshot import utils


class SplitDataset(torch.utils.data.Subset):
    """Split a dataset by a proportion and direction."""

    def __init__(self, ds: Dataset, prop: float, direction: str):
        if prop < 0 or prop > 1:
            raise ValueError("Propotion should be in [0, 1]")
        split_idx = int(prop * len(ds))
        direction = utils.strings.format_string(direction)
        if direction == "forward":
            indices = range(0, split_idx)
        elif direction == "backward":
            indices = range(split_idx, len(ds))
        else:
            raise ValueError("direction must be 'forward' or 'backward'")
        super().__init__(ds, indices=indices)

    def increment_seed(self):
        """Advance internal seed."""
        self.dataset.increment_seed()

    def collate_fn(self, batch):
        """Provide the custom collate function of the base dataset."""
        return self.dataset.collate_fn(batch)
