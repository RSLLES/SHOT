# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class StackDataset(torch.utils.data.StackDataset):
    """Stack multiple datasets."""

    def increment_seed(self):
        """Advance each dataset internal seed."""
        for ds in self.datasets.values():
            ds.increment_seed()

    def collate_fn(self, batch):
        """Collate each dataset with its own collate function."""
        batch = {
            key: ds.collate_fn([e[key] for e in batch])
            for key, ds in self.datasets.items()
        }
        return batch
