# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions around dataloaders."""

import os

from torch.utils.data import DataLoader, Dataset


def interpret_n_workers(n_workers: int, world_size: int) -> int:
    """Compute effective number of workers per process, given system and world size."""
    if n_workers < 0:
        n_workers = os.process_cpu_count()
    n_workers = max(0, n_workers // world_size) if n_workers is not None else 0
    return n_workers


def build_dl(
    ds: Dataset,
    batch_size: int,
    n_workers: int,
    shuffle: bool,
    world_size: int,
) -> DataLoader:
    """Return a DataLoader using the dataset's collate function."""
    n_workers = interpret_n_workers(n_workers, world_size=world_size)
    dl = DataLoader(
        ds,
        collate_fn=ds.collate_fn,
        batch_size=min(batch_size, max(len(ds), 1)),
        num_workers=n_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        # incompatible with utils.training.handle_oom():  OSError: Too many open files
        # persistent_workers=n_workers > 0,
        persistent_workers=False,
    )
    return dl
