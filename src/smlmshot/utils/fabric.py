# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for fabric."""

import contextlib

from lightning.fabric.connector import _is_using_cli
from lightning_fabric import Fabric


def initialize_fabric(
    seed: int, precision: str = "32-true", devices: str | int = "auto"
) -> Fabric:
    """Initialize a Fabric instance with the given seed and devices."""
    fabric = Fabric(devices=devices, precision=precision)
    fabric.seed_everything(seed)
    if not _is_using_cli():  # delegate launch call to CLI
        fabric.launch()
    return fabric


@contextlib.contextmanager
def global_zero_context(ctx_manager_factory, fabric: Fabric):
    """Wrap a context manager so it only activates on global zero device."""
    if fabric.is_global_zero:
        with ctx_manager_factory() as ctx:
            yield ctx
    else:
        yield None
