# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to inspect models."""

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def present_model(model: nn.Module) -> None:
    """Print the model name and its trainable parameter count."""
    print(f"Model '{model._get_name()}'; {count_parameters(model):,} parameters.")
