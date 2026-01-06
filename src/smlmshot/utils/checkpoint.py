# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for checkpoints; path handling, load, save."""

import os

from lightning_fabric import Fabric
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

### Logs ###


def get_ckpts_path(log_dir: str, ckpt_dir: str = None) -> (str, str):
    """Return the paths to the last and best checkpoint files."""
    root = os.path.join(log_dir, ckpt_dir) if ckpt_dir is not None else log_dir
    path_best_ckpt = os.path.join(root, "best.tar")
    path_last_ckpt = os.path.join(root, "last.tar")
    return path_last_ckpt, path_best_ckpt


### Load ###


def load_weights(
    fabric: Fabric,
    ckpt_path: str,
    model: Module,
    strict: bool = True,
    model_prefix: str = "model.",
):
    """Load model weights from a checkpoint into the model."""
    ckpt = fabric.load(ckpt_path)
    training_module_state_dict = ckpt["training_module"]
    model_state_dict = {
        k[len(model_prefix) :]: v
        for k, v in training_module_state_dict.items()
        if k.startswith(model_prefix)
    }
    model.load_state_dict(model_state_dict, strict=strict)


def load_training(
    fabric: Fabric,
    ckpt_path: str,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    training_module: Module,
) -> tuple[int, int]:
    """Load training module and optimizer states, returning epoch and step."""
    d = fabric.load(ckpt_path)
    optimizer.load_state_dict(d["optimizer"])
    scheduler.load_state_dict(d["scheduler"])
    training_module.load_state_dict(d["training_module"])
    return d["epoch"], d["step"]


### Save ###


def save_training(
    fabric: Fabric,
    path: str,
    epoch: int,
    step: int,
    optimizer_state_dict: dict,
    scheduler_state_dict: dict,
    training_module_state_dict: dict,
):
    """Save training state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict,
        "training_module": training_module_state_dict,
    }
    fabric.save(path, state)
