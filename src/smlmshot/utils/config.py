# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for hydra configuration."""

import yaml
from omegaconf import DictConfig, OmegaConf

from smlmshot.utils.git import get_head_commit_hash


def initialize_config(cfg: DictConfig):
    """Initialize configuration with resolvers and git commit hash."""
    OmegaConf.set_struct(cfg, False)
    if not OmegaConf.has_resolver("pow2"):
        OmegaConf.register_new_resolver("pow2", lambda n: 2 ** int(n))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg.runtime.git_commit_hash = get_head_commit_hash()
    return cfg


def add_git_commit_hash(cfg: DictConfig):
    """Add git commit hash to configuration."""
    cfg.runtime.git_commit_hash = get_head_commit_hash()
    return cfg


def add_eff_batch_size(cfg: DictConfig, world_size: int):
    """Compute effective batch size and assign to configuration."""
    cfg.runtime.eff_batch_size = (
        cfg.runtime.batch_size * world_size * cfg.runtime.n_accum_steps
    )
    return cfg


def add_total_steps(cfg: DictConfig, step_per_epoch: int):
    """Compute total training steps and assign to configuration."""
    if cfg.runtime.n_epochs > 0:
        cfg.runtime.total_steps = (
            cfg.runtime.n_epochs * step_per_epoch // cfg.runtime.eff_batch_size
        )
    else:
        cfg.runtime.total_steps = -1
    return cfg


def dump_config(cfg: DictConfig) -> str:
    """Dump configuration to YAML string."""
    return yaml.dump(OmegaConf.to_container(cfg), default_flow_style=False)


def load_config(filepath: str):
    """Load a config from a yaml file."""
    with filepath.open() as f:
        cfg_dict = yaml.safe_load(f)
    return OmegaConf.create(cfg_dict)
