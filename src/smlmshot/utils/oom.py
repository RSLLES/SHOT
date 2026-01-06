# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility decorator to detect OOM and restart training with a reduced batch size."""

from functools import wraps

import torch
from omegaconf import DictConfig


def handle_oom(cfg_attr_name="cfg"):
    """Handle out-of-memory errors by retrying with reduced batch size."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the cfg object from args or kwargs
            cfg = kwargs.get(cfg_attr_name)
            if cfg is None:
                for arg in args:
                    if isinstance(arg, DictConfig):
                        cfg = arg
                        break
            if cfg is None:
                raise ValueError("DictConfig not found in arguments.")

            cfgr = cfg.runtime
            while True:
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    oom = "out of memory" in str(e)
                    oom = oom or "OOM" in str(e)
                    if not oom:
                        raise
                    if cfgr.batch_size == 1:
                        print(
                            "OOM encountered. Batch size is already 1. "
                            "Cannot reduce further."
                        )
                        raise
                    cfgr.batch_size = max(1, cfgr.batch_size // 2)
                    cfgr.n_accum_steps = 2 * cfgr.n_accum_steps
                    print(
                        f"OOM encountered. Retry with batch_size={cfgr.batch_size} "
                        f"and n_accum_steps={cfgr.n_accum_steps}."
                    )
                    torch.cuda.empty_cache()

        return wrapper

    return decorator
