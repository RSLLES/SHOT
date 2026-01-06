# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools

from torch import Tensor


class HashableTensor:
    """Make tensor hashable by using their memory ID."""

    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    def __hash__(self):
        return id(self.tensor)

    def __eq__(self, other):
        return self.tensor is other.tensor


def lru_cache(maxsize=128):
    """Adaptation of lru_cache that handles Tensors hashing by id."""

    def decorator(func):
        # exit decorator: unwrap tensors
        @functools.lru_cache(maxsize=maxsize)
        def cached_wrapper(*args, **kwargs):
            args = tuple(a.tensor if isinstance(a, HashableTensor) else a for a in args)
            kwargs = {
                k: v.tensor if isinstance(v, HashableTensor) else v
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)

        # entry decorator: wrap tensor
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = tuple(
                HashableTensor(a) if isinstance(a, Tensor) else a for a in args
            )
            kwargs = {
                k: HashableTensor(v) if isinstance(v, Tensor) else v
                for k, v in kwargs.items()
            }
            return cached_wrapper(*args, **dict(kwargs))

        return wrapper

    return decorator
