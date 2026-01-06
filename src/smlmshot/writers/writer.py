# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from torch import Tensor


class WriterInterface(ABC):
    """Interface to write data in a specific format."""

    @abstractmethod
    def open(self):
        """Open the writing destination."""
        pass

    @abstractmethod
    def close(self):
        """Close the writing destination."""
        pass

    @abstractmethod
    def _write(self, data: Tensor):
        """Implement the writing procedure."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, data: Tensor, **kwargs):
        """Write data that is a Tensor of shape (N, 5) with rows (frame, x, y, z, n)."""
        if not isinstance(data, Tensor) or data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(
                "data must be a 2D of shape (N, 5) with rows (frame, x, y, z, n)"
            )
        self._write(data, **kwargs)
