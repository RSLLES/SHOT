# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.utils.data import Dataset, default_collate


class ImagesDataset(Dataset):
    """Slide a n_frames window over a stack of images, stored as a [N, H, W] tensor."""

    def __init__(self, y: Tensor, window: int) -> None:
        if y.ndim != 3:
            raise ValueError("ImageDataset expects a 3D tensor [N, H, W]")
        if window < 1:
            raise ValueError("window size should be >= 1")
        self.y = y
        self.window = window
        self.collate_fn = default_collate

    def __len__(self) -> int:
        return self.y.size(0) - self.window + 1

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.__len__():
            raise StopIteration()
        y = self.y[index : index + self.window]
        return {"y": y}
