# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor, nn

from smlmshot.utils.format import format_string


class LayerNorm(nn.Module):
    """Layer Normalization that supports channel first (B, C, **) or last (B, **, C)."""

    def __init__(self, n_features: int, channel_mode: str):
        super().__init__()
        channel_mode = format_string(channel_mode)
        if channel_mode == "first":
            self.layer = nn.GroupNorm(1, n_features)
        elif channel_mode == "last":
            self.layer = nn.LayerNorm(n_features)
        else:
            raise ValueError("channel_mode should be either 'first' or 'last'")

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.layer(x)


class LayerNormChannelFirst(LayerNorm):
    """Layer normalization for (B, C, **) input."""

    def __init__(self, n_features: int):
        super().__init__(n_features, channel_mode="first")


class LayerNormChannelLast(LayerNorm):
    """Layer normalization for (B, **, C) input."""

    def __init__(self, n_features: int):
        super().__init__(n_features, channel_mode="last")
