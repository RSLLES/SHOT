# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""U-Net model.

Modified from https://github.com/milesial/Pytorch-UNet/tree/master .
"""

from torch import nn

from .unet_parts import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """UNet model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        init_features: int = 64,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inc = DoubleConv(
            in_channels, init_features, norm_layer=norm_layer, act_layer=act_layer
        )
        self.downs = nn.ModuleList()
        features = init_features
        for _ in range(depth):
            self.downs.append(
                Down(
                    features,
                    features * 2,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            features *= 2
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(
                Up(
                    features,
                    features // 2,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            features //= 2
        self.head = OutConv(features, out_channels)

    def forward(self, x):  # noqa: D102
        x = self.inc(x)

        X = [x]
        for down in self.downs:
            X.append(down(X[-1]))

        x = X.pop(-1)
        for up, x_skip in zip(self.ups, X[::-1]):
            x = up(x, x_skip)

        x = self.head(x)
        return x
