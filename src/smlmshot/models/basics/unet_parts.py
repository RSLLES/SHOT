# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""U-Net parts.

Inspired by https://github.com/milesial/Pytorch-UNet/tree/master .
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Basic block for U-Nets."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            norm_layer(mid_channels),
            act_layer(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer(inplace=True),
        )

    def forward(self, x):  # noqa: D102
        return self.double_conv(x)


class Down(nn.Module):
    """Downscale -> double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )

    def forward(self, x):  # noqa: D102
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscale -> double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x1, x2):  # noqa: D102
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):  # noqa: D102
        return self.conv(x)
