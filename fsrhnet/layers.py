"""Small convolutional blocks used by the FSRH-Net decoder.

Sequential indices preserve checkpoint keys. GRN implements spatial-L2
residual normalization. Source attribution is in THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn


class ConvBNReLU6(nn.Sequential):
    """Convolution, batch normalization, and ReLU6 with stable key indices."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
        norm_layer=nn.BatchNorm2d, groups=1, bias=False,
    ):
        super(ConvBNReLU6, self).__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                dilation=dilation, stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=groups,
            ),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    """Channel projection followed by batch normalization."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
        norm_layer=nn.BatchNorm2d, bias=False,
    ):
        super(ConvBN, self).__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                dilation=dilation, stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    """Convolution wrapped to preserve its checkpoint key prefix."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
        bias=False,
    ):
        super(Conv, self).__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                dilation=dilation, stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            )
        )


class DepthwiseSeparableConvBNReLU6(nn.Sequential):
    """Depthwise/pointwise projection, each retaining its own normalization."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(DepthwiseSeparableConvBNReLU6, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class GRN(nn.Module):
    """Spatial-L2 residual normalization (not channel GRN)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta
