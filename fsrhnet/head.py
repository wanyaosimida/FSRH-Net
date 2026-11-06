"""Segmentation head combining multiscale, channel and spatial responses.

The inactive compatibility projection remains registered for checkpoint and
parameter-count compatibility. Output resizing belongs to the decoder.
Source attribution is provided in THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .layers import Conv, DepthwiseSeparableConvBNReLU6


class SpatialAttention(nn.Module):
    """Channel-pooled spatial gate with ReLU before sigmoid."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)

        out2 = self.relu1(out1)

        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        return y


class DilatedDepthwiseSeparableConv(nn.Module):
    """Dilated depthwise convolution followed by a pointwise projection."""

    def __init__(self, nin, nout, dilation):
        super(DilatedDepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=3, dilation=dilation,
            padding=(dilation * 2) // 2, groups=nin,
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DenseAtrousBlock(nn.Sequential):
    """Pointwise bottleneck and dilated spatial response for a dense branch."""

    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DenseAtrousBlock, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1))
        self.add_module(
            'bn1', norm_layer(
                inter_channels, **({} if norm_kwargs is None else norm_kwargs)
            )
        )
        self.add_module('relu1', nn.ReLU(True))
        self.add_module(
            'conv2', DilatedDepthwiseSeparableConv(
                nin=inter_channels, nout=out_channels, dilation=atrous_rate
            )
        )
        self.add_module(
            'bn2', norm_layer(
                out_channels, **({} if norm_kwargs is None else norm_kwargs)
            )
        )
        self.add_module('relu2', nn.ReLU(True))
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(DenseAtrousBlock, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class DenseAtrousPyramid(nn.Module):
    """Densely connected atrous responses with multiplicative input fusion."""

    def __init__(self, in_channels, atrous_rates):
        super(DenseAtrousPyramid, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates

        self.branch1 = DenseAtrousBlock(
            in_channels, out_channels // 2, out_channels, rate1, 0.1
        )
        self.branch2 = DenseAtrousBlock(
            in_channels + out_channels * 1, (in_channels + out_channels * 1) // 2,
            out_channels, rate2, 0.1,
        )
        self.branch3 = DenseAtrousBlock(
            in_channels + out_channels * 2, (in_channels + out_channels * 2) // 2,
            out_channels, rate3, 0.1,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(in_channels + 3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        u = x.clone()
        aspp1 = self.branch1(x)
        x = torch.cat([aspp1, x], dim=1)

        aspp2 = self.branch2(x)
        x = torch.cat([aspp2, x], dim=1)

        aspp3 = self.branch3(x)
        x = torch.cat([aspp3, x], dim=1)
        return self.output_projection(x) * u


class SegmentationHead(nn.Module):
    """Combine atrous, channel, and spatial responses into class logits.

    FSRH-Net uses dim=18 and dilation=(6, 12, 18). The inactive
    compatibility projection preserves tensors required by model checkpoints.
    """

    def __init__(
        self, dim, dilation=(3, 5, 7), fc_ratio=4, dropout=0., num_classes=6,
    ):
        super(SegmentationHead, self).__init__()
        self.dim = dim
        self.multiscale_context = DenseAtrousPyramid(dim, dilation)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = SpatialAttention()
        # Checkpoint-compatible projection; not used in forward.
        self.compatibility_projection = Conv(dim, 2 * dim, 1)
        self.classifier = nn.Sequential(
            DepthwiseSeparableConvBNReLU6(dim, dim, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(18, num_classes, kernel_size=1),
        )

    def forward(self, x):

        u = x.clone()
        attn = self.multiscale_context(x)

        c_attn = self.channel_pool(x)
        c_attn = self.channel_gate(c_attn)
        c_attn = c_attn * u

        s_attn = self.spatial_gate(x)
        out = self.classifier(attn + c_attn + s_attn)
        return out
