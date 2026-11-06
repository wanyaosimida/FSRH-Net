"""Spatial variation descriptors and the within-stage shared context bank.

These spatial operators describe edges and texture; they are not a Fourier
or wavelet decomposition. See the paper's spatial variation/context section.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SpatialVariationDescriptor(nn.Module):
    """Channel-averaged Sobel magnitude, absolute Laplacian, local variance."""

    def __init__(self):
        super().__init__()
        kernels = {
            "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "laplace": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        }
        for name, values in kernels.items():
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.float32).view(1, 1, 3, 3),
                persistent=False,
            )

    @staticmethod
    def _depthwise_filter(x, kernel):
        channels = x.shape[1]
        weight = kernel.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=channels)

    def forward(self, x):
        grad_x = self._depthwise_filter(x, self.sobel_x)
        grad_y = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mean_square = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1)
        variance = (mean_square - mean * mean).clamp_min(0.0)
        return torch.cat(
            [grad_mag.mean(1, keepdim=True), lap.abs().mean(1, keepdim=True),
             variance.mean(1, keepdim=True)], dim=1,
        )


class ChannelSharedContextBranch(nn.Module):
    """Factorized large spatial kernel plus a learned 3x3 compensation."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.kernel_h = nn.Parameter(torch.empty(1, 1, 1, self.kernel_size))
        self.kernel_v = nn.Parameter(torch.empty(1, 1, self.kernel_size, 1))
        self.kernel_s = nn.Parameter(torch.empty(1, 1, 3, 3))
        nn.init.normal_(self.kernel_h, mean=0.0, std=0.02)
        nn.init.normal_(self.kernel_v, mean=0.0, std=0.02)
        nn.init.normal_(self.kernel_s, mean=0.0, std=0.02)

    @staticmethod
    def _repeat_kernel(kernel, channels, x):
        return kernel.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)

    def forward(self, x):
        channels = x.shape[1]
        horizontal = F.conv2d(
            x, self._repeat_kernel(self.kernel_h, channels, x),
            padding=(0, self.kernel_size // 2), groups=channels,
        )
        peripheral = F.conv2d(
            horizontal, self._repeat_kernel(self.kernel_v, channels, x),
            padding=(self.kernel_size // 2, 0), groups=channels,
        )
        local = F.conv2d(
            x, self._repeat_kernel(self.kernel_s, channels, x),
            padding=1, groups=channels,
        )
        return peripheral + local


class SharedContextBank(nn.Module):
    """Fine/Medium/Wide kernels shared across channels, not across stages.

The same instance and scale weights are used before the scan and on its
gathered local rank-slot pseudo-grid. No second context bank is constructed.
"""

    def __init__(self, kernel_sizes=(9, 15, 21)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.branches = nn.ModuleList(
            [ChannelSharedContextBranch(kernel) for kernel in self.kernel_sizes]
        )

    def forward(self, x, scale_weights):
        if x.ndim != 4 or scale_weights.ndim != 4:
            raise ValueError("features and scale_weights must be NCHW tensors.")
        if scale_weights.shape[:2] != (x.shape[0], len(self.branches)):
            raise ValueError("scale_weights must have matching batch and three channels.")
        if scale_weights.shape[-2:] != x.shape[-2:]:
            scale_weights = F.interpolate(
                scale_weights, size=x.shape[-2:], mode="bilinear", align_corners=False,
            )
            scale_weights = scale_weights.clamp_min(0.0)
            scale_weights = scale_weights / scale_weights.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-6)
        output = torch.zeros_like(x)
        for index, branch in enumerate(self.branches):
            output = output + branch(x) * scale_weights[:, index:index + 1]
        return output
