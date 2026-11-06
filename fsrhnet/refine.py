"""FSCoRefine: context adaptation followed by scan-output refinement."""

import torch
from torch import nn

from .context import SpatialVariationDescriptor, SharedContextBank
from .layers import ConvBN, ConvBNReLU6, GRN
from .scan import RoutedSelectiveScan2D


class ScanRefinementBottleneck(nn.Module):
    """144 -> 36 -> routed SS2D -> 144, preserving the route handoff."""

    def __init__(self, in_features=144, hidden_features=36, out_features=144):
        super().__init__()
        self.input_projection = ConvBNReLU6(in_features, hidden_features, 1)
        self.routed_scan = RoutedSelectiveScan2D(d_model=hidden_features, d_state=16, dropout=0.0)
        self.output_projection = ConvBN(hidden_features, out_features, 1)
        self.activation = nn.ReLU6()

    def forward(self, x, spatial_descriptor, scale_weights, context_bank):
        x = self.input_projection(x)
        x, route_map = self.routed_scan(
            x, spatial_descriptor=spatial_descriptor,
            scale_weights=scale_weights, context_bank=context_bank,
        )
        return self.activation(self.output_projection(x)), route_map


class FSCoRefine(nn.Module):
    """Return refined features D_s and a differentiable one-channel route R_s.

    Two residual updates are sequential: context adaptation, then scan-output
    refinement. The descriptor and scale weights are computed only once.
    The shared context bank belongs to this stage and is reused by its scan.
    """

    def __init__(self, dim=144):
        super().__init__()
        self.context_norm = nn.BatchNorm2d(dim)
        self.scan_norm = nn.BatchNorm2d(dim)
        self.spatial_descriptor = SpatialVariationDescriptor()
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(3, 16, 1, bias=True), nn.GELU(), nn.Conv2d(16, 3, 1, bias=True),
        )
        self.context_bank = SharedContextBank((9, 15, 21))
        self.context_projection = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU(), GRN(dim),
        )
        self.context_scale = nn.Parameter(torch.ones(dim) * 1e-3)
        # The published main model uses drop_path=0 at all three stages.
        self.drop_path = nn.Identity()
        self.scan_refinement = ScanRefinementBottleneck(dim, dim // 4, dim)

    def forward(self, x):
        normalized = self.context_norm(x)
        descriptor = self.spatial_descriptor(normalized)
        alpha = torch.softmax(self.scale_predictor(descriptor), dim=1)
        context = self.context_bank(normalized, alpha)
        context = self.context_projection(context)
        context = context * self.context_scale.view(1, -1, 1, 1)
        x = x + self.drop_path(context)
        scan_residual, route_map = self.scan_refinement(
            self.scan_norm(x), descriptor, alpha, self.context_bank,
        )
        return x + self.drop_path(scan_residual), route_map
