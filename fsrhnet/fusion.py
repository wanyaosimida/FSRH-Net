"""Route-HSSF: feature-gated semantics and scan-informed structural correction."""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNGELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )


class LevelAlignedSkipProjection(nn.Module):
    """Use only the matching HRNet branch: pointwise/pointwise/depthwise."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_projection = ConvBNGELU(self.in_channels, self.out_channels, 1)
        self.refinement = nn.Sequential(
            ConvBNGELU(self.out_channels, self.out_channels, 1),
            ConvBNGELU(self.out_channels, self.out_channels, 3, groups=self.out_channels),
        )

    def forward(self, encoder_feature):
        if encoder_feature.ndim != 4 or encoder_feature.shape[1] != self.in_channels:
            raise ValueError(f"encoder_feature must be NCHW with {self.in_channels} channels.")
        return self.refinement(self.input_projection(encoder_feature))


class RouteHSSF(nn.Module):
    """Fuse decoder/encoder features; route enters only the structural term.

    At receiving level s, route_map is R_(s+1) from the preceding FSCoRefine
    stage. Its resized and centered factor R'_s modulates the structural
    residual. The semantic matcher receives only encoder/decoder features;
    an all-one incoming route is neutral.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        hidden = max(self.channels // 4, 8)
        self.skip_norm = nn.BatchNorm2d(self.channels)
        self.decoder_norm = nn.BatchNorm2d(self.channels)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(self.channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True), nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1,
                      groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels), nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
        )
        self.structure_scale = nn.Parameter(torch.tensor(0.25))
        self.route_coupling = nn.Parameter(torch.tensor(0.50))
        self.residual_scale = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.semantic_gate[-2].weight)
        nn.init.zeros_(self.semantic_gate[-2].bias)

    @staticmethod
    def _structure_residual(skip):
        padded = F.pad(skip, (1, 1, 1, 1), mode="replicate")
        return skip - F.avg_pool2d(padded, kernel_size=3, stride=1)

    def _validate_inputs(self, decoder, skip, route_map):
        if decoder.ndim != 4 or skip.ndim != 4:
            raise ValueError("decoder and skip must be NCHW tensors.")
        if decoder.shape[0] != skip.shape[0]:
            raise ValueError("decoder and skip batch sizes must match.")
        if decoder.shape[1] != self.channels or skip.shape[1] != self.channels:
            raise ValueError(f"decoder and skip must have {self.channels} channels.")
        if route_map is None or route_map.ndim != 4:
            raise ValueError("route_map must have shape [B, 1, H, W].")
        if route_map.shape[:2] != (skip.shape[0], 1):
            raise ValueError("route_map must have one channel and matching batch.")

    def forward(self, decoder, skip, route_map):
        self._validate_inputs(decoder, skip, route_map)
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:],
                                    mode="bilinear", align_corners=False)
        if route_map.shape[-2:] != skip.shape[-2:]:
            route_map = F.interpolate(route_map, size=skip.shape[-2:],
                                      mode="bilinear", align_corners=False)
        skip_norm = self.skip_norm(skip)
        decoder_norm = self.decoder_norm(decoder)
        gate = self.semantic_gate(torch.cat(
            [skip_norm * decoder_norm, torch.abs(skip_norm - decoder_norm)], dim=1,
        ))
        semantic_correction = gate * (skip - decoder)
        structure = self._structure_residual(skip)
        structure_gain = torch.tanh(self.structure_scale)
        route_gain = torch.tanh(self.route_coupling)
        # R'_s is derived from the incoming R_(s+1), independently of the gate.
        route_factor = 1.0 + route_gain * (route_map - 1.0)
        correction = semantic_correction + (
            structure_gain * route_factor * (1.0 - gate) * structure
        )
        return decoder + torch.tanh(self.residual_scale) * self.output_projection(correction)
