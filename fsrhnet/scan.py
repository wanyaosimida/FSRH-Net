"""Eight-direction selective scanning and spatial-variation-guided refinement.

Eight ordered scan sequences are grouped by effective step size, refined,
scattered back to their direction-channel indices, and inverse-merged. The
refinement modulation factor is returned directly as the route for Route-HSSF.
The native selective-scan kernel is supplied by ``mamba_ssm``; no alternative
kernel is silently selected. See THIRD_PARTY_NOTICES.md for the Mamba/VMamba
and MF-Mamba source attribution.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def selective_scan_fn(*args, **kwargs):
    """Load the native Mamba operator, or report an unavailable dependency."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as scan
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "FSRH-Net requires the mamba_ssm selective-scan operator for forward "
            "execution. Install a mamba_ssm build compatible with PyTorch and CUDA; "
            "this model does not silently fall back to another scan implementation."
        ) from exc
    return scan(*args, **kwargs)


def diagonal_gather(tensor):
    """Gather wrapped diagonals, including rectangular feature maps."""
    batch, channels, height, width = tensor.shape
    shift = torch.arange(height, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(width, device=tensor.device)) % width
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1, -1)
    return tensor.gather(3, index).transpose(-1, -2).reshape(
        batch, channels, height * width
    )


def antidiagonal_gather(tensor):
    """Gather wrapped anti-diagonals."""
    batch, channels, height, width = tensor.shape
    shift = torch.arange(height, device=tensor.device).unsqueeze(1)
    index = (torch.arange(width, device=tensor.device) - shift) % width
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1, -1)
    return tensor.gather(3, index).transpose(-1, -2).reshape(
        batch, channels, height * width
    )


def diagonal_scatter(tensor_flat, original_shape):
    """Invert :func:`diagonal_gather` without changing its spatial convention."""
    batch, channels, height, width = original_shape
    shift = torch.arange(height, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(width, device=tensor_flat.device)) % width
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1, -1)
    output = tensor_flat.new_zeros(batch, channels, height, width)
    output.scatter_(3, index, tensor_flat.reshape(batch, channels, width, height).transpose(-1, -2))
    return output


def antidiagonal_scatter(tensor_flat, original_shape):
    """Invert :func:`antidiagonal_gather`."""
    batch, channels, height, width = original_shape
    shift = torch.arange(height, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(width, device=tensor_flat.device) - shift) % width
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1, -1)
    output = tensor_flat.new_zeros(batch, channels, height, width)
    output.scatter_(3, index, tensor_flat.reshape(batch, channels, width, height).transpose(-1, -2))
    return output


class CrossScan(torch.autograd.Function):
    """Map NCHW features to eight ordered sequences with inverse-scatter gradients."""

    @staticmethod
    def forward(ctx, x):
        batch, channels, height, width = x.shape
        ctx.shape = (batch, channels, height, width)
        scans = x.new_empty((batch, 8, channels, height * width))
        scans[:, 0] = x.flatten(2, 3)
        scans[:, 1] = x.transpose(2, 3).flatten(2, 3)
        scans[:, 2:4] = scans[:, 0:2].flip(-1)
        scans[:, 4] = diagonal_gather(x)
        scans[:, 5] = antidiagonal_gather(x)
        scans[:, 6:8] = scans[:, 4:6].flip(-1)
        return scans

    @staticmethod
    def backward(ctx, ys):
        batch, channels, height, width = ctx.shape
        length = height * width
        axial = ys[:, 0:2] + ys[:, 2:4].flip(-1).view(batch, 2, -1, length)
        axial = axial[:, 0] + axial[:, 1].view(batch, -1, width, height).transpose(
            2, 3
        ).contiguous().view(batch, -1, length)
        axial = axial.view(batch, -1, height, width)
        diagonal = ys[:, 4:6] + ys[:, 6:8].flip(-1).view(batch, 2, -1, length)
        diagonal = diagonal_scatter(diagonal[:, 0], ctx.shape) + antidiagonal_scatter(
            diagonal[:, 1], ctx.shape
        )
        return axial + diagonal


class CrossMerge(torch.autograd.Function):
    """Inverse-align and sum eight scan outputs (the feature merge is not averaged)."""

    @staticmethod
    def forward(ctx, ys):
        batch, directions, channels, height, width = ys.shape
        ctx.shape = (height, width)
        ys = ys.view(batch, directions, channels, -1)
        axial = ys[:, 0:2] + ys[:, 2:4].flip(-1).view(batch, 2, channels, -1)
        axial = axial[:, 0] + axial[:, 1].view(batch, -1, width, height).transpose(
            2, 3
        ).contiguous().view(batch, channels, -1)
        axial = axial.view(batch, -1, height, width)
        diagonal = ys[:, 4:6] + ys[:, 6:8].flip(-1).view(batch, 2, channels, -1)
        shape = (batch, channels, height, width)
        diagonal = diagonal_scatter(diagonal[:, 0], shape) + antidiagonal_scatter(
            diagonal[:, 1], shape
        )
        return (axial + diagonal).view(batch, channels, -1)

    @staticmethod
    def backward(ctx, x):
        height, width = ctx.shape
        batch, channels, length = x.shape
        scans = x.new_empty((batch, 8, channels, length))
        scans[:, 0] = x
        scans[:, 1] = x.view(batch, channels, height, width).transpose(2, 3).flatten(2, 3)
        scans[:, 2:4] = scans[:, 0:2].flip(-1)
        scans[:, 4] = diagonal_gather(x.view(batch, channels, height, width))
        scans[:, 5] = antidiagonal_gather(x.view(batch, channels, height, width))
        scans[:, 6:8] = scans[:, 4:6].flip(-1)
        return scans.view(batch, 8, channels, height, width)


def partition_scan_outputs(delta, delta_bias, retain_ratio=0.5):
    """Partition scan channels by descending effective step size at every token.

    Keep ``torch.sort`` and its tie behavior: replacing this with a
    top-k operation would not establish equivalence at tied step sizes.
    """
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    delta = F.softplus(delta)
    channels = delta.shape[1]
    retain_channels = int(channels * retain_ratio)
    _, sorted_indices = torch.sort(delta, descending=True, dim=1)
    retain_indices, refine_indices = torch.split(
        sorted_indices, [retain_channels, channels - retain_channels], dim=1
    )
    return retain_indices.permute(0, 2, 1), refine_indices.permute(0, 2, 1)


class RoutedSelectiveScan2D(nn.Module):
    """FSCoRefine scan-output refinement and route handoff to Route-HSSF.

    Inputs are ``x: [B, d_model, H, W]``, a three-channel spatial descriptor,
    three scale-weight maps, and the *same* context-bank instance used before
    the scan. Returns ``(features: [B,d_model,H,W], route_map: [B,1,H,W])``.
    The context bank is passed explicitly and is never registered as a child
    here; its parameters belong to the enclosing FSCoRefine block.
    """

    def __init__(self, d_model=36, d_state=16, dropout=0.0):
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if not isinstance(d_state, int) or d_state <= 0:
            raise ValueError("d_state must be a positive integer.")
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = 2 * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.num_directions = 8
        self.retain_ratio = 0.5

        # Standard selective-state-space projections and transition parameters.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner, bias=True
        )
        self.act = nn.SiLU()
        projections = [
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(self.num_directions)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([p.weight for p in projections]))
        dt_projections = [self._dt_init(self.dt_rank, self.d_inner) for _ in range(self.num_directions)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projections]))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projections]))
        state_a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(state_a).repeat(self.num_directions, 1))
        self.A_logs._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(self.num_directions * self.d_inner))
        self.Ds._no_weight_decay = True
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.route_predictor = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1, bias=True),
        )
        self.modulation_gain = nn.Parameter(torch.full((2,), 0.05, dtype=torch.float32))
        self.refinement_gain = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    @staticmethod
    def _dt_init(dt_rank, d_inner):
        """Random step-size projection with inverse-softplus bias initialization."""
        projection = nn.Linear(dt_rank, d_inner, bias=True)
        bound = dt_rank ** -0.5
        nn.init.uniform_(projection.weight, -bound, bound)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inverse_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_dt)
        projection.bias._no_reinit = True
        return projection

    @staticmethod
    def _normalize_maps(x, eps=1e-6):
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(variance + eps)

    def _modulation_factors(self, dts, dt_bias, spatial_descriptor, height, width, dtype, device):
        effective_dts = F.softplus(dts + dt_bias.view(1, -1, 1))
        state_summary = torch.cat(
            [effective_dts.mean(dim=1, keepdim=True), effective_dts.amax(dim=1, keepdim=True)],
            dim=1,
        ).reshape(dts.shape[0], 2, height, width)
        if spatial_descriptor.shape[-2:] != (height, width):
            spatial_descriptor = F.interpolate(spatial_descriptor, size=(height, width), mode="bilinear", align_corners=False)
        routing_input = torch.cat(
            [self._normalize_maps(spatial_descriptor.float()), self._normalize_maps(state_summary.float())],
            dim=1,
        ).to(device=device, dtype=self.route_predictor[0].weight.dtype)
        beta = torch.softmax(self.route_predictor(routing_input), dim=1)
        beta = beta.permute(0, 2, 3, 1).reshape(dts.shape[0], height * width, 2).to(dtype=dtype)
        gains = torch.tanh(self.modulation_gain).to(device=device, dtype=dtype)
        return (
            1.0 + gains[0] * (2.0 * beta[..., 0:1] - 1.0),
            1.0 + gains[1] * (2.0 * beta[..., 1:2] - 1.0),
        )

    def _refine_scan_outputs(self, x, spatial_descriptor, scale_weights, context_bank):
        batch, _, height, width = x.shape
        length = height * width
        scans = CrossScan.apply(x)
        projected = torch.einsum(
            "b k d l, k c d -> b k c l", scans.view(batch, self.num_directions, -1, length), self.x_proj_weight
        )
        dts, bs, cs = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(batch, self.num_directions, -1, length), self.dt_projs_weight
        )
        scans = scans.float().view(batch, -1, length)
        dts = dts.contiguous().float().view(batch, -1, length)
        bs = bs.float().view(batch, self.num_directions, -1, length)
        cs = cs.float().view(batch, self.num_directions, -1, length)
        ds = self.Ds.float().view(-1)
        state_a = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        scan_output = selective_scan_fn(
            scans, dts, state_a, bs, cs, ds,
            z=None, delta_bias=dt_bias, delta_softplus=True, return_last_state=False,
        )
        batch, scan_channels, length = scan_output.shape
        routed = scan_output.clone()
        retain_index, refine_index = partition_scan_outputs(
            dts.view(batch, scan_channels, length), dt_bias, self.retain_ratio
        )
        device = routed.device
        batch_index = torch.arange(batch, device=device)[:, None, None]
        token_index = torch.arange(length, device=device)[None, :, None]
        retain_factor, refine_factor = self._modulation_factors(
            dts, dt_bias, spatial_descriptor, height, width, routed.dtype, device
        )

        retain_outputs = routed[batch_index, retain_index, token_index].to(routed.dtype)
        routed[batch_index, retain_index, token_index] = retain_outputs * retain_factor
        refine_outputs = routed[batch_index, refine_index, token_index]
        refine_channels = refine_outputs.shape[-1]
        refine_grid = refine_outputs.transpose(1, 2).reshape(batch, refine_channels, height, width)
        refined = context_bank(refine_grid, scale_weights)
        gain = torch.tanh(self.refinement_gain).to(device=refine_grid.device, dtype=refine_grid.dtype)
        refine_grid = refine_grid + gain * refined
        refine_outputs = refine_grid.flatten(2).transpose(1, 2).contiguous()
        routed[batch_index, refine_index, token_index] = refine_outputs.to(routed.dtype) * refine_factor

        routed = routed.view(batch, self.num_directions, scan_channels // self.num_directions, height, width)
        merged = CrossMerge.apply(routed)
        return merged.transpose(1, 2).reshape(batch, height, width, -1), refine_factor

    def forward(self, x, spatial_descriptor, scale_weights, context_bank):
        if x.ndim != 4 or x.shape[1] != self.d_model:
            raise ValueError(f"x must have shape [B, {self.d_model}, H, W].")
        for name, tensor in (("spatial_descriptor", spatial_descriptor), ("scale_weights", scale_weights)):
            if tensor is None or tensor.ndim != 4 or tensor.shape[:2] != (x.shape[0], 3):
                raise ValueError(f"{name} must have shape [B, 3, H, W].")
        if context_bank is None:
            raise ValueError("FSCoRefine requires the enclosing block's context_bank.")
        batch, _, height, width = x.shape
        projected = self.in_proj(x.permute(0, 2, 3, 1))
        scan_input, gate = projected.chunk(2, dim=-1)
        scan_input = scan_input.permute(0, 3, 1, 2).contiguous()
        gate = F.silu(gate)
        scan_input = self.act(self.conv2d(scan_input))
        merged, refine_factor = self._refine_scan_outputs(
            scan_input, spatial_descriptor, scale_weights, context_bank
        )
        merged = self.out_norm(merged).to(gate.dtype)
        output = self.out_proj(merged * gate)
        if self.dropout is not None:
            output = self.dropout(output)
        output = output.permute(0, 3, 1, 2)

        # The route uses scan-token coordinates directly, as defined in the paper.
        # Do not inverse-align/average this route with CrossMerge.
        route_map = refine_factor.reshape(batch, height, width, 1).permute(0, 3, 1, 2).contiguous()
        return output, route_map.to(dtype=output.dtype)
