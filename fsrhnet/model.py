"""FSRH-Net main model: a coarse-to-fine refinement/fusion hierarchy."""

from torch import nn
import torch.nn.functional as F

from .backbone import build_hrnet_w18
from .checkpoint import remap_model_state_dict
from .fusion import LevelAlignedSkipProjection, RouteHSSF
from .head import SegmentationHead
from .layers import ConvBN
from .refine import FSCoRefine


class FSRHDecoder(nn.Module):
    """Three FSCoRefine/Route-HSSF pairs; no fourth refinement stage."""

    def __init__(self, num_classes=6, dropout=0.1):
        super().__init__()
        self.refine4 = FSCoRefine(144)
        self.refine3 = FSCoRefine(144)
        self.refine2 = FSCoRefine(144)
        self.shallow_projection = ConvBN(144, 18, 1)
        self.skip_projection3 = LevelAlignedSkipProjection(72, 144)
        self.skip_projection2 = LevelAlignedSkipProjection(36, 144)
        self.skip_projection1 = LevelAlignedSkipProjection(18, 18)
        self.fusion3 = RouteHSSF(144)
        self.fusion2 = RouteHSSF(144)
        self.fusion1 = RouteHSSF(18)
        self.segmentation_head = SegmentationHead(
            18, fc_ratio=4, dilation=(6, 12, 18), dropout=dropout, num_classes=num_classes,
        )

    def forward(self, encoder1, encoder2, encoder3, encoder4, out_h, out_w):
        features, route4 = self.refine4(encoder4)
        features = self.fusion3(features, self.skip_projection3(encoder3), route4)
        features, route3 = self.refine3(features)
        features = self.fusion2(features, self.skip_projection2(encoder2), route3)
        features, route2 = self.refine2(features)
        features = self.shallow_projection(features)
        features = self.fusion1(features, self.skip_projection1(encoder1), route2)
        return F.interpolate(self.segmentation_head(features), size=(out_h, out_w),
                             mode="bilinear", align_corners=False)


class FSRHNet(nn.Module):
    """FSRH-Net main model, returning unnormalized class logits.

    Args:
        num_classes: 6 for ISPRS or 8 for UAVid (other positive counts allowed).
        dropout: Prediction-head dropout probability, default 0.1.
        backbone_pretrained: Optional explicit HRNet backbone weight path.
            None performs random initialization, with no download or hidden path.

    Inputs are three-channel optical NCHW tensors with positive spatial sizes,
    including rectangles and odd dimensions. Channel order is NIR/Red/Green
    for Vaihingen and Red/Green/Blue for Potsdam and UAVid. Match preprocessing
    to the checkpoint; this module does not reorder or normalize image bands.
    Validated precisions are FP32 and CUDA FP16-autocast, not arbitrary floating
    dtypes. The native selective scan requires a compatible CUDA-enabled
    mamba-ssm installation. Training retains BatchNorm sample-count
    constraints at coarse resolutions.
    """

    def __init__(self, num_classes=6, dropout=0.1, backbone_pretrained=None):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.backbone = build_hrnet_w18(pretrained=backbone_pretrained)
        self.decoder = FSRHDecoder(num_classes=num_classes, dropout=dropout)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current or supported earlier model-only parameter names.

        Key conversion is deterministic and rejects collisions; tensor shapes,
        missing keys and unexpected keys are checked by PyTorch. Framework
        checkpoint containers and prefixes must be removed explicitly by the
        caller. Saved state dictionaries always use the current module names.
        """
        return super().load_state_dict(
            remap_model_state_dict(state_dict), strict=strict, assign=assign,
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[0] < 1:
            raise ValueError("FSRHNet expects a nonempty three-channel optical tensor of shape [B, 3, H, W].")
        height, width = x.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("H and W must be positive.")
        if not x.is_floating_point():
            raise TypeError("FSRHNet expects floating-point inputs with the original data normalization.")
        encoder1, encoder2, encoder3, encoder4 = self.backbone(x)
        return self.decoder(encoder1, encoder2, encoder3, encoder4, height, width)
