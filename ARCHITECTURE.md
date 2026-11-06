# Classes and forward flow

`FSRHNet` is the public model entry. Its encoder is `HRNetW18`; its decoder
alternates `FSCoRefine` and `RouteHSSF` (Route-HSSF in the paper) from coarse
to fine. FSCoRefine performs context-linked scan-output refinement; Route-HSSF
provides scan-informed structural skip conditioning alongside a separate
feature-gated semantic correction.

## Module hierarchy

```text
FSRHNet
├── backbone: HRNetW18
└── decoder: FSRHDecoder
    ├── refine4 / refine3 / refine2: FSCoRefine
    │   ├── context_norm
    │   ├── spatial_descriptor: SpatialVariationDescriptor
    │   ├── scale_predictor
    │   ├── context_bank: SharedContextBank
    │   │   └── branches: ChannelSharedContextBranch × 3
    │   ├── context_projection / context_scale
    │   ├── scan_norm
    │   └── scan_refinement: ScanRefinementBottleneck
    │       ├── input_projection
    │       ├── routed_scan: RoutedSelectiveScan2D
    │       └── output_projection / activation
    ├── skip_projection3 / skip_projection2 / skip_projection1
    │   └── LevelAlignedSkipProjection
    ├── fusion3 / fusion2 / fusion1: RouteHSSF
    ├── shallow_projection
    └── segmentation_head: SegmentationHead
        ├── multiscale_context: DenseAtrousPyramid
        ├── channel_pool / channel_gate
        ├── spatial_gate: SpatialAttention
        └── classifier
```

The three refinement stages have **separate** parameters. Within each stage,
the same context-bank object and scale-weight tensor are reused before scanning
and during local scan-output refinement. The route map, not the context bank,
is passed to the next fusion stage.

## Decoder sequence

For image input `I` with spatial size `(H, W)`:

```text
(E1, E2, E3, E4) = backbone(I)

(D4, R4) = refine4(E4)
X3 = fusion3(D4, skip_projection3(E3), R4)

(D3, R3) = refine3(X3)
X2 = fusion2(D3, skip_projection2(E2), R3)

(D2, R2) = refine2(X2)
X1 = fusion1(shallow_projection(D2), skip_projection1(E1), R2)

logits = resize(segmentation_head(X1), (H, W))
```

The encoder widths are 18, 36, 72 and 144. Refinement uses 144 channels;
`shallow_projection` reduces 144 to 18 before the finest fusion. Each route
has one channel. The head produces six ISPRS or eight UAVid logits and the
decoder resizes them to the input resolution. There is no fourth refinement.

## FSCoRefine interface

`FSCoRefine.forward(x) -> (refined_features, route_map)`

1. Normalize the input using `context_norm`.
2. Compute the three-map spatial variation descriptor (Sobel magnitude,
   absolute Laplacian, local variance), then predict three scale weights.
3. Apply the shared 9/15/21 context bank and projected context residual.
4. Apply `scan_norm`, then `scan_refinement`: 144 → 36 channels, routed scan,
   and 36 → 144 channel recovery.
5. Add the scan-output residual and return the feature and route.

`RoutedSelectiveScan2D.forward(x, spatial_descriptor, scale_weights,
context_bank)` expands the scan width to 72 and uses state dimension 16.
`CrossScan` produces eight directions. `partition_scan_outputs` sorts the
576 direction-channel outputs at each sequence index by effective positive
step size, dividing them into 288 retain-role and 288 refine-role entries.
The refine-role subset reuses the context bank and scale weights on its
rank-slot pseudo-grid. The subsets are scattered back before `CrossMerge`.
The applied refine factor is reshaped directly into `route_map`, without
directional averaging or inverse-aligning that route.

## Route-HSSF interface

`RouteHSSF.forward(decoder, skip, route_map) -> fused_features`

`LevelAlignedSkipProjection` uses only the matching encoder level. Fusion
resizes decoder features and the route to the skip resolution. The semantic
gate uses normalized encoder/decoder feature products and absolute differences.
The route modulates only the high-pass structural correction, not the semantic
gate or semantic difference. The projected correction is added residually to
the decoder feature.

The producer and receiver use different stage indices. FSCoRefine at stage
`s` exports `R_s` to the fusion at level `s-1`. Equivalently, a receiving
Route-HSSF at level `s` takes `R_(s+1)`. Its `route_map` argument is that
incoming factor. After spatial resizing, the code computes
`route_factor = 1 + tanh(route_coupling) * (route_map - 1)`, corresponding to
`R'_s` in the paper. This factor multiplies the structural branch only;
it is neither a replacement for the semantic gate nor a recurrent hidden state.

## Prediction and supporting classes

`SegmentationHead` combines `DenseAtrousPyramid`, a pooled channel gate and
`SpatialAttention`, then applies the classifier. `DenseAtrousBlock` uses
`DilatedDepthwiseSeparableConv`. The supporting layers include `ConvBNReLU6`,
`ConvBN`, `Conv`, `DepthwiseSeparableConvBNReLU6`, `ConvBNGELU` and `GRN`.
The backbone retains the standard HRNet `BasicBlock`, `Bottleneck` and
`HighResolutionModule` classes.

## Weight compatibility

Current state dictionaries use the module names above. The model loader
translates supported earlier model-only keys through `fsrhnet/checkpoint.py`,
with collision detection and normal PyTorch strict checks. No duplicate
registered aliases are used in the model hierarchy.

`backbone.compatibility_head` and
`decoder.segmentation_head.compatibility_projection` retain parameter tensors
needed by existing checkpoints and parameter-count accounting. They are not
called by the forward path and are not additional method components.
