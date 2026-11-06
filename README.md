# FSRH-Net — main model

This package contains **one main model**, FSRH-Net, for cross-stage
scan-informed conditioning in high-resolution remote sensing image segmentation.

FSCoRefine links spatial context adaptation with scan-output refinement through
stage-local kernel and scale-weight reuse. Route-HSSF combines feature-gated
semantic correction with structural correction conditioned by the refinement
route from the preceding stage.

It does not include training pipelines, datasets, comparison models, ablations,
or pretrained weights. Constructing the model alone does not reproduce the
paper's segmentation accuracy.

## Model structure

```text
Three-channel optical image
  └─ HRNet-W18 → E1 / E2 / E3 / E4
       E4 → FSCoRefine4 → Route-HSSF3 (+ E3, R4)
          → FSCoRefine3 → Route-HSSF2 (+ E2, R3)
          → FSCoRefine2 → Route-HSSF1 (+ E1, R2)
          → segmentation head → input-resolution logits
```

| File | Responsibility |
| --- | --- |
| `fsrhnet/model.py` | Public `FSRHNet` entry and three-stage decoder |
| `fsrhnet/backbone.py` | HRNet-W18, four multiresolution feature branches |
| `fsrhnet/context.py` | Spatial variation descriptor and shared context bank |
| `fsrhnet/refine.py` | FSCoRefine's sequential context/scan residual updates |
| `fsrhnet/scan.py` | Eight-way scan, ranked grouping, local refinement, route handoff |
| `fsrhnet/fusion.py` | Matching-level skip projection and Route-HSSF (`RouteHSSF`) |
| `fsrhnet/head.py` | Inherited segmentation head |
| `fsrhnet/layers.py` | Small shared convolutional blocks and GRN |
| `fsrhnet/checkpoint.py` | Exact key translation for supported earlier model weights |

There are three refinement stages, not four. The final decoder-channel
projection is 144 → 18. ISPRS uses six output classes; UAVid uses eight.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the class names, module interfaces
and complete forward sequence.

## Dependencies and use

The FSRH-Net source directly imports PyTorch and the native `mamba_ssm`
selective-scan operator, in addition to the standard library. Mamba has its own
transitive dependencies, including `einops` and the causal-convolution CUDA
extension. The dependency files include these supporting packages.

Follow [INSTALL.md](INSTALL.md) for the fixed-version Linux/CUDA setup,
installation order and native-operator checks. Runtime dependencies are in
`requirements.txt`, build tools in `requirements-build.txt`, and selected
transitive version constraints in `constraints-cu118.txt`. The instructions target the
[Mamba v1.1.3 documentation](https://github.com/state-spaces/mamba/blob/v1.1.3/README.md#installation),
not the changing default branch. This package does not silently substitute
another scan kernel when the native dependency fails. A compatible CUDA
installation is required; this package has no built-in CPU forward backend.

Run from this directory, or add it to `PYTHONPATH`:

```python
import torch
from fsrhnet import FSRHNet

model = FSRHNet(num_classes=6).cuda().eval()  # 8 for UAVid
image = torch.randn(1, 3, 512, 512, device="cuda")
with torch.no_grad():
    logits = model(image)                   # [1, 6, 512, 512]
```

This example uses random weights and a synthetic input to illustrate the
interface. Real inference requires trained weights and matching preprocessing.
Inputs are three-channel optical NCHW tensors with shape `[B, 3, H, W]`;
the channels are dataset-specific:

| Dataset | Input channel order | Output classes |
| --- | --- | --- |
| ISPRS Vaihingen | Near-infrared, Red, Green (NIR-R-G) | 6 |
| ISPRS Potsdam | Red, Green, Blue (RGB) | 6 |
| UAVid | Red, Green, Blue (RGB) | 8 |

For uint8 input images, convert to floating point,
divide by 255, then apply channel-wise mean `[0.485, 0.456, 0.406]` and standard
deviation `[0.229, 0.224, 0.225]` in the channel order above. Do not divide
already-scaled data by 255 again, and do not replace Vaihingen's NIR band with
blue. Match any other checkpoint's own preprocessing. The model does not
reorder bands or normalize image values internally; it uses no DSM/nDSM.

Validated precision modes are FP32 and CUDA FP16-autocast. Float64 and a direct
`model.half()` workflow are not claimed as supported. Rectangular and odd
spatial sizes are supported. Training retains the original BatchNorm
sample-count requirements at coarse resolutions. Use `eval()` for inference.

`FSRHNet(num_classes=6, dropout=0.1, backbone_pretrained=None)` is the default.
There is no automatic download or machine-specific weight path. An optional
`backbone_pretrained` path must contain an exact encoder-local state dict,
including the retained `compatibility_head`; its earlier `last_layer` name is
accepted by the loader. It is not an arbitrary ImageNet
classification checkpoint loader. Incompatible or missing files raise errors.

## Existing weights

Model parameters are organized under `backbone.*` and `decoder.*` using the
module names in [ARCHITECTURE.md](ARCHITECTURE.md). Load a model-only dictionary
with `model.load_state_dict(state_dict, strict=True)`. This accepts both the
current names and the supported earlier FSRH-Net parameter names; translation
is centralized in `fsrhnet/checkpoint.py`. It changes key names only, preserves
tensor values and module-version metadata, and rejects key collisions.
Missing/unexpected keys and incompatible tensor shapes still raise errors with
`strict=True`; no weights are silently skipped. `model.state_dict()` always
exports the current names.

A training-framework checkpoint wrapper or prefixes such as `net.` and
`module.` must be removed explicitly before calling the model loader. Do not
use `strict=False` to hide missing model weights. Compatibility is provided for
state dictionaries, not serialized whole-model objects, old private class
imports or training-framework/optimizer checkpoints. When FSRH-Net is inside a
larger wrapper, load its extracted model-only weights through this model's
`load_state_dict` method.

The six-class model has 10,000,717 parameters. The inherited but inactive
`backbone.compatibility_head` and
`decoder.segmentation_head.compatibility_projection` stay registered for checkpoint
and parameter-count compatibility; neither is used by the forward path.
No parameters are automatically frozen. To compare predictions across
implementations, load the same state dictionary; equal random seeds alone do
not establish identical initial weights.

The conversion can also be inspected without loading weights:

```python
from fsrhnet.checkpoint import remap_model_state_dict

converted = remap_model_state_dict(state_dict)
model.load_state_dict(converted, strict=True)
```

Here `state_dict` is an already extracted model-only parameter dictionary.

## Paper-to-code numerical contract

- Sobel magnitude, absolute Laplacian and local variance describe spatial
  variation, not a Fourier or wavelet decomposition.
- Within one stage, the 9/15/21 context bank and scale weights are reused
  before and after the scan. Different stages have separate learned banks.
- Effective positive step sizes rank 576 direction–channel outputs into
  288 retain-role and 288 refine-role entries at each sequence position.
  The refined outputs are scattered to their original indices before merge.
- The retain path has no extra self-attention. The local path works on the
  rank-slot pseudo-grid described in the paper, not a direction-aligned image.
- The applied local factor is **directly reshaped** into the one-channel route
  map. The route is not directionally averaged or inverse-aligned.
- Route-HSSF uses that route only in its structural correction. The semantic
  matcher receives encoder/decoder features, not the route.
- At receiving level `s`, the incoming `route_map` is `R_(s+1)` from the
  preceding FSCoRefine stage; the resized, centered `route_factor` is `R'_s`.
  The route combines spatial descriptors and effective-step statistics, not
  a co-located average of recurrent hidden states.
- HRNet retains SyncBatchNorm and its original interpolation settings;
  decoder interpolation uses `align_corners=False`.

## Runtime notes

- Use the native CUDA operator and the dependency versions described in
  [INSTALL.md](INSTALL.md). A clean-environment installation has not been
  validated; run the documented import and forward checks before use.
- CUDA backward operations can be nondeterministic. Bitwise-identical
  gradients across repeated runs are not guaranteed.
- The dependency files constrain selected versions; they are not a complete
  hash-locked environment covering drivers, toolchains and every dependency.
- Reproducing segmentation metrics additionally requires the matching trained
  weights, dataset splits, preprocessing and evaluation protocol. The model
  definition alone does not supply those resources.

## License

The project-level license is **Apache-2.0**; see [LICENSE](LICENSE).
It applies to original FSRH-Net contributions and modifications that the
contributors have authority to license. Third-party components retain their
applicable licenses and copyright notices; see [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.

Adding this project license does not relicense third-party code or resolve the
MF-Mamba-specific redistribution terms that remain to be clarified in the
third-party notice. Do not interpret it as a blanket grant over those portions.
