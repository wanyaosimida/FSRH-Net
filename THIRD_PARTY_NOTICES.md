# Component provenance and upstream terms

The project's original contributions and modifications
that its contributors have authority to license use [Apache-2.0](LICENSE).
This file records upstream statements and exceptions; it does not grant
additional rights or apply that project license to third-party code beyond
the permissions already held. See also [NOTICE](NOTICE).

## MF-Mamba

Several supporting components derive in part from
[Mango-Mars/MF-Mamba](https://github.com/Mango-Mars/MF-Mamba), using reference
revision [3326aa9b6d05bb7bd4d9b64847f6a061198f018c](https://github.com/Mango-Mars/MF-Mamba/tree/3326aa9b6d05bb7bd4d9b64847f6a061198f018c).
The inherited convolutional utilities and segmentation head in `layers.py`
and `head.py`, and the eight-direction scan conventions in `scan.py`, are
support components rather than additional proposed methods.

In [an upstream collaborator's comment dated 2025-09-25](https://github.com/Mango-Mars/MF-Mamba/issues/1#issuecomment-3332443242),
the collaborator explains that the core code is provided for integration and
extension in users' own frameworks, including training and testing workflows.
This is relevant evidence of the intended integration/extension use.

The referenced repository and README do not specify a project-wide license.
The comment does not specify public redistribution or relicensing terms.
Accordingly, this package does not claim
that the MF-Mamba-specific inherited code is MIT-, BSD-, or Apache-licensed,
or that the upstream comment grants additional rights beyond its stated scope.
Public redistribution terms for those inherited portions remain to be clarified.

## HRNet and the unused bundled sync_bn component

`fsrhnet/backbone.py` retains the original Microsoft copyright and MIT notice
from the [upstream HRNet source](https://github.com/Mango-Mars/MF-Mamba/blob/3326aa9b6d05bb7bd4d9b64847f6a061198f018c/backbone/HRNet/models/seg_hrnet.py).
The MIT text is included in `LICENSES/HRNet-MIT.txt`.

MF-Mamba also bundles an independent `backbone/HRNet/models/sync_bn` component
under [BSD-3-Clause](https://github.com/Mango-Mars/MF-Mamba/blob/3326aa9b6d05bb7bd4d9b64847f6a061198f018c/backbone/HRNet/models/sync_bn/LICENSE).
This refactored package does **not** include that component; it uses PyTorch's
`nn.SyncBatchNorm`. That component's BSD license is not a license for the
whole MF-Mamba or FSRH-Net repository.

## Mamba and VMamba lineage

`fsrhnet/scan.py` includes Mamba/VMamba-derived state-space
initialization and scanning conventions. This source-level lineage is distinct
from the external native operator dependency. Relevant upstream terms include
[Mamba v1.1.3 Apache-2.0](https://github.com/state-spaces/mamba/blob/v1.1.3/LICENSE)
and [VMamba MIT](https://github.com/MzeroMiko/VMamba/blob/main/LICENSE).
Copies are included as [LICENSES/Mamba-Apache-2.0.txt](LICENSES/Mamba-Apache-2.0.txt)
and [LICENSES/VMamba-MIT.txt](LICENSES/VMamba-MIT.txt), with attribution in
[NOTICE](NOTICE). These copies retain available upstream terms; they do not
establish rights to MF-Mamba-specific additions or replace the need to identify
the exact inherited versions and scope when confirming redistribution.

The native selective-scan operator is imported from the separately installed
[Mamba package](https://github.com/state-spaces/mamba). Its binary extension is
not bundled here. Its external dependency license does not resolve the
MF-Mamba-specific inherited source terms described above.

## Project-level status

The project-level license is **Apache-2.0**.
The complete, unmodified license text is in [LICENSE](LICENSE).
Its grant covers the project's original contributions and modifications that
the contributors have authority to license, including the corresponding
copyright and patent grants described in that license.

Existing third-party notices and component licenses are preserved. In
particular, the MF-Mamba-specific source-rights clarification above remains
outstanding; selecting Apache-2.0 does not represent it as already resolved.
This provenance document is not an independent license grant.
