# Installation and native-operator checks

Run the commands below from the repository root, the directory containing
`fsrhnet/`. They install dependencies into a **new** environment; do not run
them over an existing training environment that you need to preserve.

## Platform and compatibility

This installation path targets Linux x86_64, an NVIDIA GPU, Python 3.8.20,
PyTorch 2.2.2+cu118 and the native Mamba 1.1.3 selective-scan extension.
This package has no built-in CPU forward backend. Native Windows, macOS and
other CUDA/Python combinations are outside the documented configuration.

A clean-environment installation has not been validated. Complete the import
and model-forward checks in section 5 before using the package.

## Dependency files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Model and Mamba runtime/import dependencies, including fixed native source URLs |
| `requirements-build.txt` | pip, setuptools, wheel, packaging and ninja needed before building extensions |
| `constraints-cu118.txt` | Selected transitive version constraints; constraints do not install packages by themselves |

No training-framework, dataset, comparison-model or visualization packages are
included. Dependencies of Mamba are still needed even though FSRH-Net does not
directly import all of them. For example, Mamba 1.1.3 imports the causal-conv1d
extension when its scan interface is imported.

The native package versions are `mamba-ssm==1.1.3` and
`causal-conv1d==1.1.3`. To select these exact un-suffixed releases without
depending on their availability on PyPI, `requirements.txt` uses official
source archives pinned to the commits behind the **v1.1.3** tags. Similar
`.post1` releases are different distributions and are not substituted:

- [Mamba: 35689e19d2a0445f46b814b56d40491664b5da0a](https://github.com/state-spaces/mamba/tree/35689e19d2a0445f46b814b56d40491664b5da0a).
- [causal-conv1d: 21a3f010d01e1c25d7a407a7769e8952b5232417](https://github.com/Dao-AILab/causal-conv1d/tree/21a3f010d01e1c25d7a407a7769e8952b5232417).

The selected constraints are not a complete hash-locked environment: they do
not pin the OS, NVIDIA driver, CUDA toolkit, compiled-wheel identity, or every
transitive dependency. Rebuilding an extension from the same source does not
guarantee identical binary output across different toolchains.
Unpinned transitive dependencies, including the `buildtools` package declared
by causal-conv1d, are resolved by pip.

## 1. Prepare the platform and a new Python environment

Use a Linux x86_64 machine with an NVIDIA driver capable of running CUDA 11.8.
For the source-archive route below, prepare the CUDA 11.8 toolkit (`nvcc`),
CUDA headers/libraries and a CUDA-compatible C++ compiler. PyTorch's cu118 wheel
supplies runtime libraries; it does not by itself install the compiler toolkit.

```bash
nvidia-smi
nvcc --version
c++ --version
conda create -n fsrhnet python=3.8.20 pip -y
conda activate fsrhnet
```

If `nvcc` is unavailable or points to a different toolkit, select the installed
CUDA 11.8 toolkit in your shell's PATH and CUDA_HOME before building. The
appropriate path is machine-specific, so none is hard-coded here. Do not set
`MAMBA_SKIP_CUDA_BUILD` or `CAUSAL_CONV1D_SKIP_CUDA_BUILD`: this model needs the
native extensions. See the fixed-version [Mamba requirements](https://github.com/state-spaces/mamba/blob/v1.1.3/README.md#installation).

## 2. Install build tools

```bash
python -m pip install -r requirements-build.txt
```

These packages support the Python build process; the CUDA toolkit and a
compatible C++ compiler must also be available.

## 3. Install the CUDA build of PyTorch first

```bash
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu118 -c constraints-cu118.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.__version__.startswith('2.2.2'); assert torch.version.cuda == '11.8'; assert torch.cuda.is_available()"
```

This uses the [official PyTorch 2.2.2 CUDA 11.8 wheel index](https://pytorch.org/get-started/previous-versions/#v222).
The standalone model does not require torchvision or torchaudio. A plain
`torch==2.2.2` requirement without the selected index does not specify which
CUDA/CPU build is intended.

## 4. Install the fixed native sources and runtime dependencies

```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

Do not start with this command in an environment without PyTorch and the
build tools. The [Mamba v1.1.3 build script](https://github.com/state-spaces/mamba/blob/v1.1.3/setup.py)
imports PyTorch while preparing/building the package. `--no-build-isolation`
allows that build to use the dependencies already installed above; it does not
install missing build tools or fix a missing CUDA compiler. See [pip's build
isolation documentation](https://pip.pypa.io/en/stable/reference/build-system/#disabling-build-isolation).

The upstream build may use a matching release wheel or compile locally.
If obtaining a wheel manually, match its Python ABI, Linux platform, PyTorch,
CUDA and C++ ABI tags to this environment; do not use an arbitrary wheel with
the same package name. Official release pages:
[Mamba v1.1.3](https://github.com/state-spaces/mamba/releases/tag/v1.1.3),
[causal-conv1d v1.1.3](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.1.3).

Stop and resolve conflicts reported by `pip check` in the new environment.
Do not hide conflicts with `--no-deps`, disable TLS verification, or silently
upgrade to a newer Mamba/causal-conv1d API. If PyTorch/CUDA is changed later,
rebuild or reinstall both native extensions for the new combination.

## 5. Check imports, then execute the actual model

First verify the native extensions and installed distribution versions:

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch
import selective_scan_cuda
import causal_conv1d_cuda
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

assert torch.cuda.is_available()
assert torch.version.cuda == "11.8"
assert metadata.version("mamba-ssm") == "1.1.3"
assert metadata.version("causal-conv1d") == "1.1.3"
print("Native imports OK:", torch.__version__)
PY
```

Then execute the model from this repository root:

```bash
python - <<'PY'
import torch
from fsrhnet import FSRHNet

model = FSRHNet(num_classes=6).cuda().eval()
image = torch.randn(1, 3, 64, 96, device="cuda", dtype=torch.float32)
with torch.no_grad():
    logits = model(image)
assert logits.shape == (1, 6, 64, 96)
assert torch.isfinite(logits).all()
print("Model forward OK:", tuple(logits.shape))
PY
```

This is an installation smoke check with random weights and synthetic input,
not an accuracy test. For real images, use the channel order and normalization
in [README.md](README.md#dependencies-and-use), and load the matching trained
model weights with `strict=True`. A model-only source directory is provided;
`pip install .` is not an installation interface for FSRH-Net itself.
