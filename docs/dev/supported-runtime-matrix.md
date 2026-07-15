# Supported runtime matrix

Phase 2 intentionally declares only combinations backed by the accepted local
baseline. A package version range is not a claim that every Python, Torch, or
CUDA combination inside a broader ecosystem is supported.

| Python | Torch | Torch CUDA runtime | Native CUDA toolkit | Verified GPU | Status |
| --- | --- | --- | --- | --- | --- |
| CPython 3.11 | 2.10 | 12.8 | 12.9 | SM 120 | supported |

The package metadata expresses the same compatibility boundary as bounded
ranges: Python `>=3.11,<3.12`, Torch `==2.10.0`, scikit-build-core
`>=0.12,<0.13`, CMake `>=4.3,<4.4`, and Ninja `>=1.13,<1.14`. CI pins the
accepted concrete versions in `constraints/ci-py311-cu128.txt`.

`cuda_runtime` means the CUDA ABI reported by `torch.version.cuda`; it is not
the host compiler version. The accepted baseline used a Torch CUDA 12.8 runtime
and CUDA toolkit/NVCC 12.9. Builds declare SASS for SM 75, 80, 86, 89, and 120,
plus PTX for SM 120. Only SM 120 has runtime acceptance evidence. The other
architectures remain `declared_unverified` until the complete solver and AD
gates run on matching hardware.

The machine-readable policy is `ci/support-matrix.toml`. Its evidence path
points to the committed Phase 0 environment manifest. Changes to a supported
row require a fresh baseline, full unit/contract/acceptance gates, the four
solver runtime matrix, AD gates, and wheel smoke on the proposed combination.

## Reproducing the CI environment

Start from CPython 3.11, then resolve the absolute constraint path before
enabling PEP 517 build isolation:

```powershell
$constraint = (Resolve-Path constraints/ci-py311-cu128.txt).Path
$env:PIP_CONSTRAINT = $constraint
python -m pip install --constraint $constraint pip==25.3 setuptools==80.9.0 wheel==0.45.1
python -m pip install --constraint $constraint --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
python -m pip install --constraint $constraint build cmake ninja numpy pytest coverage ruff scikit-build-core
python -m pip install --no-build-isolation --editable .
```

Constraints restrict versions but do not select a CUDA wheel source. The CI job
must explicitly use the official CUDA 12.8 Torch wheel index documented in
[PyTorch 2.10 installation instructions](https://pytorch.org/get-started/previous-versions/#v2100).
Release evidence must record the resolved package set, `torch.version.cuda`,
toolkit/compiler version, native build fingerprint, and GPU SM.
