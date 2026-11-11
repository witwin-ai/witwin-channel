# Supported runtime matrix

The machine-readable policy is [`ci/support-matrix.toml`](../../ci/support-matrix.toml). The
release workflow derives its job matrix, CUDA flags, expected SASS/PTX and size limit from it
through `ci/wheel_matrix.py`, and `tests/test_supported_runtime_matrix.py` keeps
`pyproject.toml` consistent with it.

| Axis | Values |
| --- | --- |
| CPython | 3.10, 3.11, 3.12, 3.13 |
| Platforms | `manylinux_2_28_x86_64`, `win_amd64` |
| PyTorch | 2.10.0 (wheels on PyPI), 2.11.0 (wheels with a `+torch211` local version on the GitHub release); CUDA 12.8 builds |
| CUDA toolkit | 12.8.1 |
| Cubins | sm_70, 75, 80, 86, 89, 90, 100, 120 |
| PTX | compute_120 |
| Wheel size limit | 90 MiB |
| Runtime-verified | sm_120 on Windows (evidence in the `[[verified]]` table) |

Every architecture that is built but not verified is expected to work and has no recorded
runtime evidence yet; CONTRIBUTING.md "Verifying a GPU architecture" lists the steps that add
it.

`cuda_runtime` means the CUDA version reported by `torch.version.cuda`; it is not the compiler
version. Python 3.10 compatibility of the package is checked statically (mypy and Ruff target
3.10); the wheel smoke runs on each built Python.

Why the PyTorch pin is exact, and why only one PyTorch release goes to PyPI, is explained in
CONTRIBUTING.md "Supported platforms and the PyTorch pin".

## Reproducing the CI environment

Start from CPython 3.11, then resolve the absolute constraint path before enabling PEP 517 build
isolation:

```bash
constraint=$(realpath constraints/ci-py311-cu128.txt)
export PIP_CONSTRAINT=$constraint
python -m pip install --constraint "$constraint" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
python -m pip install --constraint "$constraint" build cmake ninja numpy pytest coverage ruff scikit-build-core
```

Constraints restrict versions but do not select a CUDA wheel source; PyTorch must come from the
CUDA 12.8 index. Release evidence records the resolved package set, `torch.version.cuda`, the
toolkit version, the native build fingerprint and the GPU SM.
