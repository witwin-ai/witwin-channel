# Contributing to Witwin Channel

The binding rules for code changes are in [`AGENTS.md`](AGENTS.md) (identical to `CLAUDE.md`):
compute policy, launch and sync budgets, ownership, native boundary, style and validation. This
page covers the mechanics: building the extension, running the tests and cutting a release.

## What a build needs

| Item | Version | Notes |
| --- | --- | --- |
| NVIDIA GPU | compute capability 7.0 (Volta) or newer | for running tests; building needs no GPU |
| CUDA toolkit | 12.8 (nvcc, cudart, cuSOLVER headers) | must match PyTorch's CUDA 12.8 runtime |
| Host compiler | GCC 11-14 on Linux, MSVC 2022 on Windows | |
| Python | 3.10-3.13 | |
| PyTorch | the exact release in `pyproject.toml` (`torch==2.10.0`), CUDA 12.8 build | `pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128` |
| CMake, Ninja, scikit-build-core | `cmake>=4.3,<4.4`, `ninja>=1.13,<1.14`, `scikit-build-core>=0.12,<0.13` | builds run with `--no-build-isolation`, so install them yourself |
| RayD | a Git checkout at the commit in `dependencies/rayd.lock.json` | its `origin` must be the locked repository URL |
| OptiX SDK headers | 8.1 or newer | `git clone --branch v8.1.0 https://github.com/NVIDIA/optix-dev`; headers only, no library |
| A Git checkout of Channel | | the build records its commit and dirty state in the build fingerprint |

Development tools (pytest, coverage, Ruff, mypy, SciPy) are pinned in the `dev` extra of
`pyproject.toml` and in `constraints/ci-py311-cu128.txt`; install them at those versions.

## Linux: build and test a checkout

```bash
scripts/dev_build.sh --rayd /path/to/RayD    # CMake + Ninja into build/dev-86; read its header
source scripts/dev_env.sh                     # loads that build through the developer override
python -m pytest -q tests/deterministic       # always from the repository root
python ci/run_ci_tier.py quick                # see "Test tiers" below
```

`CHANNEL_DEV_CUDA_ARCH=89 scripts/dev_build.sh ...` builds for another GPU (one SM per build
directory). If the RayD checkout has moved past the locked commit, the script checks the locked
commit out into `build/rayd-<sha>` and builds against that. A clean build takes 15-35 minutes
(one translation unit, `coupled.cu`, dominates); an incremental one takes seconds to minutes.
Re-source `dev_env.sh` after every build: the fingerprint follows Git HEAD and dirty state.

The test suite never picks a build directory by itself. When it finds no validated extension it
stops and prints the three `WITWIN_CHANNEL_*` variables to set and the local builds it found.

## Building a wheel

From a Channel clone, with the toolchain above in the active Python environment:

```bash
export CMAKE_ARGS="-DRAYD_SOURCE_DIR=/path/to/RayD -DOPTIX_INCLUDE_DIR=/path/to/optix-dev/include -DCMAKE_CUDA_ARCHITECTURES=86-real"
pip wheel . --no-build-isolation --no-deps -w dist
```

On Windows (Developer PowerShell for VS 2022, CUDA 12.8 installed):

```powershell
$env:CMAKE_ARGS = "-DRAYD_SOURCE_DIR=C:/src/RayD -DOPTIX_INCLUDE_DIR=C:/src/optix-dev/include -DCMAKE_CUDA_ARCHITECTURES=120-real"
pip wheel . --no-build-isolation --no-deps -w dist
```

`CMAKE_CUDA_ARCHITECTURES` takes `<sm>-real` for a cubin and `<sm>-virtual` for PTX; the release
set is in `ci/support-matrix.toml`. Without it the build compiles five architectures.

When `RAYD_SOURCE_DIR` is not given, the build looks for the locked `rayd-torch` source bundle
installed in the active environment and validates its manifest; it never scans for other copies.

A container build that does all of this is in the [`Dockerfile`](Dockerfile).

## Test tiers

```bash
python ci/run_ci_tier.py quick     # static and generated checks, CPU-only tests; no extension
python ci/run_ci_tier.py cuda      # + full unit/contract suite, solver smoke, no-fallback, AD core
python ci/run_ci_tier.py nightly   # + coverage, Munich parity, full AD, statistics, wheel smoke
python ci/run_ci_tier.py release   # + performance, memory, scaling and wheel release gates
```

The quick tier calls `python -m ruff` and `python -m mypy`; put the pinned tools on `PATH` or set
`CHANNEL_DEV_TOOLS_BIN` before sourcing `dev_env.sh`. `AGENTS.md` section 8 says which tier a
change needs. After changing a pybind binding run `python ci/check_native_bindings.py --write`;
after changing native sources run `python ci/check_native_inventory.py --write`.

To run the solver tests on the pure-CUDA trace backend, set `WITWIN_CHANNEL_TRACE_BACKEND=cuda`
(or `RAYD_DISABLE_OPTIX=1`, which makes OptiX unusable process-wide).

## Changing the RayD dependency

Commit in the RayD checkout, then `python tools/relock_rayd.py --rayd /path/to/RayD` (`--check`
previews). It rewrites `dependencies/rayd.lock.json` and every file that pins the commit. Never
edit the digests by hand.

## Supported platforms and the PyTorch pin

The release matrix lives in [`ci/support-matrix.toml`](ci/support-matrix.toml) and the release
workflow derives everything from it through `ci/wheel_matrix.py`: CPython 3.10-3.13, Linux
(`manylinux_2_28_x86_64`) and Windows (`win_amd64`), cubins for sm_70, 75, 80, 86, 89, 90, 100
and 120 plus compute_120 PTX, and a 90 MiB wheel size limit.

**Why every wheel pins one exact PyTorch release.** `_channel` links PyTorch's C++ API (ATen,
c10, pybind11 types), which has no stable ABI across PyTorch minor releases, so a wheel built
against 2.10 cannot load under 2.11. The loader checks the PyTorch version, CUDA version and C++
ABI recorded in the build and refuses a mismatch with the install command. Relaxing the pin in
the metadata would only let pip produce environments that fail at load time. A wheel filename
cannot carry the PyTorch release, so PyPI carries wheels for one release (the `pypi` row) and the
other row's wheels carry a local version (`0.6.0+torch211`) and are attached to the GitHub
release. Making one wheel work across PyTorch releases needs a C ABI core with a thin per-version
shim, which is a separate design decision.

**sm_87 and sm_101** (Jetson Orin and Thor) run only their own cubins and exist only on aarch64,
so x86_64 wheels do not carry them; build from source on those boards.

## Verifying a GPU architecture

`ci/support-matrix.toml` `[[verified]]` lists the architectures with recorded runtime evidence.
Every other architecture in `sass` is built and expected to work but is not verified. To add one:

1. Install the release wheel (or a source build for that SM) on the GPU and record
   `python -m witwin.channel doctor --json`.
2. Run `python ci/run_ci_tier.py cuda` and the AD tier tests (`tests/ad`) on that GPU, once with
   the default trace backend and once with `WITWIN_CHANNEL_TRACE_BACKEND=cuda`.
3. Commit the doctor output and the test summaries under `docs/dev/baselines/<commit>/` and add a
   `[[verified]]` entry pointing at them.

## Releasing

1. The integration branch sets the version in `pyproject.toml`.
2. Publish a GitHub release tagged `witwin-channel-v<version>`. The workflow
   `.github/workflows/publish-witwin-channel.yml` builds every wheel in the matrix, checks
   architectures, metadata, contents and size, publishes the `pypi` wheels through trusted
   publishing and attaches the others to the release.
3. A manual `workflow_dispatch` run with scope `windows-smoke` builds one Windows wheel for
   sm_86/sm_120 only; scope `full` builds the whole matrix without publishing.
