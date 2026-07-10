# Witwin Channel

> **Warning**
> This is an experimental early release. APIs may change frequently, bugs are
> expected, and the project will be updated aggressively.

`witwin-channel` is a differentiable, geometric differentiable, GPU-first
wireless channel simulation package for the Witwin platform. It models radio
propagation through declarative scenes and solver-specific entrypoints for
deterministic radiomaps, Monte Carlo radiomaps, and receiver-level path export.

The stable public contract is:

```text
Scene + solver.solve(scene, config) + Result
```

Solvers live under `witwin.channel.deterministic`, `witwin.channel.montecarlo`,
and `witwin.channel.path`. There is no shared public tracer object.

## Capabilities

- Differentiability-first channel simulation for inverse rendering,
  optimization, and gradient-based RF digital twin workflows.
- Geometric differentiability through scene geometry, material parameters,
  endpoints, and solver outputs.
- Declarative scene construction with `witwin.core.Structure`, `Material`, and
  geometry objects.
- Scene-owned `Transmitter`, `Receiver`, and `ReceiverGrid` endpoints selected
  by name at solve time.
- Line-of-sight, multi-bounce reflection, and multi-order UTD-style diffraction
  support.
- Deterministic radiomap solving for structured grid outputs.
- Monte Carlo radiomap solving with basic and BDPT integrators.
- Path-level solving for CIR/CFR, delay, angle, interaction, and optional
  geometry payloads.
- Dr.Jit-native runtime internals with CUDA/native extension acceleration paths.
- Pytest validation, GPU acceptance tests, benchmark scripts, and Sionna RT
  comparison workflows.

## Requirements

Core solver workflows expect:

- Windows or Linux with an NVIDIA GPU.
- CUDA-capable Python environment.
- CPython 3.10-3.14.
- Dr.Jit `1.3.1`.
- PyTorch 2.10 or 2.11 with CUDA 12.8 support.
- RayD and native extension build dependencies where acceleration paths are
  required.

Release wheels are built for Linux x86_64 and Windows x86_64 with CUDA 12.8. Their fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX. This includes RTX 2080-class Turing and current data-center and RTX/RTX PRO Blackwell coverage. Linux wheels target `manylinux_2_35_x86_64`; the installed NVIDIA driver must support CUDA 12.x.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

Repository development uses the `witwin2` conda environment.

## Installation

Create and activate a local conda environment:

```powershell
conda create -n witwin2 python=3.11 -y
conda activate witwin2
```

Install the CUDA-enabled PyTorch build that matches your driver and platform,
then install this package. For example:

```powershell
python -m pip install "torch>=2.10,<2.12" --index-url https://download.pytorch.org/whl/cu128
```

From the `channel` subproject root:

```powershell
conda activate witwin2
python -m pip install -e . --no-build-isolation --no-deps
```

For non-editable local installs, keep dependency resolution disabled:

```powershell
conda activate witwin
python -m pip install . --no-deps
```

Native C++/CUDA iteration should use the existing CMake build directory when
possible instead of repeated editable reinstalls:

```powershell
cmake --build build\cp311-cp311-win_amd64 --config Release --target <native_target>
cmake --install build\cp311-cp311-win_amd64 --config Release --prefix <witwin>\Lib\site-packages
```

On Windows, close Python and Jupyter processes that have loaded the extension
before installing `.pyd` files, because loaded extension modules are locked.

## Quick Start

```python
import numpy as np
import witwin.channel as wc

scene = wc.Scene(
    structures=[
        wc.Structure(
            name="wall",
            geometry=wc.Box(
                position=(0.0, 0.0, 1.5),
                size=(0.25, 4.0, 3.0),
                device="cuda",
            ),
            material=wc.Material(eps_r=4.0, sigma_e=0.0),
        ),
    ],
    transmitters=[
        wc.Transmitter("tx", (-2.0, 0.0, 1.5)),
    ],
    receivers=[
        wc.ReceiverGrid(
            "rm",
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=(32, 32),
        ),
    ],
    frequency=3.5e9,
    device="cuda",
)

result = wc.deterministic.solve(
    scene=scene,
    transmitter="tx",
    receiver="rm",
    config=wc.deterministic.Config(
        num_samples=256,
        max_bounces=1,
        max_diffraction_order=0,
        edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
    ),
)

path_gain = np.asarray(result.path_gain, dtype=np.float32)
print(path_gain.shape, float(path_gain.max()))
```

## Solver Entry Points

### Deterministic Radiomap

Use `wc.deterministic.solve(...)` when you need a repeatable radiomap over a
scene-owned `ReceiverGrid`.

```python
config = wc.deterministic.Config(
    num_samples=512,
    max_bounces=2,
    max_diffraction_order=1,
    edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
    tuning=wc.deterministic.Tuning(enable_rd_diffraction=True),
)

result = wc.deterministic.solve(
    scene=scene,
    transmitter="tx",
    receiver="rm",
    config=config,
)
```

The result is a shared `wc.RadioMapResult` with `path_gain`, component maps,
coordinates, transmitter-axis helpers, and metadata.

### Monte Carlo Radiomap

Use `wc.montecarlo.solve(...)` for transmitter-driven radiomap sampling and
Monte Carlo integrators.

```python
config = wc.montecarlo.Config(
    num_samples=128,
    max_bounces=1,
    max_diffraction_order=0,
    integrator_options=wc.montecarlo.IntegratorOptions(
        integrator="basic",
        samples_per_tx=4096,
        accumulation_backend="auto",
        seed=7,
    ),
)

result = wc.montecarlo.solve(
    scene=scene,
    transmitter="tx",
    receiver="rm",
    config=config,
)
```

The Monte Carlo package also exposes `IntegratorOptions(integrator="bdpt")` for
BDPT workflows.

### Path Solver

Use `wc.path.solve(...)` when you need discrete paths for named receivers.
The scene must contain the selected `Receiver` endpoints.

```python
path_result = wc.path.solve(
    scene=scene,
    transmitter="tx",
    receiver=["rx0", "rx1"],
    config=wc.path.Config(
        num_samples=256,
        max_bounces=1,
        max_diffraction_order=0,
        max_num_paths=8,
        return_geometry=True,
        edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
    ),
)

coeff, delay = path_result.cir()
response = path_result.cfr(subcarriers_hz)
```

`PathResult` includes path coefficients, delays, AoD/AoA metadata,
interaction types, optional geometry, and torch-backed CIR/CFR helpers.

## Public API Overview

The recommended user-facing import is:

```python
import witwin.channel as wc
```

The umbrella namespace exports:

- Scene and endpoint types: `Scene`, `Transmitter`, `Receiver`,
  `ReceiverGrid`.
- Core geometry and materials: `Box`, `Material`, `Structure`.
- Antenna arrays: `AntennaArray`, `PlanarArray`, `ULA`, `UPA`.
- Solver namespaces: `deterministic`, `montecarlo`, `path`.
- Shared result and Dr.Jit aliases such as `RadioMapResult`, `Point3f`, and
  `Vector3f`.

Scene frequency, materials, endpoint positions, endpoint power, polarization,
and arrays are owned by `Scene` and its endpoint/material objects. Solver
configs should describe algorithm controls such as sample budgets, bounce
limits, diffraction order, edge policy, integrator settings, and advanced
tuning.

## Tutorials

Introductory notebook workflows live under `tutorials/`.

## Testing

Run tests from this directory with the `witwin` environment active:

```powershell
conda activate witwin
python -m pytest tests
```

Common targeted runs:

```powershell
python -m pytest tests/scene
python -m pytest tests/deterministic
python -m pytest tests/montecarlo
python -m pytest tests/path
python -m pytest tests --gpu
python -m pytest tests --gpu --acceptance
```

Manual validation and benchmark entrypoints live under `tests/support/bin/`.
Useful maintained commands include:

```powershell
python -m tests.support.bin.benchmark_monte_carlo_radiomap_package --mode forward --integrator basic --json
python -m tests.support.bin.benchmark_mc_basic_munich_vs_sionna --json
python -m tests.support.bin.validate_path_solver_munich --json
```

## Repository Layout

| Path | Role |
| --- | --- |
| `witwin/channel/core/` | Shared geometry, numerics, physics, runtime, scene, kernels, and result helpers |
| `witwin/channel/core/scene/` | Declarative `Scene`, mesh adaptors, endpoints, arrays, and edge policy |
| `witwin/channel/deterministic/` | Deterministic radiomap solver package |
| `witwin/channel/montecarlo/` | Monte Carlo radiomap solver package |
| `witwin/channel/path/` | Path-level solver package |
| `witwin/channel/_native/` | Native extension loaders and package-level native bindings |
| `tests/` | Regression, GPU, acceptance, and support-script coverage |
| `tutorials/` | Introductory notebook workflows |

The package layering is:

```text
channel/core/ -> channel/core/scene/ -> {channel/deterministic, channel/montecarlo, channel/path}/
```

Solver packages must not import from each other. Shared solver-neutral behavior
belongs in `witwin/channel/core/`.

## Citation

If you use this work in academic research, please cite:

```bibtex
@inproceedings{chen2026rfdt,
  title     = {Physically Accurate Differentiable Inverse Rendering
               for Radio Frequency Digital Twin},
  author    = {Chen, Xingyu and Zhang, Xinyu and Zheng, Kai and
               Fang, Xinmin and Li, Tzu-Mao and Lu, Chris Xiaoxuan
               and Li, Zhengxiong},
  booktitle = {Proceedings of the 32nd Annual International Conference
               on Mobile Computing and Networking (MobiCom)},
  year      = {2026},
  doi       = {10.1145/3795866.3796686},
  publisher = {ACM},
  address   = {Austin, TX, USA},
}
```

## License

Witwin Channel is available under a dual-license model for academic and
non-commercial research use or commercial and enterprise use. See the
[Witwin licensing page](https://witwin.ai/license) for the applicable terms.
