# Witwin Channel

Witwin Channel is a GPU-accelerated, differentiable wireless propagation
simulator for RF digital twins, coverage prediction, channel characterization,
and inverse optimization. It models a declarative scene and exposes dedicated
solvers for deterministic fields, explicit propagation paths, and Monte Carlo
radiomaps through the `witwin.channel` Python package.

> **Project status**
> Channel requires an NVIDIA GPU and runs only through its compiled CUDA
> extension. It is pre-1.0: public APIs may change between minor releases, and
> each release's changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## Capabilities

- Line-of-sight, multi-bounce specular reflection, first-order UTD
  diffraction, transmission through layered thin-sheet materials, and
  rough-surface Kirchhoff scattering.
- Point receivers and structured receiver grids for link-level and radiomap
  workflows.
- Complex fields, path gain, delay, departure/arrival angles, interaction
  geometry, CIR/CFR conversion, polarization, and antenna-array support where
  advertised by the selected solver.
- Fixed-topology JVP and VJP derivatives for deterministic, path, and Monte
  Carlo Basic solves. Supported inputs include material parameters, carrier
  frequency, endpoint positions, and mesh vertices.
- CUDA-resident scene, geometry, field, scattering, and derivative execution
  backed by the native Channel/RayD runtime.

The versioned capability manifest is available at runtime:

```python
import witwin.channel as channel

print(channel.capabilities())
```

## Propagation consumer API

External simulation packages can use the solver-neutral, versioned consumer
boundary without importing a Channel solver or internal propagation contract:

```python
from witwin.channel.propagation.consumer import (
    CONTRACT_VERSION,
    EndpointBatch,
    PropagationRequest,
    evaluate,
)

assert CONTRACT_VERSION == 7
```

The consumer publishes exact compact `K` rows with native-produced stable pair
segmentation and scalar, Complex3, or complete source-basis-to-sink-basis Jones
transport when advertised by its typed capabilities. It reuses the owning
ADR-032 compact boundary; no adapter performs a second count observation or
Torch/Python compaction. Unsupported component, response, frequency-offset, or
AD combinations fail before a partial result is returned.

The consumer supports LoS, reflection, transmission, and diffraction.
Scattering is not advertised or accepted: its current enumerated rows are
incoherent power-domain output and do not satisfy the consumer's coherent
transport or canonical pair-major row contracts.

The consumer is intentionally propagation-only. It contains no Radar target,
RCS, waveform, IQ, ADC, detection, or processing policy. See
[`docs/dev/propagation-consumer-contract.md`](docs/dev/propagation-consumer-contract.md)
for the versioning, convention, zero-copy, and failure contracts.

## Solver entry points

| Package | Use case | Primary result |
| --- | --- | --- |
| `witwin.channel.radiomap` with `Coherent` | Repeatable enumerated fields and radiomaps | `CoherentRadioMap`: field, path gain, component maps, optional path table |
| `witwin.channel.radiomap` with `Stochastic` | Monte Carlo power and radiomaps | `RadioMap`: path gain and component power/maps |
| `witwin.channel.path` | Explicit channel paths for point-to-point links | Complex coefficients, delays, angles, interactions, CIR/CFR |
| `witwin.channel.experimental.bdpt` | Bidirectional Monte Carlo propagation (experimental tier) | Path gain, component power, optional BDPT samples |

`radiomap.solve(scene, config)` picks the estimator from the configuration type;
`path` and `experimental.bdpt` own their own `Config`, `Result` and `solve`. The exact
stable exports are recorded in
[`ci/public-api-snapshot.json`](ci/public-api-snapshot.json); `experimental` modules are
outside it. Upgrading from 0.5: [`docs/migration-0.6.md`](docs/migration-0.6.md).

Complete signatures, parameter tables, tensor layouts, methods, exceptions,
and usage patterns are collected in the
[`docs/api-reference`](docs/api-reference/index.md) API Reference.

## Installation

### Prerequisites

| Item | Requirement |
| --- | --- |
| OS | Linux x86_64 (glibc 2.28 or newer) or Windows x64 |
| GPU | NVIDIA, compute capability 7.0 (Volta) or newer. Wheels contain native code for sm_70, 75, 80, 86, 89, 90, 100 and 120 and PTX for newer GPUs. Jetson (sm_87, sm_101) needs a source build. |
| NVIDIA driver | R525 or newer (CUDA 12); R555 or newer to use OptiX; R570 or newer for RTX 50-series (Blackwell). Without a usable OptiX runtime Channel traces on its slower pure-CUDA backend. |
| Python | CPython 3.10, 3.11, 3.12 or 3.13 |
| PyTorch | the exact CUDA 12.8 build the wheel names (PyTorch 2.10.0 for the PyPI wheels) |

No CUDA toolkit, OptiX SDK or compiler is needed to install a wheel.

### pip (Linux and Windows)

Install the CUDA 12.8 build of PyTorch first: the PyTorch wheels on PyPI are CPU-only on
Windows, and Channel cannot use them.

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install witwin-channel
python -m witwin.channel doctor
```

`doctor` (also installed as `witwin-channel-doctor`) prints the driver, GPU, PyTorch, build,
OptiX and trace backend it found, a verdict, and how to fix each problem.

Builds for PyTorch 2.11 are attached to each GitHub release as wheels with a `+torch211` local
version; install one by URL after `pip install torch==2.11.0 --index-url
https://download.pytorch.org/whl/cu128`. Every Channel wheel works with exactly one PyTorch
release, because the extension links PyTorch's C++ API, which changes between releases.

### Docker

The [`Dockerfile`](Dockerfile) builds the wheel from a checkout and installs it in a CUDA 12.8
image. Run containers with `--gpus all`; the image sets
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, which the NVIDIA container toolkit needs
to expose the driver's OptiX runtime.

### From source

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the toolchain (CUDA 12.8, the locked RayD commit,
OptiX 8.1 headers) and the build commands for Linux and Windows.

### Troubleshooting

- **`PyTorch ... is a CPU-only build`**: reinstall PyTorch from the CUDA index as shown above.
- **`_channel does not match the active Torch runtime`**: the installed PyTorch is not the
  release this Channel wheel was built for; the message names both and the install command.
- **`Channel cannot run on CUDA device ...`**: the GPU or driver is not covered by the build (for
  example a pre-Volta GPU, a Jetson, or a driver older than CUDA 12). The message names the GPU,
  its SM, the build's targets and the fix.
- **OptiX unavailable**: common in containers started without the `graphics` driver capability
  and on drivers older than R555. Channel then uses its pure-CUDA trace backend for every solver;
  it is several times slower on tracing-heavy scenes. `doctor` shows the reason.
- **Choosing the trace backend**: `WITWIN_CHANNEL_TRACE_BACKEND=auto` (default: OptiX when usable),
  `optix` (fail if OptiX is unusable) or `cuda`. It is read when a scene is compiled.
- **First run is slow**: the first solve in a fresh environment compiles OptiX programs (about
  4 s for LoS or reflection and about 14 s with diffraction on an RTX A6000); the driver caches
  them on disk, so later processes start in about 1-2 s. GPUs newer than every cubin in the wheel
  JIT-compile the PTX once in the same way.

## Quick start

The following CUDA example evaluates a 10-metre free-space link with the
Coherent radiomap estimator:

```python
import torch

from witwin.core import AntennaState, Scene
from witwin.channel.radiomap import Coherent, solve

scene = Scene(
    structures=[],
    endpoints=[
        AntennaState(
            1,
            "tx",
            torch.tensor([0.0, 0.0, 1.5]),
            power_w=1.0,
        ),
        AntennaState(
            2,
            "rx",
            torch.tensor([10.0, 0.0, 1.5]),
        ),
    ],
)

result = solve(
    scene,
    Coherent(max_depth=0, components={"los"}),
    reference_frequency_hz=3.5e9,
)

print(result.path_gain)  # CUDA tensor with shape (1, 1)
print(result.field)      # complex64 coherent field
```

Logical scenes, structures, materials, stable IDs, endpoints, and snapshots are
owned by `witwin.core`. Channel compiles them with
`witwin.channel.scene.compile(scene, reference_frequency_hz=...)`; every
solver entry point performs that same explicit compilation boundary. Use Core
`Mesh`, `PhysicalMaterial`, `MaterialLayer`, and
`PhysicalMaterial.perfect_conductor()`. Use Core `ReceiverGrid` for radiomap
solves, or select `witwin.channel.path` when individual path coefficients,
delays, angles, and interactions are required.

## Differentiation contract

`path`, `radiomap.Coherent`, `radiomap.Stochastic` and `experimental.bdpt` accept
`ad_mode="none"`, `"jvp"`, or `"vjp"`. Derivatives are evaluated through the
fixed topology selected by the primal solve. Visibility changes, path
birth/death, and other discrete topology discontinuities are outside this
contract. Unsupported solver/component/gradient combinations fail explicitly;
BDPT differentiates material, frequency and transmitter-power parameters and
refuses endpoint-position and mesh-vertex gradients.

## Runtime diagnostics

Use the public diagnostics instead of inspecting private extension modules:

```python
import witwin.channel as channel

print(channel.runtime_diagnostics())
print(channel.build_info())
```

`runtime_diagnostics()` reports the package, Python, PyTorch, GPU and its support status, the
driver's CUDA version, the cubin and PTX targets of the build, the OptiX probe, the requested
trace backend and the backend each compiled scene uses. `build_info()` identifies the native
ABI, build fingerprint and locked RayD source of the running extension.

## Development

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers building the extension, the test tiers, re-locking
RayD and releasing. Contributor rules are in [`AGENTS.md`](AGENTS.md). Architecture and
domain-owner documentation lives under [`docs/dev`](docs/dev).

## Citation

If you use Witwin Channel in academic research, please cite:

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
non-commercial research use or for commercial and enterprise use. See
[`LICENSE`](LICENSE) and the [Witwin licensing page](https://witwin.ai/license)
for the applicable terms.
