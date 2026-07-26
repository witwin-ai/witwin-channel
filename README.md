# Witwin Channel

Witwin Channel is a GPU-accelerated, differentiable wireless propagation
simulator for RF digital twins, coverage prediction, channel characterization,
and inverse optimization. It models a declarative scene and exposes dedicated
solvers for deterministic fields, explicit propagation paths, and Monte Carlo
radiomaps through the `witwin.channel` Python package.

> **Project status**
> Version 0.4 is a breaking replacement for the earlier 0.3 API. The supported
> runtime requires an NVIDIA GPU and the packaged native extension; APIs may
> continue to evolve before 1.0.

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

assert CONTRACT_VERSION == 4
```

The consumer publishes exact compact `K` rows with native-produced stable pair
segmentation and scalar, Complex3, or complete source-basis-to-sink-basis Jones
transport when advertised by its typed capabilities. It reuses the owning
ADR-032 compact boundary; no adapter performs a second count observation or
Torch/Python compaction. Unsupported component, response, frequency-offset, or
AD combinations fail before a partial result is returned.

Contract version 1 supports LoS, reflection, transmission, and diffraction.
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
| `witwin.channel.deterministic` | Repeatable coherent fields and radiomaps | Field, path gain, component maps, optional path table |
| `witwin.channel.path` | Explicit channel paths for point-to-point links | Complex coefficients, delays, angles, interactions, CIR/CFR |
| `witwin.channel.montecarlo.basic` | Incoherent sampled power and radiomaps | Path gain and component power/maps |
| `witwin.channel.montecarlo.bdpt` | Bidirectional Monte Carlo propagation | Path gain, component power, optional BDPT samples |

Each solver owns its own `Config`, `Result`, and `solve(scene, config)` public
contract. The exact stable exports are recorded in
[`ci/public-api-snapshot.json`](ci/public-api-snapshot.json).

## Requirements

- CPython 3.11.
- PyTorch 2.10 with CUDA 12.8 runtime support.
- Windows x64 on an NVIDIA GPU with compute capability 12.0 for the currently
  runtime-verified row. Release wheels must contain the repository-wide native
  SASS set, including SM87, and compute_120 PTX; SM87 runtime validation remains
  separate evidence.
- An ABI-compatible `witwin-channel` wheel, or a source build against the
  repository-locked RayD integration.

Channel has no production CPU compute backend. Missing CUDA, an unsupported
GPU architecture, an incompatible extension, or a required native capability
raises an error before returning a partial result.

`_channel` uses the versioned LibTorch/Python extension ABI. The Stage-I wheel
is CPython 3.11 / Torch 2.10.0 only and is not advertised as a LibTorch Stable
ABI binary. Linux release wheels are compiled inside a real
`manylinux_2_28` environment.

## Installation

Install an approved wheel into an environment that already contains the
matching CUDA-enabled PyTorch build:

```powershell
python -m pip install .\witwin_channel-0.4.0-cp311-cp311-win_amd64.whl --no-deps
```

For a source build, select the intended RayD checkout explicitly and keep the
build in the same Python environment as PyTorch:

```powershell
conda activate witwin2
$env:CMAKE_ARGS = "-DRAYD_SOURCE_DIR=E:/Code/RayD"
python -m pip install . --no-build-isolation --no-deps
```

When `RAYD_SOURCE_DIR` is omitted, the build may use a unique locked
`rayd-torch` source bundle only when its package metadata reports a clean,
matching source tree. The published `rayd-torch` 0.7.0 bundle does not satisfy
that release guard, so Channel 0.4.0 source builds must set `RAYD_SOURCE_DIR`
to the locked clean checkout. Discovery never scans a Conda prefix or loads an
arbitrary global build.

Do not mix files from the 0.3 and 0.4 implementations in one environment.

## Quick start

The following CUDA example evaluates a 10-metre free-space link with the
deterministic solver:

```python
import torch

from witwin.core import AntennaState, Scene
from witwin.channel.deterministic import Config, solve

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
    Config(max_depth=0, components={"los"}),
    reference_frequency_hz=3.5e9,
)

print(result.path_gain)  # CUDA tensor with shape (1, 1)
print(result.field)      # complex64 coherent field
```

Logical scenes, structures, materials, stable IDs, endpoints, and snapshots are
owned by `witwin.core`. Channel compiles them with
`witwin.channel.scene.compile(scene, reference_frequency_hz=...)`; the four
solver entry points perform that same explicit compilation boundary. Use Core
`Mesh`, `PhysicalMaterial`, `MaterialLayer`, and
`PhysicalMaterial.perfect_conductor()`. Use Core `ReceiverGrid` for radiomap
solves, or select `witwin.channel.path` when individual path coefficients,
delays, angles, and interactions are required.

## Differentiation contract

`path`, `deterministic`, and `montecarlo.basic` accept
`ad_mode="none"`, `"jvp"`, or `"vjp"`. Derivatives are evaluated through the
fixed topology selected by the primal solve. Visibility changes, path
birth/death, and other discrete topology discontinuities are outside this
contract. Unsupported solver/component/gradient combinations fail explicitly;
BDPT currently supports primal evaluation only.

## Runtime diagnostics

Use the public diagnostics instead of inspecting private extension modules:

```python
import witwin.channel as channel

print(channel.runtime_diagnostics())
print(channel.build_info())
```

These records identify the package, CUDA architecture, native ABI, build
fingerprint, and locked RayD source used by the running extension.

## Development

Repository development and validation use the `witwin2` Conda environment:

```powershell
conda run -n witwin2 python ci/run_ci_tier.py quick
conda run -n witwin2 python ci/run_ci_tier.py cuda
```

Architecture and domain-owner documentation lives next to the implementation
under [`src/witwin/channel`](src/witwin/channel). Contributor rules and the
full native validation matrix are defined in [`AGENTS.md`](AGENTS.md).

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
