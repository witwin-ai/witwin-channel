# Witwin Channel API Reference

This reference documents the complete supported Python interface of
`witwin-channel` 0.4.x.

The stable API baseline is [`ci/public-api-snapshot.json`](../../ci/public-api-snapshot.json):
six modules and 60 frozen exports. The scene lifecycle page additionally
documents the three supporting exports required to compile a `witwin.core`
world for direct consumer use. Implementation modules such as `runtime`,
`kernels`, `interactions`, and the remaining `propagation` modules are
intentionally excluded: they are internal contracts, not supported user APIs.

> `witwin.channel` has one production compute backend: the compiled native
> CUDA/RayD extension. There is no CPU, NumPy, Torch-numerical, or reduced
> fallback. Missing CUDA, an incompatible ABI, a missing native symbol, or an
> unsupported capability raises before a partial result is returned.

## Module index

| Module | Purpose | Reference |
| --- | --- | --- |
| `witwin.channel` | ABI state types, capabilities, and deployment reporting | [Package root](root.md) |
| `witwin.channel.scene` | Compile and cache lifecycle for direct consumer use | [Scene lifecycle](scene.md) |
| `witwin.channel.path` | Explicit paths, CIR/CFR, taps, and beamforming | [Path solver](path.md) |
| `witwin.channel.deterministic` | Deterministic coherent fields and radiomaps | [Deterministic solver](deterministic.md) |
| `witwin.channel.montecarlo.basic` | Basic Monte Carlo power estimation | [Monte Carlo Basic](montecarlo-basic.md) |
| `witwin.channel.montecarlo.bdpt` | Bidirectional path tracing with MIS | [Monte Carlo BDPT](montecarlo-bdpt.md) |
| `witwin.channel.propagation.consumer` | Solver-neutral compact propagation contract | [Propagation consumer](propagation-consumer.md) |

## Shared conventions

- Logical worlds are owned by `witwin.core`. Import `Scene`, `SceneSnapshot`,
  `Structure`, `AntennaState`, materials, and receiver grids from
  `witwin.core`.
- All four solvers use
  `solve(scene, config, *, reference_frequency_hz=...)`.
- Frequency is in hertz, position and distance are in metres, delay is in
  seconds, angles are in radians, and power is in watts.
- The phase convention is `exp(-j*k*d)` with time dependence
  `exp(+j*2*pi*f*t)`.
- Continuous real outputs are normally `float32`, complex fields are
  `complex64`, indices are `int32` or stable-ID `int64`, and validity masks are
  `bool`. Each type reference gives the exact contract.
- Supported differentiable entry points accept `ad_mode="none"`, `"jvp"`, or
  `"vjp"`. Differentiation is through the selected discrete topology; path
  birth/death and visibility discontinuities are outside that contract.

Dimension symbols used throughout this reference:

| Symbol | Meaning |
| --- | --- |
| `R`, `T` | Receiver and transmitter counts |
| `Ra`, `Ta` | Receive and transmit antenna counts |
| `P` | Actual maximum padded paths per link |
| `K` | Actual compact path-row count |
| `D` | Maximum interaction depth |
| `F` | Frequency-offset column count |
| `S` | Slot or time-label count |

## Minimal solver example

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

print(result.path_gain.shape)  # torch.Size([1, 1])
print(result.field.dtype)      # torch.complex64
print(result.field.device)     # cuda:0 (device index is environment-specific)
```

## Stability boundary

The frozen stable surface consists of the package root, the four solver entry
modules, and `witwin.channel.propagation.consumer`. The scene lifecycle is a
small supporting surface used to obtain `CompiledScene` for the consumer.

An importable name is not automatically public. In particular, `__all__`
inside an implementation-owner module controls internal wildcard imports; it
does not override the repository's frozen public API policy. No predecessor
product name or compatibility alias is part of this documentation.
