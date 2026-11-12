# Witwin Channel API Reference

This reference documents the complete supported Python interface of
`witwin-channel` 0.6.x.

The stable API baseline is [`ci/public-api-snapshot.json`](../../ci/public-api-snapshot.json):
six modules and 79 frozen exports. The scene lifecycle page additionally
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
| `witwin.channel.radiomap` | Radiomaps: `solve` dispatches on `Coherent` or `Stochastic` | [Radiomap](radiomap.md) ([Coherent](radiomap-coherent.md), [Stochastic](radiomap-stochastic.md)) |
| `witwin.channel.experimental.bdpt` | Bidirectional path tracing with MIS (experimental tier, not in the snapshot) | [Experimental BDPT](experimental-bdpt.md) |
| `witwin.channel.propagation.consumer` | Solver-neutral compact propagation contract | [Propagation consumer](propagation-consumer.md) |
| `witwin.channel.importers` | Mitsuba/Sionna XML, PLY, OBJ and glTF import into Core structures | [Importers](importers.md) |
| `witwin.channel.presets` | ITU-R P.2040 materials, TR 38.901 and dipole patterns | [Presets](presets.md) |

User guides: [Quickstart](../user/quickstart.md), [Scene import](../user/scene-import.md),
[Materials and antennas](../user/materials-and-antennas.md),
[Troubleshooting](../user/troubleshooting.md). Runnable scripts are in
[`examples/`](../../examples).

## Shared conventions

- Logical worlds are owned by `witwin.core`. Import `Scene`, `SceneSnapshot`,
  `Structure`, `AntennaState`, materials, and receiver grids from
  `witwin.core`.
- Every solver (`path.solve`, `radiomap.solve`, `experimental.bdpt.solve`) uses
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

### Receiver layout and grid axis order

Every solver uses the same receiver layout for `path_gain`, Coherent
`field`/`component_power`/`component_fields`, Stochastic `component_maps` and
BDPT `variance`:

- **Exactly one `ReceiverGrid`** of `shape=(rows, cols)`: `[T, rows, cols]`,
  following Core's row-major grid convention. Entry `[t, i, j]` is grid sample
  `(row i, column j)` at `origin + i * spacing[0] * x_axis + j * spacing[1] *
  y_axis`, so the second axis runs along `x_axis` and the third along `y_axis`.
  To show it as an image with x to the right and y up, use
  `imshow(path_gain[t].T, origin="lower")`.
- **Any other layout** (points only, several grids, grids and points mixed):
  `[T, R]`, with columns following `scene.receivers`; a grid contributes its
  samples in row-major order (`i * cols + j`).

So for a single grid, `path_gain[t].reshape(-1)[k]` is the same receiver `k` in
every solver and equals the point-receiver result at that sample. All grid
outputs are contiguous: every solver accumulates or deposits in receiver
row-major order natively.

### Result semantics

Result metadata of every solver carries `estimator` (`coherent` for enumerated
paths, `stochastic` for sampled ones), `quantity` (a dict from each requested
component to `point_sample`, the value at the receiver position, or
`cell_average`, the average over the cell or capture region), `combine`
(`field`: complex sum then squared, `power`: sum of row powers, or `none` for
uncombined rows) and `complete` (whether every path within the requested bounds
was found; for `path` and `radiomap.Coherent` it is
`metadata["discovery"]["complete"]`, and the stochastic solvers report `False`).
The metadata of each solver is typed as a `TypedDict` (`path.PathMetadata`,
`radiomap.CoherentRadioMapMetadata`, `radiomap.RadioMapMetadata`,
`experimental.bdpt.Metadata`) naming its stable keys; at run time it is a plain `dict`.

## Minimal solver example

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

print(result.path_gain.shape)  # torch.Size([1, 1])
print(result.field.dtype)      # torch.complex64
print(result.field.device)     # cuda:0 (device index is environment-specific)
```

## Stability boundary

The frozen stable surface consists of the package root, the `path` and
`radiomap` solver modules, `witwin.channel.propagation.consumer`, and the
`importers` and `presets` helper modules. `witwin.channel.experimental.*` is
importable and tested but outside the frozen surface; see
[`docs/migration-0.6.md`](../migration-0.6.md) for the 0.6.0 renames. The scene lifecycle is a
small supporting surface used to obtain `CompiledScene` for the consumer.

An importable name is not automatically public. In particular, `__all__`
inside an implementation-owner module controls internal wildcard imports; it
does not override the repository's frozen public API policy. No predecessor
product name or compatibility alias is part of this documentation.
