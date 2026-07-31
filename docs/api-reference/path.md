# `witwin.channel.path`

The Path solver publishes explicit antenna-aware propagation paths and provides
CIR, CFR, discrete-tap, filtering, and beamforming views.

```python
from witwin.channel.path import (
    Config,
    InteractionType,
    PathResult,
    RaggedPathSoA,
    solve,
)
```

## `Config`

Frozen dataclass.

```text
Config(
    max_depth: int = 1,
    components = frozenset({"los", "reflection", "diffraction"}),
    max_paths: int | None = None,
    max_paths_scope: str = "per_pair",
    ad_mode: str = "none",
    coupled_paths: bool = False,
    coupled_candidate_limit: int = 1_000_000,
    scattering_samples_per_m2: float = 8.0,
    scattering_max_paths_per_pair: int = 4096,
    scattering_power_threshold: float = 0.0,
    scattering_chain_max_depth: int = 0,
    scattering_chain_samples_per_m2: float = 2.0,
    scattering_chain_max_rows: int = 256,
    isb_boundary_taper: bool = False,
    isb_boundary_taper_width: float = 0.5,
)
```

| Parameter | Description |
| --- | --- |
| `max_depth` | Maximum reflection/transmission depth. Depth-capped components support at most 5. |
| `components` | Requested propagation components. `transmission` and `scattering` are opt-in. |
| `max_paths` | Optional positive maximum actual rows per endpoint pair. |
| `max_paths_scope` | Must be `"per_pair"`. |
| `ad_mode` | `"none"`, `"jvp"`, or `"vjp"`. |
| `coupled_paths` | Enable supported reflection-diffraction coupled paths. |
| `coupled_candidate_limit` | Positive fail-loud candidate limit; it is not a truncation policy. |
| `scattering_samples_per_m2` | Positive rough-surface sample density. |
| `scattering_max_paths_per_pair` | Positive maximum scattering rows per pair. |
| `scattering_power_threshold` | Non-negative absolute gain threshold for exported scattering rows. |
| `scattering_chain_max_depth` | Combined reflection depth around a coherent scattering vertex; 0 disables chains. |
| `scattering_chain_samples_per_m2` | Positive scattering-chain vertex density. |
| `scattering_chain_max_rows` | Positive retained chain rows per pair. |
| `isb_boundary_taper` | Enable the default-off ISB continuity taper. |
| `isb_boundary_taper_width` | Fresnel-penumbra width multiplier; defaults to 0.5. |

```python
config = Config(
    max_depth=2,
    components={"los", "reflection"},
    max_paths=64,
)
```

Invalid component, depth, AD, coupled, or scattering combinations raise during
configuration or solve preflight.

## `solve`

```text
solve(
    scene: witwin.core.Scene | witwin.core.SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> PathResult
```

### Parameters

| Parameter | Description |
| --- | --- |
| `scene` | Core logical world or snapshot. |
| `config` | Path-solver configuration. |
| `reference_frequency_hz` | Positive reference frequency in hertz; may be a scalar CUDA tensor on supported AD routes. |

### Returns

A padded `PathResult` whose path width is the actual largest per-link
cardinality.

```python
paths = solve(
    scene,
    Config(max_depth=1, components={"los", "reflection"}),
    reference_frequency_hz=3.5e9,
)
print(paths.a.shape)   # (R, Ra, T, Ta, P, time)
print(paths.tau.shape) # (R, Ra, T, Ta, P)
```

The solver uses only native CUDA/RayD production paths. A missing backend,
unsupported capability, capacity failure, or ABI error raises without a
partial result.

## `InteractionType`

`enum.IntFlag` used by `PathResult.interaction_type`.

| Member | Value | Meaning |
| --- | ---: | --- |
| `NONE` | `0` | Line of sight / no interaction. |
| `REFLECTION` | `1` | Specular reflection. |
| `DIFFRACTION` | `2` | Diffraction. |
| `TRANSMISSION` | `4` | Transmission. |
| `SCATTERING` | `8` | Rough-surface scattering. |

```python
is_reflection = (
    paths.interaction_type & int(InteractionType.REFLECTION)
) != 0
```

## `PathResult`

```text
PathResult(
    a: torch.Tensor,
    tau: torch.Tensor,
    theta_t: torch.Tensor,
    phi_t: torch.Tensor,
    theta_r: torch.Tensor,
    phi_r: torch.Tensor,
    valid: torch.Tensor,
    interaction_type: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    position: torch.Tensor,
    normal: torch.Tensor,
    num_paths: torch.Tensor,
    metadata: dict[str, Any],
    field_xyz: torch.Tensor | None = None,
    field_direction: torch.Tensor | None = None,
    tx_weights: torch.Tensor | None = None,
    rx_weights: torch.Tensor | None = None,
)
```

Frozen dataclass. Users normally obtain it from `solve` rather than constructing
it directly. The base path shape is `(R, Ra, T, Ta, P)`; `a` adds a final time
axis.

| Field | Shape and dtype | Description |
| --- | --- | --- |
| `a` | `(R, Ra, T, Ta, P, time) complex64` | Complex path coefficient. |
| `tau` | `(R, Ra, T, Ta, P) float32` | Delay in seconds. |
| `theta_t`, `phi_t` | base path shape, `float32` | Departure spherical angles in radians. |
| `theta_r`, `phi_r` | base path shape, `float32` | Source-facing arrival angles in radians. |
| `valid` | base path shape, `bool` | Padding validity. |
| `interaction_type` | `(..., P, D) int32` | Per-depth interaction flags. |
| `primitive_id`, `material_id` | `(..., P, D) int32` | Per-depth primitive and material IDs. |
| `position`, `normal` | `(..., P, D, 3) float32` | Interaction points and normals. |
| `num_paths` | `(R, Ra, T, Ta) int32` | Actual path count per link. |
| `metadata` | `dict[str, Any]` | Solver, phase, component, build, and execution records. |
| `field_xyz` | `(..., P, 3) complex64` | World-Cartesian field; omitted construction input becomes a zero tensor. |
| `field_direction` | `(..., P, 3) float32` | Field propagation direction. |
| `tx_weights`, `rx_weights` | complex tensor or `None` | Default array weights captured from the scene. |

A valid empty result has `P == 0`, for example
`a.shape == (R, Ra, T, Ta, 0, 1)`.

### Properties

| Property | Return type | Meaning |
| --- | --- | --- |
| `num_rx` | `int` | Receiver count `R`. |
| `num_rx_ant` | `int` | Receive-antenna count `Ra`. |
| `num_tx` | `int` | Transmitter count `T`. |
| `num_tx_ant` | `int` | Transmit-antenna count `Ta`. |
| `max_num_paths` | `int` | Actual padded width `P`, not provisioned capacity. |
| `num_time_steps` | `int` | Final dimension of `a`. |
| `max_depth` | `int` | Interaction width `D`. |
| `path_shape` | `tuple[int, int, int, int, int]` | `(R, Ra, T, Ta, P)`. |
| `path_count_shape` | `tuple[int, int, int, int]` | `(R, Ra, T, Ta)`. |
| `rx_id`, `tx_id` | `torch.Tensor` | Endpoint-index tensors with the same shape as `valid`. |
| `path_length_m` | `torch.Tensor` | `tau * c`; invalid padding is `-1`. |
| `types` | `torch.Tensor` | Convenience alias for `interaction_type`. |
| `vertices` | `torch.Tensor` | Convenience alias for `position`. |
| `normals` | `torch.Tensor` | Convenience alias for `normal`. |
| `objects` | `torch.Tensor` | Convenience alias for `primitive_id`. |

### `from_ragged`

```text
PathResult.from_ragged(
    ragged: RaggedPathSoA,
    *,
    max_paths_per_pair: int | None = None,
    minimum_path_width: int = 0,
    metadata: dict[str, Any] | None = None,
) -> PathResult
```

Converts stable pair-grouped ragged rows into the public padded layout.

| Parameter | Description |
| --- | --- |
| `ragged` | Input `RaggedPathSoA`. |
| `max_paths_per_pair` | Optional non-negative padded width; input counts must not exceed it. |
| `minimum_path_width` | Non-negative minimum padded width. |
| `metadata` | Result metadata; `None` uses an empty dictionary. |

When no width is supplied, construction observes the maximum pair count to
allocate exact output.

### `cir`

```text
cir(
    *,
    normalize_delays: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]
```

Returns `(a, tau)`. With `normalize_delays=True`, the earliest valid delay of
each link is shifted to zero. Invalid coefficients become zero and invalid
delays remain `-1`.

```python
coefficients, delays = paths.cir()
print(coefficients.shape)  # same as paths.a
print(delays.shape)        # same as paths.tau
```

### `cfr`

```text
cfr(
    frequencies: torch.Tensor,
    *,
    normalize_delays: bool = True,
) -> torch.Tensor
```

Evaluates the channel frequency response with
`a * exp(-j*2*pi*tau*frequency)` and sums over the path axis.

| Parameter | Description |
| --- | --- |
| `frequencies` | One-dimensional frequency or offset tensor in hertz; converted to result device and `float32`. |
| `normalize_delays` | Normalize each link's earliest delay before evaluation. |

```python
frequency_offsets = torch.tensor(
    [-1e6, 0.0, 1e6], device=paths.a.device
)
h = paths.cfr(frequency_offsets)
print(h.shape)  # (R, Ra, T, Ta, time, F), here F == 3
```

A non-one-dimensional frequency tensor raises `ValueError`.

### `taps`

```text
taps(
    bandwidth: float,
    num_taps: int,
    *,
    normalize_delays: bool = True,
) -> torch.Tensor
```

Accumulates each valid path into the nearest discrete bin using
`round(tau * bandwidth)`.

| Parameter | Description |
| --- | --- |
| `bandwidth` | Positive sampling bandwidth in hertz. |
| `num_taps` | Positive output tap count. |
| `normalize_delays` | Normalize the earliest delay before binning. |

```python
taps = paths.taps(20e6, 64)
print(taps.shape)  # (R, Ra, T, Ta, time, 64)
```

Non-positive bandwidth or tap count raises `ValueError`.

### `beamform`

```text
beamform(
    *,
    tx_weights: torch.Tensor | None = None,
    rx_weights: torch.Tensor | None = None,
)
```

Returns a signal view with complex transmit precoding and receive combining
weights. The source `PathResult` is unchanged. Omitted weights use values
captured from the solved scene.

| Parameter | Required shape |
| --- | --- |
| `tx_weights` | `(T, Ta)`, or `(Ta,)` when `T == 1` |
| `rx_weights` | `(R, Ra)`, or `(Ra,)` when `R == 1` |

Weights are converted to `complex64` on the result device. If either weight is
unavailable or a shape is wrong, the method raises `ValueError`.

```python
view = paths.beamform(
    tx_weights=torch.ones(
        (paths.num_tx, paths.num_tx_ant), device=paths.a.device
    ),
    rx_weights=torch.ones(
        (paths.num_rx, paths.num_rx_ant), device=paths.a.device
    ),
)
coefficient, delay = view.cir()
```

The return object is a method-owned signal view, not a separately exported
stable construction type.

### `filter_by_type`

```text
filter_by_type(*interaction_types: int) -> PathResult
```

Keeps paths containing any requested interaction value and compacts padding.
Pass `InteractionType` members or equivalent integers. `NONE` selects LoS.
Calling with no arguments returns the same object.

```python
reflected = paths.filter_by_type(InteractionType.REFLECTION)
print(reflected.max_num_paths <= paths.max_num_paths)  # True
```

## `RaggedPathSoA`

Stable pair-grouped structure-of-arrays storage. It is an advanced public
construction utility; most callers only need `PathResult`.

```text
RaggedPathSoA(
    num_rx: int,
    num_rx_ant: int,
    num_tx: int,
    num_tx_ant: int,
    num_time_steps: int,
    pair_offsets: torch.Tensor,
    rx_id: torch.Tensor,
    rx_ant_id: torch.Tensor,
    tx_id: torch.Tensor,
    tx_ant_id: torch.Tensor,
    field: torch.Tensor,
    delay_s: torch.Tensor,
    theta_t: torch.Tensor,
    phi_t: torch.Tensor,
    theta_r: torch.Tensor,
    phi_r: torch.Tensor,
    interaction_type: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    position: torch.Tensor,
    normal: torch.Tensor,
)
```

| Field group | Type and shape |
| --- | --- |
| `num_rx`, `num_rx_ant`, `num_tx`, `num_tx_ant`, `num_time_steps` | `int` |
| `pair_offsets` | `(pair_count + 1,) int64` |
| `rx_id`, `rx_ant_id`, `tx_id`, `tx_ant_id` | `(K,) int32` |
| `field` | `(K, time) complex64` |
| `delay_s`, `theta_t`, `phi_t`, `theta_r`, `phi_r` | `(K,) float32` |
| `interaction_type`, `primitive_id`, `material_id` | `(K, D) int32` |
| `position`, `normal` | `(K, D, 3) float32` |

Properties:

- `path_count -> int`: returns `K`.
- `pair_count -> int`: returns `R * Ra * T * Ta`.
- `max_depth -> int`: returns `D`.

### `from_flat`

```text
RaggedPathSoA.from_flat(
    *,
    num_rx: int,
    num_rx_ant: int,
    num_tx: int,
    num_tx_ant: int,
    rx_id: torch.Tensor,
    tx_id: torch.Tensor,
    field: torch.Tensor,
    delay_s: torch.Tensor,
    theta_t: torch.Tensor,
    phi_t: torch.Tensor,
    theta_r: torch.Tensor,
    phi_r: torch.Tensor,
    interaction_type: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    position: torch.Tensor,
    normal: torch.Tensor,
    rx_ant_id: torch.Tensor | None = None,
    tx_ant_id: torch.Tensor | None = None,
    max_paths_per_pair: int | None = None,
) -> RaggedPathSoA
```

Accepts flat rows, stably sorts them by receiver-major pair index, and creates
CSR `pair_offsets`. Omitted antenna IDs become zero. A positive
`max_paths_per_pair` keeps the first stable rows of each pair.

```python
ragged = RaggedPathSoA.from_flat(
    num_rx=1,
    num_rx_ant=1,
    num_tx=1,
    num_tx_ant=1,
    rx_id=torch.tensor([0], device="cuda"),
    tx_id=torch.tensor([0], device="cuda"),
    field=torch.tensor([[1 + 0j]], device="cuda"),
    delay_s=torch.tensor([1e-8], device="cuda"),
    theta_t=torch.zeros(1, device="cuda"),
    phi_t=torch.zeros(1, device="cuda"),
    theta_r=torch.zeros(1, device="cuda"),
    phi_r=torch.zeros(1, device="cuda"),
    interaction_type=torch.empty((1, 0), dtype=torch.int32, device="cuda"),
    primitive_id=torch.empty((1, 0), dtype=torch.int32, device="cuda"),
    material_id=torch.empty((1, 0), dtype=torch.int32, device="cuda"),
    position=torch.empty((1, 0, 3), device="cuda"),
    normal=torch.empty((1, 0, 3), device="cuda"),
)
print(ragged.pair_offsets)  # tensor([0, 1], device="cuda:0")
```

Endpoint IDs outside declared dimensions, invalid shapes/dtypes, or a
non-positive explicit row limit raise `ValueError`.
