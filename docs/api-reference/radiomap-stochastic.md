# `witwin.channel.radiomap.Stochastic`

The Stochastic radiomap estimator produces Monte Carlo power estimates for
links and receiver-grid radiomaps, combined in the power domain. It does not
export individual paths. It is selected by passing a `Stochastic`
configuration to [`radiomap.solve`](radiomap.md); the enumerated counterpart is
[`Coherent`](radiomap-coherent.md).

```python
from witwin.channel.radiomap import RadioMap, Stochastic, solve
```

## `Stochastic`

Frozen dataclass.

```text
Stochastic(
    samples: int = 4096,
    max_depth: int = 1,
    seed: int = 0,
    components = frozenset({"los", "reflection", "diffraction"}),
    diagnostics: bool = False,
    ad_mode: str = "none",
    workspace_limit_bytes: int | None = 1 << 30,
    receiver_radius_m: float | None = None,
    scattering_samples: int | None = None,
)
```

| Parameter | Description |
| --- | --- |
| `samples` | Positive sample count. |
| `max_depth` | Maximum path depth within the solver's advertised capability. |
| `seed` | Deterministic random-stream seed. |
| `components` | Requested propagation-component set. |
| `diagnostics` | Return additional counters and performance diagnostics. |
| `ad_mode` | `"none"`, `"jvp"`, or `"vjp"`; unsupported component combinations fail before launch. |
| `workspace_limit_bytes` | Positive native workspace budget or `None`; insufficient budget fails rather than truncating samples. |
| `receiver_radius_m` | Capture-sphere radius of point receivers, or `None` to derive it from the transmitter distance. |
| `scattering_samples` | Area samples of the scattering estimator; `None` uses `samples`. |

```python
config = Stochastic(
    samples=16_384,
    max_depth=2,
    seed=2026,
    components={"los", "reflection"},
)
```

## `solve`

```text
solve(
    scene: witwin.core.Scene | witwin.core.SceneSnapshot,
    config: Stochastic,
    *,
    reference_frequency_hz: float | torch.Tensor,
) -> RadioMap
```

### Parameters

| Parameter | Description |
| --- | --- |
| `scene` | Core logical world or snapshot. |
| `config` | A `witwin.channel.radiomap.Stochastic`; `radiomap.solve` dispatches on it. |
| `reference_frequency_hz` | Positive reference frequency in hertz. |

### Returns

A `RadioMap` containing total gain, component power, optional component maps,
metadata, and optional diagnostics.

```python
result = solve(scene, config, reference_frequency_hz=28e9)
print(result.path_gain.shape)             # for example torch.Size([1, 4, 4])
print(result.component_power.keys())      # requested supported components
print(result.component_maps["los"].shape) # same logical map shape
```

## `RadioMap`

```text
RadioMap(
    path_gain: torch.Tensor,
    component_power: dict[str, torch.Tensor],
    metadata: RadioMapMetadata,
    diagnostics: dict[str, Any] | None = None,
    component_maps: dict[str, torch.Tensor] | None = None,
)
```

| Field | Description |
| --- | --- |
| `path_gain` | `float32` total power gain; `[T, R]` for points or mixed receivers, `[T, rows, cols]` for exactly one grid (see the [receiver layout](index.md#receiver-layout-and-grid-axis-order)). |
| `component_power` | Component-name to CUDA power tensor. |
| `metadata` | `RadioMapMetadata`: sample count, seed, components, execution counts and policies (see [`RadioMapMetadata`](#radiomapmetadata)). |
| `diagnostics` | Additional diagnostics when requested; otherwise `None`. |
| `component_maps` | Component-name to spatial map when a map layout exists; otherwise `None`. |

The result is all-or-nothing. Workspace overflow, capacity failure, or a
native exception never returns a truncated estimate.

## `RadioMapMetadata`

A `TypedDict` naming the stable keys; at run time it is a plain `dict`, so key access, `in`, `.get` and JSON serialization are unchanged. Every solver publishes the result-semantics keys `estimator`, `quantity`, `combine` and `complete` (see the [index](index.md)); here `estimator` is `stochastic`, `combine` is `power` and `complete` is `False`, the `kernel` execution record, and the requested/effective configuration keys `requested_config`, `effective_config`, `requested_max_depth`, `effective_max_depth` and `component_max_depth`.

| Key | Type | Meaning |
| --- | --- | --- |
| `seed`, `samples`, `max_depth`, `ad_mode` | config values | The solve's options. |
| `path_count`, `contribution_capacity` | `int` | Contribution counts of the solve. |
| `components` | `dict[str, str]` | Status per component. |
| `rayd` | `dict[str, bool]` | RayD reflection/diffraction availability. |
| `kernel` | `KernelMetadata` | Launch, tape, memory and timing record. |
| `semantic_capabilities` | `dict` | This solver's capability record. |
| `edge_policy` | `dict` | Diffraction edge selection policy of the scene. |
| `transmission`, `scattering` | `dict` | Only when that component was requested. |
| `dropped_faces` | `dict[str, int]` | Only when compilation removed degenerate or duplicate faces. |
