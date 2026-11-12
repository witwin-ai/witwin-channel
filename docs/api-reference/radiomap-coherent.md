# `witwin.channel.radiomap.Coherent`

The Coherent radiomap estimator enumerates propagation paths and combines their
complex fields per receiver. It is intended for repeatable links, radiomaps,
and optional compact path-table export. It is selected by passing a `Coherent`
configuration to [`radiomap.solve`](radiomap.md); the Monte Carlo counterpart is
[`Stochastic`](radiomap-stochastic.md).

```python
from witwin.channel.radiomap import Coherent, CoherentRadioMap, PathTable, solve
```

## `Coherent`

Frozen dataclass; values cannot be changed after construction. The first 16
fields are the shared enumerated options, defined once and inherited by both
`radiomap.Coherent` and `path.Config` (see "Shared enumerated options" in
[`path`](path.md)); the last six are read only by the Coherent terminal stage.

```text
Coherent(
    max_depth: int = 1,
    components = frozenset({"los", "reflection", "diffraction"}),
    max_paths: int | None = None,
    max_paths_scope: str = "global",
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
    discovery: str = "auto",
    combine: Literal["field", "power"] = "field",
    return_field: bool = True,
    export_paths: bool = False,
    sort_key: str = "receiver_transmitter_depth_component",
    diagnostics: bool = False,
    scattering_coherent: bool = False,
)
```

| Parameter | Description |
| --- | --- |
| `max_depth` | Maximum reflection/transmission depth. Depth-capped components support at most 5. |
| `components` | Supported subset of `los`, `reflection`, `diffraction`, `transmission`, and `scattering`. `"diffraction"` is the only diffraction switch (first order). |
| `max_paths` | Optional positive exported-row limit, applied per `max_paths_scope`. |
| `max_paths_scope` | `"global"` (default) or `"per_pair"`. |
| `ad_mode` | `"none"`, `"jvp"`, or `"vjp"`. |
| `coupled_paths` | Enable supported reflection-diffraction coupled paths. |
| `coupled_candidate_limit` | Positive fail-loud candidate limit. |
| `scattering_samples_per_m2` | Positive rough-surface sample density. |
| `scattering_max_paths_per_pair` | Positive per-pair scattering-row limit. |
| `scattering_power_threshold` | Non-negative absolute gain threshold for exported scattering rows. |
| `scattering_chain_max_depth` | Combined reflection depth around a coherent scattering vertex; 0 disables chains. |
| `scattering_chain_samples_per_m2` | Positive chain-vertex sample density. |
| `scattering_chain_max_rows` | Positive per-pair retained chain-row limit. |
| `isb_boundary_taper` | Enable the default-off ISB continuity taper. |
| `isb_boundary_taper_width` | Fresnel-penumbra width multiplier; defaults to 0.5. |
| `discovery` | Reflection candidate source: `"auto"` (complete image tree; a depth whose tree exceeds the cost budget, and every deeper one, uses ray discovery), `"exhaustive"` (every plane-group sequence; fails before work if it cannot fit in memory) or `"rays"` (ray discovery, never complete). `metadata["complete"]` is true only when every reflection depth came from a complete source; `metadata["discovery"]["sources"]` names the source per depth. |
| `combine` | How rows reach a receiver value: `"field"` sums the complex path fields per (transmitter, receiver) and squares the sum; `"power"` sums the per-path powers. |
| `return_field` | Preserve complex field output. |
| `export_paths` | Produce a compact `PathTable`. |
| `sort_key` | Stable path ordering policy. |
| `diagnostics` | Collect additional diagnostic records. |
| `scattering_coherent` | Select the supported coherent scattering-combination policy. |

Invalid values or unsupported combinations raise `TypeError`, `ValueError`,
`NotImplementedError`, or `RuntimeError` during configuration or solve
preflight. Requested capabilities are never silently removed.

```python
config = Coherent(
    max_depth=2,
    components={"los", "reflection"},
    export_paths=True,
    diagnostics=True,
)
```

## `solve`

```text
solve(
    scene: witwin.core.Scene | witwin.core.SceneSnapshot,
    config: Coherent,
    *,
    reference_frequency_hz: float | torch.Tensor,
) -> CoherentRadioMap
```

### Parameters

| Parameter | Description |
| --- | --- |
| `scene` | Core logical world or snapshot. |
| `config` | A `witwin.channel.radiomap.Coherent`; `radiomap.solve` dispatches on it, and another solver's config is rejected. |
| `reference_frequency_hz` | Positive reference frequency in hertz; may be a scalar CUDA tensor on supported AD routes. |

### Returns

A `CoherentRadioMap` containing total gain, field, component breakdowns,
optional compact paths, metadata, and optional diagnostics.

```python
result = solve(
    scene,
    Coherent(max_depth=0, components={"los"}),
    reference_frequency_hz=3.5e9,
)
print(result.path_gain.shape)  # (T, R); one ReceiverGrid gives (T, rows, cols)
print(result.field.dtype)      # torch.complex64
```

Missing CUDA, an incompatible extension, an unsupported component/AD
combination, or a frequency/scene mismatch raises before a partial result.

## `CoherentRadioMap`

```text
CoherentRadioMap(
    path_gain: torch.Tensor,
    field: torch.Tensor,
    component_power: dict[str, torch.Tensor],
    component_fields: dict[str, torch.Tensor],
    paths: PathTable | None,
    metadata: CoherentRadioMapMetadata,
    diagnostics: dict[str, Any] | None = None,
)
```

| Field | Description |
| --- | --- |
| `path_gain` | `float32` total power gain per link or radiomap cell. |
| `field` | `complex64` total field with the same logical layout. |
| `component_power` | Component-name to power tensor. |
| `component_fields` | Component-name to complex-field tensor. |
| `paths` | Compact `PathTable` when `export_paths=True`; otherwise `None`. |
| `metadata` | `CoherentRadioMapMetadata`: solver, component, phase, build, and execution records (see [`CoherentRadioMapMetadata`](#coherentradiomapmetadata)). |
| `diagnostics` | Additional records when requested; otherwise normally `None`. |

## `CoherentRadioMapMetadata`

A `TypedDict` naming the stable keys; at run time it is a plain `dict`, so key access, `in`, `.get` and JSON serialization are unchanged. Every solver publishes the result-semantics keys `estimator`, `quantity`, `combine` and `complete` (see the [index](index.md)); here `estimator` is `coherent` and `combine` is `Coherent.combine`, the `kernel` execution record, and the requested/effective configuration keys `requested_config`, `effective_config`, `requested_max_depth`, `effective_max_depth` and `component_max_depth`.

| Key | Type | Meaning |
| --- | --- | --- |
| `max_depth`, `return_field`, `export_paths`, `max_paths`, `max_paths_scope`, `sort_key` | config values | The solve's options (`combine` is the result-semantics key). |
| `accumulation_strategy` | `str` | `coherent` (`combine="field"`) or `incoherent` (`combine="power"`). |
| `components` | `dict[str, str]` | Status per component (`enabled`, `not_requested`, ...). |
| `counts` | `dict` | `path_count`, `valid_path_count` and per-component counts. |
| `capability` | `dict[str, bool]` | Native capability flags used by the solve. |
| `kernel` | `KernelMetadata` | Launch, tape, memory and timing record. |
| `field_abi`, `phase_convention`, `coefficient_semantics` | | Field and phase conventions. |
| `coupled_paths` | `dict` | Coupled interaction-chain record. |
| `discovery` | `DiscoveryMetadata` | `requested`, `sources` (candidate source per reflection depth) and `complete`. |
| `semantic_capabilities` | `dict` | This solver's capability record. |
| `transmission`, `scattering` | `dict` | Only when that component was requested. |
| `dropped_faces` | `dict[str, int]` | Only when compilation removed degenerate or duplicate faces. |

## `PathTable`

`PathTable` contains exactly the actual `K` exported rows. It is not
capacity-shaped. Every row field has leading dimension `K`; sequence fields
use interaction width `D`.

| Field | Shape and dtype | Description |
| --- | --- | --- |
| `valid` | `(K,) bool` | Row validity. Published rows in a complete success are valid. |
| `tx_id`, `rx_id` | `(K,) int32` | 0-based transmitter and receiver indices in scene endpoint order (a grid contributes one receiver per cell), not antenna IDs. |
| `depth` | `(K,) int32` | Interaction depth. |
| `component_id` | `(K,) int32` | Component code. |
| `primitive_id`, `edge_id`, `material_id` | `(K,) int32` | First-interaction convenience columns. |
| `path_length_m`, `delay_s`, `path_gain` | `(K,) float32` | Length, delay, and gain. |
| `primitive_sequence`, `material_sequence` | `(K, D) int32` | Complete interaction sequences. |
| `interaction_positions`, `interaction_normals` | `(K, D, 3) float32` | Complete interaction geometry; `[:, 0]` is the first interaction (when `D > 0`). |
| `field_real`, `field_imag` | `(K,) float32` | Real and imaginary path-field components. |
| `coefficient` | `(K,) complex64` | Complex path coefficient. |
| `field_xyz` | `(K, 3) complex64` | World-Cartesian complex field. |
| `field_direction` | `(K, 3) float32` | Field propagation direction. |
| `phase_rad` | `(K,) float32` | Reference-frequency phase. |
| `interaction_count` | `(K,) int32` | Actual interaction count. |

```python
result = solve(
    scene,
    Coherent(export_paths=True, components={"los"}),
    reference_frequency_hz=3.5e9,
)
paths = result.paths
assert paths is not None
print(paths.delay_s.shape)               # torch.Size([K])
print(paths.interaction_positions.shape) # torch.Size([K, D, 3])
```
