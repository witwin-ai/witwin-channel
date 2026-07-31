# `witwin.channel.deterministic`

The deterministic solver discovers propagation paths and coherently
accumulates their complex fields. It is intended for repeatable links,
radiomaps, and optional compact path-table export.

```python
from witwin.channel.deterministic import Config, PathTable, Result, solve
```

## `Config`

Frozen dataclass; values cannot be changed after construction.

```text
Config(
    max_depth: int = 1,
    max_diffraction_order: int = 1,
    components = frozenset({"los", "reflection", "diffraction"}),
    coherent: bool = True,
    return_field: bool = True,
    export_paths: bool = False,
    max_paths: int | None = None,
    max_paths_scope: str = "global",
    sort_key: str = "receiver_transmitter_depth_component",
    diagnostics: bool = False,
    ad_mode: str = "none",
    coupled_paths: bool = False,
    coupled_candidate_limit: int = 1_000_000,
    scattering_samples_per_m2: float = 8.0,
    scattering_max_paths_per_pair: int = 4096,
    scattering_power_threshold: float = 0.0,
    scattering_coherent: bool = False,
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
| `max_diffraction_order` | Maximum diffraction order; the current public capability is first order. |
| `components` | Supported subset of `los`, `reflection`, `diffraction`, `transmission`, and `scattering`. |
| `coherent` | Coherently combine deterministic complex-field contributions. |
| `return_field` | Preserve complex field output. |
| `export_paths` | Produce a compact `PathTable`. |
| `max_paths` | Optional positive global exported-row limit. |
| `max_paths_scope` | Must be `"global"`. |
| `sort_key` | Stable path ordering policy. |
| `diagnostics` | Collect additional diagnostic records. |
| `ad_mode` | `"none"`, `"jvp"`, or `"vjp"`. |
| `coupled_paths` | Enable supported reflection-diffraction coupled paths. |
| `coupled_candidate_limit` | Positive fail-loud candidate limit. |
| `scattering_samples_per_m2` | Positive rough-surface sample density. |
| `scattering_max_paths_per_pair` | Positive per-pair scattering-row limit. |
| `scattering_power_threshold` | Non-negative absolute gain threshold for exported scattering rows. |
| `scattering_coherent` | Select the supported coherent scattering-combination policy. |
| `scattering_chain_max_depth` | Combined reflection depth around a coherent scattering vertex; 0 disables chains. |
| `scattering_chain_samples_per_m2` | Positive chain-vertex sample density. |
| `scattering_chain_max_rows` | Positive per-pair retained chain-row limit. |
| `isb_boundary_taper` | Enable the default-off ISB continuity taper. |
| `isb_boundary_taper_width` | Fresnel-penumbra width multiplier; defaults to 0.5. |

Invalid values or unsupported combinations raise `TypeError`, `ValueError`,
`NotImplementedError`, or `RuntimeError` during configuration or solve
preflight. Requested capabilities are never silently removed.

```python
config = Config(
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
    config: Config,
    *,
    reference_frequency_hz,
) -> Result
```

### Parameters

| Parameter | Description |
| --- | --- |
| `scene` | Core logical world or snapshot. |
| `config` | A `witwin.channel.deterministic.Config`; another solver's config is rejected. |
| `reference_frequency_hz` | Positive reference frequency in hertz; may be a scalar CUDA tensor on supported AD routes. |

### Returns

A `Result` containing total gain, coherent field, component breakdowns,
optional compact paths, metadata, and optional diagnostics.

```python
result = solve(
    scene,
    Config(max_depth=0, components={"los"}),
    reference_frequency_hz=3.5e9,
)
print(result.path_gain.shape)  # (R, T); a grid may produce (T, H, W)
print(result.field.dtype)      # torch.complex64
```

Missing CUDA, an incompatible extension, an unsupported component/AD
combination, or a frequency/scene mismatch raises before a partial result.

## `Result`

```text
Result(
    path_gain: torch.Tensor,
    field: torch.Tensor,
    component_power: dict[str, torch.Tensor],
    component_fields: dict[str, torch.Tensor],
    paths: PathTable | None,
    metadata: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
)
```

| Field | Description |
| --- | --- |
| `path_gain` | `float32` total power gain per link or radiomap cell. |
| `field` | `complex64` coherent total field with the same logical layout. |
| `component_power` | Component-name to power tensor. |
| `component_fields` | Component-name to complex-field tensor. |
| `paths` | Compact `PathTable` when `export_paths=True`; otherwise `None`. |
| `metadata` | Solver, component, phase, build, and execution records. |
| `diagnostics` | Additional records when requested; otherwise normally `None`. |

## `PathTable`

`PathTable` contains exactly the actual `K` exported rows. It is not
capacity-shaped. Every row field has leading dimension `K`; sequence fields
use interaction width `D`.

| Field | Shape and dtype | Description |
| --- | --- | --- |
| `valid` | `(K,) bool` | Row validity. Published rows in a complete success are valid. |
| `tx_id`, `rx_id` | `(K,) int32` | Stable endpoint IDs. |
| `depth` | `(K,) int32` | Interaction depth. |
| `component_id` | `(K,) int32` | Component code. |
| `primitive_id`, `edge_id`, `material_id` | `(K,) int32` | First-interaction convenience columns. |
| `path_length_m`, `delay_s`, `path_gain` | `(K,) float32` | Length, delay, and gain. |
| `interaction_position`, `interaction_normal` | `(K, 3) float32` | First-interaction convenience views. |
| `primitive_sequence`, `material_sequence` | `(K, D) int32` | Complete interaction sequences. |
| `interaction_positions`, `interaction_normals` | `(K, D, 3) float32` | Complete interaction geometry. |
| `field_real`, `field_imag` | `(K,) float32` | Real and imaginary path-field components. |
| `coefficient` | `(K,) complex64` | Complex path coefficient. |
| `field_xyz` | `(K, 3) complex64` | World-Cartesian complex field. |
| `field_direction` | `(K, 3) float32` | Field propagation direction. |
| `phase_rad` | `(K,) float32` | Reference-frequency phase. |
| `interaction_count` | `(K,) int32` | Actual interaction count. |

```python
result = solve(
    scene,
    Config(export_paths=True, components={"los"}),
    reference_frequency_hz=3.5e9,
)
paths = result.paths
assert paths is not None
print(paths.delay_s.shape)               # torch.Size([K])
print(paths.interaction_positions.shape) # torch.Size([K, D, 3])
```
