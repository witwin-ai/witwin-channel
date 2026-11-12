# `witwin.channel.experimental.bdpt`

> **Experimental tier.** This module is importable and tested but is not part of
> the stable API snapshot: its names, options and results may change in any
> release (see `docs/dev/standards/adr-0xx-experimental-api-tier.md`). It was
> `witwin.channel.montecarlo.bdpt` before 0.6.0; its behaviour is unchanged.

The bidirectional path tracing (BDPT) solver estimates received power with
multiple importance sampling and can optionally export actual BDPT samples.
Its current public contract is primal-only.

```python
from witwin.channel.experimental.bdpt import (
    BDPTPathSamples,
    Config,
    Result,
    solve,
)
```

## `Config`

Frozen dataclass.

```text
Config(
    samples: int = 4096,
    seed: int = 0,
    max_depth: int = 3,
    max_light_depth: int | None = None,
    max_diffraction_order: int = 1,
    max_scattering_order: int = 1,
    coupled_paths: bool = False,
    coupled_candidate_limit: int = 1_000_000,
    components = frozenset({"los", "reflection", "diffraction"}),
    coherent: bool = False,
    mis: str = "power_heuristic",
    power_heuristic_beta: float = 2.0,
    receiver_strategy: str = "grid_area",
    accumulation_strategy: str = "auto",
    sample_streams: int = 1,
    diagnostics: bool = False,
    export_paths: bool = False,
    max_exported_paths: int | None = None,
    ad_mode: str = "none",
    workspace_limit_bytes: int | None = 1 << 30,
)
```

| Parameter | Description |
| --- | --- |
| `samples` | Positive sample count. |
| `seed` | Random-stream seed. |
| `max_depth` | Maximum complete connected-path depth. |
| `max_light_depth` | Optional light-subpath depth limit; `None` derives it from the full depth. |
| `max_diffraction_order` | Maximum diffraction order. |
| `max_scattering_order` | Maximum scattering order. |
| `coupled_paths` | Enable supported coupled paths. |
| `coupled_candidate_limit` | Positive fail-loud coupled-candidate limit. |
| `components` | Requested propagation-component set. |
| `coherent` | Select the supported coherent/incoherent accumulation semantics. |
| `mis` | MIS policy; the standard policy is `"power_heuristic"`. |
| `power_heuristic_beta` | Positive power-heuristic exponent. |
| `receiver_strategy` | Receiver sampling strategy; grids default to `"grid_area"`. |
| `accumulation_strategy` | `"auto"` or another advertised native accumulation strategy. |
| `sample_streams` | Positive independent sample-stream count. |
| `diagnostics` | Publish additional diagnostics. |
| `export_paths` | Export `BDPTPathSamples`. |
| `max_exported_paths` | Optional positive exported-row limit. |
| `ad_mode` | Must currently be `"none"`; BDPT AD is not public. |
| `workspace_limit_bytes` | Native workspace budget or `None`. |

## `solve`

```text
solve(
    scene: witwin.core.Scene | witwin.core.SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz: float | torch.Tensor,
) -> Result
```

```python
result = solve(
    scene,
    Config(
        samples=65_536,
        seed=7,
        max_depth=3,
        export_paths=True,
        max_exported_paths=1024,
    ),
    reference_frequency_hz=28e9,
)
print(result.path_gain.shape)  # for example torch.Size([1, 4, 4])
print(result.variance.shape)   # same shape when variance is published
```

The parameters have the same scene and frequency meaning as the shared solver
contract. Invalid configuration, unsupported capability, workspace overflow,
or native failure raises before a partial estimate.

## `Result`

```text
Result(
    path_gain: torch.Tensor,
    component_power: dict[str, torch.Tensor],
    metadata: Metadata,
    diagnostics: dict[str, Any] | None = None,
    component_maps: dict[str, torch.Tensor] | None = None,
    variance: torch.Tensor | None = None,
    path_samples: BDPTPathSamples | None = None,
)
```

| Field | Description |
| --- | --- |
| `path_gain` | `float32` total power gain. |
| `component_power` | Component-name to power tensor. |
| `metadata` | `Metadata`: sample, seed, MIS, accumulation and execution records (see [`Metadata`](#metadata)). |
| `diagnostics` | Optional diagnostics. |
| `component_maps` | Optional component-name to spatial map. |
| `variance` | Estimator variance with the same shape as `path_gain`, when available. |
| `path_samples` | `BDPTPathSamples` when `export_paths=True`; otherwise `None`. |

## `Metadata`

A `TypedDict` naming the stable keys; at run time it is a plain `dict`, so key access, `in`, `.get` and JSON serialization are unchanged. Every solver publishes the result-semantics keys `estimator`, `quantity`, `combine` and `complete` (see the [index](index.md)), the `kernel` execution record, and the requested/effective configuration keys `requested_config`, `effective_config`, `requested_max_depth`, `effective_max_depth` and `component_max_depth`.

| Key | Type | Meaning |
| --- | --- | --- |
| `samples`, `seed`, `sample_streams`, `stream_count`, `mis`, `power_heuristic_beta`, `coherent`, `max_depth`, `max_light_depth`, `max_diffraction_order`, `max_scattering_order` | config values | The solve's options. |
| `combine_domain` | `str` | `coherent` or `power`. |
| `path_counts_by_strategy`, `valid_contribution_count`, `launch_count`, `workspace_bytes` | `int` / `dict[str, int]` | Execution counts. |
| `accumulation_strategy`, `sampler` | `str` | Selected accumulation and sampler. |
| `variance` | `bool` | Whether `Result.variance` is published. |
| `components` | `dict[str, str]` | Status per component. |
| `native_capabilities`, `rayd` | `dict[str, bool]` | Native capability flags. |
| `throughput_domain`, `scattering_depth_rule`, `pdf_domain`, `delta_strategy`, `sampled_delta_mass` | `str` | Estimator conventions. |
| `field_transport`, `event_classification`, `component_mask_bits`, `mis_capabilities` | `dict` | Estimator conventions. |
| `ad_status`, `ad_geometry`, `ad_differentiable_parameters` | `str` / `list[str]` | AD scope of the solve. |
| `kernel` | `KernelMetadata` | Launch, tape, memory and timing record. |
| `semantic_capabilities` | `dict` | This solver's capability record. |
| `edge_policy` | `dict` | Diffraction edge selection policy of the scene. |
| `transmission`, `scattering` | `dict` | Only when that component was requested. |
| `dropped_faces` | `dict[str, int]` | Only when compilation removed degenerate or duplicate faces. |

## `BDPTPathSamples`

Compact storage with exactly `N` exported sample rows.

```text
BDPTPathSamples(
    topology: torch.Tensor,
    contribution: torch.Tensor,
    pdf: torch.Tensor,
    mis_weight: torch.Tensor,
    component_id: torch.Tensor,
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    grid_linear_id: torch.Tensor,
    light_depth: torch.Tensor,
    sensor_depth: torch.Tensor,
    path_length_m: torch.Tensor,
)
```

| Field | Typical shape | Description |
| --- | --- | --- |
| `topology` | `(N, ...)` | Encoded complete discrete topology. |
| `contribution` | `(N,)` | Sample contribution. |
| `pdf` | `(N,)` | Sampling probability density. |
| `mis_weight` | `(N,)` | MIS weight. |
| `component_id` | `(N,)` | Propagation-component code. |
| `valid` | `(N,)` | Validity mask. |
| `tx_id`, `rx_id` | `(N,)` | 0-based transmitter and receiver indices in scene endpoint order, not antenna IDs. |
| `grid_linear_id` | `(N,)` | Linear receiver-grid index; non-grid cases use the contract sentinel. |
| `light_depth`, `sensor_depth` | `(N,)` | Light and sensor subpath depths. |
| `path_length_m` | `(N,)` | Complete path length in metres. |

```python
samples = result.path_samples
assert samples is not None
print(samples.contribution.shape)  # torch.Size([N])
print(samples.mis_weight.shape)    # torch.Size([N])
```
