# `witwin.channel.montecarlo.basic`

The Basic Monte Carlo solver produces incoherent sampled power estimates for
links and receiver-grid radiomaps. It does not export individual paths.

```python
from witwin.channel.montecarlo.basic import Config, Result, solve
```

## `Config`

Frozen dataclass.

```text
Config(
    samples: int = 4096,
    max_depth: int = 1,
    seed: int = 0,
    components = frozenset({"los", "reflection", "diffraction"}),
    diagnostics: bool = False,
    ad_mode: str = "none",
    workspace_limit_bytes: int | None = 1 << 30,
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

```python
config = Config(
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
    config: Config,
    *,
    reference_frequency_hz,
) -> Result
```

### Parameters

| Parameter | Description |
| --- | --- |
| `scene` | Core logical world or snapshot. |
| `config` | Basic Monte Carlo configuration. |
| `reference_frequency_hz` | Positive reference frequency in hertz. |

### Returns

A `Result` containing total gain, component power, optional component maps,
metadata, and optional diagnostics.

```python
result = solve(scene, config, reference_frequency_hz=28e9)
print(result.path_gain.shape)             # for example torch.Size([1, 4, 4])
print(result.component_power.keys())      # requested supported components
print(result.component_maps["los"].shape) # same logical map shape
```

## `Result`

```text
Result(
    path_gain: torch.Tensor,
    component_power: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    component_maps: dict[str, torch.Tensor] | None = None,
)
```

| Field | Description |
| --- | --- |
| `path_gain` | `float32` total power gain; point receivers normally produce `(R, T)`, while a grid may produce `(T, H, W)`. |
| `component_power` | Component-name to CUDA power tensor. |
| `metadata` | Sample count, seed, components, build identity, execution counts, and phase convention. |
| `diagnostics` | Additional diagnostics when requested; otherwise `None`. |
| `component_maps` | Component-name to spatial map when a map layout exists; otherwise `None`. |

The result is all-or-nothing. Workspace overflow, capacity failure, or a
native exception never returns a truncated estimate.
