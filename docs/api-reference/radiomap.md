# `witwin.channel.radiomap`

Radiomaps and per-link power from one entry point. The estimator is chosen by the
configuration type passed to `solve`:

| Configuration | Estimator | Result | Reference |
| --- | --- | --- | --- |
| `Coherent` | Enumerated paths, combined per receiver as a complex field (`combine="field"`) or as power (`combine="power"`) | `CoherentRadioMap` | [Coherent](radiomap-coherent.md) |
| `Stochastic` | Monte Carlo ray tubes deposited per receiver, combined as power | `RadioMap` | [Stochastic](radiomap-stochastic.md) |

```python
from witwin.channel import radiomap

field_map = radiomap.solve(scene, radiomap.Coherent(max_depth=2), reference_frequency_hz=3.5e9)
power_map = radiomap.solve(scene, radiomap.Stochastic(samples=1 << 16), reference_frequency_hz=3.5e9)
```

## `solve`

```text
solve(
    scene: witwin.core.Scene | witwin.core.SceneSnapshot,
    config: Coherent | Stochastic,
    *,
    reference_frequency_hz: float | torch.Tensor,
) -> CoherentRadioMap | RadioMap
```

A `Coherent` config returns a `CoherentRadioMap` and a `Stochastic` config returns a
`RadioMap`. Any other object (for example a `path.Config`) raises `TypeError` before any
work. Dispatch adds no GPU work: the estimator's own solve runs unchanged.

## Exports

| Name | Kind |
| --- | --- |
| `Coherent`, `Stochastic` | Estimator configurations (frozen dataclasses). |
| `CoherentRadioMap`, `RadioMap` | Results. |
| `CoherentRadioMapMetadata`, `RadioMapMetadata` | `TypedDict`s naming the stable metadata keys. |
| `PathTable` | The compact path export of a `Coherent` solve (`export_paths=True`). |
| `solve` | The dispatching entry point. |

## Result semantics

Both results carry the result-semantics metadata keys described in the
[index](index.md#result-semantics):

| Key | `CoherentRadioMap` | `RadioMap` |
| --- | --- | --- |
| `estimator` | `coherent` | `stochastic` |
| `quantity` | `point_sample` for every component | `point_sample` for LoS and transmission; `cell_average` for reflection, diffraction and scattering |
| `combine` | `Coherent.combine` (`field` or `power`) | `power` |
| `complete` | from discovery (`metadata["discovery"]["complete"]`) | `False` |

The two estimators are independent implementations; neither imports the other.
