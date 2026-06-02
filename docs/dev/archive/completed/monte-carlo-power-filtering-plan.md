Status: Draft
Category: Plan
Last reviewed: 2026-04-24

# Monte Carlo Power Filtering Plan

## Purpose

Add optional differentiable denoising filters to `witwin.channel.montecarlo` radio-map
results. The filters target Monte Carlo variance in the incoherent power maps
without changing the default solver output.

## Scope

Filtering is opt-in through `Config(...)` and remains disabled by default.

The first implementation filters only incoherent power-domain components:

- `incoherent["reflection"]`
- `incoherent["diffraction"]`

`los` is not filtered by default because it is evaluated directly at receiver
cell centers and is not a Monte Carlo estimator in the current solver. Coherent
complex field components are also out of scope because the Monte Carlo radio-map
metric is non-coherent `path_gain`.

## Public API

Add a small filtering configuration owned by `witwin.channel.montecarlo.config`:

```python
from witwin.channel.montecarlo import Config, FilterConfig

config = Config(
    filtering=FilterConfig(
        reflection={"method": "gaussian", "radius": 1, "sigma": 1.0, "blend": 0.5},
        diffraction={"method": "bilateral", "radius": 2, "sigma": 1.25, "range_sigma": 0.5},
    )
)
```

The first implementation should expose these public config types:

- `FilterConfig`
- `ComponentFilterConfig`

The contract is:

- `Config.filtering` defaults to `None`.
- Reflection and diffraction can be configured independently.
- A disabled component is passed through exactly.
- Invalid methods, negative radii, non-positive sigmas, and blends outside
  `[0, 1]` fail during `Config` construction.
- Package-root exports include the public filter config types.

## Filter Methods

### Gaussian

Use a fixed-radius 2D separable Gaussian over the receiver grid. This is the
recommended first-use filter because it is linear, stable, and fully
differentiable with respect to the input power map.

At boundaries, use normalized in-bounds weights so constant maps remain
constant.

### Bilateral

Use a compact, differentiable, power-domain bilateral-style filter:

```text
weight(i, j) = spatial_weight(i, j) * exp(-((p_i - p_j)^2) / (2 * range_sigma^2))
out_i = sum_j weight(i, j) * p_j / sum_j weight(i, j)
```

This keeps the implementation close to computer-graphics denoising while
remaining differentiable. It is nonlinear and more parameter-sensitive than the
Gaussian filter, so it should not be the default recommendation.

### Blend

Every component filter supports a continuous blend:

```text
filtered = (1 - blend) * raw + blend * candidate
```

The default blend is `1.0`.

## Solver Placement

Filtering should run after reflection and diffraction accumulation and before
`_finalize_component_totals(...)` computes `raw_total`, shadow-boundary
correction, `total`, `path_gain`, `rss`, and `sinr`.

This placement gives one shared path for:

- `Basic` non-AD
- `Basic` AD result assembly
- `BDPT` non-AD
- `BDPT` AD result assembly

The current `BasicIntegratorAD.result_from_components(...)` already rebuilds
the same `weighted_diagnostics` structure from component powers, so a shared
finalization helper can keep AD and non-AD behavior aligned.

## Differentiability Contract

The filters must be implemented with DrJit-native scalar operations:

- `dr.gather`
- arithmetic
- `dr.exp`
- `dr.select`
- normalized weighted sums

Do not introduce NumPy, Torch, DLPack, or CPU conversion in the filtering path.

The filtered result is differentiable with respect to the component power maps.
Through the existing custom-op AD paths, upstream gradients from
`result.path_gain` should flow back into the `reflection` and `diffraction`
component outputs after the filter Jacobian is applied by DrJit.

The filter is not a new unbiased Monte Carlo estimator. It is an opt-in
post-accumulation denoising transform. Metadata must report that distinction.

## Metadata

Add a `metadata["monte_carlo"]["filtering"]` payload:

```python
{
    "enabled": True,
    "domain": "incoherent_power",
    "components": {
        "reflection": {"method": "gaussian", "radius": 1, "sigma": 1.0, "blend": 0.5},
        "diffraction": {"method": "bilateral", "radius": 2, "sigma": 1.25, "range_sigma": 0.5, "blend": 1.0},
    },
    "contract": "differentiable_post_accumulation_power_denoising",
}
```

When disabled, metadata should still record `"enabled": False` so downstream
checks can distinguish default pass-through behavior from older solver payloads.

## Tests

Add focused non-GPU tests for the filter operators:

- disabled filtering returns the original DrJit array object or identical values
- Gaussian preserves a constant map
- Gaussian spreads an impulse with normalized positive weights
- component-specific filtering changes only the selected component
- bilateral preserves sharper contrast better than Gaussian on a simple step
- `dr.forward_to` through a filtered map returns finite non-zero derivatives

Add integration checks for the Monte Carlo package contract:

- default `Config()` leaves metadata filtering disabled
- opt-in config records metadata and updates `path_gain == incoherent["total"]`
- AD result assembly keeps gradients finite when filtering is enabled

GPU-heavy solver tests should stay targeted because existing Monte Carlo tests
already cover the full solve path.

## Out Of Scope

- Coherent complex field filtering.
- CPU or Torch result-space postprocessing.
- New native CUDA kernels for filtering in the first implementation.
- Heuristic smoothing inside diffraction or intersection derivative formulas.
- Changing the default `Config()` output.
