# Monte Carlo Radiomap Package Overview

Status: Active
Category: Standard
Last reviewed: 2026-04-22

## Purpose

This document is the package-level architecture overview for
`witwin.channel.montecarlo`.

Use it when you need the current ownership map for the standalone Monte Carlo
radiomap solver rather than the refactor plan that led here.

## Public API

Import the standalone solver through the package root:

```python
from witwin.channel.montecarlo import Config, FilterConfig, IntegratorOptions, RadioMapResult, Tuning, solve
```

```python
config = Config(integrator_options=IntegratorOptions(integrator="basic"))
filtered_config = Config(
    filtering=FilterConfig(
        reflection={"method": "gaussian", "radius": 1, "sigma": 1.0},
        diffraction={"method": "bilateral", "radius": 2, "sigma": 1.25},
    )
)
```

The public contract is intentionally small:

- `Config`: user-facing Monte Carlo solver settings
- `Tuning`: advanced runtime controls such as shadow-boundary backend and diffraction execution controls
- `IntegratorOptions`: integrator selection, seed, AD intent, roulette, and `samples_per_tx`
- `RadioMapResult`: traced radiomap payload and sampling helpers
- `FilterConfig` / `ComponentFilterConfig`: opt-in differentiable
  incoherent-power denoising for reflection and diffraction maps
- `solve(...)`: public solve entrypoint
- `native_extension_available`: bundled native-extension availability check

The package root is re-export only. Solver logic starts in `solver.py`.

## Solve Flow

The intended top-down call graph is:

1. `solve(...)`
2. selected integrator in `integrators/`
3. per-cell LoS phase in `path/los.py`
4. specular reflection phase in `path/reflection.py`
5. optional diffraction phase and `RadioMapResult` assembly

This is the main path a new reader should follow before drilling into AD or
native-kernel details.

## Ownership Map

Production ownership is split by domain:

- `solver.py`: public API edge, scene coercion, config resolution, integrator dispatch
- `config.py`: public config plus filtering, resolved trace, and solver-control policy
- `grid.py`: internal grid construction and cell-center geometry
- `filtering.py`: Dr.Jit-native post-accumulation power-domain filtering
- shared `witwin.channel.core.results.RadioMapResult`: public result payload plus result-boundary tensor conversion helpers
- `integrators/basic.py`: TX-emitted non-LoS transport owner plus result assembly
- `integrators/bdpt.py`: BDPT owner for cell-center LoS, specular reflection reuse, and depth-limited wedge diffraction MIS
- `path/los.py`: cell-center LoS visibility and power evaluation
- `path/reflection.py`: specular reflection tracing and wedge hit discovery support
- `path/diffraction.py`: sampled Keller-cone diffraction runtime used by the basic integrator
- `integrators/basic_ad.py`, `reflection_ad.py`, `ad_support.py`: AD-specific replay/tape logic for the basic integrator
- `integrators/metadata.py`: diagnostic and runtime metadata assembly
- `utils/`: small generic Dr.Jit math/helpers only

## Internal Runtime Objects

The package uses a small set of typed payloads at cross-module boundaries:

- `BatchPlan`
- `TraceTiming`
- `PathCounts`
- `ReflectionPhaseResult`
- `DiffractionPhaseResult`
- `MetadataInput`
- `ADContext`

These types replace the older repeated dict-shaped payloads that obscured the
call graph.

## Integrator Model

The package now treats path-family composition and sampling policy as
integrator-owned logic.

- `Config(integrator_options=IntegratorOptions(integrator="basic"))` selects the current TX-emitted non-LoS Monte Carlo path
- `Config(integrator_options=IntegratorOptions(integrator="bdpt"))` selects the staged bidirectional path-tracing path
- `solve(...)` resolves the integrator and delegates the solve directly to it
- no package-level `trace.py` compatibility shim remains

This keeps LoS, reflection, diffraction, and future transport-family sampling
under one explicit owner instead of routing them through a generic tracer layer.

LoS is no longer estimated by TX-emitted plane hits. Both integrators evaluate
LoS once per receiver cell center with a direct Tx-to-cell visibility test.
The basic integrator keeps the existing sampled specular reflection and
first-order Keller-cone diffraction estimator. The BDPT integrator keeps
reflection constrained to forward-generated specular paths, and runs
depth-limited wedge diffraction up to `max_diffraction_order=3` with balance MIS
across direct wedge-to-cell connections and Keller-cone plane-hit samples. The
diffraction budget is split by order and then by endpoint strategy, including a
single-bounce specular suffix-connection strategy. Direct multi-diffraction
chains sample wedge vertices as light-subpath vertices, reflection-prefix
diffraction states are seeded only from actually traced specular reflection
hits, and suffix reflections use a one-surface image-method connection from the
last diffraction vertex to the receiver cell. BDPT does not construct arbitrary
reflection paths. `RadioMapResult.metadata` reports the mixed receiver measures explicitly:
direct cell connections are accumulated on the discrete receiver-cell measure,
while Keller-cone samples remain continuous plane-hit samples. Inserted
reflection families are reported separately from the active diffraction-order
breakdown. Reflection-prefix diffraction states are written into deterministic
ray/depth slots and carry the forward ray solid-angle source weight before they
enter the BDPT wedge sampler. UTD diffraction applies the resolved material
gain to the direct edge term as well as the face-reflection terms, so
`reflection_coef` and `diffraction_material.gain` attenuate every diffraction
event instead of only the face-reflected subterms. BDPT diffraction sampling
defaults to `IntegratorOptions.bdpt_diffraction_sampling="sobol"` so edge-chain state
selection, edge-position sampling, receiver/suffix choices, and Keller-cone
angles come from one low-discrepancy Sobol vector per sample; use `"hash"` only
for controlled comparisons against the older hash-uniform sequence.
`Config(tuning=Tuning(shadow_boundary_mode="utd_power_smoothing"))` applies a power-domain
UTD transition completion from geometry weights and complex `F(kLa)` responses.
It does not use local-neighbor hard-boundary fallback or force an `F(0)` branch;
finite-edge endpoint support uses a Fresnel-scaled smoothstep taper around the
explicit line bounds.
`Config.filtering` defaults to disabled, but can opt into Dr.Jit-native
Gaussian or bilateral-style filtering for the incoherent reflection and
diffraction power maps before totals and `path_gain` are finalized. Metadata
reports the filtering contract under `metadata["monte_carlo"]["filtering"]`;
when diffraction filtering is enabled, the same filter is also applied to the
shadow-boundary diffraction transition-power auxiliaries so completion algebra
stays in the same filtered power domain. The transform is a differentiable
post-accumulation denoising step, not a new unbiased Monte Carlo estimator.
First-order BDPT diffraction additionally samples candidate edges from a mixed
edge-length, receiver projected-solid-angle, source-distance, and source-power
proposal. A positive edge-length baseline keeps full support, and the direct
and Keller estimators apply the per-sample edge PDF correction before
accumulation, so the proposal changes sample allocation without changing the UTD
field contribution formula.
For inspection runs, `Config(tuning=Tuning(enable_bdpt_reflection_coupled_diffraction=False))`
keeps ordinary reflection tracing enabled but removes reflection-coupled
diffraction families from the BDPT diffraction estimator. In that mode,
reflection-prefix states are not collected, suffix-reflection samples are
reallocated to direct/Keller diffraction strategies, and metadata reports
`R^n -> D` / `D -> R` families as disabled.

## Size-Budget Exceptions

Most package modules now stay within the architecture size targets. The
remaining oversized files are intentional and reviewed exceptions:

- `config.py`: keeps public config, resolved trace config, and solver-control
  policy together to avoid low-value module splits across one ownership domain
- `integrators/basic.py`: remains the current orchestration owner for
  the public solve path until a second real transport family exists
- `path/reflection.py`: keeps reflection batching and tape capture together
  because they share one active-ray loop
- `integrators/basic_ad.py`: keeps the `dr.CustomOp` capture boundary, sparse-coefficient
  cache wiring, and transport replay local to one AD owner
- `path/diffraction.py`: keeps the diffraction state layout and sampling logic
  together because they share one runtime memory contract

## Examples

Keep two example tiers:

- `examples/monte_carlo_radiomap_minimal.py`: onboarding-first public API example
- `examples/monte_carlo_radiomap_bdpt.ipynb`: BDPT diffraction MIS visualization example
- `examples/monte_carlo_radiomap_three_cubes.py`: advanced profiling and AD workflow

The minimal example is the canonical starting point for package usage.
