# ADR-0xx: Result-semantics keys in every solver's metadata

- **Status:** Accepted (owner decision for the 0.6.0 radiomap estimators).
- **Date:** 2026-11-10.
- **Decision ID:** `result-semantics-metadata`.
- **Scope:** `metadata` of `path`, `radiomap.Coherent`, `radiomap.Stochastic` and
  `experimental.bdpt` results (the 0.5 `deterministic`, `montecarlo.basic` and
  `montecarlo.bdpt`). Additive; no numerical change.
- **Amended:** 2026-11-11, with the 0.6.0 rename: the `estimator` values became
  `coherent` / `stochastic` (were `enumerated` / `sampled`) and the `combine` values
  `field` / `power` (were `coherent` / `incoherent`), matching `radiomap.Coherent(combine=...)`.

## Decision

Every solver's result metadata carries four keys, built by one helper,
`witwin.channel.capabilities.estimator_metadata`, which validates the vocabulary:

| Key | Values | Meaning |
| --- | --- | --- |
| `estimator` | `coherent`, `stochastic` | Path enumeration or a Monte Carlo estimate. |
| `quantity` | `{component: point_sample \| cell_average}` | Per requested component: value at the receiver position, or averaged over its cell / capture region. |
| `combine` | `field`, `power`, `none` | How rows reach a receiver value: complex field sum then squared, sum of row powers, or `none` when rows are returned uncombined. |
| `complete` | `True`, `False`, `None` | Whether every path within the requested bounds was found; `None` when the solve does not report it. |

Values in 0.6.0:

| Solver | estimator | quantity | combine | complete |
| --- | --- | --- | --- | --- |
| `path` | coherent | point_sample (every component) | none | from discovery |
| `radiomap.Coherent` | coherent | point_sample (every component) | field / power (`Coherent.combine`) | from discovery |
| `radiomap.Stochastic` | stochastic | LoS, transmission: point_sample; reflection, diffraction, scattering: cell_average | power | `False` |
| `experimental.bdpt` | stochastic | point_sample (every component) | field / power (`Config.coherent`) | `False` |

For `path` and `radiomap.Coherent`, `complete` is the discovery stage's per-depth
result (`metadata["discovery"]["complete"]`: true only when every reflection depth
came from a complete source, so `discovery="rays"` gives `False`); the solver
publishes it once, through the helper, next to the `discovery` block. The
propagation consumer has no result metadata dictionary and does not publish these
keys; adding them is a consumer contract change. Per the owner, `quantity` is reported per component, because MC Basic mixes
estimators: LoS and straight transmission are evaluated at the receiver position
(the MC consistency gate checks them against the deterministic cell centres to
3.3e-7), while reflection, diffraction and scattering deposit over the cell or a
point receiver's capture sphere. BDPT connects to the
receiver position and is `point_sample`.

## Evidence

`tests/test_result_semantics_metadata.py`: the helper's vocabulary checks, and
the four keys on a solve of each solver (six configurations).
