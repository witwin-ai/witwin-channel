# ADR-0xx: Result-semantics keys in every solver's metadata

- **Status:** Accepted (owner decision for the 0.6.0 radiomap estimators).
- **Date:** 2026-11-10.
- **Decision ID:** `result-semantics-metadata`.
- **Scope:** `metadata` of `path`, `deterministic`, `montecarlo.basic` and
  `montecarlo.bdpt` results. Additive; no numerical change.

## Decision

Every solver's result metadata carries four keys, built by one helper,
`witwin.channel.capabilities.estimator_metadata`, which validates the vocabulary:

| Key | Values | Meaning |
| --- | --- | --- |
| `estimator` | `enumerated`, `sampled` | Path enumeration or a Monte Carlo estimate. |
| `quantity` | `point_sample`, `cell_average` | Value at the receiver position, or averaged over its cell / capture region. |
| `combine` | `coherent`, `incoherent`, `none` | How rows reach a receiver value; `none` when rows are returned uncombined. |
| `complete` | `True`, `False`, `None` | Whether every path within the requested bounds was found; `None` when the solve does not report it. |

Values in 0.6.0:

| Solver | estimator | quantity | combine | complete |
| --- | --- | --- | --- | --- |
| `path` | enumerated | point_sample | none | from discovery |
| `deterministic` | enumerated | point_sample | coherent / incoherent (`Config.coherent`) | from discovery |
| `montecarlo.basic` | sampled | cell_average | incoherent | `False` |
| `montecarlo.bdpt` | sampled | point_sample | coherent / incoherent (`Config.coherent`) | `False` |

For `path` and `deterministic`, `complete` is the discovery stage's per-depth
result (`metadata["discovery"]["complete"]`: true only when every reflection depth
came from a complete source, so `discovery="rays"` gives `False`); the solver
publishes it once, through the helper, next to the `discovery` block. The
propagation consumer has no result metadata dictionary and does not publish these
keys; adding them is a consumer contract change. MC Basic is labelled `cell_average` because its reflection, diffraction and
scattering estimators deposit over the cell (or a point receiver's capture
sphere); its LoS and transmission components are evaluated at the cell centre
(the MC consistency gate checks them against the deterministic cell centres to
3.3e-7), which the per-solver label does not distinguish. BDPT connects to the
receiver position and is `point_sample`.

## Evidence

`tests/test_result_semantics_metadata.py`: the helper's vocabulary checks, and
the four keys on a solve of each solver (six configurations).
