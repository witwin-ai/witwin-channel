# ADR-0xx: MC Basic scattering sample count and native receiver pairs

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `mc-scattering-native-pairs`.
- **Scope:** the Kirchhoff area estimator of `witwin.channel.montecarlo.basic`
  (`interactions.scattering.scattering_map_matrix`). Estimator unchanged; reduction order
  changes.

## Context

P-14: MC Basic scattering connected every area sample to every receiver with chunked Torch
code (1.07e9 pairs: 1.38 s, ~15k kernel events, ~800 syncs) and shared `samples` with the
reflection launch, so raising the reflection sample count made scattering quadratically slower.

## Decision

- `Config.scattering_samples` sets the area-sample count (default: `samples`).
- The per-sample stage (area sampling, frames, incident weight, transmitter visibility) is
  unchanged. The all-pairs stage is native: `mc_scattering_pair_legs` writes the visibility leg
  of every (receiver, sample) pair, one RayD segment visibility launch tests them, and
  `mc_scattering_pair_sum` evaluates the Kirchhoff table through RayD's
  `scattering_table.cuh` and sums `0.5 (f_te + f_tm) cos_i cos_o / (r1^2 r2^2)` per receiver in
  a fixed order (one block per receiver, strided lanes, fixed tree). Pairs are chunked to 2^22
  per launch. The estimator, its frames and its visibility rule are the same as before.
- AD: the sum is linear in the table values. Its VJP spreads the receiver cotangent over the
  16 interpolation corners of the flat (live, per-material) tables; its JVP is the same sum over
  tangent tables. Frequency and transmitter-power derivatives flow through the scalar prefactor.
- Statistics (`tx_visible_samples`, `deposited_rows`) are read once per solve.

## Evidence

`tests/montecarlo/basic/test_basic_scattering.py` (quadrature reference, point = cell centre,
regression guards) and `tests/ad/test_mc_basic_scattering_ad.py`,
`tests/ad/test_solver_scattering_ad.py` (table-value FD, JVP/VJP duality, solver VJP to the
table leaf) pass unchanged.

## Consequences

Scattering values change in the last bits (fixed-order per-receiver sums instead of Torch
reductions). The BDPT NEE builder is untouched.
