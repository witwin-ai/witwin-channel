# ADR-0xx: Consistency gate between the enumerated and sampled estimators

- **Status:** Accepted (owner decision D5, analysis report 10.13).
- **Date:** 2026-11-10.
- **Decision ID:** `d5-estimator-consistency-gate`.
- **Scope:** `tests/accuracy/test_enumerated_sampled_consistency.py`: what
  incoherent Deterministic (`Enumerated(coherent=False)`) and MC Basic
  (`Sampled`) must agree on, and the thresholds. Test-only.

## Thresholds

Measured on the 0.6.0 gate with 8 fixed seeds of 2^20 samples per case on a
4 x 4 grid of 2 m cells; the reference is the deterministic incoherent power
averaged over 8 x 8 sub-samples of each cell (MC Basic reflection deposits a
cell average).

| Case | Rule | Measured |
| --- | --- | --- |
| PEC wall reflection, shallow (Tx 1 m above the grid) | `abs(MC mean - det) <= 5 SE` per cell | max 1.50 SE, ratio 0.985-1.016 |
| Dielectric wall reflection, shallow | same | max 1.55 SE, ratio 0.985-1.016 |
| LoS | MC equals the deterministic cell centre, relative 1e-5 | 1.9e-7 |
| Straight transmission | same | 3.3e-7 |

5 standard errors with 7 degrees of freedom is a two-sided per-cell tail of
about 0.16 %; the seeds are fixed, so the gate is reproducible.

## Known gaps (named strict xfails, measured)

- **Steep PEC wall reflection** (Tx 5.5 m above the grid): MC/deterministic
  1.060-1.119, up to 133 SE. MC Basic reflection is scalar; Deterministic projects
  the vector field onto the z dipoles, which the steep geometry makes visible.
- **Dielectric ground reflection near the Brewster angle:** 1.186-1.333, up to
  184 SE, the same scalar-versus-vector Fresnel difference.

## Not compared: diffraction

Diffraction is not expected to agree between the two estimators and is not part
of the gate (no test and no xfail). MC Basic samples Keller cones from the
reflection launch's wedge hits and deposits their power, while Deterministic
enumerates first-order UTD rows per receiver; they are different estimators of
different quantities near shadow boundaries. The deterministic diffraction
target is the full-wave reference (Plan 16 validation), owned elsewhere.
