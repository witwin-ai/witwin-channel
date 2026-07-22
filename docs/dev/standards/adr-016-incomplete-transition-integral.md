# ADR-016 (DRAFT): Incomplete pole-transition integral for finite-edge boundaries

(Renumbered from the initial ADR-014 draft: the concurrent `wt/scattering-ad`
branch already carries adr-014/adr-015 for scattering AD.)

- **Status:** ORACLE STAGE COMPLETE - device implementation NOT accepted
  (2026-07-18). The oracle protocol reached its own stop condition: the
  exact object regresses field-level acceptance. Findings and the recorded
  follow-up are in "Oracle-stage results and decision" below. The oracle
  library under `artifacts/p4-oracle/` is the retained foundation.
- **Date:** 2026-07-18
- **Kind:** Replaces the empirical single-variable corner-mend stand-ins with
  the exact truncated (incomplete) boundary-transition object.
- **Related:** ADR-012 (stationary machinery), ADR-013 (D->D; its leg-2
  coefficient call site is reserved for this swap),
  `docs/dev/audit/utd-continuity-fix-design.md` F5c/d/e/f,
  RayD `shared/include/rayd/shared/utd/utd_math.h` (`mend_beta_term_value`
  :613-654, `C_BLEND` :592).

## Problem

Two empirical elements remain in the finite-edge continuity machinery, both
self-documented as stand-ins for the same exact object:

1. `mend_beta_term_value`'s odd-part treatment: locality window
   `w = exp(-(delta/deltaW)^2)` (with a hard `w <= 1e-3` cutoff), and the
   G2 blend `B = wb + (1-wb) T_mono` with `wb = exp(-(delta/deltaB)^2)`,
   `deltaB = C_BLEND sqrt(2 pi / kL)`, `C_BLEND = 0.35` calibrated against
   the single-cube Maxwell reference. The header comment itself records the
   "complex-pole truncated transition integral (generalized Fresnel)" as the
   exact replacement.
2. The measured P2 residuals transferred from ADR-013: receivers at
   lambda/20 from PEC surfaces (second-leg kL ~ 0.46) where cascaded per-leg
   UTD is outside its asymptotic regime; and the single-cube corner-zone ISB
   toggle median (1.09 dB vs 0.29 baseline) plus shadow-gap p10/p90 spread
   (+/-4 dB) recorded in plan 09 section 2.

## Canonical object

The finite-edge, near-boundary term is the truncated stationary-phase
integral with a nearby simple pole. In the Fresnel-normalized edge variable
`t` (stationary point at 0, segment ends at `w1 < 0 < w2`) and complex pole
parameter `p` (|p| ~ boundary distance in transition units; side of the GO
boundary fixes the sign convention; loss shifts it off the real axis):

```
G(p; w1, w2) = int_{w1}^{w2} e^{-j t^2} / (t - p) dt
```

Rotating the contour by `e^{-j pi/4}` maps the infinite case onto the
Hilbert transform of a Gaussian, i.e. the Faddeeva function:
`G(p; -inf, inf) = -j pi w(e^{j pi/4} p)` (upper-half-plane convention);
this limit must reduce EXACTLY to the existing Kouyoumjian-Pathak
`F(kL a)` assembly - that is oracle check O1 below. The truncated case is
the genuinely two-variable object; the odd-part weight and the even-part
truncation both fall out of `G(p; w1, w2)` evaluated at the term's pole and
at the segment bounds, replacing `w`, `wb`, `B`, and `C_BLEND` entirely.
No window, no calibration constant.

## Oracle protocol (fp64 numpy FIRST; no device code before O1-O4 pass)

- O1 (infinite-limit identity): `G(p; -inf, inf)` reproduces the KP
  transition assembly for every beta-term over a dense (beta, n, kL) grid to
  relerr < 1e-10 (fp64).
- O2 (brute force): direct adaptive quadrature of the finite-edge
  equivalent-edge-current line integral with TRUE per-sample off-cone
  angular arguments, over a geometry matrix that includes: interior points,
  corner rays (stationary point near an edge end), deep shadow, the P2
  lambda/20 near-wall cells (exact three_cube_320 geometries of ADR-013
  G-A cells 1/4), and overlapping-transition D->D pairs. The closed-form /
  quadrature evaluation of G must match brute force to < 1 percent in
  amplitude and < 1 degree in phase over the matrix, and must degrade
  gracefully (documented) outside it.
- O3 (dose response): re-weight the recorded P1/P2 path tables offline with
  the oracle weights and re-score the ADR-013 transferred gates (G-A cells,
  G-D, G-E) plus the single-cube corner-zone/shadow-gap metrics BEFORE any
  CUDA work; the fix must move those metrics, or the ADR stops here and
  records why.
- O4 (evaluation scheme): select the device scheme (region-split closed
  forms via Faddeeva + endpoint asymptotics, vs fixed-node pole-subtracted
  quadrature) by fp64-vs-float32 error mapping; the scheme must be smooth
  (C1) across region seams, AD-differentiable (Dual-compatible, no
  data-dependent branching that breaks lockstep), and reproduce O2 within
  float32 headroom.

Device candidates for the Faddeeva core: Weideman rational approximation
(N ~ 16; single rational evaluation, Dual-friendly) with the standard
reflection identities. The two-variable regime near `|w_i - p|` small is
the part O4 must settle empirically - do not guess.

## Acceptance gates (frozen now)

1. ADR-013 transferred gates: G-A cells (0.0531,0.4531) and (0.284,0.241)
   within +/-6 dB of FDTD truth; coupled-active ISB p95 excess < +2.0 dB;
   three-cube NMSE < 0.0899; ISB p95 excess < +1.0 dB.
2. Single-cube: corner-zone ISB toggle median <= 0.5 dB (from 1.09; pre-fix
   baseline 0.29); shadow-gap p10/p90 spread halved; all recorded
   single-cube metrics not regressed (NMSE <= 0.0358, coherence >= 0.885,
   ISB p95 excess <= -0.1 dB).
3. MC / non-stationary call sites remain BIT-IDENTICAL (the mend fast path
   contract); coupled-off deterministic solves unchanged within the
   documented run-to-run band; AD lockstep suites green; drjit PTX regen
   recorded as a release chore with the RayD lock update.
4. Runtime: three-cube warm <= 4 s (from 2.6 s; the new special function
   must stay O(10) flops per boundary-active term).

## Oracle-stage results and decision (2026-07-18)

All artifacts under `artifacts/p4-oracle/` (oracle_lib.py, o1/o2/o3/o3b
reports and matrices). Executed as O1 || O2 -> O3 -> O3b (scoped rerun).

**O1 - PASSED.** The fp64 port reproduces the header operation-for-operation
(Boersma literals, C roundf N-selection, MC fast-path bit-identity). The
pole mapping is pinned analytically: for each beta-term,
`p = j sqrt(kL a(beta))`, `C = cot sqrt(x) e^{-j pi/4}/sqrt(pi)`, and
`C G(p; -inf, inf) == cot F(kL a)` exactly (Faddeeva bridge; verified to
1.0e-14 over 4212 (beta, n, kL) points including exactly-on-boundary, by an
independent contour quadrature). Mirror-argument N reselection falls out of
re-running the term at betaM. The header's Boersma F itself is a 1.4e-8
rational approximation of the true transition function; the identity is
stated against the true function, as it must be.

**O2 - PASSED, with one mapping correction.** The finite-edge bounds scale
is `sqrt(k kappa / 2)` (e^{-j t^2} kernel), not the Fresnel-C/S scale
`sqrt(k kappa / pi)` O1 lifted from `finite_wedge_truncation_factor_bounds`
- a factor sqrt(pi/2) verified across every matrix row. With the corrected
scale, G matches the brute-force equivalent-edge-current reference to
0.95%/0.23 deg (corner rays), 1.17%/0.75 deg (deep shadow), 0.13%/0.19 deg
(D->D pair), 3.7%/1.0 deg at the lambda/20 cells (kL 0.43; documented
graceful degradation). The production stand-in errs by up to 5.0-6.5 dB /
11-19 deg in the corner-truncation regime, with a systematic ~0.6-0.7 dB
T_mono undershoot even for interior stationary points, and its error signs
match the recorded ADR-013 G-A failures (cid-7 cell over-bright, cid-4 cell
over-attenuated).

**O3 - full replacement REJECTED by measurement.** Reweighting every
reconstructable diffraction term with the exact G on the recorded solves:
3 of 4 G-A cells clear (de-cancellation; (0.284,0.241) -39.4 -> 0.0 dB),
but every calibrated aggregate regresses - decisively, the artifact-free
single-cube full reweight moves corner-zone ISB toggle median 1.09 ->
5.09 dB and NMSE 0.0358 -> 0.0639. Mechanism: `C_BLEND = 0.35` was
calibrated against the Maxwell reference and therefore ABSORBS higher-order
physics the single-edge theory omits (vertex/corner waves launched at edge
ends, inter-edge vertex coupling); the exact-but-incomplete single-edge
object is ~+6.8 dB brighter in the corner-truncation regime, and removing
the calibrated compensation de-smooths the very boundaries it was tuned on.

**O3b - scoped (coupled-only cid 3/4/7) replacement: necessary-safe, NOT
sufficient.** Leaving order-1 cid-2 on the calibrated stand-in removes the
aggregate catastrophe (ISB excess 1.86 not 7.48; RSB stays PASS at 1.74)
and heals the coupled null ((0.284,0.241) -39.4 -> -1.5 dB), but G-D still
regresses (2.48 -> 4.35), and the remaining hard gates - G-A cell
(0.0531,0.4531) and the single-cube corner zone - are cid-2-dominated by
construction. Coupled-family label coverage 61.6% (cid-4 31%) is a mild
caveat on the exact projections, not on the verdict.

**DECISION:** do not implement the naked replacement (full or scoped) in
the device header. The calibrated single-variable stand-in stays, now with
a measured characterization of what it compensates: the difference between
the exact single-edge incomplete integral and the true field, i.e. the
missing vertex-wave / inter-edge physics (~6.8 dB effective extra
attenuation in the corner-truncation regime). The principled completion is
`exact G + vertex diffraction terms + edge-interaction coupling` as one
numerical unit - a research-scale follow-up with its own plan and ADR,
building on the retained oracle library (validated port, G evaluators with
the corrected bounds scale, brute-force EEC reference, dose-response
harness). Do not partially land any of it without that unit passing this
ADR's frozen gates.

## Open questions for the oracle stage

- Exact mapping from each beta-term's (delta, kL, w1, w2) to (p, w1, w2)
  including the N+- branch reselection at mirrored arguments.
- Whether the even-part monotone truncation T_mono should also be upgraded
  to the complex incomplete Fresnel (phase-correct corner wave) or kept
  real-monotone (G1 lesson: ripple-free); decide from O2/O3 evidence, not
  taste.
- Slope-diffraction terms: first/second derivative outputs currently take
  truncEven only; verify from O2 whether the incomplete object's derivative
  terms matter at the acceptance cells.
