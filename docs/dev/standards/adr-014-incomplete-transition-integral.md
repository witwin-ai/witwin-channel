# ADR-014 (DRAFT): Incomplete pole-transition integral for finite-edge boundaries

- **Status:** Draft - oracle stage in progress (plan-09 P4). Numerical-kernel
  change in the SHARED RayD UTD header; requires its own acceptance evidence
  and the drjit committed-PTX regeneration chore before release.
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
