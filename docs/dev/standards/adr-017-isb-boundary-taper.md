# ADR-017: Switchable joint ISB boundary taper (visual-continuity heuristic)

- **Status:** Accepted for implementation (2026-07-18). Numerical change
  behind a DEFAULT-OFF switch; the off path is bit-identical.
- **Kind:** Declared heuristic (user-authorized deviation from the
  no-heuristics rule, scoped to ISB visual continuity). NOT new physics;
  the vertex/near-zone physics program (plan 10) is unaffected and remains
  the principled owner of the residuals.
- **Related:** plan 10 Appendix A (mechanism discriminations), the
  rsb-taper falsification (`artifacts/rsb-taper/report.json`), the ISB
  diagnosis + projection (`artifacts/isb-taper/report.json`). Numbering:
  014/015 belong to the concurrent scattering-AD branch, 016 to the
  transition-integral oracle verdict.

## Problem and measured mechanism

The visible ISB (LoS shadow-boundary) lines are NOT corner-truncated
compensation and NOT coupled-family coincidence (stage-1 census: 84.5% /
100% pure LoS-vs-D miscompensation; coupled-coincident 3.8% / 0%). At each
crossing the LoS component and the compensating order-1 diffraction BOTH
step within ~1 receiver cell, near-anti-phase (median 8 deg off), with
magnitude ratio ~1.04-1.11 - the compensation amplitude is right, but both
members transition HARD while the FDTD truth's penumbra is 15-55 cells wide
(w_F = sqrt(lambda d1 d2/(d1+d2)) ~ 0.11 m; FDTD 10-90 width / w_F median
0.61 three-cube, 1.39 single-cube). The ~24% residual of two misaligned
hard steps is the visible line.

## Decision

One switch, smoothing BOTH members of the compensation pair with CONGRUENT
windows (the single-sided taper is falsified twice: rsb-taper 24/24
variants; the LoS-only ISB control regresses ISB p95 excess to +24/+27 dB):

- **LoS member (channel-native):** the hard occlusion gate becomes
  membership tau(c/w): c = signed clearance of the tx->rx segment past the
  occluding cube silhouette edge (positive lit), w = width_scale * w_F of
  the grazed edge; tau = C1 smoothstep through 1/2 at c = 0. LoS rows
  survive inside the taper margin.
- **D member (RayD shared header):** `mend_beta_term_value` gains a
  defaulted congruent-window override so the odd (GO-step-carrying) part of
  INCIDENT-boundary terms (bStar = +-pi) of the shadow-owning edge spreads
  over delta_beta = width_scale * w_F / s2 instead of transitioning hard.
  Reflection-boundary terms are untouched (RSB is out of scope by user
  decision). Default parameter value reproduces current behavior
  bit-identically for every existing caller (ADR-012
  `stationaryExternalIncident` precedent).
- **Config:** `isb_boundary_taper: bool = False`,
  `isb_boundary_taper_width: float = 0.5` on the deterministic and path
  configs; capabilities + public-api snapshot updated intentionally.
  OFF = bit-identical (hard gate, unchanged window); this is enforced by a
  bitwise regression test.

## Projection evidence (offline, fp64; `artifacts/isb-taper/`)

Analytic LoS reconstruction bit-exact vs recorded rows (corr 1.000000).
Best variant joint smoothstep width_scale 0.5: three-cube ISB p95 excess
+1.083 -> +0.758, RSB +0.431 -> +0.441 (flat), NMSE/coherence unchanged;
single-cube ISB -0.729 -> -0.820, RSB improves, NMSE unchanged. Rendered
maps: the green ISB seam lines vanish in both scenes with no new
artifacts; LoS-only control catastrophically fails (mechanism proof).
Deep-multipath fringing seams are intentionally untouched (plan-10 scope).

## Acceptance gates (native implementation)

1. OFF: bitwise-identical solves (both scenes, deterministic + path), all
   existing suites green, binding/governance artifacts consistent.
2. ON (width 0.5): native reproduction of the projection numbers within
   0.05 dB (three-cube ISB p95 excess <= 0.81, RSB <= 0.48, NMSE 0.0922
   +/- 0.0002; single-cube ISB <= -0.78, NMSE 0.0358 +/- 0.0002); rendered
   maps show the ISB lines removed; no new seam class in the census.
3. AD: taper factors are C1 functions of endpoint geometry; JVP/VJP
   lockstep via the existing companions; OFF-path AD bit-identical.
4. MC / non-stationary callers bit-identical (header default).
5. RayD lock refreeze + drjit committed-PTX regeneration recorded as the
   release chore for the header change.
6. Runtime: warm three-cube within +10% of 2.6 s.

## Risks

- The taper widens the LoS transition using w_F of the FIRST grazed
  silhouette edge; multi-occluder penumbra overlap is approximated by the
  nearest-edge clearance (documented; census shows no such overlap in the
  benchmark scenes).
- The D-side window rides the odd-part machinery; its interaction with the
  C_BLEND deep-shadow blend is additive by construction (different delta
  regimes) but is pinned by a direct contract test.
- Scope creep guard: the switch must never default ON, and no other
  boundary family (RSB, coupled, corner rays) may reuse it without a new
  ADR and its own projection evidence.
