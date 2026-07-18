# Plan 10 — Vertex-diffraction research charter

Date: 2026-07-18. Kind: RESEARCH CHARTER (not an ADR; no production change
is authorized by this document). Successor to the ADR-016 oracle verdict;
owner of every residual that verdict assigned to "missing vertex-wave /
inter-edge physics".

Numbering note: any ADR spawned by this charter must skip **ADR-014 and
ADR-015** (owned by the concurrent `wt/scattering-ad` branch, unmerged at
charter time) and ADR-016 (the oracle verdict). Next free number: ADR-017.

---

## 1. Frozen problem statement (necessity record)

The finite-edge continuity machinery's calibrated stand-in
(`C_BLEND = 0.35`, RayD `utd_math.h::mend_beta_term_value`) is an
*effective* model: it was calibrated against the single-cube Maxwell
reference and therefore silently absorbs physics that single-edge
diffraction theory omits. The P4 oracle campaign (ADR-016,
`artifacts/p4-oracle/`) made that quantitative:

- **Per-term evidence (O2, brute-force-validated):** the exact incomplete
  pole integral `G(p; w1, w2)` matches the fp64 equivalent-edge-current
  reference to 0.95%/0.23 deg (corner rays), 1.17%/0.75 deg (deep shadow),
  0.13%/0.19 deg (D->D pairs). Against that reference the production
  stand-in errs by **up to 5.0-6.5 dB and 11-19 deg** in the
  corner-truncation regime (stationary point near an edge end), with a
  systematic ~0.6-0.7 dB T_mono undershoot even for interior stationary
  points. (`artifacts/p4-oracle/o2_report.json`, `o2_matrix.npz`.)
- **Field-level evidence (O3/O3b):** replacing the stand-in by the exact
  single-edge object REGRESSES the calibrated gates - single-cube
  corner-zone ISB toggle median 1.09 -> 5.09 dB, NMSE 0.0358 -> 0.0639
  (artifact-free full reweight); even the coupled-only scoped swap moves
  G-D +2.48 -> +4.35 dB. The exact single-edge object is effectively
  **~+6.8 dB brighter** than the calibrated stand-in in the
  corner-truncation regime. (`o3_dose_response.json`,
  `o3b_scoped_report.json`.)

Conclusion frozen here: the calibration gap IS the vertex physics. What the
stand-in compensates is, by construction of the evidence, the field of the
waves launched where finite edges terminate - corner (vertex) waves and
their coupling into the adjacent edges of the same vertex. Any principled
replacement must therefore ship `exact G + vertex coefficient (+ adjacent-
edge coupling)` as ONE numerical unit; partial landings are forbidden
(measured to regress).

Residuals this charter owns (current recorded values):

| Residual | Current | Source |
| --- | --- | --- |
| three_cube_320 cell (0.0531, 0.4531) gap vs FDTD | -21.3 dB | ADR-013 G-A |
| three_cube_320 cell (0.284, 0.241) gap vs FDTD | -39.4 dB (coupled-scoped G alone reaches -1.5) | ADR-013 G-A / O3b |
| coupled-active ISB p95 excess | +2.48 dB | ADR-013 G-D |
| three-cube NMSE / ISB p95 excess | 0.0922 / +1.08 dB | ADR-013 G-E |
| single-cube corner-zone ISB toggle median | 1.09 dB (pre-mend baseline 0.29) | plan 09 §2 |
| single-cube shadow-gap p10/p90 spread | 8.67 dB | plan 09 §2 / O3b |

## 2. Primary mandate: an original complete closed-form coefficient

**This is the charter's central, strict requirement.** The published
vertex-diffraction results (Kouyoumjian-Pathak-era corner diffraction,
Hill-Pathak, Michaeli endpoint EEC endpoint waves, Albani /
Capolino-Maci vertex coefficients, quarter-plane spectral solutions) are
known to be non-uniform, geometry-restricted (quarter-plane / half-plane
sectors, not the PEC cube vertex where THREE mutually orthogonal 270-deg
wedges meet), or asymptotically incomplete in exactly the regime measured
here (kL ~ 0.4-1, receivers at lambda/20). **The main thread (Fable) must
therefore attempt an original derivation - beyond the existing human
literature - of the complete canonical closed-form vertex coefficient**,
targeted as follows:

- Canonical object: the DOUBLE-incomplete two-pole extension of ADR-016's
  `G(p; w1, w2)` - the corner ray sits where two Fresnel truncations and
  up to two GO-boundary poles interact; the conjecture to prove or refute
  is that the cube-vertex wave admits a closed form in iterated Faddeeva /
  generalized-Fresnel functions of the SAME family already validated in
  `oracle_lib.py`, with the three-edge coupling entering through exact
  endpoint matching (each edge's incomplete integral endpoint term must
  cancel against the vertex wave and the adjacent edges' endpoint terms so
  that the TOTAL is continuous through every corner ray by construction,
  not by calibration).
- Uniformity requirement: finite everywhere (on corner rays, on ISB/RSB
  crossings through the vertex, at grazing), reducing exactly to (a) the
  infinite-edge KP assembly when both bounds recede (O1 identity), (b) the
  single-truncation G when only one end is near, (c) the known quarter-
  plane spectral solution in that limiting geometry.
- The literature is a set of verification anchors and limiting cases, NOT
  the ceiling. If the derivation succeeds it is a research-level
  contribution (publishable); the charter explicitly allocates main-thread
  derivation time to it and forbids delegating the load-bearing
  mathematics to subagents (working-protocol lesson, plan 09 §6).

## 3. Candidate routes and their discriminating experiments

- **Route A - endpoint-integral derivation (PRIMARY, per §2).** Derive the
  vertex wave as the exact endpoint/corner contribution of the truncated
  edge integrals plus the inter-edge matching terms.
  *Discriminator D-A:* the derived closed form must match the Route-C
  reference on isolated-vertex micro-scenes to < 1% amplitude / < 1 deg
  phase over a (kL, incidence, observation) matrix INCLUDING kL in
  [0.3, 3] and lambda/20 observation radii - the regime where published
  asymptotic corner coefficients are known to fail. Falsifiable: if no
  closed form emerges, the derivation stage must produce the precise
  mathematical obstruction (which integral does not reduce) as the
  recorded outcome.
- **Route B - calibrated effective coefficient (DIAGNOSTIC ONLY).** Fit an
  effective vertex amplitude against Maxwell references. This route
  CONFLICTS with the standing no-heuristics constraint and is therefore
  admissible ONLY as an oracle-internal instrument: it measures the size
  and structure of the residual physics (e.g. confirming the ~6.8 dB
  corner-regime deficit is vertex-shaped), and serves as an upper-bound
  benchmark for Route A ("no derived coefficient may explain less of the
  residual than the fitted one"). Nothing from Route B ever ships to a
  production kernel; it cannot close this charter.
- **Route C - exact reference solutions (GROUND TRUTH, never shipped).**
  Two independent generators: (i) high-resolution witwin-maxwell FDTD
  micro-scenes isolating a single cube vertex (Yee-locked, same recipe as
  the P1 arbiter, `benchmarks/fullwave_validation/experiments/
  run_maxwell_three_cube.py` pattern); (ii) the spherical-multipole /
  spectral quarter-plane solution as the analytic anchor in its geometry.
  *Discriminator D-C:* the two generators must agree with each other in
  their overlap regime before either is used to judge Route A.

Decision rule: Route A validated by D-A/D-C clears the way to an ADR-017
device design; Route A obstructed + Route B showing the residual is NOT
vertex-shaped falsifies this charter's premise (record and stop); Route A
obstructed + Route B confirming vertex shape keeps the charter open with
the obstruction as the new problem statement.

## 4. Reusable assets (exact paths)

- `artifacts/p4-oracle/oracle_lib.py` - fp64 header port
  (operation-faithful, MC fast-path bit-identity verified);
  `beta_term_pole()` -> (p = j sqrt(kL a), C); `G_incomplete` /
  `G_infinite_closed` / `G_infinite_quad`;
  `finite_wedge_truncation_factor_bounds` port. **Known correction that
  MUST be applied when using the truncated object: the G bounds scale is
  `sqrt(k kappa / 2)`, not the as-written `sqrt(k kappa / pi)` (factor
  sqrt(pi/2) = 1.2533, O2-verified).**
- `artifacts/p4-oracle/o2_matrix.npz`, `o2_report.json`, `o2_brute.py`,
  `o2_geom.py` - the 55-row brute-force EEC matrix incl. the
  lambda/20-cell geometries reconstructed from the P2 path tables.
- `artifacts/p4-oracle/o3_dose_response.json`, `o3b_scoped_report.json` +
  the o3/o3b reweighting scripts - the validated offline dose-response
  harness (reproduces recorded fields to 1.9e-9).
- Recorded solves and references: three-cube
  `artifacts/fullwave/three-cube-metal-320/` (FDTD truth
  `visual-maxwell-metal-three-cube-5ghz-320.npz`, s_empty 62.5931; P2
  tables under `p2/`); single-cube
  `artifacts/fullwave-refs/fullwave-smoke/visual-maxwell-metal-centered-5ghz-256.npz`
  (s_empty 62.2108) and `artifacts/fullwave-refs/fullwave-fix/` diagnostic
  script library.
- Comparison figures: `artifacts/fullwave/{single,three}_cube_fullwave_vs_deterministic.png`.

## 5. Frozen acceptance gates and stop-loss

Transferred verbatim (ADR-013 measured-acceptance + ADR-016):

1. three_cube_320 cells (0.0531, 0.4531) AND (0.284, 0.241): |gap vs FDTD|
   <= 6 dB.
2. Coupled-active ISB p95 excess < +2.0 dB; three-cube NMSE < 0.0899; ISB
   p95 excess < +1.0 dB; RSB p95 excess <= +3.6 dB (must not regress).
3. Single-cube: corner-zone ISB toggle median <= 0.5 dB; shadow-gap
   p10/p90 spread <= 4.3 dB (halved from 8.67); NMSE <= 0.0358; coherence
   >= 0.885; ISB p95 excess <= -0.1 dB (no regression).
4. MC / non-stationary call sites bit-identical; coupled-off unchanged
   within the documented run-to-run band; runtime three-cube warm <= 4 s.

**Stop-loss (ADR-016 precedent, binding):** the offline oracle
dose-response of the COMPLETE unit (exact G + derived vertex coefficient +
adjacent-edge coupling) must clear gates 1-3 BEFORE any device/kernel work
begins. If it does not, the charter closes with the recorded mechanism -
no partial landing, no calibration fallback, no gate weakening.

## 6. Research-agent boundaries

- Subagents write ONLY under `artifacts/vertex-oracle/` (new) and read
  everything else; fp64 numpy/scipy oracles and FDTD micro-scene runners
  only. No CUDA/device code, no edits to `src/`, `native/`, `ci/`,
  `benchmarks/` (a versioned micro-scene experiment script, if needed,
  goes through the main thread), and **no RayD edits of any kind**.
- Subagents do not commit; the main thread commits per verified stage.
- Load-bearing attributions and all derivation mathematics are main-thread
  work; subagents build laboratories, run matrices, and check limits
  (plan 09 §6 protocol).

## 7. Is true vertex diffraction needed NOW? (assessment, 2026-07-18)

**Not for the current benchmark acceptance.** Every frozen runbook
threshold passes today (three-cube NMSE 0.0922 < 0.12, coherence 0.8467 >
0.80, ISB +1.08 < 3.0, RSB +0.43 < 4.5; single-cube at or above all
records). The vertex-owned residuals are: one lambda/20 cell at -21 dB
(spatially isolated, receiver 3 mm from a PEC face - questionable as a
physical observation point for channel modeling), the single-cube
corner-zone median 1.09 dB vs the 0.29 dB pre-mend baseline, the
shadow-gap +/-4 dB spread, and deep-shadow fine structure visible in the
gap maps below -35 dB of peak.

**It becomes necessary if any of these hold:** (a) receivers closer than
~lambda/10 to scattering surfaces are a product scenario (device-adjacent
antennas, RIS-mounted probes); (b) deep-shadow fidelity beyond ~35 dB
dynamic range matters (coverage-hole prediction); (c) the differentiable
solver's gradients are consumed near corner rays, where the calibrated
blend's gradient is model-error-dominated; or (d) the research contribution
of §2 is pursued for publication - the derivation itself is the prize.

Recommendation: run this charter as a background research track. Stage
order: Route-C micro-scene references first (cheap, reusable), Route-B
diagnostic fit second (bounds the prize), Route-A derivation third with
the D-A discriminator ready before any device ADR is drafted.
