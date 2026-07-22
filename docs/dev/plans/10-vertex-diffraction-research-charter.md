# Plan 10 — Vertex-diffraction research charter

Date: 2026-07-18. Kind: RESEARCH CHARTER (not an ADR; no production change
is authorized by this document). Successor to the ADR-016 oracle verdict;
owner of the investigation into every residual that verdict attributed to a
model-completeness gap whose leading hypothesis is vertex-wave / inter-edge
physics. That attribution remains a hypothesis until the discrimination gates
in this charter identify it against the competing mechanisms below.
Revised 2026-07-18 after literature and research-design review.

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

Conclusion frozen here: the calibrated stand-in compensates a real
model-completeness gap; vertex waves launched where finite edges terminate and
their coupling into adjacent edges are the leading, physically motivated
hypothesis, NOT a uniquely identified cause. O2 validates the single-edge
object against an equivalent-edge-current reference, while O3/O3b observe an
aggregate field response; neither experiment alone separates vertex physics
from finite-distance/asymptotic error, slope-diffraction terms, face-current
effects, polarization error, omitted path families, or near-PEC reference
error. The discrimination stages in section 3 must perform that separation.

If the leading hypothesis survives, any principled replacement must ship
`exact G + vertex coefficient (+ adjacent-edge coupling)` as ONE numerical
unit; partial landings remain forbidden because the recorded partial swaps
regress. This is a conditional integration rule, not proof that the proposed
decomposition is the unique physical one.

Residuals this charter owns (current recorded values):

| Residual | Current | Source |
| --- | --- | --- |
| three_cube_320 cell (0.0531, 0.4531) gap vs FDTD | -21.3 dB | ADR-013 G-A |
| three_cube_320 cell (0.284, 0.241) gap vs FDTD | -39.4 dB (coupled-scoped G alone reaches -1.5) | ADR-013 G-A / O3b |
| coupled-active ISB p95 excess | +2.48 dB | ADR-013 G-D |
| three-cube NMSE / ISB p95 excess | 0.0922 / +1.08 dB | ADR-013 G-E |
| single-cube corner-zone ISB toggle median | 1.09 dB (pre-mend baseline 0.29) | plan 09 §2 |
| single-cube shadow-gap p10/p90 spread | 8.67 dB | plan 09 §2 / O3b |

## 2. Primary mandate: identify, reproduce, then extend only if necessary

The literature baseline is materially closer to this problem than the initial
charter assumed. In particular, Albani, Capolino, Carluccio, and Maci,
"UTD Vertex Diffraction Coefficient for the Scattering by Perfectly Conducting
Faceted Structures," IEEE TAP 57(12), 2009,
[DOI 10.1109/TAP.2009.2027455](https://doi.org/10.1109/TAP.2009.2027455),
gives a uniform first-order high-frequency,
dyadic electromagnetic vertex coefficient for a PEC pyramid tip, finite
source/observation distance, and generalized Fresnel transition functions. Its
worked validation includes the orthogonal pyramid formed by three mutually
orthogonal edges explicitly identified with a cube/parallelepiped vertex. That
object is the mandatory first baseline, not merely a distant limiting case.

The unresolved research question is narrower and falsifiable: whether that
published high-frequency construction, composed consistently with ADR-016's
exact incomplete single-edge object and the existing D->D family, explains the
measured `kL ~ 0.4-1`, lambda/20 near-zone residuals. A new derivation is
authorized only after a faithful fp64 reproduction of the published object
fails a converged, same-geometry reference for a named mechanism.

The candidate extension, if required, is the DOUBLE-incomplete two-pole
generalization of ADR-016's `G(p; w1, w2)`: two Fresnel truncations and up to
two GO-boundary poles interact at a corner ray, with three-edge coupling
entering through endpoint matching. The conjecture to prove or refute is that
the missing near-zone correction admits a closed or uniformly computable form
in iterated Faddeeva/generalized-Fresnel functions of the same family already
validated in `oracle_lib.py`. Failure to reduce the integral is a valid result;
"closed form" is not a success condition when a demonstrably uniform numerical
canonical operator is required by the physics.

Any accepted composite object must satisfy more than visual continuity:

- the TOTAL field, not necessarily every separated coefficient, is finite and
  uniform on corner rays, ISB/RSB crossings through the vertex, and grazing
  limits;
- tangential electric-field PEC conditions hold on all incident faces to the
  declared asymptotic order;
- the electromagnetic dyadic handles both incident polarizations and
  cross-polarization, and satisfies reciprocity under source/receiver exchange;
- edge, double-diffraction, and vertex terms have explicit asymptotic order,
  phase convention, and no-double-counting rules;
- it reduces to (a) the infinite-edge KP assembly when both bounds recede,
  (b) the single-truncation G when only one end is near, (c) the published
  orthogonal-pyramid coefficient in its high-frequency domain, and (d) the
  rigorous quarter-plane result only in the geometry's actual quarter-plane
  limiting case;
- any dependence on remote edge lengths or other vertices is declared. If the
  target regime is intrinsically nonlocal, the charter must reject a universal
  local coefficient rather than hide that dependence in a fit.

Load-bearing attribution and derivation remain main-thread work; subagents may
build laboratories and independently check limits, per plan 09 section 6.

## 3. Candidate routes and discriminating experiments

The route labels express evidence roles, not execution order. Section 7 freezes
the stage order and stop conditions.

- **Route C - converged reference solutions (REFERENCE, never shipped).**
  A numerical solver is not called exact without a convergence record. Build:
  Yee-locked witwin-maxwell FDTD micro-scenes using at least three spatial
  resolutions, numerical-dispersion accounting, and a mesh-convergence error
  estimate (Richardson extrapolation only after an asymptotic convergence order
  is demonstrated). Remote face-edge and other-vertex arrivals must be isolated
  by time gating, face-size extrapolation, or an equivalent documented
  construction. The rigorous quarter-plane spectral solution is a separate
  limiting-case anchor, not an independent solution of the trihedral geometry.

  *Discriminator D-C:* the target FDTD observables must be stable under spatial
  refinement, time-window/PML checks, source calibration, and the declared
  remote-feature isolation method, with an uncertainty budget smaller than the
  model difference being judged. Quarter-plane agreement is scored only after
  the geometry is actually reduced to that limit. If this fails, improve the
  reference; do not fit or derive against an unresolved target. An independent
  same-geometry MoM/BEM/FEM cross-check is deferred, not required by this
  charter. Revisit it if FDTD uncertainty dominates the A0/A1 difference or if
  a later publication/production decision requires stronger external validity.

- **Route A0 - published-coefficient reproduction (MANDATORY BASELINE).**
  Implement a standalone fp64 oracle for the 2009 PEC pyramid-vertex dyadic
  and its generalized Fresnel functions, including the orthogonal-pyramid
  example, analytical limits, polarizations, and published numerical examples
  where reconstructable. Compose it offline with exact G without altering
  production code.

  *Discriminator D-A0:* reproduce the paper's analytical limits and reported
  examples within the reference/reporting precision, then score the target
  matrix against Route C. Record error by `kL`, source/observer distance,
  incidence, observation, polarization, transition class, and distance to a
  field null. If A0 clears the charter gates, no original derivation is needed.

- **Route B - constrained diagnostic attribution (DIAGNOSTIC ONLY).** Fit
  competing, physically labeled oracle bases: vertex/endpoint, finite-distance
  edge correction, slope diffraction, face-current/multiple-scattering, and
  omitted-topology indicators. Use disjoint train/validation geometries,
  regularization, and held-out polarizations/frequencies. Define
  "vertex-shaped" before fitting through spatial phase, distance scaling,
  polarization, and vertex/edge permutation symmetries.

  The best held-out fit is a performance ceiling and an identifiability probe,
  not a floor that a derived coefficient must equal. Nothing fitted ships to a
  production kernel and Route B cannot close the charter.

- **Route A1 - endpoint-integral extension (CONDITIONAL).** Only if A0 fails
  by a converged, vertex-identified mechanism, derive the missing vertex wave
  as the endpoint/corner contribution of truncated edge integrals plus
  inter-edge matching terms. The derivation must state whether the obstruction
  is analytic reduction, nonlocal geometry dependence, or failure of the
  assumed asymptotic decomposition.

  *Discriminator D-A1:* compare the composite against Route C over `kL` in
  `[0.3, 3]`, including lambda/20 observation radii. The primary metric is a
  normalized complex vector-field error with an absolute field floor and
  propagated reference uncertainty. The historical `<1% amplitude / <1 deg`
  phase target applies only where the reference uncertainty is smaller and the
  field magnitude is safely above the declared null floor; phase is not scored
  at a null. Both polarizations and cross-polarized components are mandatory.

The attribution matrix must explicitly try to falsify vertex ownership. A0/A1
must not receive credit for fitting errors that follow FDTD grid spacing,
near-PEC sampling, remote-face size, missing topology, or a non-vertex basis
more strongly than vertex symmetries.

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
  harness (reproduces recorded fields to 1.9e-9). This harness reweights
  EXISTING rows only; it is reusable for exact-G factors but is not by itself
  a complete vertex oracle because a vertex contribution can exist where an
  edge-diffracted row is absent.
- Recorded solves and references: three-cube
  `artifacts/fullwave/three-cube-metal-320/` (recorded FDTD reference
  `visual-maxwell-metal-three-cube-5ghz-320.npz`, s_empty 62.5931; P2
  tables under `p2/`); single-cube
  `artifacts/fullwave-refs/fullwave-smoke/visual-maxwell-metal-centered-5ghz-256.npz`
  (s_empty 62.2108) and `artifacts/fullwave-refs/fullwave-fix/` diagnostic
  script library. These references remain valid regression records but do not
  become working canonical micro-scene references until Route C records
  convergence, isolation evidence, and an uncertainty budget.
- Comparison figures: `artifacts/fullwave/{single,three}_cube_fullwave_vs_deterministic.png`.
- Mandatory literature baseline: Albani et al. 2009,
  [DOI 10.1109/TAP.2009.2027455](https://doi.org/10.1109/TAP.2009.2027455);
  quarter-plane compatibility/reference work must be catalogued separately
  from orthogonal-pyramid/trihedral references.
  The S0 catalogue must also include Ivrissimtzis, "Edge wave vertex and edge
  diffraction," Radio Science 24(6), 1989,
  [DOI 10.1029/RS024I006P00771](https://doi.org/10.1029/RS024I006P00771), and
  Assier/Abrahams, "A Surprising Observation in the Quarter-Plane Diffraction
  Problem," SIAM J. Applied Mathematics 81(1), 2021,
  [DOI 10.1137/19M1258785](https://doi.org/10.1137/19M1258785), together with
  the primary works those papers identify.

## 5. Frozen acceptance gates and stop-loss

Physics gates transferred from ADR-013 measured acceptance and ADR-016; the
runtime limit is the stricter ADR-016 bound. This is an intentional combined
gate set, not a verbatim copy of either source:

1. three_cube_320 cells (0.0531, 0.4531) AND (0.284, 0.241): |gap vs FDTD|
   <= 6 dB.
2. Coupled-active ISB p95 excess < +2.0 dB; three-cube NMSE < 0.0899; ISB
   p95 excess < +1.0 dB; RSB p95 excess <= +3.6 dB (must not regress).
3. Single-cube: corner-zone ISB toggle median <= 0.5 dB; shadow-gap
   p10/p90 spread <= 4.3 dB (halved from 8.67); NMSE <= 0.0358; coherence
   >= 0.885; ISB p95 excess <= -0.1 dB (no regression).
4. MC / non-stationary call sites bit-identical; coupled-off unchanged
   within the documented run-to-run band; runtime three-cube warm <= 4 s.

For oracle comparisons, gates 1-3 are evaluated with the Route-C extrapolated
reference and its uncertainty. Legacy point estimates remain reported for
continuity with ADR-013/016. A change smaller than combined reference
uncertainty is inconclusive, not a pass. Per-term amplitude/phase criteria use
the section-3 complex-field metric and null floor.

**Stop-loss (ADR-016 precedent, binding):** the offline oracle
dose-response of the COMPLETE identified unit (exact G + A0 or A1 vertex
object + adjacent-edge coupling, or the alternative mechanism selected by the
attribution matrix) must clear gates 1-3 BEFORE any device/kernel work begins.
The complete oracle must synthesize explicit vertex contributions even where
the recorded path table has no edge-diffracted row; row reweighting alone
cannot satisfy this gate. If the complete unit does not clear the gates, the
charter closes with the recorded mechanism - no partial landing, no
calibration fallback, no gate weakening.

## 6. Oracle completeness and native-device feasibility

### 6.1 Standalone topology-complete oracle

Before field-level dose response, the oracle must enumerate every applicable
vertex-centered contribution independently of current production path-row
existence. This includes `TX -> vertex -> RX`, the endpoint terms assigned to
all incident edges, and any edge->vertex / vertex->edge coupling required by
the selected composite. It must:

- remain defined on both sides of each edge-ray existence boundary;
- record vertex ID, incident/observation face and edge IDs, polarization basis,
  phase origin, and the exact rule that prevents double counting with edge and
  D->D rows;
- demonstrate invariance under cube-axis permutations and source/receiver
  reciprocity;
- score the standalone vertex field, the exact-G edge field, and their coherent
  total separately before scoring aggregate solver fields.

The existing O3 scripts may assemble recorded components, but cannot substitute
for this topology-complete oracle. Production ownership is only a preliminary
constraint here: a future ADR must assign discrete vertex rows to
`propagation.topology`, continuous vertex geometry to `propagation.geometry`,
and field/dyadic evaluation to `propagation.fields`, while preserving the
existing fused native ABI ownership and forbidding solver-to-solver imports.

### 6.2 O4 feasibility gate before production acceptance

Clearing physics gates authorizes an ADR and a measured native prototype, not
immediate production landing. ADR-017 (or the next free ADR) must define and
verify:

- a region-split or fixed-work float32 evaluation whose error map has headroom
  against the fp64 oracle and is C1 across every algorithmic seam;
- registered native primal, JVP, VJP, and backward companions, with unsupported
  tangents failing before partial output and no finite-difference production
  derivative;
- one declared fusion/launch contract with no new host/device round trips,
  scalar extraction, avoidable synchronization, persistent tape, or
  materialized intermediate merely to mirror Python modules;
- exact/duality/reciprocity tests for all polarization branches and gradients
  near, but not exactly at, declared physical singularities/nulls;
- a repeatable warm benchmark with CUDA-event or explicitly synchronized
  timing, plus Nsight Systems evidence for launches/transfers/synchronization
  and Nsight Compute evidence for registers, occupancy, spills, branch
  efficiency, and memory traffic on the hot kernel;
- the <=4 s three-cube warm gate, launch-count and temporary-memory budgets,
  GPU model, compiler flags, launch configuration, and profiler counters.

Fast math, mixed precision, altered reduction order, and special-function
approximations are numerical changes and require their own accuracy evidence.
If O4 cannot meet native AD, fusion, residency, or runtime requirements, no
Torch/CPU or host-evaluated lookup fallback is permitted. A resident device
table is admissible only if a new accepted ADR treats its construction,
interpolation, native AD, accuracy, residency, and fusion as the production
algorithm rather than as recovery from a failed coefficient. Otherwise record
the obstruction and stop before production acceptance.

## 7. Execution stages and bounded decision rules

Each stage ends in a versioned report under `artifacts/vertex-oracle/` and an
explicit continue/stop decision. A stage may be repeated only for a named
reference defect or new mathematical hypothesis; "research remains open" is
not itself a decision.

1. **S0 - literature and convention lock.** Reconstruct A0's formulas,
   geometry, dyadic bases, generalized Fresnel definitions, branch conventions,
   and published validation cases. Record the exact overlap and mismatch with
   ADR-016 G and current edge/D->D terms.
2. **S1 - Route C references.** Produce the converged FDTD matrix, uncertainty
   budget, remote-feature isolation evidence, and separate quarter-plane
   limiting-case anchor. D-C must pass before attribution or derivation. An
   independent same-trihedral solver is a deferred upgrade, not an S1 gate.
3. **S2 - Route A0 reproduction.** Pass D-A0, compose A0 + exact G offline,
   and run the topology-complete dose response. If it clears sections 2 and 5,
   skip A1 and proceed to O4/ADR design.
4. **S3 - Route B attribution.** Run only when A0 leaves a converged residual.
   If a non-vertex basis explains the held-out residual as well as or better
   than the vertex basis, this charter closes or is explicitly re-chartered
   around that mechanism.
5. **S4 - Route A1 extension.** Run only when S3 identifies a vertex-owned
   residual. One derivation attempt and one reformulation from a precisely
   recorded obstruction are allowed. If neither yields a uniform object,
   decide between a non-shipping numerical canonical oracle and a documented
   negative result; do not keep the charter indefinitely open.
6. **S5 - complete field gate.** The topology-complete composite must clear
   all section-5 physics gates and the section-2 physical identities. Failure
   closes the charter without a partial landing.
7. **S6 - O4 and device ADR.** Perform section 6.2 only after S5 passes.
   Production work requires the accepted ADR, complete binding/governance
   updates, native AD companions, CUDA-tier validation, and measured profiler
   evidence.

## 8. Research-agent boundaries

- Subagents write ONLY under `artifacts/vertex-oracle/` (new) and read
  everything else; fp64 numpy/scipy oracles and FDTD micro-scene runners
  only. No CUDA/device code, no edits to `src/`, `native/`, `ci/`,
  `benchmarks/` (a versioned micro-scene experiment script, if needed,
  goes through the main thread), and **no RayD edits of any kind**.
- Subagents do not commit; the main thread commits per verified stage.
- Load-bearing attributions and all derivation mathematics are main-thread
  work; subagents build laboratories, run matrices, and check limits
  (plan 09 §6 protocol).

## 9. Is true vertex diffraction needed NOW? (assessment, 2026-07-18)

**Not for the current benchmark acceptance.** Every frozen runbook
threshold passes today (three-cube NMSE 0.0922 < 0.12, coherence 0.8467 >
0.80, ISB +1.08 < 3.0, RSB +0.43 < 4.5; single-cube at or above all
records). The candidate vertex-attributed residuals are: one lambda/20 cell at -21 dB
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
blend's gradient is model-error-dominated; or (d) the literature reproduction,
near-zone validity result, or a genuinely necessary A1 extension is pursued as
a research contribution.

Recommendation: run this charter as a background research track in the bounded
S0-S6 order above. The first deliverables are the literature/convention lock
and converged same-geometry Route-C references; no original derivation or
device ADR begins merely because the recorded residuals exist.

---

## Appendix A - Recorded mechanism-discrimination results (2026-07-18)

Two competing mechanisms from section 1 were tested offline the same day
(workflow artifacts under `artifacts/po-ptd-projection/`); both are
FALSIFIED as the source of, or the fix for, the visible seam lines. This
narrows the section-1 hypothesis space toward the value-level
corner-continuation physics this charter investigates.

**A. PO+PTD reorganization (face-current mechanism) - NO-GO at this
electrical size.** The offline projection replaced GO finite-face
reflection by the exact PO aperture factor (full 2x2 in-plane Hessian with
cross terms, fp64 2D quadrature over the true rectangles; operator
self-consistency PASSED: |A_PO| passes continuously through 0.5000 at the
face edge, endpoint wave matches D_PO to the leading 1/w order) and the
same faces' order-1 D by the Ufimtsev fringe ratio. Result: at 5 GHz the
0.2 m faces are only ~1-2 Fresnel zones wide, so physical PO ripples
0.16-1.42 over the WHOLE face; the RSB seam class GREW (three-cube 490 ->
1145 px, single-cube 113 -> 414), RSB p95 excess exploded +0.43 -> +12.98
dB, coherence fell 0.8467 -> 0.8382, and the fringe swap worsened ISB
(+1.08 -> +2.43). Verdict: GO+UTD's uniform resummation IS the better
effective theory at ~3.3-lambda faces; the FDTD-vs-deterministic seam gap
is NOT Kirchhoff face-current physics. Caveat recorded: the projection
could only modify lit-side rows (no shadow-side reflection rows exist in
the tables), understating a true device's shadow-side leakage - but the
lit-side ripple alone is disqualifying.

**B. Slope diffraction (C1 mechanism) - eliminated as a visible-seam
fix.** Audit proof (header consumption path): `incidentDerivativeVector`
is zeroed on EVERY stationary leg (utd_math.h:1467 external-incident
branch, :1470 plain stationary branch); only the MC branch consumes the
caller's derivative Jones, so slope diffraction is inert for order-1,
coupled, and D->D stationary evaluations alike - the kernel-side
`jones_zero()` writes are redundant documentation. Measured seam census at
coupled-active FDTD-smooth cells: C0 VALUE JUMPS dominate (136 cells, mean
6.90 dB, p95 15.0, max 23.6 dB) over C1 slope kinks (99 cells, mean 2.69
dB/cell, which retain a ~1.6 dB residual value gap anyway). Slope terms
are an O(1/k) correction to a family that moves the total >2 dB in only
4.9% of cells; realistic visible impact is sub-dB smoothing of a few
hundred pixels. Threading the derivative remains a documented completeness
item (fix sketch in the audit report), but it is NOT a path to visual
seamlessness and is not scheduled by this charter.

Consequence for the stage order: the visible hard lines are pinned as
value-level support-toggle steps (mean ~7 dB) at RSB / corner-ray /
coupled boundaries - exactly the object of stages S0-S2. The two
falsifications raise the prior that the corner-continuation (vertex-wave)
basis owns the residual, but section 3's discrimination stages remain the
arbiter.
