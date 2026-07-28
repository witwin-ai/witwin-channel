# Audit: physical-boundary discontinuities after the no-taper rerun

- **Date:** 2026-07-28
- **Status:** candidate evidence; not an accepted production change
- **Scope:** deterministic single-cube and `three_cube_320` PEC comparisons at
  5 GHz, with the full-wave arrays immutable and the explicit ISB boundary
  taper disabled
- **Supersedes for this candidate:** the implementation conclusions, but not
  the historical measurements, in
  `fullwave-deterministic-discontinuity-audit.md` and
  `utd-continuity-fix-design.md`
- **Related proposals:** ADR-044 and ADR-045

## Executive conclusion

This pass removed three independent confounders before attempting another
boundary fit:

1. RayD's stationary direct-diffraction source state now retains the physical
   short-dipole `sin(theta)` amplitude and exact axial null.
2. RayD's line-Fermat point uses a local-coordinate closed form, and Channel's
   coupled D-D discovery now publishes only a certified ordinary interior
   stationary pair.
3. CUDA and OptiX diffraction visibility now trace toward the same
   exterior-bisector-offset point.

The explicit ISB boundary taper is off (`isb_boundary_taper=false`, effective
width `0.0`) in every benchmark entry point and recorded result. No smoothing
was added or retuned.

These corrections improve the global envelope fit and suppress the largest
shadow-region spikes, but they do **not** complete the ISB/RSB/vertex repair.
In the three-cube case, the ISB/RSB p95 tails regress even as their medians and
maxima improve. A one-sided refinement shows that the dominant remaining jump
is an already-existing D-D coefficient, not a path birth. The missing physical
owner is a uniform finite-edge/vertex/inter-edge closure. Applying a naked
exact finite-edge factor or another scalar taper cannot provide the required
complex cancellation.

## Reproducibility boundary

All modifications are isolated from the original checkout:

| Owner | Worktree | Branch | Base |
|---|---|---|---|
| Channel | `.codex_tmp/worktrees/fullwave-physical-boundary` | `codex/fullwave-physical-boundary` | `73a553fdb6a19bccd762b0e4d6a851d9fba07c21` |
| RayD | `.codex_tmp/worktrees/rayd-physical-boundary-safe` | `codex/direct-source-sintheta-boundary` | `49c58c4cb8212f6babb920cc88fb937509826cc5` |
| Baseline evidence | `E:/Code/witwin-platform/.worktrees/fullwave-ground-truth` | unchanged | clean |

RayD tests and builds use `witwin3`. Channel tests and builds use `witwin2`,
as required by the Channel repository contract.

The candidate Channel extension is a Release, SM120, OptiX-enabled build:

- build fingerprint:
  `ea180ceec82bd230fd500a0901d4e84cfda1b421e54088a4ca916e0eb920a19b`
- extension SHA-256:
  `4fef92c3cebac52768709293cb643b34dd9b8a3c20f7a3406c6f781e7f303bce`
- Torch `2.10.0`; CUDA compiler `12.9.41`
- source-linked dirty RayD and Channel worktrees are developer evidence only

The four reference arrays were read only. Their final SHA-256 values are:

| Reference | SHA-256 |
|---|---|
| single metal | `d2b87b6c4ad6a693c7f4a35c84b44d5c504ad789cb5bb3c2b60fbf4887b51ab0` |
| single empty | `ac0db283958f8034be3e1efb64e28c6b3a6730abc74976483ee3402218a2aaab` |
| three metal | `e6296dd0c6f12f91df80565c118041bfc2566d9031cc1cccb2abcc86932a7944` |
| three empty | `26c17a39577ba601c17771982b235218e0c137b5745cbc5a7b5536d16bec8172` |

## Confirmed defects and candidate corrections

### 1. Stationary direct-source amplitude

`direct_source_vector` normalized the transverse dipole projection. That
removed the norm

\[
\left\lVert
  \left(I-\hat{\mathbf r}\hat{\mathbf r}^{T}\right)\mathbf p
\right\rVert=\sin\theta
\]

and replaced the axial null with a basis fallback. RayD now transports the
unnormalized projection in the shared float/Dual template. Tests cover
analytic angles, axial zero, the `cos(theta)` tangent, finite differences,
native JVP, and JVP/VJP contraction.

This is a source-physics correction, not a boundary closure. It intentionally
changes direct D and the first leg of D-D.

### 2. Artificial Mend cutoff

`mend_beta_term_value` hard-zeroed a nonzero asymptotic term for
`w <= 1e-3`. The candidate removes that empirical threshold and returns zero
only at actual exponential underflow. This eliminates a small coefficient
step without introducing a blending window.

### 3. Diffraction visibility parity

The OptiX route traced visibility toward

\[
Q_{\mathrm{vis}}=Q+10^{-4}
\frac{n_0+n_1}{\lVert n_0+n_1\rVert},
\]

while the shared CUDA/Dr.Jit route used `Q`. The same candidate could
therefore appear or disappear solely by trace backend. The shared algorithm
now uses the same exterior-bisector offset. The published interaction point,
path length, delay, and field geometry remain based on the physical `Q`.

### 4. RayD line-Fermat cancellation

The former unfold/rotate/cross construction lost precision near nearly
parallel or short-leg configurations. RayD now evaluates the stationary
parameter in an edge-local weighted-axial form:

\[
u=\frac{\rho_t z_s+\rho_s z_t}{\rho_s+\rho_t}.
\]

The float and Dual paths instantiate the same expression. Non-finite inputs,
zero edge axes, and the undefined both-on-line case fail with `NaN`; they do
not receive an arbitrary point.

For the audited single-cube `rx=24362, edge 13 -> 9` witness, the native
strict bracket was `[0.164044395, 0.164044410]`. The independent FP64 joint
Fermat root was `(u, v) = (0.10442336348, 0.16404439881)`, while the former
sequential replay falsely reported an approximately `7.45e-7 m` discrepancy.
The new exported point differs from the FP64 oracle by at most
`1.42e-8 m`.

### 5. Coupled D-D stationarity and ordinary-field domain

The old Channel path alternated two unbounded line maps exactly sixteen times,
published the final iterate without a convergence certificate, and admitted
legs down to `1e-6 m` although RayD's ordinary UTD field requires
`1e-4 m`.

The candidate reduces the convex two-edge Fermat problem to a one-dimensional
signed derivative, requires a strict interior bracket, performs at most 32
FP32 bisections, reconstructs the analytic first-edge partner at the final
midpoint, and certifies both interaction-position widths against

\[
8\,\epsilon_{32}\,S_{\mathrm{local}}.
\]

All reduced math is evaluated relative to edge 1, avoiding subtraction of
large world coordinates across a short middle leg. A shared vertex, a
zero/sub-UTD middle leg, a flat interval, an endpoint solution, or a same-line
pair is rejected as outside the ordinary sequential-D-D physical domain.
A regular bracket that cannot meet the representable position certificate
device-fails; it is not silently dropped.

This changes the component probe from 31 to 28 D-D rows. The known
`33 -> 30` and `35 -> 30` singular candidates are inactive; the slow but
ordinary `42 -> 30` candidate remains active.

## Component isolation

The final candidate was probed by interaction sequence:

| Component | Topology | Maximum complex-field delta |
|---|---|---:|
| R | identical | exactly `0` |
| D | identical, 8 rows changed | `4.99e-6` |
| R-D | identical | `4.07e-11` |
| D-R | identical | `1.57e-11` |
| D-D | 31 rows to 28 rows | not rowwise comparable |

The R-D/D-R deltas are floating-point-scale effects of the combined RayD
candidate, including the removed Mend cutoff; there is no topology
contamination. Pure reflection is a bitwise sentinel.

## Full-wave benchmark result

The candidate outputs contain 676,009 paths for the single cube and 3,563,755
paths for the three-cube case. Solve times were `0.745 s` and `2.906 s`.

| Metric | Single baseline | Single candidate | Three baseline | Three candidate |
|---|---:|---:|---:|---:|
| calibrated magnitude NMSE | 0.035830 | **0.035050** | 0.092190 | **0.091248** |
| magnitude correlation | 0.87334 | **0.87612** | 0.79748 | **0.79972** |
| ISB median jump, dB | 1.089 | **0.274** | 1.066 | **0.569** |
| ISB p95 jump, dB | 2.540 | **2.104** | **5.666** | 6.424 |
| ISB max jump, dB | 5.367 | **4.870** | 17.268 | **14.511** |
| RSB median jump, dB | 0.747 | **0.452** | 0.953 | **0.804** |
| RSB p95 jump, dB | 3.554 | **3.368** | **6.319** | 6.735 |
| RSB max jump, dB | **6.968** | 9.018 | 38.082 | **27.368** |
| shadow adjacent max, dB | 16.623 | **11.968** | 40.481 | **29.695** |

The global and median improvements are real, but the three-cube p95 regression
prevents a claim that the boundary problem is solved. Shadow p95/p99 also
move slightly upward in the three-cube case. These tails remain visible rather
than being hidden by a taper.

## Residual localization

A one-sided receiver refinement down to `h = 4.88e-5 m` converges to a nonzero
complex jump of approximately `7.14e-4`. Only two small D-D births occur at
the smallest offsets; they explain about 2.3% of the dominant change. The main
term is an already-present `30 -> 45` D-D coefficient.

This rules out three tempting but incorrect fixes:

- increasing the D-D fixed-point iteration count;
- fading rows solely when they are born or die;
- smoothing the final accumulated field.

The coefficient has reached a boundary where the infinite/ordinary
sequential-wedge decomposition changes ownership. The compensating endpoint
and vertex field is absent.

## Required physical closure

The next implementation must be one atomic numerical decision derived from a
common generalized Fresnel/equivalent-current representation:

1. an exact incomplete finite-edge `G` term for each edge saddle;
2. a physical vertex endpoint sum over all incident edges;
3. a uniform two-edge D-D term that replaces, rather than overlays, the
   ordinary cascade in their shared transition region;
4. overlap subtraction/partition of unity so the edge, vertex, and two-edge
   terms have one owner and one asymptotic limit;
5. the reflected incident Jones state for RSB closure, not a direct-source
   scalar proxy.

Deleting the existing finite-edge corner surrogate and enabling the exact
`G` factor alone is not safe. In the single-cube counterfactual it moves the
corner-ISB median from approximately `1.09 dB` to `5.09 dB` and worsens NMSE
from about `0.0358` to `0.0639`, because the exact saddle exposes the still
missing vertex/inter-edge term. The historical calibrated `C_BLEND` surrogate
is therefore frozen, not retuned, in this candidate until the complete unit is
available. This is not acceptance of that surrogate as the final physics.

No receiver-domain smoothing should be considered before that closure passes
independent analytic and full-wave gates. If a residual remains afterward,
the only defensible smoothing is a declared measurement operator—finite
receiver aperture, bandwidth, or sampling convolution—with its scale supplied
by the measurement contract rather than fitted to the benchmark.

## Validation completed

RayD (`witwin3`):

- native diffraction contract: 1/1 passed;
- CUDA/OptiX multipath parity: 8 passed;
- finite-edge/path/accumulation suites: 8 passed;
- shared ownership checks: 4 passed;
- float/Dual, finite-difference, translation, scaling, axis reversal,
  degeneracy, and public-exporter line-Fermat regressions passed.

Channel (`witwin2`, final clean extension):

- coupled topology, deterministic coupled paths, coupled AD, enumerated owner
  and integration, geometry, single-wedge deterministic, and full-wave
  validation selections: 79 passed;
- dedicated coupled-D-D file: 5 passed;
- CUDA unit/contract acceptance: 2,155 passed and 3 skipped;
- four-solver smoke: 14 passed; no-fallback: 26 passed; CUDA AD core:
  121 passed;
- complete nightly `tests/ad`: 318 passed, 2 skipped, and 1 expected failure;
- Ruff, mypy, import graph, 234-binding contract coverage, orphan modules,
  product identity, production dependencies, and the public API/binding
  manifests passed. The binding manifest's D-D doc field was intentionally
  updated from the removed alternating projection to the convex bracketed
  Fermat contract;
- all candidate diffs pass `git diff --check`.

The top-level `cuda` runner itself stops at its clean-worktree hygiene gate,
as expected for an uncommitted review candidate. The four CUDA commands it
defines were therefore executed directly, without omitting a numerical gate,
using the explicit RayD worktree and a repository-local pytest base temporary
directory. An initial concurrent run had 80 setup errors from denied access to
the shared user temporary directory and nine environment/test-path failures;
the clean rerun above eliminated all of them.

The complete candidate manifest is
`artifacts/fullwave/physical-boundary-20260727/final_candidate_run_manifest.json`;
the metric report is
`artifacts/fullwave/physical-boundary-20260727/candidate_comparison_report.json`;
the convergence and component-isolation reports are stored beside them.

## Merge disposition

Do not merge this dirty developer candidate as-is. ADR-044 and ADR-045 remain
Proposed. RayD must first publish a clean reachable commit and source manifest;
Channel must update its lock and run the required CUDA/nightly gates from the
clean source-linked build. The uniform vertex/inter-edge closure is a separate
numerical ADR and must not be smuggled into either correctness patch.
