# Plan 16: Deterministic uniform boundaries, vertex diffraction, and surface-current correctness

## Status

Proposed. This plan follows the boundary audit and ADR-045/ADR-046. It is a
physics-first roadmap; smoothing is permitted only as a final presentation
operation after the field representation and topology limits are correct.

## Objective

Make the deterministic solver converge toward the immutable single-cube and
three-cube full-wave references without hiding ISB, RSB, vertex, overlap, or
deep-shadow defects. Preserve coherent complex fields, source amplitude,
polarization, phase convention, reciprocity, and the production native-CUDA
ownership rules.

## Baseline and non-regression contract

- Keep the four archived Maxwell reference NPZ files byte-for-byte unchanged and
  record their SHA-256 before and after every campaign.
- Run with `isb_boundary_taper=False`; report the effective width as `0.0`.
- Use the declared `reference_frequency_hz`; never infer or replay it on host.
- Report complex-field NMSE, magnitude correlation, ISB/RSB median-p95-max,
  shadow adjacent jumps, topology births/deaths, component support, path count,
  wall time, peak memory, launch counts, and synchronization counts.
- Render deterministic and full-wave field panels with `inferno`; residual,
  boundary, and support maps retain their diagnostic colormaps.
- Freeze single-cube and three-cube candidate metrics as regression envelopes,
  not as evidence that the remaining physics is complete.

## Phase 0 — Boundary ledger and probes

1. Publish per-row component, interaction sequence, edge/face/vertex IDs,
   stationary parameters, visibility decisions, transition-function arguments,
   Jones bases, complex coefficient, and accumulated contribution through a
   debug-only resident diagnostic path.
2. Add receiver trajectories normal and tangential to each ISB/RSB, every finite
   edge endpoint, cube corner, and deep-shadow support boundary.
3. Classify each jump as topology, coefficient, basis/sign, overlap, numerical
   root, or sampling/aggregation. A fix must name one class and one owner.

Exit gate: every top-1% jump in both scenes has a reproducible row-level cause.

## Phase 1 — Uniform single-edge field

Replace piecewise GO plus ordinary UTD assembly near ISB/RSB with one uniform
representation derived from the same canonical incident/outgoing geometry.
The transition variable, GO limiting term, diffraction term, phase reference,
and polarization basis must be evaluated in one native owner and share primal,
JVP, and VJP branch decisions. Verify the two one-sided limits analytically and
numerically, including PEC and dielectric material limits.

Exit gate: no finite discontinuity at ISB/RSB for a fixed finite-edge topology;
reciprocity and AD duality remain within existing tolerances.

## Phase 2 — Finite-edge endpoint and vertex diffraction

1. Introduce an explicit vertex interaction identity rather than assigning an
   endpoint limit arbitrarily to an adjacent edge.
2. Implement a canonical vertex diffraction coefficient or a validated
   equivalent uniform endpoint construction. Its domain begins exactly where
   ordinary interior-edge stationarity ends.
3. Define deterministic ownership at shared mesh vertices using stable world IDs
   and geometry-derived adjacency; remove duplicate incident-edge contributions.
4. Verify continuity edge-interior -> endpoint transition -> vertex cone, plus
   invariance to triangle tessellation, edge orientation, and path reversal.

Exit gate: endpoint/vertex trajectories have bounded, resolution-convergent
complex fields and no double count.

## Phase 3 — Coupled R-D, D-R, and D-D uniform limits

Use ADR-046's certified convex D-D stationary pair as the ordinary-domain input.
Build uniform coupled limits for stationary points approaching endpoints and for
short middle legs. Do not clamp an ordinary UTD coefficient into a vertex domain.
R-D/D-R must use the same reflection Jones operator and face-side convention as
standalone reflection. D-D must define its endpoint/vertex handoff and preserve
reversal invariance.

Exit gate: coupled rows have certified roots, ordinary legs satisfy the RayD
minimum-distance domain, and births/deaths are paired with a uniform replacement.

## Phase 4 — Overlap subtraction and canonical composition

Create a component overlap ledger for LOS, reflection, transmission,
diffraction, coupled R-D/D-R/D-D, endpoint/vertex, and any later surface-current
term. Derive subtraction/partition rules from asymptotic limits, not spatial
heuristics. Enforce one canonical phase origin and source-to-sink Jones map.

Exit gate: adding a higher-order family does not create an O(1) jump or duplicate
power in a lower-order asymptotic region.

## Phase 5 — Surface-current correction

A surface-current method is feasible as a residual correction, not a second
uncoordinated solver. Use equivalent electric/magnetic currents derived from the
already computed incident tangential fields on illuminated faces, integrate them
with a native CUDA quadrature/FMM-style owner, and subtract the asymptotic
specular/edge content already present in ray terms. Begin with PEC physical
optics, then add impedance/dielectric currents only after the PEC overlap proof.

Adaptive quadrature is driven by wavelength, phase curvature, visibility, and
edge distance. All samples remain resident; host code may construct only static
compile-time tables allowed by architecture. Surface currents are expected to
improve shadow interiors and finite-face diffraction but cannot repair wrong
source amplitude, visibility, topology, or Jones bases.

Exit gate: the residual correction converges under surface refinement, reduces
held-out full-wave error, and approaches zero in the asymptotic regions owned by
existing ray terms.

## Phase 6 — Performance controls

- Vertex diffraction: enumerate only geometry-adjacent visible candidates and
  compact natively; expected cost is proportional to surviving vertex paths.
- Surface current: use hierarchical/adaptive face tiles, frequency-stable static
  resources, fused evaluation, and receiver batching. A naive face-sample x
  receiver product is forbidden as the production design.
- Publish time and memory by component. Each new family requires an independent
  off/on benchmark and a bounded candidate budget that fails loudly rather than
  truncating.

Expected cost: vertex diffraction should be a moderate increment for cube-scale
scenes; surface current can dominate runtime without hierarchy, but an adaptive
residual implementation should be activated selectively for shadow/finite-face
regions. No complexity claim is accepted without measured single/three-cube
launch, memory, and throughput evidence.

## Phase 7 — Optional smoothing

Only after Phases 1-5 pass may a visualization or measurement-aperture filter be
added. It must be explicitly labeled, conserve the intended complex/aperture
quantity, never alter exported path fields, and never be used in correctness
metrics. Solver-internal empirical spatial blending remains prohibited.

## Required test matrix

- Analytic: free-space short dipole, PEC half-plane/wedge, canonical reflection,
  line-Fermat and coupled stationarity, reciprocity, and asymptotic limits.
- Native contracts: primal/JVP/VJP parity, finite/degenerate domains, stream and
  device residency, stable row order, failure-state all-or-nothing behavior.
- Geometry invariance: translation, scale, edge-axis reversal, path reversal,
  tessellation, shared-vertex ownership, and endpoint perturbations.
- Full wave: immutable single cube and three cubes at the locked grid/frequency,
  plus at least one held-out frequency and geometry before acceptance.
- Performance: cold/warm solve, component launch counts, GPU peak memory,
  cardinality transfers, and throughput.

## Stop conditions

Stop and require a new accepted ADR if a proposal needs Torch/CPU production
physics, a second numerical owner, host compaction, silent truncation, a hidden
fallback, finite-difference production AD, or empirical smoothing inside the
correctness path. Stop surface-current work if overlap subtraction cannot be
stated and tested before implementation.