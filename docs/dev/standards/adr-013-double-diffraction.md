# ADR-013: Coupled double diffraction (D->D)

- **Status:** Accepted for implementation (plan-09 P2). Numerical change;
  coupled-off solves stay byte-identical.
- **Date:** 2026-07-17
- **Kind:** New order-2 compensator family member (component id 7) inside the
  existing `coupled_paths` gate. New native discovery + field kernels and ABI
  symbols. No RayD source change (the shared UTD header already provides every
  primitive this design needs).
- **Related:** ADR-011 (coupled paths), ADR-012 (stationary coupled leg,
  external-incident mode), ADR-009 (fusion ownership), ADR-004 (numerical
  duplication), plan 09 P1 arbiter data (`docs/dev/fullwave-validation.md`,
  ADR-011 arbiter addendum).

## Context: three measured defect classes, one missing term

The P1 full-wave arbiter (three_cube_320 vs witwin-maxwell FDTD) plus per-path
dissection of the P1 path tables established three defect classes in the
coupled-ON deterministic map. All three are manifestations of the same missing
order-2 term: cascaded edge diffraction TX -> e1 -> e2 -> RX.

1. **Pure-D blockage support toggles.** A direct diffraction path's
   edge->RX segment is occluded by another cube; the row is dropped by the
   visibility gate and the total steps 10-18 dB where FDTD moves < 1.6 dB.
   Measured targets (`artifacts/fullwave/three-cube-metal-320/p2_recon/`):
   (0.0469, 0.4969) 18.27 dB det vs 1.30 dB FDTD; (0.1656, 0.2469) 13.01 dB
   vs 1.55 dB (coupled-ON does not heal it: 11.68 dB); (0.0094, 0.6969)
   10.18 dB vs 0.56 dB. The healing term is diffraction around the occluder:
   e1 = the original edge, e2 = an occluder edge, born exactly at the
   visibility cut with the second leg's transition function.

2. **Coupled-row support toggles at specular-face-exit.** Coupled R->D / D->R
   rows are KILLED when the R-leg specular point leaves the finite face
   (RayD `reflection_epc_algo.h:525-544` barycentric containment ->
   `store_invalid`; no clamp, no extrapolation). The compensator vanishes
   discontinuously. The healing term is e1 = the face-boundary edge the
   specular exits through, e2 = the original coupled edge: at that boundary
   the first leg's observation point crosses e1's reflection-shadow boundary,
   so its UTD transition function jumps by exactly the lost reflection
   amplitude (Kouyoumjian-Pathak compensation applied leg-wise).

3. **Near-boundary over-cancellation ("anti-phase" cells).** At the P1 worst
   regression cells (e.g. (0.0531, 0.4531): OFF gap -0.5 dB -> ON gap
   -59.9 dB) the coupled rows are LEGITIMATE cross-body paths (verified by
   path-table dissection: D at cube1's NW edge -> R off cube2's east face,
   plus R off cube1's west face -> D at cube2's east edges) whose receiver
   sits ~lambda/20 from the reflecting plane; with r_te ~ -1 they form the
   physically required wall image and collectively cancel the order-1
   diffraction. FDTD shows the cancellation must be PARTIAL (~5 dB below the
   diffraction-only field, not -60 dB): the specular points sit ~lambda/4
   from the face boundary, where the finite face reflects far less than the
   GO value. In uniform theory the reduction is carried by the secondary
   source's edge term - again D(e1 = cube1 edge as source) -> D(e2 = the
   reflecting face's boundary edge). UTD arithmetic closes numerically:
   D + (-D) + D->D(transition ~ D/2) ~ 0.5 D, matching the measured ~5 dB
   suppression.

   NOTE (misattribution corrected): a recon hypothesis that the anti-phase
   cells were *degenerate same-wedge rows* (edge coplanar with the reflection
   group surviving the (eps, 1-eps) plane gate by float32 jitter) was REFUTED
   by dissecting the path table at (0.0531, 0.4531): all five coupled rows
   there have |refl - edge| >= 0.27 m and legitimate cross-body geometry. No
   candidate-exclusion or gating change is part of this ADR.

Consequently P2 is a PURE ADDITION: no change to cid 3/4 discovery, gating,
or field semantics. Continuity is obtained from physics (per-leg UTD
transition functions born exactly on the existing hard boundaries), not from
any smoothing device.

## Decision

### D1: enumeration (new cid 7, inside `coupled_paths`)

Ordered edge pairs (e1, e2) from the same `selected_edges` set used by
coupled discovery, e1 != e2, excluding collinear pairs (edges on the same
line within the shared geometry epsilon; they belong to the same physical
edge). `candidates_per_pair = E*(E-1)` (three-cube: 36*35 = 1260). One
direction only (the ordered pair already covers both traversals; there is no
reverse request and no new x2). Discovery mirrors
`propagation/topology/discovery/coupled.py`: the plan/iterator gains a
`dd` request stream with `component_id = 7` after the (False,3)/(True,4)
requests of each chunk; the per-block budget uses
`per_receiver_candidates = tx * (2*groups*edges + edges*(edges-1))` so the
existing rx-streamed wrapper and the 1M cap govern the union. Depth = 2,
`interaction_type_sequence = [2, 2]` (two diffraction events).

### D2: native discovery kernels (channel-native owned)

New symbols alongside the coupled ones in
`native/channel_native/kernels/coupled_topology.cu` and
`native/channel_native/rayd/geometry.cpp`:

- `cn_coupled_dd_prepare_cuda`: per candidate (tx, rx, e1, e2) solve the
  two-edge Fermat point pair (Q1, Q2) minimizing
  |tx-Q1| + |Q1-Q2| + |Q2-rx| by alternating the existing closed-form
  single-edge projection (fixed 16 iterations, deterministic order,
  float32). Validity: both parameters strictly inside their finite segments
  (same `kGeometryEpsilon` semantics as `coupled_rd_prepare_kernel`
  `inside_edge`, coupled_topology.cu:103-106) and the three segment
  directions well-defined. Emits Q1, Q2, per-leg lengths.
- visibility: three segment queries (tx->Q1, Q1->Q2, Q2->rx) through the
  existing `raydn_visibility_forward` C-ABI batch call (geometry.cpp:291
  pattern), ANDed into the row validity.
- `cn_coupled_dd_finalize_cuda`: assemble the 2-interaction sequence
  (edge ids in both slots), path length, delay, valid mask.
- `raydn_coupled_dd_geometry_forward`: the orchestrator bound to Python
  (mirrors `cn_raydn_coupled_rd_geometry_forward`, geometry.cpp:204).

Python: `propagation/geometry/coupled.py` gains `query_coupled_dd_geometry`
(typed facade over the new bridge symbol);
`propagation/enumerated/coupled.py` row-build sets `component_id = 7`,
depth 2, both interactions type 2, material_sequence = -1 (edges carry
wedge materials, resolved by the field stage exactly as coupled rows do).

### D3: native field kernel (two sequential wedge operators, one launch)

New `coupled_dd_field_kernel` in
`native/channel_native/kernels/field_transport.cu` (pattern:
`coupled_rd_field_kernel`, field_transport.cu:348-537):

- **Leg 1** (e1): `PairInputs` with `sourcePos = tx`,
  spherical incident from tx (same `free_space_complex3` construction as the
  coupled reverse branch, field_transport.cu:415-420),
  `selectStationaryPoint = 1`, real `edge_line_min/max` for e1,
  `stationaryExternalIncident = 0` (the incident IS the direct source),
  observation point = Q2_frozen (from discovery). Output: complex vector
  field at Q2 plus the leg-1 outgoing direction.
- **Leg 2** (e2): `PairInputs` with `sourcePos = Q1_frozen`,
  `incidentJones` = leg-1 output projected on e2's incident basis,
  `selectStationaryPoint = 1`, `stationaryExternalIncident = 1`,
  `sPrimeFrozen = |Q1_frozen - Q2_frozen|`, real bounds for e2,
  observation = rx. The ADR-012 re-extrapolation
  (scale = (sPrimeFrozen/sPrimeNew) * exp(-j k (sPrimeNew - sPrimeFrozen)))
  treats the leg-1 wave as locally spherical from Q1. This is exact at
  Q2_frozen and second-order in the re-anchor displacement - the same
  approximation class ADR-012 froze for the coupled R-leg; documented, not
  hidden.
- Both legs therefore inherit the full continuity machinery (edge
  re-anchoring, monotone even truncation, corner-mend gamma, boundary
  blend) - the G3 lesson is honored from day one.
- Receiver projection, tx_power scaling, outputs identical in shape to the
  coupled kernel (field_vector, coefficient, path_field, path_gain,
  direction_out).

**Known physical limitation (recorded, deferred):** cascading per-leg UTD
transition functions is leading-order wrong when the two edges sit in each
other's transition regions (overlapping transition zones need the
generalized two-variable transition function). That machinery is plan-09 P4
(the same generalized Fresnel integral replaces the corner-mend gamma). The
leg-2 coefficient evaluation must stay a single call site so P4 can swap it.

### D4: AD companions

`cn_field_coupled_dd_backward` / `cn_field_coupled_dd_jvp` twins (pattern:
`field_wedge_ad_coupled.cu::coupled_rd_row_dual` after ADR-012 G4-3, which
calls `compute_pair_vector_contribution` directly so the truncation/mend
derivatives flow in lockstep). tx/rx gradients flow through the live
re-anchoring inside the kernels; Q1/Q2/bounds are frozen seeds (detached),
and mesh-vertex gradients through cid-7 rows raise the same loud refusal as
cid 3/4 (`coupled_paths_mesh_vertex` exclusion). Primal/dual numerical
lockstep entries go into the duplication ledger.

### D5: accumulation and public surface

- `accum_slot()` (deterministic_accum.cu:36-50) gains `cid 7 ->
  kCoupledSlot (5)`; all backward/jvp/fwd64 variants inherit via the shared
  helper. The public component list is unchanged ("coupled" now aggregates
  cids 3, 4, 7; path tables keep cid 7 distinct for audits).
- NO new Config field: `coupled_paths=True` now means the uniform order-2
  compensator family {R->D, D->R, D->D}. A partial family is non-uniform by
  measurement (P1: net-wash aggregates; G3: seam injection), so a separate
  toggle would only enable known-wrong configurations. The A/B acceptance
  baseline is the frozen P1 artifact set, not a runtime toggle.
- `capabilities.py` coupling blocks gain
  `"coupled_double_diffraction": True` (deterministic + path + bdpt as
  applicable) -> intentional `ci/public-api-snapshot.json` regeneration +
  migration note (ADR-003 process).

### D6: governance (moves together, per CLAUDE.md)

New ABI symbols (`raydn_coupled_dd_geometry_forward`, `field_coupled_dd`,
`field_coupled_dd_backward`, `field_coupled_dd_jvp`; internal `cn_*_cuda`
helpers are not pybind symbols): `ci/native-binding-manifest.json` (179 ->
183), `ci/check_contract_coverage.py::EXPECTED_NATIVE_BINDING_COUNT`,
`ci/contract-coverage-manifest.json` (contract test + e2e caller per
symbol), `docs/dev/audit/phase9-native-owner-inventory.json` (+ body
hashes), `tests/test_native_owner_inventory.py` expectations, negative
no-fallback tests (symbol-missing, budget-exceeded with the new candidate
arithmetic), the accumulator oracle
(`tests/kernels/test_ops_facade.py::test_deterministic_accumulate_flat_matches_torch_reference`
gains cid-7 rows mapping to slot 5), capabilities test, launch-ledger
baseline refresh. Import graph: no new cross-domain edges (all additions
live inside existing owners).

## Acceptance gates (numeric; three_cube_320 vs the frozen P1 FDTD reference)

| Gate | Metric | Before (P1 ON) | Required after |
| --- | --- | ---: | --- |
| G-A | anti-phase cells gap vs truth: (0.0531,0.4531) / (-0.059,0.491) / (-0.028,0.672) / (0.284,0.241) | -59.9 / -31.8 / -31.8 / -25.6 dB | each within +/-6 dB |
| G-B | blockage steps (0.0469,0.4969) / (0.1656,0.2469) / (0.0094,0.6969) | 18.3 / 11.7 / 10.2 dB | each < 4 dB (FDTD: 1.3/1.6/0.6) |
| G-C | region-A envelope NMSE (coupled-active cells) | 0.350 (OFF 0.331) | < 0.331 |
| G-D | coupled-active ISB p95 excess | +3.92 dB (OFF +1.11) | < +2.0 dB |
| G-E | aggregate: NMSE / coherence / ISB p95 excess | 0.0934 / 0.8455 / +2.04 | < 0.0899 / > 0.845 / < +1.0 |
| G-F | RSB p95 excess | +3.53 dB | <= +3.6 dB (order-2 R-R-D is P3, not this ADR) |
| G-G | single-cube | - | coupled-off bitwise vs P1 build; full continuity suite green; single-cube-256 benchmark metrics not regressed |
| G-H | runtime | 2.13 s warm | < 10 s warm three-cube 320^2 |

Plus: all cuda-tier suites green; every governance artifact in D6 updated in
the same change; if any gate fails, per-path decompose the worst cell and
name the mechanism before touching thresholds (never weaken a gate to pass).

## Risks

- **Overlapping transition zones** (D3 note): edge pairs closer than a few
  Fresnel widths carry leading-order cascade error until P4. The acceptance
  cells at lambda/5-lambda/4 spacing are inside UTD's validity only
  marginally; gates G-A/G-B thresholds (+/-6, <4 dB) reflect that.
- **Cost**: candidates/rx grows 1296 -> 2556 (~2x coupled solve time,
  bounded by G-H). The streaming budget math must count the union, or a
  block sized for cid 3/4 alone would overrun the 1M guard.
- **Same-line edge pairs**: collinear e1/e2 pairs are excluded as duplicate
  physical edges; the in-group resolution rule for coplanar-group edge
  duplicates follows the existing selected-edges semantics. If the exclusion
  epsilon is wrong the symptom is doubled cid-7 rows on straight polylines -
  covered by a direct contract test.
- **Misattribution guard**: the anti-phase mechanism was misdiagnosed once
  (see Context). The acceptance protocol therefore requires per-path
  decomposition evidence at gate G-A cells, not just field-level metrics.

## Revisit condition

Revisit when P4 lands the two-variable transition function (upgrade the
leg-2 call site), when P3 adds R->R->D (shares the discovery streaming), or
if a future scene class needs diffraction order > 2 (new ADR).
