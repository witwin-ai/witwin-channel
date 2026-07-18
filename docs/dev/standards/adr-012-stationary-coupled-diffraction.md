# ADR-012: Stationary-path coupled reflection-diffraction leg

- **Status:** Proposed (draft; acceptance evidence in progress). Companion to
  ADR-011 (deterministic coupled paths), which recorded this as its own required
  follow-up numerical ADR.
- **Date:** 2026-07-17
- **Kind:** Numerical-kernel change (fullwave-ground-truth G4). It changes the
  coefficient the coupled reflection-diffraction leg carries; coupled-off solves
  stay byte-identical.
- **Related:** ADR-011 (deterministic coupled paths, "Risks" section), ADR-010
  (native scattering kernels), ADR-009 (native fusion ownership), ADR-004
  (numerical duplication), the F5c/d/e + G1/G2 continuity design in
  `docs/dev/audit/utd-continuity-fix-design.md`, and the G3 verify artifacts
  under `artifacts/fullwave-fix/verify-g3/`.

## Context

ADR-011 turned on the coupled reflection-diffraction compensator (component ids
3 and 4) for the deterministic grid solver and accumulated it coherently. That
closed the dominant RSB class (the *missing* compensator). Its "Risks" section
recorded a residual: the coupled diffraction leg ran with
`selectStationaryPoint = 0` and pseudo-infinite edge bounds
(`edgeLineMin/Max = -/+1e5`) in both the primal
(`field_transport.cu::coupled_rd_field_kernel`) and the AD twin
(`field_wedge_ad_coupled.cu::coupled_rd_row_dual`). On that path the E1c/d/e
corner-mend, the G1 monotone even truncation, and the G2 boundary-distance blend
are all inert (they are stationary-path-only), so the coupled leg carried the
plain truncated infinite-wedge coefficient and injected its own extension-plane
("shared-plane") steps.

The G3 verifier quantified the residual: the flagship occlusion-RSB compensation
worked (33.2 -> 11.5 dB at the grid, 40.7 -> 8.1 dB at 0.5 mm, 19/24 occlusion
pairs improved), but the coupled leg's own discontinuities regressed the order-1
RSB failure fraction (21.4 -> 24.1 %) and added genuine non-toggle steps.

## Decision

Give the coupled R->D / D->R diffraction leg the stationary-path physics so it
inherits the same continuity machinery as order-1 diffraction. Three parts.

### G4-1: external-incident stationary mode (shared RayD header)

`PairInputsT` gains a discrete `float stationaryExternalIncident` flag (default
0, so every existing order-1 and Monte Carlo caller is bit-identical). In the
vector path (`compute_pair_vector_at_angles` /
`compute_pair_vector_contribution_no_completion`), when
`selectStationaryPoint > 0.5` AND the flag is set, the full stationary machinery
runs (edge re-anchoring via `pair_state_at_stationary_point`, monotone even
`finite_wedge_monotone_truncation`, `corner_mend_gamma`, and the G2 blend), but
the incident field is NOT `direct_source_vector`. Instead the frozen EXTERNAL
`incidentJones` (the coupled kernel's image-source spherical wave, projected onto
`incidentBasis` at the original edge point) is re-extrapolated from the frozen
edge point to the re-anchored stationary point Q*:

```
scale = (sPrimeFrozen / sPrimeNew) * exp(-j k (sPrimeNew - sPrimeFrozen))
sPrimeFrozen = |sourcePos - edgePos_original|   (captured before re-anchoring)
sPrimeNew    = |sourcePos - Q*|
```

The composite amplitude/phase become `1/(2 k sPrimeNew)` and `e^{-j k sPrimeNew}`
exactly, so the re-extrapolation is EXACT for the spherical image-source wave the
coupled kernel constructs. The reconstructed vector keeps the original
polarization direction (frozen; same approximation order as the frozen txPol in
`direct_source_vector`) and is reprojected onto the re-anchored incident basis by
the existing `jones_from_vector`. `sPrimeFrozen` is captured in
`compute_pair_vector_contribution_no_completion` before the re-anchor and threaded
to `compute_pair_vector_at_angles` as a new defaulted parameter.

### G4-2: coupled kernels run the stationary path with real bounds

`coupled_rd_field_kernel` and its AD twin set `selectStationaryPoint = 1` and
`stationaryExternalIncident = 1`, and receive real per-row `edge_line_min` /
`edge_line_max` (offsets of the edge-segment endpoints from the passed edge point
along the edge axis), replacing the `-/+1e5` infinite bounds. The owning facade
(`propagation/fields/evaluation.py::_evaluate_coupled_fields`) computes the bounds
host-side from the edge tables: the tables carry `line_min/line_max` relative to
the segment reference origin, shifted by the Keller point's arc offset. The bounds
are frozen (detached): coupled rows carry no edge-geometry gradient (ADR-011
mesh-vertex exclusion), and the tx/rx gradient flows through source/target inside
the native re-anchoring.

`diffraction_source` is verified consistent with the stationary geometry: forward
R->D unfolds the transmitter across the reflection plane (image source), reverse
D->R keeps the real transmitter and unfolds the receiver; in both cases the frozen
incident is a genuine spherical wave from `diffraction_source` to the edge, so the
re-extrapolation is exact. The slab face operators (frozen at the Keller point via
`omega < 0`) and the reverse-direction reflection are unchanged.

### G4-3: ABI, AD lockstep, governance

`cn_field_coupled_rd`, `cn_field_coupled_rd_backward`, and
`cn_field_coupled_rd_jvp` each grow two tensor arguments (`edge_line_min`,
`edge_line_max`) appended before `frequency_hz`. No new pybind symbol is
introduced, so the 174-binding baseline is unchanged; the binding manifest
baseline, the owner-inventory body hashes, and the Python facade/autograd wrappers
move together. The AD twin now calls `compute_pair_vector_contribution` directly
(like the order-1 diffraction dual), so the T_mono / gamma / B derivatives flow
through the dual in lockstep with the primal; the former manual frozen-finite-
factor inline is removed.

## Rationale

The continuity machinery is a single production-native implementation in the
shared RayD header. Reusing it via a flag (rather than duplicating a coupled-only
truncation) obeys the no-duplicate-physics rule and keeps the primal and the dual
in lockstep. The external re-extrapolation is the physically exact continuation of
the same image-source spherical wave the coupled kernel already builds, with the
same direction-frozen polarization approximation the leg already made.

## Acceptance evidence

1. **Host probe (float + Dual)**, `artifacts/fullwave-fix/verify-g4/probe.cpp`:
   with the external `incidentJones` set to `direct_source_vector`'s spherical
   wave at the frozen edge point and an interior config where Q* == E0, the
   external-incident stationary mode reproduces the plain stationary mode to
   relerr 8.6e-08 (float) and 9.6e-08 (Dual value + tangent). The flag is inert on
   the Monte Carlo (non-stationary) path (bit-identical), and the flag=0
   stationary and MC outputs are bit-identical between the pre-G4 and G4 headers.
2. **Coupled AD FD lockstep**: `tests/ad/test_solver_diffraction_coupled_ad.py`
   is green, including `test_coupled_endpoint_position_grad_matches_fd` (tx and
   rx), so jvp/vjp still match central differences of the new primal.
3. **Coupled-off byte identity**: coupled-off deterministic / order-1 diffraction
   / MC paths use `stationaryExternalIncident = 0` and are unchanged (host-probe
   bit-identity + `tests/deterministic/test_field_continuity.py` green).
4. **Three-cube occlusion-RSB** (`artifacts/fullwave-fix/verify-g4/g4_summary.json`):
   the coupled leg's own extension-plane steps are reduced toward the coupled-off
   baseline while the flagship compensation is retained. Order-1 RSB failure
   fraction: 21.45 % (off) -> 24.06 % (G3 on) -> **21.88 %** (G4 on). No-toggle
   failure fraction: 3.62 -> 4.50 -> **4.09 %**. NOTOG genuine steps: 3889 -> 5056
   -> **4543**. On the flagship row (y = 0.457) the max adjacent step drops
   33.2 dB (off) / 24.1 dB (G3 on, an injected extension-plane step next to the
   compensated RSB) to **11.6 dB** (G4 on); the same 8 of 24 occlusion pairs are
   brought below 3 dB as G3.

## Consequences

- Coupled rows carry the mended (stationary, truncated, corner-blended)
  coefficient instead of the plain infinite-wedge one. Coupled-off solves are
  byte-identical to ADR-011.
- The coupled field ABI grows two tensor arguments (no new symbol). The binding
  baseline and owner inventory are updated in lockstep.

## Risks

- The polarization direction is frozen at the original edge point and the edge
  bounds are frozen for AD; both are the same approximation order as the coupled
  leg's existing frozen-Jones treatment and are exact for the amplitude/phase of
  the spherical image-source wave. A fully live coupled polarization / edge
  gradient remains a documented future refinement.
- Coupled diffraction order stays 1 and reflection depth inside a coupled path
  stays 1 (ADR-011). Raising either is an independent numerical change.
