# ADR-011: Deterministic coupled reflection-diffraction paths

- **Status:** Proposed (draft; acceptance evidence pending)
- **Date:** 2026-07-17
- **Kind:** Numerical-kernel + accumulation change (fullwave-ground-truth G3 / audit
  M1). This is NOT a move-only architecture change: it adds coherent coupled
  reflection-diffraction contributions to the deterministic grid solver's field
  totals, which changes deterministic output for scenes with coupled paths.
- **Related:** ADR-007 (propagation data ownership), ADR-008 (enumerated
  propagation as the shared oracle), ADR-009 (native fusion ownership), ADR-010
  (native scattering/C_r kernels), Plan 05 section 6 (coupled contract), Plan 08
  section 2 (native extension is the only production compute backend). Companion
  audit: `docs/dev/audit/fullwave-deterministic-discontinuity-audit.md` and the
  three-cube multipath findings under `artifacts/fullwave-fix/threecube/`.

## Context

The single-cube UTD-continuity work (F1-F6, ADR-adjacent, RayD branch
`fix/utd-continuity`) closed the order-1 ISB/RSB and shadow-interior
discontinuities. The three-cube multipath validation exposed a residual class the
order-1 fixes cannot reach: **reflection shadow boundaries that need a
diffraction-of-the-reflected-field compensator**.

The flagship is the RSB1 bucket in
`artifacts/fullwave-fix/threecube/jump_summary.json`
(`buckets.RSB1.max_db = 28.505` dB, 134 pairs, median 5.38 dB, p95 15.4 dB). At
the flagship cell near (0.049, 0.457) cube2's specular reflection is occluded by
cube3's silhouette (`h1_occlusion.json`: `H1_occlusion = 23`, of which
`occlusion_cube3 = 18`). Geometric optics toggles the reflected ray off in one
cell with no compensating term, producing a hard 28.5 dB step. The physical
compensator is the coupled path **TX -> reflect(cube2) -> diffract(cube3 edge)
-> RX** (component id 3, "reflection_diffraction") and its reciprocal
**TX -> diffract -> reflect -> RX** (component id 4, "diffraction_reflection").
Order-1 edge diffraction (component id 2) diffracts the *direct* field and
compensates the *direct*-field ISB; it cannot compensate a *reflected*-field RSB.

The coupled physics already exists and is production-native. The path solver
enumerates, evaluates, and exports coupled rows today:

- Discovery: `propagation/topology/discovery/coupled.py`
  (`prepare_coupled_candidate_plan`, `iter_coupled_candidate_requests` emit
  component ids 3 and 4) and `propagation/enumerated/coupled.py`
  (`_coupled_reflection_diffraction_topology_order2`).
- Geometry: `propagation/geometry/coupled.py` -> native
  `raydn_coupled_rd_geometry_forward`.
- Field: `propagation/fields/evaluation.py::_evaluate_coupled_fields`
  (lines 529-715) -> native `field_coupled_rd` (+ `_backward`, `_jvp`,
  `coupled_rd_prepare` companions). These native symbols are owned, manifested,
  and contract-covered (`ci/contract-coverage-manifest.json`:
  `native-field-coupled`, `field-coupled`, `path-coupled`).
- The shared enumerated engine already wires coupled: `engine.py:94` reads
  `getattr(config, "coupled_paths", False)`; `engine.py:177-195` enumerates cid
  3/4 when `coupled_paths and max_depth >= 2 and {reflection, diffraction}
  subset components`; `evaluate_path_fields` calls `_evaluate_coupled_fields`
  unconditionally (evaluation.py:925), gated only by the presence of cid 3/4
  rows, not by the `components` set.

Two gaps keep this off the deterministic grid solver:

1. **Config gate.** `deterministic/config.py` has no `coupled_paths` /
   `coupled_candidate_limit` fields, so `engine.py:94`'s `getattr` defaults to
   `False` and no cid 3/4 rows are ever enumerated for a deterministic solve.
2. **Accumulator drop.** Even if cid 3/4 rows existed and their fields were
   evaluated (they would be, through the shared engine), the deterministic flat
   accumulator discards them. `kernels/deterministic_accum.cu::accum_slot`
   (lines 32-43) maps cid 0/1/2 -> slot 0/1/2, cid 5 -> slot 3, cid 6 -> slot 4,
   and returns **-1 for cid 3/4** ("consumed per path by the path API, never
   materialized here"). Slot -1 rows are dropped by the scatter/gather gates. So
   the coupled compensation is computed and then thrown away before it reaches
   `power_total` / `field_total` / `component_power`.

The path solver avoids gap 2 because it exports per-row `PathResult` fields
(`path/result.py::from_evaluated_paths`) and never runs the cell accumulator; cid
3/4 are named "reflection_diffraction"/"diffraction_reflection" in
`path/pipeline.py::_COMPONENT_ID` and flow through as ordinary rows. The
deterministic grid solver's whole job is the cell accumulation, which is exactly
where the coupled rows are lost.

## Decision

Enable the existing shared coupled enumeration for the deterministic grid solver
and accumulate its coherent contribution. Four parts.

### D1: enable coupled enumeration on the deterministic Config

Add two fields to `deterministic/config.py::Config`, mirroring
`path/config.py:32-33` verbatim:

```python
coupled_paths: bool = False
coupled_candidate_limit: int = <default, see D4>
```

Validation (mirror `path/config.py:55-73`): when `coupled_paths` is set, require
`max_depth >= 2` and `{"reflection", "diffraction"}.issubset(components)`;
require `0 < coupled_candidate_limit <= _MAX_COUPLED_CANDIDATES`. The engine gate
already consumes both fields via `getattr` (verified: `engine.py:94`, `:177-195`,
`:188-190`), and `sequence_width = max(max_depth, 2)` at `engine.py:95` already
reserves the two-interaction row width. No engine signature change is required
for D1; adding the fields is sufficient to turn enumeration on.

### D2: coherent accumulation of ids 3/4 in the native flat accumulator

The coupled coefficient is a coherent complex-3 Jones contribution
(`_evaluate_coupled_fields` fills `path_field` = receiver-projected scalar,
`field_xyz`, `path_gain`, exactly like reflection/diffraction). It must join the
coherent field sum. Map cid 3 and cid 4 into **one new coupled field slot** in
`kernels/deterministic_accum.cu`:

- `kAccumSlotCount`: 5 -> 6; add `constexpr int kCoupledSlot = 5;`.
- `accum_slot`: add `if (component_id == 3 || component_id == 4) return
  kCoupledSlot;` (before the final `return -1;`). Both directions land in slot 5
  and sum coherently in-cell: `component_field[5] = E_RD + E_DR`.
- The coupled slot is a normal coherent field slot. The finalize's power-domain
  special case stays bound to `kScatteringSlot == 4`; slot 5 falls through to the
  coherent `real*real + imag*imag` branch and adds into `real_sum`/`imag_sum`,
  so `field_total`/`power_total` pick it up automatically. The
  backward/jvp/fwd64 companions share `accum_slot`, so they inherit the mapping;
  the jvp gain-tangent gate `(!coherent || slot == kScatteringSlot)` correctly
  excludes slot 5 in coherent mode (coupled is a field slot, not a power slot).

Python owner side (`deterministic/accumulation.py`):

- `_NATIVE_COMPONENT_SLOTS`: add `"coupled": 5`.
- Expose "coupled" as an exported component when `coupled_paths` is set (see D3).

Single slot (not two) because R->D and D->R are the same physical component,
coherently summed at each cell; the finalize sums all field slots identically
whether they share one slot or two, so a second slot only fragments the
diagnostic without changing the total. This keeps the accumulator growth minimal
(one slot) and the public component set to a single new "coupled" key.

### D3: component / export semantics — new "coupled" slot, NOT folded into diffraction

Two candidate mappings for the accumulator slot were analyzed.

**Option A — fold cid 3/4 into the diffraction slot (slot 2).** Rejected. It is
inconsistent with the pipeline's exact-power swap for the diffraction component
(`deterministic/pipeline.py:271-283`). That swap replaces
`component_power["diffraction"]` with `|diffraction_vector_field|^2`, where
`diffraction_vector_field` is the per-cell coherent vector sum built **only from
order-1 D rows** in `_diffraction_topology_order1`
(`enumerated/diffraction.py:129-217`); the coupled field kernel is a different
launch and contributes nothing to that buffer. Folding therefore breaks in two
ways:

- Coherent mode: the native slot-2 field sum would include the coupled rows
  (correct for the total), but the swap overwrites `component_power["diffraction"]`
  with the order-1-only vector power, silently erasing the coupled contribution
  from the component diagnostic.
- Incoherent mode: `pipeline.py:282-283` does
  `path_gain = path_gain - previous_diffraction + exact_diffraction`, where
  `previous_diffraction` is the native slot-2 power (order-1 + coupled) and
  `exact_diffraction` is the order-1-only vector power. The coupled power is
  subtracted out of the total and never added back — a real total-power bug.

The swap cannot be made consistent cheaply: making it coupled-aware would require
`diffraction_vector_field` to also accumulate the coupled per-cell vectors, but
those vectors are produced by `field_coupled_rd` downstream in
`evaluate_path_fields`, across the enumerated/fields boundary from where the
order-1 vector field is built. In coherent mode the native slot-2 power
`|sum(order1 + coupled)|^2` is not additively separable into order-1 and coupled
parts, so no post-hoc correction recovers the split. Making the swap consistent
therefore requires tracking order-1 and coupled powers separately — which is
exactly a separate slot. Folding buys nothing and adds a bug.

**Option B — new "coupled" slot (slot 5).** Recommended and specified in D2. The
diffraction slot stays pure order-1, so the exact-power swap
(`pipeline.py:271-283`) is untouched and remains valid. Coupled gets its own
`component_power["coupled"]` / `component_fields["coupled"]` diagnostic, and the
coherent total naturally includes it. Consumers:

- `deterministic/pipeline.py`: extend `extra_components` (currently
  `deterministic/pipeline.py:254-256`, the optional-component tuple) to append
  `"coupled"` when `config.coupled_paths`, so
  `accumulate_flat_components::exported_names` (`accumulation.py:114`) surfaces
  the "coupled" key. Add a coupled path-count to the metadata component counts
  (mirror the transmission/scattering count at `pipeline.py:249-253` with
  `cid in (3, 4)`); the native `deterministic_component_counts` only counts cid
  0/1/2 and needs no change.
- `deterministic/pipeline.py::_metadata`: set the deterministic
  `semantic_capabilities` coupling flag true (see D-capabilities below) and add a
  `coupled_paths` metadata block mirroring `path/metadata.py:129-137`.
- `benchmarks/fullwave_validation/backends.py`: add `"coupled"` to `_COMPONENTS`
  (line 12) so the benchmark requests coupled and exports the coupled component
  map. `solve_deterministic` already forwards `result.component_fields` for every
  component with `numel > 0`, so the coupled component appears automatically once
  requested.

Public-API and capability surface:

- `capabilities.py:126-136` (`solvers.deterministic`): flip
  `supports_reflection_diffraction_coupling` False -> True and add the coupling
  geometry/topology/candidate-limit keys that the path solver already carries
  (`capabilities.py:111-115`). This changes the `capabilities` return value; the
  `capabilities` export has a `contract_sha256` in
  `ci/public-api-snapshot.json`, so the snapshot needs an intentional update and
  migration note.
- The deterministic `Config` contract hash
  (`ci/public-api-snapshot.json`, `witwin.channel_native.deterministic` ->
  `Config` -> `contract_sha256`) changes because two fields are added. Intentional
  snapshot update + migration note required (ADR-003).

### D4: RX streaming for the 65 536-receiver grid

Measured for the three_cube 256x256 case (see Candidate scaling below): the full
grid needs **84.9M** coupled candidate evaluations, **84.9x** over the 1M budget
`_MAX_COUPLED_CANDIDATES`. `prepare_coupled_candidate_plan`
(`discovery/coupled.py:58-64`) raises before any launch when the theoretical
total exceeds `min(candidate_limit, _MAX_COUPLED_CANDIDATES)`, so a full-grid
deterministic coupled solve is currently impossible.

Memory is *not* the blocker: `iter_coupled_candidate_requests` already streams the
candidate index in chunks of 65 536, holding peak device memory to ~33 MB per
launch (measured) regardless of receiver count. The blocker is purely the
plan-level total-candidate guard. Decision:

- The deterministic engine streams coupled discovery over **receiver blocks**,
  mirroring the diffraction rx-chunk pattern already used at
  `enumerated/diffraction.py:161` (`iter_diffraction_rx_chunk_requests`, audit
  P-2). Each block calls the existing
  `_coupled_reflection_diffraction_topology_order2` with a receiver slice, offsets
  the returned `rx_id` by `rx_start` (exactly as diffraction does at
  `enumerated/diffraction.py:205-206`), and concatenates the blocks. Block size
  is derived so `rx_block * candidates_per_pair * 2 <= coupled_candidate_limit`.
- This keeps `coupled_candidate_limit` a **real per-block work/safety budget** (a
  city-scale scene with 10^4 edges still fails loudly per block instead of
  enumerating 10^13 candidates), preserves the shared discovery function's
  total-cap contract for the path solver unchanged, and bounds both the per-launch
  memory (~33 MB) and the per-block row buffer.
- Recommended deterministic default `coupled_candidate_limit`: keep the 1M hard
  cap semantics but pick a default that yields a reasonable block count. With
  `candidates_per_pair = 648`, a 1M budget gives 771 rx/block -> 86 blocks; an 8M
  budget (if the hard cap is raised for the deterministic solver in a follow-up)
  gives ~6 000 rx/block -> 11 blocks. The recon recommends starting at the
  existing 1M cap (86 blocks, ~5 s, no cap change, no path-solver impact) and
  treating a raised cap as an independent optimization.
- Do **not** relax the shared `_MAX_COUPLED_CANDIDATES` guard or change
  `iter_coupled_candidate_requests` semantics: those are shared with the path
  solver and changing them alters the path Config contract. The rx-block loop
  lives in the deterministic engine branch (`engine.py:177-195`) or a
  deterministic-owned helper, isolating the change.

## Rationale

Coupled physics is a single production-native implementation shared by discovery,
geometry, and field evaluation. Duplicating it or re-deriving a deterministic-only
coupled term would violate the monorepo no-duplicate-physics rule and ADR-008's
single-source-of-truth intent. The only deterministic-specific work is (a) turning
the shared enumeration on and (b) giving the coupled rows a home in the cell
accumulator. A new coupled slot keeps the diffraction exact-power swap intact,
preserves per-component diagnostics, and matches the path solver's separate
treatment of cid 3/4 at the physics level while collapsing them to one coherent
component at the grid level (where only the coherent sum is observable).

## Acceptance protocol (numerical evidence to freeze BEFORE merge)

Per CLAUDE.md, a numerical/output change needs its own ADR (this) plus acceptance
evidence. Collect:

1. **Bitwise-unchanged baseline for coupled-off cells.** Every deterministic,
   path, and MC cell that does not request `coupled_paths` MUST be bitwise
   identical to the pre-change runtime exact-hash profile. The accumulator slot
   addition must not perturb slots 0-4; assert component_power/field slots
   0-4 and totals unchanged on the existing `tests/kernels` accumulator oracle and
   on a coupled-off deterministic grid solve.
2. **Accumulator oracle extended.**
   `tests/kernels/test_ops_facade.py::test_deterministic_accumulate_flat_matches_torch_reference`
   (lines 2071-2122) currently pins the 5-slot layout (`torch.zeros((5, ...))`,
   `slot_of = {0:0,1:1,2:2,5:3,6:4}`, `expected_field_total =
   component_field[:4].sum`). Extend to 6 slots: `slot_of` adds `3:5, 4:5`, add
   cid 3/4 input rows, `expected_field_total = component_field[[0,1,2,3,5]].sum`
   (include coupled slot 5, exclude scattering slot 4), coupled component power =
   `|E_RD + E_DR|^2`. This is the direct contract test for the new slot.
3. **Coupled continuity at the flagship.** Deterministic line scans across the
   RSB1 boundary near (0.049, 0.457) with and without `coupled_paths`: assert the
   coupled-on max adjacent |E_total| jump drops materially from the 28.5 dB
   baseline and that no coupled-on cell introduces a NEW > 3 dB step that the
   order-1 continuity regression (design F6) would catch. Quantify the residual
   against the Maxwell three-cube reference.
4. **No-fallback negative tests (repo rule).** Assert that a coupled deterministic
   solve raises loudly (no Torch/CPU/zero-tensor fallback) when: RayD native is
   unavailable; `coupled_paths` is set with `max_depth < 2`; `coupled_paths` set
   without both reflection and diffraction components; the per-block candidate
   budget is exceeded. Mirror the path solver's negative tests
   (`tests/path/test_path_reflection_diffraction_sequences.py`) for the
   deterministic solver.
5. **AD lockstep unchanged.** `coupled_paths` under `ad_mode in {jvp, vjp}` must
   route the existing native coupled companions (`field_coupled_rd_backward/_jvp`,
   `coupled_rd_prepare_*`) and the extended accumulator backward/jvp; run
   `tests/ad/test_deterministic_accum_ad.py` and
   `tests/ad/test_solver_diffraction_coupled_ad.py` with the 6-slot accumulator.
   Mesh-vertex gradients through coupled rows stay a loud refusal
   (`_evaluate_coupled_fields:583-590`), matching the path solver's
   `coupled_paths_mesh_vertex` ad-exclusion.
6. **Manifest / owner governance.** No new pybind ABI symbol is introduced (the
   change is internal to `cn_deterministic_accumulate_flat` and its
   backward/jvp/fwd64), so the 174-symbol binding baseline
   (`docs/dev/baselines/.../static/bindings.json`, `EXPECTED_BINDING_COUNT = 174`)
   is unchanged; confirm the semantic projection still matches. Update the
   accumulator's cell-shape contract test and any resource sentinel that pins
   `kAccumSlotCount = 5`. Update `ci/public-api-snapshot.json` (deterministic
   Config + capabilities) and add the migration note.
7. **Launch-ledger delta documented.** Coupled adds ~1 296 native geometry
   launches + 1-2 coupled field launches for the full grid; record the new
   deterministic launch count in metadata and freeze the coupled-on ledger as a
   new baseline artifact (do not compare it against the coupled-off ledger).

## P1 full-wave arbiter decision (2026-07-17)

The plan-09 P1 arbiter compared coupled-OFF and coupled-ON against a
Yee-grid-coincident witwin-maxwell FDTD reference on the versioned
`three_cube_320` / `metal` case (recorded metrics and setup:
`docs/dev/fullwave-validation.md`, "Three-cube full-wave reference";
artifacts under `artifacts/fullwave/three-cube-metal-320/`).

**Decision: the benchmark default stays coupled ON**
(`benchmarks/fullwave_validation/backends.py`, `coupled_paths = max_depth >= 2`).

Evidence summary:

- FDTD shows **no step** at the flagship occlusion RSB (y about 0.457): the
  truth profile is smooth (-15 to -21 dB) where coupled-OFF carries a 39.3 dB
  adjacent-cell step and an entire reflected-shadow sector 35-53 dB below
  truth. Coupled-ON fills that sector to within 3-20 dB and caps the row jump
  at 9.2 dB. The compensator is real physics confirmed by the arbiter, not a
  smoothing device.
- Aggregate metrics are a statistical wash (NMSE 0.0899 OFF vs 0.0934 ON,
  coherence 0.8475 vs 0.8455): the coupled component's own
  enumeration-existence seams give back what the healed sector gains.
- The dominant ON liability is now precisely characterized: past the
  face-edge-exit existence boundary the coupled term degenerates into an
  **anti-phase, equal-magnitude duplicate of order-1 diffraction** (measured
  example at `(x=0.0531, y=0.4531)`: diffraction 3.174e-3 at +57.0 deg vs coupled
  3.174e-3 at -130.5 deg, total collapsing to 3.4e-6 — a -59.9 dB gap versus
  truth where OFF was within 0.5 dB), and ISB edges touching coupled-active
  cells carry +3.92 dB p95 excess versus +1.11 dB with OFF. Healing these
  sector edges is the P2 (D-to-D) acceptance gate; ON keeps the benchmark
  sensitive to that healing.
- Coupled-OFF remains bitwise identical to the pre-ADR-011 baseline, so
  single-cube continuity gates are unaffected by this default.

## Consequences

- The deterministic grid solver gains the reflection-diffraction coupling
  compensator, closing the M1 RSB class that order-1 UTD cannot reach. Coupled-off
  solves are byte-identical to today.
- One new accumulator slot (`kAccumSlotCount 5 -> 6`) and one new public component
  name ("coupled"). Deterministic Config and capabilities contracts change
  intentionally.
- The shared discovery/geometry/field coupled kernels are unchanged; the path
  solver is unaffected (its total-cap `coupled_candidate_limit` contract is
  preserved by keeping the rx-block loop deterministic-side).

## Risks

- **Coupled legs do NOT inherit the stationary G2 mend.** The coupled diffraction
  leg is evaluated with `selectStationaryPoint = 0` in both the primal
  (`field_transport.cu::coupled_rd_field_kernel:490`) and AD
  (`field_wedge_ad_coupled.cu:275`) kernels, using the plain
  `finite_wedge_truncation_factor`. The G2 odd-blend corner-mend and monotone
  even-part truncation (design F5c/d/e) are stationary-path-only
  (`selectStationaryPoint = 1`); the non-stationary path keeps gamma==1 semantics
  by design (design doc F5c "Known residual"). So coupled rows carry the plain
  truncated infinite-wedge coefficient and may retain a residual extension-plane
  step ("shared-plane disease"). This does NOT block the primary M1 win (the
  dominant 28.5 dB error is the *missing* compensator, fixed by accumulating it at
  all), but the coupled compensator's own continuity must be quantified in
  acceptance evidence step 3. If a coupled RSB itself shows a > 3 dB residual step,
  a stationary-path coupled evaluation (or an EEC-style coupled truncation) is a
  documented follow-up, its own numerical ADR. NOTE: this corrects the task
  premise that "coupled legs run the same stationary path" — they run the
  non-stationary path.
- **No R->D / order-1 double counting.** Order-1 D (cid 2) diffracts the direct
  field; coupled R->D (cid 3) diffracts the once-reflected field; D->R (cid 4) is
  the reciprocal order. These are distinct terms in the multi-interaction UTD
  expansion with distinct incident fields, Keller points, and path lengths, so a
  receiver on the RSB that also sits near a direct-field ISB legitimately receives
  both an order-1 D and a coupled compensation without double counting. Within
  coupled, R->D and D->R are distinct interaction orders (TX != RX, so not
  reciprocal duplicates) and coherently sum; each physical coupled path maps to a
  single (coplanar-group representative, edge) candidate resolved to one row by the
  native `surface_group_members` in-group resolution, so there is no intra-coupled
  duplication. Confirm empirically with a per-cell contribution audit at the
  flagship (evidence step 3).
- **Numerical-ADR requirement.** This draft IS the required ADR; merge is gated on
  the acceptance evidence above. Do not mix the accumulator slot change into an
  architecture-cleanup commit.
- **Brute-force candidate cost.** The discovery is O(rx * coplanar_groups *
  selected_edges) with no TX face/edge visibility prefilter, so candidates_per_pair
  = 648 for three_cube and the full grid is ~85M candidate evaluations (~5 s). A
  TX-visibility prefilter mirroring diffraction's native
  `diffraction_tx_visible_state_plan`
  would cut this several-fold and is a documented optional optimization, not
  required for the benchmark.
- **Public-surface churn.** Config + capabilities + public-api-snapshot +
  benchmark component list all move together; missing any one breaks a governance
  gate. Enumerated in the implementation map.

## Revisit condition

Revisit when: (a) a stationary-path (or EEC) coupled diffraction evaluation is
needed because the non-stationary coupled compensator shows a measurable residual
step at the RSB; (b) coupled diffraction order rises above 1 or reflection depth
inside a coupled path rises above 1 (currently `max_reflections_in_coupled_path =
1`); or (c) a TX face/edge visibility prefilter is added to bound the candidate
count for city-scale grids. Each is an independent numerical/architecture change
with its own evidence.

## Revisit condition (c) evaluated 2026-07-18: TX-visibility prefilter — NEGATIVE

Measured evidence (probes under `artifacts/ws2-perf/_cull_*.py`): a
TX-visibility cull of the coupled candidate axes is NOT conservative — the
shared (group, edge) candidate grid forces a symmetric shrink that removes
geometrically-valid SECOND-leg paths (edges lit by the reflected field,
groups lit by the diffracted field: exactly the compensator this ADR
exists to add). On three_cube_320 it deletes 50/56/43% of valid cid-3/4/7
rows and regresses the frozen P2 gates (RSB p95 excess +3.53 -> +5.22 dB,
NMSE 0.0923 > coupled-OFF). And it does not achieve city feasibility:
post-cull Munich/SF remain 30x/484x over the 1e6 per-receiver budget
because the cid-7 D->D term is quadratic in edges and edge2 is not
TX-prefilterable (a single receiver already exceeds the budget, so
rx-streaming cannot rescue it). Conclusion: no TX-visibility prefilter
ships. City-scale coupled requires a different candidate architecture -
receiver-tile-local edge sets or edge2-aware (edge-to-edge visibility)
pruning, and/or gating cid-7 at city scale - each its own numerical ADR
with re-frozen evidence.
