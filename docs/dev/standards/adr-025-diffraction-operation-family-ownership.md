# ADR-025: Diffraction operation-family ownership

- **Status:** Accepted (2026-07-19)
- **Date:** 2026-07-19
- **Kind:** Move-only numerical ownership, semantic naming, and native-boundary
  decision. This ADR does not authorize a diffraction-physics change, a new
  estimator, a winner-selection derivative, or a fusion/launch change.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-004 (numerical duplication), ADR-009 (native fusion ownership), ADR-012
  (stationary coupled diffraction), ADR-013 (double diffraction), ADR-018
  (BDPT standalone diffraction realignment), ADR-023 (direct typed RayD
  integration), and ADR-024 (shared RF ownership).

## Context

The word "diffraction" currently names several operations with different
inputs, outputs, tapes, numerical order, compilation modes, and solver policy.
Treating them as one family would split complete primal/JVP/VJP owners or move
Channel-specific estimators and fused coupled rows into RayD merely because
they call a generic UTD primitive.

RayD already owns order-1 OptiX path export and visibility. Channel currently
owns a fixed-winner pure-wedge field reevaluator that reproduces the same
generic UTD coefficient, including its backward and JVP companions. That
three-entry family is the duplicate generic runtime owner that should move.
The MC UTD fixed-tape estimator and coupled R-D/D-D field operations are
different complete operations: extracting their UTD subexpressions would add
launches and materialized intermediates and would break their AD lockstep.

The current surface also retains one misleading BDPT-named RayD sample-tape
producer and a live Python/Torch transmitter visibility prefilter. Plan 13
Phase 4 already deleted the other dead diffraction bindings after a four-axis
reachability audit. Phase 7 must freeze the final ownership and cleanup
contract before either repository changes numerical source ownership.

## Decision

### 1. Operation-family matrix

The following matrix is authoritative. An owner applies to the complete
operation family, not to a substring in a symbol name.

| Operation family | Complete entries or boundary | Authoritative numerical owner after activation | Decision |
|---|---|---|---|
| order-1 path export / visibility and sample-tape production | `rayd_diffraction_paths_order1_forward`; renamed sample-tape producer | RayD | Keep exporter and producer distinct typed operations; discrete winner selection is not differentiated. |
| pure-wedge fixed-winner field | `field_diffraction_wedge`, `field_diffraction_wedge_backward`, `field_diffraction_wedge_jvp` | RayD | Move all three entries together in Phase 8A; retain Channel fields/autograd facades and stable `_channel` names. |
| MC UTD fixed-tape estimator | `mc_utd_diffraction_tape_accumulate`, backward, JVP | Channel | Keep the three-entry estimator whole. Do not extract a UTD sub-launch. |
| coupled R-D field | `field_coupled_rd`, backward, JVP | Channel | Keep reflection-slab plus UTD row fusion whole; consume RayD public device primitives. |
| coupled D-D field | `field_coupled_dd`, backward, JVP | Channel | Keep the two-wedge one-launch row fusion whole; consume RayD public device primitives. |
| coupled R-D stationary geometry | `coupled_rd_prepare`, backward, JVP | Channel | Keep the continuous stationary-geometry family whole. A future move needs a separate ADR and evidence. |
| composed R-D/D-D geometry | `coupled_rd_geometry_forward`, `coupled_dd_geometry_forward` | Channel operation / RayD primitives | Channel owns prepare/finalize and row semantics; RayD owns typed EPC/visibility primitives. |
| deterministic/path/MC packing, compaction, vector accumulation, and edge discovery | current `deterministic_*`, `path_diffraction_block`, `mc_diffraction_*` operations | Channel | Keep solver/propagation policy in Channel. |
| BDPT connection, PDF, MIS, and storage | no dedicated live diffraction ABI after Phase 4; standalone diffraction uses ADR-018 enumerated paths | Channel, only when a real BDPT caller exists | Do not recreate deleted bindings. Any future entry needs a real BDPT E2E caller and complete governance coverage. |

The machine-readable copy is
`docs/dev/audit/phase13-diffraction-family-matrix.json`. It records the current
Phase 7 owner separately from the accepted post-activation owner so accepting
this ADR cannot be mistaken for completing Phase 8A.

### 2. Pure-wedge family moves as one exact owner

RayD first lands a pushed, dormant typed candidate for the three pure-wedge
entries. Channel then performs one atomic Phase 8A pin/switch/delete commit:

1. pin the pushed RayD commit and integration-header identity;
2. dispatch all three stable Channel ABI entries to the typed `rayd::torch`
   family;
3. delete the Channel numerical implementation and its CMake source entry; and
4. update the binding, coverage, current-owner, duplication, launch/resource,
   migration, and no-fallback records together.

There is no forwarding source, runtime feature flag, fallback, second compiled
owner, or interval in which primal and derivative entries have different
owners. Rollback is a lock-pin rollback to the preceding complete owner.

The move preserves `wedge_row_eval<T>` expression and evaluation order,
optional winner vertices, fixed-winner geometry AD, material/frequency/source/
target/winner-vertex leaves, request gating, output names/shapes, empty-row
behavior, exception timing, and the existing three-launch contract: one launch
for each active primal, backward, or JVP call and no launch for zero rows. The
RayD order-1 exporter remains a separate discovery/export operation; fusing it
with the fixed-winner reevaluator is a numerical/fusion change outside this
ADR.

### 3. Fast-math and precise-math boundary

The pure-wedge family remains compiled with `--use_fast_math` after its move.
It is locked to RayD's OptiX order-1 exporter, which evaluates the same UTD
path with fast math. Phase 8A must compare compiler commands, PTX/SASS,
registers, occupancy, launch geometry, and exported-field parity before
activation.

Fast math does not spread across the ownership boundary. The MC UTD
fixed-tape family, coupled R-D/D-D fields, `coupled_rd_prepare`, transmission,
and other precise families remain precise. Scattering's separate `--fmad=false`
contract does not apply to diffraction. A compiler flag inherited from a
target or moved translation unit that changes any of these boundaries stops
the migration.

### 4. Channel-retained complete families

The MC UTD primal/backward/JVP family owns one fixed-tape estimator contract:
proposal and Jacobian terms, finite-thickness slab response, cell atomics, RNG,
map schema, and output exactness. RayD's sample-tape producer is an upstream
opaque producer, not the primal of this Channel derivative family.

The coupled R-D and D-D families retain their complete fused row operations,
including ADR-012 stationary external-incident behavior and ADR-013 two-wedge
ordering. Their backward/JVP companions remain in the same owner, use the same
frozen winner geometry, and may include RayD shared UTD/RF device headers. They
must not call a new pure-wedge ABI from inside the row or materialize a UTD
intermediate. `coupled_rd_prepare` likewise remains a complete Channel
primal/backward/JVP family. These decisions preserve launch count, numerical
order, tape lifetime, and solver schemas.

### 5. Sample-tape producer has a semantic name

The live `bdpt_diffraction_accumulation_forward` binding is a RayD fused
sampling/visibility tape producer called by MC Basic. It is not a BDPT
operation and it is not the MC UTD estimator primal. Phase 8B renames the
native binding and owning facade to `rayd_diffraction_sample_tape_forward`.
The old name is deleted in the same commit; no alias or re-export remains.

This is rename-only. The RayD implementation, launches, random consumption,
row order, and complete output tuple remain exact. Unconsumed map columns are
not trimmed. Converting the producer into a tape-only kernel would change the
fusion/resource contract and requires its own profiler-backed ADR.

### 6. Legacy deletion and reachability

The Phase 4 deletion audit remains authoritative for
`bdpt_diffraction_connection_samples_from_tape`,
`bdpt_diffraction_point_connection_samples`,
`bdpt_diffraction_state_pack`, `bdpt_diffraction_state_wi`, and
`bdpt_diffraction_edge_geometry`. They failed all four required axes and must
not be reintroduced:

1. static production caller;
2. dynamic binding/registry lookup;
3. stable public import; and
4. real BDPT end-to-end execution.

The Phase 4 `mc_diffraction_discover_edges{,_counted}` rename is final and its
Channel owner is retained. Phase 8B must leave zero live
`raydn_diffraction_*` facades and zero instances of the old sample-tape name.
Any other `bdpt_diffraction_*` entry is retained only if all four axes prove a
real BDPT owner/caller. Name-based contract coverage is not reachability. A
dead entry is deleted with its facade, registration, source/helper, tests,
manifest rows, coverage row, inventories, and budgets in the same commit.

The closed and pending records are frozen in
`docs/dev/audit/phase13-diffraction-legacy-audit.json`.

### 7. Transmitter-visible state selection becomes one native operation

Phase 8B deleted `propagation.geometry.diffraction._tx_visible_diffraction_states`
and its Torch loop/compaction path. The Channel-owned composed native planning
entry `diffraction_tx_visible_state_plan` now calls the typed RayD axial-edge
visibility primitive directly; Python only validates and assembles the named
result.
ADR-028 supersedes this ADR's provisional compact-selection storage contract:
the accepted result preserves the twelve input tensors as exact aliases and
returns a device-resident active mask at capacity `N`, avoiding a host-visible
selected count.

The operation freezes these exact semantics:

- fractions are ordered `(0.02, 1/3, 2/3, 0.98)` and evaluated as
  `t_min + fraction * (t_max - t_min)`;
- a state survives if any of its four source-to-edge sample segments is
  visible, with no ignored face;
- active rows preserve input row identity/order and all twelve row-aligned state
  fields remain exact object/storage aliases;
- empty input returns the established correctly shaped CUDA result without a
  launch; dtype/device/shape/resource errors fail before a success result;
- the caller's active CUDA stream is used and no host scalar extraction,
  synchronization, Python loop, Torch geometry/reduction, CPU path, or fallback
  remains.

Phase 8B acceptance proves the ADR-028 capacity/mask storage contract and that
solver-visible values and row order are exact. The implementation may not hide
a new persistent tape, host wait, device transfer, or unbounded peak-memory
regression. Phase 12 records the intentionally changed inactive-lane/launch
behavior and may replace composed visibility with one typed native launch only
with profiler evidence.
If the complete operation cannot meet those resource and exactness gates, the
Phase 8B migration stops; the Torch implementation is not accepted as fallback.

## Acceptance evidence

### Phase 7 decision evidence

- The machine family matrix contains exactly the nine rows above and records
  complete primal/backward/JVP sets.
- The legacy audit carries forward the Phase 4 four-axis deletions, the
  completed MC discovery rename, and the one pending sample-tape rename.
- Static governance tests prove this ADR is accepted, the current Phase 7
  implementation has not been moved, only the pure-wedge translation unit has
  the diffraction fast-math flag, and `AGENTS.md`/`CLAUDE.md` are identical.
- Plan 13, the propagation owner README, feature list, and migration note point
  to this decision without claiming Phase 8A activation.

### Required Phase 8A evidence

- exact pure-wedge parity against RayD-exported `field_xyz`, including empty,
  batched, boundary, ISB/RSB, finite-edge, and optional-winner cases;
- material, frequency, source, target, and winner-vertex JVP/VJP lockstep,
  adjoint dot-product tests, and test-only finite-difference oracles;
- ADR-012/013 coupled R-D/D-D primal/JVP/VJP regression and MC fixed-tape
  seed/RNG invariance, proving retained families did not change;
- compiler command, fast-math, PTX/SASS, register, occupancy, active-stream,
  launch, synchronization, copy, peak-memory, and packaging evidence; and
- missing symbol/capability, ABI, dtype/shape/device, unsupported-SM, and
  no-fallback negatives plus Path, Deterministic, MC Basic, and BDPT E2E.

### Required Phase 8B evidence

- exact rename manifest/coverage/E2E evidence with zero old aliases and
  unchanged sample-tape output/launch/RNG records;
- the four-axis audit for every remaining legacy diffraction candidate;
- exact four-fraction visibility and active-mask/active-row parity for empty,
  all-visible, partially visible, accepted passthrough views, rejected
  non-contiguous numerical inputs, invalid dtype/shape/device, capacity-limit,
  and current-stream cases; and
- static proof that production Python contains no transmitter visibility
  geometry loop, host Boolean extraction, or Torch numerical reconstruction.

Run the relevant `quick` and `cuda` tiers for both implementation phases and
the `nightly` performance/codegen evidence required by Plan 13. Tests,
tolerances, launch/resource budgets, manifests, and allowlists are not widened
to obtain acceptance.

## Stop conditions

Stop Phase 8A or 8B before activation if any of the following occurs:

- a family is split across numerical owners, or both repositories compile a
  production implementation;
- pure-wedge expressions, evaluation order, output schema, fixed-winner AD,
  fast-math codegen, launch/resource behavior, or exported-field parity differ
  without a separate accepted numerical ADR;
- an MC or coupled family gains a UTD sub-launch, materialized intermediate,
  persistent tape, RNG change, or precise/fast compilation drift;
- a rename leaves an alias or trims/recomputes the sample-tape output;
- a deleted/retained legacy decision lacks all four reachability axes;
- transmitter state selection changes the four fractions or stable rows, or
  retains Torch/CPU geometry, host synchronization, or a fallback; or
- the RayD candidate is not pushed and lockable by exact commit/header
  identity, or required static/CUDA/nightly/package evidence is incomplete.

## Consequences

RayD becomes the sole generic pure-wedge numerical owner only after the atomic
Phase 8A activation. Channel remains the complete owner of MC UTD and
coupled diffraction operations and of solver packing/policy. Phase 8B then
removes misleading legacy identity and the production Torch visibility
geometry without conflating tape production with estimator evaluation. The
accepted boundary prevents a directory- or name-based migration from splitting
fusion and AD contracts.
