# Cell-State Memory Phase 2/3 Implementation Notes

Date: 2026-04-01

This note records the concrete implementation shipped for Phase 2 and Phase 3 of the cell-state memory plan.

## Scope

The Phase 2 / Phase 3 rollout landed the two structural changes that the memory plan called out as the highest-value state-side fixes:

- split propagation-hot state from optional cold metadata
- replace duplicated fixed history arrays with parent-link lineage

The public `Scene + Tracer + Result` architecture remains unchanged.

## Phase 2: Hot State vs Cold Metadata

Diffraction state arrays now distinguish between:

- hot propagation fields that are required for packing, pruning, replay input, and field accumulation
- cold metadata retained only for path replay, audit, and related diagnostics

The hot packed-state layout remains the authoritative representation for gather/concat/subset operations. Cold metadata is kept as optional side data instead of being treated as mandatory packed payload.

Cold metadata currently includes:

- `path_length_prefix`
- `first_interaction_pos`
- `is_direct_tx`
- `source_type_code`
- `approximation_mode_code`

Field-only diffraction preparation now disables cold metadata retention unless a caller explicitly asks for state-audit output. Path export and audit flows keep the cold metadata enabled.

## Phase 3: Parent-Link Lineage

Builders no longer write duplicated `path_edge_idx_*` and `path_reflection_depth_*` arrays into every child state on the active path.

Instead, each state now carries compact lineage metadata:

- parent state id
- current state id
- last edge id
- last reflection-depth delta

That lineage is finalized into a shared replay store after first-order build and after every higher-order expansion stage. When a complete path history is needed, the replay code reconstructs it lazily by walking parent links backward and reversing the collected events.

The following consumers now replay lineage instead of reading fixed history slots:

- path collectors
- diffraction audit export
- pruning-sort tie-break history handling
- validation helpers
- regression benchmarks comparing native and Dr.Jit state payloads

## Acceptance-Oriented Behavior

The Phase 2 / Phase 3 implementation now satisfies the intended mode split:

- field-only traces can execute with `cold_metadata_retained=False`
- path export still reconstructs path histories and interaction slots from lineage
- audit flows still reconstruct ordered edge/reflection sequences on demand
- packed-state profiling now reports hot bytes, lineage bytes, cold-metadata bytes, and lineage mode explicitly

Current profiling constants on the active lineage layout are:

- packed hot stride: `88` floats / `352` bytes
- packed core floats: `86`
- lineage overhead: `12` bytes per state when retained
- cold metadata overhead: `28` bytes per state when retained

## Native Packed-State Status

The C++ packed-state kernel has not been migrated to the new parent-link lineage layout yet.

For the new split-lineage state layout:

- native packed-state primal dispatch is disabled intentionally
- gather/concat/subset fall back to the updated Dr.Jit implementation
- the fallback is temporary and exists only until the C++ packed-state format is updated to the same layout

This keeps the new state model correct without reintroducing duplicated history arrays.

## Validation

Targeted regression coverage run in `witwin2`:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\diffraction\test_higher_order_edge_bvh_phase4b.py tests\diffraction\test_slope_diffraction_propagation.py tests\trace\test_solver_modes_and_guardrails.py tests\validation\test_validation_state_audit.py -q
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\mixed\test_alternating_mixed_chain_generalization.py tests\trace\test_path_monitor.py tests\backend\test_native_kernel_consistency.py -q
```

Observed results:

- `6 passed, 6 skipped`
- `2 passed, 39 skipped`

Benchmark evidence:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases high_diffractions path_export --json
```

Key observations from that run:

- `high_diffractions`
  - execution intent: `field_scalar_only`
  - `cold_metadata_retained=false`
  - packed stride: `88` floats / `352` bytes
  - allocator peak during trace: `2.13 GiB`
  - peak diffraction states after prune: `13,547`

- `path_export`
  - execution intent: `path_export`
  - `cold_metadata_retained=true`
  - state bytes: `352` hot + `12` lineage + `28` cold = `392`
  - diffraction path count: `8,650,252`
  - path-collection time: `0.331 s`
  - allocator peak during trace: `6.792 GiB`

## Remaining Follow-Up

The main unfinished Phase 2 / Phase 3 item is native packed-state parity for the new layout:

- migrate `packed_state.h` / CUDA packing logic to parent-link lineage
- re-enable native packed-state dispatch on the new layout
- keep the current benchmark and parity suite as the re-enable gate
