# Cell-State Memory Phase 5 + Phase 6 Implementation Notes

Date: 2026-04-02

This note records the concrete implementation shipped for Phase 5 and Phase 6 of the cell-state memory optimization plan.

## Scope

The completed work covers two items:

- move bounded-mode pruning earlier so weak diffraction states can be removed before expensive higher-order and inserted-reflection expansion
- shrink the hot packed-state representation itself after the Phase 2 to Phase 4 storage-model changes

The public `Scene + Tracer + Result` API remains unchanged.

## Phase 5: Earlier Pruning Before Cartesian Expansion

Bounded modes now apply a documented source-side pruning policy before the higher-order and inserted-reflection builders run:

- `accuracy` + default memory profile keeps the previous behavior and does not apply implicit early pruning
- `fast_approximate` and `memory_safe` resolve a `pre_expansion_policy` in diffraction solver metadata
- the current bounded-mode rule keeps only the strongest source frontier before expansion, using a source budget derived from the explicit per-order state budgets

The shipped policy is:

- `higher_order_source_budget = total_state_budget_per_order / 4`
- `inserted_source_budget = min(total_state_budget_per_order / 4, inserted_reflection_state_budget)`

Both budgets are rounded up to at least one state when the corresponding downstream budget is non-zero.

Instrumentation was extended so the profiling payload now reports:

- the resolved `pre_expansion_policy`
- per-order source counts before and after pre-pruning
- peak higher-order and inserted source counts before and after pre-pruning

This keeps the Phase 5 rollout explicit and makes it easy to verify that bounded modes actually reduce expansion pressure instead of only reporting a policy name.

## Phase 6: Packed-State Width Reduction

The native packed-state buffer was reduced from the previous hot layout to a smaller authoritative hot layout:

- previous hot stride: `88` floats / `352` bytes
- current hot stride: `72` floats / `288` bytes

The reduction comes from removing redundant stored fields from the native packed buffer:

- stored `incident_vector_{x,y,z}` complex channels were removed
- stored `incident_normal_derivative_vector_{x,y,z}` complex channels were removed
- the old packed presence slots for face operators were removed

The native unpack path now reconstructs the removed vector transport fields from:

- `incident_jones_{u,v}`
- `incident_derivative_jones_{u,v}`
- `incident_basis_{u,v,k}`

using the existing `vector_from_jones(...)` transport helper.

The native CUDA primal path remains enabled for the compact layout:

- native packed-state gather
- native packed-state concat
- native packed-state primal subset

For AD-sensitive gather/subset calls, the implementation currently keeps the validated Dr.Jit path to preserve forward- and backward-mode correctness after the vector fields became derived data instead of stored hot payload.

## Validation

Targeted validation was run in `witwin2`:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install -e . --no-build-isolation --no-deps
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\trace\test_solver_modes_and_guardrails.py -q
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_concat_and_subset_match_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_inserted_reflection_field_gather_matches_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_jvp_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_subset_backward_matches_drjit_reference
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases high_diffractions path_export --json
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases high_diffractions --memory-profile memory_safe --json
```

Observed test results:

- `tests\trace\test_solver_modes_and_guardrails.py`: `3 passed`
- native packed-state parity/JVP/backward coverage: `5 passed`

Observed benchmark highlights:

- `high_diffractions` with default accuracy profile:
  - trace time: `5.26s`
  - Dr.Jit allocator peak: `5.303 GiB`
  - packed hot stride: `72` floats / `288` bytes
  - pre-expansion policy: disabled
  - peak candidate-pair chunk pressure: `225,576`

- `high_diffractions` with `memory_safe`:
  - trace time: `2.25s`
  - Dr.Jit allocator peak: `1.923 GiB`
  - peak higher-order source states: `2048 -> 512` after pre-prune
  - peak inserted source states: `2048 -> 512` after pre-prune
  - peak total states before/after prune: `3072 -> 2048`
  - peak candidate-pair chunk pressure: `9,216`

- `path_export` with default accuracy profile:
  - trace time: `0.52s`
  - Dr.Jit allocator peak: `9.576 GiB`
  - packed bytes per diffraction state: `288` hot + `12` lineage + `28` cold = `328`
  - sparse diffraction reference count: `8,650,252`

These results show the intended Phase 5 and Phase 6 effects:

- bounded modes now reduce candidate growth before expansion on the stress workload instead of only relying on post-expansion budgets
- the hot packed-state footprint is materially smaller than the previous native layout baseline

## Remaining Follow-Up

The main follow-up after this rollout is not another storage-model rewrite. The next meaningful memory lever remains workload-level pressure beyond the packed-state representation itself, for example:

- additional early culling strategies beyond power-top-K
- tighter receiver/tile-local expansion control
- further acceptance coverage for broader end-to-end workloads

The compact layout is now in place, and the bounded-mode pruning policy is active and measurable.
