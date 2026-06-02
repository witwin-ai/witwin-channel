# Cell-State Memory Native-First Follow-Up Round 2

Date: 2026-04-02

This note records the second native-first follow-up pass after the original Phase 7 replay refactor and the first native follow-up note.

## Scope

This round closes the three native gaps that remained after the first follow-up:

- move diffraction path slot/depth assembly into the native packed-state helper path
- keep AD-sensitive `gather_inserted_reflection_state_fields(...)` on the native packed-state primal path
- replace the Python-only paired pre-expansion prune orchestration with a native paired pruning-sort path that emits both budgets from one ranking pass

## Implemented Changes

### 1. Native AD Inserted-Reflection EPC Gather

Files:

- `witwin/channel/kernels/packed_state/native_impl.py`
- `witwin/channel/kernels/packed_state/bind.h`
- `witwin/channel/kernels/packed_state/packed_state.h`
- `witwin/channel/kernels/packed_state/packed_state.cu`

`gather_inserted_reflection_state_fields(...)` no longer falls back to the Dr.Jit replay-field gather just because the input states carry AD.

The new path is:

- native packed-state primal gather for the replay fields
- native unpack into the replay-field dictionary
- custom AD reattachment for the differentiable leaves (`edge_pos`, `path_length_prefix`, `first_interaction_pos`)

This keeps the native packed-state hot path active for AD-sensitive sparse replay workloads while preserving the reference JVP behavior used by the regression suite.

### 2. Native Diffraction Path Slot/Depth Assembly

Files:

- `witwin/channel/kernels/packed_state/drjit_impl.py`
- `witwin/channel/kernels/packed_state/native_impl.py`
- `witwin/channel/kernels/packed_state/__init__.py`
- `witwin/channel/kernels/packed_state/bind.h`
- `witwin/channel/kernels/packed_state/packed_state.h`
- `witwin/channel/kernels/packed_state/packed_state.cu`
- `witwin/channel/monitors/path/collectors.py`

Sparse diffraction replay no longer builds the per-path type/depth slots entirely in Python.

The new helper:

- materializes lineage history once on the Python side
- sends the path edge slots, prefix depth, inserted-reflection depth slots, and optional geometry buffers to a native CUDA helper
- lets CUDA assemble the per-path interaction type slots, geometry lookup indices, and optional geometry outputs in one pass

Python still wraps the returned slot tensors into the `PathResult`-level structures, but the slot/depth assembly itself is no longer Python-loop bound.

### 3. Native Paired Pre-Expansion Pruning

Files:

- `witwin/channel/kernels/pruning_sort/drjit_impl.py`
- `witwin/channel/kernels/pruning_sort/native_impl.py`
- `witwin/channel/kernels/pruning_sort/__init__.py`
- `witwin/channel/kernels/pruning_sort/bind.h`
- `witwin/channel/kernels/pruning_sort/pruning_sort.h`
- `witwin/channel/kernels/pruning_sort/pruning_sort.cu`
- `witwin/channel/trace/diffraction/builders/__init__.py`

Bounded-mode pre-expansion pruning now has a paired native CUDA path.

Instead of ranking the same `prev_states` twice when the higher-order and inserted-reflection source budgets differ, the native pruning-sort helper now:

- sorts the pruning tuple once on the GPU
- emits the higher-order top-K subset
- emits the inserted-reflection top-K subset
- returns both subsets in canonical ascending state-index order so the downstream state gather path remains unchanged

When the budgets are equal, the existing shared-result fast path is still used.

## Validation

Validated in `witwin2` with:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install -e . --no-build-isolation --no-deps
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\backend\test_native_kernel_consistency.py::test_native_packed_state_inserted_reflection_field_gather_matches_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_inserted_reflection_field_gather_ad_avoids_drjit_fallback tests\backend\test_native_kernel_consistency.py::test_native_diffraction_path_slot_builder_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_ad_avoids_drjit_fallback tests\backend\test_native_kernel_consistency.py::test_native_packed_state_subset_ad_avoids_drjit_fallback tests\backend\test_native_kernel_consistency.py::test_native_pruning_sort_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_pruning_sort_history_tiebreak_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_pruning_sort_pair_matches_drjit_reference
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\trace\test_path_monitor.py::test_path_monitor_collects_first_order_diffraction_paths tests\trace\test_path_monitor.py::test_diffraction_state_path_materialization_uses_minimal_native_gather tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_selects_before_materialization tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_materializes_in_chunks tests\trace\test_path_monitor.py::test_path_monitor_reflection_replay_uses_endpoints_without_full_geometry
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q tests\trace\test_solver_modes_and_guardrails.py
```

Observed results:

- backend native consistency nodes: `8 passed`
- targeted path replay/path monitor nodes: `6 passed`
- `tests\trace\test_solver_modes_and_guardrails.py`: `3 passed`

## Runtime Snapshots

### 1. Bounded high-diffraction field workload

Command:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases high_diffractions --memory-profile memory_safe --json
```

Observed:

- trace wall clock: `3.240s`
- Dr.Jit allocator peak: `1.562 GiB`
- hot packed-state stride: `72` floats / `288` bytes
- peak higher-order source states: `1354 -> 512`
- peak inserted-reflection source states: `1354 -> 512`
- max Cartesian pairs per chunk: `9216`
- pre-expansion prune timing: `0.00259s`

This confirms that the paired native pre-expansion path is live in the bounded memory-safe workload and still keeps source growth capped before higher-order and inserted-reflection expansion.

### 2. Sparse path-export workload

Command:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases path_export --json
```

Observed:

- trace wall clock: `0.430s`
- Dr.Jit allocator peak: `9.533 GiB`
- sparse diffraction references: `8,650,252`
- path collection total: `0.1566s`
- path collection field evaluation: `0.1065s`
- path collection slot assembly during trace: `0.0s` because sparse refs remain deferred

The path-export trace remains dominated by field evaluation and sparse reference emission, not by slot replay work.

### 3. Native-heavy multipath AD workload

Command:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_multipath_ad --parameters tx_x cube1_x --modes jvp --grid-size 256 --n-rays 1280 --repeats 1 --warmup-runs 1 --workload full_field
```

Observed steady-state results:

- `tx_x`: trace `0.258s`, JVP `0.460s`, total `0.718s`
- `cube1_x`: trace `0.222s`, JVP `0.549s`, total `0.849s`
- all three traced backends resolved to native custom ops:
  - reflection: `native_cuda_custom_op`
  - diffraction: `native_cuda_custom_op`
  - suffix: `native_cuda_custom_op`

Internal trace timing for these runs shows:

- diffraction remains the dominant trace component:
  - `tx_x`: diffraction `0.223s`, reflection `0.033s`
  - `cube1_x`: diffraction `0.186s`, reflection `0.035s`
- within diffraction, the largest preparation costs are still:
  - inserted-reflection state building
  - higher-order candidate/incident-field work
  - suffix accumulation

In this real workload, pruning and sparse replay are no longer the dominant hot spots.

## Remaining Gaps

This round materially improves the native path, but a few boundaries remain:

- lineage-history materialization still happens in Python before the native slot builder runs
- sparse replay wrapper assembly at the `PathResult` layer is still Python-side
- pre-expansion pruning is now paired and native, but candidate generation plus pruning is not yet fused into one CUDA kernel

The next native work should focus on:

1. native lineage replay inputs for the slot builder
2. native assembly of the final sparse replay wrapper payloads
3. only then a larger candidate-generation-plus-prune fusion, if a bounded workload shows it matters
