# Cell-State Memory Native-First Follow-Up Notes

Date: 2026-04-02

This note records the native-first follow-up work after the Phase 7 path replay refactor.

## Scope

The completed work focuses on the parts of the cell-state memory rollout that most directly affect the native CUDA path:

- keep AD-sensitive packed-state gather/subset on the native primal hot path
- reduce sparse diffraction path replay cost by gathering only the minimal replay fields needed for path materialization
- remove duplicate pre-expansion source-prune passes when bounded-mode higher-order and inserted-reflection source budgets resolve to the same budget
- profile representative native-heavy workloads before and after the follow-up so the next native work is driven by measured hotspots

## Profiling Findings Before The Follow-Up

Local hotspot profiling on representative workloads showed three distinct regimes.

### 1. Main field workload

Representative workload:

- `samples.save_multipath_main_component_gradient_figure`
- `tx_x`
- `grid_size=256`
- `n_rays=1280`

Observed before the follow-up:

- forward field workload was not bottlenecked by packed-state or pruning
- AD field workload still used Dr.Jit packed-state fallback paths for:
  - `subset_state_arrays`
  - `gather_state_arrays`
  - `concat_state_arrays`
  - inserted-reflection field gathers

The fallback cost was not dominant in absolute wall-clock time, but it meant the advertised native packed-state path was not actually staying native in the AD-sensitive diffraction workload.

### 2. Path export with sparse replay

Representative workload:

- `PathMonitor` built from the `benchmark_cell_state_memory` `path_export` scene
- `grid_size=128`
- `reflection_n_rays=10000`
- `max_diffractions=2`

Observed before the follow-up:

- `PathResult.from_raw_collections(...)` dominated path-export wall-clock time
- `_materialize_diffraction_state_path_refs(...)` still spent meaningful time on full packed-state replay even though only replay fields were needed
- in the uncapped case, native packed-state replay hot gather was called more than two thousand times during sparse path replay

### 3. Bounded high-diffraction field workload

Representative workload:

- `benchmark_cell_state_memory`
- `high_diffractions`
- `memory_profile="memory_safe"`

Observed before the follow-up:

- bounded-mode pre-expansion pruning worked correctly
- native/source-prune cost was small relative to the whole field trace
- higher-order and inserted-reflection source budgets often resolved to the same top-K budget, so the pipeline still paid for duplicate source pruning on the same `prev_states`

## Implemented Changes

## 1. AD-Sensitive Packed-State Gather/Subset Stay On Native Primal

Files:

- `witwin/channel/kernels/packed_state/native_impl.py`

`gather_state_arrays(...)` and `subset_state_arrays(...)` no longer fall back to the full Dr.Jit SoA gather/subset when the input state arrays carry AD.

The shipped implementation keeps the native primal path:

- native packed-state pack
- native packed-state gather
- native packed-state unpack

and then reattaches AD on the differentiable leaves of the gathered state via leaf-wise custom AD wrappers.

This keeps the packed-state hot path native while preserving the validated JVP/backward behavior from the existing regression suite.

Current boundary:

- AD-sensitive inserted-reflection field gathers still keep the Dr.Jit path
- this follow-up only removes the full packed-state gather/subset fallback

## 2. Sparse Path Replay Uses Minimal Native Field Gather

Files:

- `witwin/channel/monitors/path/collectors.py`

Sparse diffraction path materialization no longer calls the full `gather_state_arrays(...)` replay path.

It now uses `gather_inserted_reflection_state_fields(...)`, which already has a native CUDA raw launcher and returns only the fields required for sparse replay materialization:

- `edge_pos`
- `prefix_reflection_depth`
- `intermediate_reflection_depth`
- `suffix_reflection_depth`
- `order`
- retained cold metadata such as `first_interaction_pos`

This is enough for:

- departure/arrival angle reconstruction
- depth computation
- lineage/history replay through gathered metadata
- optional geometry reconstruction in `_build_type_and_geometry_slots(...)`

without unpacking the full hot diffraction state for every sparse replay chunk.

## 3. Shared Pre-Expansion Source Prune When Budgets Match

Files:

- `witwin/channel/trace/diffraction/builders/__init__.py`

Bounded-mode source pruning now reuses a single shared pre-expansion prune result when:

- higher-order source budget
- inserted-reflection source budget

resolve to the same value.

This removes the duplicate source-prune pass on the same `prev_states` for the common `memory_safe` top-K case while keeping the existing budgeting/reporting behavior intact.

This is intentionally a low-risk fusion step. Profiling did not justify a larger dedicated C++/CUDA pruning rewrite yet.

## Validation

Targeted validation was run in `witwin2`:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q tests\trace\test_solver_modes_and_guardrails.py
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_concat_and_subset_match_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_inserted_reflection_field_gather_matches_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_jvp_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_ad_avoids_drjit_fallback tests\backend\test_native_kernel_consistency.py::test_native_packed_state_subset_backward_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_subset_ad_avoids_drjit_fallback
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\trace\test_path_monitor.py::test_path_monitor_collects_first_order_diffraction_paths tests\trace\test_path_monitor.py::test_diffraction_state_path_materialization_uses_minimal_native_gather tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_selects_before_materialization tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_materializes_in_chunks tests\trace\test_path_monitor.py::test_path_monitor_reflection_replay_uses_endpoints_without_full_geometry
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\main\test_multipath_main.py
```

Observed results:

- `tests\trace\test_solver_modes_and_guardrails.py`: `3 passed`
- packed-state native regression nodes: `7 passed`
- targeted path-monitor replay nodes: `6 passed`
- `tests\main\test_multipath_main.py`: `1 passed`

Local note for `test_multipath_main.py`:

- first run after the native-first follow-up took `325.73s`
- immediate hot-cache rerun took `27.33s`

So the steady-state main workload remained healthy, while the first run still pays substantial one-time cold-start cost in this environment.

## Measured Runtime Snapshots

Direct local wall-clock measurements after the follow-up:

- main field forward workload (`tx_x`, `256x256`, `1280` rays): `0.818s`
- main field AD workload (`tx_x`, `256x256`, `1280` rays): `1.331s`
- bounded high-diffraction field workload (`high_diffractions`, `memory_safe`): `0.601s`
- path export with `max_num_paths=8`: `0.671s`
- uncapped path export (`resolved_max_num_paths=988`): `6.012s`

Internal hotspot measurements on the path-export replay path showed the main effect of the minimal native replay gather:

- uncapped sparse replay:
  - `_materialize_diffraction_state_path_refs(...)` dropped from about `5.39s` to about `3.13s`
  - full packed-state native gather time inside sparse replay dropped from about `1.39s` to about `0.004s`
- top-K sparse replay (`max_num_paths=8`):
  - `PathResult.from_raw_collections(...)` dropped from about `1.48s` to about `0.41s`
  - `_materialize_diffraction_state_path_refs(...)` dropped from about `0.062s` to about `0.034s`

These hotspot numbers come from the same local profiling harness before and after the follow-up, not from the public benchmark script.

## Remaining Gaps

The follow-up improves the native path materially, but it does not finish every possible native migration.

Still not native-end-to-end:

- AD-sensitive inserted-reflection EPC gather still uses the Dr.Jit path
- sparse path replay slot assembly and lineage-history expansion still run in Python
- bounded pre-expansion pruning is only lightly fused at the orchestration layer; there is still no dedicated CUDA kernel that sorts and emits both shared source subsets plus replay metadata in one pass

Based on the measured hotspots, the next native work should prioritize:

1. path replay slot/depth assembly, not pruning
2. optional native AD handling for inserted-reflection EPC gathers
3. only then a larger fused pruning kernel, if a bounded-mode workload shows it matters in practice
