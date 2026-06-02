# Cell-State Memory Native Packed-State + Phase 4 Implementation Notes

Date: 2026-04-01

This note records the concrete implementation shipped to finish the pending native packed-state migration and the Phase 4 sparse path-export rollout from the cell-state memory plan.

## Scope

The completed work covers two items:

- migrate the C++ packed-state layout to the active parent-link / hot-cold split state model
- replace diffraction path-export raw payload construction with sparse state references plus lazy materialization

The public `Scene + Tracer + Result` API remains unchanged.

## Native Packed-State Migration

The bundled CUDA packed-state path is now re-enabled for the active lineage layout.

The key design change is that the native packed buffer no longer tries to serialize history arrays or cold metadata. Instead:

- the C++ packed-state buffer stores only propagation-hot diffraction fields
- Python reattaches optional cold metadata after native gather/concat/subset
- Python also reattaches parent-link lineage metadata after native gather/concat/subset

This keeps the native buffer compact while preserving the Phase 2 / Phase 3 lineage model.

Current packed-state constants on the migrated native layout are:

- packed hot stride: `88` floats / `352` bytes
- packed core floats: `86`
- packed core pointer count: `84`
- cold metadata stays outside the native packed buffer
- parent-link lineage stays outside the native packed buffer

The inserted-reflection field gather path was migrated to the same split model: the CUDA kernel returns only hot inserted-reflection fields, and Python gathers the required cold fields (`path_length_prefix`, `first_interaction_pos`, `source_type_code`) plus lineage metadata from the source state arrays.

## Phase 4: Sparse Diffraction Path References

Diffraction path export no longer builds the full raw path payload during trace collection.

Instead, `collect_diffraction_state_paths(...)` now emits a sparse payload:

- `rx_index`
- `local_rx_index`
- `state_idx`
- `a`
- `tau`
- shared references to `state_arrays`, `edge_data`, and optional geometry lookup data

The collector records this as `payload_kind="diffraction_state_refs_v1"` and defers AoD/AoA, interaction-type slots, and optional geometry-slot reconstruction.

Full path payload materialization now happens lazily inside `PathResult.from_raw_collections(...)`:

- sparse diffraction refs are detected during normalization
- the code gathers the referenced states on demand
- interaction slots and optional geometry are reconstructed only for final result assembly

This means the trace-side path collector no longer duplicates full per-path state/path payloads.

## Additional Collector-Side Memory Fix

While wiring Phase 4, the path benchmark showed that the collector still performed a second full-state gather after diffraction field evaluation.

That extra gather was removed. After `pair_valid`, the collector now gathers only:

- `edge_pos`
- `path_length_prefix`

for the kept references needed to finish `a` and `tau`.

The full diffraction state replay is left to the lazy materialization stage.

## Validation

Targeted regression coverage run in `witwin2`:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install -e . --no-build-isolation --no-deps
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_concat_and_subset_match_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_inserted_reflection_field_gather_matches_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_gather_jvp_matches_drjit_reference tests\backend\test_native_kernel_consistency.py::test_native_packed_state_subset_backward_matches_drjit_reference tests\trace\test_path_monitor.py::test_path_monitor_collects_reflection_paths_and_geometry tests\trace\test_path_monitor.py::test_path_monitor_collects_first_order_diffraction_paths
```

Observed result:

- `7 passed`

Benchmark evidence:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases path_export --json
```

Key observations from the local run:

- diffraction collector payload kind: `diffraction_state_refs_v1`
- deferred materialization: `true`
- sparse diffraction reference count: `8,650,252`
- allocator peak during trace: `10.28 GiB`
- local intermediate run before the final selective-gather fix peaked at `20.11 GiB`

So the shipped Phase 4 collector removed the trace-time full raw path expansion and also removed the extra post-field full-state gather. The remaining path-export peak is now dominated by the large Cartesian pair evaluation itself rather than the final raw path payload representation.

## Remaining Follow-Up

The main remaining memory item from the original plan is the later cell-side compaction work beyond this phase:

- sparse cell references are now in place for diffraction path export
- full path payload duplication is deferred out of the trace collector
- the next major memory lever would be further reducing temporary pair-evaluation pressure, not restoring any dense path payload layout

During validation, the broader `tests\trace\test_path_monitor.py --gpu` file still contains unrelated current-tree failures around LoS/reference expectations and reflection-discovery sharing. The targeted diffraction/reflection path-export nodes listed above pass.
