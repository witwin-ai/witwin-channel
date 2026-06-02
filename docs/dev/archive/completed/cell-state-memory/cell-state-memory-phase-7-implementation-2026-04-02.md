# Cell-State Memory Phase 7 Implementation Notes

Date: 2026-04-02

This note records the concrete implementation shipped for Phase 7 of the cell-state memory optimization plan.

## Scope

The completed work covers the Phase 7 path-export replay refactor on top of the Phase 4 sparse diffraction path payload:

- keep the public `PathResult` payload padded and semantically unchanged
- stop fully materializing sparse diffraction path refs before global `max_num_paths` selection
- keep geometry replay lazy behind `return_geometry=True`
- bound second-pass sparse replay with chunked materialization so large kept path sets do not rebuild one dense intermediate at a time

The public `Scene + Tracer + Result` API remains unchanged.

## Problem Before Phase 7

Phase 4 had already converted diffraction path export to sparse state references with
`payload_kind="diffraction_state_refs_v1"`, but `PathResult.from_raw_collections(...)` still normalized each raw collection eagerly.

That meant the pipeline still:

1. materialized every sparse diffraction ref into dense AoD/AoA, interaction-type, and optional geometry slots
2. concatenated those dense payloads across all raw collections
3. only then applied the final receiver/path selection and padding rules

So trace-side duplication was gone, but result shaping could still rebuild more dense path payload than the final public result needed.

## Shipped Design

`PathResult.from_raw_collections(...)` now uses a two-pass selection-aware assembly path.

### First Pass: Lightweight Summary Only

Each raw collection is summarized with only the fields needed for global selection and ordering:

- `rx_index`
- `a`
- `tau`
- payload kind and depth hints

This lets the code determine:

- per-receiver path counts
- final `max_num_paths`
- strength-based top-K truncation
- receiver/tau ordering

without first replaying sparse diffraction references into full path payloads.

### Second Pass: Selected-Only Replay

After the final selected path indices are known:

- sparse diffraction refs are replayed only for the kept local path indices
- dense raw collections are subset before normalization so non-diffraction paths follow the same selected-only contract
- `max_depth` is computed from the kept paths only

The public `PathResult` output layout is unchanged:

- amplitudes, delays, AoD/AoA
- interaction codes
- optional geometry/object slots
- padded receiver-path tensors plus masks and counts

### Chunked Replay

Phase 7 also adds bounded second-pass replay for sparse selections:

- selected sparse paths are replayed in fixed-size chunks
- the current internal chunk size is `4096` kept paths per replay batch
- depth estimation for sparse diffraction refs is chunked as well

This keeps the replay stage from constructing one large dense intermediate even when `max_num_paths=None` and the final public output still retains many paths.

### Geometry Laziness

Optional geometry construction stays lazy:

- `return_geometry=False` does not build vertex, normal, or object slots
- `return_geometry=True` rebuilds geometry only for the selected kept paths and then pads those slots into the final public layout

## Code Touchpoints

- `witwin/channel/monitors/path/collectors.py`
  - sparse diffraction ref materialization now accepts selected path indices and can subset the sparse payload before replay
- `witwin/channel/result.py`
  - summary-first selection
  - selected-only normalization
  - selected-path chunked replay into the final padded output tensors
- `tests/trace/test_path_monitor.py`
  - regression coverage for selected-only sparse replay
  - regression coverage for chunked sparse replay

## Validation

Targeted validation was run in `witwin2`:

```bash
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest --gpu -q tests\trace\test_path_monitor.py::test_path_monitor_collects_reflection_paths_and_geometry tests\trace\test_path_monitor.py::test_path_monitor_collects_first_order_diffraction_paths tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_selects_before_materialization tests\trace\test_path_monitor.py::test_path_result_sparse_diffraction_replay_materializes_in_chunks tests\trace\test_path_monitor.py::test_path_monitor_reflection_replay_uses_endpoints_without_full_geometry
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q tests\validation\test_validation_state_audit.py
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_cell_state_memory --cases path_export --json
```

Observed results:

- targeted path-monitor coverage: `6 passed`
- validation state-audit coverage: `2 skipped`

Observed benchmark highlights for the default uncapped `path_export` stress case:

- trace time: `0.576s`
- Dr.Jit allocator peak: `9.533 GiB`
- sparse diffraction reference count: `8,650,252`
- payload kind: `diffraction_state_refs_v1`
- packed bytes per diffraction state: `288` hot + `12` lineage + `28` cold = `328`

The targeted regression tests add the key Phase 7 proof points that are not obvious from the uncapped benchmark alone:

- sparse diffraction refs are selected before materialization, so top-K truncation only replays the final kept paths
- sparse diffraction replay is chunked, so the second pass no longer rebuilds one large dense intermediate payload per raw collection

## Remaining Follow-Up

Phase 7 finishes the path-result replay refactor, but two follow-ups remain outside this specific rollout:

- the current `benchmark_cell_state_memory` path-export case still runs with `max_num_paths=None`, so it is best for uncapped stress scenes rather than for highlighting top-K replay savings
- broader acceptance / full GPU-suite coverage is still pending for the overall memory rollout

The main Phase 7 architectural goals are now in place:

- public path output remains semantically equivalent
- sparse diffraction refs stay sparse until final selection
- geometry remains optional and lazy
- second-pass replay is bounded instead of all-at-once
