# PathMonitor Acceleration Plan

Status: Active
Category: Optimization
Last reviewed: 2026-04-03

## Purpose

This document is the consolidated plan for PathMonitor-specific acceleration work. It folds together:

- the public architecture and constraints from `docs/dev/plans/path-monitor-design.md`,
- the completed storage and replay baseline from `docs/dev/optimization/cell-state-memory-rollout.md`,
- the remaining kernel candidates from `docs/dev/optimization/memory-optimization-and-kernel-candidates.md`,
- the native tiling direction from `docs/dev/optimization/receiver-tiled-cuda-path-family-refactor-plan.md`.

The goal is not to redesign the public API. The goal is to reduce PathMonitor wall time while keeping the stable `Scene + Tracer + Result` architecture, GPU-first execution, and current diffraction-family semantics.

## Current Baseline

The cell-state rollout already removed the largest structural memory waste for path export:

- sparse diffraction path references are emitted during trace,
- selected-only replay happens lazily in `PathResult.from_raw_collections(...)`,
- hot state storage is compact and lineage-based,
- native gather/subset helpers cover the main sparse replay path.

That work is complete and remains the baseline. New PathMonitor work should build on it, not reopen it.

## Current Bottlenecks

Recent targeted PathMonitor profiling on the current branch shows two remaining hotspots.

### Large path-export workload

Configuration:

- `100` RX
- `4` boxes
- `1000` reflection rays
- `3` reflection bounces
- `2` diffraction orders

Observed steady-state timing:

- `diff.state_prep`: about `49.9 ms`
- `assembly.from_raw`: about `15.3 ms`
- `diff.utd_eval`: about `11.5 ms`

Interpretation:

- diffraction state preparation is the dominant end-to-end bottleneck,
- path-result assembly is still material but is no longer the first problem.

### Warm-cache multi-frame workload

Configuration:

- `1` TX
- `5` RX
- `30` frames
- repeated traces with warm reflection and scene caches

Observed steady-state timing after the latest low-risk assembly cleanup:

- `diff_eval`: about `18.6 ms`
- `assembly`: about `15.1 ms`

Interpretation:

- for repeated small-RX path export, final assembly is still a large fraction of the total call,
- but the biggest full-path win still comes from reducing or reusing diffraction-side work.

## Guiding Decisions

- Keep FieldMonitor and PathMonitor configurable independently for diffraction depth.
- Treat `PathMonitor` as a bounded export workload, not as a dense-field monitor in disguise.
- Prefer reducing enumerated work before adding heavier native kernels.
- Keep native/CUDA work focused on hot loops that still materialize many small GPU ops or repeated state-prep slices.

## Phase Map

### Phase 0: Completed Baseline

Scope:

- completed cell-state memory rollout through Phase 7 plus native-first follow-up work
- sparse path references, lazy selected-only replay, compact lineage-based state storage

Acceptance:

- keep `cell-state-memory-rollout.md` as the frozen baseline summary
- do not reintroduce dense path payload materialization before final selection

### Phase 1: Workload Shaping and Explicit Diffraction Controls

Status: Completed on 2026-04-03

Scope:

- add per-monitor `max_diffractions` overrides for both `FieldMonitor` and `PathMonitor`
- let `max_diffractions=None` inherit the tracer-wide trace setting
- make `PathMonitor` default to first-order diffraction
- update benchmark and profiling scripts so any deeper-order PathMonitor workload is requested explicitly

Why this phase comes first:

- it is low risk,
- it removes accidental higher-order path export from the default workload,
- discrete path export rarely needs the same default diffraction depth as dense field accumulation.

Expected effect:

- default PathMonitor traces should usually be faster because higher-order and mixed diffraction state growth is no longer enabled unless requested,
- benchmark data becomes easier to interpret because the requested path-export depth is explicit.

### Phase 2: PathMonitor State-Prep Reuse

Status: Completed on 2026-04-03

Scope:

- add PathMonitor diffraction state-prep caching keyed by scene mesh version, `tx`, shared receiver `z` slice, and effective solver controls
- share cached direct-diffraction state prep across repeated `Tracer.trace(...)` calls when the cached state is AD-safe to reuse
- reuse the same prepared state within a trace call across multiple PathMonitor executions that resolve to the same cache key
- invalidate the cache on scene mesh updates
- surface cache hit/miss reporting in `PathResult.metadata["runtime_reuse"]` and per-group metadata

Shipped behavior:

- persistent reuse stays enabled only for direct-diffraction state prep whose inputs are stable across trace calls
- mixed reflection-prefix diffraction falls back to per-trace reuse because its prepared state depends on reflection-discovery detail
- path-only state-prep slimming remains part of the later state-prep reduction phase

Observed effect:

- local repeated-trace benchmark on the current branch
- configuration: single wedge, `5` RX, fixed `tx`, first-order PathMonitor diffraction, receiver `x/y` motion with shared `z`
- steady-state total PathMonitor time dropped from about `25.9 ms` with cache clears every frame to about `22.8 ms` with cache reuse
- the diffraction section inside monitor timing dropped from about `19.6 ms` to about `16.6 ms`

Acceptance:

- repeated-path traces with identical `tx` and receiver-height slices show a measurable drop in diffraction-side time
- scene updates invalidate cached PathMonitor state prep
- no change to exported path content or solver provenance

### Phase 3: Native Path Assembly Kernel

Status: Completed on 2026-04-03

Scope:

- replace the remaining Python/Torch-heavy path selection and slot-packing path inside `PathResult.from_raw_collections(...)`
- push per-RX top-k selection, ranking, slot assignment, and depth-aware packing into a native helper or CUDA kernel

Why this phase matters:

- assembly is still around `15 ms` in warm-cache small-RX workloads,
- the current path is structurally many small GPU ops plus Python orchestration,
- this is the clearest dedicated PathMonitor kernel candidate.

Candidate kernel shape:

- input: flat raw summaries (`rx_index`, `|a|`, `tau`, depth hints, sparse payload refs)
- output: kept path indices, per-RX slot ids, packed valid mask, packed depth metadata

Shipped behavior so far:

- path selection and summary partitioning now run on lightweight concatenated `rx/a/tau` summaries before any sparse replay/materialization
- no-geometry reflection path export now carries per-path reflection depth plus AoD/AoA angles in `reflection_path_refs_v1`, so `PathResult.from_raw_collections(...)` can rebuild reflection slots without replaying the reflection chain a second time
- no-geometry assembly now also reuses summary-level torch caches for already-materialized dense raws and cached reflection refs instead of re-normalizing those chunks through the generic raw-materialization path
- selected-depth sizing for reflection refs is now driven by the selected `path_depth` subset instead of a larger summary-wide max-depth hint, which keeps slot packing aligned with the allocated result depth

Observed effect:

- local A/B benchmark on the current branch
- configuration: warm-cache repeated path export, `1` TX, `5` RX, `20` frames, `4` boxes, `1000` reflection rays, `3` reflection bounces, first-order diffraction, `return_geometry=False`
- steady-state `assembly.from_raw` dropped from about `8.5 ms` on the legacy replay-in-assembly reflection path to about `4.2 ms` on the current shipped assembly path
- the final summary-cache tightening reduced assembly-only `from_raw_collections(...)` time by about another `4.7%` relative to the intermediate reflection-fast-path-only branch on the same captured raw workload
- steady-state end-to-end cached pipeline time remains about `21.0 ms` on the shipped branch versus about `24.4 ms` on the legacy replay-in-assembly path

Acceptance:

- `PathResult.from_raw_collections(...)` wall time drops materially on cached multi-frame path-export workloads
- sparse replay remains selected-only and chunked

### Phase 4: Diffraction State-Prep Reduction for Path Export

Status: Completed on 2026-04-03

Scope:

- specialize diffraction state preparation for discrete receivers instead of dense field monitors
- narrow path-only retained fields even further before replay
- replace full-state gather on the PathMonitor path with reduced-layout path-specific gathers for evaluation and replay

Why this phase is higher value than more PathResult polishing:

- current large-workload profiling is dominated by `diff.state_prep`, not final packing
- end-to-end wins require shrinking or reusing the source-state construction path itself

Shipped behavior:

- PathMonitor diffraction state prep now emits a reduced path-export state layout instead of carrying the full field-oriented state payload into path collection and sparse replay
- path collection no longer does full `gather_state_arrays(...)` on every visible `(state, rx)` pair when the reduced layout is available
- sparse path replay/materialization also uses reduced-layout replay gathers rather than reconstructing a full diffraction state just to rebuild slots and angles
- metadata reports the selected path state layout through `diffraction_groups[*].state_layout`, `diffraction_groups[*].path_collection["state_layout"]`, and `runtime_reuse["diffraction_state_prep_cache"]["state_layout"]`

Observed effect:

- local A/B benchmark on the current branch
- configuration: single wedge, `5` RX, first-order PathMonitor diffraction, repeated traces, forced `full` path state layout versus shipped reduced layout
- steady-state total PathMonitor time dropped from about `30.4 ms` to about `25.4 ms`
- the diffraction section inside monitor timing dropped from about `20.0 ms` to about `17.0 ms`

Acceptance:

- PathMonitor path-export traces show a measurable reduction in diffraction-side time when compared against the full path-state layout
- no regression in higher-order or mixed-family completeness when explicitly requested

### Phase 5: Path-Family Native Replay and Tiling

Status: Completed on 2026-04-03

Scope:

- adapt the native-first family/tile direction from `receiver-tiled-cuda-path-family-refactor-plan.md` to discrete PathMonitor export
- promote reusable reflection or diffraction families to compact descriptors before replay
- avoid rebuilding identical replay work for nearby receivers or repeated RX arrays when a native family descriptor can be reused

Shipped behavior:

- PathMonitor reflection collection now prepares one compact reflection EPC descriptor per unique path-family chunk, then reuses that descriptor across every receiver pair in the chunk instead of re-gathering reflection hot data per `(path, rx)` replay.
- The prepared descriptor keeps the replay path native-compatible by carrying the flattened image-source, plane, and material-slot payload plus the original source-path index mapping needed for geometry validation and endpoint reconstruction.
- `epc_reflection_chain_to_target(...)` now accepts a prepared EPC descriptor, so PathMonitor collection can reuse the same replay hot data while keeping the existing native/custom-op replay kernels on the hot path.
- Path tracing now also exposes `Tracer.trace(..., monitor_overrides={...})` and `Tracer.trace_many([...])`, which provides a stable way to exercise repeated RX-array traces and per-request `tx/rx` overrides without mutating scene monitors between runs.

Observed effect:

- local reflection EPC microbenchmark on the current branch
- configuration: `4` blocker boxes, `3` reflection bounces, largest first-order family chunk with `6` unique paths, `512` discrete receivers, `return_geometry=False`
- pair replay workload: `3072` `(path, rx)` pairs
- prepared family-descriptor replay reduced the replay chunk from about `1.16 ms` to about `1.13 ms`
- the measured replay-only improvement on that workload is about `2.9%`

Acceptance:

- path-export workloads with many RX or repeated array traces reuse native EPC descriptors instead of repeating full family replay
- new path-export hot paths stay native for both primal and supported AD modes

### Phase 6: Rollout Gates and Benchmark Closure

Status: Completed on 2026-04-03

Scope:

- freeze a repeatable PathMonitor benchmark matrix
- define performance gates for:
  - default first-order path export,
  - explicit multi-order path export,
  - warm-cache repeated multi-frame traces,
  - geometry-off versus geometry-on export,
  - field-plus-path mixed monitor calls

Acceptance:

- every performance phase has a benchmark delta and a correctness gate before becoming the new default
- docs and profiling scripts report the requested PathMonitor diffraction depth explicitly

Shipped behavior:

- the frozen Phase 6 benchmark matrix now lives in `optimization/path-monitor-phase6-rollout-gates.md`
- `python -m tests.support.bin.benchmark_path_monitor_phase6 --strict-gates` runs the canonical closure matrix and exits nonzero if any rollout gate fails
- the matrix covers:
  - default first-order path export,
  - explicit higher-order path export,
  - geometry-off versus geometry-on path export,
  - standalone field-only and path-only baselines for mixed-monitor comparison,
  - mixed field-plus-path trace calls,
  - warm-cache `Tracer.trace_many(...)` with multi-TX, multi-RX, and per-request `tx/rx` overrides
- the warm-cache benchmark excludes first-use compilation noise by running one compile warmup before clearing trace caches and measuring the cold frame
- the `profile_path_monitor*.py` scripts now print the requested and effective PathMonitor diffraction depth explicitly

Observed effect:

- reviewed baseline from `python -m tests.support.bin.benchmark_path_monitor_phase6 --strict-gates`
- `default_first_order_path_export`: about `17.60 ms`
- `explicit_multi_order_path_export`: about `18.60 ms`
- `geometry_off_path_export`: about `8.82 ms`
- `geometry_on_path_export`: about `11.16 ms`
- `field_only_baseline`: about `11.11 ms`
- `path_only_baseline`: about `22.71 ms`
- `mixed_field_path_trace`: about `31.60 ms`
- `warm_cache_trace_many`: cold about `18.41 ms/request`, warm steady-state about `16.59 ms/request`
- all four Phase 6 rollout gates passed on the reviewed branch

## Recommended Execution Order

1. Keep Phases 1 through 6 shipped and treat them as the current PathMonitor baseline.
2. Run the Phase 6 benchmark command before and after any new PathMonitor performance work.
3. If a new hotspot matters for rollout decisions, add it to the frozen Phase 6 matrix before changing the default workload again.
4. Only reopen deeper native family/tile work once the Phase 6 matrix shows a new dominant bottleneck.

## Relationship To Other Documents

- `path-monitor-design.md` remains the public-architecture reference.
- `cell-state-memory-rollout.md` remains the completed storage and replay baseline.
- `memory-optimization-and-kernel-candidates.md` remains the general kernel-candidate catalog.
- `receiver-tiled-cuda-path-family-refactor-plan.md` remains the broader dense-field native replay direction; this document only pulls the parts that are relevant to PathMonitor after the nearer bottlenecks are addressed.
