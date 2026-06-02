# PathMonitor Phase 6 Rollout Gates

Status: Active
Category: Optimization
Last reviewed: 2026-04-03

## Purpose

This document freezes the PathMonitor Phase 6 benchmark matrix and rollout gates.

It is the authoritative closure document for the shipped PathMonitor optimization stack:

- Phase 1 workload shaping and explicit diffraction controls
- Phase 2 PathMonitor diffraction state-prep reuse
- Phase 3 native/no-replay path assembly fast paths
- Phase 4 reduced diffraction state layout for path export
- Phase 5 prepared path-family EPC descriptors
- Phase 6 benchmark closure and rollout gates

Use this document together with `optimization/path-monitor-acceleration-plan.md`. The plan remains the phased narrative; this document is the frozen benchmark and gate contract.

## Canonical Command

Run the complete Phase 6 matrix with:

```bash
python -m tests.support.bin.benchmark_path_monitor_phase6 --strict-gates
```

The command is intentionally opinionated:

- it runs the frozen benchmark matrix only,
- it reports requested and effective PathMonitor diffraction depth explicitly,
- it exits nonzero when any rollout gate fails.

## Frozen Benchmark Matrix

### 1. `default_first_order_path_export`

- Scene: `triple_wedge`
- TX: `1`
- RX: `6`
- Reflection: disabled
- Tracer `max_diffractions`: `3`
- PathMonitor requested `max_diffractions`: `1`
- `return_geometry=False`
- Purpose: freeze the default bounded PathMonitor export workload

### 2. `explicit_multi_order_path_export`

- Scene: `triple_wedge`
- TX: `1`
- RX: `6`
- Reflection: disabled
- Tracer `max_diffractions`: `3`
- PathMonitor requested `max_diffractions`: `3`
- `return_geometry=False`
- Purpose: explicit higher-order comparison point for the default workload

### 3. `geometry_off_path_export`

- Scene: `reflection_wall`
- TX: `1`
- RX: `2`
- Reflection rays: `8192`
- Reflection bounces: `1`
- PathMonitor requested `max_diffractions`: `0`
- `return_geometry=False`
- Purpose: no-geometry export baseline for reflection-heavy path assembly

### 4. `geometry_on_path_export`

- Scene: `reflection_wall`
- TX: `1`
- RX: `2`
- Reflection rays: `8192`
- Reflection bounces: `1`
- PathMonitor requested `max_diffractions`: `0`
- `return_geometry=True`
- Purpose: geometry-on comparison point for path export overhead

### 5. `field_only_baseline`

- Scene: `mixed_boxes`
- TX: `1`
- Field grid: `24 x 24`
- Reflection rays: `1024`
- Reflection bounces: `1`
- Tracer `max_diffractions`: `1`
- Purpose: standalone field baseline for mixed-monitor orchestration

### 6. `path_only_baseline`

- Scene: `mixed_boxes`
- TX: `1`
- RX: `4`
- Reflection rays: `1024`
- Reflection bounces: `1`
- PathMonitor requested `max_diffractions`: `1`
- `return_geometry=False`
- Purpose: standalone path baseline for mixed-monitor orchestration

### 7. `mixed_field_path_trace`

- Scene: `mixed_boxes`
- TX: `1`
- Field grid: `24 x 24`
- RX: `4`
- Reflection rays: `1024`
- Reflection bounces: `1`
- PathMonitor requested `max_diffractions`: `1`
- `return_geometry=False`
- Purpose: freeze the mixed field-plus-path call path and its reuse behavior

### 8. `warm_cache_trace_many`

- Scene: `single_wedge`
- TX: `2`
- RX: `4`
- Frames: `6`
- API path: `Tracer.trace_many([...], monitor=base_monitor, monitor_overrides=...)`
- PathMonitor requested `max_diffractions`: `1`
- `return_geometry=False`
- Purpose: freeze the repeated multi-TX, multi-RX, per-request `tx/rx` override workload

Warm-cache measurement rules:

- one compile warmup run is executed before measurement,
- `Tracer._clear_trace_caches()` is called after that warmup,
- the measured cold frame therefore means "no runtime cache, already compiled",
- the warm metric is the steady-state median per-request time from later frames.

## Rollout Gates

### Gate: `first_order_vs_multi_order_bounded`

- Formula: `median(default_first_order_path_export) / median(explicit_multi_order_path_export) <= 1.05`
- Correctness gate:
  - default effective depth must be `1`
  - explicit effective depth must be `3`
  - explicit diffraction path count must be at least the default diffraction path count

### Gate: `geometry_off_vs_on_bounded`

- Formula: `median(geometry_off_path_export) / median(geometry_on_path_export) <= 1.10`
- Correctness gate:
  - `a`, `tau`, `valid`, `num_paths`, `types`, and `rx_positions` must match
  - geometry-off result must not expose geometry arrays
  - geometry-on result must expose geometry arrays

### Gate: `mixed_monitor_vs_separate_bounded`

- Formula: `median(mixed_field_path_trace) / (median(field_only_baseline) + median(path_only_baseline)) <= 1.10`
- Correctness gate:
  - mixed path payload must match the standalone path payload
  - mixed field total must match the standalone field total

### Gate: `warm_cache_trace_many_reuse`

- Formula: `steady_state_median_per_request / cold_frame_per_request <= 1.00`
- Correctness gate:
  - final-frame results must preserve per-request `tx_pos`
  - final-frame results must preserve per-request overridden RX positions
  - every final-frame result must report persistent PathMonitor diffraction-state cache hits

## Current Baseline Run

Current frozen baseline from `python -m tests.support.bin.benchmark_path_monitor_phase6 --strict-gates` on the reviewed branch:

- `default_first_order_path_export`: `17.60 ms`
- `explicit_multi_order_path_export`: `18.60 ms`
- `geometry_off_path_export`: `8.82 ms`
- `geometry_on_path_export`: `11.16 ms`
- `field_only_baseline`: `11.11 ms`
- `path_only_baseline`: `22.71 ms`
- `mixed_field_path_trace`: `31.60 ms`
- `warm_cache_trace_many`: cold `18.41 ms/request`, warm steady-state `16.59 ms/request`

Observed gate ratios:

- `first_order_vs_multi_order_bounded`: `0.946`
- `geometry_off_vs_on_bounded`: `0.791`
- `mixed_monitor_vs_separate_bounded`: `0.934`
- `warm_cache_trace_many_reuse`: `0.901`

All rollout gates passed on the reviewed branch.

## Phase Closure Table

### Phase 1

- Benchmark delta: default first-order PathMonitor export was previously measured at about `22.6 ms` versus about `57.7 ms` for explicit two-order export on the double-wedge path workload.
- Correctness gate: PathMonitor default must stay first-order unless the monitor requests a deeper order explicitly.

### Phase 2

- Benchmark delta: repeated same-`z` path traces were previously measured at about `25.9 ms` without reuse versus about `22.8 ms` with PathMonitor diffraction state-prep reuse.
- Correctness gate: repeated traces must report cache hits, and scene updates must invalidate the cache.

### Phase 3

- Benchmark delta: `assembly.from_raw` was previously measured at about `8.5 ms` on the replay-in-assembly path versus about `4.2 ms` on the shipped fast path.
- Correctness gate: no-geometry reflection export must not replay reflection chains in assembly and must preserve the selected path payload.

### Phase 4

- Benchmark delta: reduced path-export diffraction state layout was previously measured at about `30.4 ms` total versus about `25.4 ms` after the reduced layout shipped.
- Correctness gate: reduced-layout path export must preserve path completeness and geometry-on correctness.

### Phase 5

- Benchmark delta: prepared reflection EPC descriptors were previously measured at about `1.16 ms` versus about `1.13 ms` on the reflection-family EPC microbenchmark.
- Correctness gate: prepared-descriptor replay must match direct replay and remain compatible with multi-TX, multi-RX, and trace-time monitor overrides.

### Phase 6

- Benchmark delta: no new acceleration target. The deliverable is the frozen benchmark matrix and rollout gates in this document plus the strict benchmark command.
- Correctness gate: all four Phase 6 rollout gates must pass before new PathMonitor performance work is treated as the updated baseline.

## Maintenance Rules

- If a new PathMonitor optimization changes the intended default workload, update both this document and `benchmark_path_monitor_phase6.py` in the same change.
- If a new hot path matters for default rollout decisions, add it as a new matrix item before changing the gate thresholds.
- Do not silently repurpose an existing benchmark ID for a different workload shape.
