# Trace Architecture Boundary Refactor

Status: Completed
Category: Archive
Last reviewed: 2026-04-05

## Completion Summary

This refactor plan was completed on 2026-04-05.

Implemented boundary changes:

- monitor packages now use short local module names such as `monitor.py`, `trace.py`, `result.py`, `helpers.py`, and `samples.py`,
- field, path, and radio-map result types now live under their owning monitor packages,
- root `witwin/channel/result.py` now owns only the aggregate `Result`,
- `Tracer` is now a thinner facade over `TraceSession`, `TraceCacheManager`, and monitor-specific executors,
- scene runtime state now lives under `witwin/channel/scene/runtime_state.py` with helper modules split into `builder.py`, `query_backend.py`, and `runtime_queries.py`.

Validation completed during implementation:

1. import smoke for `witwin.channel`, result modules, trace session/cache modules, and `SceneRuntime`
2. `python -m pytest tests/trace/test_path_monitor.py --gpu -q`
3. `python -m pytest tests/scene/test_radio_map_monitors.py --gpu -q`
4. `python -m pytest tests/backend/test_runtime_backend_switch.py tests/scene/test_field_monitors.py::test_scene_accepts_field_monitors_and_resolves_them tests/scene/test_radio_map_monitors.py tests/trace/test_path_monitor.py --gpu -q`

## Purpose

This document defines a repository-specific refactor plan to:

- reduce large files in the trace and result layers,
- lower coupling between declarative scene data, trace orchestration, and monitor-specific payload shaping,
- stabilize module ownership boundaries,
- move monitor-specific result code into the corresponding monitor folders.

The plan keeps the current public model intact:

- `Scene`
- `Tracer`
- `Result`

The goal is not to force a full OOP rewrite of solver internals. Numerical kernels, state-array transforms, and DrJit-native hot paths should remain function-oriented.

## Scope

In scope:

- `witwin/channel/scene/`
- `witwin/channel/trace/`
- `witwin/channel/monitors/`
- `witwin/channel/result.py`

Out of scope:

- `kernels/` CUDA and native implementation ownership
- core numerical model changes
- public API redesign of `Scene + Tracer + Result`
- new compatibility shims

## Current Problems

### 1. `Scene` owns too much runtime state directly

`Scene` is the public declarative object, but it also owns a large mutable runtime surface:

- compiled vertices/faces
- RayD scene handle
- query backend handle
- edge caches
- triangle material/runtime buffers
- mesh versioning

That state is initialized in [scene.py](E:/Code/witwin-platform/channel/witwin/channel/scene/scene.py) but mostly mutated by external functions in [builder.py](E:/Code/witwin-platform/channel/witwin/channel/scene/builder.py) and [runtime.py](E:/Code/witwin-platform/channel/witwin/channel/scene/runtime.py).

This creates an awkward ownership split: the class owns the state, but helper modules own the behavior.

### 2. `Tracer` is still carrying too many responsibilities

[tracer.py](E:/Code/witwin-platform/channel/witwin/channel/trace/tracer.py) currently mixes:

- config resolution,
- monitor resolution,
- monitor override logic,
- per-monitor dispatch,
- reflection-detail reuse,
- diffraction cache ownership,
- `trace_many()` aggregation,
- scene update helpers.

This is stable enough for users, but it is too broad to remain the only orchestration owner.

### 3. Result code is centralized instead of living with the owning monitor type

[result.py](E:/Code/witwin-platform/channel/witwin/channel/result.py) currently holds:

- tensor adapters,
- field result objects,
- radio-map result objects,
- path result objects,
- raw path normalization and replay helpers,
- the top-level aggregate `Result`.

This file crosses multiple ownership boundaries. It is the clearest candidate for splitting.

### 4. Some boundaries are already drifting

The current tree contains a broken import path for radio-map tracing in `Tracer`, which indicates refactor work already happened without a final ownership pass. Boundary cleanup should happen before further functional work is layered on top.

## Refactor Principles

1. Keep solver kernels and numeric transforms function-oriented.
2. Move orchestration, caching, and payload assembly toward explicit owner objects.
3. Put monitor-specific result types inside the matching monitor folder.
4. Inside a monitor package, prefer short file names such as `monitor.py`, `trace.py`, `result.py`, `samples.py`, and `helpers.py`. Do not repeat the package name in the file name.
5. Avoid thin re-export shims. If a module remains at the root, it must own meaningful logic.
6. Prefer small objects with narrow lifetimes over large permanent service classes.
7. Preserve the current public architecture: `Scene + Tracer + Result`.

## Target Ownership Model

### Scene Layer

- `Scene` remains the public declarative object.
- `SceneRuntime` becomes the owner of compiled runtime state.
- compiler functions remain implementation helpers, but they act on `SceneRuntime`, not on the public `Scene` directly.

### Trace Layer

- `Tracer` becomes a thin facade.
- `TraceSession` owns one trace call's resolved context, caches, and monitor dispatch state.
- monitor-specific execution lives in dedicated executors.

### Monitor Layer

- each monitor package owns:
  - its declarative monitor object,
  - its trace orchestration entrypoint,
  - its monitor-specific result types and builders.

### Result Layer

- root `result.py` remains only for the aggregate `Result` object plus shared cross-monitor result protocols/helpers that cannot be owned by a single monitor package.
- monitor-specific result classes move into the corresponding folders:
  - `monitors/field/`
  - `monitors/path/`
  - `monitors/radio_map/`

## Target File Layout

```text
witwin/channel/
  result.py                              # aggregate Result only
  scene/
    scene.py                             # public Scene
    runtime_state.py                     # SceneRuntime
builder.py                           # build/rebuild helpers
    query_backend.py                     # query backend objects
    runtime_queries.py                   # edge/material/runtime query helpers
  trace/
    tracer.py                            # thin public facade
    session.py                           # TraceSession
    cache.py                             # diffraction/reflection cache owners
    executors/
      field.py                           # FieldTraceExecutor
      path.py                            # PathTraceExecutor
      radio_map.py                       # RadioMapTraceExecutor
  monitors/
    common.py                            # monitor-shared normalization only
    field/
      monitor.py
      trace.py
      result.py                          # MonitorResult, field-specific payload types
    path/
      monitor.py
      trace.py
      result.py                          # PathResult + path payload builders
    radio_map/
      monitor.py
      trace.py
      result.py                          # RadioMapResult
      samples.py
      helpers.py
      scheduler.py
```

## Naming Rule For Monitor Packages

The package directory already carries the domain context. File names inside the package should therefore stay short.

Preferred pattern:

- `monitors/field/monitor.py`
- `monitors/field/trace.py`
- `monitors/field/result.py`
- `monitors/path/monitor.py`
- `monitors/path/trace.py`
- `monitors/path/result.py`
- `monitors/radio_map/monitor.py`
- `monitors/radio_map/trace.py`
- `monitors/radio_map/result.py`
- `monitors/radio_map/samples.py`
- `monitors/radio_map/helpers.py`

Avoid patterns like:

- `field_monitor.py`
- `path_monitor.py`
- `radio_map_monitor.py`
- `trace_field.py`
- `trace_path.py`
- `trace_radio_map.py`
- `trace_radio_map_samples.py`
- `trace_radio_map_helpers.py`

The same rule applies to new modules created during the refactor unless a shorter name would become ambiguous at the package level.

## Required Result Placement

This plan adopts the following explicit ownership rule.

### Field results

Move field-monitor result types from the current root result module into:

- `witwin/channel/monitors/field/result.py`

Expected ownership:

- `MonitorCoordinates`
- `MonitorField`
- `MonitorJones`
- `MonitorVector`
- `MonitorResult`

### Path results

Move path-monitor result types and the heavy path normalization pipeline into:

- `witwin/channel/monitors/path/result.py`

Expected ownership:

- `PathResult`
- path raw payload normalization
- path replay helpers
- path tensor packing and filtering logic

If the path conversion code is still too large after the move, split it under the same folder, for example:

- `monitors/path/result.py`
- `monitors/path/result_builders.py`
- `monitors/path/result_normalization.py`

### Radio-map results

Move radio-map result types into:

- `witwin/channel/monitors/radio_map/result.py`

Expected ownership:

- `RadioMapCoordinates`
- `RadioMapResult`

### Root aggregate result

Keep only the cross-monitor aggregate object in:

- `witwin/channel/result.py`

Expected ownership:

- `Result`
- shared aggregate helpers that are genuinely cross-monitor and not just historical leftovers

`result.py` must not continue to own monitor-specific payload normalization after the refactor.

## Target Class Boundaries

### `Scene`

Owns:

- structures
- monitors
- metadata
- user-facing scene mutation API

Does not directly own low-level runtime mutation behavior beyond delegating to a runtime owner.

### `SceneRuntime`

Owns:

- compiled mesh buffers
- query backend handle
- RayD scene handle
- edge runtime caches
- triangle runtime data
- mesh version and dirty flags

Primary methods:

- `rebuild()`
- `update_vertices()`
- `ensure_edge_runtime()`
- `get_edge_data()`

### `Tracer`

Owns:

- validated top-level configuration
- public `trace()` and `trace_many()` API
- creation of per-call sessions

Does not own monitor-specific execution logic.

### `TraceSession`

Owns:

- resolved transmitter position
- trace-local monitor overrides
- reflection-detail reuse cache
- diffraction-state cache access
- per-call dispatch bookkeeping

Primary methods:

- `run()`
- `run_monitor()`
- `build_result()`

### `FieldTraceExecutor`

Owns:

- `FieldMonitor` trace orchestration
- LoS/reflection/diffraction composition for field outputs
- field metadata assembly

### `PathTraceExecutor`

Owns:

- `PathMonitor` trace orchestration
- grouped receiver handling
- path export metadata assembly
- construction of path result payloads

### `RadioMapTraceExecutor`

Owns:

- `RadioMapMonitor` trace orchestration
- scheduler selection
- coherent/incoherent accumulation branching
- radio-map metadata assembly

### `Result`

Owns:

- aggregate monitor map
- aggregate path monitor map
- primary monitor access

Does not own field/path/radio-map payload normalization.

## Phased Plan

### Phase 0: Stabilization

Goals:

- fix broken or drifting trace-module imports,
- decide final module names before moving logic,
- document owner boundaries.

Tasks:

- restore or replace the missing radio-map trace entrypoint referenced by `Tracer`,
- choose final module names for `SceneRuntime`, `TraceSession`, and executor modules,
- freeze the short-name convention for files inside monitor packages,
- add this document to the active docs index.

Acceptance:

- importing `witwin.channel.trace.tracer` works in the intended environment,
- no trace module imports a missing monitor module,
- monitor package target names use short local names instead of repeated package prefixes,
- target module names are frozen in this document.

### Phase 1: Result relocation

Goals:

- split `result.py` by monitor ownership first,
- keep behavior unchanged,
- make later trace refactors easier.

Tasks:

- move field result classes into `monitors/field/result.py`,
- move path result classes and path normalization code into `monitors/path/result.py`,
- move radio-map result classes into `monitors/radio_map/result.py`,
- shrink root `result.py` to aggregate-only ownership.

Acceptance:

- `witwin.channel.result.Result` remains the aggregate entrypoint,
- `MonitorResult` is defined only under `monitors/field/`,
- `PathResult` is defined only under `monitors/path/`,
- `RadioMapResult` is defined only under `monitors/radio_map/`,
- root `result.py` no longer contains monitor-specific result construction logic,
- public imports used by tests continue to work after direct-import updates.

### Phase 2: Tracer thinning

Goals:

- make `Tracer` a facade,
- move per-call state into a session owner,
- isolate monitor-specific dispatch.

Tasks:

- add `TraceSession`,
- move reflection-detail cache coordination into session scope,
- move monitor dispatch into executor classes,
- keep `Tracer.trace()` and `Tracer.trace_many()` as public API only.

Acceptance:

- `Tracer` no longer contains monitor-specific payload assembly,
- monitor execution for field/path/radio-map happens through dedicated executors,
- trace-local cache logic is not duplicated across monitors,
- `trace_many()` aggregation still behaves identically for radio-map outputs.

### Phase 3: Scene runtime split

Goals:

- separate public scene declaration from compiled runtime ownership,
- stop mutating large runtime state through unrelated helper modules.

Tasks:

- introduce `SceneRuntime`,
- move runtime state fields out of `Scene` where practical,
- retarget compile/runtime helper functions to the runtime owner,
- keep `Scene` public API unchanged.

Acceptance:

- runtime buffers and backends are owned by `SceneRuntime`,
- `Scene` remains the user-facing declarative object,
- scene rebuild and vertex-update paths no longer depend on free functions mutating arbitrary `Scene` internals,
- query and edge runtime access go through stable runtime-owned APIs.

### Phase 4: Cleanup and line-count reduction

Goals:

- make each file easier to own,
- prevent another god-object cycle.

Tasks:

- split any remaining oversized result or executor modules,
- rename repeated monitor-local file names to short package-local names where needed,
- remove stale imports and dead helper paths,
- update development docs if the final layout differs from this draft.

Acceptance:

- no single monitor result module grows back into a second all-in-one result file,
- `Tracer`, root `result.py`, and public scene modules each have a narrow ownership description that matches the code,
- monitor-local files do not keep redundant domain prefixes when the package path already provides the context,
- there are no compatibility shims whose only purpose is to re-export a moved symbol.

## Acceptance Matrix

The refactor is complete only when all conditions below are true.

### Structural acceptance

- monitor-specific result classes live under the matching monitor folder.
- root `result.py` owns only aggregate or truly shared result concerns.
- `Tracer` is not the owner of field/path/radio-map orchestration details.
- `Scene` is not the owner of all mutable runtime implementation details.

### Dependency acceptance

- dependency direction remains aligned with the repository layering rules.
- solver modules do not import monitor result modules.
- monitor result modules do not import back upward into `Tracer`.
- no new root-level Markdown or duplicate architecture plan is introduced outside `docs/dev/`.

### Behavioral acceptance

- no numerical behavior changes are introduced by the boundary refactor.
- current reflection, diffraction, path export, and radio-map tests continue to pass.
- monitor metadata payloads remain semantically equivalent unless explicitly documented.

### Maintainability acceptance

- result ownership can be inferred from path alone.
- each moved file has one clear owner topic.
- large helper blocks are moved with their owning domain instead of being left behind as generic leftovers.

## Validation Strategy

Prefer targeted validation after each phase, then broader regression coverage.

Recommended checkpoints:

1. import smoke checks for `Scene`, `Tracer`, `Result`, and monitor result modules
2. targeted monitor tests for field, path, and radio-map
3. targeted scene update and runtime query tests
4. broader `pytest tests --gpu` once the phase is internally stable

Use the canonical workflow in [test-and-acceptance-workflow.md](E:/Code/witwin-platform/channel/docs/dev/standards/test-and-acceptance-workflow.md).

## Risks

### Risk: accidental compatibility shims

Moving result classes can tempt a thin root re-export layer. This should be avoided unless the root module still owns meaningful aggregate logic.

### Risk: path result split creates a second giant file

`PathResult` is the heaviest result type. If needed, split builders/normalizers within `monitors/path/` instead of keeping everything in one file.

### Risk: runtime split leaks through the public API

`SceneRuntime` must stay an implementation detail. Public scene construction still has to look like `Scene(structures=[...])`.

### Risk: trace session over-engineering

`TraceSession` should stay small and trace-call scoped. It should not turn into a second `Tracer`.

## Completion Condition

This plan is considered complete when:

- the active code matches the ownership rules in this document,
- result classes are located under the corresponding monitor folders,
- root `result.py`, `Tracer`, and `Scene` each have stable and narrow responsibilities,
- the remaining function-oriented numeric core stays intact.
