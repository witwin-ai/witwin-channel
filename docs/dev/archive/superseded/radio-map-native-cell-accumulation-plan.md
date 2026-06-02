# Radio-Map Native Cell-Accumulation Plan

Status: Active
Category: Plan
Last reviewed: 2026-04-04

## Objective

Make `RadioMapMonitor` converge toward the production radio-map architecture used
by Sionna-style coverage solvers while staying within this repository's
standards:

1. radio maps remain first-class `Scene + Tracer + Result` outputs,
2. the default production radio-map path becomes direct in-loop accumulation
   into cells,
3. dense runtime cost is driven by traced samples and path families rather than
   by exporting a path-times-receiver product,
4. hot paths stay Dr.Jit-native at the boundary and CUDA-native inside,
5. `FieldMonitor` and `RadioMapMonitor` continue to share low-level geometry and
   kernel infrastructure only where that reuse is structurally correct.

This document is the focused follow-up plan for the remaining architecture and
kernel work after the current `RadioMapMonitor` baseline, coherent native path,
and explicit `cell_accumulation` parity path.

## Why This Plan Exists

The current repository now has:

1. a first-class `RadioMapMonitor`,
2. a baseline reducer for incoherent and coherent metrics,
3. an explicit `cell_accumulation` path that avoids baseline radio-map path
   export,
4. a radiomap-owned scheduler,
5. a native coherent axis-aligned path.

However, the current implementation still falls short of the target production
shape in two important ways:

1. the incoherent native path still performs part of scalarization and final
   power reduction outside the production CUDA accumulation loop,
2. the top-level orchestration still carries target-driven path-solver
   structure, while the desired production radio-map path should be sample- or
   ray-driven and accumulate contributions directly into cells.

The goal is to close those gaps explicitly instead of extending the current
hybrid architecture indefinitely.

## Implementation Snapshot

As of the current review, the repository has landed the core semantic and
runtime work for Phases 0 through 4 plus the supported default flip from
Phase 6 on the eligible axis-aligned incoherent path:

1. receiver-model selection is explicit and surfaced in result metadata,
2. tracer intent is split into explicit radio-map coherent/incoherent families,
3. the supported native incoherent production path accumulates directly into
   final cell buffers during reflection and diffraction replay,
4. scheduler metadata now records why tiling was or was not selected,
5. `auto` selects the native incoherent backend on supported workloads, and
   the benchmark harness now freezes the opt-in `512 x 512` dense-map gates.

The main remaining architectural item from this plan is the Phase 5 goal of a
true solver-native multi-transmitter radio-map TX stack. The current
implementation now streams `trace_many(...)` radio-map aggregation without
materializing an extra dense RSS stack tensor, but it still performs
post-trace aggregation rather than solver-native TX-stack accumulation.

## Target Architecture

The production radio-map stack should be organized around the following split.

### 1. Public Radio-Map Contract

`RadioMapMonitor` remains the public entrypoint, but the semantic contract is
made explicit:

1. default production metric is non-coherent cell-averaged received power,
2. coherent radio maps are supported as an explicit opt-in mode,
3. receiver model is explicit:
   - `matched_isotropic` for Sionna-style default coverage semantics,
   - `projected_polarized` for current polarization-projected workflows,
4. quadrature remains explicit and fixed for deterministic validation,
5. transmitter association and SINR remain result-level derived metrics.

### 2. Radio-Map Execution Intent

`Tracer` should treat radio maps as their own execution family instead of
feeding them through a `path_export` intent and then repairing the structure
inside `trace_radio_map_monitor(...)`.

The top-level intent split should become:

1. `field`
2. `path`
3. `radio_map_coherent`
4. `radio_map_incoherent`

This does not require duplicate geometry logic. It only makes the execution
contract explicit at the tracer boundary.

### 3. Radio-Map Cell Accumulator

Production radiomap accumulation should own explicit cell buffers:

1. `power_buffer`
2. optional `coherent_real_buffer`
3. optional `coherent_imag_buffer`
4. optional per-component diagnostics buffers

The kernel-facing contract should be:

1. trace a family contribution,
2. compute the target cell index or compact cell-hit list,
3. atomic-add directly into the final monitor cell buffers,
4. avoid materializing exported per-path records for dense production runs.

### 4. Shared Infra Boundary

The parts that should remain shared with other monitors are:

1. scene compilation,
2. reflection-family discovery,
3. diffraction state preparation,
4. low-level EPC descriptors,
5. visibility logic,
6. CUDA custom-op extension plumbing.

The parts that should become radio-map-owned are:

1. execution intent,
2. scheduling policy,
3. cell-hit mapping,
4. accumulation buffers,
5. result assembly,
6. production backend selection policy.

## Design Direction For Direct In-Loop Cell Accumulation

The intended production path is not:

1. export path payloads,
2. gather receiver indices,
3. reduce onto cells afterwards.

The intended production path is:

1. trace or replay a contribution family,
2. intersect that family with the measurement surface during the same solver
   pass,
3. compute the corresponding cell index immediately,
4. accumulate the contribution into the cell buffer before leaving the kernel or
   tightly-coupled native replay loop.

For axis-aligned planar radio maps, this means:

1. reflection EPC computes the target hit point on the measurement plane,
2. diffraction replay computes the target hit point or valid receiver-plane
   crossing,
3. the native grid adapter converts that hit point to a cell index,
4. the kernel performs `atomicAdd` into the cell buffers.

For fixed quadrature, each quadrature sample set can still be handled
independently, but each set must accumulate directly into its own cell buffer
instead of exporting per-path products.

## Scope And Non-Goals

In scope for this plan:

1. axis-aligned planar radio maps,
2. default first-order diffraction,
3. native incoherent production path,
4. coherent-path cleanup where it still depends on field-style abstractions,
5. result semantics needed to align with Sionna-style coverage maps,
6. benchmark and parity gates for dense maps.

Not in scope for the first production closure:

1. mesh radio-map surfaces,
2. higher-order diffraction beyond the already supported repository contract,
3. arbitrary smoothing of cell-boundary discontinuities,
4. CPU fallback paths,
5. replacing the existing baseline reducer, which remains the reference parity
   implementation.

## Current Gaps

The main remaining gaps are:

1. `trace_radio_map.py` still mixes backend dispatch, state preparation,
   quadrature reduction, diagnostics, and result packaging.
2. `cell_accumulation` still performs final scalarization and power reduction in
   Python/Dr.Jit orchestration around the native replay path.
3. the tracer still describes radiomap work as `path_export` intent.
4. the default receiver model is still closer to projected-polarization path
   reduction than to Sionna's matched-isotropic default radio-map semantics.
5. `auto` still keeps incoherent maps on the baseline backend because the native
   ownership boundary is not complete enough yet.

## Phased Plan

### Phase 0. Freeze The Semantic Contract

Deliverables:

1. document and implement explicit receiver-model selection for radiomaps,
2. define the exact meaning of `incoherent` and `coherent` in public metadata,
3. keep `projected_polarized` available for backward-compatible workflows,
4. make `matched_isotropic` the target default for Sionna-aligned coverage maps.

Acceptance:

1. baseline reference tests cover both receiver models,
2. metadata records receiver model and combine mode,
3. docs clarify that coherent maps are not the default coverage semantics.

### Phase 1. Separate Radio-Map Execution From Path-Export Intent

Deliverables:

1. add radio-map-specific execution intent in tracer orchestration,
2. split `trace_radio_map.py` into:
   - backend resolution,
   - state preparation,
   - accumulator invocation,
   - result assembly,
3. keep the existing baseline path as the parity reference path.

Acceptance:

1. `Tracer.trace(...)` and `Tracer.trace_many(...)` keep public behavior,
2. runtime metadata reports explicit radiomap execution intent,
3. radiomap code no longer depends on path-export-specific naming for native
   execution.

### Phase 2. Native Reflection Cell Accumulation

Deliverables:

1. introduce a radiomap-specific reflection cell accumulator that computes cell
   hits and accumulates directly into cell buffers,
2. remove reflection-side Python scatter-reduce from the production incoherent
   path,
3. keep reflection-family discovery shared, but keep accumulation ownership
   radiomap-local.

Acceptance:

1. baseline-vs-native parity on wall and multipath fixtures,
2. runtime backend metadata distinguishes reflection cell accumulation from
   replay-only modes,
3. `512 x 512` reflection-only dense maps do not regress in memory.

### Phase 3. Native Diffraction Cell Accumulation

Deliverables:

1. extend the diffraction native path so scalarization and power accumulation are
   fused into the radiomap accumulation route,
2. compute valid cell hits and atomic-add directly into radiomap cell buffers,
3. keep state preparation shared, but move the final accumulation endpoint fully
   under radiomap kernel ownership.

Acceptance:

1. wall parity passes for direct diffraction and mixed reflection-diffraction,
2. the production runtime path no longer reports
   `native_utd_pair_vector_replay` for the default incoherent backend,
3. dense-map memory and timing no longer show the previous replay-orchestration
   bottleneck.

### Phase 4. Sample-Driven Production Scheduler

Deliverables:

1. replace the remaining target-driven scheduling assumptions with radio-map
   sample-driven scheduling,
2. make scheduler decisions depend on traced-family workload and cell-hit density
   rather than inherited field-style receiver tiling assumptions,
3. allow direct dense replay when tiling does not reduce enough work.

Acceptance:

1. scheduler metadata explains why tiling was or was not used,
2. `512 x 512 center` stays stable across repeated runs,
3. no dense-map path explodes into many tiny tile calls without a measured
   workload reduction.

### Phase 5. Multi-TX Native Radio-Map Accumulation

Deliverables:

1. move multi-transmitter radiomap accumulation closer to solver-native
   production execution,
2. reduce the dependence on `trace_many()` post-aggregation for core radiomap
   use cases,
3. keep result-level SINR and association utilities, but build them on native TX
   stacks when possible.

Acceptance:

1. RSS and association parity remain correct,
2. solver metadata records native TX-stack execution when used,
3. multi-TX dense maps do not require exporting separate dense products per TX
   before aggregation.

### Phase 6. Production Default Flip

Deliverables:

1. promote native incoherent accumulation to the `auto` backend when eligibility
   conditions are met,
2. retain baseline as explicit parity and fallback mode,
3. freeze benchmark gates for dense maps before the default flip.

Acceptance:

1. strict parity suite passes,
2. dense-map benchmark matrix shows consistent wins or neutral results over
   baseline on supported workloads,
3. `auto` never selects a native path with known parity gaps.

## Benchmark And Acceptance Matrix

The plan should be considered complete only when the following matrix is stable.

### Correctness

1. baseline-vs-native parity for:
   - LoS-only,
   - reflection-only,
   - diffraction-only,
   - mixed reflection and diffraction,
   - coherent and incoherent where supported,
   - both receiver models.
2. multi-TX RSS, SINR, and association parity.
3. oriented-plane baseline behavior remains unchanged while axis-aligned native
   production paths evolve.

### Performance

Required benchmark cases:

1. `512 x 512 center`
2. reflection-dominant scene
3. diffraction-dominant scene
4. mixed wall scene
5. multi-TX scene

Required performance outcome:

1. native incoherent must at least match baseline on supported axis-aligned
   dense maps,
2. dense-map runtime must not scale through path-export payload growth,
3. memory usage must remain bounded by cell buffers, solver state, and tile
   descriptors rather than per-path export products.

## Validation Rules

Before each phase is considered landed:

1. update the active plan and benchmark notes,
2. run targeted GPU pytest coverage,
3. run the radiomap benchmark harness,
4. record any parity exceptions in `docs/dev/bugs/known-bugs.md`,
5. do not flip `auto` defaults until parity and dense-map benchmarks both pass.

## Risks

The main risks are:

1. coherent and incoherent semantics can drift if they continue sharing too much
   reducer logic,
2. fused accumulation kernels can hide parity bugs unless per-component
   diagnostics stay available,
3. direct cell accumulation can become nondifferentiable in unexpected places if
   cell-hit mapping is not treated carefully,
4. overgeneralizing the first production kernel to oriented or mesh surfaces too
   early will slow down the axis-aligned closure.

## Immediate Next Actions

The next concrete implementation steps should be:

1. add an explicit receiver-model field to `RadioMapMonitor` and result metadata,
2. split `trace_radio_map.py` so backend resolution and result assembly are no
   longer interleaved with native execution,
3. design the native reflection cell-accumulation contract around
   `hit_point -> cell_index -> atomicAdd`,
4. extend the diffraction native path to own scalarization and power
   accumulation in the same production route,
5. freeze a dense-map benchmark gate that blocks regressions before the `auto`
   backend is widened.
