# Cell-State Path Memory Optimization Plan

## 1. Scope

This document defines a phased optimization plan for the memory blow-up caused by storing large
path/state payloads per monitor cell (or per receiver-cell equivalent) in the current channel
tracing pipeline.

The plan is written for the current `Scene + Tracer + Result` architecture and assumes the
repository's existing constraints:

- GPU-first execution remains the default.
- New user-facing tracing paths must stay compatible with the current declarative scene model.
- The optimization must preserve correctness first and should not introduce heuristic smoothing or
  ad hoc approximations in diffraction/intersection derivatives.

This is a development-plan document. It does not itself change runtime behavior.

## 2. Problem Statement

### 2.1 Symptom

The current implementation can require excessive memory when:

- a large number of diffraction/reflection states are generated,
- each state carries a large path-history payload,
- and that payload is then materialized again at cell or receiver granularity.

In practical terms, the failure mode is not just "too many cells" or "too many states". The real
issue is a multiplicative expansion:

- `O(states)` for the global state table,
- then `O(states x cells)` or `O(states x receivers)` when a state payload is expanded into
  per-cell/per-receiver structures.

### 2.2 Why the Current State Object Is Expensive

The packed diffraction-state buffer is already large before any cell duplication:

- `PACKED_CORE_FLOATS = 93` in
  `witwin/channel/kernels/packed_state/packed_state.h`
- each history slot adds `2` more packed entries
- packed stride is padded to 4-float alignment

So the packed state stride is:

```text
stride(history_size) = align4(93 + 2 * history_size)
bytes_per_state = 4 * stride(history_size)
```

Examples:

- `history_size = 1` -> `align4(95) = 96 floats` -> `384 B/state`
- `history_size = 2` -> `align4(97) = 100 floats` -> `400 B/state`
- `history_size = 3` -> `align4(99) = 100 floats` -> `400 B/state`
- `history_size = 4` -> `align4(101) = 104 floats` -> `416 B/state`

That cost is for one global state entry. It becomes untenable if repeated per cell.

### 2.3 Where the Current Design Repeats Data

The current implementation repeats path-history data in two ways:

1. State-to-state history copying

- higher-order diffraction construction explicitly writes full
  `path_edge_idx_*` / `path_reflection_depth_*` arrays into each child state
- hotspots:
  - `witwin/channel/trace/diffraction/builders/higher.py`
  - `witwin/channel/trace/diffraction/builders/prefix.py`
  - `witwin/channel/trace/diffraction/builders/tx.py`

2. State-to-cell expansion

- field accumulation paths are already mostly streaming and scatter directly into grid outputs
- but any path-level or debug path that keeps full per-cell path/state payloads risks exploding to
  `O(states x cells)`
- relevant code paths:
  - `witwin/channel/monitors/field/grid_diffraction.py`
  - `witwin/channel/kernels/suffix_grid/*`
  - `witwin/channel/monitors/path/collectors.py`

### 2.4 Current Good Direction That Should Be Preserved

Several existing components already follow the correct pattern:

- stream work in chunks instead of materializing full Cartesian products
- scatter field contributions directly into the target grid
- apply explicit state budgets before later expansion

Relevant examples:

- `witwin/channel/trace/diffraction/constants.py:_cartesian_chunk_size`
- `witwin/channel/kernels/suffix_grid/native_impl.py`
- `witwin/channel/monitors/field/grid_diffraction.py`
- `witwin/channel/trace/diffraction/builders/__init__.py`

The optimization plan below extends this pattern across the full pipeline instead of introducing a
second storage-heavy execution path.

## 3. Optimization Goals

### 3.1 Primary Goals

- Eliminate full path/state duplication at cell granularity.
- Eliminate repeated full-history copying when constructing child states.
- Preserve numerical correctness of field accumulation and path replay.
- Keep the default field-monitor path GPU-native and streaming.
- Make path export and audit data optional and lazily reconstructable.

### 3.2 Secondary Goals

- Reduce peak VRAM.
- Reduce pack/unpack/concat pressure for state arrays.
- Reduce the size of temporary Cartesian products.
- Keep AD behavior correct for supported differentiable paths.

### 3.3 Non-Goals

- No new CPU fallback architecture.
- No lossy approximation as the main fix.
- No redesign of the public `Scene + Tracer + Result` model.
- No requirement that every internal debug/audit structure remain materialized at all times.

## 4. Design Principles

1. A cell must not own a full copy of a path state.
2. A state must not own a full copy of all ancestor history if a parent link is sufficient.
3. Field accumulation should remain streaming and scatter-based.
4. Path-level output should be reconstructed on demand from compact lineage data.
5. Expensive metadata must be optional and split from the hot propagation path.
6. Every phase must have benchmark and correctness gates before rollout.

## 5. Target End State

The desired final architecture is:

- a compact global state table for propagation-critical data,
- a separate lineage graph for path reconstruction,
- sparse cell-to-state references only when path export is explicitly requested,
- pure scatter-based field accumulation for normal monitor tracing,
- optional cold metadata for audit/debug/geometry export.

At the end of the roadmap:

- field tracing should not materialize per-cell path payloads,
- state construction should not duplicate full path histories,
- path monitors should reconstruct paths from lineage rather than from duplicated arrays,
- budgets should act as guardrails rather than as the only protection against structural blow-up.

## 6. Phased Plan

## Phase 0: Baseline, Instrumentation, and Failure Characterization

### Objective

Establish hard numbers for where memory is spent and which pipeline stages dominate peak VRAM.

### Deliverables

- a repeatable benchmark entrypoint under `tests/support/bin/`
- phase timing and memory reporting in the diffraction builder and monitor tracing paths
- a standard benchmark matrix covering:
  - dense field monitor
  - high `max_diffractions`
  - high reflection-ray counts
  - path-monitor workloads

### Required Metrics

- `n_states` per order
- `history_size`
- packed-state stride
- estimated bytes per state
- peak states before and after pruning
- number of Cartesian pairs created per chunk
- peak CUDA memory
- wall time by stage:
  - state preparation
  - higher-order expansion
  - inserted reflection construction
  - suffix accumulation
  - path collection

### Code Touchpoints

- `witwin/channel/monitors/field/trace_field.py`
- `witwin/channel/trace/diffraction/builders/__init__.py`
- `witwin/channel/trace/diffraction/api.py`
- benchmark script under `tests/support/bin/`

### Acceptance Criteria

- one command can reproduce the memory profile for a fixed scene
- per-phase memory reports can identify whether the peak is caused by:
  - state storage
  - Cartesian expansion
  - path collection
  - suffix/grid accumulation

### Risks

- instrumentation itself can perturb timing
- benchmark scenes may miss the true pathological production case

## Phase 1: Immediate Containment Without Structural Redesign

### Objective

Stop the worst-case blow-ups before deeper refactors land.

### Changes

1. Make field mode and path-export mode explicitly different execution intents.
2. Ensure field mode does not retain geometry/audit/path payloads unless requested.
3. Tighten and expose state-budget defaults for stress workloads.
4. Apply earlier guardrails to mixed-path depth and inserted reflection counts.

### Specific Actions

- keep `diffraction_state_budget`,
  `inserted_reflection_state_budget`,
  and `max_inserted_reflections_per_path` explicit in stress-facing configurations
- ensure path-monitor collection does not run on field-only traces
- document a "memory-safe mode" profile in developer docs

### Code Touchpoints

- `witwin/channel/config.py`
- `witwin/channel/monitors/trace_common.py`
- `witwin/channel/trace/diffraction/api.py`
- `witwin/channel/monitors/path/*`

### Acceptance Criteria

- the largest current memory regressions are bounded by config alone
- field-only runs stop building unnecessary path-level payloads

### Risks

- this phase mitigates symptoms but does not remove structural duplication
- over-aggressive budgets can reduce coverage if used as defaults in accuracy mode

## Phase 2: Split Hot State From Cold Metadata

### Objective

Separate propagation-critical data from audit/debug/path-reconstruction metadata.

### Rationale

The current packed state stores many fields that are not needed in the hot accumulation path for
every execution mode. Carrying them through all kernels increases:

- packed buffer size
- gather/scatter cost
- concat cost
- pack/unpack traffic

### New Data Model

Introduce two logical layers:

- `HotStateArrays`
  - required for propagation and accumulation
  - e.g. edge position, edge direction, incident field, incident vector/Jones data, face operators,
    order, minimal depth counters
- `ColdStateMetadata`
  - optional path provenance and audit information
  - e.g. full path lineage, source-type diagnostics, geometry replay helpers, audit-only flags

### Candidate Hot Fields

- `edge_idx`
- `edge_pos`
- `edge_dir`
- `n0`
- `n_face_n`
- `wedge_n`
- `adjacent_face0`
- `adjacent_face1`
- `source_pos`
- `path_length_prefix`
- `first_interaction_pos`
- incident field/Jones/vector values needed by UTD and transport
- face operators/material response needed during propagation
- `order`
- minimal reflection-depth counters if needed for ownership classification

### Candidate Cold Fields

- `source_type_code`
- `approximation_mode_code`
- full path lineage
- optional geometry/audit tags
- replay-only helper structures

### Code Touchpoints

- `witwin/channel/trace/diffraction/state/arrays.py`
- `witwin/channel/kernels/packed_state/packed_state.h`
- `witwin/channel/kernels/packed_state/native_impl.py`
- `witwin/channel/trace/diffraction/state/audit.py`

### Acceptance Criteria

- field-only traces can execute without materializing cold metadata
- packed hot-state stride decreases materially
- audit/path replay can still be enabled explicitly

### Risks

- if hot/cold boundaries are chosen poorly, kernels may need extra gathers from cold data
- AD-sensitive code paths may depend on fields that look "cold" but are actually hot

## Phase 3: Replace Full History Arrays With Parent-Link Lineage

### Objective

Remove repeated copying of full path history into every child state.

### Rationale

This is the highest-value structural change. Today, when a new diffraction or inserted-reflection
state is created, the builder writes an entire set of `path_edge_idx_*` and
`path_reflection_depth_*` arrays into the child state. That causes lineage memory to scale with:

- number of states
- maximum history size

instead of only with the number of transitions.

### New Lineage Representation

Replace full-history slots with a parent-linked DAG or tree:

- `lineage_parent_state_idx`
- `lineage_last_edge_idx`
- `lineage_last_interaction_type`
- `lineage_last_reflection_depth_delta`
- `lineage_order`

Optional:

- `lineage_generation`
- `lineage_root_kind`

### Reconstruction Strategy

When a complete path sequence is needed:

1. start from the terminal state,
2. follow `parent_state_idx` backward,
3. collect edge/reflection events,
4. reverse the collected sequence to produce the forward path.

This reconstruction is only needed for:

- path-monitor exports
- audit reports
- debugging

It is not needed during normal field accumulation.

### Required Refactors

- builder code must stop writing `path_edge_idx_*` arrays
- path-monitor collectors must replay lineage instead of reading fixed history slots
- audit payloads must reconstruct sequences lazily
- packed-state format must remove or downgrade full-history storage

### Code Touchpoints

- `witwin/channel/trace/diffraction/builders/higher.py`
- `witwin/channel/trace/diffraction/builders/prefix.py`
- `witwin/channel/trace/diffraction/builders/tx.py`
- `witwin/channel/trace/diffraction/state/arrays.py`
- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/trace/diffraction/state/audit.py`

### Acceptance Criteria

- builders no longer allocate `path_edge_idx_*` / `path_reflection_depth_*` per child state
- full path replay remains correct for path-monitor output
- peak memory for higher-order state construction drops materially on high-order benchmarks

### Risks

- replay latency for path export will increase
- debugging lineage bugs is harder than reading flat history arrays

### Mitigation

- keep a temporary developer-only compatibility mode during migration
- add replay-vs-legacy parity tests on small scenes before removing the old layout

## Phase 4: Replace Per-Cell Path Payloads With Sparse Cell-to-State References

### Objective

Ensure cells never store full path/state payloads.

### Rationale

Even after state slimming, storing a complete path object per cell is the wrong scaling law. The
cell should only know which compact state contributions affect it.

### New Cell-Side Representation

Use sparse references when path export is requested:

- `cell_idx`
- `state_idx`
- `contribution`
- optional `tau`
- optional lightweight flags

Suggested storage layouts:

- COO during construction
- CSR after compaction for efficient per-cell traversal

### Rules

- Field monitors:
  - do not store per-cell path payloads
  - only scatter into accumulated outputs
- Path monitors:
  - may build sparse cell-to-state/path-reference tables
  - must not duplicate the full state payload per cell

### Code Touchpoints

- `witwin/channel/monitors/field/grid_diffraction.py`
- `witwin/channel/kernels/suffix_grid/*`
- `witwin/channel/monitors/path/collectors.py`
- any monitor-side path export buffers

### Acceptance Criteria

- no code path stores a full state/path object per cell
- path export memory scales with number of references, not with duplicated payload size
- field-monitor memory remains approximately independent of exported-path metadata

### Risks

- sparse structures complicate downstream formatting
- path-monitor output reshaping may require additional compaction work

## Phase 5: Move Pruning Earlier and Reduce Cartesian Expansion Before It Happens

### Objective

Reduce the number of candidate pairs that ever reach expensive per-cell or per-receiver stages.

### Rationale

Chunking limits temporary size, but it does not change total work. The bigger win is to reduce the
candidate set before Cartesian expansion.

### Candidate Early-Pruning Strategies

1. State power pruning

- prune weak states before `state x rx` or `state x cell` expansion

2. Coarse geometric culling

- bounding-box or directional culling before exact visibility tests

3. Top-K per order or per edge cluster

- retain only the strongest states where exact exhaustive retention is not required by mode

4. Receiver/tile-local culling

- only expand states into receiver tiles that are plausibly reachable/visible

### Mode Policy

- `accuracy` mode:
  - only explicit user budgets
  - no hidden heuristic pruning
- bounded-performance mode:
  - early pruning allowed under clearly documented rules

### Code Touchpoints

- `witwin/channel/trace/diffraction/builders/__init__.py`
- `witwin/channel/trace/diffraction/api.py`
- `witwin/channel/trace/diffraction/state/pruning.py`
- candidate construction in `builders/higher.py`

### Acceptance Criteria

- total candidate pairs created on stress scenes decrease materially
- accuracy mode remains semantically unchanged except for explicit user budgets

### Risks

- early pruning can silently change coverage if it leaks into accuracy mode
- culling heuristics may bias path families if not validated carefully

## Phase 6: Reduce Packed-State Width and Representation Cost

### Objective

After structural duplication is removed, shrink the representation itself.

### Rationale

Representation compression is only worth doing after the storage model is fixed. Otherwise the
project risks compressing the wrong thing.

### Candidate Optimizations

1. Narrow integer fields

- small enums and depth counters should become `uint8` or `uint16` where safe

2. Separate packed formats by mode

- field-only packed format
- replay/audit packed format

3. Reduce redundant stored fields

- remove values that can be recomputed cheaply from parent or scene data

4. Keep packed representation authoritative for hot operations

- avoid repeated pack/unpack churn around gather/concat/subset workflows

### Code Touchpoints

- `witwin/channel/kernels/packed_state/packed_state.h`
- `witwin/channel/kernels/packed_state/packed_state.cu`
- `witwin/channel/kernels/packed_state/native_impl.py`
- `witwin/channel/trace/diffraction/state/arrays.py`

### Acceptance Criteria

- hot packed-state stride is materially smaller than the current baseline
- gather/concat/subset remain correct and benchmarkable

### Risks

- mixed precision or narrow integer choices can introduce subtle bugs
- representation forks can increase maintenance cost if not disciplined

## Phase 7: Path Export Replay and Audit Refactor

### Objective

Rebuild path-monitor and audit functionality on top of compact lineage and sparse references.

### Rationale

This phase makes the structural changes user-viable. The path monitor must still provide:

- per-path amplitude
- delay
- angles
- interaction types
- optional geometry

but it should reconstruct those outputs from compact lineage, not from duplicated state payloads.

### Changes

1. Rewrite path collectors to replay from lineage DAGs.
2. Build geometry lazily only when `return_geometry=True`.
3. Allow tile- or receiver-chunked path export to cap peak memory.
4. Keep padded/masked public output, but construct it from sparse intermediate storage.

### Code Touchpoints

- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/monitors/path/trace_path.py`
- `witwin/channel/trace/reflection/api.py`
- related path-monitor result shaping code

### Acceptance Criteria

- public path output remains semantically equivalent
- path export peak memory no longer tracks full duplicated path payload size
- geometry export remains optional and lazy

### Risks

- receiver-path padding and sorting logic becomes more complex
- replay can become a runtime hotspot if done naively

### Mitigation

- chunk replay by receiver tile
- cache compact replay intermediates where they are shared

## Phase 8: Rollout, Validation, and Cleanup

### Objective

Safely replace the legacy storage model.

### Rollout Strategy

1. Introduce new internal structures behind developer/runtime switches.
2. Validate field parity first.
3. Validate path-monitor parity second.
4. Remove legacy full-history storage after tests and benchmarks are stable.

### Validation Matrix

- small deterministic scenes:
  - EPC parity
  - interaction sequence parity
- medium stress scenes:
  - peak VRAM comparison
  - runtime comparison
- gradient scenes:
  - finite gradients
  - parity within accepted tolerance
- path-monitor scenes:
  - path count parity
  - geometry parity when enabled

### Cleanup Tasks

- remove deprecated history-slot fields
- remove compatibility branches once stable
- update developer docs and internal benchmarks

## 7. Recommended Execution Order

The recommended order is:

1. Phase 0
2. Phase 1
3. Phase 3
4. Phase 4
5. Phase 2
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

### Why This Order

- Phase 1 stops the immediate bleeding.
- Phase 3 removes the largest structural duplication inside the state graph.
- Phase 4 removes the largest structural duplication at the cell layer.
- Phase 2 then becomes easier, because hot/cold boundaries are clearer after lineage is separated.
- Phase 6 should be delayed until the architecture is stable.

## 8. Concrete Milestones

### Milestone A

- instrumentation and benchmark baseline exist
- clear worst-case scene is reproducible

### Milestone B

- field mode no longer materializes path-export payloads
- explicit budgets cap worst-case failures

### Milestone C

- state lineage uses parent links
- legacy fixed history arrays remain only in compatibility mode

### Milestone D

- cell-side path storage becomes sparse-reference-only
- no full per-cell path/state duplication remains

### Milestone E

- hot-state packed stride is reduced
- replay/audit logic runs from lineage reconstruction

### Milestone F

- legacy history-slot implementation removed
- benchmark and acceptance suites pass on the new path

## 9. Expected Impact

### Memory

The largest expected wins should come from:

1. removing full-history copying from each child state
2. eliminating full path/state duplication at cell granularity
3. avoiding unnecessary path-monitor payload construction on field-only runs

### Runtime

Runtime may initially improve or regress depending on the phase:

- memory pressure should drop early
- path replay may become slower before it is optimized
- field mode should improve once less data is carried through gather/concat paths

### Engineering Complexity

The biggest complexity increase comes from:

- lineage replay
- sparse reference structures
- dual hot/cold data paths

That complexity is justified because the current flat replicated storage does not scale.

## 10. Testing Requirements

Every structural phase must add or update:

- unit tests for lineage correctness
- regression tests for field totals
- path-monitor parity tests
- memory benchmark snapshots
- gradient sanity tests where AD is expected

Specific test themes:

- child state lineage reconstruction equals previous flat history on small scenes
- path collector replay equals legacy path reconstruction
- field totals are unchanged when no pruning semantics change
- sparse cell-reference output produces the same final padded path result

## 11. Open Questions

These questions should be resolved before or during implementation:

1. Which currently packed fields are truly required in the hot path for all accumulation modes?
2. Do any AD-sensitive kernels require lineage information directly, or can lineage remain entirely
   cold?
3. Should path replay use state indices directly, or should it use a separate compact lineage arena?
4. For path-monitor export, is per-cell sparse reference storage enough, or is a per-receiver sparse
   layout more natural for the public output format?
5. Which replay outputs should be cached versus recomputed?

## 12. Recommendation

If only one deep refactor can be funded first, it should be:

- Phase 3: parent-link lineage

If two can be funded together, do:

- Phase 3: parent-link lineage
- Phase 4: sparse cell-to-state references

That combination attacks both structural multipliers:

- state-to-state history duplication
- state-to-cell duplication

Without those two changes, later compression and kernel tuning will only reduce the slope of the
problem, not remove the cause.
