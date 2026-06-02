# Deterministic Radiomap Package Architecture Plan

Status: Draft
Category: Plan
Last reviewed: 2026-04-14

## Objective

Refactor `witwin/channel/deterministic/` so it reaches the same architecture
bar now established by `witwin.channel.montecarlo`:

1. a small and explicit public API,
2. package-owned runtime objects instead of monitor-era compatibility shells,
3. domain ownership that follows deterministic radiomap concepts rather than
   historical `RadioMapMonitor` layering,
4. typed cross-module state instead of repeated dict-shaped payloads,
5. deletion of low-value wrappers and duplicate runtime paths in the same
   migration, not as an indefinite follow-up,
6. preserved forward results, backward gradients, and GPU-first execution.

This is a package-architecture and maintainability plan. It is not a physics
redesign.

## Relationship To Existing Plans

This document owns the standalone package architecture plan for
`witwin.channel.deterministic`.

`docs/dev/plans/radio-map-deterministic-decoupling-plan.md` remains the focused
runtime and kernel-decoupling plan for the deterministic radiomap workload and
its launch-count, repeatability, and native-accumulation gates.

The separation is intentional:

1. this document answers "how should the standalone package be shaped and what
   should be deleted or migrated?",
2. the decoupling plan answers "which deterministic radiomap execution
   contracts and native paths are being optimized or corrected?"

If the two documents disagree on ownership, this document owns package
structure and public API, while the decoupling plan owns workload-specific
performance gates.

## Non-Goals

This plan does not:

1. add CPU fallback paths,
2. add new public feature surface,
3. change diffraction or reflection physics formulas,
4. introduce Torch, NumPy, or DLPack into solver hot paths,
5. preserve old public names or compatibility wrappers just because they
   already exist,
6. rewrite every native kernel before the Python ownership cleanup is done,
7. keep two production runtimes alive long-term for the same deterministic
   solve path.

## Current Audit Summary

As of 2026-04-14, the standalone deterministic package is already substantial:

1. `183` tracked files,
2. `114` Python files,
3. `69` native source/header files,
4. about `61525` total lines.

The package is functional, but its structure still reflects its migration
history more than its actual ownership model.

### Current strengths

1. A standalone package already exists under `witwin/channel/deterministic/`.
2. Reflection, diffraction, native kernels, and result conversion are already
   separated into recognizable domains.
3. The package already has its own native extension loader, examples, and
   integration coverage.
4. Result-boundary tensor conversion is already mostly confined to
   `result.py` and `utils/tensor_conversion.py`.

### Current architecture problems

1. The package root still exports monitor-era public names:
   - `AxisAlignedGridSpec`
   - `RadioMapConfig`
   - `RadioMapResult`
   - `RadioMapSolver`
2. Package-root solve still routes through a compatibility shell:
   - root `__init__.py`
   - `trace.py`
   - `monitor.py`
   - `deterministic/trace.py`
3. The package still mirrors the old `channel/monitors/radio_map/` structure
   too closely instead of owning a package-native layout.
4. Several major modules are far beyond a readable ownership size:
   - `deterministic/cell_accumulation.py`: `2918` lines
   - `reflection/api.py`: `1842` lines
   - `kernels/monitors/field/radio_map_accumulate/native_impl.py`: `1673` lines
   - `deterministic/samples.py`: `1639` lines
   - `kernels/monitors/common/suffix_grid/native_impl.py`: `1366` lines
   - `reflection/epc.py`: `1319` lines
   - `diffraction/field.py`: `1172` lines
   - `kernels/trace/utd/native_impl.py`: `1053` lines
   - `deterministic/trace.py`: `725` lines
   - `config.py`: `769` lines
5. `common.py` is still a mixed-responsibility sink rather than one concept.
6. `monitor.py` exists mostly to reconstruct a package-local
   `RadioMapMonitor`, which is a low-value compatibility layer inside a
   standalone package.
7. `trace.py` is a wrapper layer whose main role is adapting package config
   back into the monitor-facing tracer path.
8. Internal state still crosses module boundaries as large anonymous dicts:
   - diagnostics payloads,
   - runtime backend metadata,
   - timing payloads,
   - accumulation payloads,
   - solver-control bundles.
9. Kernel ownership is still partly field-centric:
   - `kernels/monitors/field/...`
   - `kernels/monitors/common/...`
   even when the runtime contract is deterministic radiomap-specific.
10. The package carries naming duplication:
   - package path already says `deterministic_radiomap`,
   - public classes repeat `RadioMap` or `AxisAligned`,
   - subpackage `deterministic/` repeats the package name again.

## Lessons From The Monte Carlo Refactor

The Monte Carlo package established several rules this deterministic plan
should copy directly.

### What to repeat

1. Narrow the public surface first.
2. Introduce explicit owner modules before touching low-level performance
   details.
3. Replace repeated dict payloads with a small set of typed runtime objects.
4. Prefer static owner classes over large free-function modules when a module
   owns one runtime concept.
5. Split by domain ownership:
   - config,
   - grid,
   - solver,
   - integrator,
   - path families,
   - kernels,
   - result boundary.
6. Keep Torch conversion only at the result boundary.
7. Update examples, tests, and docs in the same migration.

### What to avoid

1. Do not leave the old runtime fully alive beside the new one for long.
2. Do not create one more wrapper layer to hide the old ownership.
3. Do not move names without deleting the abandoned aliases.

## Target Public Contract

The target standalone package contract should match the Monte Carlo package
shape unless a repo-wide standard later mandates a different single naming
scheme.

```python
from witwin.channel.deterministic import Config, GridSpec, Result, Solver
```

### Public naming policy

1. `AxisAlignedGridSpec` becomes `GridSpec`.
2. `RadioMapConfig` becomes `Config`.
3. `RadioMapResult` becomes `Result`.
4. `RadioMapSolver` becomes `Solver`.
5. Package root remains re-export only.
6. Keep one public solve entry only. Do not keep both `Solver` and `Tracer` as
   equal public fronts.

### Public behavior policy

1. Preserve the deterministic radiomap numerical contract.
2. Preserve the current explicit scene type requirement.
3. Preserve GPU-first behavior.
4. Preserve current result-boundary conversion helpers and sampling helpers.
5. Preserve current deterministic reflection and diffraction capability set
   unless a separate correctness change says otherwise.

## Target Package Shape

The package should read top-down from public API to runtime families.

```text
witwin/channel/deterministic/
  __init__.py
  _native/
  solver.py
  config.py
  grid.py
  result.py
  metadata.py
  diagnostics.py
  materials.py
  validation.py
  types.py
  integrators/
    __init__.py
    coherent.py
    incoherent.py
    scheduler.py
    samples.py
  reflection/
    __init__.py
    api.py
    epc.py
    paths.py
  diffraction/
    __init__.py
    api.py
    builders/
    geometry/
    state/
    operator.py
    utd.py
  path/
    __init__.py
    collectors.py
  kernels/
    ...
  utils/
    ...
```

This is a target ownership map, not a promise to create many tiny files.
Modules should only be split when the new owner is real.

## Static Class Policy

The deterministic package should not keep large owner modules as bags of free
functions.

When a module owns one stable runtime concept, prefer one explicit static class
namespace over dozens of sibling free functions.

Use static classes for domains such as:

1. config normalization and validation,
2. grid operations and cell indexing,
3. material lookup and material math,
4. deterministic diagnostics allocation and finalization,
5. metadata assembly,
6. scheduler and sample-planning policies,
7. reflection and diffraction owner operations where the methods share one
   runtime contract.

Do not use a static class when:

1. the code is a small immutable dataclass payload,
2. the module is a pure math module with a few tightly related formulas,
3. the class would only wrap one function and add no ownership value.

The intended rule is:

1. delete low-value free functions that merely live at module scope because of
   migration history,
2. group package-owned behavior under a named static owner class,
3. keep state in dataclasses and behavior in static owner classes,
4. avoid mixing large free-function sinks with a second wrapper class in the
   same module.

## Ownership Rules

### 1. Package root

`__init__.py` re-exports only:

1. `Config`
2. `GridSpec`
3. `Result`
4. `Solver`
5. `native_extension_available`

No coercion helpers, scene validation helpers, or solve orchestration should
remain in the package root.

### 2. Solver edge

`solver.py` owns:

1. public solve signature,
2. scene type checks,
3. conversion from public `Config` to resolved runtime config,
4. integrator selection and dispatch,
5. final call into the deterministic runtime.

It must not rebuild a `RadioMapMonitor` compatibility object.

### 3. Config

`config.py` owns:

1. public config dataclass,
2. resolved trace config,
3. solver-control resolution,
4. literal validation and normalization,
5. package-owned runtime option naming.

It must not also own result conversion or runtime execution.

The behavior in this module should be grouped under explicit static owners such
as `Config` and config-resolution helpers that are methods on that owner, not a
flat free-function surface.

### 4. Grid

`grid.py` owns:

1. `GridSpec`,
2. resolved receiver-grid geometry,
3. coordinate construction,
4. cell indexing helpers,
5. deterministic radiomap accumulation-facing geometry helpers.

It should not depend on `monitor.py`.

The operational surface of this module should be a small set of static owners
such as `GridSpec`, `Grid`, and `GridOps`, not a long list of unrelated module
functions.

### 5. Integrators

The current `deterministic/` subpackage should be replaced by an
integrator-owned structure.

The package is already deterministic, so `deterministic/` is redundant.
The owner split should instead be by solve family:

1. coherent transport family,
2. incoherent transport family,
3. shared deterministic scheduling and sample planning.

### 6. Reflection

`reflection/` owns:

1. reflection path generation,
2. EPC and reflection replay contracts,
3. reflection-family result payloads,
4. reflection-specific native adapter calls.

It must not own result assembly or public package coercion.

Where reflection behavior is spread across large helper modules, consolidate it
under explicit static owners instead of preserving free-function dumps.

### 7. Diffraction

`diffraction/` owns:

1. diffraction builders,
2. diffraction state layout,
3. geometry and visibility helpers specific to deterministic diffraction,
4. diffraction operator evaluation,
5. package-owned diffraction metadata and auditing support.

It must not be split by historical migration accidents such as separate helper
layers that only rename one function call.

Diffraction submodules should prefer explicit static owners for runtime
behavior, builder orchestration, and operator evaluation rather than broad
free-function modules.

### 8. Result boundary

`result.py` owns:

1. result dataclasses,
2. tensor conversion,
3. sampling helpers,
4. NumPy/Torch views,
5. immutable metadata projection.

Torch and NumPy stay here, not in hot runtime paths.

### 9. Kernels

Kernel layout should follow deterministic radiomap ownership rather than
generic field-monitor naming where possible.

That means:

1. keep truly shared receiver-tile or suffix-grid kernels under a clear shared
   owner,
2. move deterministic radiomap-specific accumulation or replay kernels under a
   deterministic radiomap owner,
3. stop naming deterministic-radiomap-specific kernels as generic field kernels
   when no other runtime truly owns them.

## Delete, Migrate, And Keep Decisions

### Delete

These should be deleted once the replacement owner exists:

1. package-root coercion helpers in `__init__.py`,
2. `trace.py` as a public compatibility wrapper,
3. package-local `monitor.py` if no external caller truly needs a monitor
   object inside the standalone package,
4. `payload.py` if it is only an intermediate compatibility container,
5. `orchestration.py` if its content is absorbed into `config.py` or
   `solver.py`,
6. `common.py` after its helpers are reassigned to real owner modules,
7. large owner-style free functions once they are migrated under static owner
   classes,
8. the redundant `deterministic/` package name once its modules are moved under
   `integrators/`.

### Migrate

These should be migrated, not preserved in place:

1. `deterministic/trace.py` into integrator-owned orchestration,
2. `deterministic/samples.py` into coherent/incoherent sample planning plus
   scheduler-owned helpers,
3. `deterministic/cell_accumulation.py` into smaller accumulation owners split
   by receiver contract and result mode,
4. `grid_reflection.py` into `reflection/` or `grid.py` depending on whether
   its owner is reflection or grid geometry,
5. kernel wrappers under `kernels/monitors/field/` into deterministic
   radiomap-owned kernel modules where they are not genuinely shared.

### Keep

These domains should remain first-class:

1. `reflection/`
2. `diffraction/`
3. `kernels/`
4. `utils/`
5. `_native/`
6. `result.py`

The goal is not fewer directories. The goal is clearer owners and fewer
duplicate execution layers.

## Typed Runtime Objects To Introduce

Cross-module deterministic runtime state should stop being open-ended dicts.
Use a small, stable set of typed payloads.

At minimum:

1. `TraceTiming`
2. `RuntimeBackends`
3. `PathCounts`
4. `ReflectionPhaseResult`
5. `DiffractionPhaseResult`
6. `AccumulationBuffers`
7. `MetadataInput`
8. `ReceiverGrid`
9. `ResolvedSurface`

Rules:

1. use dataclasses only for stable repeated payloads,
2. do not create wrapper dataclasses for three local variables used in one
   function,
3. do not pass giant free-form dicts between reflection, diffraction, and
   result assembly once the owner types exist.

## Phased Execution Plan

### Phase 0: Baseline Freeze And Audit

Outputs:

1. capture current package import contract,
2. capture deterministic forward and backward baseline outputs on maintained
   scenes,
3. record current package file map and top oversized files,
4. identify any external imports of package-local `RadioMap*` names.

Acceptance:

1. no code movement yet,
2. baseline tests are green,
3. benchmark and plot entrypoints are recorded before refactor begins.

### Phase 1: Public API Contraction

Changes:

1. add `Config`, `GridSpec`, `Result`, and `Solver`,
2. move public solve entry into `solver.py`,
3. reduce package root to re-exports,
4. update examples and tests to the short names,
5. remove `RadioMapSolver` and related aliases in the same phase unless an
   explicit temporary migration note is approved.

Acceptance:

1. public examples import only the short names,
2. integration package tests pass,
3. package root contains no solve logic.

### Phase 2: Remove Monitor-Era Compatibility Shells

Changes:

1. stop rebuilding `RadioMapMonitor` inside the standalone package,
2. delete or internalize `monitor.py`,
3. delete `trace.py` compatibility routing,
4. make `GridSpec` and package-owned config sufficient for runtime dispatch.

Acceptance:

1. `Solver.solve(...)` no longer routes through a package-local monitor object,
2. no package-local `Tracer` shim remains on the main solve path,
3. forward results are unchanged within existing tolerances.

### Phase 3: Runtime Typing Pass

Changes:

1. add typed runtime payloads to `types.py`,
2. replace repeated dict-shaped payloads across integrator, reflection,
   diffraction, diagnostics, and result assembly,
3. reduce cross-module anonymous metadata bundles,
4. replace owner-style free functions with static class namespaces in the
   modules that survive the phase.

Acceptance:

1. reflection, diffraction, and result boundaries pass typed payloads,
2. major solve functions become shorter and easier to review,
3. no new low-value wrapper layer is added,
4. surviving owner modules expose behavior primarily through static classes
   rather than large free-function surfaces.

### Phase 4: Integrator Ownership Rewrite

Changes:

1. replace `deterministic/` with `integrators/`,
2. split current orchestration by coherent and incoherent solve family where
   ownership differs,
3. move scheduler and sample planning under integrator-owned modules,
4. shrink `deterministic/trace.py`, `deterministic/samples.py`, and
   `deterministic/cell_accumulation.py` by moving real sub-owners out,
5. make the surviving integrator-facing owners static classes instead of new
   free-function modules.

Acceptance:

1. no redundant `deterministic/` package remains,
2. no orchestration module exceeds the approved exception budget without a
   documented reason,
3. coherent and incoherent flow are understandable from integrator entrypoint
   to phase outputs.
4. integrator-owned behavior is exposed through static owners rather than
   another free-function sink.

### Phase 5: Reflection Cleanup

Changes:

1. split `reflection/api.py` by path generation, replay, and accumulation
   owners if needed,
2. keep `epc.py` only if it remains a cohesive reflection owner,
3. move any grid-specific or result-specific helpers out of reflection.

Acceptance:

1. reflection modules each own one primary concept,
2. no reflection module acts as a catch-all dump for unrelated helpers,
3. reflection forward and backward parity tests remain green.

### Phase 6: Diffraction Cleanup

Changes:

1. simplify `diffraction/api.py`, `diffraction/field.py`, and
   `diffraction/state/*` ownership,
2. keep math-heavy UTD or geometry modules cohesive where they are true single
   concepts,
3. move auditing, profiling, and state-export helpers under explicit owners,
4. avoid dynamic re-export magic in `diffraction/__init__.py`.

Acceptance:

1. diffraction package is searchable by owner,
2. state layout, builder flow, and operator evaluation each have explicit
   homes,
3. no mixed "runtime plus helper dump" module remains.

### Phase 7: Kernel Ownership Cleanup

Changes:

1. rename or move deterministic-radiomap-specific kernels away from
   field-centric names where appropriate,
2. keep only genuinely shared kernels under shared owners,
3. align Python wrappers with the final ownership map,
4. remove duplicate wrapper layers around native kernel entrypoints.

Acceptance:

1. kernel names match runtime owners,
2. Python wrapper modules are thin but not meaningless,
3. deterministic radiomap no longer presents its private kernels as generic
   field-monitor infrastructure.

### Phase 8: Final Deletion Pass

Changes:

1. delete replaced wrappers and aliases,
2. delete abandoned compatibility modules,
3. delete dead imports and dead tests that only existed for old names,
4. update docs to the final package contract.

Acceptance:

1. there is one production deterministic runtime path,
2. old public names are gone unless explicitly grandfathered in a written note,
3. no duplicate implementation remains in parallel files.

### Phase 9: Documentation, Examples, And Tests

Changes:

1. add a package overview standard for `witwin.channel.deterministic`,
2. provide one onboarding-first minimal example,
3. keep one advanced example for profiling and gradients,
4. update integration tests to the final package contract.

Acceptance:

1. a new reader can follow `Solver.solve(...)` to the main runtime owners,
2. docs/dev index points at the active deterministic package plan,
3. examples use only the supported public contract.

## Test And Acceptance Matrix

Every phase must preserve the following.

### API gates

1. package imports succeed in the `witwin2` environment,
2. public examples use only supported names,
3. no stale alias-only tests remain after the alias is deleted.

### Numerical gates

1. deterministic radiomap forward parity holds on maintained reference scenes,
2. coherent and incoherent outputs remain within current tolerances,
3. diffraction enable/disable behavior remains unchanged.

### AD gates

1. TX position gradients remain within maintained tolerances,
2. geometry gradients remain within maintained tolerances,
3. supported material gradients remain within maintained tolerances.

### Performance and runtime gates

1. no intentional regression in maintained deterministic benchmark commands,
2. no extra CPU fallback path is introduced,
3. no new Torch or NumPy dependency enters hot runtime paths.

### Maintained entrypoints

Use the existing maintained deterministic radiomap commands and tests already
tracked in the repository, including package integration coverage and the
committed forward/backward profiling and plotting entrypoints used by the
deterministic radiomap workstream.

## Rollout Strategy

Use small, behavior-preserving passes.

Rules:

1. separate pure rename/move commits from behavior changes,
2. do not combine kernel rewrites with public API renames unless necessary,
3. land owner-introducing changes before delete passes,
4. delete old code in the same series once the replacement is proven,
5. do not leave "temporary" compatibility shims without a dated removal plan.

Recommended execution order:

1. Phase 0 and Phase 1
2. Phase 2 and Phase 3
3. Phase 4
4. Phase 5 and Phase 6
5. Phase 7
6. Phase 8 and Phase 9

## Risks

### Risk 1: Hidden external imports of old public names

Mitigation:

1. audit repository imports first,
2. update examples and tests in the same change,
3. only preserve an alias if a documented repo-wide migration requires it.

### Risk 2: Architecture cleanup stalls and old wrappers survive

Mitigation:

1. treat deletion as part of each phase,
2. do not mark a phase complete while both old and new solve paths remain
   active,
3. require a final duplicate-implementation audit before closing the plan.

### Risk 3: Oversplitting into low-value modules

Mitigation:

1. create new modules only for real owners,
2. keep math-heavy cohesive modules intact where appropriate,
3. prefer fewer readable owners over many tiny wrappers.

### Risk 4: Performance regressions caused by ownership churn

Mitigation:

1. run maintained forward and backward checkpoints at every phase boundary,
2. keep kernel behavior unchanged during structure-only passes,
3. isolate performance fixes into follow-up commits if needed.

## Completion Criteria

This plan is complete only when all of the following are true:

1. `witwin.channel.deterministic` exposes the short public contract:
   `Config`, `GridSpec`, `Result`, `Solver`,
2. package root is re-export only,
3. the main solve path does not rebuild a package-local `RadioMapMonitor`,
4. the redundant `trace.py` compatibility shim is gone,
5. the redundant `deterministic/` package layer is gone,
6. typed runtime payloads replaced the major repeated dict bundles,
7. oversized catch-all modules were split into real owners,
8. deterministic-radiomap-specific kernels have deterministic-radiomap-owned
   wrappers and naming,
9. examples, integration tests, and docs all use the final package contract,
10. no duplicate legacy runtime remains in parallel production files.
