# Monte Carlo Radiomap Package Architecture Plan

Status: Draft
Category: Plan
Last reviewed: 2026-04-13

## Objective

Refactor `witwin/channel/montecarlo/` so it meets the maintainability bar of
a large open source project:

1. the public API is small, explicit, and stable,
2. package structure reflects domain ownership instead of implementation
   history,
3. oversized mixed-responsibility modules are split into cohesive units,
4. repeated dict-shaped runtime payloads are replaced by a small number of
   typed state objects,
5. scene adaptation, Monte Carlo tracing, AD support, and native-kernel
   boundaries are readable without reverse-engineering the whole package,
6. a new reader can understand the solver from package entrypoint to major
   phases without first learning every optimization detail.

This is an architecture and maintainability plan. It is not a solver redesign.
The estimator contract stays:

1. transmitter-driven Monte Carlo accumulation,
2. axis-aligned matched-isotropic incoherent radio-map output,
3. optional first-order diffraction,
4. native CUDA acceleration where supported,
5. Dr.Jit-first internals.

Update note, 2026-04-13:
the active implementation no longer uses a package-level `trace.py` /
`Tracer` layer. The current architecture replaces that boundary with
`integrators/unidirectional.py` plus a placeholder
`integrators/bidirectional.py`. Historical references below to the standalone
tracer should be read as the transport-integrator orchestration owner until
this draft plan is fully rewritten.

## Non-Goals

This plan does not:

1. add CPU fallback paths,
2. expand the public feature set,
3. change numerical behavior as part of the structure-only pass,
4. rewrite the native kernels first,
5. merge the standalone package back into `witwin.channel`,
6. preserve low-value compatibility aliases just because they already exist in
   old notes or docs.

## Current Audit Summary

The package already has a usable high-level story, but its internal shape is
not yet at "large maintainable open source project" quality.

### Current strengths

1. The public solve contract is narrow:
   - `RadioMapSolver.solve(...)`
   - `AxisAlignedGridSpec`
   - `RadioMapConfig`
   - `RadioMapResult`
2. The solver is conceptually decomposed into:
   - scene adaptation,
   - grid construction,
   - specular tracing,
   - diffraction tracing,
   - result conversion,
   - AD replay support.
3. There is a maintained example and integration coverage.

### Current architecture problems

1. Several modules are too large to be read as single concepts:
   - `trace.py`: `882` lines
   - `scene/adapter.py`: `879` lines
   - `config.py`: `756` lines
   - `diffraction/state.py`: `757` lines
   - `reflection.py`: `593` lines
- `integrators/unidirectional_ad.py`: `566` lines
2. `config.py` mixes three different layers:
   - user-facing config,
   - resolved trace parameters,
   - solver-control policy.
3. `trace.py` is both orchestration module and implementation dump:
   - backend checks,
   - batching,
   - reflection phase,
   - diffraction phase,
   - metadata assembly,
   - AD-mode branching,
   - final result construction.
4. `scene/adapter.py` owns too many responsibilities:
   - scene wrapper,
   - geometry merge,
   - edge extraction,
   - triangle material views,
   - RayD query forwarding,
   - vertex updates,
   - runtime cache invalidation.
5. `common.py` is a sink module:
   - ray sampling,
   - grid intersection,
   - scatter helpers,
   - batch planning,
   - constants.
6. AD logic is structurally separate but still too concentrated:
   - custom op definition,
   - sparse-coefficient cache setup,
   - detached workload handling,
   - transport replay,
   - component wiring.
7. `diffraction/__init__.py` uses dynamic re-export search through
   `__getattr__`, which hides ownership and weakens discoverability.
8. Public naming is inconsistent across docs and code:
   - docs mention `MonteCarloRadioMapSolver`
   - code exports `RadioMapSolver`
9. The package has no package-local architecture overview or minimal example;
   the current example is a research harness, not onboarding-first material.

## Architecture Standard For This Package

This package should follow the active Python package standard in
`docs/dev/standards/20-python-package-architecture-standard.md` with the
following package-specific rules.

### 1. Keep the public API thin

Package-root exports should stay limited to stable nouns:

1. `GridSpec`
2. `Config`
3. `Result`
4. `Solver`
5. `native_extension_available`

`__init__.py` should re-export only. It should not contain solve logic,
coercion policy, or hidden ownership.

### 2. Organize by domain, not by implementation accidents

The package should read top-down:

1. public API,
2. scene runtime,
3. solver runtime,
4. path families,
5. AD support,
6. native kernels,
7. pure utilities.

The current "one big trace file plus support fragments" shape should be
eliminated.

### 3. One module should own one primary concept

Examples:

1. config normalization belongs in a config or controls module,
2. batch planning belongs in a batching module,
3. metadata construction belongs in a metadata module,
4. scene runtime queries belong in scene runtime modules,
5. result-to-torch conversion belongs at the result boundary, not in a generic
   utility sink.

### 4. Stateful lifecycle should have explicit owners

The current repeated dict payloads are a sign that the solver needs a few
real owner objects:

1. one object for scene runtime state,
2. one object for one trace session,
3. one object for the specular phase,
4. one object for the diffraction phase,
5. one object for AD runtime context.

This does not justify adding generic "manager" or "builder" classes. The owner
must correspond to real repeated state.

### 5. Keep pure helpers small and truly generic

`utils/` should contain only pure, reusable helpers:

1. constants,
2. Dr.Jit array helpers,
3. geometry math,
4. polarization math,
5. small numeric conversions.

Scene-specific, result-specific, and solver-specific helpers must not remain in
`utils/`.

### 6. Remove dynamic export magic from core domains

`diffraction/__init__.py` should use explicit imports. Large open source code
bases need searchability and ownership clarity more than re-export cleverness.

### 7. Use typed runtime objects at cross-module boundaries

Loose dict payloads should not cross package-internal boundaries when they are
stable and repeated. Use small dataclasses for:

1. batch plans,
2. path counts,
3. timing payloads,
4. metadata inputs,
5. AD context,
6. scene precompute summaries,
7. phase outputs.

### 8. Adopt explicit size budgets

These are architectural targets, not hard failures:

1. public modules: usually `< 200` lines,
2. orchestration modules: usually `< 350` lines,
3. algorithm modules: usually `< 450` lines,
4. split any module before it grows past `500` lines unless it is a strongly
   cohesive math kernel module,
5. functions should usually stay below `60` logical lines,
6. if a function grows past `100` logical lines, it must justify why it still
   owns one concept.

### 9. Keep File Count Roughly Flat

This package should not "improve architecture" by doubling the number of
modules.

The target is to keep the production Python file count in roughly the current
range. Prefer:

1. replacing one oversized file with one or two cohesive files,
2. absorbing low-value helper modules back into their real owner,
3. using clear in-file sections before creating a new module,
4. keeping the existing `scene/` and `diffraction/` package boundaries,
5. limiting net new production modules to a very small number.

Do not introduce new package layers such as `runtime/`, `paths/`, or `ad/`
unless a later phase proves they are necessary after the low-value cleanup is
done.

## Target Public Contract

The public contract should be renamed to short context-aware names. The package
path already carries the domain context, so the public classes do not need
`RadioMap` or `AxisAligned` prefixes in their primary names.

```python
from witwin.channel.montecarlo import (
    GridSpec,
    Config,
    Result,
    Solver,
)
```

### Naming policy

1. Rename the public classes to:
   - `AxisAlignedGridSpec` -> `GridSpec`
   - `RadioMapConfig` -> `Config`
   - `RadioMapResult` -> `Result`
   - `RadioMapSolver` -> `Solver`
2. Stop using `MonteCarloRadioMapSolver`, `MonteCarloRadioMapConfig`, and
   `MonteCarloRadioMapResult` in active docs.
3. Stop using `AxisAlignedGridSpec`, `RadioMapConfig`, `RadioMapResult`, and
   `RadioMapSolver` as the target public API names.
4. Keep the Monte Carlo identity in the package path, not duplicated in every
   public class name.
5. Do not keep long-name compatibility aliases in the final package shape
   unless a separate migration decision explicitly requires them.

This matches the active architecture standard: avoid repeating context that is
already obvious from the package path.

## Target Package Layout

```text
witwin/channel/montecarlo/
  __init__.py
  solver.py
  config.py
  grid.py
  result.py
  profiler.py
  types.py
  trace.py
  common.py
  materials.py
  reflection.py
  reflection_ad.py
integrators/unidirectional_ad.py
  ad_support.py
  scene/
    __init__.py
    adapter.py
    types.py
  diffraction/
    __init__.py
    state.py
    tracing.py
    field.py
    visibility.py
  kernels/
    monte_carlo/
    sparse_coeff/
  utils/
    constants.py
    drjit_ops.py
    geometry.py
    plane_axes.py
    polarization.py
    power.py
```

### Ownership summary

1. `solver.py`
   - public solve entry only
2. `config.py`
   - public config plus internal resolved-control section
3. `grid.py`
   - public grid spec and internal grid runtime construction
4. `result.py`
   - public result object and result-boundary conversion helpers
5. `trace.py`
   - tracer orchestration only
6. `reflection.py`
   - LoS plus reflection phase
7. `diffraction/`
   - diffraction-specific runtime only
8. `scene/adapter.py`
   - lightweight scene adapter and scene-local queries
9. `integrators/unidirectional_ad.py`, `reflection_ad.py`, `ad_support.py`
   - AD-only logic, without creating a deeper package layer
10. `utils/`
   - pure helper math only

## File-Level Change Plan

### A. `__init__.py`

Current problem:

1. package root contains solve logic and coercion helpers.

Target:

1. move `RadioMapSolver` implementation into `solver.py`,
2. keep `__init__.py` as re-export only,
3. export the short public names:
   - `GridSpec`
   - `Config`
   - `Result`
   - `Solver`
4. allow at most trivial import-time version or extension checks.

### B. `solver.py` (new)

Target ownership:

1. public `Solver`,
2. public input coercion at the API edge,
3. one short `solve(...)` method that:
   - adapts the scene,
   - resolves runtime controls,
   - calls the tracer directly,
   - returns `Result`.

`solve(...)` should not contain phase logic.

### C. `config.py`

Current problem:

1. user config and runtime policy are mixed together.

Target:

1. keep `Config` as the public entry type,
2. keep `TraceConfig`, `ResolvedTraceConfig`, `resolve_trace_config(...)`, and
   `resolve_solver_controls(...)` in the same module if that avoids creating a
   low-value extra file,
3. split the file by explicit sections:
   - public config
   - internal resolved config
   - solver-control policy
4. move and rename `AxisAlignedGridSpec` to `GridSpec` in `grid.py`,
5. replace repeated string-policy checks with normalized typed internal values
   as early as possible.

### D. `grid.py`

Target:

1. own `GridSpec`,
2. own `RadioMapGrid`,
3. keep world-space cell-center construction,
4. keep surface descriptor logic,
5. do not accumulate unrelated quadrature or monitor-generic logic here.

`grid.py` is already relatively cohesive and should stay that way.

### E. `result.py`

Current problem:

1. result conversion depends on helper modules that are effectively
   result-specific.

Target:

1. keep `Result` and `RadioMapCoordinates`,
2. move torch/tensor conversion helpers next to the result boundary,
3. keep user-facing sampling helpers such as `sample_metric_positions(...)`,
4. do not let result code depend on generic utility sinks for package-specific
   conversions.

### F. `scene/adapter.py`

Current problem:

1. one file owns wrapper, precompute, query, and mutation lifecycle.

Target:

1. move solver-facing runtime queries into `witwin.channel.core.scene.Scene` instead
   of maintaining package-local scene modules,
2. absorb any remaining scene-specific query helpers into the owning solver
   files or shared `channel_scene` APIs,
3. avoid recreating scene-local wrapper or precompute modules once the shared
   scene contract is sufficient.

### G. `common.py`

Current problem:

1. it mixes several runtime concerns.

Target:

1. keep `common.py` if that helps hold file count flat,
2. narrow it to Monte Carlo runtime primitives only:
   - ray sampling
   - plane/grid hit helpers
   - scatter helpers
   - batch planning
3. move anything scene-specific, result-specific, or AD-specific out of it,
4. do not let it grow into a second architecture sink,
5. if `common.py` remains, keep it on a tight size budget:
   - target `<= 300` lines
   - hard review required before it exceeds `400` lines.

### H. `reflection.py`

Current problem:

1. the module name says "reflection", but the implementation owns LoS plus
   reflection batching and tape capture.

Target:

1. keep the file name `reflection.py` to avoid churn and extra package layers,
2. keep LoS and reflection in one phase module because they share ray batches,
3. own:
   - path tape storage,
   - active ray loop,
   - plane contribution evaluation,
   - reflection bounce updates,
   - optional wedge collection for diffraction handoff.

The important change is ownership, not renaming the file.

### I. `diffraction/`

Current strengths:

1. diffraction already has a meaningful subpackage split.

Required changes:

1. stop dynamic symbol export from `diffraction/__init__.py`,
2. keep state sampling inside `state.py` unless a later split is justified,
3. keep `tracing.py` for the runtime phase loop,
4. keep `field.py` for UTD field/power evaluation,
5. keep `visibility.py` for visibility tests only.

The diffraction package is close to a good shape. It needs ownership cleanup,
not a full redesign.

### J. `trace.py`

Current problem:

1. one file owns nearly the entire solver lifecycle.

Target:

Keep `trace.py` as the single tracer/orchestration module.

Required changes:

1. keep `Tracer` as the only orchestration owner between `Solver` and the
   phase modules,
2. split large local blocks into tracer methods or small helper dataclasses,
   not a second orchestration class,
3. make the main flow explicit:
   - `Tracer.trace(...)`
   - `Tracer.primal(...)`
   - reflection phase
   - diffraction phase
   - metadata/result assembly
4. replace repeated phase dict payloads with a very small number of typed
   payloads where they materially improve readability.

### K. `integrators/unidirectional_ad.py`, `reflection_ad.py`, `ad_support.py`

Target:

1. keep the existing file count roughly flat,
2. let `integrators/unidirectional_ad.py` own AD orchestration and the custom op,
3. let `reflection_ad.py` own reflection-specific AD math,
4. let `ad_support.py` own tape and reusable AD support structures,
5. split further only if one of these files remains too large after cleanup,
6. any move of tape ownership or sparse-coefficient helpers must preserve the
   Dr.Jit capture boundary used by the `dr.CustomOp` closure.

### L. `utils/`

Keep only modules that are genuinely package-generic:

1. `constants.py`
2. `drjit_ops.py`
3. `geometry.py`
4. `plane_axes.py`
5. `polarization.py`
6. `power.py`

Move out:

1. `mesh_buffers.py` -> `scene/geometry.py`
2. `tensor_conversion.py` -> `result.py` boundary when practical
3. `torch_bridge.py` -> `result.py` boundary when practical
4. any scene-specific material or transform helper that is not reused outside
   one domain.

Prefer folding narrow boundary helpers into their owning file before creating a
new support module.

## Class-Level Change Plan

### 1. Use `witwin.channel.core.scene.Scene` directly

Responsibilities:

1. keep merged geometry, edge/material runtime views, and RayD query helpers on
   `witwin.channel.core.scene.Scene`,
2. let Monte Carlo solver code consume that shared scene contract directly,
3. avoid package-local scene wrappers or duplicated scene-runtime lifecycles.

### 2. Keep `Tracer` as the sole orchestration class

Responsibilities:

1. own one solve flow from primal trace to result assembly,
2. coordinate reflection, diffraction, metadata, and AD branching,
3. stay thin enough that the call graph remains easy to follow.

Do not add `TraceSession` or another second-layer orchestration object between
`Solver` and the phase modules.

### 3. Keep reflection logic module-owned

Responsibilities:

1. trace LoS and reflection batches,
2. record optional tape data,
3. return a typed phase result or a tightly scoped dict payload.

Do not introduce a required `SpecularTracer` class unless the module still
remains too broad after cleanup.

### 4. Keep `DiffractionTracer`, but narrow its surface

Responsibilities:

1. batch diffraction samples,
2. evaluate visibility and field,
3. scatter contributions,
4. return a typed phase result.

It should not own unrelated scene setup or metadata assembly.

### 5. Introduce a small typed runtime payload set

Recommended dataclasses:

1. `BatchPlan`
2. `TraceTiming`
3. `PathCounts`
4. `ReflectionPhaseResult`
5. `DiffractionPhaseResult`
6. `ADContext`

Do not create more typed wrappers than needed. The goal is to replace repeated
cross-module dict payloads, not to wrap every local variable.

## Function-Level Rules

The refactor should enforce the following function rules throughout the package.

### 1. Every non-trivial function must declare a return type

This is mandatory for:

1. public functions,
2. cross-module helpers,
3. class methods,
4. phase functions.

### 2. Every non-trivial function needs a short purpose comment

The current package already uses this style in many places. Make it
consistent.

### 3. Stop passing "bags of stuff"

If several functions share the same repeated argument cluster, move ownership to
an object or typed payload.

### 4. Ban low-value one-line wrappers

Do not add wrapper functions that only rename another function call, especially
around:

1. array math,
2. small metadata reads,
3. scatter helpers,
4. trivial conversions.

### 5. Prefer module-qualified imports for owned domains

Examples:

1. `from . import common as mc_common`
2. `from . import reflection as mc_reflection`
3. `from . import diffraction as mc_diff`

This keeps ownership readable at the call site.

### 6. Keep the public solve path easy to trace

A new reader should be able to follow this call graph without jumping through
dynamic exports or sink modules:

1. `Solver.solve(...)`
2. `Tracer.trace(...)`
3. reflection phase
4. `DiffractionTracer.trace_batches()`
5. `Result.from_solver(...)`

### 7. Preserve Dr.Jit CustomOp capture boundaries

AD modules that define `dr.CustomOp` subclasses must document which Dr.Jit
variables are captured by the op's closure.

Moving captured state across module boundaries requires verifying that:

1. the same symbolic/JIT trace graph remains intact,
2. the closure still references live Dr.Jit values from the intended solve
   scope,
3. JVP/VJP behavior and kernel history do not regress because of capture
   breakage.

## Documentation And Example Changes

Structure changes alone are not enough for open source maintainability.

### Required documentation changes

1. Add a short package-architecture overview doc for the standalone package.
2. Update active docs to use the short public names:
   - `Solver`
   - `Config`
   - `Result`
3. Add one minimal example that only demonstrates:
   - scene creation,
   - grid creation,
   - config creation,
   - `Solver.solve(...)`,
   - reading `path_gain`.

### Example policy

Keep both example tiers:

1. `examples/monte_carlo_radiomap_minimal.py`
   - onboarding-first
2. `examples/monte_carlo_radiomap_three_cubes.py`
   - research and profiling workflow

The current three-cube example should not remain the only way to learn the
package.

## Test Strategy

The refactor should preserve or improve the current integration coverage while
adding more architecture-local tests.

### Required test layers

1. public API tests
   - package-root imports
   - solve contract
   - config validation
2. scene runtime tests
   - geometry refresh
   - material lookup
   - ray query passthrough
3. phase tests
   - specular-only path
   - diffraction-enabled path
4. AD tests
   - JVP/VJP smoke tests
   - detached workload path
5. parity tests
   - parity with current standalone behavior
   - parity with legacy monitor where applicable

### Required non-functional checks

1. no dynamic re-export search in core packages,
2. if `common.py` remains, it is a bounded Monte Carlo runtime helper module,
   not a cross-domain sink, and it stays within its size budget,
3. no new file above the size budget without justification,
4. no active doc references to stale `MonteCarloRadioMap*` names.

## Rollout Plan

### Phase 1: Public boundary cleanup

1. add `solver.py`,
2. reduce `__init__.py` to re-exports,
3. rename the package-root public classes to:
   - `GridSpec`
   - `Config`
   - `Result`
   - `Solver`
4. standardize docs and tests on the short public names,
5. add the minimal example.

### Phase 2: Config and grid boundary cleanup

1. separate public and internal config sections clearly inside `config.py`,
2. move `AxisAlignedGridSpec` to `grid.py`,
3. keep package-root imports stable.

### Phase 3: Scene runtime split

1. simplify `scene/adapter.py`,
2. absorb or delete `runtime_queries.py`,
3. preserve vertex-update behavior and AD-friendly scene data.

### Phase 4: Tracer cleanup

1. keep `Tracer` as the only orchestration owner,
2. move phase-local logic out of long methods,
3. replace the worst dict payloads with small typed phase results.

### Phase 5: Path-family cleanup

1. narrow reflection module ownership without renaming `reflection.py`,
2. keep LoS and reflection together,
3. narrow diffraction package ownership,
4. delete dynamic diffraction exports.

### Phase 6: AD cleanup

1. narrow `integrators/unidirectional_ad.py`, `reflection_ad.py`, and `ad_support.py`,
2. consolidate tape ownership only where it does not risk breaking the
   `dr.CustomOp` closure boundary,
3. keep sparse-coefficient and transport logic domain-local without creating a
   new package layer.

### Phase 7: Utility shrink and final cleanup

1. delete `common.py` only if it still behaves like a sink module after the
   cleanup passes,
2. otherwise keep `common.py` narrow and within its size budget,
3. move non-generic helpers out of `utils/`,
4. enforce module size budgets,
5. update docs and examples.

## Acceptance Criteria

This plan is complete when all of the following are true.

### Package shape

1. `__init__.py` is re-export only.
2. file count stays roughly in the current range and does not balloon through
   micro-modules.
3. `scene/adapter.py` is no longer a mixed-responsibility dump.
4. `trace.py` is a readable tracer/orchestration module, not a second
   monolithic sink.
5. AD code is clearer without requiring a deeper package tree.

### Readability

1. a new reader can follow the public solve call graph in five hops or fewer,
2. no core package relies on dynamic symbol lookup to expose its API,
3. no core module exceeds the size budget without a written justification.

### Contract stability

1. public package-root imports are stable after the short-name rename,
2. forward results match current behavior,
3. AD behavior remains covered by regression tests,
4. native extension boundary stays explicit and unchanged in semantics,
5. multi-TX forward throughput regression does not exceed `2%` compared to the
   pre-refactor baseline.

### Documentation

1. the development-doc index links this plan,
2. active docs use the real public names,
3. the package has one minimal example and one advanced example.

## Recommended First Patch

If the work is staged conservatively, the first patch should do only the
highest-signal boundary cleanup:

1. add `solver.py`,
2. shrink `__init__.py`,
3. rename the public package-root classes to `GridSpec`, `Config`, `Result`,
   and `Solver`,
4. clarify `config.py` sections without spawning extra packages,
5. standardize docs on the short public names,
6. keep the orchestration path as `Solver -> Tracer -> reflection/diffraction
   -> Result`.

That first patch improves reader comprehension immediately without touching
solver math or native kernels.
