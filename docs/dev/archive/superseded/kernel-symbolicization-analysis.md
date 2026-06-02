# Kernel Symbolicization Analysis

Status: Active
Category: Optimization
Last reviewed: 2026-04-08

## Scope

This note captures the current symbolic/JIT analysis for the radio-map Monte Carlo optimization work. It is intentionally scoped to the execution paths that matter for `RadioMapMonitor` throughput.

The immediate objective is not "make every RayD path symbolic". The immediate objective is narrower:

- keep no-diffraction radio-map LoS and reflection accumulation inside the Dr.Jit graph
- remove unnecessary `eval` / `make_opaque` barriers from the hot path
- reduce Python-side batch-loop overhead

RayD-wide symbolic support remains a separate project and is explicitly deferred in the optimization plan below.

## Current Findings

## 1. The old native Monte Carlo scatter callback is not graph-correct

The previous `native_monte_carlo` scatter path mutated the backing CUDA memory of a Dr.Jit array through a native callback and then attempted to expose the result with `dr.make_opaque(...)`.

That model is not reliable for Dr.Jit graph semantics:

- raw device memory can change
- the logical Dr.Jit variable value does not automatically update from that out-of-band mutation
- `make_opaque(...)` does not turn an arbitrary external write into a graph-visible value refresh

For radio-map Monte Carlo accumulation, this means the correct direction is:

- prefer graph-native `dr.scatter_reduce(...)` for accumulation
- use a real Dr.Jit custom op only if we later need a fused primitive with explicit graph ownership

## 2. The remaining no-diffraction performance gap is still execution-shape driven

The previous large gap to Sionna was not caused by Dr.Jit itself. It was mainly caused by:

- fixed conservative batch sizes
- repeated Python outer loops
- repeated `scene.ray_intersect(...)` launch groups
- repeated accumulation barriers

After switching to adaptive batch sizing, the remaining work is:

- allow the no-diffraction path to run full-batch when memory permits
- keep accumulation in the Dr.Jit graph
- move the LoS / reflection loop closer to a single Dr.Jit-controlled loop body

## 3. RayD is not globally symbolic-friendly today

RayD currently contains explicit symbolic blockers.

### Module-level blocker

`rayd` disables symbolic loops during module initialization:

- `E:/Code/RayDi/src/rayd.cpp`

### Slang interop blocker

The Slang interop layer is eager/scalar oriented and repeatedly forces evaluation:

- `E:/Code/RayDi/src/slang_interop.cpp`

Examples include:

- `drjit::eval(...)`
- `drjit::sync_thread()`
- lane extraction into scalar result structs

This layer should not be treated as the first target for symbolicization.

### Scene query blocker

The upper `Scene` query layer is closer to symbolic support, but still inserts eager barriers:

- `E:/Code/RayDi/src/scene/scene.cpp`

Known blockers:

- split-scene `intersect()` performs `eval + sync` after the static query before tracing the dynamic scene
- split-scene `shadow_test()` does the same
- `trace_reflections()` evaluates broadphase inputs and then launches a pointer-driven reflection pipeline

### Lower OptiX query layer

The lower OptiX query path is the most promising place for future symbolic support:

- `E:/Code/RayDi/src/scene/scene_optix.cpp`

`OptixScene::intersect()` and `OptixScene::shadow_test()` already route through `jit_optix_ray_trace(...)` and produce Dr.Jit arrays. They are much closer to a symbolic-compatible execution shape than the higher-level reflection pipeline.

## Practical Consequence

For the current radio-map work, we should not block on RayD-wide symbolicization.

The correct near-term strategy is:

1. Skip RayD-wide symbolic refactors for now.
2. Keep using the current RayD query path in evaluated-mode loops where necessary.
3. Remove graph breaks in the radio-map accumulation path first.
4. Revisit RayD symbolic support later, starting with `intersect()` and `shadow_test()`, not `trace_reflections()`.

## Channel-Side Symbolic Blockers

Even if RayD accepted symbolic scene queries, the radio-map path in `channel` still has its own graph blockers.

### Vector-power custom op

`radiomap_vector_power(...)` currently calls a native custom op path whose forward launcher explicitly performs `dr.eval(...)` before dispatch:

- `E:/Code/witwin-platform/channel/witwin/channel/kernels/monitors/field/radio_map_accumulate/native_impl.py`

This makes it incompatible with fully symbolic loop execution.

### Previous Monte Carlo accumulation callback

The old Monte Carlo scatter callback path also introduced explicit graph boundaries. This is why the current refactor moves accumulation toward direct `dr.scatter_reduce(...)`.

## Difficulty Assessment

## Minimal useful goal: symbolic-friendly scene queries

If the only requirement is to support radio-map LoS / reflection loop control better, then a limited RayD project focused on:

- `Scene.intersect()`
- `Scene.shadow_test()`

is difficult but manageable.

This is a medium-sized runtime refactor, not a research project.

## Full RayD symbolicization

If the requirement is:

- symbolic `intersect`
- symbolic `shadow_test`
- symbolic `trace_reflections`
- symbolic nearest-edge queries
- symbolic Slang interop

then the project becomes substantially larger.

`trace_reflections()` is the main escalation point because it currently depends on a raw-pointer launch contract instead of a Dr.Jit-owned graph primitive.

## Recommended Optimization Order

The current repository should proceed in this order:

1. Keep the radio-map no-diffraction path on graph-native accumulation.
2. Remove unnecessary Python batch loops and other loop-control overhead in `channel`.
3. Keep diffraction work separate until the no-diffraction path is stable.
4. Defer RayD symbolic work.
5. If RayD symbolic work becomes necessary, start with `intersect()` and `shadow_test()` only.

## Explicit Deferral

The following items are deferred for the current radio-map optimization cycle:

- RayD-wide symbolic loop enablement
- symbolic Slang interop support
- symbolic `trace_reflections()` support inside RayD
- symbolic nearest-edge query support inside RayD

Those are valid future topics, but they are not the shortest path to better radio-map Monte Carlo throughput today.
