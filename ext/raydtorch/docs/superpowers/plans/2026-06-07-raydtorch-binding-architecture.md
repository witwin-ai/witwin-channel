# RayDTorch Binding Architecture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: Draft
Last reviewed: 2026-06-07

**Goal:** Make RayDTorch's native operators callable from C++ and the Torch dispatcher (not only from Python pybind), give the Scene/cache a typed dispatcher-friendly lifetime, and keep Python-only glue on pybind11. The driver is letting downstream consumers (notably witwin `channel`, see its plan 16 "Torch Runtime Decoupling") call RayD primitives as Torch-native ops without a Python re-entry and without compiling RayDTorch sources into a second extension.

**Architecture:** One shared library, three binding tiers chosen by call-site requirement:
1. `TORCH_LIBRARY` operators for functional tensor primitives (forward / backward / JVP). Callable from Python (`torch.ops.raydtorch.*`) and C++ (dispatcher), autograd-composable, schema-checked.
2. `torch::class_<SceneCache>` (TorchBind) for the Scene/cache lifecycle handle. It becomes a typed argument to the ops, is reference-counted, and removes the global `get_scene(int64)` registry from the dispatcher path.
3. `pybind11` for Python-only surfaces: `build_info`, capability probes, debug/introspection, one-off utilities, and any conversion that returns arbitrary Python objects.
   The OptiX context / pipeline / SBT stays an internal C++ owned resource and is not bound at all (only its status is reported through a probe).

**Tech Stack:** PyTorch CUDA C++ extension, `torch/library.h` (`TORCH_LIBRARY`, `torch::class_`), `torch/extension.h` + pybind11 (Python-only tier), ATen, CUDA Runtime/Driver, OptiX, CMake/scikit-build-core, unittest, finite-difference gradient tests.

**Repository:** This plan is for `E:\Code\RayDTorch`.

**Environment:** All commands use the conda environment `witwin2`.

**Namespace:** Register ops under the `raydtorch` dispatcher namespace (`torch.ops.raydtorch.*`) and the custom class under `torch.classes.raydtorch.Scene`. A shorter namespace (e.g. `raydt`) is an allowed call-site convenience; if chosen, keep it consistent across ops and class. The Python package name stays `raydtorch`; the pybind extension name stays `_raydtorch`.

---

## Context

The native compute layer is already cleanly factored: each op is a stateless flat function `(scene_handle: int64, at::Tensor...) -> tensors (+ tape)` that forwards to a `*_cuda` launcher. Example: `intersect_forward / intersect_backward / intersect_jvp` in `src/torch_ext/scene/ops_intersect.cpp`.

Current binding facts that this plan changes:

- Ops are exposed only through **pybind** `m.def(...)` inside `PYBIND11_MODULE(_raydtorch, m)` (`src/torch_ext/module.cpp` calls `bind_scene_ops`, `bind_intersect_ops`, `bind_edge_ops`, `bind_reflection_ops`, `bind_diffraction_ops`). They are Python-callable only; nothing is registered with the Torch dispatcher.
- Op functions return `py::tuple`, which couples the op layer to pybind.
- The Scene/cache is passed as an opaque `int64_t scene_handle` resolved through a process-global registry (`get_scene(scene_handle)` -> `SceneCache &`).
- Autograd orchestration lives in Python `raydtorch/autograd.py` as `torch.autograd.Function` subclasses (e.g. `_IntersectFunction`) that call `_C.<op>` and manage the saved tape via `save_for_backward` / `save_for_forward`.

This design is fine for the standalone Python package, but it blocks three things a Torch-native consumer needs:

1. **C++ / dispatcher callability.** A consumer that wants RayD primitives inside its own native op (or wants `torch.ops`-level composition, `torch.compile` opacity, FakeTensor/meta) cannot reach pybind functions.
2. **A typed, dispatcher-passable Scene.** `int64 + global registry` cannot be a typed op argument and forces manual lifetime plus a process-global singleton.
3. **Consumer-owned autograd.** Per the redner-style ownership in channel plan 16, the *consumer* should own the `torch.autograd.Function` and call RayD's forward/backward/jvp as plain primitives. RayDTorch should expose those three flat primitives, not force its own Python autograd wrapper into the consumer's hot path.

## Motivation

- witwin `channel` plan 16 wants RayD geometry/multipath primitives as Torch-native ops backing `torch.ops.wc.*`, with channel owning the autograd functions and the RF material/accumulation kernels. The cheapest correct boundary is: RayDTorch registers its primitives as dispatcher ops; channel calls them (from Python for coarse orchestration, or from C++ if it ever fuses) and wraps them in channel-owned `torch.autograd.Function`s.
- `TORCH_LIBRARY` registration is simultaneously a Python API and a C++ API from one registration, with a stable schema contract. This is the native C++ interface the consumer needs without hand-writing a parallel C++ header library or exporting internal types (`SceneCache`, `IntersectForwardOutputs`) as an ABI.
- Keeping RayDTorch a single shared library (consumer depends on the installed package, ops self-register on import) avoids the duplicate-statics hazard of compiling RayDTorch translation units into a second extension: two scene registries, two OptiX contexts, two CUDA module caches in one process.

## Decision Rule (call-site driven, not uniformity)

Pick the binding tier from what the call site requires, not from a desire to make everything one mechanism:

- Must be callable from C++ / participate in autograd / run in a hot path  -> **`TORCH_LIBRARY` op**.
- Must be a typed handle passed into ops, with managed lifetime, reachable from C++  -> **`torch::class_`**.
- Only Python ever touches it, may return arbitrary Python objects  -> **`pybind11`**.
- Heavy device-global resource the public never holds (OptiX context/pipeline/SBT)  -> **internal C++ owned, not bound**; expose status via a probe only.

## Non-Goals

- Do not compile RayDTorch `.cu`/`.cpp` sources directly into a consumer's extension. RayDTorch stays one shared library; consumers call through the dispatcher.
- Do not hand-write a separate `raydtorch_native_core` C++ header/library API or export internal structs as a stable ABI. The dispatcher schema is the C++ contract.
- Do not register an `Autograd`-key kernel for these primitives inside RayDTorch. Consumers own the `torch.autograd.Function`. RayDTorch's `raydtorch/autograd.py` remains a reference/standalone path, not the consumer hot path.
- Do not break the existing public `raydtorch` Python API used by tests and standalone users. Behavior and tensor ABI stay identical; only the underlying binding mechanism changes.
- Do not introduce a Dr.Jit dependency. The package stays Dr.Jit-free.
- Do not fold forward/backward/jvp into a single fused op. Keep them separate primitives so the consumer can schedule its own tape policy.

## Target Binding Layout

One `.so`/`.pyd`, three registrations coexisting (standard and supported):

```text
TORCH_LIBRARY(raydtorch, m):          // hot functional primitives, C++ + Python callable
  m.class_<SceneCache>("Scene")       //   lifecycle handle as a typed dispatcher value
     .def(torch::init<...>())         //     build(vertices, faces, ...) -> Scene
     .def("update_vertices", ...)     //     methods limited to IValue types (Tensor/int/...)
     .def("sync", ...);
  m.def("intersect_forward(__torch__.torch.classes.raydtorch.Scene scene, Tensor ray_o, "
        "Tensor ray_d, Tensor ray_tmax, Tensor active) -> Tensor[]");
  m.def("intersect_backward(...) -> Tensor[]");
  m.def("intersect_jvp(...) -> Tensor[]");
  // edge / reflection / diffraction forward+backward+jvp ...

TORCH_LIBRARY_IMPL(raydtorch, CUDA, m):
  m.impl("intersect_forward",  &intersect_forward_op);   // op funcs return std::tuple/std::vector<at::Tensor>
  m.impl("intersect_backward", &intersect_backward_op);  //   (no py::tuple in the op layer)
  m.impl("intersect_jvp",      &intersect_jvp_op);

PYBIND11_MODULE(_raydtorch, m):       // Python-only glue
  m.def("build_info", ...);           //   returns py::dict
  // capability probes, debug dumps, one-off utilities, arbitrary-object conversions

// OptiX context / pipeline / SBT: internal C++ owned singleton/resource, not bound.
```

Notes:

- `SceneCache` must inherit `torch::CustomClassHolder` and be constructed/held via `c10::intrusive_ptr<SceneCache>`.
- Multi-tensor returns use `Tensor[]` in the schema (or a fully enumerated named tuple if a stable named contract is wanted). The tape tensors stay part of the forward return so the consumer can save them.
- Custom-class method/argument types are limited to IValue-convertible types. The Scene interface (`build / update_vertices / sync / be passed to ops`) is IValue-clean, so this constraint costs nothing here. Anything with a "dirty" non-IValue C++ interface stays on pybind.
- Capsules are retired for the Scene path: the custom class is the typed, lifetime-managed opaque handle. Keep capsules only for throwaway Python-internal plumbing, and never pass a capsule into a dispatcher op.

## Implementation Tasks

### Task 0: Decouple the op layer from pybind return types

**Files:** `src/torch_ext/scene/ops_intersect.cpp`, `src/torch_ext/scene/ops_scene.cpp`, `src/torch_ext/edge/ops_edge.cpp`, `src/torch_ext/reflection/ops.cpp`, `src/torch_ext/diffraction/ops.cpp`

- [ ] Change op function return types from `py::tuple` to `std::tuple<at::Tensor, ...>` or `std::vector<at::Tensor>`.
- [ ] Remove `<torch/extension.h>`/pybind includes from files that will host only dispatcher ops; keep `<torch/library.h>` and ATen.
- [ ] Leave the `*_cuda` launchers and `require_*` tensor validators unchanged.
- [ ] Confirm all op arguments are already dispatcher-compatible types (`int64_t`, `at::Tensor`, `int64_t flags`). Flag any that are not.

### Task 1: Pilot TORCH_LIBRARY registration for scene/intersect

**Files:** new `src/torch_ext/scene/library.cpp` (or extend existing), `CMakeLists.txt`

- [ ] Register `intersect_forward/backward/jvp` (and the scene-build/intersect ops) under `TORCH_LIBRARY(raydtorch, ...)` with explicit schemas and `TORCH_LIBRARY_IMPL(raydtorch, CUDA, ...)`.
- [ ] Keep the existing pybind `bind_intersect_ops` temporarily so nothing breaks during migration.
- [ ] Verify `torch.ops.raydtorch.intersect_forward(...)` is callable from Python and returns the same tensors as `_C.intersect_forward(...)`.
- [ ] Add a minimal C++ dispatcher-call smoke test (or a gtest/`load_inline` probe) proving the op is reachable without Python.

### Task 2: Scene as a TorchBind custom class

**Files:** `include/raydtorch/scene/cache.h`, `src/torch_ext/scene/scene_cache.cpp`, `src/torch_ext/scene/ops_scene.cpp`

- [ ] Make `SceneCache` derive from `torch::CustomClassHolder`; construct via `c10::make_intrusive<SceneCache>(...)`.
- [ ] Register `m.class_<SceneCache>("Scene")` with `torch::init<...>` (build from geometry tensors), `update_vertices`, and `sync`.
- [ ] Change op schemas to take `__torch__.torch.classes.raydtorch.Scene` instead of `int64 scene_handle`.
- [ ] Remove the global `get_scene(scene_handle)` registry from the dispatcher path (op bodies read the cache from the passed Scene). Keep the registry only if a legacy pybind path still needs it during migration, and delete it in Task 5.
- [ ] Verify lifetime: dropping the last Python/C++ reference frees the cache and its OptiX/CUDA buffers.

### Task 3: Migrate edge / reflection / diffraction ops

**Files:** `src/torch_ext/edge/ops_edge.cpp`, `src/torch_ext/reflection/ops.cpp`, `src/torch_ext/diffraction/ops.cpp`, `CMakeLists.txt`

- [ ] Register forward/backward/jvp for edge queries, reflection trace/EPC/visibility/accumulation, and diffraction paths/accumulation/coherent-direct under `TORCH_LIBRARY(raydtorch, ...)`.
- [ ] Take `Scene` (custom class) where these ops currently take `scene_handle`.
- [ ] Keep the tape tensors in each forward return.

### Task 4: Keep the pybind tier for Python-only surfaces

**Files:** `src/torch_ext/module.cpp`

- [ ] Reduce `PYBIND11_MODULE(_raydtorch, ...)` to Python-only surfaces: `build_info`, capability/OptiX-status probes, debug dumps, one-off utilities, arbitrary-object conversions.
- [ ] Ensure the pybind module and the `TORCH_LIBRARY`/`torch::class_` registrations coexist in the same shared library and both load on `import raydtorch`.
- [ ] Confirm OptiX context/pipeline/SBT remain internal and unbound; expose only status through a probe.

### Task 5: Update the Python package to call the dispatcher

**Files:** `raydtorch/scene.py`, `raydtorch/autograd.py`, `raydtorch/__init__.py`, `raydtorch/types.py`

- [ ] `raydtorch.Scene` becomes a thin wrapper around `torch.classes.raydtorch.Scene`.
- [ ] `raydtorch/autograd.py` `torch.autograd.Function`s call `torch.ops.raydtorch.<op>` instead of `_C.<op>`; behavior (tape save/restore, `mark_non_differentiable`, JVP) unchanged. This stays the reference/standalone autograd path.
- [ ] Remove pybind op `m.def`s that are now dispatcher ops; delete the global scene registry once nothing depends on it.
- [ ] Keep the public tensor ABI and dataclass result types (`Intersection`, `NearestPointEdge`, `ReflEpcField`, `DfrAccum`, ...) identical.

### Task 6: Consumer-callability proof

**Files:** `tests/` (new), short C++ or Python harness

- [ ] Demonstrate the redner-style consumer pattern: a `torch.autograd.Function` defined outside RayDTorch that calls `torch.ops.raydtorch.intersect_forward/backward/jvp` with a `torch.classes.raydtorch.Scene`, and matches finite differences.
- [ ] Demonstrate one op invoked through the C++ dispatcher (no Python) returning correct results, to prove the C++ boundary exists.

### Task 7: Parity and no-regression gates

**Files:** existing `tests/`

- [ ] All existing opt-in RayD parity tests (intersect, multi-mesh ids, nearest-edge, visibility, reflection trace, diffraction paths, direct/Keller/suffix accumulation, order-2/3 chains, coherent direct) pass unchanged through the dispatcher path.
- [ ] Existing VJP/JVP tests pass for geometry, edge, reflection trace, EPC, and diffraction accumulation.
- [ ] No Dr.Jit import anywhere; single shared library; no process-global scene registry on the dispatcher path.
- [ ] Performance unchanged within noise versus the current pybind path on the maintained benchmark (`docs/raydtorch_native_performance.md` shapes).

### Task 8: Documentation

**Files:** `README.md`, `docs/api_reference.md`

- [ ] Document the three-tier binding model and the `torch.ops.raydtorch.*` / `torch.classes.raydtorch.Scene` surface.
- [ ] Document the consumer integration contract: depend on the installed package, call dispatcher ops, own your `torch.autograd.Function`, do not compile RayDTorch sources into your own extension.
- [ ] Cross-reference witwin channel plan 16.

## Verification Gates

- `torch.ops.raydtorch.*` ops and `torch.classes.raydtorch.Scene` are callable from both Python and the C++ dispatcher from a single loaded shared library.
- Op layer has no `py::tuple` return types and no pybind dependency; pybind remains only for Python-only glue.
- The Scene lifetime is reference-counted through the custom class; no global `get_scene` registry remains on the dispatcher path.
- Forward returns expose the saved tape; forward/backward/jvp remain separate primitives; no `Autograd`-key kernel is registered in RayDTorch.
- Existing public Python API, tensor ABI, result dataclasses, parity tests, and FD gradient tests all pass unchanged.
- No Dr.Jit dependency; benchmark performance within noise of the pre-migration pybind path.

## Sequencing Note

This migration is not a blocker for a consumer's first integration slice. A consumer (channel path-primal slice) can validate its design against the existing pybind `_C.*` / `raydtorch.Scene` Python API first, then this plan lands the clean `TORCH_LIBRARY` + `torch::class_` boundary once the design is proven and profiling justifies removing the Python re-entry. Do Task 0-2 (op decouple + intersect pilot + Scene custom class) first as the smallest end-to-end proof, then fan out Task 3 across the remaining op families.

## Open Questions

- Dispatcher namespace: `raydtorch` (explicit) vs a shorter `raydt`/`raydt` for call-site brevity. Decide once and keep consistent.
- Should the Scene custom class be picklable/serializable (TorchScript `__getstate__`/`__setstate__`), or is in-process lifetime sufficient for all consumers?
- For multi-tensor returns, keep `Tensor[]` (simple, positional) or define fully named schemas per op (self-documenting, more boilerplate)?
- Do any current ops take or return non-IValue types that block a clean schema and must stay on pybind?
- Does any consumer need a `Meta`/`FakeTensor` registration for shape inference under `torch.compile`, or is dispatcher opacity sufficient initially?

## Relationship to Other Plans

- Implements the RayDTorch-side boundary assumed by witwin `channel/docs/dev/plans/16-torch-runtime-decoupling-plan.md`. It replaces that plan's "reusable native CMake target / direct source linkage" idea (which does not exist today and carries a duplicate-statics hazard) with dispatcher-level integration from a single shared library.
- Builds on `2026-06-06-raydtorch-native-cuda-ad.md` (native CUDA/OptiX forward + fixed-winner VJP/JVP). That plan owns the kernels and gradient math; this plan owns how they are exposed.
- See `docs/raydtorch_native_gap_analysis.md` for the current native parity gap list, which is orthogonal to the binding mechanism.
