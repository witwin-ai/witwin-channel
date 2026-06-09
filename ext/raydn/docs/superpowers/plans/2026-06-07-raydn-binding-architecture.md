# rayd-native Binding Architecture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: Draft, revised after architecture review
Last reviewed: 2026-06-07

**Goal:** Make rayd-native (`raydn`) native operators callable from C++ and the Torch dispatcher (not only from Python pybind), give the Scene/cache a typed dispatcher-friendly lifetime, and keep Python-only glue on pybind11. The driver is letting downstream consumers (notably witwin `channel`, see its plan 16 "Torch Runtime Decoupling") call RayD primitives as Torch-native ops without a Python re-entry and without compiling rayd-native sources into a second extension.

**Architecture:** One shared library, migrated in two proof stages before the broad fan-out. First register one small dispatcher pilot while keeping the current `int64_t scene_handle` registry, proving that the existing compute layer can be called through `torch.ops` without changing Scene lifetime. Then split Scene construction into an intrusive-pointer cache factory plus a legacy handle wrapper, introduce TorchBind `Scene`, and move the pilot op to the typed Scene argument. After both proofs pass, migrate the remaining op families.

Final binding tiers are chosen by call-site requirement:
1. `TORCH_LIBRARY` operators for functional tensor primitives (forward / backward / JVP). Callable from Python (`torch.ops.raydn.*`) and C++ after the extension shared library is loaded, schema-checked, and usable from consumer-owned autograd wrappers. During Tasks 0-10 the loaded extension is `_raydn`; after the final rename it is `_raydn`.
2. `torch::class_<SceneCache>` (TorchBind) for the Scene/cache lifecycle handle. It becomes a typed argument to the ops, is reference-counted, and removes the global `get_scene(int64)` registry from the dispatcher path.
3. `pybind11` for Python-only surfaces: `build_info`, capability probes, debug/introspection, one-off utilities, and any conversion that returns arbitrary Python objects.
   The OptiX context / pipeline / SBT stays an internal C++ owned resource and is not bound at all (only its status is reported through a probe).

**Tech Stack:** PyTorch CUDA C++ extension, `torch/library.h` (`TORCH_LIBRARY`, `torch::class_`), `torch/extension.h` + pybind11 (Python-only tier), ATen, CUDA Runtime/Driver, OptiX, CMake/scikit-build-core, unittest, finite-difference gradient tests.

**Repository:** This plan is for `E:\Code\RayDN` as the current worktree. The target package/distribution naming is `rayd-native` with Python import package `raydn`.

**Environment:** All commands use the conda environment `witwin2`.

**Naming decision:** Use `rayd-native` as the final distribution/display name and `raydn` as the final import/code namespace. Register dispatcher ops under `raydn` (`torch.ops.raydn.*`) and the custom class under `torch.classes.raydn.Scene` from the first dispatcher proof, because these names are the new native contract. Keep the existing Python import package `raydn` and pybind extension `_raydn` during Tasks 0-10 so binding behavior can be proven without rename churn. The Python package, extension target/module, packaging metadata, tests, docs, and user-facing strings are hard-cut to `raydn` only in the final rename task. `rayd-native` is not used as an import or dispatcher namespace because hyphens are not valid there.

---

## Context

The native compute layer is already cleanly factored: each op is a stateless flat function `(scene_handle: int64, at::Tensor...) -> tensors (+ tape)` that forwards to a `*_cuda` launcher. Example: `intersect_forward / intersect_backward / intersect_jvp` in `src/torch_ext/scene/ops_intersect.cpp`.

Current binding facts that this plan changes:

- Ops are currently exposed only through **pybind** `m.def(...)` inside `PYBIND11_MODULE(_raydn, m)` (`src/torch_ext/module.cpp` calls `bind_scene_ops`, `bind_intersect_ops`, `bind_edge_ops`, `bind_reflection_ops`, `bind_diffraction_ops`). They are Python-callable only; nothing is registered with the Torch dispatcher.
- Op functions return `py::tuple`, which couples the op layer to pybind.
- The Scene/cache is passed as an opaque `int64_t scene_handle` resolved through a process-global registry (`get_scene(scene_handle)` -> `SceneCache &`).
- Autograd orchestration lives in Python `raydn/autograd.py` as `torch.autograd.Function` subclasses (e.g. `_IntersectFunction`) that call `_C.<op>` and manage the saved tape via `save_for_backward` / `save_for_forward`.

This design is fine for the standalone Python package, but it blocks three things a Torch-native consumer needs:

1. **C++ / dispatcher callability.** A consumer that wants RayD primitives inside its own native op (or wants `torch.ops`-level composition, `torch.compile` opacity, FakeTensor/meta) cannot reach pybind functions.
2. **A typed, dispatcher-passable Scene.** `int64 + global registry` cannot be a typed op argument and forces manual lifetime plus a process-global singleton.
3. **Consumer-owned autograd.** Per the redner-style ownership in channel plan 16, the *consumer* should own the `torch.autograd.Function` and call RayD's forward/backward/jvp as plain primitives. rayd-native should expose those three flat primitives, not force its own Python autograd wrapper into the consumer's hot path.

## Motivation

- witwin `channel` plan 16 wants RayD geometry/multipath primitives as Torch-native ops backing `torch.ops.wc.*`, with channel owning the autograd functions and the RF material/accumulation kernels. The cheapest correct boundary is: rayd-native registers its primitives as dispatcher ops; channel calls them (from Python for coarse orchestration, or from C++ if it ever fuses) and wraps them in channel-owned `torch.autograd.Function`s.
- `TORCH_LIBRARY` registration is simultaneously a Python API and a C++ API from one registration, with a stable schema contract. This is the native C++ interface the consumer needs without hand-writing a parallel C++ header library or exporting internal types (`SceneCache`, `IntersectForwardOutputs`) as an ABI.
- Keeping rayd-native a single shared library (consumer depends on the installed package, ops self-register on import) avoids the duplicate-statics hazard of compiling rayd-native translation units into a second extension: two scene registries, two OptiX contexts, two CUDA module caches in one process.

## Decision Rule (call-site driven, not uniformity)

Pick the binding tier from what the call site requires, not from a desire to make everything one mechanism:

- Must be callable from C++ / participate in autograd / run in a hot path  -> **`TORCH_LIBRARY` op**.
- Must be a typed handle passed into ops, with managed lifetime, reachable from C++  -> **`torch::class_`**.
- Only Python ever touches it, may return arbitrary Python objects  -> **`pybind11`**.
- Heavy device-global resource the public never holds (OptiX context/pipeline/SBT)  -> **internal C++ owned, not bound**; expose status via a probe only.

## Non-Goals

- Do not compile rayd-native `.cu`/`.cpp` sources directly into a consumer's extension. rayd-native stays one shared library; consumers call through the dispatcher.
- Do not hand-write a separate native core C++ header/library API or export internal structs as a stable ABI. The dispatcher schema is the C++ contract.
- Do not register an `Autograd`-key kernel for these primitives inside rayd-native initially. Consumers own the `torch.autograd.Function`. The `raydn/autograd.py` wrapper remains a reference/standalone path, not the consumer hot path.
- Do not change tensor ABI or result dataclass semantics while renaming the package. The import rename happens last and is a hard cut to `raydn`; do not add a `raydn` compatibility shim.
- Do not introduce a Dr.Jit dependency. The package stays Dr.Jit-free.
- Do not fold forward/backward/jvp into a single fused op. Keep them separate primitives so the consumer can schedule its own tape policy.

## Target Binding Layout

One `.so`/`.pyd`, three registrations coexisting (standard and supported). Fixed-layout ops use fixed multi-return schemas; use `Tensor[]` only where the number of returned tensors is genuinely variable.

```text
TORCH_LIBRARY(raydn, m):          // hot functional primitives, C++ + Python callable
  m.class_<SceneCache>("Scene")       //   lifecycle handle as a typed dispatcher value
     .def(torch::init<                //     flat, index-aligned per-mesh lists
        std::vector<at::Tensor>,      //       vertices[], faces[], uv[], face_uv[],
        std::vector<at::Tensor>,      //       to_world_left[], to_world_right[]
        std::vector<at::Tensor>,      //       (empty tensor = absent optional / identity)
        std::vector<at::Tensor>,
        std::vector<at::Tensor>,
        std::vector<at::Tensor>,
        std::vector<int64_t>>())      //       mesh_flags[]: bit0 face_normals, bit1 edges, bit2 dynamic
     .def("update_vertices", ...)     //     methods limited to IValue types (Tensor/int/...)
     .def("sync", ...)
     .def("version", ...)
     .def("num_meshes", ...)
     .def("edge_count", ...);
  m.def("intersect_forward(__torch__.torch.classes.raydn.Scene scene, Tensor ray_o, "
        "Tensor ray_d, Tensor ray_tmax, Tensor active) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("intersect_backward(...) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("intersect_jvp(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  // edge / reflection / diffraction forward+backward+jvp ...

TORCH_LIBRARY_IMPL(raydn, CUDA, m):
  m.impl("intersect_forward",  &intersect_forward_op);   // fixed-output funcs return std::tuple<at::Tensor, ...>
  m.impl("intersect_backward", &intersect_backward_op);  //   (no py::tuple in the op layer)
  m.impl("intersect_jvp",      &intersect_jvp_op);

PYBIND11_MODULE(_raydn, m):   // Python-only glue during Tasks 0-10; _raydn after final rename
  m.def("build_info", ...);           //   returns py::dict
  // capability probes, debug dumps, one-off utilities, arbitrary-object conversions

// OptiX context / pipeline / SBT: internal C++ owned singleton/resource, not bound.
```

Notes:

- `SceneCache` must inherit `torch::CustomClassHolder` and be constructed/held via `c10::intrusive_ptr<SceneCache>`.
- Do not try to replace the current global registry in the same patch that first introduces dispatcher ops. The migration has two proof stages:
  1. register a pilot dispatcher op using the current `int64_t scene_handle`;
  2. split Scene construction into an intrusive-pointer cache factory, then switch the pilot op to `torch.classes.raydn.Scene`.
- Scene construction should be factored as `c10::intrusive_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes)`. The legacy `int64_t create_scene(std::vector<MeshRecord>)` becomes a wrapper that calls the cache factory and stores the intrusive pointer in the old registry during migration.
- Fixed multi-tensor returns use fixed tuple schemas, not `Tensor[]`. The forward output-vs-tape split is still documented next to each registration (e.g. `intersect_forward` = 10 public outputs + 3 tape: `[t,p,n,geo_n,uv,barycentric,shape_id,prim_id,local_prim_id,global_prim_id, tape_prim_id,tape_barycentric,tape_t]`). Use `Tensor[]` only where the number of tensors is genuinely variable.
- Custom-class method/argument types are limited to IValue-convertible types. `update_vertices`/`sync` are already IValue-clean (int, Tensor). The **build path is not**: today `create_scene_op(py::list mesh_specs)` reads `py::dict` specs, which cannot cross the dispatcher. It is reshaped in Task 3 into flat, index-aligned per-mesh lists with an empty-tensor sentinel for absent optionals and an `int64` flag bitmask. The ergonomic dict/kwargs API stays in the Python `raydn.Scene` wrapper until the final rename task, then becomes `raydn.Scene`; the wrapper assembles the lists and calls the IValue-clean init. Anything else with a non-IValue C++ interface stays on pybind until a task explicitly reshapes it.
- Optional Tensor arguments in dispatcher schemas use `Tensor?` / `c10::optional<at::Tensor>` when a real `None` is part of the public contract, and empty tensor sentinels only when the existing ABI already treats empty as meaningful. Do not carry `py::object` into a dispatcher implementation.
- No `Autograd`-key kernel is registered for these ops initially (the consumer owns the `torch.autograd.Function`). Consequence: calling `torch.ops.raydn.*_forward` directly on `requires_grad` inputs runs forward without tracking gradients. Document this loudly; keep the Python wrappers in `raydn/autograd.py` during Tasks 0-10, then `raydn/autograd.py` after the final rename, as the only blessed grad-enabled path. Do **not** add an Autograd-key "raise" kernel until a focused test proves it does not break calls from inside `torch.autograd.Function.forward`.
- Capsules are retired for the Scene path: the custom class is the typed, lifetime-managed opaque handle. Keep capsules only for throwaway Python-internal plumbing, and never pass a capsule into a dispatcher op.
- C++ consumers must ensure the extension shared library has been loaded before calling the dispatcher. During Tasks 0-10 this is `_raydn` and Python consumers satisfy it by `import raydn`; after the final rename this is `_raydn` and Python consumers satisfy it by `import raydn`. Pure C++ tests must explicitly load the extension or link/load the library before `c10::Dispatcher::findSchemaOrThrow`.

## Internal Value Container

The op layer, the `*_cuda` launchers, and the op return structs (`IntersectForwardOutputs`, `EdgeForwardOutputs`, ...) keep `at::Tensor` as the value container. Do not introduce a custom host-side owning math type as the inter-function or inter-library currency.

- Zero-copy across the boundary is automatic: the op receives a `Tensor`, the launcher holds it, the kernel reads `data_ptr`. Nothing is converted, so there is no copy to eliminate. A custom owning buffer would instead reimplement the CUDA caching allocator or fall back to `cudaMalloc`/`cudaFree` (a device-synchronizing free), and would still have to convert at every `Tensor`-typed dispatcher op.
- Host allocation cost is amortized by torch's `CUDACachingAllocator` (alloc/free hit a pool, not the driver). Keep it cheap downstream by batching `[N, ...]` rather than per-element tensors, using `at::empty` (not `zeros`) for fully-overwritten buffers, and reusing a persistent per-solve workspace for scratch.
- A `struct` of named `at::Tensor` (the existing `*Outputs` pattern) is the right "typed bundle": keep the names, keep the elements as tensors.
- The shared math vocabulary lives at the **device** layer, header-only and ATen-free: `common/complex.cuh` (`Complex`/`Complex3`) and `common/math.cuh` (`float3` operators). The channel consumer includes these directly in its own kernels. This costs zero copies (on-device) and introduces no cross-`.so` C++ ABI -each extension recompiles the headers -consistent with the Non-Goal of exporting internal structs as a stable ABI. A non-owning accessor view (e.g. `{const float3* p; int n;}`) is allowed only as a local kernel-launch convenience inside one launcher, never as a value crossing an op boundary.

## Implementation Tasks

### Task 0: Add a dispatcher-safe op core for intersect only

**Files:** `src/torch_ext/scene/ops_intersect.cpp`

- [ ] Split `intersect_forward_op`, `intersect_backward_op`, `intersect_backward_t_op`, and `intersect_jvp_op` into dispatcher-safe core functions that return fixed `std::tuple<at::Tensor, ...>`.
- [ ] Keep the existing pybind functions as thin wrappers that call the core functions and convert the result to `py::tuple`.
- [ ] Keep all arguments as the current `int64_t scene_handle` for this task; do not touch Scene lifetime yet.
- [ ] Leave the `*_cuda` launchers and `require_*` tensor validators unchanged.
- [ ] Run the existing intersect tests through the pybind path to prove behavior is unchanged.

### Task 1: Register an int64-handle dispatcher pilot for intersect

**Files:** new `src/torch_ext/scene/library.cpp` (or extend existing), `CMakeLists.txt`

- [ ] Register only `intersect_forward`, `intersect_backward`, `intersect_backward_t`, and `intersect_jvp` under `TORCH_LIBRARY(raydn, ...)` with explicit fixed tuple schemas and `int scene_handle` arguments.
- [ ] Register the CUDA implementations with `TORCH_LIBRARY_IMPL(raydn, CUDA, ...)`.
- [ ] Keep the existing pybind `bind_intersect_ops` temporarily so nothing breaks during migration.
- [ ] Verify `torch.ops.raydn.intersect_forward(...)` is callable from Python and returns the same tensors as `_C.intersect_forward(...)`.
- [ ] Add a C++ dispatcher-call smoke test that first loads `_raydn`, then calls `c10::Dispatcher::singleton().findSchemaOrThrow(...)`; this proves the C++ boundary exists after the shared library is loaded before package rename.
- [ ] Do not register Scene construction in the dispatcher yet.

### Task 2: Split Scene construction from the legacy global registry

**Files:** `include/raydn/scene/cache.h`, `src/torch_ext/scene/scene_cache.cpp`

- [ ] Make `SceneCache` derive from `torch::CustomClassHolder`.
- [ ] Change the legacy registry from `std::unique_ptr<SceneCache>` to `c10::intrusive_ptr<SceneCache>`.
- [ ] Add `c10::intrusive_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes)` that contains the current build logic from `create_scene`.
- [ ] Keep legacy `int64_t create_scene(std::vector<MeshRecord> meshes)` as a wrapper that calls `create_scene_cache`, assigns/stores the handle, and returns the handle.
- [ ] Keep `destroy_scene`, `get_scene`, `scene_version`, `scene_num_meshes`, `scene_edge_count`, `update_mesh_vertices`, and `sync_scene` working for the legacy pybind path.
- [ ] Verify all current Scene/cache tests still pass before introducing TorchBind.

### Task 3: Introduce TorchBind Scene beside the legacy handle path

**Files:** `include/raydn/scene/cache.h`, `src/torch_ext/scene/ops_scene.cpp`, new or existing dispatcher registration file

- [ ] Register `m.class_<SceneCache>("Scene")` with `update_vertices`, `sync`, `version`, `num_meshes`, `edge_count`, and a flat, index-aligned per-mesh `torch::init`:
  - args: `vertices: Tensor[]`, `faces: Tensor[]`, `uv: Tensor[]`, `face_uv: Tensor[]`, `to_world_left: Tensor[]`, `to_world_right: Tensor[]`, `mesh_flags: int[]`.
  - all lists share one length = mesh count; an **empty tensor** is the sentinel for an absent optional (`uv`/`face_uv` = none, `to_world_*` = identity).
  - `mesh_flags[i]` is a bitmask: bit0 `use_face_normals`, bit1 `edges_enabled`, bit2 `dynamic` (avoids the `std::vector<bool>` proxy and leaves room for future flags).
  - the init body assembles `std::vector<MeshRecord>` exactly as `create_scene_op` does today, then calls `create_scene_cache(...)`; reuse the existing geometry/OptiX assembly unchanged.
- [ ] Add a Python smoke test that builds `torch.classes.raydn.Scene(...)` directly and calls its `version`, `num_meshes`, and `edge_count` methods.
- [ ] Verify lifetime: dropping the last Python/C++ reference frees the cache and its OptiX/CUDA buffers.

### Task 4: Move the intersect dispatcher pilot to typed Scene

**Files:** `src/torch_ext/scene/ops_intersect.cpp`, dispatcher registration file, tests

- [ ] Add typed-Scene core wrappers for `intersect_forward`, `intersect_backward`, `intersect_backward_t`, and `intersect_jvp` that accept `c10::intrusive_ptr<SceneCache>` and use `*scene` instead of `get_scene(scene_handle)`.
- [ ] Change the dispatcher schemas for the pilot intersect ops from `int scene_handle` to `__torch__.torch.classes.raydn.Scene scene`.
- [ ] Keep legacy pybind wrappers on `int64_t scene_handle` until the Python package is migrated.
- [ ] Verify `torch.ops.raydn.intersect_forward(scene_obj, ...)` matches the legacy `_C.intersect_forward(handle, ...)`.
- [ ] Verify the C++ dispatcher smoke test passes with a typed custom-class argument after explicitly loading `_raydn` before package rename.

### Task 5: Migrate edge op cores and dispatcher schemas

**Files:** `src/torch_ext/edge/ops_edge.cpp`, dispatcher registration file, `CMakeLists.txt`, `tests/raydn_native/test_edge_queries.py`

- [ ] Repeat the Task 0 pattern for edge queries: core functions return fixed tuples or named output structs converted to fixed tuples; pybind wrappers stay thin during migration.
- [ ] Register `nearest_edge_forward`, `nearest_edge_forward_noad`, `nearest_edge_ray_forward`, `nearest_edge_backward`, `nearest_edge_backward_optional`, `nearest_edge_jvp`, and `nearest_edge_jvp_optional` under `TORCH_LIBRARY(raydn, ...)`.
- [ ] Use typed `Scene` for scene-dependent forward ops and backward/JVP ops that need scene geometry.
- [ ] Convert optional upstream grads/tangents from `py::object` to `Tensor?` / `c10::optional<at::Tensor>` in dispatcher cores.
- [ ] Keep all forward tape tensors in the fixed return tuple, with an index map documented next to each schema.
- [ ] Verify pybind and dispatcher outputs match for point nearest-edge, ray nearest-edge, backward, and JVP.
- [ ] Run `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydn_native.test_edge_queries -v`.

### Task 6: Migrate reflection visibility / trace / EPC / accumulation dispatcher schemas

**Files:** `src/torch_ext/reflection/ops.cpp`, dispatcher registration file, `CMakeLists.txt`, `tests/raydn_native/test_multipath.py`

- [ ] Repeat the dispatcher-safe core pattern for `visibility_forward`.
- [ ] Repeat the pattern for `trace_reflections_forward`, `trace_reflections_forward_noad`, `trace_reflections_forward_reduced`, `trace_reflections_backward`, `trace_reflections_backward_optional`, `trace_reflections_jvp`, and `trace_reflections_jvp_optional`.
- [ ] Repeat the pattern for `trace_refl_epc_field_forward`, `trace_refl_epc_field_backward`, and `trace_refl_epc_field_jvp`.
- [ ] Register `reflection_dedup_forward` and `reflection_accumulation_forward` if they are kept as Tensor hot-path primitives; otherwise explicitly document why they remain pybind-only.
- [ ] Convert optional active masks, upstream grads, and tangents from `py::object` to dispatcher-safe optional Tensor or empty-sentinel semantics without adding Python-side staging/copies.
- [ ] Keep all forward tape tensors in fixed tuple schemas, with index maps documented next to each schema.
- [ ] Run `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydn_native.test_multipath -v`.

### Task 7: Migrate diffraction paths and accumulation dispatcher schemas

**Files:** `src/torch_ext/diffraction/ops.cpp`, dispatcher registration file, `CMakeLists.txt`, `tests/raydn_native/test_multipath.py`

- [ ] Repeat the dispatcher-safe core pattern for `diffraction_paths_order1_forward`.
- [ ] Repeat the pattern for `diffraction_accumulation_forward`, `diffraction_accumulation_direct_backward`, `diffraction_accumulation_direct_jvp`, `diffraction_accumulation_chain_backward`, and `diffraction_accumulation_chain_jvp`.
- [ ] Register `diffraction_coherent_accumulation_forward`.
- [ ] Convert optional active masks, recursive active masks, upstream grads, and tangents from `py::object` to dispatcher-safe optional Tensor or empty-sentinel semantics without adding Python-side staging/copies.
- [ ] Keep all forward tape tensors in fixed tuple schemas, with index maps documented next to each schema.
- [ ] Run `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydn_native.test_multipath -v`.

### Task 8: Migrate dispatcher-safe common utility ops and multi-mesh helpers

**Files:** `src/torch_ext/common/ops_stats.cpp`, `src/torch_ext/common/ops_camera.cpp`, `src/torch_ext/scene/ops_scene.cpp`, dispatcher registration file, `raydn/autograd.py`, `raydn/camera.py`, `raydn/types.py`, tests

- [ ] Decide and document each non-scene pybind Tensor op: camera sampling/backward, `intersection_valid`, `default_dfr_material`, `reflection_trace_stats`, and `diffraction_path_stats` should move to dispatcher if they are Tensor hot-path primitives.
- [ ] Register dispatcher schemas for all migrated common utility ops and keep pybind wrappers thin until Python call sites move.
- [ ] Replace `split_scene_vertex_grad(py::tuple)` with a dispatcher-safe helper surface. Prefer `Tensor[] split_scene_vertex_grad(Scene scene, Tensor grad_vertices)` because the return count is genuinely variable by mesh count.
- [ ] Replace `pack_scene_vertex_tangents(py::args)` with a dispatcher-safe helper surface. Prefer `Tensor? pack_scene_vertex_tangents(Scene scene, Tensor?[] tangents)` or an explicit empty-tensor sentinel list, and document the chosen convention next to the schema.
- [ ] Update Python autograd wrappers to call `torch.ops.raydn.*` for migrated ops while the Python import package still remains `raydn`.
- [ ] Keep behavior, saved tape policy, `mark_non_differentiable`, VJP, and JVP unchanged.
- [ ] Run focused camera, public API contract, intersect gradient, edge, and multipath tests.

### Task 9: Reduce pybind to Python-only glue and remove dispatcher-path registry dependency

**Files:** `src/torch_ext/module.cpp`, op binding files, `src/torch_ext/scene/scene_cache.cpp`, `include/raydn/scene/cache.h`, Python package files

- [ ] Remove pybind `m.def`s for ops that now have dispatcher registrations and Python dispatcher call sites.
- [ ] Keep pybind only for Python-only surfaces: `build_info`, capability/OptiX-status probes, debug dumps, one-off utilities, and arbitrary-object conversions.
- [ ] Keep legacy `int64_t` scene registry only while any remaining Python path needs it; delete it after all scene-dependent Python calls use `torch.classes.raydn.Scene`.
- [ ] Ensure the pybind module and the `TORCH_LIBRARY`/`torch::class_` registrations coexist in the same shared library and both load on `import raydn` during this task.
- [ ] Confirm OptiX context/pipeline/SBT remain internal and unbound; expose only status through a probe.
- [ ] Run `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydn_native -v`.

### Task 10: Consumer-callability proof and no-regression gates

**Files:** `tests/` (new), short C++ or Python harness, existing `tests/`

- [ ] Demonstrate the redner-style consumer pattern before package rename: a `torch.autograd.Function` defined outside rayd-native calls `torch.ops.raydn.intersect_forward/backward/jvp` with `torch.classes.raydn.Scene` and matches finite differences.
- [ ] Demonstrate one op invoked through the C++ dispatcher after explicit `_raydn` library load returning correct results, to prove the C++ boundary exists without pybind function calls.
- [ ] Demonstrate that directly calling `torch.ops.raydn.*_forward` on `requires_grad` inputs does not silently replace the documented autograd path; document the observed behavior.
- [ ] All existing opt-in RayD parity tests pass unchanged through the dispatcher path.
- [ ] Existing VJP/JVP tests pass for geometry, edge, reflection trace, EPC, and diffraction accumulation.
- [ ] No Dr.Jit import anywhere; single shared library; no process-global scene registry on the dispatcher path.
- [ ] Performance unchanged within noise versus the current pybind path on the maintained benchmark (`docs/raydn_native_performance.md` shapes).

### Task 11: Final package rename to `raydn`

**Files:** `pyproject.toml`, `CMakeLists.txt`, current `raydn/` package files, target `raydn/` package files, tests, docs

- [ ] Rename the import package from `raydn` to `raydn` and update packaging metadata so the distribution/display name is `rayd-native`.
- [ ] Rename the pybind extension target/module from `_raydn` to `_raydn`, keeping the shared library self-registration behavior unchanged.
- [ ] `raydn.Scene` remains a thin wrapper around `torch.classes.raydn.Scene`.
- [ ] The wrapper still exposes `is_ready`, `num_meshes`, `version`, `edge_count`, `add_mesh`, `build`, `update_mesh_vertices`, and `sync` with the current public behavior.
- [ ] `raydn/autograd.py` `torch.autograd.Function`s continue to call `torch.ops.raydn.<op>`; behavior (tape save/restore, `mark_non_differentiable`, JVP) unchanged. This stays the reference/standalone autograd path.
- [ ] Keep the public tensor ABI and dataclass result types (`Intersection`, `NearestPointEdge`, `ReflEpcField`, `DfrAccum`, ...) identical.
- [ ] Hard-cut all tests and docs to import `raydn` only; do not keep a `raydn` compatibility shim.

### Task 12: Documentation

**Files:** `README.md`, `docs/api_reference.md`

- [ ] Document the three-tier binding model and the `torch.ops.raydn.*` / `torch.classes.raydn.Scene` surface.
- [ ] Document the consumer integration contract: depend on the installed `rayd-native` package, call dispatcher ops, own your `torch.autograd.Function`, do not compile rayd-native sources into your own extension.
- [ ] Cross-reference witwin channel plan 16.

## Verification Gates

- Before the final package rename, `torch.ops.raydn.*` ops and `torch.classes.raydn.Scene` are callable from both Python and the C++ dispatcher after loading `_raydn`; after the final rename the same dispatcher surface works after loading `_raydn`.
- Final dispatcher op layer has no `py::tuple` return types and no pybind dependency; pybind remains only for Python-only glue.
- The Scene lifetime is reference-counted through the custom class; no global `get_scene` registry remains on the dispatcher path.
- Fixed-layout forward returns use fixed tuple schemas and expose the saved tape; forward/backward/jvp remain separate primitives; no `Autograd`-key kernel is registered initially.
- Public tensor ABI, result dataclasses, parity tests, and FD gradient tests all pass unchanged; Python import paths intentionally hard-cut to `raydn`.
- No Dr.Jit dependency; benchmark performance within noise of the pre-migration pybind path.

## Sequencing Note

This migration is not a blocker for a consumer's first integration slice. A consumer (channel path-primal slice) can validate its design against the existing pybind `_C.*` / `raydn.Scene` Python API first, then this plan lands the clean `TORCH_LIBRARY` + `torch::class_` boundary once the design is proven and profiling justifies removing the Python re-entry. Do Task 0-1 first as the smallest dispatcher proof, Task 2-4 as the Scene lifetime proof, then migrate one op family at a time in Tasks 5-8. Reduce pybind and prove no regressions in Tasks 9-10. Do the package/import/extension rename last in Task 11 so rename churn cannot mask binding regressions.

## Open Questions

- Does any consumer need a `Meta`/`FakeTensor` registration for shape inference under `torch.compile`? Note: custom-class ops are expected graph breaks initially; fixed tuple schemas still need Meta kernels if they must participate in FakeTensor shape propagation. Deferred unless a consumer requires compile-time shape inference.
- Should direct dispatcher calls on `requires_grad` tensors raise, warn, or remain silent? Recommended default for the first implementation is silent plus documentation, because an Autograd-key raise kernel may interfere with calls made inside `torch.autograd.Function.forward`.

Resolved since first draft:

- **Scene construction:** flat, index-aligned per-mesh lists into `torch::init` (Task 3). The dict/kwargs ergonomic layer stays in the Python `raydn.Scene` wrapper.
- **Multi-tensor returns:** fixed-layout ops use fixed tuple schemas with a documented output-vs-tape index map. Use `Tensor[]` only for genuinely variable-length returns.
- **Non-IValue surfaces:** only `create_scene_op(py::list)`; resolved by the flat Scene init. All 28 compute ops are already IValue-clean.
- **Internal value container:** `at::Tensor` at the host layer; share device `.cuh` math with the consumer (see Internal Value Container).
- **Migration sequencing:** first prove dispatcher registration with legacy `int64_t scene_handle`, then introduce TorchBind Scene, then migrate the rest of the op families.
- **Naming:** final distribution/display name is `rayd-native`; final Python import package is `raydn`; Torch dispatcher namespace and Torch custom-class namespace are `raydn` from the dispatcher proof onward; pybind extension target remains `_raydn` until the final rename to `_raydn`.
- **Import compatibility:** package rename is last; when it happens, hard cut to `import raydn`; do not keep a temporary `raydn` compatibility shim.
- **Scene serialization:** Scene is in-process only. Do not implement TorchScript pickling/serialization for `torch.classes.raydn.Scene`.

## Relationship to Other Plans

- Implements the rayd-native-side boundary assumed by witwin `channel/docs/dev/plans/16-torch-runtime-decoupling-plan.md`. It replaces that plan's "reusable native CMake target / direct source linkage" idea (which does not exist today and carries a duplicate-statics hazard) with dispatcher-level integration from a single shared library.
- Builds on `2026-06-06-raydn-native-cuda-ad.md` (native CUDA/OptiX forward + fixed-winner VJP/JVP). That plan owns the kernels and gradient math; this plan owns how they are exposed.
- See `docs/raydn_native_gap_analysis.md` for the current native parity gap list, which is orthogonal to the binding mechanism.
