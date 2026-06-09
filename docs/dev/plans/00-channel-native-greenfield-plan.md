# Channel Native RayDN Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `witwin.channel_native` as a new DrJit-free Torch/CUDA RF channel runtime that consumes the current RayD native Torch code snapshot as vendored RayDN sources under `ext/raydn`.

**Architecture:** Channel Native owns the public RF scene model, material semantics, solver orchestration, accumulation contracts, metadata, and fixed-topology AD boundaries. RayDN owns reusable CUDA/OptiX geometry, edge, reflection, diffraction, dispatcher, and scene-cache primitives. Production hot paths call Channel Native native ops or RayDN dispatcher/native C++ surfaces; they must not call Python `raydn` wrappers from solver execution.

**Tech Stack:** Python 3.10+, PyTorch CUDA tensors, C++17, CUDA 17 mode, OptiX, RayDN vendored source snapshot, CMake/scikit-build-core, pybind11/Torch extension bindings, pytest, optional Nsight profiling.

---

## Current Source Snapshot

The upstream source lives at:

```text
E:\Code\RayDTorch
```

The current upstream code has been renamed internally to RayDN:

```text
distribution/display name: rayd-native
Python package:            raydn
native extension module:   _raydn
Torch dispatcher namespace: torch.ops.raydn.*
Torch class namespace:      torch.classes.raydn.*
```

The vendored snapshot for this project is:

```text
ext/raydn
```

`ext/raydtorch` may exist as an older snapshot, but new Channel Native work must target `ext/raydn`. Do not add a `raydtorch` compatibility layer.

## Product Boundary

`witwin.channel_native` is not a compatibility wrapper over the existing channel package. It is a new implementation with a scene-oriented public API:

```python
import witwin.channel_native as cn

scene = cn.Scene(
    structures=[...],
    transmitters=[...],
    receivers=[...],
    frequency=3.5e9,
)

result = cn.montecarlo.basic.solve(
    scene,
    cn.montecarlo.basic.Config(samples=4096, seed=1),
)
```

The first complete solver is Monte Carlo basic. Deterministic, path, PSDR, and Monte Carlo BDPT packages are reserved API shells until their phases begin.

## Hard Rules

- DrJit is forbidden in package dependencies, imports, native code, runtime metadata, and production tests.
- `import witwin.channel_native` must not import Python `raydn`.
- Production solver hot paths must not call Python `raydn.Scene` or Python `raydn.autograd` wrappers.
- RayDN is consumed from `ext/raydn` through native C++/CUDA/OptiX targets, Torch dispatcher ops, Torch custom classes, or a narrow Channel Native CMake shim.
- Public constructors use structured scene objects. Raw tensor-only scene construction is allowed only as an internal or test helper.
- Geometry, materials, and assignments are separate runtime stores with separate invalidation versions.
- Material lookup for hot paths happens inside fused Channel Native CUDA primitives or launch families, not through Python-created per-face/per-path material tensors.
- CPU fallback solvers are not allowed. Small pure-Torch references may exist only under `tests/support` for validation.
- Boundary, visibility silhouette, diffraction topology, primitive-id, edge-id, path-topology, visibility-decision, compaction-order, and random-sample-decision gradients are out of scope for the first complete solver.
- Unsupported AD combinations raise explicit errors. They must not silently return zero gradients.

## Target Layout

```text
channel_native/
  pyproject.toml
  CMakeLists.txt
  README.md
  docs/dev/plans/00-channel-native-greenfield-plan.md
  cmake/RayDNNative.cmake
  ext/raydn/
  src/witwin/channel_native/
    __init__.py
    core/
      __init__.py
      objects.py
      scene.py
      materials.py
      runtime/
        __init__.py
        geometry.py
        material_store.py
        assignments.py
        compiled_scene.py
        raydn.py
        workspace.py
      kernels/
        __init__.py
        extension.py
        ops.py
        autograd.py
        reference.py
    montecarlo/
      __init__.py
      basic/
        __init__.py
        config.py
        result.py
        metadata.py
        sampling.py
        backend.py
        solver.py
      bdpt/
        __init__.py
    deterministic/
      __init__.py
      config.py
      result.py
      solver.py
    path/
      __init__.py
      config.py
      result.py
      solver.py
    psdr/
      __init__.py
      config.py
      result.py
      solver.py
  native/channel_native/
    bindings.cpp
    build_info.cpp
    scene_bridge.cpp
    montecarlo_basic.cpp
    kernels/
      complex.cuh
      material_eval.cuh
      los.cu
      reflection.cu
      diffraction.cu
      accum.cu
  tests/
    test_import_contract.py
    core/
    kernels/
    montecarlo/basic/
    reserved_api/
    support/
```

## Runtime Contract

Runtime tensors are Torch-owned CUDA tensors. The authoritative stores are separated so edits invalidate only the relevant cache:

```python
from dataclasses import dataclass
import torch

@dataclass(slots=True, frozen=True)
class GeometryStore:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    edges: torch.Tensor
    edge_adj_faces: torch.Tensor
    edge_param_range: torch.Tensor
    face_structure_id: torch.Tensor
    face_surface_id: torch.Tensor
    version: int

@dataclass(slots=True, frozen=True)
class MaterialStore:
    eps_r: torch.Tensor
    mu_r: torch.Tensor
    sigma_e: torch.Tensor
    gain: torch.Tensor
    model_id: torch.Tensor
    model_params: torch.Tensor
    frequency_hz: float
    version: int

@dataclass(slots=True, frozen=True)
class AssignmentStore:
    face_material_id: torch.Tensor
    edge_material_id0: torch.Tensor
    edge_material_id1: torch.Tensor
    surface_material_id: torch.Tensor
    structure_material_id: torch.Tensor
    version: int
```

`CompiledScene` owns these stores, the RayDN scene/cache handle, and solver workspace. Vertex edits invalidate geometry and RayDN acceleration state. Material edits invalidate only material state. Assignment edits invalidate only assignment state. Frequency edits invalidate material state and wavelength-dependent workspace.

## Phase 0: Bootstrap Package And Import Contract

**Outcome:** The package installs, imports cleanly, and exposes native build metadata without DrJit or Python RayDN imports.

**Files:**
- Create: `pyproject.toml`
- Create: `CMakeLists.txt`
- Create: `src/witwin/channel_native/__init__.py`
- Create: `src/witwin/channel_native/core/kernels/extension.py`
- Create: `native/channel_native/bindings.cpp`
- Create: `native/channel_native/build_info.cpp`
- Create: `tests/test_import_contract.py`
- Create: `tests/kernels/test_build_info.py`

- [ ] **Step 1: Write import tests**

```python
import sys

def test_channel_native_import_does_not_import_drjit_or_raydn():
    sys.modules.pop("witwin.channel_native", None)
    import witwin.channel_native  # noqa: F401

    assert "drjit" not in sys.modules
    assert "raydn" not in sys.modules
```

- [ ] **Step 2: Write build-info smoke test**

```python
from witwin.channel_native.core.kernels.extension import build_info

def test_build_info_contract():
    info = build_info()
    assert info["backend"] == "channel-native"
    assert info["uses_dr_jit"] is False
    assert "uses_raydn_native" in info
    assert "cuda_available" in info
    assert "optix_available" in info
```

- [ ] **Step 3: Run tests and confirm they fail before implementation**

```powershell
conda run -n witwin2 python -m pytest tests/test_import_contract.py tests/kernels/test_build_info.py -q
```

Expected: import or build-info tests fail because package/native module is not implemented.

- [ ] **Step 4: Implement minimal package and native module**

`build_info()` must return:

```python
{
    "backend": "channel-native",
    "uses_dr_jit": False,
    "uses_raydn_native": False,
    "cuda_available": False,
    "optix_available": False,
}
```

Native C++ should expose the same fields through `_channel_native.build_info()` once the extension is available. The Python facade may return conservative values when the extension is not built yet, but tests for native metadata should require the extension path after build setup lands.

- [ ] **Step 5: Run acceptance**

```powershell
conda run -n witwin2 python -m pytest tests/test_import_contract.py tests/kernels/test_build_info.py -q
```

Expected: all selected tests pass.

## Phase 1: RayDN Vendored Native Link

**Outcome:** Channel Native can compile or load against vendored RayDN native sources without Python hot-path dependency.

**Files:**
- Existing: `ext/raydn`
- Create: `cmake/RayDNNative.cmake`
- Modify: `CMakeLists.txt`
- Create: `src/witwin/channel_native/core/runtime/raydn.py`
- Create: `native/channel_native/scene_bridge.cpp`
- Create: `tests/kernels/test_raydn_native_link.py`
- Create: `tests/core/test_raydn_scene_contract.py`

- [ ] **Step 1: Write native capability tests**

```python
from witwin.channel_native.core.kernels.extension import build_info

def test_build_info_reports_raydn_key():
    info = build_info()
    assert "uses_raydn_native" in info
    assert isinstance(info["uses_raydn_native"], bool)
```

- [ ] **Step 2: Write no-Python-RayDN scene construction test**

```python
import sys

def test_raydn_scene_wrapper_does_not_import_python_raydn():
    sys.modules.pop("raydn", None)
    from witwin.channel_native.core.runtime.raydn import RayDNScene

    assert RayDNScene.__name__ == "RayDNScene"
    assert "raydn" not in sys.modules
```

- [ ] **Step 3: Run tests and confirm they fail**

```powershell
conda run -n witwin2 python -m pytest tests/kernels/test_raydn_native_link.py tests/core/test_raydn_scene_contract.py -q
```

Expected: tests fail because RayDN integration does not exist yet.

- [ ] **Step 4: Add `cmake/RayDNNative.cmake`**

The CMake module must prefer a vendored RayDN target if one is exported. If RayDN only builds `_raydn`, define `cn_raydn_shim` as the narrow target that compiles only required `ext/raydn/include` and `ext/raydn/src/torch_ext` sources.

The first shim target must not link or call Python `raydn` APIs.

- [ ] **Step 5: Add `RayDNScene` wrapper**

```python
class RayDNScene:
    """Opaque wrapper for a native RayDN scene/cache handle."""

    def __init__(self, handle: object | None = None) -> None:
        self._handle = handle

    @property
    def handle(self) -> object | None:
        return self._handle
```

Replace the `object | None` handle with the actual native custom class or C++ handle once the native bridge exists.

- [ ] **Step 6: Run acceptance**

```powershell
conda run -n witwin2 python -m pytest tests/kernels/test_raydn_native_link.py tests/core/test_raydn_scene_contract.py -q
```

Expected: all selected tests pass; `raydn` is not present in `sys.modules`.

## Phase 2: Public Scene And Runtime Stores

**Outcome:** Public scene objects compile into separated CUDA runtime stores.

**Files:**
- Create: `src/witwin/channel_native/core/objects.py`
- Create: `src/witwin/channel_native/core/materials.py`
- Create: `src/witwin/channel_native/core/scene.py`
- Create: `src/witwin/channel_native/core/runtime/geometry.py`
- Create: `src/witwin/channel_native/core/runtime/material_store.py`
- Create: `src/witwin/channel_native/core/runtime/assignments.py`
- Create: `src/witwin/channel_native/core/runtime/compiled_scene.py`
- Create: `tests/core/test_public_scene.py`
- Create: `tests/core/test_runtime_stores.py`
- Create: `tests/core/test_compiled_scene_invalidation.py`

- [ ] **Step 1: Write public constructor tests for `Structure`, `Transmitter`, `ReceiverPoint`, and `ReceiverGrid`.**
- [ ] **Step 2: Write material tests for `Dielectric`, `LossyDielectric`, and `PerfectConductor`.**
- [ ] **Step 3: Write store validation tests for CUDA device, dtype, shape, and contiguity.**
- [ ] **Step 4: Write invalidation tests proving geometry, materials, assignments, and frequency versions change independently.**
- [ ] **Step 5: Run tests and confirm they fail for missing objects.**
- [ ] **Step 6: Implement dataclasses and validators with no solver behavior.**
- [ ] **Step 7: Implement `Scene.compile()` returning `CompiledScene`.**
- [ ] **Step 8: Run acceptance.**

```powershell
conda run -n witwin2 python -m pytest tests/core/test_public_scene.py tests/core/test_runtime_stores.py tests/core/test_compiled_scene_invalidation.py -q
```

## Phase 3: Reserved Solver API Shells

**Outcome:** Deterministic, path, PSDR, and MC BDPT packages import cleanly and fail explicitly when called.

**Files:**
- Create: `src/witwin/channel_native/deterministic/config.py`
- Create: `src/witwin/channel_native/deterministic/result.py`
- Create: `src/witwin/channel_native/deterministic/solver.py`
- Create: `src/witwin/channel_native/path/config.py`
- Create: `src/witwin/channel_native/path/result.py`
- Create: `src/witwin/channel_native/path/solver.py`
- Create: `src/witwin/channel_native/psdr/config.py`
- Create: `src/witwin/channel_native/psdr/result.py`
- Create: `src/witwin/channel_native/psdr/solver.py`
- Create: `src/witwin/channel_native/montecarlo/bdpt/__init__.py`
- Create: `tests/reserved_api/test_reserved_solver_imports.py`
- Create: `tests/reserved_api/test_reserved_solver_errors.py`

- [ ] **Step 1: Write import tests for all reserved packages.**
- [ ] **Step 2: Write `solve(...)` error tests expecting solver-specific `NotImplementedError` messages.**
- [ ] **Step 3: Run tests and confirm they fail.**
- [ ] **Step 4: Add minimal `Config` and `Result` dataclasses.**
- [ ] **Step 5: Add `solve(scene, config)` functions that raise explicit phase messages.**
- [ ] **Step 6: Run acceptance.**

```powershell
conda run -n witwin2 python -m pytest tests/reserved_api/test_reserved_solver_imports.py tests/reserved_api/test_reserved_solver_errors.py -q
```

## Phase 4: Kernel Facade And Metadata

**Outcome:** Native solver calls pass through one Python facade and report inspectable metadata.

**Files:**
- Create: `src/witwin/channel_native/core/kernels/ops.py`
- Create: `src/witwin/channel_native/core/kernels/autograd.py`
- Create: `src/witwin/channel_native/core/kernels/reference.py`
- Create: `native/channel_native/montecarlo_basic.cpp`
- Create: `native/channel_native/kernels/complex.cuh`
- Create: `native/channel_native/kernels/material_eval.cuh`
- Create: `native/channel_native/kernels/accum.cu`
- Create: `tests/kernels/test_ops_facade.py`
- Create: `tests/kernels/test_metadata_contract.py`

- [ ] **Step 1: Write facade validation tests for CUDA device, dtype, contiguity, and shape errors.**
- [ ] **Step 2: Write metadata schema tests requiring launch counts, RayDN native usage, accumulation strategy, bytes, fusion flags, and AD status.**
- [ ] **Step 3: Run tests and confirm they fail.**
- [ ] **Step 4: Register `torch.ops.cn.noop_metadata` as the first native op.**
- [ ] **Step 5: Add `ops.py` wrappers and metadata validators.**
- [ ] **Step 6: Add native complex and analytic material helper headers.**
- [ ] **Step 7: Add tests proving solvers call native code only through `core.kernels.ops`.**
- [ ] **Step 8: Run acceptance.**

```powershell
conda run -n witwin2 python -m pytest tests/kernels/test_ops_facade.py tests/kernels/test_metadata_contract.py -q
```

## Phase 5: Monte Carlo Basic Primal Solver

**Outcome:** `witwin.channel_native.montecarlo.basic.solve(...)` produces CUDA primal results with metadata.

**Files:**
- Create: `src/witwin/channel_native/montecarlo/basic/config.py`
- Create: `src/witwin/channel_native/montecarlo/basic/result.py`
- Create: `src/witwin/channel_native/montecarlo/basic/metadata.py`
- Create: `src/witwin/channel_native/montecarlo/basic/sampling.py`
- Create: `src/witwin/channel_native/montecarlo/basic/backend.py`
- Create: `src/witwin/channel_native/montecarlo/basic/solver.py`
- Create: `native/channel_native/kernels/los.cu`
- Create: `native/channel_native/kernels/reflection.cu`
- Create: `native/channel_native/kernels/diffraction.cu`
- Create: `native/channel_native/kernels/accum.cu`
- Create: `tests/montecarlo/basic/test_basic_config.py`
- Create: `tests/montecarlo/basic/test_basic_solver_smoke.py`
- Create: `tests/montecarlo/basic/test_basic_solver_metadata.py`

- [ ] **Step 1: Write `Config` tests for sample count, max depth, seed, component mask, accumulation strategy, diagnostics flag, and capability flags.**
- [ ] **Step 2: Write `Result` tests for path gain or radiomap tensors, component powers, optional diagnostics, and metadata.**
- [ ] **Step 3: Write CUDA smoke tests for empty-space LoS.**
- [ ] **Step 4: Write reflection smoke tests gated by RayDN capability.**
- [ ] **Step 5: Write diffraction smoke tests gated by RayDN capability.**
- [ ] **Step 6: Run tests and confirm they fail.**
- [ ] **Step 7: Implement primal LoS contribution path.**
- [ ] **Step 8: Integrate RayDN-backed reflection primitive.**
- [ ] **Step 9: Integrate first-order diffraction when RayDN reports capability; otherwise return explicit capability-disabled metadata.**
- [ ] **Step 10: Implement fused native accumulation into receiver outputs.**
- [ ] **Step 11: Record path counts, valid contribution counts, launch counts, accumulation strategy, RayDN primitive usage, and unsupported AD status in metadata.**
- [ ] **Step 12: Run acceptance.**

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/basic/test_basic_config.py tests/montecarlo/basic/test_basic_solver_smoke.py tests/montecarlo/basic/test_basic_solver_metadata.py -q
```

## Phase 6: Parity, Seed Stability, And Performance Gates

**Outcome:** MC basic has maintained scenes, fixed-seed checks, and benchmark output.

**Files:**
- Create: `tests/support/scenes.py`
- Create: `tests/support/reference_channel.py`
- Create: `tests/montecarlo/basic/test_basic_parity_small_scene.py`
- Create: `tests/montecarlo/basic/test_basic_seed_stability.py`
- Create: `tests/montecarlo/basic/test_basic_performance_gate.py`
- Create: `benchmarks/bench_mc_basic.py`

- [ ] **Step 1: Add an empty-space LoS reference scene.**
- [ ] **Step 2: Add a single-wall reflection reference scene.**
- [ ] **Step 3: Add a wedge or edge diffraction scene behind capability gating.**
- [ ] **Step 4: Add optional reference comparisons against the existing channel package only under `tests/support`.**
- [ ] **Step 5: Add fixed-seed stability tests for path counts and toleranced outputs.**
- [ ] **Step 6: Add benchmark JSON fields: wall time, launch count, intermediate bytes, output bytes, RayDN primitive usage, and accumulation strategy.**
- [ ] **Step 7: Set performance budgets only after recording the first native baseline.**
- [ ] **Step 8: Run acceptance.**

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/basic/test_basic_parity_small_scene.py tests/montecarlo/basic/test_basic_seed_stability.py -q
conda run -n witwin2 python benchmarks/bench_mc_basic.py --scene small --samples 4096 --json
```

## Verification Gates

The rewrite is not complete until all of these are proven from current-state evidence:

- `ext/raydn` contains the current tracked RayDN source snapshot from `E:\Code\RayDTorch`.
- `import witwin.channel_native` does not import DrJit.
- `import witwin.channel_native` does not import Python `raydn`.
- `_channel_native.build_info()` reports `backend="channel-native"` and `uses_dr_jit=false`.
- RayDN native capability is detected through C++/CUDA linkage or reported as unavailable with explicit metadata.
- Public scene construction compiles to separated GeometryStore, MaterialStore, and AssignmentStore tensors.
- Material parameters are not duplicated per face in the authoritative runtime store.
- MC basic LoS smoke test passes on CUDA.
- MC basic reflection smoke test passes on CUDA when RayDN capability is available.
- First-order diffraction is implemented and tested or explicitly capability-disabled.
- Result metadata exposes launch, fusion, accumulation, memory, AD, and RayDN-native dependency facts.
- Fixed-seed tests pass on maintained small scenes.
- Deterministic, path, PSDR, and MC BDPT packages import cleanly and raise explicit not-implemented errors.
- No production solver test calls Python `raydn.Scene`.
- No production solver test uses CPU fallback computation.

## Command Set

Use the `witwin2` environment:

```powershell
conda run -n witwin2 python -m pip install -e . --no-deps
conda run -n witwin2 python -m pytest tests/test_import_contract.py -q
conda run -n witwin2 python -m pytest tests/kernels -q
conda run -n witwin2 python -m pytest tests/core -q
conda run -n witwin2 python -m pytest tests/reserved_api -q
conda run -n witwin2 python -m pytest tests/montecarlo/basic -q
```

Native iteration should prefer incremental CMake builds after the first editable install.

For direct native skeleton checks on Windows, load the Visual Studio compiler
environment and configure with the `witwin2` Python explicitly:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && "C:\Users\Asixa\miniconda3\envs\witwin2\Scripts\cmake.exe" -S . -B artifacts\cmake-witwin2-explicit-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=C:\Users\Asixa\miniconda3\Scripts\ninja.exe -DPython_EXECUTABLE=C:\Users\Asixa\miniconda3\envs\witwin2\python.exe'
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && "C:\Users\Asixa\miniconda3\envs\witwin2\Scripts\cmake.exe" --build artifacts\cmake-witwin2-explicit-release --config Release'
$env:PYTHONPATH = (Resolve-Path 'artifacts\cmake-witwin2-explicit-release').Path
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -c "import _channel_native; print(_channel_native.build_info())"
```

## Phase Ordering

1. Phase 0: package bootstrap and import/build-info contract.
2. Phase 1: RayDN vendored native source linkage.
3. Phase 2: public Scene and separated runtime stores.
4. Phase 3: reserved solver API shells.
5. Phase 4: kernel facade and metadata contract.
6. Phase 5: MC basic primal solver.
7. Phase 6: parity, seed stability, and performance gates.
8. Fixed-topology AD expansion for MC basic.
9. Path solver implementation.
10. Deterministic solver implementation.
11. PSDR research solver implementation.
12. MC BDPT implementation.

The first release should ship only after Phase 6 passes.
