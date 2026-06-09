# RayDN Native Munich MC Radiomap Implementation Plan

**Goal:** Implement `witwin.channel_native` MC radiomap as a full native Torch/CUDA/OptiX backend using vendored `ext/raydn`, with LoS, reflection, diffraction, scene loading, materials, and Munich parity against the original `witwin.channel`.

**Non-negotiable boundary:** `witwin.channel_native` remains the public API. Solver hot paths must not call Python `raydn.Scene`, `raydn.autograd`, or the original `witwin.channel` solver. DrJit must not be imported by `witwin.channel_native`. RayDN is consumed through native extension loading, `torch.classes.raydn.Scene`, `torch.ops.raydn.*`, or narrow C++ bridge functions.

---

## Acceptance Contract

1. `import witwin.channel_native` does not import `drjit`, `mitsuba`, `sionna`, `raydn`, or original `witwin.channel`.
2. `_channel_native.build_info()` reports:
   - `uses_dr_jit=False`
   - `uses_raydn_native=True`
   - `optix_available=True` when the RayDN extension is built and loadable
3. `Scene.load_mitsuba(...munich.xml...)` reproduces the original Munich scene counts:
   - structures: `11`
   - triangles: `38936`
   - diffraction edges with half-plane edge policy: `51650`
4. `Scene.compile()` owns a RayDN native scene handle built from Torch CUDA tensors and does not construct Python `raydn.Scene`.
5. MC basic returns component maps:
   - `components["los"]`
   - `components["reflection"]`
   - `components["diffraction"]`
   - `path_gain == los + reflection + diffraction` under the incoherent power contract used by the original MC basic backend.
6. Munich parity tests compare `channel_native` to original `channel` for identical reduced config:
   - grid size `32`
   - samples `4096`
   - `max_bounces=2`
   - `max_diffraction_order=1`
   - seed `11`
   - frequency `2.4e9`
   - tx `(8.5, 21.0, 27.0)`
   - bounds `((-120, 120), (-120, 140))`
   - plane z `1.5`

## Implementation Stages

### Stage 1: Native RayDN Backend Availability

Modify `channel_native` CMake to build vendored `ext/raydn` alongside `_channel_native`. Install `_raydn` as a private native extension artifact, not as a public Python dependency. Add a loader in `witwin.channel_native.core.kernels.extension` that can load `_raydn` without importing `raydn`.

Required files:
- `CMakeLists.txt`
- `native/channel_native/build_info.cpp`
- `src/witwin/channel_native/core/kernels/extension.py`
- `src/witwin/channel_native/core/kernels/raydn_backend.py`
- `tests/kernels/test_raydn_native_link.py`

### Stage 2: Native Scene Handle Bridge

Build RayDN scene handles from `channel_native.Scene.compile()` using `torch.classes.raydn.Scene`. Store the handle in `CompiledScene.raydn`. The bridge must take per-structure vertices/faces/UV/transform tensors from `channel_native` runtime stores and create a RayDN scene cache directly.

Required files:
- `src/witwin/channel_native/core/runtime/raydn.py`
- `src/witwin/channel_native/core/runtime/compiled_scene.py`
- `src/witwin/channel_native/core/scene.py`
- `tests/core/test_raydn_scene_contract.py`

### Stage 3: Original-Compatible Scene Loading And Materials

Implement `Scene.load_mitsuba()` with the original loader semantics needed for Munich:
- XML path resolution and `source_root`
- merge-shapes behavior
- per-shape material extraction
- default ITU-style material parameters
- surface and structure assignment
- edge policy and diffraction-edge selection compatible with original `channel`

The loader may use Mitsuba/Sionna only inside `Scene.load_mitsuba()`, never during solver hot paths or package import.

Required files:
- `src/witwin/channel_native/core/scene_loader.py`
- `src/witwin/channel_native/core/materials.py`
- `src/witwin/channel_native/core/edge_policy.py`
- `tests/scene/test_munich_loader_parity.py`

### Stage 4: MC LoS, Reflection, Diffraction Component Maps

Replace the current LoS-only MC solver with a full component pipeline:
- LoS: RayDN visibility-aware direct-path power over receiver grid.
- Reflection: call `torch.ops.raydn.reflection_accumulation_forward` with original MC sampling semantics, grid mapping, wavelength, transmitter polarization, active mask, and max bounce depth.
- Diffraction: build first-order diffraction states from selected scene edges, then call `torch.ops.raydn.diffraction_accumulation_forward` with direct/Keller/suffix sample budgets matching original MC basic config.

Required files:
- `src/witwin/channel_native/montecarlo/basic/config.py`
- `src/witwin/channel_native/montecarlo/basic/backend.py`
- `src/witwin/channel_native/montecarlo/basic/solver.py`
- `src/witwin/channel_native/montecarlo/basic/result.py`
- `native/channel_native/montecarlo_basic.cpp` only for narrow glue where dispatcher-only calls are insufficient
- `tests/montecarlo/basic/test_basic_components_contract.py`

### Stage 5: Path Solver Multipath Export

Use the same RayDN native scene bridge for reflection and diffraction path export. LoS remains the current native `_channel_native.path_los_export`; reflection and diffraction stop being capability-disabled once RayDN is linked.

Required files:
- `src/witwin/channel_native/path/solver.py`
- `tests/path/test_munich_path_parity.py`

### Stage 6: Munich Parity And Visual Artifacts

Add reduced Munich parity scripts that emit JSON and PNG artifacts:
- original total radiomap
- native total radiomap
- native components
- native/original dB delta
- path count comparison

Required files:
- `tests/support/bin/benchmark_munich_native_vs_original.py`
- `tests/montecarlo/basic/test_munich_radiomap_parity.py`
- `artifacts/` generated only during runs

## Explicitly Out Of Scope

No Python `raydn` wrapper fallback. No DrJit fallback. No CPU fallback solver. No claim of Munich parity until component maps and parity gates pass.
