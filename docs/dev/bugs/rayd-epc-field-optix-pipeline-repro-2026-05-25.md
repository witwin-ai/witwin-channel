# RayD EPC Field OptiX Pipeline Reproduction

Status: Resolved
Category: Bug
Last reviewed: 2026-05-25

This note records a reproducible `optixPipelineCreate(multipath)` failure in
RayD's native reflection EPC field path after rebuilding and reinstalling RayD
into the `witwin2` conda environment.

## 2026-05-25 Resolution Update

Current installed RayD native extension after the fix:

```text
C:\Users\Asixa\miniconda3\envs\witwin2\Lib\site-packages\rayd\rayd.cp311-win_amd64.pyd
Length: 7261696
LastWriteTime: 2026-05-25 15:13:52
```

Root cause summary:

- The original EPC/diffraction failures were cold `optixPipelineCreate(multipath)` failures caused by primary-only multipath raygen entries that were too broad: multiple `optixTrace()` sites plus large visibility/export/control-flow bodies.
- The later deterministic/path reflection failure had a second trigger: `Scene::trace_reflections()` cold-created its OptiX pipeline after Dr.Jit active-mask sanitization and AD geometry materialization. Downstream Channel AD/deterministic workloads could therefore enter OptiX pipeline linking after Dr.Jit had already materialized live inputs.
- Instruction count and OptiX pipeline statistics were useful diagnostics, but the proven reflection failure was an ordering/cold-pipeline problem, not a Python API change and not bad scene data.

Fix summary:

- Reflection EPC/direct field and diffraction path export use smaller operation-specific native entries where staging is compatible with the operation.
- `Scene::trace_reflections()` keeps the original performance intent: one native OptiX launch for the reflection chain. The fix was to add/use a primary-scene reflection raygen (`__raygen__reflection_trace_primary`) and cold-create the pipeline before `sanitize_reflection_active(...)` and before any AD `drjit::eval(...)`.
- No fallback was introduced. Native-selected paths still raise if their native implementation is broken.
- Public Channel/RayD API callers do not need changes for this fix.

Additional regression coverage added in RayD:

- `tests.test_project_metadata.ProjectMetadataTests.test_trace_reflections_builds_cold_pipeline_before_drjit_materialization` statically guards the `trace_reflections()` source order.
- `tests.drjit.test_geometry.GeometryCoreTests.test_trace_reflections_cold_pipeline_survives_materialized_ad_inputs` runs a fresh subprocess, materializes AD inputs before first reflection trace, and asserts one native `trace_reflections` OptiX launch with gradients preserved.
- `tests.test_project_metadata.ProjectMetadataTests.test_reflection_trace_primary_ptx_entry_stays_single_launch_friendly` guards the embedded PTX: the primary reflection raygen has one `_optix_trace` callsite and the removed trailing-primary entry stays absent.

## Environment

- Channel workspace: `E:\Code\witwin-platform\channel`
- RayD workspace: `E:\Code\RayDi`
- Python environment: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`
- Dr.Jit version observed: `1.3.1`
- RayD Python package path observed: `E:\Code\RayDi\rayd\__init__.py`
- RayD native extension path observed:
  `C:\Users\Asixa\miniconda3\envs\witwin2\Lib\site-packages\rayd\rayd.cp311-win_amd64.pyd`

## Rebuild and Install RayD

From `E:\Code\RayDi`, rebuild and install RayD into `witwin2`:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install --no-build-isolation --force-reinstall --no-deps -ve . -Cbuild-dir=build/cp311-cp311-win_amd64-clean-codex
```

The clean build performed on 2026-05-25 regenerated the CUDA/C++ objects,
including `reflection_epc_field.obj`, and installed:

```text
C:\Users\Asixa\miniconda3\envs\witwin2\Lib\site-packages\rayd\rayd.cp311-win_amd64.pyd
LastWriteTime: 2026-05-25 05:52:25
Length: 6800384
```

Confirm the loaded native extension:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -c "import importlib.util, pathlib, rayd; spec=importlib.util.find_spec('rayd.rayd'); p=pathlib.Path(spec.origin); print('rayd package', rayd.__file__); print('native origin', spec.origin); print('native size', p.stat().st_size)"
```

Expected confirmation from the reproduced environment:

```text
rayd package E:\Code\RayDi\rayd\__init__.py
native origin C:\Users\Asixa\miniconda3\envs\witwin2\Lib\site-packages\rayd\rayd.cp311-win_amd64.pyd
native size 6800384
```

## RayD-Native Reproduction

Run this from `E:\Code\witwin-platform\channel` or any directory that can import
the installed `rayd` package:

```powershell
@'
import math
import drjit as dr
import drjit.cuda as cuda
import rayd as pj

n = 8
reflector = pj.Mesh(
    cuda.Array3f(
        [-4.0, 4.0, -4.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, -1.0, 1.0, 1.0],
    ),
    cuda.Array3i([0, 0], [1, 3], [3, 2]),
)
scene = pj.Scene()
scene.add_mesh(reflector)
scene.build()

tx = cuda.Array3f([0.0 + 0.02 * i for i in range(n)], [-2.0] * n, [0.0] * n)
rx = cuda.Array3f([6.0 + 0.02 * i for i in range(n)], [-2.0] * n, [0.0] * n)

options = pj.ReflEpcFieldOptions()
options.expected_prim_ids = cuda.Int([0] * n)
options.surface_group_id = cuda.Int([0, 0])
options.surface_group_size = cuda.Int([2])
options.surface_group_members = cuda.Int([0, 1])
options.surface_max_group_size = 2
options.visibility_ignore_mode = "surface_group"
options.slot_plane_point = cuda.Array3f([-4.0] * n, [0.0] * n, [-1.0] * n)
options.slot_plane_normal = cuda.Array3f([0.0] * n, [-1.0] * n, [0.0] * n)
options.slot_eta_r = cuda.Float([4.0] * n)
options.slot_mu_r = cuda.Float([1.0] * n)
options.slot_sigma = cuda.Float([0.0] * n)
options.slot_gain = cuda.Float([1.0] * n)
options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
options.omega = 2.0 * math.pi * 3.5e9
options.wavelength = 299792458.0 / 3.5e9
options.return_geom = False
options.return_endpoints = False

result = scene.trace_refl_epc_field(
    tx,
    rx,
    max_bounces=1,
    options=options,
    active=cuda.Bool([True] * n),
)
dr.eval(result.valid, result.field_x_re, result.field_x_im)
print("PASS")
'@ | C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -
```

Observed failure:

```text
jit_optix_log(): [COMPILER] COMPILE ERROR: failed to create pipeline
RuntimeError: OptiX error in optixPipelineCreate(multipath)
```

## RayD Width Matrix

This matrix distinguishes small-width success from batched-width failures:

```powershell
@'
import subprocess
import sys

case = r'''
import math
import drjit as dr
import drjit.cuda as cuda
import rayd as pj

n = int(__import__("sys").argv[1])
reflector = pj.Mesh(
    cuda.Array3f([-4.0, 4.0, -4.0, 4.0], [0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 1.0, 1.0]),
    cuda.Array3i([0, 0], [1, 3], [3, 2]),
)
scene = pj.Scene()
scene.add_mesh(reflector)
scene.build()

tx = cuda.Array3f([0.0 + 0.02 * i for i in range(n)], [-2.0] * n, [0.0] * n)
rx = cuda.Array3f([6.0 + 0.02 * i for i in range(n)], [-2.0] * n, [0.0] * n)

options = pj.ReflEpcFieldOptions()
options.expected_prim_ids = cuda.Int([0] * n)
options.surface_group_id = cuda.Int([0, 0])
options.surface_group_size = cuda.Int([2])
options.surface_group_members = cuda.Int([0, 1])
options.surface_max_group_size = 2
options.visibility_ignore_mode = "surface_group"
options.slot_plane_point = cuda.Array3f([-4.0] * n, [0.0] * n, [-1.0] * n)
options.slot_plane_normal = cuda.Array3f([0.0] * n, [-1.0] * n, [0.0] * n)
options.slot_eta_r = cuda.Float([4.0] * n)
options.slot_mu_r = cuda.Float([1.0] * n)
options.slot_sigma = cuda.Float([0.0] * n)
options.slot_gain = cuda.Float([1.0] * n)
options.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
options.omega = 2.0 * math.pi * 3.5e9
options.wavelength = 299792458.0 / 3.5e9
options.return_geom = False
options.return_endpoints = False

result = scene.trace_refl_epc_field(
    tx,
    rx,
    max_bounces=1,
    options=options,
    active=cuda.Bool([True] * n),
)
dr.eval(result.valid, result.field_x_re, result.field_x_im)
print("PASS")
'''

for n in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32):
    proc = subprocess.run(
        [sys.executable, "-c", case, str(n)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if proc.returncode == 0:
        print(f"n={n}: PASS")
    else:
        combined = (proc.stderr + proc.stdout).splitlines()
        marker = next(
            (
                line
                for line in combined
                if "optixPipelineCreate" in line
                or "COMPILE ERROR" in line
                or "Traceback" in line
            ),
            combined[-1] if combined else "no output",
        )
        print(f"n={n}: FAIL {marker}")
'@ | C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -
```

Observed pattern after the clean rebuild:

```text
n=1: PASS
n=2: PASS
n=4: PASS
n=8: FAIL jit_optix_log(): [COMPILER] COMPILE ERROR: failed to create pipeline
n=12: FAIL jit_optix_log(): [COMPILER] COMPILE ERROR: failed to create pipeline
n=16: FAIL jit_optix_log(): [COMPILER] COMPILE ERROR: failed to create pipeline
```

The exact pass/fail set can vary by optional output flags, but `n=8` is a
stable minimal failing width in the reproduced environment.

## Channel Path Solver Reproduction

From `E:\Code\witwin-platform\channel`:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\path\test_example_path_solver_minimal.py -q --gpu
```

Observed failure:

```text
RuntimeError: OptiX error in optixPipelineCreate(multipath)
```

The failing Channel call stack reaches:

```text
witwin\channel\path\solver.py
witwin\channel\deterministic\trace\path_export.py
witwin\channel\deterministic\reflection\epc.py:1601
scene._rayd_scene.trace_refl_epc_field(...)
```

## Deterministic Solver Control Check

The deterministic RayD EPC and deterministic radiomap smoke tests passed in the
same environment:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\deterministic\test_reflection_rayd_epc_backend.py tests\deterministic\test_example_deterministic_radiomap_three_cubes.py -q --gpu
```

Observed result:

```text
10 passed
```

This narrows the active failure to the RayD native batched
`trace_refl_epc_field(...)` path used by the path solver's default native
reflection backend.

