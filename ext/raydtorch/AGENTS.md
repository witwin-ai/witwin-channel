# RayDTorch Agent Rules

## Native CUDA/OptiX Incremental Build

- Do not use `python -m pip install --no-build-isolation -e .` for every debug iteration. That command creates or refreshes a full editable build/install flow and is too slow for CUDA/OptiX work.
- The persistent native build directory is `artifacts/skbuild`. It is ignored by git and should be reused across edits.
- If `artifacts/skbuild` does not exist, initialize it once:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install --no-build-isolation -e . -Cbuild-dir=artifacts/skbuild
```

- After native `.cpp`, `.cu`, `.h`, CMake, or PTX embedding changes, use the incremental helper:

```powershell
powershell -ExecutionPolicy Bypass -File E:\Code\RayDTorch\scripts\dev_build_native.ps1
```

- The helper runs `cmake --build artifacts/skbuild --config Release --target _raydtorch` and copies the resulting `_raydtorch*.pyd` to the conda site-packages path that the editable import hook actually loads.
- Run focused tests with the environment Python directly, for example:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydtorch_native -v
```

- Use full `pip install -e .` again only when intentionally regenerating the editable install metadata, changing packaging behavior, or recreating the persistent build directory from scratch.

## Native Numeric And Performance Acceptance

Use the current worktree and command output as authoritative. Do not use a full
editable reinstall for normal CUDA/OptiX iteration; use the incremental helper
above and then run focused numeric/performance tests.

Run the CUDA tests with the `witwin2` environment Python:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_edge_queries -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath tests.raydtorch_native.test_scene_cache -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydtorch_native -v
```

Latest recorded native test results, after the nearest-edge no-AD fast path and
RayD edge topology/cache updates:

- `tests.raydtorch_native.test_edge_queries -v`: 9 tests passed.
- `unittest discover tests.raydtorch_native -v`: 61 tests passed, 12 skipped.

Run external RayD parity explicitly; the normal discover run skips these tests:

```powershell
$env:RAYDTORCH_RUN_DR_JIT_PARITY='1'
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_drjit_parity -v
```

Latest recorded opt-in RayD parity result:

- 12 tests passed.
- Covered forward parity: scene intersection, multi-mesh global ids, point
  nearest edge, visibility, reflection tracing, diffraction paths, direct
  diffraction accumulation, Keller accumulation, suffix accumulation,
  order-2/order-3 chain accumulation, and coherent direct accumulation.
- The run may print `jitc_llvm_init(): LLVM API initialization failed ..`;
  this warning appeared in the passing run and did not invalidate the parity
  assertions.

Run same-script RayD vs RayDTorch performance comparison:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30 --dynamic
```

Latest recorded same-script static-vs-static performance result, stable repeat
run:

```json
{
  "dynamic": false,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2342.0363000041107,
    "diffraction_direct_ms": 0.4519966666218049,
    "intersect_ms": 0.14462333335056124,
    "nearest_edge_ms": 1.4299183333302306,
    "reflection_trace_ms": 0.34219333332051366
  },
  "raydtorch": {
    "build_ms": 1547.1698999972432,
    "diffraction_direct_ms": 0.43191333328043885,
    "intersect_ms": 0.10184333332290407,
    "nearest_edge_ms": 1.4051099999051075,
    "reflection_trace_ms": 0.3025700001065464
  },
  "repeat": 60,
  "warmup": 8
}
```

Latest recorded same-script dynamic-vs-dynamic performance result:

```json
{
  "dynamic": true,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2337.4531000008574,
    "diffraction_direct_ms": 0.9759666667378042,
    "intersect_ms": 0.12905000015355958,
    "nearest_edge_ms": 1.5716533331821363,
    "reflection_trace_ms": 0.32191333327015553
  },
  "raydtorch": {
    "build_ms": 1547.6975999990827,
    "diffraction_direct_ms": 0.42821333336178213,
    "intersect_ms": 0.11110666673630476,
    "nearest_edge_ms": 1.4978466667040873,
    "reflection_trace_ms": 0.3103666667205592
  },
  "repeat": 30,
  "warmup": 5
}
```

Latest isolated RayDTorch reflection trace microbenchmark on the same grid:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current acceptance interpretation for the covered benchmark shape:

- Numeric parity is currently demonstrated for the covered forward cases and
  fixed-winner Torch VJP/JVP tests.
- RayDTorch is faster than RayD in the latest same-script static run for scene
  build, `intersect`, point `nearest_edge`, reflection trace, and direct
  diffraction accumulation.
- RayDTorch is faster than RayD in the latest same-script dynamic run for scene
  build, `intersect`, point `nearest_edge`, reflection trace, and direct
  diffraction accumulation.
- The nearest-edge regression was fixed by keeping RayD's SoA OptiX query path,
  using a persistent params buffer, and adding a Torch no-AD path that avoids
  autograd tape allocation when neither reverse-mode nor forward-mode AD is
  required.
- Keep running release-size and Nsight-backed benchmarks before claiming broad
  performance superiority across all multipath/diffraction workloads.
