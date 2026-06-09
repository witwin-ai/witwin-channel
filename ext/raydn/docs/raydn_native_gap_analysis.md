# RayDN Native Gap Analysis

Status: RayDN now contains RayD-style OptiX PTX pipelines for scene
intersection, edge query, reflection, and diffraction paths. The current code
also includes Torch VJP/JVP kernels for the supported continuous outputs under
the fixed-winner contract. The current same-script RayD/RayDN benchmark
shows parity or better for the covered grid-64/query-4096 static and dynamic
cases; release-size and Nsight-backed runs should remain the broader
performance gate.

## Current Multipath Implementation

The current code should be treated as a source-port with active release-scale
performance validation remaining:

- `src/torch_ext/common/optix_pipeline.cpp` owns the shared OptiX launch
  pipeline/cache.
- `src/torch_ext/reflection/ops.cpp` owns reflection Torch/PyBind bindings.
- `src/torch_ext/diffraction/ops.cpp` owns diffraction Torch/PyBind bindings.
- `src/torch_ext/reflection/pipeline.cpp` and
  `src/torch_ext/diffraction/pipeline.cpp` own the reflection and diffraction
  PTX entry configurations separately.
- `src/torch_ext/reflection/{visibility_optix.cu,trace_optix.cu,epc_optix.cu,accum_optix.cu,dedup.cu,epc_field.cu}`
  contain the reflection-side native kernels/glue.
- `src/torch_ext/diffraction/{paths_optix.cu,accum_optix.cu,accum_ad.cu}`
  contain the diffraction path search, accumulation, and direct AD kernels.
- `CMakeLists.txt` builds scene, edge, reflection, and diffraction PTX targets.

## RayD Multipath Kernel Coverage

The corresponding RayD source files exist in `E:\Code\RayDi`. Reflection and
diffraction forward kernels now have RayDN source ports, including:

- `E:\Code\RayDi\src\multipath\segment_visibility.cu`
- `E:\Code\RayDi\src\multipath\reflection_trace.cu`
- `E:\Code\RayDi\src\multipath\reflection_accumulation.cu`
- `E:\Code\RayDi\src\multipath\reflection_dedup.cu`
- `E:\Code\RayDi\src\multipath\reflection_epc.cu`
- `E:\Code\RayDi\src\multipath\reflection_epc_field.cu`
- `E:\Code\RayDi\src\multipath\diffraction_paths.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- `E:\Code\RayDi\src\multipath\diffraction_accumulation_ad.cu`

These RayD files should remain the parity source of truth when extending the
current ports.

## Performance Gate Status

Current status:

- Same-script, same-data, same-batch RayD vs RayDN performance comparison
  is implemented in `tests/benchmark_rayd_vs_raydn.py`.
- Current corrected same-script results cover both static-vs-static and
  dynamic-vs-dynamic runs. RayDN is faster for scene build, `intersect`,
  point `nearest_edge`, reflection trace, and direct diffraction accumulation
  on the recorded grid-64/query-4096 benchmark.
- The earlier point `nearest_edge` regression was fixed by removing the
  measured-slower AoS query path, keeping RayD's SoA OptiX query layout, using a
  persistent OptiX params buffer, and adding a Torch no-AD forward path that
  skips autograd tape allocation/writes for non-AD callers.
- Remaining performance work should focus on release-size runs, Nsight
  confirmation of the hot kernels, and any larger workloads that stress
  reflection/diffraction accumulation atomics.

## Required Acceptance Gate

Before this work can be considered complete, RayDN needs:

1. Same-script RayD/RayDN performance runs for the release benchmark shapes.
2. Nsight-backed confirmation or accepted thresholds for the release workloads.
3. Full native and opt-in RayD parity test runs after any performance changes.
