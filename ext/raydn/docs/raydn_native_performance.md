# RayDN Native Performance

Measured on Windows with NVIDIA GeForce RTX 5080 and Torch CUDA 12.8.

Command:

```powershell
conda run -n witwin2 python -m tests.benchmark_raydn_native --grid 192 --queries 65536
```

Current RayDN-native result:

```json
{
  "build_ms": 1550.21,
  "dynamic_sync_ms": 2.29,
  "grid": 192,
  "intersect_ms": 0.241,
  "nearest_edge_ms": 0.423,
  "queries": 65536
}
```

The benchmark covers native scene build, dynamic vertex sync, OptiX triangle intersection, and nearest-edge query throughput. Query timings are per benchmark iteration over 65,536 inputs.

## RayD Comparison Status

Same-script benchmark command:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydn --grid 64 --queries 4096 --warmup 5 --repeat 30
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydn --grid 64 --queries 4096 --warmup 5 --repeat 30 --dynamic
```

The command above is a fast RayD/RayDN multipath regression shape. It is
not the only acceptance shape: it casts only 4,096 rays. For RayD latest-style
intersection pressure and Mitsuba comparison, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_stress `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --scenario rayd-latest:64:128 `
  --scenario release:192:256 `
  --repeats 5 --warmup 2 --mitsuba-preliminary
```

This stress script matches RayD's `mesh_resolution/ray_grid_side` convention.
`rayd-latest:64:128` casts 16,384 rays, while `release:192:256` casts 65,536
rays. It reports RayDN, RayD, and Mitsuba static/dynamic intersection
performance for full materialized fields and reduced t-only paths; Mitsuba
`ray_intersect_preliminary` is reported as an extra Mitsuba-only lower-level
baseline when `--mitsuba-preliminary` is set.

For scaling sweeps instead of a few fixed sizes, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_sweep `
  --preset standard `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary
```

The sweep emits `sweep.json`, `sweep.csv`, and PNG grouped-bar plots under
`artifacts/benchmarks/scaling/<preset>/`. Presets:

- `smoke`: quick script/plot validation.
- `standard`: up to 768 mesh resolution, about 1.18M triangles, and 1M requested rays.
- `large`: up to 1024 mesh resolution, about 2.10M triangles, and 10M requested rays.
- `extreme`: includes 100,663,296 requested rays.

Large ray counts are represented by a fixed ray batch plus a batch count. By
default the script measures per-batch throughput and projects total time for the
requested ray count. Add `--execute-total-rays` when the goal is to actually run
all batches for the 10M/100M ray entries.

Current scaling interpretation:

- In `artifacts/benchmarks/scaling/codex_current_large_forward_r30/sweep.json`,
  RayDN is faster than RayD on the large static full intersection point.
  At 2.10M triangles and a 1,048,576-ray batch projected to 100.66M requested
  rays, static full is RayDN 0.3760 ms/batch vs RayD 0.6258 ms/batch.
- In the same large static t-only/reduced case, RayDN is also faster:
  RayDN 0.1478 ms/batch vs RayD 0.2136 ms/batch.
- Dynamic RayD/RayDN scene update timings in the same run also favor
  RayDN for this shape: full 1.4799 ms/batch vs RayD 22.8009 ms/batch,
  reduced 1.2745 ms/batch vs RayD 22.7541 ms/batch.
- Dynamic Mitsuba numbers are recorded, but they are not the primary comparison:
  Mitsuba dynamic scene updates perform additional work not directly comparable
  to RayD/RayDN.

RayD latest-style multipath path export is covered separately:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_multipath `
  --preset smoke --rayd-source local --rayd-root E:\Code\RayDi
```

This benchmark adds RayDN to RayD's path-level Mitsuba comparison for:

- `reflection_trace`: parallel reflectors, public reduced path fields.
- `diffraction_export`: synthetic single-edge diffraction path export.

It writes all outputs under one folder, for example
`artifacts/benchmarks/multipath/smoke_all/`, with `multipath.json`,
`multipath.csv`, and `time_ms_multipath.png`. The plot is a grouped bar chart of
absolute average time in ms only; no SVG or throughput plot is emitted.

Latest multipath result with RayD warm-started in the same script,
`--preset standard --backends raydn rayd_path --no-plots`. The
1,048,576-ray / 4-bounce reflection row is from the focused verification
`--workloads reflection_trace --ray-count 1048576 --max-bounces 4 --repeats 20
--warmup 5`, run after the standard five-sample pass showed a single outlier.

| Workload | Size | RayDN ms | RayD ms | Speedup | Status |
|---|---:|---:|---:|---:|---|
| reflection trace, 2 bounces | 65,536 rays | 0.0971 | 0.1348 | 1.39x | RayDN faster |
| reflection trace, 4 bounces | 65,536 rays | 0.1138 | 0.2259 | 1.99x | RayDN faster |
| reflection trace, 2 bounces | 1,048,576 rays | 0.2683 | 0.7791 | 2.90x | RayDN faster |
| reflection trace, 4 bounces | 1,048,576 rays | 0.4241 | 0.7000 | 1.65x | RayDN faster |
| diffraction export | 65,536 states | 0.3065 | 0.4769 | 1.56x | RayDN faster |
| diffraction export | 1,048,576 states | 0.6312 | 0.8531 | 1.35x | RayDN faster |

All listed path-length checksums matched RayD.

AD backward is measured with `--include-backward`; plots are absolute projected
time grouped by backend, not throughput or speedup plots:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -B -m tests.benchmark_raydn_rayd_mitsuba_sweep `
  --preset smoke --mesh-resolution 64 --mesh-resolution 128 --mesh-resolution 256 `
  --total-rays 16384 --total-rays 65536 --ray-batch-side 256 `
  --repeats 5 --warmup 3 --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary --include-backward `
  --output-dir artifacts\benchmarks\scaling\ad_uv_tape
```

RayDN static AD/VJP is now measured through the public API:
`scene.intersect(...).t.backward(upstream)`, with nonuniform upstream gradients.
The old `Scene.intersect_t_sum_vjp` / `t_sum_*` scalar-loss benchmark interface
has been removed and is guarded by `tests.raydn_native.test_public_api_contract`.
Full-output `Scene.intersect` VJP/JVP now also routes missing upstream gradients
and missing tangents through native optional kernels instead of Python
`torch.zeros_like`; explicit upstream gradients and tangents are read in the CUDA
kernel with tensor strides, so non-contiguous or expanded gradients do not need a
C++ ATen `.contiguous()` staging copy. This preserves generic upstream-gradient
semantics and does not assume a sum loss.
Point `Scene.nearest_edge`, `Scene.trace_reflections`, and
`Scene.trace_refl_epc_field` AD inputs use the same stride-aware nullable-input
pattern for their upstream gradients and JVP tangents. Custom autograd nodes now
also disable unused-output gradient materialization, so `its.t.backward(upstream)`
does not receive zero tensors for unused full intersection fields.

Current public AD/VJP status, 73.7K triangles / 65.5K rays, 30 repeats,
8 warmup, RayD package warm-started in the same script, full VJP explicitly
materialized with `--materialize-full-vjp`:

| Mode | RayDN ms | RayD ms | Status |
|---|---:|---:|---|
| static forward full | 0.0565 | 0.1402 | RayDN faster |
| static forward reduced | 0.0294 | 0.1298 | RayDN faster |
| static public VJP full | 0.2484 | 0.0916 | RayD faster |
| static public VJP reduced | 0.2074 | 0.0739 | RayD faster |
| dynamic public VJP full | 0.3795 | 2.5689 | RayDN faster |
| dynamic public VJP reduced | 0.3035 | 2.3201 | RayDN faster |

The native reduced VJP kernel itself is not the bottleneck: direct native
`intersect_forward_tape + intersect_backward_t` measures about 0.068 ms on the
4,096-ray and 65,536-ray stress shapes. The remaining public static AD gap is
PyTorch eager autograd overhead: graph construction, tensor metadata allocation,
and `AccumulateGrad` around `.backward()`. CUDA Graph capture works for the
direct native forward and backward calls individually, but capture of the public
`.backward()` path currently fails with CUDA stream-capture invalidation. Closing
this gap fairly requires reducing the public autograd boundary count or using an
internal cached VJP executor without recognizing the loss or benchmark shape.
`torch.compile` is not expected to fuse through the OptiX/custom-extension/
autograd boundaries reliably; the reliable optimization path is semantic native
fusion while preserving the public API and generic upstream-gradient contract.

## Public API And No-Fallback Contract

The public RayDN surface intentionally remains the original user-facing API:
`scene.intersect(...)`, `scene.trace_reflections(...)`, and the multipath /
diffraction methods on `Scene`. Benchmark-specific public names such as
`trace_reflections_minimal`, `intersect_t_sum`, and `intersect_t_sum_vjp` are not
part of the API and are guarded by `tests.raydn_native.test_public_api_contract`.
Reduced/full output timing is selected through the normal operation semantics
(`RayFlags.None` vs full flags, AD state, and whether full public fields are
materialized in the benchmark), not through a separate Python method.

The production public wrappers do not have a Python fallback that reconstructs
results from PyTorch eager tensor operations. They validate arguments and dispatch
to native pybind/CUDA/OptiX entrypoints. The previous chain-diffraction AD
fallback/staging helpers (`_contig_states`, `_contig_material`, Python
`.contiguous()` slicing) have been removed. If a path is not implemented natively,
it must fail explicitly rather than silently assembling GPU tensors with
`torch.stack`, `torch.cat`, `torch.where`, `torch.sum`, `reshape` repair,
valid-mask filtering, or checksum kernels in Python.

Backward paths are generic VJP/JVP paths. Intersection forward saves the hit tape
needed for the requested outputs, and backward consumes real upstream gradients
such as `grad_t`, `grad_p`, `grad_n`, `grad_geo_n`, `grad_uv`, and
`grad_barycentric`; a sum loss is only the special case where `grad_t` is all
ones. Reflection, EPC, and diffraction AD paths follow the same rule: missing
upstreams/tangents are nullable native inputs interpreted as zero in CUDA, while
explicit nonuniform or strided upstream tensors are read with their strides.

Diffraction direct and chain accumulation now pass logical state counts into
native no-AD and AD bindings. Padded, non-contiguous `DfrStates`/material views
therefore use the same native path as contiguous benchmark inputs without Python
prefix slicing. Direct and chain AD bindings also avoid C++ ATen
`split_vec3(...).contiguous()`, `flatten_optional_f32(...)`, and
`stack_vec3(...)` staging around state vectors, upstream grid gradients, and
returned vector gradients.

The stress benchmark also has an opt-in PyTorch-loss wrapper mode:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_stress `
  --scenario ad_uv_tape_256_65k:192:256 --backends raydn rayd mitsuba `
  --repeats 30 --warmup 8 --rayd-source package --include-backward `
  --torch-loss-backward --materialize-full-vjp --require-mitsuba
```

This exposes feasible external backends through `torch.autograd` and measures a
Torch `loss.backward()` wrapper. RayD dynamic scenes and Mitsuba static/dynamic
scenes can be wrapped. RayD warm-started static scenes cannot be fairly wrapped
with the current RayD public API because the differentiable vertex array is bound
at scene build; reconnecting a later Torch vertex tensor would require rebuilding
the static scene or changing RayD itself. The latest strict materialized run is
recorded in
`artifacts/benchmarks/scaling/ad_uv_tape/no_fallback_torch_loss_materialized_current.json`.

| Torch-loss mode | RayDN ms | RayD ms | Mitsuba ms | Status |
|---|---:|---:|---:|---|
| dynamic full | 0.3559 | 3.0661 | 2.2322 | RayDN faster |
| dynamic reduced | 0.3411 | 3.2010 | 2.2117 | RayDN faster |
| static full | 0.3340 | unsupported | 2.2374 | RayDN faster than Mitsuba |
| static reduced | 0.2798 | unsupported | 2.2802 | RayDN faster than Mitsuba |

Latest RayD-warm-started `rayd-latest:64:128` AD spot check after native
optional-gradient migration and disabled unused-output gradient materialization,
16,384 rays, 30 repeats, 8 warmup:

| Mode | RayDN ms | RayD ms | Status |
|---|---:|---:|---|
| static public VJP full | 0.1621 | 0.0879 | RayD faster |
| static public VJP reduced | 0.1371 | 0.0735 | RayD faster |
| dynamic public VJP full | 0.2901 | 1.6450 | RayDN faster |
| dynamic public VJP reduced | 0.2619 | 1.4679 | RayDN faster |

A lazy `Intersection` materialization experiment reduced the static full VJP
number when the caller only consumed `.t`, but it was not retained: it would make
the "full VJP" benchmark no longer materialize full public intersection fields
before the VJP, which is an ambiguous and potentially unfair comparison.

Current same-script static-vs-static result after native validity, camera
stride, diffraction active-mask/path-export/accumulation input-stride migration,
direct diffraction AD strided binding migration, and RayD warmup in the same
harness. This run used `--grid 64 --queries 4096 --warmup 8 --repeat 60` with
the installed RayD package:

```json
{
  "dynamic": false,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2385.078599996632,
    "diffraction_direct_ms": 0.8474633332904583,
    "diffraction_paths_ms": 0.5784999998771431,
    "intersect_flags_none_ms": 0.26136499994512025,
    "intersect_ms": 0.21221833352077132,
    "nearest_edge_ms": 1.7859500000971213,
    "reflection_trace_ms": 0.4695400001461773
  },
  "raydn": {
    "build_ms": 1551.944499995443,
    "diffraction_direct_ms": 0.24948999998741783,
    "diffraction_paths_ms": 0.10571333332336508,
    "intersect_flags_none_ms": 0.038123333312493436,
    "intersect_ms": 0.04897000011017857,
    "nearest_edge_ms": 1.4218099999804206,
    "reflection_trace_ms": 0.03780333330117477
  },
  "repeat": 60,
  "warmup": 8
}
```

Current same-script dynamic-vs-dynamic result, `--grid 64 --queries 4096
--warmup 5 --repeat 30 --dynamic`:

```json
{
  "dynamic": true,
  "grid": 64,
  "queries": 4096,
  "rayd": {
    "build_ms": 2441.7838000081247,
    "diffraction_direct_ms": 0.47562999970978126,
    "diffraction_paths_ms": 0.4139266665636872,
    "intersect_flags_none_ms": 0.16480999959943196,
    "intersect_ms": 0.17362000022937232,
    "nearest_edge_ms": 1.8150333334536601,
    "reflection_trace_ms": 0.5681900001945905
  },
  "raydn": {
    "build_ms": 1545.6587000080617,
    "diffraction_direct_ms": 0.30103666649665684,
    "diffraction_paths_ms": 0.06877333313847582,
    "intersect_flags_none_ms": 0.029380000099384535,
    "intersect_ms": 0.058646666972587504,
    "nearest_edge_ms": 1.3477100001182407,
    "reflection_trace_ms": 0.057016666687559336
  },
  "repeat": 30,
  "warmup": 5
}
```

Post-native-validity/camera/diffraction-stride same-script check, `grid=64`,
`queries=4096`, with RayD kept warm-started inside the same harness:

| Mode | RayD ms | RayDN ms | Status |
|---|---:|---:|---|
| static build | 2385.079 | 1551.944 | RayDN faster |
| static intersect `RayFlags.None` | 0.2614 | 0.0381 | RayDN faster |
| static intersect `RayFlags.All` | 0.2122 | 0.0490 | RayDN faster |
| static nearest edge | 1.7860 | 1.4218 | RayDN faster |
| static reflection trace | 0.4695 | 0.0378 | RayDN faster |
| static diffraction direct | 0.8475 | 0.2495 | RayDN faster |
| static diffraction paths | 0.5785 | 0.1057 | RayDN faster |
| dynamic build | 2441.784 | 1545.659 | RayDN faster |
| dynamic intersect `RayFlags.None` | 0.1648 | 0.0294 | RayDN faster |
| dynamic intersect `RayFlags.All` | 0.1736 | 0.0586 | RayDN faster |
| dynamic nearest edge | 1.8150 | 1.3477 | RayDN faster |
| dynamic reflection trace | 0.5682 | 0.0570 | RayDN faster |
| dynamic diffraction direct | 0.4756 | 0.3010 | RayDN faster |
| dynamic diffraction paths | 0.4139 | 0.0688 | RayDN faster |

An isolated RayDN-only microbenchmark of the no-AD reflection trace path on
the same grid measured:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current interpretation for this benchmark shape:

- RayDN `intersect` is faster in the latest static and dynamic same-script
  runs.
- RayDN large static full and reduced intersection are faster in the latest
  RayD/RayDN sweep run.
- RayDN public static AD `.backward()` is still slower than RayD's warmed
  static JIT VJP in the latest stress run. No-active vertices-only t-VJP now
  skips the empty active sentinel and saves a smaller autograd context, and
  default empty `tmax` is normalized away in that internal request. Direct native
  `intersect_forward + intersect_backward_t` remains faster than RayD; PyTorch
  eager autograd fixed overhead dominates the public `.backward()` timing.
- Acceptance benchmarks and public contract tests do not use benchmark-specific
  RayDN shortcuts such as `_minimal` or `t_sum_*`. Static VJP timings use
  public `.backward(upstream)` with nonuniform upstream gradients.
- `Intersection.is_valid()` now calls a native CUDA validity kernel for both
  full-output `shape_id >= 0` and reduced `isfinite(t)` semantics. `Camera`
  public methods and AD kernels consume strided input/upstream tensors directly,
  diffraction path/accumulation active masks are interpreted by native
  width/stride parameters instead of Python or C++ expanding them to full width,
  and `Scene.trace_dfr_paths(...)` now passes source/receiver endpoints plus
  `DfrStates` and material tensors directly to a stride-aware native OptiX path
  export binding rather than staging contiguous Python slices.
- Multi-mesh `Scene.intersect`, point `Scene.nearest_edge`,
  `Scene.trace_reflections`, and `Scene.trace_refl_epc_field` no longer build
  global AD vertices with Python `torch.cat`. They keep each mesh vertex tensor
  as an autograd input and use native scene-cache helpers to pack JVP tangents
  and split global VJP gradients back to mesh leaves.
- `Scene.intersect` full-output backward/JVP and point `Scene.nearest_edge`
  backward/JVP now treat absent upstreams/tangents as zero in native code, not
  with Python `torch.zeros_like`; nearest-edge JVP also receives hidden
  `tape_s/tape_d` zero tangents from the same native JVP kernel. Tests cover
  explicit non-sum upstreams and patch Python zero-fill helpers around these
  paths. `Scene.intersect` JVP now also applies `RayFlags` in native code, so
  inactive public tangent fields are returned as native empty tensors and the
  kernel does not write disabled outputs instead of Python creating
  `Tensor.new_empty` sentinels. Forward-AD `active=None` is passed through the
  public wrapper as nullable input, with the saved empty active context returned
  by the native binding as a hidden output.
- `Scene.trace_reflections` backward/JVP now also uses optional native
  entrypoints for absent upstreams/tangents, so the Python public wrapper no
  longer creates empty grad sentinels or zero tangent tensors. The AD forward
  path also receives the saved empty active-mask context from native code when
  `active=None`, rather than creating a Python CUDA sentinel. The reflection-chain
  AD implementation uses fused native CUDA kernels for the bounce chain and does
  not use the old C++ ATen `select/sum/where` bounce loop.
- `Scene.trace_refl_epc_field` backward/JVP now uses fused native CUDA kernels
  for nullable upstreams/tangents. The kernels consume real nonuniform upstream
  gradients, compute field/t VJP or JVP internally, and avoid Python
  `torch.zeros_like` plus the previous C++ ATen `sin/cos/reshape/mul` chain.
  AD forward also saves omitted active masks through a native hidden output. EPC
  forward now fuses source/receiver AoS-to-SoA setup, `receiver - source`, ray
  `tmax`, temp initialization, constant material/polarization defaults, optional
  y/z field writes, and first primitive-id extraction into native CUDA kernels
  instead of C++ ATen `zeros/full/ones/sub/sum/sqrt/select/reshape` fan-out.
- `Scene.visible` now sends AoS endpoint tensors directly into the segment
  visibility OptiX raygen; the raygen writes `visible`, first-blocking primitive,
  and the legacy `tape_t=inf` output. The Python wrapper no longer calls
  endpoint `.contiguous()`, and the native op no longer performs endpoint
  `select().contiguous()` SoA splits or `at::full` output initialization for the
  public segment-visibility path.
- `Scene.accum_dfr_direct(...)` and `Scene.accum_dfr(...)` backward/JVP now pass
  absent output upstreams and absent state/material tangents as nullable native
  inputs; the CUDA AD kernels treat null pointers as zero. Direct and chain
  diffraction JVP also receive their public zero field tangents from native code
  rather than Python `torch.zeros_like`. Tests cover explicit non-sum upstream
  matrices and patch Python `torch.zeros`/`torch.zeros_like` around these calls.
- `Scene.accum_dfr_direct(...)` forward now passes unused recursive inputs as
  nullable native arguments, so direct no-AD and AD forward no longer create
  Python-side empty CUDA sentinels for the order-1 path.
- `Scene.accum_dfr_direct(...)` AD backward/JVP now passes the logical
  `states.count` into native direct AD bindings and reads state vec3/scalar
  fields, material gain/valid masks, tangents, gradient outputs, and gradient
  write buffers with tensor strides. The direct AD binding no longer uses
  `split_vec3(...).contiguous()`, `flatten_optional_f32(...)`, or
  `stack_vec3(...)` staging. Tests compare strided/padded states and material
  views against contiguous logical-prefix inputs with nonuniform upstream
  `grad_outputs`, so this is not a `sum()`-loss-only optimization.
- `Scene.accum_dfr(...)` chain AD now passes logical initial and recursive state
  counts into native chain AD bindings and reads initial/recursive states,
  material gain/valid masks, tangents, upstream grid gradients, and returned
  vector-gradient buffers with tensor strides. The chain AD binding no longer
  uses Python `_contig_states` / `_contig_material` staging or C++ ATen
  `split_vec3(...).contiguous()`, `flatten_optional_f32(...)`, or
  `stack_vec3(...)` around the public AD path.
- `Scene.trace_dfr_paths(...)` and `Scene.accum_dfr_coherent_direct(...)` also
  pass omitted active masks as nullable native inputs, avoiding Python-side
  empty active sentinels in these public multipath calls. The path-export wrapper
  is covered by strided endpoint/state/material tests with `states.count` smaller
  than the physical tensor length, so this optimization is not tied to the
  contiguous benchmark layout.
- `Scene.accum_dfr_direct(...)` and `Scene.accum_dfr_coherent_direct(...)`
  no-AD forward paths now pass a logical state limit and read state/material
  scalar fields plus vec3 state fields with tensor strides in the native
  OptiX/CUDA path. The no-AD `diffraction_accumulation_forward_op` and
  `diffraction_coherent_accumulation_forward_op` bindings no longer use C++ ATen
  `split_vec3(...).contiguous()` staging for state vectors.
- `Scene.accum_dfr(...)` chain no-AD now uses the same stride-aware native
  accumulation forward binding for initial and recursive states, including a
  recursive logical state limit. Tests patch Python `Tensor.contiguous` around
  the strided chain call, so this coverage is not satisfied by hidden Python
  staging.
- RayDN multipath reflection trace and diffraction export are faster in the
  latest RayD path-export benchmark.
- RayDN `diffraction_direct` is faster in the latest static and dynamic
  same-script runs.
- RayDN point `nearest_edge` is faster in the latest static and dynamic
  same-script runs after the no-AD Torch path removed unnecessary tape
  allocation/writes for non-AD callers and the point/ray query SoA setup moved
  from ATen `select().contiguous()` calls to hand-written native split kernels.
  A direct-AoS OptiX experiment was not retained after memcheck exposed an
  illegal read on a large-grid high-z query; the native split path passed that
  same memcheck case.
- RayDN scene build is faster in the latest same-script package-RayD run
  shown above, but build should remain an open item rather than a universal
  superiority claim: build timing is sensitive to RayD source selection,
  pipeline/cache state, OptiX compaction, and first-use initialization.
- RayDN reflection trace is faster in the latest static and dynamic
  same-script runs.
- Keep release-size and Nsight-backed runs as the broader performance gate;
  this document records parity for the covered benchmark, not universal
  superiority for every multipath/diffraction workload.
- Nsight Systems was not available on PATH in this environment. Nsight Compute
  2025.2 was available, but counter collection failed with `ERR_NVGPUCTRPERM`;
  enable NVIDIA GPU performance counters or run an elevated profiler session
  before adding Nsight-backed claims.

## Parity Coverage Status

The current opt-in external RayD parity test covers same-scene forward cases for:

- `intersect`
- multi-mesh global ids
- point `nearest_edge`
- visibility
- reflection tracing
- diffraction paths
- direct, Keller, suffix, order-2, and order-3 diffraction accumulation
- coherent direct diffraction accumulation

Torch-native AD tests cover fixed-winner VJP/JVP for:

- `intersect` VJP/JVP
- `intersect(..., flags=RayFlags.None)` VJP/JVP through hidden tape
- point/ray `nearest_edge` VJP/JVP
- reflection trace VJP/JVP
- reflection EPC forward/VJP/JVP
- diffraction accumulation forward/VJP/JVP

Current default native discover result after the no-fallback chain AD migration:

- `python -m unittest discover tests.raydn_native -v`: 106 passed, 12 skipped.
- `RAYDN_RUN_DR_JIT_PARITY=1 python -m unittest tests.raydn_native.test_drjit_parity -v`:
  12 passed. The run printed `jitc_llvm_init(): LLVM API initialization failed ..`,
  as in earlier passing parity runs.

Visibility returns a discrete bool and has no continuous gradient contract.

## Multipath Implementation Status

RayDN now contains source ports for the RayD reflection and diffraction
`src/multipath` execution paths, including segment visibility, reflection trace,
reflection EPC, EPC field, reflection dedup, reflection accumulation,
diffraction path search, diffraction accumulation, chain accumulation, suffix
reflection, and coherent direct accumulation. Performance remains the active
completion risk.

See `docs/raydn_native_gap_analysis.md` for the tracked gap list.
