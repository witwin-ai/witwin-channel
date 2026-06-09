# RayDN CUDA / OptiX Performance Analysis

> Static source analysis and benchmark record from the cuda-optimize workflow.
> Items are labelled `[verified]` (provable from source) or `[needs Nsight]`
> (impact needs profiler measurement). Nsight Systems was not available on PATH
> in the current environment. Nsight Compute 2025.2 was available, but a focused
> capture failed with `ERR_NVGPUCTRPERM`; enable NVIDIA GPU performance counters
> or run an elevated profiler session before adding Nsight-backed claims.
>
> - Target GPU: NVIDIA GeForce RTX 5080 (Blackwell, compute_120), Torch CUDA 12.8, Windows.
> - Benchmark of record: `tests/benchmark_raydn_native.py`,
>   `tests/benchmark_rayd_vs_raydn.py`, and the RayD-latest-style
>   three-backend intersection stress benchmark
>   `tests/benchmark_raydn_rayd_mitsuba_stress.py`.
> - Open performance risks per
>   [raydn_native_performance.md](raydn_native_performance.md):
>   **warm-started local scene build** and **public static AD/VJP overhead**.

## Current implementation status (2026-06-08)

Implemented in the current worktree:

- CUDA build defaults now target modern GPUs instead of stale `sm_52`:
  `75-real;80-real;86-real;89-real;120-real;120-virtual`.
- Single-config native builds default to `Release` when the caller did not set
  `CMAKE_BUILD_TYPE`.
- OptiX PTX builds use fast math behind `RAYDN_OPTIX_FAST_MATH` and now
  compile explicit `compute_75` PTX instead of relying on nvcc's old default
  virtual architecture.
- `Scene.intersect(ray, flags=...)` now exposes the RayD-compatible
  `RayFlags.None/Geometric/ShadingN/UV/All` contract. The t-only/minimal
  path is reached through `flags=getattr(rt.RayFlags, "None")`.
- No-AD `intersect` uses an on-demand native path:
  - `None`: launches OptiX and returns only `t`.
  - `Geometric`: returns `t`, `p`, barycentric, ids, and `geo_n`.
  - `ShadingN`: returns `t` and `n`.
  - `UV`: returns `t` and `uv`.
  - `All`: returns the full legacy intersection fields.
- Reverse-mode and forward-mode AD now use the same public RayFlags contract.
  Differentiable calls still keep hidden hit tape for VJP/JVP, but the tape is
  reduced to `global_prim_id + (u,v) + t`; `RayFlags.None` no longer
  materializes public `p/n/uv/bary/ids`.
- `RayFlags.None` reverse-mode AD now uses a dedicated t-only backward native
  path. It does not allocate or pass full public-output gradient tensors for
  `p/n/geo_n/uv/barycentric`, skips unused ray/tmax gradient outputs via
  `ctx.needs_input_grad`, and uses warp-labeled aggregation for the t-only
  `grad_vertices` scatter.
- Edge topology construction is now fully GPU-side for the build path:
  per-face edge candidates are emitted on CUDA, sorted with CUB by canonical
  edge key, segmented, scanned, and expanded into the RayD-compatible boundary /
  manifold / non-manifold wedge records. The build path no longer calls
  `mesh.faces.cpu()`.
- Edge-search radii no longer copy six full edge SoA tensors back to CPU;
  CUDA reductions now finalize bbox/max-edge-length on device and read back
  only the 7 scalar values needed for host-side radius selection.
- The edge OptiX pipeline is split by payload pressure: point/ray queries use
  a 5-payload point/ray PTX/module/pipeline, while top-k keeps its separate
  16-payload PTX/module/pipeline.
- Edge point/ray nearest-edge queries now use one native CUDA setup launch plus
  one OptiX launch per query batch. The setup kernel splits public `[N,3]`
  point/ray tensors into SoA inputs without ATen `select().contiguous()`;
  raygen then iterates the existing tight-radius edge GAS tiers on the GPU and
  early-outs when a query resolves. Top-k keeps the separate 16-payload pipeline.
- Static triangle and edge GAS builds use `OPTIX_BUILD_FLAG_ALLOW_COMPACTION`
  and compact when OptiX reports a smaller output. Dynamic triangle/edge paths
  skip compaction to preserve update/rebuild buffer compatibility.
- Reflection no-AD tracing avoids allocating/writing autograd tape-only arrays.
- Default ray `tmax` and default mesh transforms now use empty identity/unbounded
  sentinels instead of constructing full GPU tensors for common defaults.
- Camera sample/world/ray transforms now route through native CUDA kernels with
  native VJP wrappers that consume the real upstream gradients.
- Scene global geometry refresh no longer uses per-mesh `at::cat`, `at::full`,
  `at::arange`, or `faces + offset` tensor ops. It allocates the final global
  buffers and packs each mesh with a native CUDA kernel.
- Multi-mesh AD for `Scene.intersect`, point `Scene.nearest_edge`,
  `Scene.trace_reflections`, and `Scene.trace_refl_epc_field` no longer uses
  Python `torch.cat` to build a global vertex tensor. The wrappers keep the
  original mesh tensors as autograd inputs, pack JVP tangents with native
  scene-cache kernels, and split global VJP gradients back to mesh leaves in
  C++.
- `Scene.intersect` full-output VJP/JVP now passes nullable upstream gradients
  and tangents into native optional kernels. Missing `p/n/geo_n/uv/barycentric`
  gradients and missing ray/vertex tangents are interpreted as zero inside the
  CUDA kernel rather than by Python `torch.zeros_like`; explicit nonuniform
  upstream gradients are still consumed directly. Full-output backward and
  JVP tangent inputs are read with tensor strides in CUDA, so non-contiguous or
  expanded upstreams do not require C++ ATen `.contiguous()` staging copies.
  JVP public outputs are also materialized according to `RayFlags` inside native
  code, so inactive `p/n/geo_n/uv/barycentric` tangents no longer require Python `Tensor.new_empty`
  sentinels and the CUDA kernel skips the disabled output writes. Forward-AD
  `active=None` also stays nullable through the Python public wrapper; the
  native binding returns the saved empty active-mask context as a hidden output.
- Point `Scene.nearest_edge` VJP/JVP uses the same optional-gradient pattern:
  Python no longer creates zero upstream tensors or sums duplicate `edge_t`
  upstreams before calling native code. Its JVP hidden `tape_s/tape_d` zero
  tangents are now written by the same native edge JVP kernel instead of Python
  `torch.zeros_like`. Explicit upstream gradients and tangents are read with
  tensor strides in CUDA, avoiding C++ ATen `.contiguous()` staging.
- `Scene.trace_reflections` VJP/JVP now routes missing upstream gradients and
  missing tangents through internal optional native entrypoints. Python no
  longer creates empty grad sentinels or `torch.zeros_like` tangent tensors for
  this public wrapper; explicit upstream gradients for `t` and `image_sources`
  are still passed through. AD forward also saves omitted active masks from a
  native hidden output instead of creating Python empty active-mask sentinels.
- `Scene.trace_refl_epc_field` backward/JVP now uses fused native CUDA kernels
  for the EPC field derivative path. Python passes nullable upstream gradients
  and tangents; the CUDA kernels compute `receiver - source`, field
  `sin/cos/(1+t)` derivatives, t-only intersection VJP/JVP, and
  source/receiver/vertex gradients without Python `torch.zeros_like` or the
  previous C++ ATen tensor-algebra chain. EPC upstream gradients and tangents
  are also consumed with tensor strides, so sliced/transposed inputs do not need
  C++ `.contiguous()` staging. AD forward also accepts omitted active masks as
  nullable native input and returns the saved active context as a hidden output.
- `Scene.trace_refl_epc_field` forward now fuses the public ray setup and EPC
  temporary initialization into `reflection_epc_forward_setup_kernel`: source
  and receiver AoS-to-SoA splitting, `receiver - source`, ray `tmax`, valid/path
  defaults, per-slot id/normal defaults, first-blocked defaults, and first-tape
  barycentric initialization are written in one CUDA launch. The field kernel
  also writes the first resolved/trace primitive ids directly, avoiding the
  previous C++ ATen `zeros/full/ones/sub/sum/sqrt/select/reshape/contiguous`
  fan-out around the OptiX EPC and field launches.
- `Scene.visible` public segment visibility now passes contiguous `[N,3]`
  endpoints as AoS pointers directly into the OptiX raygen. The raygen writes
  `visible`, first-blocking primitive, and the legacy `tape_t=inf` output in the
  same launch, avoiding Python endpoint copies plus the previous native
  `select().contiguous()` SoA split and `at::full` initialization fan-out.
- Reflection trace uses bounce-major internal output storage when
  `max_bounces > 1`, then returns the existing public `[ray, bounce]` tensors.
  This targets coalesced per-bounce stores without changing the Python or AD
  contract. `max_bounces == 1` keeps the original layout to avoid a pointless
  transpose in the RayD comparison benchmark.
- Reflection trace hit-gather now uses trace-specific packed triangle buffers:
  the scene cache keeps the existing SoA arrays for other kernels and additionally
  writes four `[N,4]` float tensors for p0/e1/e2/face-normal. Trace raygen reads
  four aligned `float4` records instead of 12 scattered component arrays.
- Visibility-style OptiX traces use first-hit termination where ignore lists are
  not required.
- Reflection and diffraction accumulation now use warp-aggregated same-cell
  atomics for the hot complex/power field scatter paths.
- Reflection accumulation also has a thresholded staged reduce path for future
  nonzero-material accumulation workloads: one per-ray/per-depth record is staged
  and reduced by cell with CUB before scatter.
- Diffraction no-AD, no-suffix direct/Keller accumulation has an additional
  thresholded staging path: OptiX writes one `(cell, value)` record per sample,
  then a CUDA/CUB radix sort + reduce-by-key collapses high-contention cells
  before scattering to the output grid.
- Coherent diffraction direct/multi field accumulation also has a thresholded
  staging path: each state/cell lane writes one keyed 8-float record, CUB reduces
  by direct-vs-multi cell key, and one scatter pass updates the 12 field arrays
  plus per-cell counts.
- Diffraction path export uses warp-aggregated path-slot reservation and a
  single primary-scene launch for order-1 export. It also avoids redundant
  zero-field stores and skips unused p1/p2 component writes for order-1 paths.
- Diffraction path export initializes all public output buffers with one native
  CUDA kernel and writes `p0` directly into the public AoS tensor from OptiX,
  avoiding the previous `at::zeros/full` fan-out and `at::stack(...).contiguous()`.
- Diffraction direct no-AD accumulation bypasses autograd tape export; AD/JVP/VJP
  still uses the full tape-producing path.
- Diffraction path export, coherent direct accumulation, and direct/chain
  accumulation forward accept nullable active arguments in the internal native
  bindings. Public wrappers now pass `None` for omitted active masks instead of
  allocating Python-side empty CUDA sentinels. Public `Scene.accum_dfr_direct(...)`
  direct no-AD and AD forward paths also pass `None` for unused recursive inputs.
- Diffraction direct/chain accumulation backward and JVP now pass missing output
  upstreams plus missing state/material tangents as nullable native arguments.
  The CUDA AD kernels read null pointers as zero, so Python no longer creates
  `torch.zeros`/`torch.zeros_like` tensors for `Scene.accum_dfr_direct(...)` or
  `Scene.accum_dfr(...)` backward/JVP missing inputs. Direct and chain JVP also
  return the zero tangents for currently non-differentiated public field
  components from the native binding instead of Python `torch.zeros_like`.
- `Scene.trace_dfr_paths(...)` no longer calls Python `_contig_states`,
  `_contig_material`, or endpoint `.contiguous()` staging. The internal native
  binding accepts a logical `state_limit` plus tensor strides for endpoints,
  state fields, material gain/valid masks, and active masks, so non-contiguous
  padded views and `states.count < physical rows` use the same OptiX export path
  as the contiguous benchmark layout.
- `Scene.accum_dfr_direct(...)` and `Scene.accum_dfr_coherent_direct(...)`
  no-AD forward paths pass a logical state limit and read state/material scalar
  fields plus vec3 state fields with tensor strides in native code. The no-AD
  accumulation bindings no longer use C++ ATen `split_vec3(...).contiguous()`
  staging for state vectors. `Scene.accum_dfr(...)` chain no-AD now uses the
  same internal binding for initial and recursive states, including a recursive
  logical state limit.
- `Scene.accum_dfr_direct(...)` AD backward/JVP now keeps the existing
  full-output/tape contract but consumes strided state/material tensors,
  strided tangents, strided upstream grid gradients, and a logical
  `states.count` directly in native CUDA. The direct AD C++ binding no longer
  performs `split_vec3(...).contiguous()`, `flatten_optional_f32(...)`, or
  `stack_vec3(...)` staging. Chain AD now uses the same no-fallback pattern for
  initial and recursive states: logical counts are passed to native code,
  state/material/tangent/upstream tensors are read with strides, and returned
  vec3 gradients are written directly into AoS tensors instead of being assembled
  with C++ ATen `stack`.
- Dynamic edge sync reuses compatible edge GAS/AABB/temp buffers and rebuilds
  in place. A direct `OPTIX_BUILD_OPERATION_UPDATE` refit was tested but caused
  a severe post-sync nearest-edge traversal regression on the benchmark shape,
  so it is not retained.
- Clarification: there is no CPU edge-topology fallback in the current build
  path. The remaining host readbacks in scene build are scalar control values
  needed for exact PyTorch tensor sizing or OptiX compaction decisions, not
  host-side geometry/topology computation.

Intentional non-fast paths that remain:

- Public API and benchmark acceptance tests guard against benchmark-specific
  shortcuts: no `_minimal` RayDN surface and no `t_sum_*` scalar-loss
  interface are used for acceptance timing.
- Intersect `RayFlags.All` AD still routes through the generic dense
  backward/JVP kernels because the public contract includes `p/n/geo_n/uv`
  and barycentric gradients. `RayFlags.None` now has a t-only native VJP kernel
  for arbitrary upstream `grad_t`, but public static `.backward()` remains
  slower than RayD on some measured shapes because PyTorch eager autograd still
  builds and executes the graph around a dense `grad_vertices` tensor.
- A lazy `Intersection` materialization experiment for static reverse-AD
  `flags=All` was measured but not retained. It helped the `.t.backward(...)`
  case, but it would make a "full VJP" benchmark avoid materializing full public
  intersection fields before the VJP, so it is not a fair full-output comparison.
- Reflection-chain backward/JVP now uses fused native CUDA kernels and consumes
  nullable, stride-aware upstream gradients/tangents for `t` and `image_sources`.
  It no longer uses the old C++ ATen bounce loop (`select`, `sum`, `where`-style
  masking, zero tensors, and copies). Remaining cost here is tensor allocation
  and the public autograd boundary, not PyTorch eager tensor algebra.
- `Scene.trace_refl_epc_field` forward still allocates public outputs and
  temporary tensors through C++ ATen before launching native CUDA/OptiX work,
  but the GPU tensor algebra and initialization fan-out have been moved into
  fused native kernels. Removing the remaining allocation overhead needs scratch
  reuse or a broader executor, not Python wrapper cleanup.
- Diffraction direct and chain AD no longer keep Python fallback or C++ ATen
  SoA/upstream staging around the full-output/tape contract. They still allocate
  public gradient/tangent output tensors through C++ ATen before launching native
  CUDA kernels. Removing that remaining allocation cost needs scratch reuse or a
  broader fused executor, not Python wrapper cleanup.
- Edge GAS true refit remains a guarded backlog item. Current sync uses
  rebuild-in-place when compatible, otherwise a full edge-accel rebuild path.

Latest verification:

- Incremental native build succeeded via `scripts/dev_build_native.ps1`.
- `python -m unittest tests.raydn_native.test_edge_queries -v`: 11 passed,
  including non-contiguous nearest-edge upstream/JVP tangent coverage.
- `python -m unittest tests.raydn_native.test_intersect_grad -v`: 13 passed,
  including non-contiguous upstream `grad_p`, expanded `grad_t`, and
  non-contiguous JVP tangent coverage.
- `python -m unittest tests.raydn_native.test_public_api_contract -v`:
  19 passed, including source-level guards against upstream-gradient/tangent
  `.contiguous()` staging, benchmark-only public APIs, unused-output gradient
  materialization, reflection-chain ATen bounce-loop regressions, and
  `trace_dfr_paths` / chain-diffraction Python contiguous staging regressions.
- `python -m unittest tests.raydn_native.test_multipath -v`: 32 passed,
  including non-contiguous reflection EPC upstream/JVP tangent coverage and
  non-contiguous `trace_reflections().image_sources` upstream VJP finite-difference
  coverage, plus strided diffraction path-export endpoint/state/material inputs
  and no-AD/AD chain accumulation strided initial/recursive logical-count coverage.
- `python -m unittest discover tests.raydn_native -v`: 106 passed, 12 skipped.
- Opt-in RayD parity:
  `RAYDN_RUN_DR_JIT_PARITY=1 python -m unittest tests.raydn_native.test_drjit_parity -v`:
  12 passed. The known `jitc_llvm_init()` warning appeared and did not affect assertions.
- `compute-sanitizer --tool memcheck` on the large-grid `z=0.5` point
  nearest-edge reproducer completed with `ERROR SUMMARY: 0 errors`.
- High-contention staged diffraction accumulation parity check at
  `direct_samples=4096`: max `power` diff `1.43e-6`, max `field_x_re` diff
  `3.05e-5`; direct counts and edge-use counts both matched at `4096`.
- High-contention coherent staged smoke: repeating one coherent state 4x on a
  32x32 grid triggered the staged path; all six direct field components matched
  the single-state result after dividing by 4, direct count scaled from `1024`
  to `4096`, and multi count stayed `0`.
- Reflection staged accumulation path compiles and is wired behind a high-contention
  threshold. Current public native smoke uses default air-air material parameters, so
  reflection accumulation remains zero and does not yet prove a speedup for that path.
- Multipath smoke benchmark:
  `python -m tests.benchmark_raydn_rayd_mitsuba_multipath --preset smoke --rayd-source local --rayd-root E:\Code\RayDi`
  completed for RayDN, RayD path, and Mitsuba path; outputs are under
  `artifacts/benchmarks/multipath/smoke_all/` and contain JSON, CSV, and one
  PNG grouped-bar chart only.
- Latest multipath standard RayD-path comparison used
  `python -m tests.benchmark_raydn_rayd_mitsuba_multipath --preset standard --backends raydn rayd_path --no-plots`.
  RayDN was faster for reflection trace at 65,536 rays / 2 and 4 bounces,
  1,048,576 rays / 2 bounces, and diffraction export at 65,536 and 1,048,576
  states. The 1,048,576-ray / 4-bounce reflection row was rerun with
  `--repeats 20 --warmup 5` after one standard-run outlier; the focused result
  was RayDN `0.4241 ms` vs RayD `0.7000 ms`.
- `git diff --check`: no whitespace errors; Git only reported existing LF/CRLF
  conversion warnings for touched files.
- RayD-warm-started AD spot check, `rayd-latest:64:128`, 16,384 rays,
  30 repeats, 8 warmup after native optional-gradient migration and disabled
  unused-output gradient materialization:
  static public VJP full RayDN `0.1621 ms` vs RayD `0.0879 ms`,
  static public VJP reduced RayDN `0.1371 ms` vs RayD `0.0735 ms`,
  dynamic public VJP full RayDN `0.2901 ms` vs RayD `1.6450 ms`,
  dynamic public VJP reduced RayDN `0.2619 ms` vs RayD `1.4679 ms`.
- Post-native-validity/camera/diffraction-stride same-script check, `grid=64`,
  `queries=4096`, RayD warm-started by the same harness:
  static and dynamic RayDN were faster for both intersection modes, nearest
  edge, reflection trace, direct diffraction, diffraction paths, and scene build
  in the package-RayD run shown below. Scene build remains workload/source/cache
  sensitive and should not be treated as universally settled.

Latest RayD vs RayDN static benchmark, `grid=64`, `queries=4096`,
`warmup=8`, `repeat=60`, RayD warm-started by the same harness:

| Operation | RayD ms | RayDN ms | Status |
|---|---:|---:|---|
| build | 2385.079 | 1551.944 | RayDN faster |
| intersect `RayFlags.None` | 0.2614 | 0.0381 | RayDN faster |
| intersect `RayFlags.All` | 0.2122 | 0.0490 | RayDN faster |
| nearest edge | 1.7860 | 1.4218 | RayDN faster |
| reflection trace | 0.4695 | 0.0378 | RayDN faster |
| diffraction direct | 0.8475 | 0.2495 | RayDN faster |
| diffraction paths | 0.5785 | 0.1057 | RayDN faster |

Latest dynamic benchmark, `grid=64`, `queries=4096`, `warmup=5`,
`repeat=30`:

| Operation | RayD ms | RayDN ms | Status |
|---|---:|---:|---|
| build | 2441.784 | 1545.659 | RayDN faster |
| intersect `RayFlags.None` | 0.1648 | 0.0294 | RayDN faster |
| intersect `RayFlags.All` | 0.1736 | 0.0586 | RayDN faster |
| nearest edge | 1.8150 | 1.3477 | RayDN faster |
| reflection trace | 0.5682 | 0.0570 | RayDN faster |
| diffraction direct | 0.4756 | 0.3010 | RayDN faster |
| diffraction paths | 0.4139 | 0.0688 | RayDN faster |

Latest RayDN-native multi-bounce check, `grid=64`, `queries=4096`,
`warmup=5`, `repeat=30`, `max_bounces=4`:

| Operation | RayDN ms |
|---|---:|
| build | 1544.059 |
| dynamic sync | 0.935 |
| intersect `RayFlags.None` | 0.0380 |
| intersect `RayFlags.All` | 0.0602 |
| nearest edge | 16.283 |
| reflection trace | 0.2359 |
| diffraction direct | 0.2204 |
| diffraction paths | 0.2909 |

The native benchmark's `nearest_edge` uses random 3D points after a dynamic sync,
which drives many queries to the full-radius edge tier; it is not the same
near-surface edge-query shape as the RayD comparison benchmark above.

For this benchmark shape, RayDN now meets the RayD-comparison target for
scene build, intersect, nearest-edge, reflection trace, direct diffraction, and
order-1 diffraction path export. Keep release-size and Nsight-backed benchmarks
before claiming broad superiority across all scenes and multipath workloads.

The `grid=64`, `queries=4096` RayD/RayDN benchmark is intentionally a
fast multipath regression test and is too light to be the only performance
claim. RayD's current Mitsuba comparison benchmark uses `mesh_resolution` /
`ray_grid_side` scenarios, with the default `64:128` shape already casting
16,384 rays. RayDN now has a matching three-backend stress harness:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_stress `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --scenario rayd-latest:64:128 `
  --scenario release:192:256 `
  --repeats 5 --warmup 2 --mitsuba-preliminary
```

This script reports static and dynamic `full` and `reduced` intersection
performance for RayDN, RayD, and Mitsuba. The `reduced` mode maps to
RayDN/RayD `RayFlags.None` and Mitsuba `ray_intersect(..., RayFlags.Minimal,
False)`. With `--mitsuba-preliminary`, it also reports Mitsuba's
`ray_intersect_preliminary` t-only path as a Mitsuba-only lower-level baseline.
Mitsuba is not used for nearest-edge, reflection, or diffraction totals because
those public APIs do not have a one-to-one equivalent in the current RayDN
benchmark.

For real scaling curves rather than hand-picked sizes, run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_sweep `
  --preset large `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary
```

The sweep script writes JSON, CSV, and PNG grouped-bar plots under
`artifacts/benchmarks/scaling/<preset>/`. The `large` preset reaches about
2.10M triangles and 10M requested rays; `extreme` includes a 100,663,296-ray
entry. For 10M/100M requested rays, the default mode measures a fixed ray batch
and projects total time from the required batch count. Add `--execute-total-rays`
to run every batch explicitly.

Latest scaling findings from `artifacts/benchmarks/scaling/extreme/sweep.json`:

| Case | RayDN | RayD | Mitsuba | Interpretation |
|---|---:|---:|---:|---|
| Static full, 2.10M tris / 100.66M requested rays | 0.3991 ms/batch | 0.3723 ms/batch | 0.5856 ms/batch | RayD is ~7% faster than RayDN; both beat Mitsuba public full. |
| Static reduced/t-only, same case | 0.1840 ms/batch | 0.1999 ms/batch | 0.2583 ms/batch | RayDN is fastest among public APIs. |
| Mitsuba preliminary, same case | n/a | n/a | 0.1785 ms/batch | Lower-level preliminary API can slightly beat RayDN reduced here. |

Across the 15 extreme static/full points, RayD is faster than RayDN in 13
points; the largest observed RayD advantage is about 33% at 524K triangles /
1.05M requested rays. The cause is not traversal quality alone: RayDN
`RayFlags.All` materializes the legacy Torch `Intersection` surface
(`p`, `n`, `geo_n`, `uv`, barycentric and four id arrays), while the RayD/Mitsuba
comparison path is closer to a lean public hit record. In static/reduced mode,
where the public contract is just `t`, RayDN is faster than RayD throughout
the extreme sweep and faster than Mitsuba's public minimal API; Mitsuba
`ray_intersect_preliminary` remains the strongest lower-level t-only baseline.

Latest multipath path-export benchmark:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_multipath `
  --preset smoke --rayd-source local --rayd-root E:\Code\RayDi
```

This is the RayD latest-style multipath path benchmark, not the Sionna solver
benchmark: `reflection_trace` uses the parallel-reflector scene with public
reduced path fields, and `diffraction_export` uses the synthetic single-edge
state path.
The script writes `multipath.json`, `multipath.csv`, and
`time_ms_multipath.png` to one output folder and does not emit SVG or throughput
plots. The smoke run is only a path/plot validation because it uses two timed
samples; release claims should use higher repeats and larger `ray_count` /
`state_count` sweeps.

Latest AD backward coverage:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -B -m tests.benchmark_raydn_rayd_mitsuba_sweep `
  --preset smoke --mesh-resolution 64 --mesh-resolution 128 --mesh-resolution 256 `
  --total-rays 16384 --total-rays 65536 --ray-batch-side 256 `
  --repeats 5 --warmup 3 --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary --include-backward `
  --output-dir artifacts\benchmarks\scaling\ad_uv_tape
```

The sweep plots absolute time only. Operation charts use grouped bars with one
subplot per mode and backend bars per case; throughput is still present in JSON
as a derived field but is not plotted. Representative high-repeat AD point:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_stress `
  --scenario ad_uv_tape_256_65k:192:256 --backends raydn rayd `
  --repeats 120 --warmup 24 --rayd-source package `
  --include-backward
```

| 73.7K triangles / 65.5K rays, current public VJP | RayDN | RayD | Status |
|---|---:|---:|---|
| Static forward full | 0.0474 ms | 0.1411 ms | RayDN faster |
| Static forward reduced | 0.0283 ms | 0.1183 ms | RayDN faster |
| Static public VJP full | 0.1142 ms | 0.0947 ms | RayD faster |
| Static public VJP reduced | 0.1117 ms | 0.0717 ms | RayD faster |
| Dynamic public VJP full | 0.2241 ms | 2.3145 ms | RayDN faster |
| Dynamic public VJP reduced | 0.2296 ms | 2.3185 ms | RayDN faster |

Conclusion for AD: `RayFlags.None` AD now has a native t+tape forward path and a
t-only VJP kernel that accepts arbitrary upstream gradients; the removed
`t_sum_*` benchmark interface is no longer part of the public or acceptance
surface. Direct native `intersect_forward + intersect_backward_t` measures
about 0.055-0.069 ms on the 16K/65K-ray stress shapes, faster than RayD's
static VJP numbers. The no-active vertices-only t-VJP path now avoids the empty
active sentinel, saves a smaller autograd context, and normalizes default empty
`tmax` out of the internal request. The remaining public static backward gap is
the PyTorch eager autograd engine and tensor-allocation layer around
`.backward()`, not the CUDA VJP math kernel. `torch.compile` still graph-breaks
on the pybind custom extension boundary in this workload and did not remove that
fixed cost.

The stress benchmark also supports `--torch-loss-backward`, which wraps feasible
RayD/Mitsuba paths through `torch.autograd` and measures a Torch
`loss.backward()` shell. RayD dynamic and Mitsuba static/dynamic paths can be
wrapped. RayD warm-started static scenes cannot be fairly wrapped without
rebuilding the static scene or changing RayD, because the differentiable vertex
array is bound at scene build. The benchmark reports that case as unsupported
instead of timing a dummy or stale-gradient path.

Latest no-fallback strict materialized three-backend Torch-loss run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydn_rayd_mitsuba_stress `
  --scenario ad_uv_tape_256_65k:192:256 --backends raydn rayd mitsuba `
  --repeats 30 --warmup 8 --rayd-source package --include-backward `
  --torch-loss-backward --materialize-full-vjp --require-mitsuba `
  --json-output artifacts\benchmarks\scaling\ad_uv_tape\no_fallback_torch_loss_materialized_current.json
```

73.7K triangles / 65.5K rays:

| Torch-loss mode | RayDN | RayD | Mitsuba |
|---|---:|---:|---:|
| dynamic full | 0.3559 ms | 3.0661 ms | 2.2322 ms |
| dynamic reduced | 0.3411 ms | 3.2010 ms | 2.2117 ms |
| static full | 0.3340 ms | unsupported | 2.2374 ms |
| static reduced | 0.2798 ms | unsupported | 2.2802 ms |

## Contents

- [Executive summary](#executive-summary)
- [Priority table](#priority-table)
- [P0 â€?Build configuration](#p0--build-configuration)
- [P1 â€?Scene build](#p1--scene-build)
- [P1 â€?Nearest-edge query](#p1--nearest-edge-query)
- [P1 â€?Reflection trace](#p1--reflection-trace)
- [P2 â€?Reflection accumulation / EPC / dedup](#p2--reflection-accumulation--epc--dedup)
- [P2 â€?Diffraction](#p2--diffraction)
- [Cross-cutting â€?host glue, allocations, syncs](#cross-cutting--host-glue-allocations-syncs)
- [Measurement plan](#measurement-plan)
- [Suggested execution order](#suggested-execution-order)

---

## Executive summary

The three slowest paths each have a clear primary suspect that is provable from source:

1. **Scene build** originally had two avoidable host stalls: CPU edge topology extraction
   via `faces.cpu()` and host-side bbox/max-edge reduction. Both are now GPU-side;
   remaining build work is mostly OptiX GAS/IAS construction and tensor setup.

2. **Reflection trace** originally wrote per-bounce outputs in a **ray-major layout**
   (`slot = ray*B + bounce`) across ~17â€?4 separate SoA arrays. The current worktree now
   uses bounce-major internal storage for `max_bounces > 1`, while returning the same public
   `[ray, bounce]` tensors. Nsight should still verify store coalescing and transpose cost.

3. **Nearest-edge query** previously issued **one OptiX launch + one finalize kernel +
   one params H2D copy per radius tier**. Point/ray now launches once and loops over the
   existing tight edge GAS tiers inside raygen; top-k remains on its own high-payload path.

Above all of these sits a **build-configuration defect**: the cached build targets
`sm_52` (Maxwell) on a Blackwell GPU. Fixing the architecture is zero-risk and may lift
every kernel before any code is touched.

This document is both a status record and a remaining backlog. Confirm impact ordering with
the [measurement plan](#measurement-plan) before investing in the remaining larger refactors:
split-scene support, exact stack sizing, materialized reflection-accumulation benchmarks,
and additional GAS/IAS sync work.

---

## Priority table

| Level | Items | One-liner |
|---|---|---|
| **P0** | 1, 2, 3 | Architecture, build type, fast-math/PTX arch fixes landed |
| **P1 build** | 4, 5, 6, 7 | GPU edge topology + D2H radii fixed; edge sync partially improved; static GAS compaction landed |
| **P1 edge** | 11, 12 | Point/ray tier launches collapsed; point/ray payload split landed |
| **P1 refl** | 15, 16, 17 | Bounce-major trace writes and packed hit gather landed; split-mode is inactive in current Torch call sites |
| **P2 accum** | 19, 20, 25, 26 | Same-cell warp atomics, path-counter aggregation, reflection staged reduce, and thresholded diffraction sort/reduce landed |

Impact legend: **High** = likely measurable on the benchmark; **Med** = real but
secondary; **Low** = correctness-neutral cleanup / small constant.

---

## P0 â€?Build configuration

Highest leverage, smallest change. Do these first; they may shift the whole ranking.

### 1. CUDA architecture compiled as `sm_52` on a Blackwell GPU `[verified] â€?High`

- Evidence: `artifacts/skbuild/CMakeCache.txt` â†?`CMAKE_CUDA_ARCHITECTURES:STRING=52`.
  Neither [CMakeLists.txt](../CMakeLists.txt) nor [pyproject.toml](../pyproject.toml) pins an
  architecture, so CMake fell back to the legacy default of 52.
- Impact: every non-OptiX `.cu` kernel (`cache_kernels`, `geometry_*`, `dedup`,
  `epc_field`, `backward`, `accum_ad`, â€? is generated for Maxwell and JIT-recompiled from
  PTX at runtime, using old-arch scheduling/occupancy heuristics, plus first-run JIT latency.
- Fix direction depends on whether the build is for **local dev** or a **published wheel**:
  - **Local dev (single known GPU):** `set(CMAKE_CUDA_ARCHITECTURES native)` (builds only for
    the build machine's GPU â†?`sm_120` here) or explicit `120`. Simplest and fastest to
    compile; do **not** ship this.
  - **Published wheel (others install it):** use a **multi-architecture list with a low
    virtual/PTX baseline**, e.g.
    `set(CMAKE_CUDA_ARCHITECTURES 75-real;80-real;86-real;89-real;120-real;120-virtual)`
    (or via pyproject:
    `[tool.scikit-build.cmake.define] CMAKE_CUDA_ARCHITECTURES = "75-real;80-real;86-real;89-real;120-real;120-virtual"`).
    The `-real` entries embed optimized SASS for each target GPU (no JIT); the trailing
    `120-virtual` keeps PTX for forward-compat with future GPUs.
- Compatibility note (why this matters for distribution): PTX JIT is **forward-only** â€?
  `compute_XX` PTX runs on compute capability â‰?XX, never below; `sm_XX` SASS is arch-bound.
  - The **lowest virtual/PTX entry sets the minimum supported GPU.** To keep supporting old
    cards, lower the baseline (e.g. add `52-virtual` for Maxwell, `61`/`75` otherwise).
  - Adding `120` does **not** drop old-GPU support â€?only *replacing* the broad baseline with
    a single high arch (`120` or `native`) does. A `120`-only or `native` wheel will **fail to
    load on any pre-Blackwell GPU**, so never publish those.
  - The current `52` build embeds `sm_52` SASS + `compute_52` PTX, so it runs everywhere from
    Maxwell up â€?but on this RTX 5080 it runs *only* via `compute_52` PTX JIT (slow, plus
    first-run JIT latency), which is exactly the defect this item fixes.
  - Trade-off: each `-real` arch adds a cubin â†?larger wheel and longer compile time. Pick the
    `-real` set to match the GPUs users actually have (Turing 75 / Ampere 80,86 / Ada 89 /
    Blackwell 120 above).
- Risk: none for correctness. Re-measure the whole benchmark â€?this can move every number.

### 2. `CMAKE_BUILD_TYPE` absent / empty in the cache `[implemented] â€?Med`

- Evidence: `CMAKE_BUILD_TYPE` does not appear in `artifacts/skbuild/CMakeCache.txt`.
- Impact: host glue (`scene_cache.cpp`, the `ops.cpp` files â€?heavy STL + tensor code) may
  compile without `/O2`. This directly touches the build path, which is CPU-bound.
- Implemented: single-config native builds now default to `Release` when
  `CMAKE_BUILD_TYPE` was not explicitly provided. Multi-config Visual Studio builds still
  use the requested configuration.

### 3. OptiX PTX compiled without `--use_fast_math` (and no arch flag) `[implemented] â€?Med`

- Evidence: every `--ptx` custom command in [CMakeLists.txt:75-310](../CMakeLists.txt#L75)
  passes only `--std=c++17`.
- Impact: the diffraction / accumulation kernels do many `sincosf` / `sqrtf` / divides (see
  items 28, 19) at full precision. OptiX re-optimizes the PTX, but fast-math semantics must be
  set at the source compile.
- Implemented: PTX custom commands use `RAYDN_OPTIX_NVCC_FLAGS`, which includes
  `--gpu-architecture=compute_75` and, by default, `--use_fast_math`. The explicit PTX
  architecture was required for warp intrinsics such as `__match_any_sync`.

---

## P1 â€?Scene build

Target of record: `build_ms` â‰?1550 (native, grid 192) / â‰?142 (grid 64 vs RayD 95).

### 4. Edge topology built on the CPU, single-threaded, via `std::map` `[implemented] - High`

- Original problem: `mesh.faces.cpu()` forced a synchronous D2H; then every triangle inserted its 3 undirected edges into a host map. At grid 192 the triangle count is large, so this was the most probable build hotspot.
- Implemented: `build_edge_topology_cuda()` emits all 3 edges per face on CUDA, sorts canonical `(min,max)` edge keys with CUB radix sort, marks key runs, scans output counts, and expands boundary/manifold/non-manifold RayD-compatible wedge records on the GPU. Non-manifold edges emit every unordered incident-face pair; the build path has no host topology implementation or CPU fallback.

### 5. Edge search radii via six `.cpu()` D2H + serial CPU bbox reduction `[implemented] - High`

- Location: [scene_cache.cpp:484-497](../src/torch_ext/scene/scene_cache.cpp#L484)
  (six `scene.edge_*.cpu()` copies) feeding
  [compute_edge_search_radii:383-449](../src/torch_ext/scene/scene_cache.cpp#L383).
- Problem: the edge SoA was just produced on the GPU, then copied back to the host as six
  independent synchronous transfers, only to run a serial min/max + max-edge-length loop.
- Implemented: `compute_edge_search_stats_cuda()` now reduces bbox and max edge length on
  CUDA. The first kernel writes per-block partials; a second CUDA kernel finalizes those
  partials into 7 scalar values. The host reads only those final scalars for radius
  selection.

### 6. Up to 3 edge GAS rebuilt on every build *and* every sync `[partial] â€?Med-High`

- Location: [scene_cache.cpp:498-556](../src/torch_ext/scene/scene_cache.cpp#L498) â€?loops
  `radii.size()` times, each iteration `optixAccelComputeMemoryUsage` + `optixAccelBuild`.
- Problem: three custom-primitive GAS are sized and built serially. Historically
  `sync_scene` called `build_edge_accel` again, reallocating and rebuilding every tier.
- Fix direction: (a) reassess whether 3 radius tiers are needed or whether one GAS with
  raygen-side tiered `tmax` suffices; (b) for the dynamic path, refit with `ALLOW_UPDATE` +
  `OPTIX_BUILD_OPERATION_UPDATE` instead of full rebuilds.
- Current state: compatible dynamic syncs reuse edge AABB/GAS/temp buffers and rebuild
  the GAS in place. A direct `OPTIX_BUILD_OPERATION_UPDATE` refit was tested and rejected
  because it made post-sync nearest-edge traversal much slower on the native benchmark.
  The point/ray query launch collapse is implemented without merging tiers into one
  broad-radius GAS; the existing tight-radius GAS tiers remain to protect traversal quality.

### 7. No acceleration-structure compaction anywhere `[implemented static GAS] - Med`

- Original problem: GAS output buffers were left uncompacted: larger VRAM footprint and worse traversal cache locality.
- Implemented: static triangle and edge GAS builds request compacted size and call `optixAccelCompact()` when OptiX reports a smaller output. Dynamic triangle/edge paths skip compaction to preserve update/rebuild buffer compatibility.

### 8. `refresh_global_geometry` does many tiny tensor ops + 12 separate SoA buffers `[verified] â€?Med`

- Location: [scene_cache.cpp:172-241](../src/torch_ext/scene/scene_cache.cpp#L172).
- Problem: per-mesh `at::full` / `at::arange` / `(faces + vertex_offset)` / `at::cat` spawn
  many small kernels and temporaries; then 12 distinct `tri_*` tensors
  (`tri_p0_x â€?tri_fn_z`) are allocated.
- Fix direction: write a packed layout directly from one `compute_*` kernel; generate
  shape/local ids in a single kernel; minimize `cat` calls.

### 9. Triangle GAS built serially per mesh `[verified] â€?Med`

- Location: [scene_cache.cpp:584-586](../src/torch_ext/scene/scene_cache.cpp#L584).
- Fix direction: for multi-mesh scenes, batch into a single build with multiple build
  inputs, or build on concurrent streams.

### 10. IAS instances constructed on host and re-copied every sync `[verified] â€?Low-Med`

- Location: [build_triangle_ias:115-134](../src/torch_ext/scene/scene_cache.cpp#L115).
- Problem: the identity transforms never change, yet the instance buffer is rebuilt and
  re-`cudaMemcpyAsync`'d on every sync.
- Fix direction: reuse the instance buffer and refit the IAS (`ALLOW_UPDATE`).

---

## P1 â€?Nearest-edge query

### 11. Tiered point/ray query host launches `[implemented] - Med-High`

- Original problem: point and ray nearest-edge queries looped over `scene.edge_accels` on
  the host. Each radius tier issued a `cudaMemcpyAsync(params)` + `optixLaunch`, and ray
  queries also launched a finalize kernel after every tier.
- Implemented: point/ray launch params carry the tier handle/radius array. `edge_optix.cu`
  loops over tiers inside raygen, keeps the tight per-tier GAS/AABB bounds, and early-outs
  when the query resolves. Ray query payload 4 now carries the current tier radius bits;
  validity is `edge_id != invalid`, so the 5-payload point/ray pipeline stays intact.
- Measured effect on the RayD comparison shape after this change: nearest-edge static
  improved from the previous RayDN record of `1.0648 ms` to `1.0081 ms`, while RayD
  measured `1.2564 ms` in the same run. Dynamic nearest-edge measured RayDN
  `1.0102 ms` vs RayD `1.1727 ms`.

### 11b. Point/ray query SoA setup used ATen split/copy `[implemented] - Med`

- Original problem: public point/ray nearest-edge still prepared OptiX SoA query
  inputs with C++ ATen `select(1, axis).contiguous()` calls, creating 3 launches
  for point queries and 6 launches for ray queries before the OptiX query.
- Implemented: point queries now launch one native CUDA split kernel, and ray
  queries launch one native CUDA split kernel that writes both origin and
  direction SoA arrays. This preserves the existing public API and OptiX SoA
  traversal path while removing PyTorch eager tensor ops from this setup.
- A direct-AoS OptiX experiment was rejected after `compute-sanitizer` found an
  illegal device read in the large-grid, high-z query case. The retained native
  split path passes the same `z=0.5` large-grid memcheck with 0 errors.
- Latest same-script nearest-edge check after this change: static RayDN
  `0.9638 ms` vs RayD `1.2084 ms`; dynamic RayDN `0.9618 ms` vs RayD
  `1.2475 ms` (`grid=64`, `queries=4096`, `warmup=5`, `repeat=30`).

### 12. Edge pipeline reserves 16 payload registers for all raygens `[implemented] - Med`

- Original problem: payload count is per-pipeline, so point/ray raygens paid the full 16-register top-k reservation even though point uses 4 payload values and ray uses 5.
- Implemented: `edge_optix.cu` now compiles into separate point/ray-only and top-k-only PTX modules. Runtime creates a 5-payload point/ray pipeline and a 16-payload top-k pipeline, each with its own module and SBT records. Confirm register reduction with `launch__registers_per_thread`.

### 13. AoS point/edge coordinates loaded as 3 scalar loads `[verified] â€?Low`

- Location: [edge_forward.cu:17](../src/torch_ext/edge/edge_forward.cu#L17) `make_aos_f3`;
  [edge_optix.cu:31-51](../src/torch_ext/edge/edge_optix.cu#L31).
- Fix direction: with 16-byte alignment, vectorize via `float4`/reinterpret (the edge SoA is
  already split into component arrays, so this mainly helps the AoS query points).

### 14. Ray query anyhit calls `optixIgnoreIntersection()` for every candidate `[verified] â€?Low-Med`

- Location: [edge_optix.cu:380-391](../src/torch_ext/edge/edge_optix.cu#L380). The bigger the
  search radius, the more candidates, the more anyhit invocations.
- Fix direction: couple with items 6/11 to tighten the radius tiers; ensure custom-primitive
  AABBs are not over-inflated.

---

## P1 â€?Reflection trace

### 15. Ray-major output layout â†?strided, uncoalesced stores `[implemented for B>1] â€?High`

- Location: [trace_optix.cu:187-216](../src/torch_ext/reflection/trace_optix.cu#L187), feeding
  ~24 independent SoA output arrays defined in
  [trace_params.h:44-67](../include/raydn/reflection/trace_params.h#L44).
- Original problem: outputs were indexed `slot = ray_index * B + bounce`. For a fixed
  bounce, adjacent threads wrote addresses that differed by `B`, so each warp store could
  degenerate into many transactions.
- Implemented: for `max_bounces > 1`, raygen writes bounce-major storage
  (`bounce * n_rays + ray`) and host glue transposes back to public ray-major tensors.
  `max_bounces == 1` keeps ray-major storage because both layouts are contiguous and a
  transpose would only add overhead.
- Confirm with `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio` on
  `__raygen__reflection_trace`.

### 16. Each hit re-gathers 12 scattered `tri_*[global_prim]` to recompute hit point/normal `[implemented] - Med`

- Original problem: `global_prim` is effectively random across a warp, and trace raygen
  reloaded p0/e1/e2/fn as 12 independent component arrays for every hit.
- Implemented: scene cache now writes trace-only packed triangle tensors alongside the
  existing SoA arrays. `trace_optix.cu` reads p0/e1/e2/fn through four aligned `float4`
  records and keeps the SoA compatibility read path only for internal callers
  that still provide component arrays.
- Measured effect: single-bounce RayD comparison remains traversal-bound and moves with
  benchmark noise. The native `max_bounces=4`, `warmup=5`, `repeat=30` check measured
  reflection trace at `0.2359 ms`, while an earlier repeat-60 run after the same layout
  change measured `0.2182 ms`. Treat the packed layout as implemented but not yet proven
  by Nsight counters.

### 17. `split_mode` traces twice per bounce (primary + secondary) `[verified] â€?Med`

- Location: [trace_optix.cu:128-136](../src/torch_ext/reflection/trace_optix.cu#L128) and the
  trailing segment [:238-246](../src/torch_ext/reflection/trace_optix.cu#L238).
- Current Torch call sites set `split_mode=0`, so this is not active in the latest
  RayD/RayDN comparison benchmark. RayDN currently builds one triangle IAS for the
  scene, so there is no split-scene double trace on the hot path.
- Future split-scene support should still prefer merging static/dynamic instances into one
  IAS so one traversal replaces the second per-bounce trace.

### 18. Shared multipath pipeline uses a hardcoded oversized stack `[verified] â€?Low`

- Location: [optix_pipeline.cpp:231](../src/torch_ext/common/optix_pipeline.cpp#L231) â€?
  `optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2)`. The context pipelines, by contrast,
  compute exact sizes via `optixUtilComputeStackSizes`.
- Fix direction: compute the multipath stack precisely too, reducing VRAM and potential spill.

---

## P2 â€?Reflection accumulation / EPC / dedup

> Mostly from a focused sub-agent read of the accumulation modules; treat impact as
> `[needs Nsight]` unless noted.

### 19. Complex-field accumulation via up to 7 `atomicAdd` into hashed cells `[implemented warp aggregation + staged reduce hook] â€?High`

- Location: `accum_optix.cu:377-384`, coherent branch `:445-460`.
- Problem: many threads atomically add into the same cell â€?serialized, complex-valued
  (re/im split into separate atomics per component).
- Implemented: hot reflection and diffraction field/power scatter paths now use same-cell
  warp aggregation before the global atomic. Multi-output paths share one cached
  `WarpCellGroup` inside each output branch, avoiding repeated `__match_any_sync` /
  leader discovery for each field component. This changes same-warp floating-point
  addition order, so parity tests are the correctness guard. Diffraction direct/Keller
  also has a heavier no-AD/no-suffix staged sort/reduce path described in item 26.
- Reflection accumulation now has an optional high-contention staging path:
  `__raygen__reflection_accumulation` writes one `(cell, ReflAccumStagedValue)` per
  `ray/depth` slot when the host predicts at least 2048 samples and at least 4 samples
  per cell; `reduce_refl_accum_staged_cuda()` then uses CUB radix sort and reduce-by-key
  before scattering the seven field/power outputs plus reflection count. Existing
  low-contention calls keep the warp-aggregated atomic path.
- Caveat: the current public native smoke path creates default material parameters with
  zero reflection coefficient, so this staged reflection path is compiled and parity-safe
  but not yet tied to a nonzero reflection-accumulation benchmark.

### 20. Audit occlusion/visibility ray flags for `TERMINATE_ON_FIRST_HIT | DISABLE_ANYHIT` `[needs check] â€?High if confirmed`

- Already correct: `visibility_optix.cu` uses `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`.
- To verify: the `DISABLE_ANYHIT`-only traces in `accum_optix.cu` / `epc_optix.cu`. **If a
  trace is an occlusion/shadow test**, add `TERMINATE_ON_FIRST_HIT` (and resolve in miss /
  `optixHitObject`). **If it is a nearest-reflection-point search**, the current closest-hit
  semantics are correct â€?do not change. Decide per call site.

### 21. EPC raygen holds ~500 B of stack-local arrays per thread `[needs Nsight] â€?Med`

- Location: `epc_optix.cu:437-550` â€?five `[ReflEpcMaxBounces=8]` float3 arrays.
- Problem: high register / local-memory footprint depresses occupancy.
- Fix direction: shrink live state, process bounces in chunks, check for local-memory spills
  (`launch__registers_per_thread`, local load/store metrics).

### 22. dedup blocks the host with `cudaStreamSynchronize` to return `unique_count` `[needs Nsight] â€?Med`

- Location: dedup.cu host path (count copy + sync).
- Fix direction: keep the count device-resident and consume it from a follow-up kernel;
  avoid the host stall that serializes the next op.

### 23. EPC field forward setup fan-out `[implemented setup fusion; needs Nsight] â€?Med`

- Location: `ops.cpp:602-796`, `epc_optix.cu`, `epc_field.cu`.
- Original problem: the public EPC forward prepared ray/slot/material tensors
  with C++ ATen operations around the OptiX EPC launch and EPC field launch.
- Implemented: ray AoS-to-SoA setup, ray direction/tmax, EPC temp defaults,
  per-slot id/normal defaults, first-blocked defaults, first-tape barycentric
  defaults, material defaults, polarization defaults, optional y/z field writes,
  and first resolved/trace primitive extraction now live in native CUDA kernels.
- Remaining work: public and temporary tensor allocation still goes through
  C++ ATen. Nsight should verify whether allocation, OptiX EPC traversal, or the
  EPC field kernel is now the dominant forward cost before adding scratch reuse
  or more fusion.

### 24. dedup compact writes 13 scattered fields per bounce `[needs Nsight] â€?Med`

- Location: `dedup.cu:285-302`.
- Fix direction: structured / merged writes.

---

## P2 â€?Diffraction

### 25. Path counter is a single global `atomicAdd(out_count, 1)` `[implemented warp aggregation] â€?High`

- Location: `paths_optix.cu:250` / `:394`.
- Problem: a global atomic serializes all hit-writes.
- Implemented: path export reserves output slots per warp using one `atomicAdd` per active
  warp group. Prefix-sum allocation remains a larger alternative.

### 26. Coherent UTD atomics per cell (6 complex components + counters) `[implemented warp aggregation + staged reduce] â€?High`

- Location: `accum_optix.cu:445-460` (same class of issue as item 19).
- Implemented: coherent direct/multi field outputs now use same-cell warp aggregation for
  the field components and per-cell counts, sharing one cached `WarpCellGroup` across the
  output channels within each direct or multi branch.
- Implemented for no-AD/no-suffix direct/Keller order-1 accumulation: when
  `launch_count >= 2048` and at least 4 samples per cell are expected, OptiX stages
  `(cell, float4(power, field_x_re, direct_count, keller_count))`, then
  `reduce_dfr_accum_staged_cuda()` uses CUB radix sort and reduce-by-key before one
  scatter pass updates the output tensors. AD, recursive, suffix, and low-contention
  direct paths keep the existing warp-aggregated atomic path.
- Implemented for coherent direct/multi UTD accumulation: when
  `state_count * cell_count >= 2048` and at least 4 states per cell are expected, OptiX
  stages an 8-float direct/multi keyed value per lane. `reduce_dfr_coherent_accum_staged_cuda()`
  sorts and reduces by key, then scatters reduced direct and multi field components plus
  per-cell counts. Low-contention coherent calls keep the existing warp-aggregated atomic path.
- Remaining larger work: use Nsight to decide whether staged sort cost beats atomics across
  larger coherent workloads and tune the threshold if needed.

### 27. Path output scatters 12 SoA complex components via `out_idx` `[needs Nsight] â€?Med`

- Location: `paths_optix.cu:261-280`.
- Fix direction: merged / packed writes.

### 28. `sincosf` / `sqrtf` / phase math at full precision `[needs Nsight] â€?Med`

- Location: `paths_optix.cu:256-259` and similar.
- Fix direction: pairs with P0-3 (`--use_fast_math`); validate against parity tests.

### 29. AD unit-JVP loop: 36+ serial `add_unit_vjp` + scattered `atomicAdd(ptr+index)` `[needs Nsight] â€?Med (backward only)`

- Location: `accum_ad.cu:1518-1627`, `:1774-1775`.
- Problem: many small atomics + per-call `nullptr` branches + large intermediate structs
  (register pressure).
- Fix direction: batch gradient accumulation, prune the nullptr branches, shrink live state.

---

## Cross-cutting â€?host glue, allocations, syncs

### 30. Many small `at::zeros` / `at::full` / `at::empty` allocations before launches `[verified] â€?Low-Med`

- Location: `reflection/ops.cpp:252-283`, `diffraction/ops.cpp:255-278`, and similar.
- Fix direction: batch-allocate output buffers / reuse scratch buffers across calls.

### 31. Every OptiX launch re-copies the full params struct to device `[verified] â€?Low`

- Location: [optix_pipeline.cpp:306](../src/torch_ext/common/optix_pipeline.cpp#L306),
  [edge_forward.cu:257](../src/torch_ext/edge/edge_forward.cu#L257), and each `ops.cpp`.
- Problem: large struct re-transferred per launch.
- Fix direction: cache the invariant portion, or stage from a pinned host buffer.

### 32. `hitgroup_record_capacity` rounds up to a minimum of 64 SBT records `[verified] â€?Low`

- Location: [optix_pipeline.cpp:107](../src/torch_ext/common/optix_pipeline.cpp#L107).
- Problem: most pipelines need a single record; an oversized SBT hurts cache locality.
- Fix direction: verify the minimum is justified; size to the actual record count.

### 33. intersect raygen loads ray origin/dir as 3 scalar loads `[verified] â€?Low`

- Location: [optix_intersect.cu:22-29](../src/torch_ext/scene/optix_intersect.cu#L22).
- Fix direction: vectorize after guaranteeing alignment.

---

## Measurement plan

Measure before and after each change; confirm the ranking before the large refactors
(items 4, 5, 15). On Windows, Nsight Compute counters need elevated rights or the GPU
performance-counter restriction lifted, else `ERR_NVGPUCTRPERM`.

```powershell
# 0) Repository benchmark of record
conda run -n witwin2 python -m tests.benchmark_raydn_native --grid 192 --queries 65536

# 1) System timeline â€?is scene build CPU-bound? where are the D2H copies / AS builds?
nsys profile -o prof --stats=true --force-overwrite=true `
  python -m tests.benchmark_raydn_native --grid 192 --queries 65536

# 2) Reflection trace â€?store coalescing + occupancy (item 15)
ncu --set full -k "raygen__reflection_trace" -c 5 -o refl <app>
ncu --metrics `
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,`
  launch__registers_per_thread,launch__occupancy_limit_registers `
  -k "raygen__reflection_trace" -c 5 <app>

# 3) Nearest-edge â€?payload register pressure / occupancy (item 12)
ncu --metrics `
  launch__registers_per_thread,launch__occupancy_limit_registers,`
  sm__warps_active.avg.pct_of_peak_sustained_active `
  -k "raygen__edge" -c 5 <app>
```

What to look for:

- Build: in the `nsys` timeline, confirm that the old item-4/item-5 host gaps are gone.
  Remaining build time should mostly be CUB topology kernels, OptiX GAS/IAS builds,
  compaction, and small host control work.
- Reflection trace: compare `...op_st.ratio` for `max_bounces > 1` before/after the
  bounce-major internal layout to verify the expected store coalescing and transpose cost.
- Edge: `launch__occupancy_limit_registers` being the limiter confirms the 16-payload cost
  of item 12.

---

## Suggested execution order

1. **Nsight validate landed changes**: PTX fast-math/arch, bounce-major reflection
   trace (`max_bounces > 1`), and warp-aggregated accumulation atomics.
2. **Build path**: GPU edge topology (item 4), GPU edge stats (item 5), and static AS
   compaction (item 7) are landed. Next build-path work should be driven by Nsight
   evidence around GAS/IAS construction and scalar synchronization.
3. **Edge query**: split point/ray vs top-k payload pressure and point/ray tier-launch
   collapse are landed; next step is Nsight validation of payload registers, launch count,
   and traversal cost across larger/random query distributions.
4. **Reflection remaining work**: revisit split-scene double traces (item 17) and validate
   packed hit-gather register/memory counters with Nsight on multi-bounce scenes.
5. **Accumulation next level**: validate the staged direct/Keller, coherent UTD, and
   reflection hooks with Nsight on larger workloads; tune thresholds and add a
   nonzero-material reflection accumulation benchmark.

Accuracy guard: items 3, 19, 25, 26, 28 (fast-math, reassociated/atomic reductions) change
the floating-point contract. Keep the native AD and opt-in RayD parity tests green after each.
