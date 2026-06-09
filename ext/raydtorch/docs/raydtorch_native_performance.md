# RayDTorch Native Performance

Measured on Windows with NVIDIA GeForce RTX 5080 and Torch CUDA 12.8.

Command:

```powershell
conda run -n witwin2 python -m tests.benchmark_raydtorch_native --grid 192 --queries 65536
```

Current RayDTorch-native result:

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
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_rayd_vs_raydtorch --grid 64 --queries 4096 --warmup 5 --repeat 30 --dynamic
```

The command above is a fast RayD/RayDTorch multipath regression shape. It is
not the only acceptance shape: it casts only 4,096 rays. For RayD latest-style
intersection pressure and Mitsuba comparison, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydtorch_rayd_mitsuba_stress `
  --rayd-source local --rayd-root E:\Code\RayDi `
  --scenario rayd-latest:64:128 `
  --scenario release:192:256 `
  --repeats 5 --warmup 2 --mitsuba-preliminary
```

This stress script matches RayD's `mesh_resolution/ray_grid_side` convention.
`rayd-latest:64:128` casts 16,384 rays, while `release:192:256` casts 65,536
rays. It reports RayDTorch, RayD, and Mitsuba static/dynamic intersection
performance for full materialized fields and reduced t-only paths; Mitsuba
`ray_intersect_preliminary` is reported as an extra Mitsuba-only lower-level
baseline when `--mitsuba-preliminary` is set.

For scaling sweeps instead of a few fixed sizes, use:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.benchmark_raydtorch_rayd_mitsuba_sweep `
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

- In `artifacts/benchmarks/scaling/extreme/sweep.json`, RayD is faster than
  RayDTorch in most static `full` intersection points. At 2.10M triangles and
  100.66M requested rays, static full is RayDTorch 0.3991 ms/batch, RayD
  0.3723 ms/batch, and Mitsuba public full 0.5856 ms/batch.
- In the same extreme static t-only/reduced case, RayDTorch is faster than RayD
  and Mitsuba public minimal: RayDTorch 0.1840 ms/batch, RayD 0.1999 ms/batch,
  Mitsuba public reduced 0.2583 ms/batch. Mitsuba preliminary is a lower-level
  baseline at 0.1785 ms/batch.
- Dynamic Mitsuba numbers are recorded, but they are not the primary comparison:
  Mitsuba dynamic scene updates perform additional work not directly comparable
  to RayD/RayDTorch.

AD backward is measured with `--include-backward`; plots are absolute projected
time grouped by backend, not throughput or speedup plots:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -B -m tests.benchmark_raydtorch_rayd_mitsuba_sweep `
  --preset smoke --mesh-resolution 64 --mesh-resolution 128 --mesh-resolution 256 `
  --total-rays 16384 --total-rays 65536 --ray-batch-side 256 `
  --repeats 5 --warmup 3 --rayd-source local --rayd-root E:\Code\RayDi `
  --mitsuba-preliminary --include-backward `
  --output-dir artifacts\benchmarks\scaling\ad_uv_tape
```

High-repeat static AD point, 131K triangles / 65.5K rays:

| Mode | RayDTorch ms | RayD ms | Status |
|---|---:|---:|---|
| forward full | 0.0643 | 0.1318 | RayDTorch faster |
| forward reduced | 0.0504 | 0.1332 | RayDTorch faster |
| backward `t_sum_full` | 0.5326 | 0.1915 | RayD faster |
| backward `t_sum_reduced` | 0.4936 | 0.2065 | RayD faster |

This is the main remaining RayD advantage found so far: static differentiable
intersection backward, even after RayDTorch's public `RayFlags.None` AD path was
added and hidden tape was reduced to `(global_prim_id, u, v, t)`.

Current same-script static-vs-static result:

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

Current same-script dynamic-vs-dynamic result:

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

An isolated RayDTorch-only microbenchmark of the no-AD reflection trace path on
the same grid measured:

```json
{
  "trace_reflections_forward_noad_ms": 0.291,
  "trace_reflections_forward_ad_outputs_ms": 0.461,
  "scene_trace_reflections_python_ms": 0.319
}
```

Current interpretation for this benchmark shape:

- RayDTorch `intersect` is faster in the latest static and dynamic same-script
  runs.
- RayDTorch `diffraction_direct` is faster in the latest static and dynamic
  same-script runs.
- RayDTorch point `nearest_edge` is faster in the latest static and dynamic
  same-script runs after the no-AD Torch path removed unnecessary tape
  allocation/writes for non-AD callers.
- RayDTorch scene build is faster in the latest static and dynamic same-script
  runs.
- RayDTorch reflection trace is faster in the latest static and dynamic
  same-script runs.
- Keep release-size and Nsight-backed runs as the broader performance gate;
  this document records parity for the covered benchmark, not universal
  superiority for every multipath/diffraction workload.

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

Visibility returns a discrete bool and has no continuous gradient contract.

## Multipath Implementation Status

RayDTorch now contains source ports for the RayD reflection and diffraction
`src/multipath` execution paths, including segment visibility, reflection trace,
reflection EPC, EPC field, reflection dedup, reflection accumulation,
diffraction path search, diffraction accumulation, chain accumulation, suffix
reflection, and coherent direct accumulation. Performance remains the active
completion risk.

See `docs/raydtorch_native_gap_analysis.md` for the tracked gap list.
