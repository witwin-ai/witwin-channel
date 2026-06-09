# Channel Native SF Radiomap Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `witwin.channel_native` radiomap performance competitive with, and then faster than, Sionna on large San Francisco planar and terrain radiomap workloads.

**Architecture:** Replace the current all-samples staged reflection accumulation path with adaptive accumulation and a Sionna-style streaming planar radiomap path. Then add compacted active-ray scheduling and terrain measurement-surface support so large scenes scale with useful contributions rather than total ray/depth slots.

**Tech Stack:** Python test harness, PyTorch extension dispatcher, native CUDA, OptiX, CUB, RayDN vendored extension, Sionna 2.0.1 reference benchmarks in `E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1`.

---

## Baseline Evidence

### Hardware And Environment

- GPU: NVIDIA GeForce RTX 5080, 16303 MiB, driver 596.49.
- Python: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`.
- Nsight Compute is installed at `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.bat`.
- Nsight Compute counters currently fail with `ERR_NVGPUCTRPERM`; enable NVIDIA GPU performance counters before relying on SM, DRAM, occupancy, or stall metrics.

### Same-Problem San Francisco Planar 2D Grid Benchmark

Config:

- Scene: Sionna San Francisco XML.
- Transmitter: `[468, 106, 70]`.
- Bounds: `x=[-520,720]`, `y=[-480,470]`.
- Plane: `z=1.5`.
- Grid: `256x256`.
- Samples: `100,000,000`.
- Components: LoS + specular reflection.
- `max_depth=1`.
- `diffraction=False`, `refraction=False`, `diffuse_reflection=False`.

Observed:

- Sionna planar 2D grid median: `25.09 ms`.
- Channel Native planar 2D grid median: `19,450.69 ms`.
- Current Native is about `775x` slower.

Native 100M reflection breakdown:

- sample directions: `39.16 ms`.
- launch inputs: `32.23 ms`.
- `reflection_accumulation_forward`: `23,866.87 ms`.
- store: `0.19 ms`.
- finalize: `0.09 ms`.
- peak allocation: `37.75 GB`.

Root cause:

- `reflection_accumulation_forward_op()` enables staged accumulation when sample count is high.
- For `100M` rays and `max_depth=1`, it stages `200M` ray/depth slots.
- Each staged value stores `cell + ReflAccumStagedValue`.
- The path then runs CUB radix sort and reduce-by-key over all staged slots.
- The workload is sparse, so this sorts mostly invalid or useless entries.

### Latest Streaming-Planar Evidence

Current optimized Native path:

- Adds `streaming_planar`, which generates the Fibonacci rays inside the OptiX raygen and avoids the 100M ray-direction tensor.
- Avoids staged sort/reduce for the SF planar workload.
- Treats the planar measurement surface like Sionna: analytical plane hit before the scene blocker and transparent continuation after measurement-plane hits.
- Skips unused complex field-map atomics for streaming radiomap-only output.
- Uses a power-only fast path when no field maps or staged values are requested.

Measured on the same SF planar 2D grid 100M problem:

- Earlier stable warm run: Sionna `23.49 ms`, Native `23.91 ms`.
- After power-only fast path, current GPU clock/load state was slower for both backends: Sionna-only median `29.73 ms`, Native-only median `28.06 ms`.
- Same-run both comparison after the fast path: Sionna `30.26 ms`, Native `29.76 ms`.
- Native peak allocation is about `192 MB`, far below the previous `37.75 GB` staged path.

Sionna 25ms mechanism:

- `RadioMapSolver._shoot_and_bounce()` spawns source rays with `fibonacci_lattice`, initializes antenna fields, traces the scene, intersects the planar measurement rectangle, and scatters `PlanarRadioMap.add_paths()` in one Dr.Jit/Mitsuba fused loop.
- For planar path gain, `PlanarRadioMap.add_paths()` only needs `squared_norm(e_field)` multiplied by `solid_angle / cos(theta)` and the normalization factor; it does not need to materialize a complex field map per cell.
- It keeps ray state resident and does not allocate per-sample direction, per-depth staged cell, or per-depth staged value arrays.

Remaining Native bottlenecks:

- Terminal reflected rays still use a full closest-hit OptiX trace with six payload registers for blocker visibility. A dedicated shadow/visibility trace with `TERMINATE_ON_FIRST_HIT | DISABLE_ANYHIT | DISABLE_CLOSESTHIT` is the next highest-value OptiX change.
- The raygen still carries both general accumulation state and streaming-planar state in one kernel. Specializing the `max_depth=1` streaming radiomap kernel should reduce live registers and branch pressure.
- Result parity is not solved: Sionna reports `142` nonzero cells and path-gain sum about `2.93e-10`; Native reports `95` nonzero cells and path-gain sum about `8.75e-11`. LoS is zero for both in this SF setup, so the mismatch is in first-order reflection geometry/material/field semantics, not in direct LoS.

Critical files:

- `src/witwin/channel_native/montecarlo/basic/raydn_components.py`
- `ext/raydn/src/torch_ext/reflection/ops.cpp`
- `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- `ext/raydn/src/torch_ext/reflection/accum_reduce.cu`
- `ext/raydn/include/raydn/reflection/accum_params.h`

---

## Acceptance Contract

1. Same-problem SF planar 2D grid benchmark reports Native faster than Sionna for 100M samples.
   - Target: Native median `< 25 ms`.
   - Stretch target: Native median `< 15 ms`.
2. Native 100M planar benchmark must not allocate more than 8 GB peak GPU memory for `max_depth=1`.
3. Native `max_depth=5` planar benchmark must not OOM on a 16 GB RTX 5080.
4. Result correctness must stay within documented Monte Carlo tolerance against the pre-change Native output and Sionna reference:
   - path gain sum relative tolerance: `0.35` for MC sampling differences.
   - nonzero cell count may differ, but visual support should cover the same dominant regions.
5. Existing tests continue to pass:
   - `python -m pytest tests/kernels/test_ops_facade.py tests/montecarlo/basic/test_basic_component_maps.py`
6. No solver hot path may import `drjit`, `mitsuba`, `sionna`, or original `witwin.channel`.
7. New benchmark scripts must use explicit GPU synchronization or CUDA events; enqueue-only timing is invalid.

---

## File Structure

Create:

- `benchmarks/bench_sf_planar_radiomap.py`  
  Repeatable same-problem Sionna vs Native planar benchmark with JSON output.

- `benchmarks/bench_native_reflection_accumulation.py`  
  Native-only component and kernel-stage benchmark for sample scaling, max-depth scaling, and accumulation-mode comparison.

- `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`  
  Unit and integration tests for strategy selection, memory-bounded execution, and parity on small deterministic scenes.

Modify:

- `src/witwin/channel_native/montecarlo/basic/config.py`  
  Add accumulation strategy controls and diagnostics flags.

- `src/witwin/channel_native/montecarlo/basic/raydn_components.py`  
  Pass strategy controls to RayDN reflection accumulation.

- `src/witwin/channel_native/montecarlo/basic/result.py`  
  Add optional performance metadata needed by benchmark and tests.

- `src/witwin/channel_native/montecarlo/basic/solver.py`  
  Surface reflection strategy metadata and keep component behavior stable.

- `ext/raydn/include/raydn/reflection/accum_params.h`  
  Add strategy enum fields and compacted-contribution metadata.

- `ext/raydn/src/torch_ext/reflection/ops.cpp`  
  Replace sample-count-only staging trigger with adaptive strategy selection and new compacted path bindings.

- `ext/raydn/src/torch_ext/reflection/accum_optix.cu`  
  Add direct atomic mode, compact-valid mode, and streaming planar mode.

- `ext/raydn/src/torch_ext/reflection/accum_reduce.cu`  
  Add compacted reduce-by-key path that sorts only valid contributions.

- `docs/dev/plans/02-channel-native-sf-radiomap-performance-plan.md`  
  Track implementation progress.

---

## Task 1: Lock The Benchmark Contract

**Files:**

- Create: `benchmarks/bench_sf_planar_radiomap.py`
- Create: `benchmarks/bench_native_reflection_accumulation.py`

- [ ] **Step 1: Create the same-problem planar benchmark**

Create `benchmarks/bench_sf_planar_radiomap.py` with these command-line options:

```python
parser.add_argument("--samples", type=int, default=100_000_000)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--backend", choices=("sionna", "native", "both"), default="both")
parser.add_argument("--json", type=Path, default=Path("artifacts/sf_planar_radiomap_benchmark.json"))
```

The benchmark must use:

```python
TX = (468.0, 106.0, 70.0)
BOUNDS = ((-520.0, 720.0), (-480.0, 470.0))
GRID = (256, 256)
PLANE_Z = 1.5
FREQUENCY = 3.5e9
MAX_DEPTH = 1
COMPONENTS = {"los", "reflection"}
```

- [ ] **Step 2: Time Sionna correctly**

Use `dr.eval(rm.path_gain)` and `dr.sync_thread()` inside the timed region:

```python
start = time.perf_counter()
rm = solver(scene, **kwargs)
dr.eval(rm.path_gain)
dr.sync_thread()
path_gain = np.asarray(rm.path_gain, dtype=np.float64)
elapsed_ms = (time.perf_counter() - start) * 1000.0
```

Expected JSON fields:

```json
{
  "backend": "sionna",
  "samples": 100000000,
  "median_ms": 25.0,
  "times_ms": [],
  "path_gain_sum": [],
  "nonzero": [],
  "shape": [1, 256, 256]
}
```

- [ ] **Step 3: Time Native correctly**

Use CUDA events around `solve(scene, config)` and synchronize before reading tensors:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result = solve(scene, config)
end.record()
torch.cuda.synchronize()
elapsed_ms = float(start.elapsed_time(end))
```

- [ ] **Step 4: Create Native reflection-stage benchmark**

Create `benchmarks/bench_native_reflection_accumulation.py` that measures:

- `_sample_directions`
- `mc_reflection_launch_inputs`
- `torch.ops.raydn.reflection_accumulation_forward`
- `mc_store_scaled_component_map`
- `mc_finalize_component_maps`

It must report sample counts `1M`, `10M`, `100M` and max depths `1`, `3`, `5`.

- [ ] **Step 5: Run the baseline**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_planar_radiomap.py --backend both --samples 100000000 --repeats 5 --json artifacts\sf_planar_baseline.json
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_native_reflection_accumulation.py --samples 1000000 10000000 100000000 --max-depths 1 3 5 --json artifacts\native_reflection_baseline.json
```

Expected:

- Sionna median near `25 ms`.
- Native current median near `19 s` for 100M, max-depth 1.
- Native max-depth 5 either OOMs or reports failure cleanly.

- [ ] **Step 6: Commit benchmark harness**

```powershell
git add benchmarks\bench_sf_planar_radiomap.py benchmarks\bench_native_reflection_accumulation.py
git commit -m "Add SF planar radiomap performance benchmarks"
```

---

## Task 2: Add Accumulation Strategy Controls

**Files:**

- Modify: `src/witwin/channel_native/montecarlo/basic/config.py`
- Modify: `src/witwin/channel_native/montecarlo/basic/raydn_components.py`
- Modify: `src/witwin/channel_native/montecarlo/basic/metadata.py`
- Test: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`

- [ ] **Step 1: Add failing tests for strategy validation**

Create `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`:

```python
import pytest

from witwin.channel_native.montecarlo.basic import Config


def test_reflection_accumulation_strategy_accepts_known_values():
    for strategy in ("auto", "atomic", "staged", "compact"):
        config = Config(reflection_accumulation_strategy=strategy)
        assert config.reflection_accumulation_strategy == strategy


def test_reflection_accumulation_strategy_rejects_unknown_value():
    with pytest.raises(ValueError, match="reflection_accumulation_strategy"):
        Config(reflection_accumulation_strategy="bad")
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\montecarlo\basic\test_reflection_accumulation_strategy.py -q
```

Expected: fails because `Config` has no `reflection_accumulation_strategy`.

- [ ] **Step 3: Add config fields**

Modify `Config`:

```python
_VALID_REFLECTION_ACCUMULATION_STRATEGIES = frozenset({"auto", "atomic", "staged", "compact"})

reflection_accumulation_strategy: str = "auto"
reflection_compact_min_samples: int = 262_144
reflection_staged_min_samples_per_cell: int = 64
```

Validation:

```python
if self.reflection_accumulation_strategy not in _VALID_REFLECTION_ACCUMULATION_STRATEGIES:
    raise ValueError("reflection_accumulation_strategy is not supported")
if self.reflection_compact_min_samples < 0:
    raise ValueError("reflection_compact_min_samples must be non-negative")
if self.reflection_staged_min_samples_per_cell < 0:
    raise ValueError("reflection_staged_min_samples_per_cell must be non-negative")
```

- [ ] **Step 4: Thread strategy into `raydn_components.py`**

Change `reflection_component_maps_with_wedges()` signature:

```python
reflection_accumulation_strategy: str = "auto",
reflection_compact_min_samples: int = 262_144,
reflection_staged_min_samples_per_cell: int = 64,
```

Pass these values from `solver.py`.

- [ ] **Step 5: Add metadata**

Modify `metadata.py` to include:

```python
"reflection_accumulation_strategy": config.reflection_accumulation_strategy,
"reflection_compact_min_samples": config.reflection_compact_min_samples,
"reflection_staged_min_samples_per_cell": config.reflection_staged_min_samples_per_cell,
```

- [ ] **Step 6: Run tests**

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\montecarlo\basic\test_reflection_accumulation_strategy.py tests\montecarlo\basic\test_basic_component_maps.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src\witwin\channel_native\montecarlo\basic tests\montecarlo\basic\test_reflection_accumulation_strategy.py
git commit -m "Add reflection accumulation strategy controls"
```

---

## Task 3: Stop Staged Sort/Reduce From Triggering On Sparse SF Planar Workloads

**Files:**

- Modify: `ext/raydn/include/raydn/reflection/accum_params.h`
- Modify: `ext/raydn/src/torch_ext/reflection/ops.cpp`
- Modify: `src/witwin/channel_native/montecarlo/basic/raydn_components.py`
- Test: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`

- [ ] **Step 1: Add strategy enum in RayDN C++**

Add to `accum_params.h`:

```cpp
enum ReflAccumStrategy : int {
    RAYDN_REFL_ACCUM_AUTO = 0,
    RAYDN_REFL_ACCUM_ATOMIC = 1,
    RAYDN_REFL_ACCUM_STAGED = 2,
    RAYDN_REFL_ACCUM_COMPACT = 3,
};
```

Add fields:

```cpp
int accumulation_strategy;
int compact_min_samples;
int staged_min_samples_per_cell;
```

- [ ] **Step 2: Extend `reflection_accumulation_forward_op` signature**

Add parameters after `wedge_sample_stride`:

```cpp
int64_t accumulation_strategy,
int64_t compact_min_samples,
int64_t staged_min_samples_per_cell
```

Map Python strings in `raydn_components.py`:

```python
strategy_id = {
    "auto": 0,
    "atomic": 1,
    "staged": 2,
    "compact": 3,
}[reflection_accumulation_strategy]
```

- [ ] **Step 3: Replace current staged trigger**

Current trigger:

```cpp
const bool staged_accum =
    stage_sample_count_fits &&
    stage_sample_count >= kStagedReflAccumMinSamples &&
    stage_sample_count >= cell_count * kStagedReflAccumMinSamplesPerCell;
```

Replace with:

```cpp
const bool force_staged =
    accumulation_strategy == RAYDN_REFL_ACCUM_STAGED;
const bool force_atomic =
    accumulation_strategy == RAYDN_REFL_ACCUM_ATOMIC;
const bool auto_staged =
    accumulation_strategy == RAYDN_REFL_ACCUM_AUTO &&
    stage_sample_count_fits &&
    stage_sample_count >= kStagedReflAccumMinSamples &&
    stage_sample_count >= cell_count * staged_min_samples_per_cell &&
    max_bounces_i <= 1 &&
    ray_count <= 10'000'000;
const bool staged_accum =
    !force_atomic &&
    stage_sample_count_fits &&
    (force_staged || auto_staged);
```

This intentionally prevents the 100M SF planar workload from entering the all-slot sort path.

- [ ] **Step 4: Add test for strategy metadata**

Add:

```python
def test_reflection_accumulation_strategy_is_reported():
    config = Config(reflection_accumulation_strategy="atomic")
    assert config.reflection_accumulation_strategy == "atomic"
```

- [ ] **Step 5: Build and test**

Run the repo build command used for the current environment, then:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\montecarlo\basic\test_reflection_accumulation_strategy.py tests\montecarlo\basic\test_basic_component_maps.py -q
```

- [ ] **Step 6: Benchmark atomic strategy**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_planar_radiomap.py --backend native --samples 100000000 --repeats 5 --json artifacts\sf_planar_native_atomic.json
```

Expected:

- No OOM.
- Peak allocation below 8 GB.
- Median substantially below `19,450 ms`.

- [ ] **Step 7: Commit**

```powershell
git add ext\raydn src\witwin\channel_native\montecarlo\basic tests\montecarlo\basic\test_reflection_accumulation_strategy.py
git commit -m "Avoid staged reflection accumulation for sparse large grids"
```

---

## Task 4: Implement Compact-Valid Reflection Accumulation

**Files:**

- Modify: `ext/raydn/include/raydn/reflection/accum_params.h`
- Modify: `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- Modify: `ext/raydn/src/torch_ext/reflection/accum_reduce.cu`
- Modify: `ext/raydn/src/torch_ext/reflection/ops.cpp`
- Test: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`

- [ ] **Step 1: Add compact buffers**

Add params:

```cpp
int *compact_count;
int *compact_cell;
ReflAccumStagedValue *compact_value;
int compact_capacity;
```

- [ ] **Step 2: Add compact write path in `accumulate_plane()`**

After contribution validation, before staged path:

```cpp
if (params.compact_cell != nullptr && params.compact_value != nullptr &&
    params.compact_count != nullptr) {
    const int slot = atomicAdd(params.compact_count, 1);
    if (slot < params.compact_capacity) {
        params.compact_cell[slot] = cell;
        ReflAccumStagedValue value;
        value.a = make_float4(contribution_power,
                              contribution_field.x.r,
                              contribution_field.x.i,
                              contribution_field.y.r);
        value.b = make_float4(contribution_field.y.i,
                              contribution_field.z.r,
                              contribution_field.z.i,
                              1.0f);
        params.compact_value[slot] = value;
    }
    return true;
}
```

- [ ] **Step 3: Add reduce function for compacted values**

In `accum_reduce.cu`, add:

```cpp
void reduce_refl_accum_compacted_cuda(
    const at::Tensor &compact_count,
    const at::Tensor &compact_cell,
    const at::Tensor &compact_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_reflection_count);
```

Implementation may initially sort `compact_capacity` entries, but must only emit valid entries. The next task removes capacity waste.

- [ ] **Step 4: Allocate compact buffers**

In `ops.cpp`, for compact mode:

```cpp
at::Tensor compact_count = at::zeros({1}, iopts);
at::Tensor compact_cell = at::full({ray_count}, -1, iopts);
at::Tensor compact_value = at::zeros({ray_count, 8}, fopts);
```

For `max_depth > 1`, use `ray_count * (max_bounces + 1)` until active compaction lands.

- [ ] **Step 5: Use compact mode for auto strategy**

Set auto strategy:

```cpp
const bool compact_accum =
    accumulation_strategy == RAYDN_REFL_ACCUM_COMPACT ||
    (accumulation_strategy == RAYDN_REFL_ACCUM_AUTO &&
     ray_count >= compact_min_samples &&
     !staged_accum);
```

- [ ] **Step 6: Benchmark compact mode**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_native_reflection_accumulation.py --strategy compact --samples 1000000 10000000 100000000 --max-depths 1 --json artifacts\native_reflection_compact.json
```

Expected:

- 100M compact mode must beat the old staged path.
- Peak allocation must be lower than old `37.75 GB`.

- [ ] **Step 7: Commit**

```powershell
git add ext\raydn src\witwin\channel_native\montecarlo\basic tests\montecarlo\basic\test_reflection_accumulation_strategy.py
git commit -m "Add compact-valid reflection accumulation path"
```

---

## Task 5: Implement Sionna-Style Streaming Planar Radiomap Kernel

**Files:**

- Modify: `ext/raydn/include/raydn/reflection/accum_params.h`
- Modify: `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- Modify: `ext/raydn/src/torch_ext/reflection/ops.cpp`
- Modify: `src/witwin/channel_native/montecarlo/basic/raydn_components.py`
- Test: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`

- [ ] **Step 1: Add streaming strategy**

Extend strategy enum:

```cpp
RAYDN_REFL_ACCUM_STREAMING_PLANAR = 4
```

Extend Python validation with `"streaming_planar"`.

- [ ] **Step 2: Add measurement-plane-before-scene logic**

In OptiX raygen, at each depth:

1. Trace scene and get `si_scene`.
2. Intersect current ray with the measurement plane analytically.
3. If plane hit is valid and `t_plane < t_scene`, accumulate immediately and keep ray state unchanged for the measurement-plane interaction.
4. If scene hit is valid and before plane, update reflection state and continue.

The core branch should mirror Sionna logic:

```cpp
const bool mp_int = plane_hit.valid && plane_hit.t < blocker_t;
if (mp_int && (depth > 0 || params.los_enabled != 0)) {
    accumulate_plane_hit(ray_index, depth, plane_hit, image_source, field);
}
if (hit.hit == 0u || mp_int || depth >= params.max_bounces) {
    break;
}
```

- [ ] **Step 3: Avoid all staging in streaming mode**

Streaming mode must write directly to output using warp-aggregated atomics or block-local bins. It must not allocate `stage_cell`, `stage_value`, `sorted_values`, or `reduced_values`.

- [ ] **Step 4: Add a parity test on a small wall scene**

Test:

```python
def test_streaming_planar_matches_atomic_small_scene():
    atomic = solve(scene, Config(samples=8192, components={"reflection"}, reflection_accumulation_strategy="atomic"))
    streaming = solve(scene, Config(samples=8192, components={"reflection"}, reflection_accumulation_strategy="streaming_planar"))
    torch.testing.assert_close(streaming.path_gain, atomic.path_gain, rtol=2e-2, atol=1e-12)
```

- [ ] **Step 5: Benchmark streaming planar**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_planar_radiomap.py --backend both --samples 100000000 --repeats 5 --json artifacts\sf_planar_streaming_planar.json
```

Expected:

- Native 100M median below Sionna median.
- Peak allocation below 4 GB.

- [ ] **Step 6: Commit**

```powershell
git add ext\raydn src\witwin\channel_native\montecarlo\basic tests\montecarlo\basic\test_reflection_accumulation_strategy.py
git commit -m "Add streaming planar reflection radiomap kernel"
```

---

## Task 6: Add Active-Ray Compaction For Multi-Bounce Workloads

**Files:**

- Modify: `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- Modify: `ext/raydn/src/torch_ext/reflection/ops.cpp`
- Create: `ext/raydn/src/torch_ext/reflection/active_queue.cu`
- Create: `ext/raydn/include/raydn/reflection/active_queue.h`
- Test: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`

- [ ] **Step 1: Add active ray state struct**

Create `active_queue.h`:

```cpp
struct ReflActiveRay {
    float3 origin;
    float3 direction;
    float3 image_source;
    float3 field_x;
    int ray_index;
    int depth;
};
```

- [ ] **Step 2: Add compact kernel**

Create `active_queue.cu` with a prefix-sum based compaction function:

```cpp
void compact_reflection_active_rays_cuda(
    const at::Tensor &active_flags,
    const at::Tensor &in_state,
    at::Tensor &out_state,
    at::Tensor &out_count);
```

- [ ] **Step 3: Convert multi-bounce loop to active queues**

For `max_depth > 1`, process:

```text
depth 0 queue -> trace -> accumulate -> emit next queue
depth 1 queue -> trace -> accumulate -> emit next queue
depth N queue -> trace -> accumulate -> emit next queue until N == max_depth or queue is empty
```

Do not launch later depths over the original `ray_count`.

- [ ] **Step 4: Benchmark max-depth 5**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_native_reflection_accumulation.py --strategy streaming_planar --samples 100000000 --max-depths 5 --json artifacts\native_reflection_depth5_active_queue.json
```

Expected:

- No OOM on RTX 5080 16 GB.
- Runtime scales with active rays, not `samples * (max_depth + 1)` slots.

- [ ] **Step 5: Commit**

```powershell
git add ext\raydn tests\montecarlo\basic\test_reflection_accumulation_strategy.py
git commit -m "Compact active rays for multi-bounce reflection"
```

---

## Task 7: Add Terrain Measurement Surface Support

**Files:**

- Modify: `src/witwin/channel_native/core/objects.py`
- Modify: `src/witwin/channel_native/core/scene.py`
- Modify: `src/witwin/channel_native/montecarlo/basic/raydn_components.py`
- Modify: `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- Test: `tests/montecarlo/basic/test_terrain_measurement_surface.py`

- [ ] **Step 1: Add public measurement surface object**

Add:

```python
@dataclass(frozen=True, slots=True)
class ReceiverMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    name: str = ""
```

Validation mirrors `Structure`.

- [ ] **Step 2: Add loader support for Sionna Terrain**

Add helper:

```python
Scene.load_mitsuba_receiver_mesh(xml, object_name="Terrain", z_offset=1.5)
```

It returns a `ReceiverMesh` with transformed vertices.

- [ ] **Step 3: Add mesh measurement branch**

When the receiver is `ReceiverMesh`, use the measurement mesh as part of the OptiX scene and detect hits where `si_scene.shape == measurement_surface`.

- [ ] **Step 4: Add Sionna canonical SF terrain benchmark mode**

Extend `benchmarks/bench_sf_planar_radiomap.py` or create `benchmarks/bench_sf_terrain_radiomap.py` with:

```python
measurement_surface = scene.objects["Terrain"].clone(as_mesh=True)
transform_mesh(measurement_surface, translation=[0.0, 0.0, 1.5])
samples_per_tx = 100_000_000
max_depth = 5
```

- [ ] **Step 5: Benchmark terrain**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_terrain_radiomap.py --backend both --samples 100000000 --repeats 5 --json artifacts\sf_terrain_native_vs_sionna.json
```

Expected:

- Native completes without OOM.
- Native result covers the same dominant terrain regions as Sionna.
- Native median is faster than Sionna canonical terrain median.

- [ ] **Step 6: Commit**

```powershell
git add src\witwin\channel_native ext\raydn tests\montecarlo\basic\test_terrain_measurement_surface.py benchmarks
git commit -m "Add terrain measurement surface radiomap support"
```

---

## Task 8: Profile And Tune OptiX Raygen

**Files:**

- Modify: `ext/raydn/src/torch_ext/reflection/accum_optix.cu`
- Modify: `ext/raydn/src/torch_ext/reflection/pipeline.cpp`
- Modify: `ext/raydn/src/torch_ext/common/optix_pipeline.cpp`

- [ ] **Step 1: Enable GPU performance counters**

On Windows, enable NVIDIA performance counters:

```text
NVIDIA Control Panel -> Desktop -> Enable Developer Settings
NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters -> Allow access to all users
```

Reboot if required.

- [ ] **Step 2: Capture NCU**

Run:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.bat' --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers -k regex:.*reflection.* C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_native_reflection_accumulation.py --strategy streaming_planar --samples 100000000 --max-depths 1 --repeats 1
```

- [ ] **Step 3: Tune payload and registers**

If register pressure limits occupancy:

- Pack hit payload to fewer registers.
- Shorten live ranges for `field`, `image_source`, and material coefficients.
- Split material coefficient math into a helper used only after a scene hit.
- Add `__launch_bounds__` only after measuring nearby register/occupancy tradeoffs.

- [ ] **Step 4: Tune trace flags**

Verify that nearest-hit traces keep closest-hit semantics. For pure visibility or shadow tests, use:

```cpp
OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
OPTIX_RAY_FLAG_DISABLE_ANYHIT |
OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
```

- [ ] **Step 5: Benchmark after each change**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_planar_radiomap.py --backend both --samples 100000000 --repeats 5 --json artifacts\sf_planar_after_optix_tuning.json
```

- [ ] **Step 6: Commit**

```powershell
git add ext\raydn
git commit -m "Tune OptiX reflection radiomap raygen"
```

---

## Task 9: Final Acceptance And Regression Gates

**Files:**

- Modify: `tests/montecarlo/basic/test_basic_component_maps.py`
- Modify: `tests/montecarlo/basic/test_reflection_accumulation_strategy.py`
- Create: `tests/performance/test_sf_planar_radiomap_budget.py`

- [ ] **Step 1: Add performance budget test**

Create a skipped-by-default performance test:

```python
import os
import pytest

from benchmarks.bench_sf_planar_radiomap import run_native_planar_benchmark


@pytest.mark.skipif(os.environ.get("WITWIN_PERF_TESTS") != "1", reason="performance test")
def test_sf_planar_100m_under_sionna_budget():
    result = run_native_planar_benchmark(samples=100_000_000, repeats=5)
    assert result["median_ms"] < 25.0
    assert result["peak_allocated_bytes"] < 8 * 1024**3
```

- [ ] **Step 2: Run functional tests**

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\kernels\test_ops_facade.py tests\montecarlo\basic\test_basic_component_maps.py tests\montecarlo\basic\test_reflection_accumulation_strategy.py -q
```

- [ ] **Step 3: Run performance tests**

```powershell
$env:WITWIN_PERF_TESTS='1'
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\performance\test_sf_planar_radiomap_budget.py -q
```

- [ ] **Step 4: Generate final benchmark artifacts**

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_planar_radiomap.py --backend both --samples 100000000 --repeats 10 --json artifacts\sf_planar_final_native_vs_sionna.json
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe benchmarks\bench_sf_terrain_radiomap.py --backend both --samples 100000000 --repeats 5 --json artifacts\sf_terrain_final_native_vs_sionna.json
```

- [ ] **Step 5: Commit final gates**

```powershell
git add tests benchmarks docs\dev\plans\02-channel-native-sf-radiomap-performance-plan.md
git commit -m "Add SF radiomap performance acceptance gates"
```

---

## Risk Register

1. **Monte Carlo output changes with accumulation order.**  
   Use relative tolerance for sums and visual support comparisons. Do not require bitwise parity.

2. **Atomic mode may become slow in dense scenes.**  
   Keep `staged` and `compact` strategies available. Use adaptive selection based on measured valid contribution rate.

3. **Compaction can add overhead for small workloads.**  
   Keep small sample counts on direct atomic path.

4. **Sionna and Native may not use identical electromagnetic contracts.**  
   Keep performance and numeric parity reports separate. Compare same config, but do not claim exact physical equivalence unless validated.

5. **Nsight counters may be unavailable.**  
   If `ERR_NVGPUCTRPERM` persists, report CUDA event breakdown and allocator peak memory. Do not invent SM/DRAM counter data.

---

## Completion Criteria

- SF planar 2D grid 100M Native median is below Sionna median.
- SF planar 2D grid 100M Native peak memory is below 8 GB.
- SF planar max-depth 5 no longer OOMs.
- Native has a documented strategy selector for `atomic`, `compact`, `staged`, and `streaming_planar`.
- Terrain measurement surface support exists and can run the Sionna canonical San Francisco terrain benchmark.
- All functional tests pass in `witwin2`.
- Final benchmark JSON artifacts are generated under `artifacts/`.
