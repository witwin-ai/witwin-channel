# RayD OptiX Diffraction Kernel Implementation Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-22

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **RayD source location (READ FIRST).** Tasks that touch RayD must edit and build the local source tree at `E:\Code\RayDi`. Do not use the PyPI `rayd` package for implementation or verification. For native iteration, use the `witwin2` environment and CMake build/install flow described in `AGENTS.md`.

**Goal:** Move the hot diffraction paths used by Monte Carlo basic, Monte Carlo BDPT, and the standalone path solver from DrJit symbolic loops plus repeated visibility launches into RayD-owned CUDA/OptiX kernels.

**Architecture:** The core migration unit is a RayD-native diffraction state ABI shared by grid-accumulation solvers and path-export solvers. Channel remains responsible for public configuration, result assembly, metadata, backend dispatch, and strict unsupported-workload rejection; RayD owns scene traversal, visibility, UTD evaluation inputs, per-sample path sampling, and grid/path output kernels. The first shipped path must be non-AD forward only and must raise on unsupported AD workloads rather than silently falling back.

**Tech Stack:** Python 3.11 in `witwin2`, DrJit CUDA arrays at runtime boundaries, RayD C++/CUDA/OptiX native extension, CMake/MSBuild on Windows, bundled channel tests under `tests/`, and RayD tests under `E:\Code\RayDi\tests`.

---

## Scope

This plan is about diffraction performance and solver integration. It does not change public scene construction, material APIs, receiver-grid APIs, or result payload shapes.

In scope:

- Monte Carlo basic first-order diffraction grid accumulation.
- Monte Carlo BDPT direct/Keller first-order grid accumulation.
- Monte Carlo BDPT reflection-prefix first-order grid accumulation using RayD reflection wedge prefix payloads.
- Monte Carlo BDPT higher-order edge-chain diffraction for order 2 and order 3.
- Standalone path solver diffraction candidate/path export.
- Optional deterministic/radiomap consumers through shared channel adapters after the Monte Carlo/path APIs are stable.

Out of scope for the first implementation:

- Native AD through diffraction kernels.
- Heuristic smoothing or soft visibility approximations.
- Torch/NumPy/DLPack bridges inside solver hot paths.
- CPU fallback implementations.
- Suffix reflection correctness changes before the dedicated suffix phase.

## Current Bottleneck Summary

The current BDPT diffraction cost comes from `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`: per-order Python dispatch, per-strategy symbolic loops, repeated visibility calls, per-sample UTD/MIS recomputation, and `dr.scatter_reduce` grid writes. Basic diffraction uses fewer families, but still evaluates sampled diffraction in DrJit and scatters through channel grid helpers. The path solver already uses RayD heavily for reflection discovery, but diffraction path export still needs a native candidate interface rather than grid accumulation.

RayD already has the relevant low-level ingredients:

- Native OptiX scene traversal.
- `visible`, `visible_pair`, `visible_chain`, and axial edge visibility kernels.
- Native reflection accumulation with wedge event output.
- The new reflection wedge prefix payloads committed in RayD `659fe08`.

The missing piece is a RayD-owned diffraction kernel family and a channel adapter layer that can feed it from existing `DiffractionStates` / path solver state.

## New RayD ABI

### `DiffractionGrid`

Use a struct equivalent to the existing `AccumGrid` unless RayD can reuse `AccumGrid` directly.

Fields:

```cpp
struct DiffractionGrid {
    int axis;
    float position;
    float coord0_min;
    float coord0_max;
    float coord1_min;
    float coord1_max;
    int resolution0;
    int resolution1;
    float cell_area;
};
```

### `DiffractionMaterial`

Reuse the existing RayD material payload shape when possible.

Fields:

```cpp
template <typename Float_>
struct DiffractionMaterialData {
    Float_ eta_r;
    Float_ sigma;
    Float_ mu_r;
    Float_ gain;
    MaskT<Float_> valid;
};
```

### `DiffractionStateTable`

This is the central ABI. It represents sampled diffraction states, not only geometric edges, so Monte Carlo basic, BDPT direct states, and BDPT prefix states can share one kernel.

Fields:

```cpp
template <typename Float_>
struct DiffractionStateTableData {
    int count;
    IntT<Float_> edge_index;
    Vec3T<Float_> edge_pos;
    Vec3T<Float_> edge_dir;
    Float_ edge_line_min;
    Float_ edge_line_max;
    Vec3T<Float_> face0_normal;
    Vec3T<Float_> face1_normal;
    IntT<Float_> face0_prim_id;
    IntT<Float_> face1_prim_id;
    Float_ exterior_angle;
    Vec3T<Float_> source_pos;
    Float_ source_power;
    Vec3T<Float_> incident_direction;
    Vec3T<Float_> initial_direction;
    IntT<Float_> prefix_reflection_depth;
};
```

Notes:

- `source_pos/source_power/initial_direction/prefix_reflection_depth` are required for RayD reflection-prefix diffraction.
- `incident_direction` should be the direction arriving at the diffraction state.
- `face*_prim_id` allows native material lookup without per-sample channel gathers.
- The initial ABI should not include `prefix_prim_by_bounce`; suffix reflection and AD replay can add it later if truly needed.

### `DiffractionAccumOptions`

Fields:

```cpp
struct DiffractionAccumOptions {
    float wavelength;
    float k;
    int seed;
    int samples;
    int max_order;
    int direct_samples;
    int keller_samples;
    int suffix_samples;
    int strategy_mask;
    int sample_sequence;
    int receiver_model;
    int collect_edge_use;
    int collect_debug_counts;
};
```

Constants:

```cpp
enum DiffractionStrategyMask {
    RAYD_DIFF_DIRECT = 1 << 0,
    RAYD_DIFF_KELLER = 1 << 1,
    RAYD_DIFF_SUFFIX_REFLECTION = 1 << 2,
};

enum DiffractionSampleSequence {
    RAYD_DIFF_HASH = 0,
    RAYD_DIFF_SOBOL = 1,
};

enum DiffractionReceiverModel {
    RAYD_DIFF_MATCHED_ISOTROPIC = 0,
};
```

### `DiffractionAccumResult`

Fields:

```cpp
template <typename Float_>
struct DiffractionAccumResultData {
    int grid_cell_count;
    Float_ diffraction_power;
    ComplexT<Float_> diffraction_field_x;
    ComplexT<Float_> diffraction_field_y;
    ComplexT<Float_> diffraction_field_z;
    IntT<Float_> direct_count;
    IntT<Float_> keller_count;
    IntT<Float_> suffix_count;
    IntT<Float_> visibility_reject_count;
    IntT<Float_> utd_reject_count;
    IntT<Float_> edge_use_count;
};
```

The first version may set field components to zero if only incoherent matched-isotropic power is implemented. The struct should include them from the start to avoid ABI churn once coherent path support is added.

### `DiffractionPathOptions`

Path solver export needs a separate result shape because it emits compact paths instead of grid atomics.

Fields:

```cpp
struct DiffractionPathOptions {
    float wavelength;
    float k;
    int seed;
    int max_order;
    int max_paths;
    int max_receivers;
    int strategy_mask;
    int sample_count;
    int return_geometry;
    int receiver_model;
};
```

### `DiffractionPathResult`

Fields:

```cpp
template <typename Float_>
struct DiffractionPathResultData {
    IntT<Float_> count;
    IntT<Float_> tx_index;
    IntT<Float_> rx_index;
    IntT<Float_> order;
    IntT<Float_> edge_index_0;
    IntT<Float_> edge_index_1;
    IntT<Float_> edge_index_2;
    Float_ delay;
    ComplexT<Float_> field_x;
    ComplexT<Float_> field_y;
    ComplexT<Float_> field_z;
    Vec3T<Float_> point_0;
    Vec3T<Float_> point_1;
    Vec3T<Float_> point_2;
};
```

## New RayD Methods

### `Scene.accumulate_diffraction_order1`

Purpose:

Grid accumulation for first-order direct and Keller diffraction. This is the first performance target because it benefits Monte Carlo basic and BDPT.

Python binding:

```python
result = rayd_scene.accumulate_diffraction_order1(
    states,
    grid,
    material,
    options,
    active=True,
)
```

C++ signature:

```cpp
template <bool Detached>
DiffractionAccumResultT<Detached> Scene::accumulate_diffraction_order1(
    const DiffractionStateTableT<Detached>& states,
    const DiffractionGrid& grid,
    const DiffractionMaterialT<Detached>& material,
    const DiffractionAccumOptions& options,
    const MaskT<Detached>& active) const;
```

Implementation:

- One OptiX launch with one lane per sample.
- Sample state slot by edge-length CDF or precomputed CDF supplied by channel.
- Sample edge point.
- Direct strategy samples a receiver cell and checks source-to-edge and edge-to-cell visibility.
- Keller strategy samples cone angle and plane hit, then checks source and target visibility.
- Compute UTD power inside CUDA.
- Atomic add power and optional field into grid buffers.
- Record direct/Keller accepted counts and debug counters.

### `Scene.accumulate_prefix_diffraction_order1`

Purpose:

Bridge RayD reflection wedge prefix payloads into first-order diffraction without returning to DrJit reflection.

Python binding:

```python
result = rayd_scene.accumulate_prefix_diffraction_order1(
    wedge_events,
    edge_table,
    grid,
    material,
    options,
    active=True,
)
```

Implementation:

- Accept RayD `WedgeEvents` from `accumulate_reflections`.
- Convert each valid wedge event to a diffraction state on the native side or through a channel state adapter.
- Use the same accumulation kernel as `accumulate_diffraction_order1`.
- Preserve `source_points`, `source_power`, `initial_directions`, and `bounce_depth`.

First implementation choice:

- Prefer channel-side conversion from `WedgeEvents` to `DiffractionStateTable` because channel already has `best_edge_indices_from_hit_data`.
- Move best-edge discovery native only after parity is established.

### `Scene.accumulate_diffraction_chains`

Purpose:

Higher-order BDPT diffraction, order 2 and order 3.

Python binding:

```python
result = rayd_scene.accumulate_diffraction_chains(
    initial_states,
    recursive_states,
    grid,
    material,
    options,
    active=True,
)
```

Implementation:

- One launch per order or one launch with `options.max_order`.
- Sample full edge chain inside CUDA.
- Use OptiX trace calls in the same raygen program for inter-edge visibility.
- Accumulate only direct receiver-cell connection in the first version.
- Add Keller chain strategy after direct chain parity is established.

### `Scene.accumulate_diffraction_reflection_suffix`

Purpose:

Fix BDPT coupled suffix reflection as a dedicated phase, not as an incidental extension of the order-1 kernel.

Python binding:

```python
result = rayd_scene.accumulate_diffraction_reflection_suffix(
    initial_states,
    recursive_states,
    reflection_candidates,
    grid,
    material,
    options,
    active=True,
)
```

Implementation:

- Do not sample uniformly over all triangles.
- Input a compact candidate reflection-surface table.
- Use a fused visibility chain with primitive-ignore support inside native OptiX.
- Return suffix accepted count and reject counters separately.

### `Scene.trace_diffraction_paths`

Purpose:

Standalone path solver integration. This emits compact path candidates rather than grid maps.

Python binding:

```python
paths = rayd_scene.trace_diffraction_paths(
    tx_positions,
    rx_positions,
    edge_table,
    material,
    options,
    active=True,
)
```

Implementation:

- One launch over `(tx, rx, sample)` or a flattened candidate lane.
- Support first-order direct diffraction first.
- Support order 2 and order 3 after `accumulate_diffraction_chains` is stable.
- Output compact path fields, delays, edge indices, and optional geometry points.

## Channel Runtime Interfaces

### `witwin.channel.core.scene.Scene`

Modify `witwin/channel/core/scene/scene.py`:

```python
def accumulate_diffraction_order1(
    self,
    *,
    states,
    grid,
    config,
    samples: int,
    direct_samples: int,
    keller_samples: int,
    seed: int,
    active=True,
):
    ...

def accumulate_diffraction_chains(
    self,
    *,
    initial_states,
    recursive_states,
    grid,
    config,
    samples: int,
    max_order: int,
    seed: int,
    active=True,
):
    ...

def trace_diffraction_paths(
    self,
    *,
    tx_positions,
    rx_positions,
    config,
    max_order: int,
    max_paths: int,
    seed: int,
    return_geometry: bool,
    active=True,
):
    ...
```

These wrappers must:

- Reject AD inputs for RayD native diffraction.
- Convert channel grid/material/config into RayD structs.
- Convert channel `DiffractionStates` into RayD `DiffractionStateTable`.
- Preserve DrJit-native tensors at the boundary.
- Never introduce NumPy/Torch/DLPack in solver internals.

### Monte Carlo Config

Modify `witwin/channel/montecarlo/config.py`:

```python
AccumulatePrimalMode = Literal["auto", "drjit", "rayd_optix"]
```

Default:

- `auto` keeps current behavior until RayD parity gates pass.
- `rayd_optix` explicitly requests native diffraction accumulation.
- AD workloads with `rayd_optix` raise unless a native AD path exists.

Metadata:

```python
metadata["runtime_backends"]["diffraction"] = {
    "implementation": "rayd_diffraction_order1_accumulation",
    "cell_scatter_backend": "rayd_optix_atomic_add",
    "state_sampler": "...",
    "strategy_mask": ["direct", "keller"],
    "ad_contract": "explicit_non_ad_backend_raises_on_ad_inputs",
}
```

### Monte Carlo Basic Integration

Modify:

- `witwin/channel/montecarlo/integrators/basic.py`
- `witwin/channel/montecarlo/trace/diffraction.py`

Dispatch rule:

```python
use_rayd_diffraction = (
    not collect_ad_tapes
    and config.diffraction_execution.accumulate_primal == "rayd_optix"
    and n_diffraction_states > 0
    and max_diffraction_order == 1
)
```

Behavior:

- Keep existing reflection and wedge discovery.
- Build `DiffractionStateTable` from `DiffractionStates`.
- Call `scene.accumulate_diffraction_order1(...)`.
- Add returned power/field/counts to `weighted_diagnostics`.
- Explicit `rayd_optix` raises for AD tape collection instead of falling back.

### Monte Carlo BDPT Integration

Modify:

- `witwin/channel/montecarlo/integrators/bdpt.py`
- `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`

Dispatch stages:

1. `max_depth == 1`, direct/Keller only:
   - Call `scene.accumulate_diffraction_order1(initial_states, ...)`.
2. `reflection_coupled_diffraction == True`:
   - Raise for explicit `rayd_optix` until the native suffix-reflection kernel lands.
3. `max_depth > 1`, no suffix:
   - Call `scene.accumulate_diffraction_order1(...)` for order 1 and `scene.accumulate_diffraction_chains(...)` for order 2/3 direct plus Keller strategies.
4. suffix enabled:
   - Use DrJit only when `accumulate_primal != "rayd_optix"` until `accumulate_diffraction_reflection_suffix` lands.

Required guard:

```python
if accumulate_primal == "rayd_optix" and collect_ad_tapes:
    raise RuntimeError(...)
if accumulate_primal == "rayd_optix" and include_reflection_coupled:
    raise RuntimeError(...)
```

This avoids mixed RayD/DrJit execution in the explicit native mode.

### Path Solver Integration

Modify:

- `witwin/channel/path/`
- `witwin/channel/deterministic/reflection/paths.py` only if the path solver still routes diffraction through deterministic helpers.
- Shared result assembly under `witwin/channel/core/results/` only if new native fields need mapping.

New internal entry:

```python
native_paths = scene.trace_diffraction_paths(
    tx_positions=...,
    rx_positions=...,
    config=resolved_path_config,
    max_order=config.max_diffraction_order,
    max_paths=config.max_num_paths,
    seed=config.seed,
    return_geometry=config.return_geometry,
)
```

Behavior:

- First version supports first-order diffraction path export.
- Output maps into existing `PathResult` fields.
- `return_geometry=False` skips native geometry arrays.
- `return_geometry=True` fills diffraction points from `DiffractionPathResult`.
- Reflection path scheduling remains unchanged.

Metadata:

```python
metadata["runtime_backends"]["diffraction"] = {
    "implementation": "rayd_trace_diffraction_paths_order1",
    "path_export_backend": "rayd_optix_compact_paths",
    "max_order": 1,
}
```

## Implementation Tasks

### Task 1: RayD Structs And Python Bindings

**Files:**

- Modify: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation.h`
- Modify: `E:\Code\RayDi\include\rayd\rayd.h`
- Modify: `E:\Code\RayDi\src\rayd.cpp`
- Test: `E:\Code\RayDi\tests\drjit\test_diffraction_accumulation.py`

- [x] Add `DiffractionGrid`, `DiffractionStateTableData`, `DiffractionMaterialData`, `DiffractionAccumOptions`, and `DiffractionAccumResultData`.
- [x] Expose Python classes `DiffractionStateTable`, `DiffractionAccumOptions`, and `DiffractionAccumResult`.
- [x] Write a test that constructs a one-state table and verifies field widths.
- [x] Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation
```

Expected first run: fails because the classes do not exist.

Expected after implementation: passes.

### Task 2: Channel State Adapter

**Files:**

- Modify: `witwin/channel/core/scene/scene.py`
- Modify: `witwin/channel/montecarlo/trace/diffraction.py`
- Test: `tests/montecarlo/test_monte_carlo_radiomap_integrators.py`

- [x] Add a private conversion helper from channel `DiffractionStates` to RayD `DiffractionStateTable`.
- [x] The helper must preserve `edge_index`, edge geometry, face normals, source point, source power, incident direction, initial direction, and prefix depth.
- [x] Add a unit test using a tiny synthetic `DiffractionStates` object.
- [x] Run:

```powershell
conda run -n witwin2 python -m pytest tests\montecarlo\test_monte_carlo_radiomap_integrators.py -q
```

### Task 3: RayD Order-1 Grid Kernel

**Files:**

- Create: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation_params.h`
- Create: `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- Modify: `E:\Code\RayDi\src\multipath\pipelines.cpp`
- Modify: `E:\Code\RayDi\src\scene\scene.cpp`
- Modify: `E:\Code\RayDi\CMakeLists.txt`
- Test: `E:\Code\RayDi\tests\drjit\test_diffraction_accumulation.py`

- [x] Add `__raygen__diffraction_order1_accumulation`.
- [x] Implement source-to-edge and edge-to-target visibility inside the raygen program.
- [x] Implement direct receiver-cell connection.
- [x] Implement Keller cone plane-hit connection.
- [x] Atomic-add scalar power into grid.
- [x] Return direct/Keller counts.
- [x] Regenerate committed PTX header.
- [x] Build and install into `witwin2`.
- [x] Run the RayD test suite for the new kernel.

### Task 4: Monte Carlo Basic Native Diffraction

**Files:**

- Modify: `witwin/channel/montecarlo/config.py`
- Modify: `witwin/channel/montecarlo/integrators/basic.py`
- Modify: `witwin/channel/montecarlo/trace/diffraction.py`
- Test: `tests/montecarlo/test_monte_carlo_radiomap_integrators.py`

- [x] Add `DiffractionExecutionConfig.accumulate_primal = "auto" | "drjit" | "rayd_optix"`.
- [x] Dispatch Basic first-order diffraction to `Scene.accumulate_diffraction_order1` when non-AD and explicitly requested.
- [x] Raise for explicit `rayd_optix` AD workloads instead of falling back.
- [x] Add metadata showing `rayd_optix_atomic_add`.
- [x] Add parity test versus DrJit on a tiny fixed-seed scene.

### Task 5: BDPT Order-1 Native Diffraction

**Files:**

- Modify: `witwin/channel/montecarlo/integrators/bdpt.py`
- Modify: `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`
- Test: `tests/montecarlo/test_monte_carlo_radiomap_integrators.py`

- [x] Add a native dispatch for `max_depth == 1` and suffix disabled.
- [x] Use the same native state table for direct selected-edge states and prefix states.
- [x] Add test for `integrator="bdpt"`, `max_diffraction_order=1`, `enable_bdpt_reflection_coupled_diffraction=False`, `rayd_optix`.
- [x] Add strict rejection test for `enable_bdpt_reflection_coupled_diffraction=True` with explicit `rayd_optix`.
- [x] Run the Monte Carlo test subset.

### Task 6: BDPT Chain Native Diffraction

**Files:**

- Modify: `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- Modify: `E:\Code\RayDi\src\scene\scene.cpp`
- Modify: `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`
- Test: `tests/montecarlo/test_monte_carlo_radiomap_integrators.py`

- [x] Add `Scene.accumulate_diffraction_chains`.
- [x] Support order 2 direct receiver-cell connection.
- [x] Extend direct-chain fast path to order 3.
- [x] Add order 2 and order 3 Keller-chain accumulation to the RayD kernel.
- [x] Route BDPT order 1/2/3 direct and Keller strategies through strict RayD native dispatch for explicit `rayd_optix`.
- [x] Add debug counters for inter-edge visibility rejects.
- [x] Add strict no-fallback tests for BDPT `rayd_optix` direct/Keller chains, AD rejection, and suffix rejection.
- [x] Add parity tests against DrJit within Monte Carlo tolerance.

### Task 7: Path Solver Diffraction Export

**Files:**

- Create: `E:\Code\RayDi\include\rayd\multipath\diffraction_paths.h`
- Create: `E:\Code\RayDi\src\multipath\diffraction_paths.cu`
- Modify: `E:\Code\RayDi\src\rayd.cpp`
- Modify: `witwin/channel/path/`
- Test: path-solver tests under `tests/`

- [x] Add `Scene.trace_diffraction_paths`.
- [x] Support first-order diffraction path export.
- [x] Map native output to existing `PathResult` without public API changes.
- [x] Respect `return_geometry`.
- [x] Add a first-order path smoke test with finite coefficients and stable path counts.
- [x] Add a metadata test for `rayd_trace_diffraction_paths_order1`.

### Task 8: Suffix Reflection Native Kernel

**Files:**

- Modify: `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- Modify: `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`
- Test: `tests/montecarlo/test_shadow_boundary_backend.py` or a new focused BDPT suffix test.

- [x] Replace uniform triangle suffix sampling with a compact reflection-candidate table.
- [x] Fuse edge-to-reflection and reflection-to-target visibility in native OptiX.
- [x] Avoid `segment_visible(ignore_prim_idx=...)` inside DrJit symbolic loops.
- [x] Add a regression test for `BDPT + RayD + max_diffraction_order=1 + reflection_coupled_diffraction=True`.

### Task 9: Benchmarks And Gates

**Files:**

- Modify: `tests/support/bin/benchmark_monte_carlo_radiomap_package.py`
- Modify or create benchmark helpers under `tests/support/bin/`
- Update: `docs/dev/optimization/`

- [x] Repair the missing `_benchmark_runtime` import or replace it with an in-tree helper.
- [x] Add benchmark modes for `basic-rayd-diffraction`, `bdpt-rayd-diffraction`, and `path-rayd-diffraction`.
- [x] Record three-cube and Munich-style timings.
- [x] Gate first-order grid diffraction at minimum 2x over current DrJit diffraction on the same sample budget.
- [x] Gate order 2/3 BDPT at minimum 2x over current DrJit chain path.
- [x] Gate path-solver first-order diffraction export with no path-count regressions on maintained smoke scenes.

## Performance Targets

| Target | Baseline | Acceptance |
| --- | --- | --- |
| MC basic order-1 diffraction | Current DrJit `Diffraction.trace_batches` | At least 2x faster on 64k samples |
| BDPT order-1 direct/Keller | Current `BDPTDiffractionMIS.trace_direct_batches` + `trace_keller_batches` | At least 2x faster on 64k samples |
| BDPT order 2/3 chain | Current DrJit chain direct/Keller loops | At least 2x faster, target 4x |
| BDPT coupled prefix order 1 | Current DrJit reflection-prefix state path | RayD reflection prefix handoff works without disabling coupled mode |
| Path solver first-order diffraction | Current path diffraction export path | Same path counts on smoke scene, lower wall time |

## Verification Commands

Run after each RayD native change:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_reflection_accumulation
```

Run after each channel integration change:

```powershell
conda run -n witwin2 python -m pytest tests\montecarlo\test_monte_carlo_radiomap_integrators.py -q
conda run -n witwin2 python -m pytest tests\montecarlo\test_shadow_boundary_backend.py -q
```

Run before claiming plan phase completion:

```powershell
conda run -n witwin2 python -m pytest tests\montecarlo\test_monte_carlo_radiomap_integrators.py tests\montecarlo\test_monte_carlo_shadow_boundary_smoothing.py tests\montecarlo\test_shadow_boundary_backend.py -q
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_reflection_accumulation
```

## Rollout Policy

- Keep `auto` on existing DrJit behavior until parity and benchmark gates pass.
- Expose `rayd_optix` as explicit opt-in first.
- Switch `auto` for MC basic order-1 only after benchmark and parity gates pass.
- Switch `auto` for BDPT only after coupled prefix and suffix behavior are stable.
- Do not switch path solver defaults until path counts, delays, and geometry payloads are stable across maintained smoke scenes.

## Open Questions

- Whether RayD should own best-edge discovery from wedge hit data in Phase 1 or keep it channel-side until parity passes.
- Whether first coherent vector diffraction should ship in the order-1 kernel or wait until scalar power parity is closed.
- Whether path solver first-order diffraction should enumerate exact finite-edge candidates or use the same sampled state distribution as radio maps.
- Whether suffix reflection should use reflection hit history, surface-group adjacency, or a dedicated surface BVH top-K query for candidate generation.
