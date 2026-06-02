# Path Reflection Scheduling Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce standalone path-solver reflection runtime on Munich multi-Tx/multi-Rx workloads by removing wasted first-bounce sampling and moving multi-bounce reflection prefix scheduling out of Python hot loops.

**Architecture:** Keep the public `Scene + witwin.channel.path.solve(scene, config) + PathResult` contract unchanged. Optimize the shared deterministic reflection discovery layer because the path solver already depends on `witwin.channel.deterministic.reflection.paths.trace_paths`; deterministic field/radiomap callers must keep current behavior through tests and metadata. Treat Sionna as a reference baseline only, not ground truth.

**Tech Stack:** Python 3.11 in `witwin2`, DrJit, RayD 0.3.0 native extension, CUDA/OptiX, PyTorch result adapters at public result boundaries.

---

## Baseline From 2026-05-20

All Python commands below must run in `witwin2`.

Munich pressure validator, 2 Tx / 3 Rx, 3.5 GHz, `max_bounces=1`, `max_diffraction_order=0`, `warmup=1`, `repeats=3`:

| Samples | LoS median | Reflection median | Reflection count parity | Tau max delta vs Sionna reference |
| --- | ---: | ---: | --- | ---: |
| 8192 | 13.23 ms | 104.44 ms | 28 / 28 | 8.53e-14 s |
| 32768 | 7.73 ms | 132.73 ms | 28 / 28 | 1.14e-13 s |
| 131072 | 7.56 ms | 143.94 ms | 28 / 28 | 1.14e-13 s |

Warm-start path-solver profiling, 32768 samples, 2 Tx / 3 Rx:

| Case | Total | Reflection stage | RayD discovery wrapper | EPC replay | Result assembly |
| --- | ---: | ---: | ---: | ---: | ---: |
| `return_geometry=False` | 129-136 ms | 118-124 ms | 86-95 ms | 21-24 ms | 4-5 ms |
| `return_geometry=True` after geometry JIT | 155 ms | 113 ms | 84 ms | 42 ms total EPC calls | 34 ms |

Internal scheduler breakdown for `trace_paths`, 32768 rays, 2 Tx, warm-start:

| Reflection depth | RayD `trace_reflections` | `collect_prefix_paths` | Analytic first-bounce enumeration | Direction generation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.1 ms | 82.0 ms | 0.6 ms | 0.1 ms |
| 2 | 4.1 ms | 338.9 ms | 0.6 ms | 0.1 ms |

Interpretation:

1. RayD native tracing is not the steady-state bottleneck in this workload.
2. Current first-order reflection does unnecessary work: `trace_paths` runs sampled RayD discovery and `collect_prefix_paths`, then replaces depth-1 output with `enumerate_first_bounce_surface_paths`.
3. Multi-bounce reflection is dominated by Python-side prefix scheduling and bucketing in `collect_prefix_paths`, not by RayD trace.
4. `return_geometry=True` has a one-time materialization/JIT spike near one second, but warm-start geometry assembly is tens of milliseconds, not the main steady-state cost.

Post first-order fast path, same Munich setup:

| Samples | Reflection median | Speedup vs baseline | Reflection count parity | Tau max delta vs Sionna reference |
| --- | ---: | ---: | --- | ---: |
| 32768 | 69.67 ms | 1.91x | 28 / 28 | 1.14e-13 s |
| 131072 | 72.49 ms | 1.99x | 28 / 28 | 1.14e-13 s |

Current multi-bounce scheduler profile after the first-order fast path, `max_bounces=2`, 32768 rays, 2 Tx:

| Stage | Time |
| --- | ---: |
| RayD `trace_reflections` | 4.29 ms |
| `collect_prefix_paths` | 328.58 ms |
| `enumerate_first_bounce_surface_paths` | 0.82 ms |
| Current `trace_paths` total | 341.54 ms |

Depth-1 output is now analytic (`20438` first-bounce surface paths per Tx in this scene). Depth-2 remains the sampled RayD prefix set (`860` paths for Tx0 and `936` for Tx1) to preserve existing correctness semantics.

Post channel-native exact prefix compaction, same `max_bounces=2` scheduler profile:

| Stage | Time |
| --- | ---: |
| RayD `trace_reflections` | 3.2-3.9 ms |
| Old Python `collect_prefix_paths` reference | 288-293 ms |
| Current `trace_paths` total | 5.7-6.6 ms |

The channel-native path preserves the old Python depth-2 prefix counts (`860` for Tx0 and `936` for Tx1). A direct gather path was rejected because it produced `885` and `989` paths and therefore changed prefix bucketing semantics.

Munich path-solver smoke, `max_bounces=2`, 32768 samples, 2 Tx / 3 Rx, `return_geometry=True`, `max_num_paths=32`:

| Metric | Value |
| --- | ---: |
| Median runtime | 190.5 ms |
| Valid exported paths | 133 |

Three-bounce Munich scheduler pressure after channel-native exact prefix compaction, 2 Tx:

| Samples | RayD trace | Old Python collector | Current `trace_paths` total | Depth-2 / depth-3 counts |
| --- | ---: | ---: | ---: | --- |
| 32768 | 3.35-4.72 ms | 0.792-0.880 s | 8.74-10.45 ms | Tx0: 861 / 991, Tx1: 947 / 1565 |
| 131072 | 4.76-5.07 ms | 0.953-1.050 s | 10.92-11.14 ms | Tx0: 1132 / 1363, Tx1: 1041 / 1886 |

Three-bounce standalone path-solver Munich pressure, 2 Tx / 3 Rx, `return_geometry=True`, `max_num_paths=256`:

| Samples | Witwin reflection median | Sionna reference median | Ratio | Witwin / Sionna valid paths | Count mismatches | Common tau max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 346.66 ms | 26.55 ms | 13.06x | 327 / 115 | 6 | 0.0 s |
| 131072 | 343.63 ms | 27.88 ms | 12.33x | 323 / 117 | 6 | 0.0 s |

The third-order comparison intentionally treats Sionna as a baseline, not truth. Common sorted delay overlap is exact in these runs, while Sionna exports fewer third-order paths for every Tx/Rx pair. The remaining steady-state gap is therefore no longer the RayD trace or the prefix scheduler; it is dominated by receiver replay, path export, and result materialization under Witwin's public `PathResult` contract.

Post reflection-geometry cache optimization, the same third-order standalone path-solver pressure removes the second EPC replay previously used only to materialize hit points, normals, and object slots:

| Samples | Witwin reflection median | Sionna reference median | Ratio | Witwin / Sionna valid paths | Count mismatches | Common tau max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 206.34 ms | 30.53 ms | 6.76x | 327 / 115 | 6 | 0.0 s |
| 131072 | 208.38 ms | 30.74 ms | 6.78x | 323 / 117 | 6 | 0.0 s |

The geometry cache keeps first-pass reflection EPC geometry in `_REFLECTION_PATH_REFS_PAYLOAD`, so `assemble_result_payload(..., return_geometry=True)` can pack selected reflection paths without replaying EPC. This improves third-order Munich geometry export by roughly 40% while preserving path counts and delay overlap. The remaining cost is the first receiver replay itself: all exact source path candidates still need chain-to-Rx EPC, visibility, field transport, angle calculation, and optional geometry slot packing.

A one-run wrapped EPC profile for 32768 samples, third-order Munich, 2 Tx / 3 Rx, geometry enabled, reported about `196 ms` total with these receiver-replay calls:

| Tx-local call | Depth | Pairs | EPC time |
| --- | ---: | ---: | ---: |
| 0 | 1 | 61314 | 13.95 ms |
| 0 | 2 | 2583 | 26.62 ms |
| 0 | 3 | 2973 | 38.48 ms |
| 1 | 1 | 61314 | 13.17 ms |
| 1 | 2 | 2841 | 27.17 ms |
| 1 | 3 | 4695 | 40.46 ms |

The next optimization target is therefore high-order receiver EPC replay, not first-bounce surface enumeration. A fused native export path should batch same-depth multi-Tx path refs, evaluate EPC/visibility/field/angles/geometry in one kernel family, and write directly into per-link top-k result slots to avoid Python/Torch staging and repeated small high-order EPC launches.

Post batched multi-transmitter reflection export, the path solver now mirrors Sionna's compact-buffer idea at the Witwin `SourcePathSet` layer: all transmitter-local source paths with the same reflection depth are merged into one compact path set, each source path carries a transmitter-owner index, and receiver EPC replay runs once per depth instead of once per `(Tx, depth)`.

| Samples | Witwin reflection median | Sionna reference median | Ratio | Witwin / Sionna valid paths | Count mismatches | Common tau max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 114.42 ms | 28.19 ms | 4.06x | 327 / 115 | 6 | 0.0 s |
| 131072 | 127.78 ms | 39.00 ms | 3.28x | 323 / 117 | 6 | 0.0 s |

The same wrapped EPC profile for 32768 samples drops to about `113 ms` total and three receiver-replay calls:

| Depth | Pairs | EPC time |
| --- | ---: | ---: |
| 1 | 122628 | 14.41 ms |
| 2 | 5424 | 28.61 ms |
| 3 | 7668 | 40.57 ms |

This confirms why Sionna remains faster: Sionna keeps source and target ownership inside one `PathsBuffer`, shrinks it once, detaches geometry, and runs image/field processing over compact arrays. Witwin now does the same across transmitters for reflection export, but still pays Python/Torch result-selection staging and a generic high-order EPC kernel path before packing `PathResult`.

Implementation boundary:

1. Channel now owns the exact prefix-compaction adapter and native kernel because the bottleneck is Witwin's `SourcePathSet` scheduling contract, not RayD tracing.
2. RayD does not need to change for the current optimization. A future upstream RayD helper can use the same adapter hook (`rayd.compact_reflection_prefix_paths`) if we want this capability available outside Witwin.
3. The native kernel intentionally uses exact primitive-chain comparison plus Python-matching double-precision image-source quantization. It does not adopt Sionna's hash-only candidate-loss semantics.

## Sionna Speed Analysis

Sionna 2.0.1 is fast on this benchmark because its path solver is a compact DrJit/Mitsuba pipeline:

1. `PathSolver.__call__` generates candidates into one `PathsBuffer`, schedules the arrays once, shrinks the buffer, detaches path geometry, runs image method, computes fields, then discards invalid paths.
2. Specular chain uniqueness is handled during shoot-and-bounce by hash counters in DrJit arrays. This is GPU-friendly and avoids Python dictionaries, but the implementation explicitly accepts possible candidate loss from hash collisions.
3. `ImageMethod` reuses the same `PathsBuffer` and runs symbolic DrJit loops over compact candidate arrays. It does not build per-depth Python `SourcePathSet` objects before receiver replay.
4. Geometry is detached after candidate generation. Sionna therefore avoids differentiating through candidate discovery and does not carry the same AD/result-contract burden as Witwin path export.

Witwin is currently heavier in three places:

1. Multi-bounce reflection converts RayD's compact chain into exact per-depth prefix buckets in Python (`collect_prefix_paths`). This scalarizes chain entries, constructs tuple keys, sorts by first-seen ray, then rebuilds DrJit arrays. This is the dominant multi-bounce bottleneck.
2. Path export replays each source path toward every receiver through EPC to produce the public `PathResult` contract: `a`, `tau`, AoD/AoA, interaction type slots, and optional geometry/object slots. Sionna's buffer stays closer to its internal representation.
3. `return_geometry=True` materializes extra vertex, normal, and object-slot tensors. Warm-start cost is moderate, but the first geometry call pays a large JIT/materialization spike.

This means the path solver is not slow because RayD reflection tracing is slow. It is bloated around scheduling and export materialization. The first-order wasted RayD/prefix pass is removed. The next meaningful optimization is a correctness-preserving native prefix compaction path for depths >= 2.

## File Map

- Modify: `witwin/channel/deterministic/reflection/paths.py`
  - Owns RayD reflection trace invocation, analytic first-bounce enumeration, and Python prefix scheduling.
- Modify: `witwin/channel/deterministic/trace/path_export.py`
  - Owns path-solver reflection EPC replay and reflection path refs. Only touch for timing metadata or materialization reduction.
- Modify: `witwin/channel/path/solver.py`
  - Owns standalone path-solver stage timing metadata. Only touch to expose finer diagnostics.
- Modify: `tests/support/bin/validate_path_solver_munich.py`
  - Extend repeatable Munich path validation/profiling knobs.
- Create: `tests/support/bin/profile_path_reflection_scheduler.py`
  - Checked-in profiler for RayD trace vs scheduling vs path export warm-start timings.
- Modify: `tests/path/test_endpoint_api_contract.py`
  - Add narrow path-solver API regressions for first-order and multi-bounce reflection behavior if needed.
- Create or modify: `tests/deterministic/test_reflection_path_discovery.py`
  - Add deterministic reflection discovery tests for the shared `trace_paths` behavior.
- Optional cross-repo change: `E:\Code\RayDi`
  - Add native compacted-prefix output only after Python-side fast path is verified and measured.

## Success Criteria

- Munich path validator remains correct for LoS and first-order reflection:
  - Reflection path counts unchanged for the maintained 2 Tx / 3 Rx setup.
  - Reflection tau set max delta remains within `1e-10 s` against the reference baseline.
- Warm-start `max_bounces=1`, `num_samples=32768`, `return_geometry=False` reflection median drops by at least 40% from the current ~130 ms profile.
- Warm-start internal scheduler for depth 1 no longer calls RayD or `collect_prefix_paths`.
- Warm-start depth 2 has a documented scheduling profile and a follow-up target: Python `collect_prefix_paths` must be either reduced by native compaction or isolated as the dominant remaining bottleneck.
- Public path API, deterministic solver API, and RayD import/version contracts remain unchanged.

---

## Task 1: Check In The Warm-Start Scheduler Profiler

**Files:**
- Create: `tests/support/bin/profile_path_reflection_scheduler.py`

- [x] **Step 1: Create the profiler script**

Create `tests/support/bin/profile_path_reflection_scheduler.py` with the same structure as the temporary profiling script used for this investigation:

```python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import drjit as dr
import rayd

CHANNEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIONNA_SOURCE_ROOT = CHANNEL_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src"
DEFAULT_MUNICH_XML = DEFAULT_SIONNA_SOURCE_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
DEFAULT_TX_POSITIONS = ((8.5, 21.0, 27.0), (45.0, 15.0, 22.0))
DEFAULT_RX_POSITIONS = ((45.0, 90.0, 1.5), (30.0, 55.0, 1.5), (60.0, 20.0, 1.5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=DEFAULT_MUNICH_XML)
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--max-bounces", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true", default=False)
    return parser
```

The implementation must:

```python
def _time_stage(stats: dict[str, float], label: str, fn):
    start = time.perf_counter()
    out = fn()
    dr.sync_thread()
    stats[label] = stats.get(label, 0.0) + time.perf_counter() - start
    return out
```

and must measure exactly these stage labels:

```python
"select_ray_directions"
"rayd.trace_reflections"
"collect_prefix_paths"
"enumerate_first_bounce_surface_paths"
```

- [x] **Step 2: Run profiler once**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m tests.support.bin.profile_path_reflection_scheduler --num-samples 32768 --max-bounces 2 --warmup 1 --repeats 3 --json
```

Expected:

- JSON output includes per-repeat timings.
- `rayd.trace_reflections` is materially lower than `collect_prefix_paths` after warmup.
- No correctness assertions are made in this profiler; it is a timing diagnostic.

---

## Task 2: Add A First-Order Fast-Path Regression Test

**Files:**
- Modify: `tests/deterministic/test_reflection_path_discovery.py` or create it if no focused file exists.

- [x] **Step 1: Write a test that fails on the current behavior**

Add a test that monkeypatches the RayD scene object to raise if `trace_reflections` is called for first-order reflection discovery.

```python
def test_first_order_reflection_discovery_uses_analytic_surface_enumeration(monkeypatch):
    scene = _single_wall_scene(device="cuda")
    tx = Tx(position=(-2.0, -1.0, 1.5))
    wave = Wave.from_frequency(3.5e9)

    class _NoRayDTrace:
        def trace_reflections(self, *args, **kwargs):
            raise AssertionError("first-order analytic discovery must not call RayD trace_reflections")

    monkeypatch.setattr(scene, "_rayd_scene", _NoRayDTrace())

    trace_data = paths.trace_paths(
        tx=tx,
        scene=scene,
        wave=wave,
        n_rays=32768,
        max_reflections=1,
        mode="3d",
        material=Material(reflection_coef=1.0),
        ray_sampling="full_sphere",
        sampling_axis="y",
        sampling_bounds=((-3.0, 3.0), (0.0, 3.0)),
        sampling_plane_position=0.0,
        tri_data=scene._triangle_runtime(),
    )

    first = trace_data["source_paths_per_bounce"][0]
    assert int(first.chain_depth) == 1
    assert int(first.n_paths) >= 1
    assert trace_data["reflection_model"] == "materialized"
```

- [x] **Step 2: Verify red**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/deterministic/test_reflection_path_discovery.py::test_first_order_reflection_discovery_uses_analytic_surface_enumeration -q
```

Expected before implementation:

- FAIL with `AssertionError: first-order analytic discovery must not call RayD trace_reflections`.

---

## Task 3: Implement The First-Order Analytic Fast Path

**Files:**
- Modify: `witwin/channel/deterministic/reflection/paths.py`

- [x] **Step 1: Move metadata resolution before sampled direction generation**

In `trace_paths`, compute material metadata and sampling metadata without generating ray directions:

```python
reflection_model, reflection_model_source = trace_material_info(
    scene=scene,
    material=material,
)
ray_sampling_info = common.resolve_sampling_info(
    axis=sampling_axis,
    bounds=sampling_bounds,
    tx=tx,
    mode=mode,
    plane_position=sampling_plane_position,
    ray_sampling=ray_sampling,
)
```

- [x] **Step 2: Return analytic first-bounce paths before RayD trace**

Add this before `select_ray_directions` and before `rayd_scene.trace_reflections`:

```python
if int(max_reflections) == 1:
    return {
        "source_paths_per_bounce": (
            enumerate_first_bounce_surface_paths(tx=tx, tri_data=tri_data),
        ),
        "ray_sampling_info": ray_sampling_info,
        "reflection_model": reflection_model,
        "reflection_model_source": reflection_model_source,
        "reflection_discovery_backend": "analytic_first_bounce",
    }
```

- [x] **Step 3: Keep existing multi-bounce behavior unchanged**

After the fast path, leave existing RayD direction generation and trace code in place for `max_reflections > 1`.

- [x] **Step 4: Verify green**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/deterministic/test_reflection_path_discovery.py::test_first_order_reflection_discovery_uses_analytic_surface_enumeration -q
```

Expected:

- PASS.

- [x] **Step 5: Run path and scene targeted tests**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/path/test_endpoint_api_contract.py tests/scene/test_scene_segment_visibility_rayd.py -q
```

Expected:

- PASS.

---

## Task 4: Validate First-Order Munich Correctness And Warm-Start Speed

**Files:**
- Modify: `tests/support/bin/validate_path_solver_munich.py` only if it needs extra timing fields.

- [x] **Step 1: Run Munich correctness and pressure benchmark**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m tests.support.bin.validate_path_solver_munich --warmup 1 --repeats 3 --num-samples 32768 --skip-ad --no-strict --json --output .codex_tmp\munich_path_validate_32768_after_reflection_fast_path.json
```

Expected:

- `checks.reflection_correctness.passed == true`
- `checks.reflection_correctness.tau_comparison.count_mismatches == 0`
- `checks.reflection_correctness.tau_comparison.max_tau_delta_s <= 1e-10`
- Reflection median is materially below the baseline `132.73 ms`.

- [x] **Step 2: Run higher sample pressure**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m tests.support.bin.validate_path_solver_munich --warmup 1 --repeats 3 --num-samples 131072 --skip-ad --no-strict --json --output .codex_tmp\munich_path_validate_131072_after_reflection_fast_path.json
```

Expected:

- Reflection correctness remains true.
- Reflection median no longer scales with sampled RayD discovery for first-order reflection.

---

## Task 5: Add Fine-Grained Solver Timing Metadata

**Files:**
- Modify: `witwin/channel/path/solver.py`
- Modify: `witwin/channel/deterministic/trace/path_export.py`

- [ ] **Step 1: Add reflection collection sub-timings**

In `collect_reflection_paths`, build a local timing dictionary:

```python
stage_timing = {
    "discovery_seconds": 0.0,
    "receiver_replay_seconds": 0.0,
    "packing_seconds": 0.0,
}
```

Wrap:

- `discover_reflection_paths`
- the per-bounce `epc_reflection_chain_to_target` replay loop
- final `_finalize_reflection_path_refs`

Store it under raw metadata:

```python
metadata={
    "n_paths": total_paths,
    "per_bounce_counts": tuple(
        int(paths.n_paths) if paths is not None else 0
        for paths in detail.source_paths_per_bounce
    ),
    "timing": stage_timing,
}
```

- [ ] **Step 2: Surface aggregate timing in path result metadata**

In `witwin/channel/path/solver.py`, aggregate reflection raw collection metadata:

```python
metadata["reflection_path_export"] = {
    "collections": tuple(r.get("metadata", {}).get("timing", {}) for r in reflection_raw_collections),
}
```

- [ ] **Step 3: Verify metadata shape**

Run a one-case script:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m tests.support.bin.validate_path_solver_munich --warmup 1 --repeats 1 --num-samples 32768 --skip-ad --no-strict --json
```

Expected:

- JSON contains reflection correctness.
- Result metadata inspection from an ad hoc solve contains `reflection_path_export.collections`.

---

## Task 6: Multi-Bounce Scheduling Optimization Design Gate

**Files:**
- Modify: `tests/support/bin/profile_path_reflection_scheduler.py`
- Do not modify RayD yet.

- [x] **Step 1: Profile multi-bounce warm-start scheduler**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m tests.support.bin.profile_path_reflection_scheduler --num-samples 32768 --max-bounces 2 --warmup 1 --repeats 3 --json
```

Expected:

- `rayd.trace_reflections` remains near single-digit milliseconds after warmup.
- `collect_prefix_paths` dominates; current observed value is about `339 ms` for 2 Tx at depth 2.

- [x] **Step 2: Decide implementation route from measurements**

Use this rule:

```text
If collect_prefix_paths is > 50% of end-to-end reflection time, optimize scheduling.
If RayD trace is > 50%, inspect RayD launch audit and OptiX payload instead.
```

Current evidence selects scheduling.

---

## Task 7: Native Or Vectorized Prefix Scheduling For Multi-Bounce Reflection

**Files:**
- Modify first in Witwin: `witwin/channel/deterministic/reflection/paths.py`
- Optional RayD changes after Witwin profiling proves the ABI: `E:\Code\RayDi\src\multipath\reflection_dedup.cu`, `E:\Code\RayDi\src\scene\scene.cpp`, `E:\Code\RayDi\src\rayd.cpp`

- [x] **Step 1: Document current Python bottleneck**

Add a code comment above `collect_prefix_paths`:

```python
# This routine is intentionally simple but Python-bound: it scalarizes RayD chain
# entries to bucket unique prefix paths. Keep it out of first-order reflection
# and replace it with native compaction before increasing multi-bounce budgets.
```

- [x] **Step 2: Add an explicit backend marker**

In `trace_paths`, add to returned metadata:

```python
"reflection_discovery_backend": (
    "analytic_first_bounce"
    if int(max_reflections) == 1
    else "rayd_trace_python_prefix_compaction"
),
```

- [x] **Step 3: Prototype native prefix compaction behind a private helper**

Add a private helper in Witwin first:

```python
def _source_path_sets_from_native_prefix_payload(payload):
    return tuple(
        SourcePathSet(
            image_source=entry.image_source,
            discovery_count=entry.discovery_count,
            chain_depth=int(entry.chain_depth),
            n_paths=int(entry.n_paths),
            path_prim_idx=tuple(entry.path_prim_idx),
            path_plane_point=tuple(entry.path_plane_point),
            path_plane_normal=tuple(entry.path_plane_normal),
            path_hit_point=tuple(entry.path_hit_point),
        )
        for entry in payload
    )


def _collect_prefix_paths_native_if_available(
    chain,
    *,
    chain_depth,
    surface_canonical_prims,
    image_source_tolerance,
):
    native = getattr(rayd, "compact_reflection_prefix_paths", None)
    if native is None:
        return collect_prefix_paths(
            chain,
            chain_depth=chain_depth,
            surface_canonical_prims=surface_canonical_prims,
            image_source_tolerance=image_source_tolerance,
        )
    payload = native(
        chain,
        surface_canonical_prims,
        float(image_source_tolerance),
        int(chain_depth),
        2,
    )
    return _source_path_sets_from_native_prefix_payload(payload)
```

Expected behavior:

- No public API change.
- If RayD lacks the helper, current behavior remains.
- When RayD grows the helper, Witwin uses it without changing path solver call sites.

- [x] **Step 4: RayD ABI target**

The RayD native helper should accept:

```text
ReflectionChain[Detached]
canonical_prim_table
image_source_tolerance
max_prefix_depth
min_prefix_depth
```

and return per depth:

```text
n_paths
image_source
discovery_count
path_prim_idx[depth]
path_plane_point[depth]
path_plane_normal[depth]
path_hit_point[depth]
```

This is exactly the `SourcePathSet` payload required by Witwin.

Current implementation note:

The first shipping helper is channel-native, not RayD-native. It accepts RayD flat chain arrays through the Witwin deterministic native extension and returns representative chain indices plus discovery counts, which Python wraps into `SourcePathSet`. The RayD ABI target remains documented for a future upstream helper, but it is not required for the current speedup.

- [x] **Step 5: Keep first-bounce analytic even for multi-bounce**

For `max_reflections > 1`, output should remain:

```python
source_paths_per_bounce = (
    enumerate_first_bounce_surface_paths(tx=tx, tri_data=tri_data),
    *native_or_python_prefix_paths[1:],
)
```

This keeps first-bounce deterministic and prevents sampled first-bounce coverage artifacts.

---

## Task 8: Deterministic Solver Impact Audit

**Files:**
- Modify tests under `tests/deterministic/`
- Modify no public deterministic APIs.

- [ ] **Step 1: Run deterministic reflection tests**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/deterministic/test_reflection_epc_f_weight_reference.py tests/deterministic/test_reflection_secondary_visibility_reference.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run path-level tests**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/path/test_endpoint_api_contract.py tests/path/test_example_path_solver_minimal.py -q
```

Expected:

- PASS.

- [ ] **Step 3: Confirm no public API change**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; conda run -n witwin2 python -m pytest tests/path/test_endpoint_api_contract.py::test_channel_path_package_is_canonical_path_solver -q
```

Expected:

- PASS.

## Notes On Deterministic Solver Ownership

The optimization does involve the deterministic package because path solver reflection discovery is shared through `witwin.channel.deterministic.reflection.paths.trace_paths`. It should not change `witwin.channel.deterministic.solve` public API. Deterministic field/radiomap behavior may change only in metadata and speed for first-order reflection discovery; numerical output must remain unchanged.

`RayD trace_reflections_accumulating` is not a replacement for path solver reflection export. It accumulates directly into receiver grids and does not provide the discrete `a`, `tau`, AoD/AoA, interaction type slots, and optional geometry required by `PathResult`.

## Execution Order

1. Check in profiler and baseline JSON outputs.
2. Add first-order fast-path regression test.
3. Implement first-order analytic fast path.
4. Re-run Munich correctness and pressure benchmarks.
5. Add fine-grained timing metadata if needed for ongoing profiling.
6. Profile multi-bounce scheduling after first-order fix.
7. Only then decide whether to implement RayD native prefix compaction.

## Completion Checklist

- [x] First-order path solver no longer calls RayD trace for reflection discovery.
- [x] Munich LoS/reflection correctness remains passing.
- [x] Warm-start reflection median is recorded before and after optimization.
- [x] Multi-bounce bottleneck is measured with a checked-in script.
- [x] Any native scheduling work has an adapter matching `SourcePathSet`.
- [ ] `FEATURE_LIST.md` is updated only if a new user-visible profiling workflow or public capability is added.
