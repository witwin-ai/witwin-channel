# RadioMap Monte Carlo Diffraction Kernel Launch Analysis

Status: Active
Category: Optimization
Last reviewed: 2026-04-08

## Purpose

This document records the current Dr.Jit kernel-launch structure for the three-cube radiomap Monte Carlo benchmark and pinpoints why Witwin diffraction remains slower than the Sionna RT 2.0.0 reference even after the estimator and sampling contract were aligned more closely.

The focus here is not field-estimator math. The dominant gap is execution shape: JIT launch count, OptiX launch count, and graph fragmentation.

## Measurement Contract

- Scene: three-cube benchmark scene used by `tests/support/bin/compare_radiomap_sionna_three_cubes.py`
- Grid: `256 x 256`
- Isolation: serial execution, one benchmark process at a time, subprocess-per-framework mode
- Timing mode: hot steady-state with warmup before the measured run
- Kernel tracing: `dr.scoped_set_flag(dr.JitFlag.KernelHistory, True)` with `dr.kernel_history((dr.KernelType.JIT,))`
- Reference files:
  - `tests/output/tmp_kernel_hist_witwin_no_diff_1e6.json`
  - `tests/output/tmp_kernel_hist_witwin_with_diff_1e6.json`
  - `tests/output/tmp_kernel_hist_sionna_no_diff_1e6.json`
  - `tests/output/tmp_kernel_hist_sionna_with_diff_1e6.json`
  - `tests/output/tmp_kernel_hist_witwin_with_diff_1e7.json`
  - `tests/output/tmp_kernel_hist_sionna_with_diff_1e7.json`

## Benchmark Results

### Post-change updates

The same serial `1e6` three-cube benchmark was remeasured twice on 2026-04-08:

- After retaining symbolic specular for `with_diff`:
  - `1e6 no_diff`: `6.77 ms`, `5` JIT kernels, `1` OptiX kernel
  - `1e6 with_diff`: `13.42 ms`, `43` JIT kernels, `8` OptiX kernels
- After replacing the post-loop direct-TX wedge state build with an in-loop
  fixed wedge-state store modeled after Sionna's `RadioMapSolver`:
  - `1e6 no_diff`: `7.29 ms`, `5` JIT kernels, `1` OptiX kernel
  - `1e6 with_diff`: `10.98 ms`, `8` JIT kernels, `2` OptiX kernels

Delta versus the previous `1e6 with_diff` baseline in this document:

- JIT kernels: `53 -> 8`
- OptiX kernels: `17 -> 2`
- wall time: `23.66 ms -> 10.98 ms`

This second change is the important one for launch shape. It brings Witwin
`with_diff` to the same kernel-count envelope as the local Sionna RT 2.0.0
reference.

### Serial, isolated benchmark results

`1e6 with_diff`

- Witwin: `23.66 ms`, `53` JIT kernels, `17` OptiX kernels
- Sionna: `11.87 ms`, `8` JIT kernels, `2` OptiX kernels

`1e7 with_diff`

- Witwin: `54.78 ms`, `59` JIT kernels, `19` OptiX kernels
- Sionna: `16.07 ms`, `8` JIT kernels, `2` OptiX kernels

`1e6 no_diff`

- Witwin: `5` JIT kernels
- Sionna: `3` JIT kernels

### Immediate conclusions

- `with_diff` used to be the problem because wedge discovery and state
  preparation lived outside the main symbolic trace.
- `no_diff` remains close in launch count and is not the main optimization
  target.
- The critical fix was not cache reuse. It was moving direct-TX wedge discovery
  and state storage into the symbolic specular loop, which let Dr.Jit record a
  compact graph similar to Sionna's.

## Exact Kernel Distribution

### Witwin `1e6 with_diff`

- `2` kernels of `size=1e6`
- `2` OptiX launches total
- `4` kernels of `size=65,536`
- `1` kernel of `size=256`
- `1` kernel of `size=54`

### Sionna `1e6 with_diff`

- `2` OptiX launches
- `8` JIT kernels total
- no `size=25` state-prep train

### Witwin `1e7 with_diff`

- `18` kernels of `size=9,934,848`
- `1` kernel of `size=2,390,784`
- `27` kernels of `size=25`
- `5` kernels of `size=65,152`
- metadata for the same run reported:
  - `ray_batch_size = 9,563,392`, `ray_batch_count = 2`
  - `diffraction_batch_size = 2,390,784`, `diffraction_batch_count = 5`

This confirms that the extra `1e7` overhead comes from a small amount of batching, while the base launch structure is already too fragmented in the single-batch `1e6` case.

## Code-Level Source Mapping

### 1. Specular still falls back to the manual path when diffraction is enabled

Current control flow:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1621`
- `use_manual_specular = return_timing or collect_diffraction_wedges`

When `max_diffractions > 0`, the specular half of the solve does not stay on the tighter no-diff symbolic loop. It switches to `_trace_specular_batch_manual(...)`:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:504`
- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1631`

That manual path performs repeated `scene.ray_intersect(...)` calls per depth:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:544`

For the measured `1e6 with_diff` case, direct monkeypatch counting reported:

- `scene.ray_intersect = 11`
- `scene.ray_test = 0`

This already explains a large portion of the `17` OptiX launches.

### 2. Diffraction visibility is still expressed as repeated segment queries

Current symbolic diffraction loop:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1122`

Visibility still calls `_segment_visibility_mask(...)` four times:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1214`
- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1219`
- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1230`
- `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1235`

And `_segment_visibility_mask(...)` currently disables `scene.ray_test(...)` when Dr.Jit is recording:

- `witwin/channel/trace/diffraction/geometry/visibility.py:177`
- `symbolic_recording = bool(dr.flag(dr.JitFlag.Recording))`
- `witwin/channel/trace/diffraction/geometry/visibility.py:188`

So symbolic diffraction still falls back to `_intersect_rays_ad_with_prim(...)` instead of the cheaper scene-level shadow query.

### 3. The `size=25` kernel train tracks the diffraction-state pool

Measured metadata for the same cases reported:

- `diffraction_state_pool.total = 25`
- `diffraction_state_pool.kept = 25`

The repeated `size=25` kernels correlate directly with this state pool width. They are not sample-count kernels; they are state-pool bookkeeping kernels.

Likely sources are the current state-prep and sampler path:

- `_prepare_length_proportional_state_sampler(...)`
  - `witwin/channel/monitors/radio_map/monte_carlo/trace.py:776`
- `_unique_discovered_edge_indices(...)`
  - `witwin/channel/monitors/radio_map/monte_carlo/trace.py:949`
- `_build_discovered_tx_first_order_state_arrays(...)`
  - `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1033`
- `_gather_monte_carlo_diffraction_state_arrays(...)`
  - `witwin/channel/monitors/radio_map/monte_carlo/trace.py:893`
- `_orient_monte_carlo_diffraction_face_view(...)`
  - `witwin/channel/monitors/radio_map/monte_carlo/trace.py:1096`

The exact per-kernel attribution still needs one more round of instrumentation, but the width match is too strong to ignore.

### 4. Field evaluation is not the first-order reason for the launch gap

Sionna does not rely on a dedicated native field-evaluator kernel for this path. The key difference is that Sionna keeps the diffraction solve in a far more compact Dr.Jit graph.

Kernel-history evidence supports this:

- Witwin `1e6 with_diff`: `53` JIT kernels
- Sionna `1e6 with_diff`: `8` JIT kernels

If the dominant gap were only the Jones/material arithmetic inside the field evaluator, we would expect similar launch counts with different execution times. That is not what the measurements show.

## RayDi 0.2.0 Findings

### Source inspection

RayDi now supports symbolic shadow queries through the unified OptiX scene path.

Evidence:

- `E:/Code/RayDi/src/scene/scene.cpp:78`
  - `uses_symbolic_optix_query_path()` returns `jit_flag(JitFlag::Recording);`
- `E:/Code/RayDi/src/scene/scene.cpp:1728`
  - `Scene::shadow_test(...)` routes symbolic recording to `optix_scene_->shadow_test(...)`
- `E:/Code/RayDi/src/scene/scene_optix.cpp:866`
  - `OptixScene::shadow_test(...)` directly issues `jit_optix_ray_trace(...)`

### Minimal runtime probe

A dedicated probe confirmed that `scene.ray_test(...)` can execute inside a symbolic loop with the current RayDi version.

This means the current visibility fallback in Witwin is outdated for the no-ignore case.

### Post-probe update: no-ignore visibility fallback removal

After enabling `scene.ray_test(...)` directly for the no-ignore symbolic path in:

- `witwin/channel/trace/diffraction/geometry/visibility.py`

the measured `1e6 with_diff` query mix changed from the previous all-`ray_intersect` behavior to:

- `scene.ray_intersect = 3`
- `scene.ray_test = 8`

However, the kernel-history totals remained effectively unchanged:

- `53` JIT kernels
- `17` OptiX kernels

This is an important result:

- changing the query primitive alone is not enough
- the remaining problem is query multiplicity and graph fragmentation, not merely the choice of `ray_intersect` versus `ray_test`

## Why The New Change Worked

The decisive result is that the scene-query mix did not materially change after
the in-loop wedge-state-store rewrite:

- `scene.ray_intersect = 2`
- `scene.ray_test = 8`

Yet the kernel totals dropped to:

- Witwin `1e6 with_diff`: `8` JIT kernels / `2` OptiX kernels
- Sionna `1e6 with_diff`: `8` JIT kernels / `2` OptiX kernels

That means the final gap was not raw query count by itself. The remaining gap
was graph fragmentation caused by the post-loop wedge pipeline:

- `depth0` hit capture
- post-loop wedge reconstruction
- `unique` compaction
- small-width edge/material state assembly
- separate state-pool sampler preparation

Once direct-TX wedge discovery and unique state insertion moved into the
symbolic specular loop, Dr.Jit recorded the whole `with_diff` path as a compact
Sionna-like graph even though the logical visibility checks were still the same.

## Lessons Learned

The main engineering lessons from this optimization round are:

- Optimize for graph shape first, not cache reuse first.
  Cache can hide setup cost across repeated traces, but it does not fix the
  kernel-launch structure of a single hot execution. The decisive improvement
  here came from changing where work happened in the graph, not from memoizing
  the old pipeline.

- Matching Sionna's control-flow structure mattered more than matching its
  arithmetic details.
  The important Sionna pattern was: discover wedges inside the main symbolic
  specular loop, store them in a fixed GPU buffer, and sample that buffer
  directly during diffraction. Reproducing that execution shape removed the
  persistent launch train.

- Small post-loop bookkeeping stages are dangerous even when they look cheap.
  The old `unique -> gather edge subset -> material assembly -> sampler prep`
  chain operated on only about `25` wedge states, but it still fragmented the
  graph and generated a long tail of extra kernels.

- A fixed-size scatter-based state store is often better than a dynamically
  rebuilt compact array.
  When the state family is small and bounded by scene structure, it is usually
  better to keep a preallocated GPU buffer plus `scatter_inc`/`scatter` than to
  rebuild a fresh compact state array after tracing.

- Scene-query counts and kernel counts must be measured separately.
  After the final rewrite, the query mix was still `2` `ray_intersect` calls and
  `8` `ray_test` calls, yet the launch count matched Sionna. That would have
  been easy to miss if only scene-query counts had been instrumented.

- Replacing `ray_intersect` with `ray_test` is cleanup, not a complete strategy.
  The visibility cleanup in `visibility.py` was still correct and useful, but it
  did not materially reduce launch count by itself. It only helped once the
  surrounding graph was compact.

- Avoid feature-triggered fallback paths in hot loops.
  The earlier `collect_diffraction_wedges` switch silently pushed the reflection
  solve off the symbolic path and multiplied launches. If a feature must be
  enabled in production, the hot path should stay on the same execution family.

- Do not chase field-math fusion before removing structural fragmentation.
  The data from this benchmark showed that launch count, not diffraction field
  arithmetic, was the dominant issue. Field-evaluator fusion would have attacked
  the wrong constant factor first.

- Symbolic-safety constraints should shape the design early.
  The failed `_fused_diffraction_visibility_masks(...)` attempt showed that some
  seemingly reasonable helper patterns are not symbolic-safe because they rely
  on symbolic concat/gather forms that Dr.Jit rejects. Designing around those
  constraints early saves time.

## Status Of The Reduction Plan

Completed on the current branch:

- remove the specular manual path from normal `with_diff` execution
- keep direct-TX wedge discovery on the symbolic path
- delete the old post-loop `size=25` wedge-state preparation train by replacing
  it with an in-loop fixed state store
- preserve the no-ignore symbolic `scene.ray_test(...)` cleanup in
  `visibility.py`

The original launch-count objective for the hot `1e6 with_diff` benchmark is
met. Further work, if needed, should focus on secondary goals such as:

- extending the same compact launch structure to larger `samples_per_tx`
  contracts like `1e7`
- reducing the remaining `no_diff` gap from `5` JIT kernels toward Sionna's `3`
- simplifying the diffraction visibility structure only if it improves code
  clarity or larger-scene scaling

## Follow-up From This Change

The older `_fused_diffraction_visibility_masks(...)` experiment is still worth
keeping in mind as a symbolic-safety note.

An additional experiment tried to route symbolic diffraction through
`_fused_diffraction_visibility_masks(...)` as well. That worked in the
non-symbolic timing path, but it failed in the symbolic Monte Carlo loop for a
different reason: the current batched helper needs to reindex symbolic point
arrays, and Dr.Jit rejected both tested forms:

- concatenating symbolic point arrays triggered dirty-variable failures during
  later evaluation
- gathering from symbolic point arrays triggered `cannot gather from a symbolic variable`

After the in-loop wedge-state-store rewrite, this failed fusion path is no
longer a blocker for the original launch target:

- the `size=25` wedge-state-preparation train is gone from the hot `1e6`
  benchmark
- the measured `with_diff` kernel envelope already matches Sionna at `8` JIT /
  `2` OptiX
- any future visibility fusion work should be judged on larger-scene scaling or
  code clarity, not on the original `1e6 with_diff` launch-count gap
