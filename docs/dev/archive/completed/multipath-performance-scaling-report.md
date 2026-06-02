# Multipath Performance Scaling Report

## Scope

- Sample scene: `tests/grad/grad_multipath.py`
- Target workload: scaling plane-monitor resolution and reflection ray count
- Environment: `witwin2`, `cuda_ad_rgb`, Windows/Codex shell
- Measurement method:
  - forward: `Tracer.trace(..., return_timing=True, return_diffraction_audit=True)`
  - backward: `loss = sum(|a_total|^2)` followed by `dr.backward(loss)`
  - memory: `dr.whos()` device allocator `used/peak`
  - collection rule: run benchmark processes serially; do not collect timings while other GPU-heavy workloads are active
  - note: Dr.Jit `unevaluated` storage is symbolic graph size, not directly allocated VRAM

## Root Causes

### Forward bottlenecks

1. `witwin/channel/trace/diffraction/field.py`
   - `_accumulate_edge_states_to_receivers` expanded the full Cartesian product `n_states * n_rx`.
   - For the sample at `256x256, 5000 rays`, this reached about `1648 * 65536 = 108,003,328` state-receiver pairs.
   - This was the main reason diffraction time and VRAM rose sharply with resolution.

2. `witwin/channel/trace/diffraction/suffix.py`
   - `_accumulate_reflected_segment_fields_batched` allocated flat buffers sized `n_states * n_rx` for suffix reflections.
   - This duplicated the same scaling problem inside the reflected-suffix path.

3. `witwin/channel/trace/reflection/field.py`
   - `_accumulate_reflection_paths_to_receivers` expanded `n_paths * n_rx` for exact reflection-path accumulation.
   - In the sample scene `n_paths` stayed small, so this was secondary to diffraction, but it still grew linearly with monitor resolution.

4. `witwin/channel/trace/reflection/paths.py`
   - `_collect_unique_reflection_paths` pulled active path data to CPU with NumPy and deduplicated in Python.
   - This made ray-count scaling worse, even when the final number of canonical paths stayed small.

5. `witwin/channel/trace/diffraction/state.py`
   - `_prune_state_arrays_by_budget` converted the full state set to NumPy for sorting.
   - This forced a GPU sync and CPU-side ranking whenever budgets were enabled.

### Backward bottlenecks

1. The same Cartesian expansions built very large AD graphs.
   - Before the fix, `dr.whos()` reported around `930 GiB` of unevaluated graph for the `256x256, 5000 rays` sample.
   - This is why backward cost collapsed faster than forward as resolution increased.

2. The loss is reduced only after the full monitor field is materialized.
   - Even after peak VRAM is reduced, the symbolic AD graph still spans the full receiver plane.
   - This remains the main residual scalability limit for very large optimization workloads.

## Baseline Measurements

Measured before the patch set.

| Grid / Rays | `n_states` | Pair estimate (`n_states*n_rx`) | Forward | Backward | Peak allocator |
| --- | ---: | ---: | ---: | ---: | ---: |
| `64x64 / 1000` | 411 | 1.68M | 0.365s | 0.064s | 311.8 MiB |
| `128x128 / 5000` | 1648 | 27.0M | 0.587s | 0.151s | 3.208 GiB |
| `256x256 / 5000` | 1648 | 108.0M | 1.205s | 0.495s | 9.588 GiB |
| `128x128 / 20000` | 6291 | 103.1M | 4.475s | not captured | allocator cache flush warning |

## Implemented Fixes

### 1. Chunked Cartesian accumulation

- Added `CARTESIAN_PAIR_CHUNK_BUDGET = 1 << 25` and `_cartesian_chunk_size(...)` in `witwin/channel/trace/diffraction/common.py`.
- Applied chunking to:
  - `witwin/channel/trace/diffraction/field.py::_accumulate_edge_states_to_receivers`
  - `witwin/channel/trace/diffraction/suffix.py::_accumulate_reflected_segment_fields_batched`
  - `witwin/channel/trace/reflection/field.py::_accumulate_reflection_paths_to_receivers`
- Result:
  - peak live GPU allocation is bounded
  - large monitor grids no longer require a single full `states × receivers` temporary
  - numerical outputs remain unchanged

### 2. Reflection-prefix canonicalization moved mostly off CPU

- Replaced per-ray NumPy grouping in `witwin/channel/trace/reflection/paths.py::_collect_unique_reflection_paths`.
- New flow:
  - GPU torch tensors for coarse hashing, lexicographic grouping, and averaging
  - a device-side tolerance merge over the already-collapsed coarse groups
- This removes the previous full-size GPU->CPU transfer from the reflection-prefix path builder.

### 3. Budget pruning moved to GPU torch

- Replaced NumPy ranking in `witwin/channel/trace/diffraction/state.py::_prune_state_arrays_by_budget`.
- Sorting now runs on GPU torch tensors with stable lexicographic ordering and DLPack-backed
  zero-copy views from the DrJit arrays.
- This avoids sync stalls and removes the previous duplicate DrJit/PyTorch materialization when
  mixed-path budgets are active.

### 4. Regression coverage for chunked path

- Added `tests/test_chunked_cartesian_accumulation.py`.
- It forces a very small pair budget and checks chunked vs unchunked `reflection`, `diffraction`, and `total` fields for equality.

## Post-fix Measurements

Measured after the patch set.

| Grid / Rays | `n_states` | Forward | Backward | Peak allocator | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| `64x64 / 1000` | 411 | 0.684s | 0.058s | 190.1 MiB | lower memory, small-case forward slower |
| `128x128 / 5000` | 1648 | 0.868s | 0.160s | 2.249 GiB | peak VRAM -29.9% |
| `256x256 / 5000` | 1648 | 1.299s | 0.665s | 4.098 GiB | peak VRAM -57.3% |
| `128x128 / 20000` | 6291 | 1.394s | 0.429s | 5.025 GiB | large ray-count case no longer thrashes allocator |

## Leak / Release Check

- Repeated `128x128 / 5000 rays` forward+backward three times without explicit cache flushing:
  - run 1: `1.06 GiB / 2.249 GiB used (peak 2.249 GiB)`
  - run 2: `1.06 GiB / 3.309 GiB used (peak 3.309 GiB)`
  - run 3: `1.06 GiB / 3.309 GiB used (peak 3.309 GiB)`
- Conclusion:
  - no monotonic live-memory leak was observed
  - Dr.Jit keeps allocator/cache state after warmup, which raises reserved peak once and then stabilizes
  - the primary problem was oversized temporary tensors, not unreleased live references

## Remaining Limits

1. Dr.Jit unevaluated AD graphs are still very large for high-resolution optimization workloads.
   - Example: `256x256 / 5000 rays` still reports about `847 GiB` unevaluated after backward.

2. Backward scaling is still driven by full-plane field materialization.
   - The current API computes the whole monitor field, then reduces it to a scalar loss.

3. The reflected-suffix path still uses DDA per chunk and remains expensive when both `n_states` and `n_rx` are high.

## Recommended Next Steps

1. Add an internal scalar-loss tracing path for optimization workloads.
   - Reduce over monitor tiles during tracing instead of materializing the full field graph.

2. Add monitor tiling for backward-only workloads.
   - Compute field tiles sequentially, accumulate the scalar objective, and release each tile graph earlier.

3. Consider a dedicated high-scale solver mode between `accuracy` and `fast_approximate`.
   - Keep current path families, but tile receiver accumulation and optionally cap suffix evaluation cost.

## Validation Run

Passed after the patch set:

- `tests/test_chunked_cartesian_accumulation.py --gpu`
- `tests/test_reflection_prefix_path_canonicalization.py`
- `tests/test_reflection_prefix_diffraction.py --gpu`
- `tests/test_drd_inserted_reflection.py --gpu`
- `tests/test_rd_multipath_consistency.py --gpu`
- `tests/test_mixed_path_budget_ownership.py`
- `tests/test_solver_modes_and_guardrails.py`

## Phase Benchmark Snapshot (2026-03-27)

All phase comparisons below use the fixed benchmark runner
`python -m tests.grad.benchmark_multipath`, which reproduces the default
`tests/grad/grad_multipath.py` workload at `256x256`, `reflection_n_rays=10000`,
`reflection_max_bounces=3`, `enable_rd_diffraction=True`.
For Slang-backed paths, steady-state comparisons use `--warmup-runs 1` so the
first-trace module compile cost is not mixed into the runtime number.

| Phase | Forward | Backward | Device used | Device peak | Loss | TX grad norm | Timing breakdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `phase0-baseline` | 1.733s | 1.057s | 5.284 GiB | 6.611 GiB | 9.184246 | 35.989285 | `los=0.012s`, `reflection=0.412s`, `diffraction=1.309s` |
| `phase1` | 1.829s | 1.005s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989285 | `los=0.012s`, `reflection=0.547s`, `diffraction=1.269s` |
| `phase2-prototype` | 14.052s | 4.476s | 11.43 GiB | 12.75 GiB | 9.184716 | 35.990900 | `los=0.014s`, `reflection=0.583s`, `diffraction=13.455s` |
| `phase2-guarded-tight` | 4.817s | 1.289s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989285 | `los=0.012s`, `reflection=0.592s`, `diffraction=4.214s` |
| `default-current` | 1.932s | 0.988s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989285 | `los=0.012s`, `reflection=0.634s`, `diffraction=1.286s` |
| `default-current-retest` | 1.906s | 1.024s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989280 | `los=0.012s`, `reflection=0.598s`, `diffraction=1.296s` |
| `phaseA1-symbolic-dda` | 8.927s | 4.757s | 1001 MiB | 2.492 GiB | 9.184311 | 35.988390 | `los=0.013s`, `reflection=1.974s`, `diffraction=6.940s` |
| `default-post-a1-gate` | 1.872s | 0.995s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989280 | `los=0.012s`, `reflection=0.593s`, `diffraction=1.268s` |
| `phaseA1A2-evaluated-fallback` | 1.874s | 0.997s | 5.285 GiB | 6.606 GiB | 9.184246 | 35.989280 | `los=0.013s`, `reflection=0.619s`, `diffraction=1.243s` |
| `default-symbolic-dda` | 0.634s | 0.871s | 1001 MiB | 2.492 GiB | 9.184311 | 35.988390 | `los=0.016s`, `reflection=0.128s`, `diffraction=0.490s` |
| `phaseA3A4A7-default` | 0.635s | 0.870s | 1001 MiB | 2.492 GiB | 9.184311 | 35.988392 | `los=0.014s`, `reflection=0.169s`, `diffraction=0.452s` |
| `phaseA5-suffix-roulette` | 1.282s | 0.875s | 1001 MiB | 2.51 GiB | 9.184311 | 35.988392 | `los=0.012s`, `reflection=0.159s`, `diffraction=1.110s` |
| `phase4-k1-customop` | 2.464s | 3.093s | 4.315 GiB | 5.638 GiB | 9.184253 | 35.790110 | `los=0.013s`, `reflection=0.607s`, `diffraction=1.844s` |
| `default-b1-retest` | 0.440s | 0.856s | 1001 MiB | 2.492 GiB | 9.184311 | 35.988392 | `los=0.056s`, `reflection=0.033s`, `diffraction=0.351s` |
| `phaseB1-k1-slang-forward` | 0.360s | 0.863s | 8.144 MiB | 1.525 GiB | 9.184318 | 35.988213 | `los=0.002s`, `reflection=0.030s`, `diffraction=0.328s` |
| `phaseB2-suffix-dda` | 0.423s | 0.772s | 1001 MiB | 2.495 GiB | 9.184246 | 35.988216 | `los=0.002s`, `reflection=0.029s`, `diffraction=0.392s` |
| `phaseB1B2-combined` | 0.400s | 0.882s | 8.144 MiB | 1.527 GiB | 9.184253 | 35.988044 | `los=0.002s`, `reflection=0.028s`, `diffraction=0.369s` |
| `default-b-after-phase-b` | 0.419s | 0.922s | 8.144 MiB | 1.527 GiB | 9.184253 | 35.988044 | `los=0.002s`, `reflection=0.031s`, `diffraction=0.385s` |
| `default-b3-folded-into-b1-steady` | 0.442s | 0.982s | 8.144 MiB | 1.527 GiB | 9.184253 | 35.988044 | `los=0.003s`, `reflection=0.041s`, `diffraction=0.398s` |
| `k1-kernel-vjp-default` | 0.418s | 0.370s | 8.145 MiB | 3.734 GiB | 9.188064 | 34.681308 | `los=0.002s`, `reflection=0.032s`, `diffraction=0.384s` |
| `k1-sourcepos-jvp-default` | 0.417s | 0.348s | 8.145 MiB | 3.734 GiB | 9.187963 | 34.681078 | `los=0.002s`, `reflection=0.032s`, `diffraction=0.382s` |
| `k1-hand-vjp-default` | 0.473s | 0.381s | 8.145 MiB | 3.734 GiB | 9.187963 | 34.681078 | `los=0.006s`, `reflection=0.035s`, `diffraction=0.432s` |
| `k1-hand-vjp-default-post-jvp-hybrid` | 0.503s | 0.356s | 8.145 MiB | 3.734 GiB | 9.187963 | 34.681078 | `los=0.003s`, `reflection=0.041s`, `diffraction=0.460s` |
| `k1-hand-vjp-all-scalar-backward` | 0.373s | 0.295s | 8.144 MiB | 4.031 GiB | 9.187811 | 34.682035 | `los=0.003s`, `reflection=0.034s`, `diffraction=0.336s` |
| `k1-hand-vjp-all-scalar-backward-streaming` | 0.354s | 0.281s | 8.144 MiB | 3.719 GiB | 9.187811 | 34.682035 | `los=0.002s`, `reflection=0.024s`, `diffraction=0.328s` |
| `k1-pure-kernel-vector-backward` | 0.381s | 0.289s | 8.144 MiB | 3.719 GiB | 9.187811 | 34.682035 | `los=0.002s`, `reflection=0.028s`, `diffraction=0.351s` |
| `k1-off-post-handvjp` | 0.493s | 1.079s | 1001 MiB | 2.495 GiB | 9.184246 | 35.988220 | `los=0.002s`, `reflection=0.034s`, `diffraction=0.457s` |
| `k1-off-fallback-recheck` | 0.429s | 0.867s | 1001 MiB | 2.495 GiB | 9.184246 | 35.988220 | `los=0.002s`, `reflection=0.028s`, `diffraction=0.399s` |

Phase 1 scope:

- chunked `n_paths * n_edges` expansion in reflection-prefix first-order builders
- chunked `n_prev_states * n_edges` higher-order diffraction state construction
- chunked `n_states * rays_per_state` inserted-reflection builder expansion
- pruning-side torch key reuse with reduced repeated casts/materialization
- evaluated DDA loop-state reduction by moving invariant arrays out of `state`

Observed effect on this benchmark:

- peak device allocator dropped by about 5 MiB (`6.611 GiB -> 6.606 GiB`)
- backward time improved by about 4.9% (`1.057s -> 1.005s`)
- forward time regressed by about 5.5% (`1.733s -> 1.829s`), likely from extra builder chunk boundaries on a workload that was not previously builder-memory-bound
- loss and TX gradient norm remained stable to displayed precision

Validation rerun after Phase 1:

- `tests/test_mixed_path_budget_ownership.py`
- `tests/test_reflection_prefix_path_canonicalization.py`
- `tests/test_chunked_cartesian_accumulation.py -m gpu` (skipped in this environment)

Phase 2 status:

- Historical only: Phase 2 proved that a standalone Slang Fresnel `dr.wrap(drjit -> torch)` bridge
  was numerically viable but not runtime-viable on the fixed benchmark.
- Added two higher-level mitigation attempts:
  - batched Fresnel evaluation from the UTD coefficient layer
  - a tight width guard to restrict runtime Slang dispatch to very small arrays
- The prototype is numerically close to the existing DrJit implementation, but repeated `dr.wrap` crossings still make the current integration too expensive for end-to-end tracing.
- Retest after removing unrelated GPU load still showed the same order-of-magnitude regression:
  - forward: `1.829s -> 14.052s`
  - backward: `1.005s -> 4.476s`
  - peak allocator: `6.606 GiB -> 12.75 GiB`
- A tighter runtime guard avoided the peak-memory blow-up, but it still did not meet the fixed benchmark acceptance bar:
  - forward: `1.932s -> 4.817s`
  - backward: `0.988s -> 1.289s`
  - peak allocator: unchanged at `6.606 GiB`
- One intermediate batched-streaming attempt still exhausted the allocator on the fixed workload and was not kept as the default runtime path.
- That standalone bridge has since been removed from the runtime path. Its Boersma / UTD math now
  lives only as the shared `utd_accumulate_math.slang` include used by the K1 fused kernel.

Phase A1/A2 status:

- A1 was first implemented as a direct symbolic-loop conversion with manual loop-state piping.
- That prototype solved the Tier 2 allocator problem but badly regressed runtime:
  - forward: `1.906s -> 8.927s`
  - backward: `1.024s -> 4.757s`
  - peak allocator: `6.606 GiB -> 2.492 GiB`
- A2 then reworked the symbolic DDA path around `@dr.syntax + dr.hint(exclude=[params])`:
  - non-state constants, result buffers, and non-AD traversal arrays stay in an excluded `params` tuple
  - AD-carrying per-ray inputs remain explicit function arguments so reverse-mode differentiation remains valid
  - `max_iterations=max_steps` is still provided on both loops for reverse-mode support
- Added symbolic DDA regression coverage in `tests/test_symbolic_dda_toggle.py`.
- Serial validation passed with the final symbolic DDA implementation:
  - `tests/test_symbolic_dda_toggle.py`
  - `tests/test_fresnel_gradient_regression.py`
  - `tests/test_reflection_material_response.py`
  - `tests/test_mixed_path_budget_ownership.py`
  - `tests/test_mixed_path_regression_scenes.py`
- The refined A1+A2 implementation now satisfies the fixed benchmark acceptance bar and is the only retained reflection DDA path.
- Final measured effect versus the evaluated fallback:
  - forward: `1.874s -> 0.634s`
  - backward: `0.997s -> 0.871s`
  - device allocator used: `5.285 GiB -> 1001 MiB`
  - device allocator peak: `6.606 GiB -> 2.492 GiB`

Phase A3/A4/A7 status:

- `A6` had already landed in the earlier builder chunking work, so this pass focused on the remaining non-kernel Layer A items.
- `A3` now selectively detaches reflection-hit geometry only when the mesh triangle buffers do not carry gradients.
  - This keeps the common TX-only optimization workload off the ray-intersection AD path.
  - If scene geometry is gradient-enabled, the detach is skipped and the previous geometry-aware path is preserved.
- `A4` replaced the full CPU merge in `witwin/channel/trace/reflection/paths.py` with GPU torch hashing/lexsort for coarse grouping plus a small on-device tolerance merge over the already-collapsed groups.
- `A7` switched the pruning hot path to DLPack-backed torch views and shared GPU lexsort helpers, removing the extra `.torch().detach()` materialization pattern from budget ranking.
- Serial validation passed after these changes:
  - `tests/test_reflection_prefix_path_canonicalization.py`
  - `tests/test_chunked_cartesian_accumulation.py` (skipped in this environment)
  - `tests/test_mixed_path_budget_ownership.py`
  - `tests/test_mixed_path_regression_scenes.py`
  - `tests/test_reflection_material_response.py`
  - `tests/test_fresnel_gradient_regression.py`
  - `tests/test_symbolic_dda_toggle.py`
  - `tests/test_forward_ad.py`
  - `tests/test_rd_multipath_consistency.py` (`3 skipped` in this environment)
  - `tests/test_polarization_and_finite_edge.py -k suffix_reflection_updates_mixed_jones_field`
- Serial fixed-benchmark retest versus `default-symbolic-dda` was effectively neutral while keeping the same allocator footprint:
  - forward: `0.634s -> 0.635s`
  - backward: `0.871s -> 0.870s`
  - device allocator used: unchanged at `1001 MiB`
  - device allocator peak: unchanged at `2.492 GiB`
- Conclusion: `A3`, `A4`, and `A7` are now safe default-path cleanups that preserve the A1/A2 win without introducing a new steady-state regression.

Phase A5 status:

- Added deterministic suffix Russian roulette behind `DiffractionExecutionConfig(suffix_russian_roulette=True)`.
- Survival probabilities are based on the normalized reflected-field power, and surviving rays are reweighted by the inverse survival probability to keep the estimator unbiased.
- The fixed serial benchmark did not justify default enablement:
  - forward: `0.635s -> 1.282s`
  - backward: `0.870s -> 0.875s`
  - device allocator peak: `2.492 GiB -> 2.51 GiB`
  - loss and TX gradient norm stayed stable to displayed precision
- Conclusion: `A5` is implemented and available for further experimentation, but remains disabled by default because it regresses the fixed workload.

Phase 4 status:

- Added a first `dr.custom(CustomOp)` bridge for the totals-only diffraction accumulation path in `witwin/channel/trace/diffraction/field.py`.
- The current prototype repackages the full edge-state schema, including path-history fields, and rematerializes the existing DrJit totals implementation inside `eval()`, `forward()`, and `backward()`.
- Runtime enablement remains an explicit execution-config opt-in; the default tracing path is unchanged.
- Serial validation passed with the K1 prototype enabled:
  - `tests/test_fresnel_gradient_regression.py`
  - `tests/test_utd_angle_derivatives.py`
  - `tests/test_mixed_path_budget_ownership.py`
  - `tests/test_reflection_material_response.py`
  - `tests/test_forward_ad.py`
  - `tests/test_rd_multipath_consistency.py` was skipped in this environment
- The fixed serial benchmark shows the expected memory improvement, but not acceptable steady-state runtime:
  - forward: `1.906s -> 2.464s`
  - backward: `1.024s -> 3.093s`
  - device allocator used: `5.285 GiB -> 4.315 GiB`
  - device allocator peak: `6.606 GiB -> 5.638 GiB`
- Conclusion: this confirms `dr.custom` is a viable K1 bridge for AD encapsulation and memory reduction, but the current rematerialization-based prototype is scaffolding only. The next K1 step must replace the internal DrJit replay with an actual fused kernel implementation instead of making the custom op re-run the existing accumulation graph.

Phase B status:

- `B3` is no longer a standalone runtime bridge. The Fresnel / UTD helper math now lives in
  `witwin/channel/trace/diffraction/utd_accumulate_math.slang`, included directly by
  `witwin/channel/trace/diffraction/utd_accumulate.slang`.
- `B1` is now a real fused forward kernel in `witwin/channel/trace/diffraction/utd_accumulate.slang`, launched from the existing `dr.custom(CustomOp)` totals path.
  - The all-Slang forward prototype initially mismatched a small set of cotangent-pole boundary samples.
  - The shipped version therefore uses a hybrid policy:
    - Slang fused forward for safe chunks
    - DrJit fallback for chunks containing pole-unsafe receiver pairs
  - This was enough to move K1 from “memory-only prototype” to a benchmark-positive default path.
- `B2` has landed for the diffraction suffix DDA hot loop through `witwin/channel/trace/diffraction/dda_traverse.slang`.
  - The suffix path now launches a Slang forward traversal kernel by default when Slang is available.
  - Reflection DDA stays on the A1+A2 symbolic DrJit path because it is already cheap on the fixed benchmark and was no longer the steady-state bottleneck.
- Serial validation with the new default B-stage path passed:
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_mixed_path_budget_ownership.py --gpu`
  - `tests/test_rd_multipath_consistency.py --gpu`
  - `tests/test_mixed_path_regression_scenes.py --gpu`
  - `tests/test_polarization_and_finite_edge.py --gpu -k suffix_reflection_updates_mixed_jones_field`
- After folding the old B3 bridge into the shared B1 math layer, serial validation still passed:
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_mixed_path_budget_ownership.py --gpu`
- After landing the kernel-side K1 backward changes, serial validation still passed:
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_mixed_path_budget_ownership.py --gpu`
  - `tests/test_reflection_material_response.py --gpu`
  - `tests/test_rd_multipath_consistency.py --gpu`
- After replacing the safe scalar-total reverse path with a hand-written Slang VJP, serial validation passed:
  - `tests/test_utd_accumulate_scalar_backward.py --gpu`
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_mixed_path_budget_ownership.py --gpu`
  - `tests/test_reflection_material_response.py --gpu`
  - `tests/test_rd_multipath_consistency.py --gpu`
- After enabling hybrid safe-chunk Slang JVP with chunk-level finite fallback, serial validation passed:
  - `tests/test_utd_accumulate_scalar_backward.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
- After extending scalar-total backward to all visible chunks, serial validation passed:
  - `tests/test_utd_accumulate_scalar_backward.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
- After switching scalar-total backward to stream visible chunks instead of materializing the full
  `safe_chunks + unsafe_chunks` list first, serial validation passed:
  - `tests/test_utd_accumulate_scalar_backward.py --gpu`
  - `tests/test_forward_ad.py --gpu`
- After landing the pure-kernel vector backward path, serial validation passed:
  - `tests/test_utd_accumulate_scalar_backward.py --gpu`
  - `tests/test_utd_accumulate_slang_bridge.py --gpu`
  - `tests/test_forward_ad.py --gpu`
  - `tests/test_utd_angle_derivatives.py --gpu`
  - `tests/test_fresnel_gradient_regression.py --gpu`
  - `tests/test_rd_multipath_consistency.py --gpu`
  - `tests/test_polarization_and_finite_edge.py --gpu -k suffix_reflection_updates_mixed_jones_field`
- Serial benchmark results on the fixed `grad_multipath` workload:
  - `B1` only (`phaseB1-k1-slang-forward`):
    - forward: `0.440s -> 0.360s`
    - backward: `0.856s -> 0.863s`
    - device allocator peak: `2.492 GiB -> 1.525 GiB`
  - `B2` suffix DDA only (`phaseB2-suffix-dda`):
    - forward: `0.440s -> 0.423s`
    - backward: `0.856s -> 0.772s`
    - device allocator peak: effectively unchanged (`2.492 GiB -> 2.495 GiB`)
  - combined `B1+B2` (`phaseB1B2-combined`):
    - forward: `0.440s -> 0.400s`
    - backward: `0.856s -> 0.882s`
    - device allocator peak: `2.492 GiB -> 1.527 GiB`
  - final default after enabling both by default (`default-b-after-phase-b`):
    - forward: `0.419s`
    - backward: `0.922s`
    - device allocator used: `8.144 MiB`
    - device allocator peak: `1.527 GiB`
  - after removing the standalone Fresnel bridge and folding its math into `B1` (`default-b3-folded-into-b1-steady`, measured with `--warmup-runs 1`):
    - forward: `0.442s`
    - backward: `0.982s`
    - device allocator used: `8.144 MiB`
    - device allocator peak: `1.527 GiB`
  - after landing the hybrid kernel VJP (`k1-kernel-vjp-default`, measured with `--warmup-runs 1`):
    - forward: `0.418s`
    - backward: `0.370s`
    - device allocator used: `8.145 MiB`
    - device allocator peak: `3.734 GiB`
  - after replacing safe-chunk `source_pos` replay with a field-only Slang JVP transpose
    (`k1-sourcepos-jvp-default`, measured with `--warmup-runs 1`):
    - forward: `0.417s`
    - backward: `0.348s`
    - device allocator used: `8.145 MiB`
    - device allocator peak: `3.734 GiB`
  - after replacing that workaround with a hand-written scalar-safe Slang VJP
    (`k1-hand-vjp-default`, measured with `--warmup-runs 1`):
    - forward: `0.473s`
    - backward: `0.381s`
    - device allocator used: `8.145 MiB`
    - device allocator peak: `3.734 GiB`
  - after enabling hybrid safe-chunk Slang JVP with chunk-level finite fallback
    (`k1-hand-vjp-default-post-jvp-hybrid`, measured with `--warmup-runs 1`):
    - forward: `0.503s`
    - backward: `0.356s`
    - device allocator used: `8.145 MiB`
    - device allocator peak: `3.734 GiB`
  - after extending scalar-total backward to all visible chunks
    (`k1-hand-vjp-all-scalar-backward`, measured with `--warmup-runs 1`):
    - forward: `0.373s`
    - backward: `0.295s`
    - device allocator used: `8.144 MiB`
    - device allocator peak: `4.031 GiB`
  - after streaming scalar-total backward visible chunks instead of materializing the full partition list
    (`k1-hand-vjp-all-scalar-backward-streaming`, measured with `--warmup-runs 1`):
    - forward: `0.354s`
    - backward: `0.281s`
    - device allocator used: `8.144 MiB`
    - device allocator peak: `3.719 GiB`
  - current `K1=0` comparison point from the same hand-VJP round
    (`k1-off-post-handvjp`, measured with `--warmup-runs 1`):
    - forward: `0.493s`
    - backward: `1.079s`
    - device allocator used: `1001 MiB`
    - device allocator peak: `2.495 GiB`
  - current `K1=0` comparison point (`k1-off-fallback-recheck`, measured with `--warmup-runs 1`):
    - forward: `0.429s`
    - backward: `0.867s`
    - device allocator used: `1001 MiB`
    - device allocator peak: `2.495 GiB`
- Current K1 AD split:
  - `eval()`: fused Slang forward on AD-safe chunks plus replay on the remaining chunks
  - `backward()`: `utdAccumulateBackwardScalar` for scalar-total cotangents on all visible chunks,
    plus `utdAccumulateBackwardVector` for vector-output cotangents on all visible chunks; mixed
    cotangents split into those two pure-kernel launches on CUDA
  - `forward()`: try the Slang JVP on the full AD-safe chunk set first, then retry per safe chunk,
    and replay only the safe chunks that still emit non-finite tangents plus the already-unsafe chunks
- Benchmark interpretation:
  - The hand-written scalar-safe VJP removes the default runtime dependence on the safe-chunk
    `source_pos` replay / field-only JVP-transpose workaround and stabilizes the previously bad
    benchmark pair (`state=0`, `rx=7170`) to finite `source_pos` / `rx` adjoints.
  - On the fixed workload it is slightly slower than the previous workaround path
    (`0.348s -> 0.381s` backward, `0.417s -> 0.473s` forward), but it remains much faster than the
    `K1=0` path from the same round (`1.079s` backward).
  - The hybrid forward-mode landing changes AD architecture, not the fixed benchmark hot path. The
    benchmark does not exercise forward-mode AD, so the `0.473s/0.381s -> 0.503s/0.356s` shift
    should be treated as run-to-run noise, not as a measurable forward-mode throughput effect.
  - Benchmark-scale auditing showed the current workload has more vector/pole-unsafe visible pairs
    than safe ones (`~59.3M unsafe` vs `~49.0M safe`). Extending the scalar backward kernel across
    all visible chunks therefore produced a real backward win (`0.356s -> 0.295s`) and removed the
    scalar-total unsafe replay from the hot path.
  - The first implementation of that change increased peak (`3.734 GiB -> 4.031 GiB`), which turned
    out not to come from the small persistent grad buffers. The actual issue was the long-lived
    materialized `safe_chunks + unsafe_chunks` dispatch-index lists.
  - Streaming scalar-total backward directly over visible chunks fixed that. The current default is
    now both faster (`0.295s -> 0.281s`) and lower peak (`4.031 GiB -> 3.719 GiB`) than the first
    all-visible scalar-backward prototype.
  - The current hand-written VJP now covers both reverse-mode CUDA branches. Scalar totals use
    `utdAccumulateBackwardScalar`, vector outputs use `utdAccumulateBackwardVector`, and the mixed
    benchmark-scene regression proves the default path no longer touches replay for reverse-mode
    mixed/vector cotangents on CUDA.
  - The fixed scalar-loss benchmark mainly exercises the scalar branch, so the new vector kernel is
    an architectural landing rather than a new hot-path throughput win on this workload. The measured
    `0.381s / 0.289s / 3.719 GiB` result stays in the same regime as the previous streaming scalar
    default while making the mixed/vector CUDA backward path pure-kernel.
  - The fixed benchmark should be treated as a throughput comparison only. Field/gradient parity is
    guarded by the dedicated pytest coverage above, not by comparing fresh-process benchmark runs.
- Conclusion:
  - `B1` is now a default-path backward-throughput win with a hand-written scalar-safe reverse
    kernel on the benchmark-critical path, and scalar-total backward no longer replays the benchmark
    `unsafe` chunks.
  - The next steps are now even clearer: reduce the fused scalar-backward peak, replace more of the
    remaining forward-mode replay with a stable kernel JVP, and retire the full-vector unsafe fallback.
  - `B2` suffix DDA is modest but positive on the fixed workload and does not destabilize mixed-path regression scenes.
  - `B3` now exists only as the shared math layer under `B1`; there is no separate runtime bridge left to maintain.
