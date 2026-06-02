# Channel Memory Optimization & CUDA Kernel Candidates

## Overview

This document analyzes the core memory bottlenecks in `witwin.channel.trace` and identifies
components that would benefit from dedicated GPU kernels to reduce intermediate memory allocation.

Current state: the chunked Cartesian accumulation patch (see `multipath_performance_scaling_report.md`)
reduced peak VRAM by up to 57%, but the pipeline still creates large numbers of intermediate DrJit
arrays per chunk. The root cause is that DrJit's lazy-evaluation model allocates a separate GPU buffer
for every intermediate expression, and multi-step computations (gather → compute → scatter) cannot be
fused across DrJit operations.

## Execution Notes

- Fixed benchmark entrypoint: `tests/grad/benchmark_multipath.py`, which reproduces the
  default `tests/grad/grad_multipath.py` workload (`256x256`, `reflection_n_rays=10000`,
  `reflection_max_bounces=3`, `enable_rd_diffraction=True`) and reports:
  - `Tracer.trace(..., return_timing=True)` timing
  - `loss = sum(|a_total|^2)` backward wall time
  - `dr.whos()` device allocator used/peak
  - total-field loss and TX gradient norm
- Benchmark collection must be serialized. Run one benchmark process at a time and avoid
  competing GPU workloads while recording phase comparisons.
- When a phase changes Slang sources, use `python -m tests.grad.benchmark_multipath --warmup-runs 1`
  for the steady-state comparison so one-time per-process module compilation does not pollute the
  runtime number.
- Q3 in the quick-win list is not primarily about `.detach()` itself. The larger issue is the
  pruning path materializing extra torch temporaries during dtype conversion, lexicographic
  key stacking, and sorting. Reducing repeated `.torch()` calls and avoiding unnecessary casts
  is the first fix; a deeper zero-copy bridge is optional follow-up work.
- K1 cannot use `dr.wrap(...)` because the fused accumulation kernel performs internal atomic
  adds. The correct bridge is `dr.custom(CustomOp)` so the operation remains a pure input-output
  node from DrJit's perspective while encapsulating side effects inside the kernel.
- K3 overlaps heavily with K1. If K1 lands with packed-state gather + UTD evaluation + atomic
  accumulation, K3 should be removed from the roadmap instead of maintained as a second fused path.
- The standalone Slang Fresnel `dr.wrap(drjit -> torch)` bridge has been retired. Its Boersma /
  UTD math now lives inside the K1 kernel module as the shared `utd_accumulate_math.slang`
  layer, so there is no separate Slang-Fresnel runtime toggle anymore.
- The first K1 bridge is now implemented with `dr.custom(CustomOp)` in the totals-only
  diffraction accumulation path, selected through `DiffractionExecutionConfig(accumulate_primal="custom_op_partitioned", ...)`.
  On the fixed serial benchmark it reduced device peak memory from `6.606 GiB` to
  `5.638 GiB`, but backward time regressed from `1.024s` to `3.093s` because the prototype
  still rematerializes the existing DrJit accumulation in `eval()/forward()/backward()`.
  This validates the bridge shape, not the final performance target.
- K1 has now been upgraded beyond the replay prototype:
  - `eval()` uses the fused Slang totals kernel on AD-safe chunks and replays only the remaining chunks.
  - `backward()` on CUDA now splits into two pure-kernel branches at the host bridge:
    - `utdAccumulateBackwardScalar` handles scalar-total cotangents with a hand-written guarded VJP
      for the scalar UTD path, including explicit `phi/phi'`, `s/s'`, and `D/D_slope` adjoints
      plus the geometric reverse pass from the cached oriented-angle representation.
    - `utdAccumulateBackwardVector` handles vector-output cotangents with a hand-written transport
      adjoint plus a kernel-side gain adjoint that reconnects `directGain/derivativeGain` back into
      the existing guarded scalar UTD reverse pass.
  - The remaining `source_pos` issue did **not** turn out to be vector-transport-only. A temporary
    field-only kernel confirmed that `bwd_diff(...)` on the scalar UTD field path still emitted
    `NaN` `source_pos` / `rx` adjoints on some benchmark-safe pairs, even with zero vector
    cotangents. The hand-written scalar VJP replaces that path and removes the earlier safe-chunk
    `source_pos` replay / field-only JVP-transpose workaround from the default runtime.
  - The hand-written scalar-safe VJP has pair-level regression coverage for both replay parity on a
    small safe chunk and a previously unstable benchmark pair (`state=0`, `rx=7170`), which now
    produces finite `source_pos` / `rx` adjoints.
  - Benchmark-scale validation then showed that the scalar backward kernel is also numerically close
    on the current `unsafe` benchmark chunks when the output cotangent is scalar-only. The default
    K1 backward path now uses `utdAccumulateBackwardScalar` across **all visible chunks** whenever
    scalar totals are present, and `utdAccumulateBackwardVector` across **all visible chunks**
    whenever vector cotangents are present.
  - The scalar-only backward entrypoint no longer materializes the full `safe_chunks + unsafe_chunks`
    partition list first. It streams visible chunks directly through `accumulate_edge_state_totals_slang_backward(...)`,
    which removes the long-lived dispatch-index lists from the benchmark-critical scalar-total path.
  - `forward()` now uses a hybrid safe-chunk JVP policy:
    - first try the full safe-chunk Slang JVP on the whole AD-safe set
    - if that emits non-finite tangents, retry per safe chunk
    - only the still-non-finite safe chunks, plus all already-unsafe chunks, fall back to DrJit replay
  - This keeps forward-mode AD on a numerically EPC fallback when needed, while no longer
    forcing the entire forward-mode path through replay on scenes where the Slang JVP is already finite.
  - The fused K1 bridge is currently an explicit execution-config opt-in; keep it disabled by default until
    its first-launch CUDA path is repeatable across identical forward traces.
- The fused K1 forward path keeps a conservative DrJit fallback for chunks containing
  cotangent-pole-unsafe receiver pairs. Those chunks were the only observed source of large
  forward mismatches in the all-Slang prototype, so the default path now uses a hybrid
  safe-chunk Slang / unsafe-chunk DrJit split.
- A1+A2 are now implemented as the default DDA path. The final implementation uses
  `@dr.syntax + dr.hint(exclude=[params])` plus explicit AD-carrying loop inputs instead of
  the earlier manual loop-state piping prototype. The reflection DDA path now keeps only the
  symbolic implementation.
- B2 has now landed for the diffraction suffix DDA hot loop in
  `witwin/channel/trace/diffraction/dda_traverse.slang`. Suffix traversal is now selected explicitly
  through `DiffractionExecutionConfig(suffix_dda="symbolic"|"evaluated"|"slang")` instead of an
  environment-variable toggle. Reflection DDA remains on the optimized symbolic DrJit path
  because it is no longer the dominant steady-state cost on the fixed benchmark.
- The final fixed serial benchmark comparison is:
  - evaluated fallback: forward `1.874s`, backward `0.997s`, peak `6.606 GiB`
  - default symbolic DDA: forward `0.634s`, backward `0.871s`, peak `2.492 GiB`
- After the B-stage fused-kernel landing, the earlier default benchmark was:
  - default after Phase B: forward `0.419s`, backward `0.922s`, peak `1.527 GiB`
- After the kernel-side VJP landing on K1, the current serial benchmark comparison is:
  - earlier kernel-VJP default `K1=1`: forward `0.418s`, backward `0.370s`, peak `3.734 GiB`
  - earlier source-pos-JVP default `K1=1`: forward `0.417s`, backward `0.348s`, peak `3.734 GiB`
  - current hand-VJP default `K1=1`: forward `0.473s`, backward `0.381s`, peak `3.734 GiB`
  - current hand-VJP + hybrid forward-JVP default `K1=1`: forward `0.503s`, backward `0.356s`, peak `3.734 GiB`
  - current hand-VJP + all-visible scalar-backward streaming default `K1=1`: forward `0.354s`, backward `0.281s`, peak `3.719 GiB`
  - current pure-kernel vector-backward default `K1=1`: forward `0.381s`, backward `0.289s`, peak `3.719 GiB`
  - current `K1=0` fallback recheck: forward `0.493s`, backward `1.079s`, peak `2.495 GiB`
- Interpretation:
  - K1 backward is now fully kernel-side on scalar-total chunks; the earlier safe-chunk
    `source_pos` replay / JVP-transpose workaround is gone from the default runtime path, and
    scalar-total cotangents no longer replay the current `unsafe` benchmark chunks either.
  - The hand-written scalar VJP is slightly slower than the previous source-pos-JVP workaround on
    this benchmark, but it is still much faster than `K1=0` on backward (`0.381s` vs `1.079s`) and
    it fixes the previously unstable scalar-safe `source_pos` adjoint path.
  - The new forward-mode landing does not materially change the fixed throughput benchmark because
    `grad_multipath` measures reverse-mode, not forward-mode. The small steady-state drift
    (`0.473s/0.381s -> 0.503s/0.356s`) should be treated as normal benchmark variance, not as a
    forward-mode throughput claim.
  - Extending the scalar backward kernel from safe chunks to **all visible chunks** is a real
    throughput win on the fixed workload because the benchmark has more currently-unsafe than safe
    visible pairs (`~59.3M unsafe` vs `~49.0M safe`).
  - The first all-visible scalar-backward attempt improved throughput but raised peak because it
    still kept the full `safe_chunks + unsafe_chunks` dispatch-index lists alive. Streaming visible
    chunks directly through the scalar backward path fixed that: the current retest improved again
    (`0.356s -> 0.281s`) and reduced peak below the previous hand-VJP default (`3.734 GiB -> 3.719 GiB`).
  - Reverse-mode CUDA backward is now pure-kernel on both branches: scalar-total cotangents route
    through `utdAccumulateBackwardScalar`, vector-output cotangents route through the new
    `utdAccumulateBackwardVector`, and mixed cotangents are split into those two kernel launches.
  - The fixed `grad_multipath` workload still measures the scalar-loss path, so the new vector
    kernel mainly changes architecture and mixed/polarization workloads rather than this benchmark.
    On the fixed scalar benchmark it stays in the same throughput/peak regime as the previous
    streaming scalar-default (`0.381s / 0.289s / 3.719 GiB` versus `0.354s / 0.281s / 3.719 GiB`).
  - Even with that increase, the default path remains well below the original Phase 0 peak
    (`6.611 GiB`), so this is still a net win versus the historical baseline.
- The earlier A1-only symbolic prototype (`8.927s / 4.757s`) is kept in the report as a
  historical baseline only; it is no longer representative of the current implementation.

---

## Tier 1 — Cartesian Pair Expansion (Highest Impact)

Even with `CARTESIAN_PAIR_CHUNK_BUDGET = 2^25`, each chunk materializes ~33M elements across ~50+
DrJit arrays simultaneously.

| Location | Arrays per chunk | Est. peak per chunk |
|---|---|---|
| `diffraction/field.py:222-306` `_accumulate_edge_states_to_receivers` | ~50 (gather ~25 + UTD ~15 + scatter ~10) | 1.5–2 GB |
| `diffraction/suffix.py:136-269` `_accumulate_reflected_segment_fields_chunk` | 9 result arrays of `n_states × n_rx` + DDA loop ~22 | 1–3 GB |
| `reflection/field.py:31-126` `_accumulate_reflection_paths_to_receivers` | ~30 (paths + chain eval + scatter) | ~1 GB |

### Root cause in `_accumulate_edge_states_to_receivers`

`_gather_state_arrays` (state.py:328) creates **25+ new arrays** of size `n_pairs` in a single call.
The subsequent `_edge_state_field_to_targets` then creates another ~20 arrays for UTD evaluation
(angles, Fresnel integrals, coefficients, phase, etc.).  None of these intermediates can be fused
by DrJit because each is a separate traced operation.

---

## Tier 2 — DDA Loop State Explosion

Both DDA implementations carry ~20 state variables through `dr.while_loop(mode="evaluated")`, which
materializes **all state variables at every iteration**.

| Path | State elements | Params | Loop bound |
|---|---|---|---|
| `reflection/dda.py:12-213` | 20 | 28 | `2*(nx+ny)` |
| `diffraction/suffix.py:30-133` | 22 | 22 | `2*(nx+ny)` |

For 256×256 grid, 5000 rays: `max_steps ≈ 1024`, evaluated state per iteration =
`22 × n_rays × 4B ≈ 440 MB` for reflection DDA alone.

The suffix DDA is worse: its result buffers are `n_states × n_rx`, which at 256×256 with 1648 states
means 9 arrays × `1648 × 65536 × 4B ≈ 3.6 GB` for results alone.

---

## Tier 3 — State Concatenation & Pruning

### `_concat_state_arrays` (state.py:206-304)

Concatenates ~30 fields using per-field `dr.zeros` → `dr.scatter` → `dr.eval`.  For 2–3 non-empty
sources this triples the intermediate footprint during merge.

### `_prune_state_arrays_by_budget` (state.py:391-446)

Converts ~10+ DrJit arrays to PyTorch tensors (`.torch().detach()`), causing **double allocation**
(DrJit buffer + PyTorch tensor).  `_torch_lexsort` creates additional sort buffers:
O(n_states × num_keys).  Peak: ~3× base state memory during pruning.

---

## Tier 4 — Builder Cartesian Products (Unbounded)

These Cartesian products have **no chunking** applied:

| Location | Product | Typical size |
|---|---|---|
| `builders.py:169` `_build_reflection_first_order_state_arrays` | `n_paths × n_edges` | 50 × 100 = 5K (manageable now, grows with scene) |
| `builders.py:309` `_build_higher_order_state_arrays` | `n_prev_states × n_edges` | 768 × 100 = 76.8K after pruning |
| `builders.py:471` `_build_inserted_reflection_state_arrays` | `n_states × rays_per_state` | 768 × 128 = 98K rays, each with ~20 gathered fields |

---

## Tier 5 — UTD Computation Intermediate Bloat

### `_edge_state_field_to_targets` (field.py:44-187)

For `width` elements creates ~40+ intermediate arrays:
- 4 angle arrays (phi, phi_prime, s, s_prime)
- 6 broadcast geometry arrays (3 components each)
- 4 mask arrays
- 2 complex diffraction coefficients
- 2 scale/phase arrays
- 2 broadcast incident fields
- Angle derivative + vector transport (~10 more)

At `width = 33M` (chunk size): ~5 GB of intermediates.

### `fresnel_integral` (utd.py:28-100)

Creates `arg, arg^2, ..., arg^11` = 11 power arrays, plus 24 `dr.select` results for the
12-term Boersma polynomial (×2 for real/imag).  Called 4× per UTD coefficient.

---

## CUDA Kernel Candidates (Ranked by Impact)

### K1 — Fused UTD Accumulation Kernel (Highest Priority)

**Replaces**: `_accumulate_edge_states_to_receivers` inner loop.

```
Input:  state_arrays[n_states], rx_pos[n_rx], k, wavelength, material
Output: direct_total[n_rx], multi_total[n_rx]  (complex scalar + complex vector)
```

Single kernel: read state by index → compute angles → Fresnel integral → UTD coefficient →
phase + spreading → atomic-add to output.  No intermediate arrays materialized.

**Estimated savings**: eliminate ~50 intermediate arrays of chunk_size → 3–5 GB per trace.

Current status:

- A first `dr.custom(CustomOp)` prototype now wraps the totals-only path in
  `witwin/channel/trace/diffraction/field.py`.
- It already proves that K1 can cut allocator pressure before a fused kernel exists.
- It is no longer the original replay-only prototype.
- The fused forward milestone has landed, and reverse-mode is now partially fused as well:
  - `eval()`: Slang fused forward on AD-safe chunks
  - `backward()`: `utdAccumulateBackwardScalar` on scalar-total cotangents across all visible chunks,
    `utdAccumulateBackwardVector` on vector-output cotangents across all visible chunks, and mixed
    cotangents split into those two kernel launches on CUDA
  - `forward()`: try the Slang JVP on the full AD-safe set first, then retry per safe chunk, and
    replay only the safe chunks that still emit non-finite tangents plus the already-unsafe chunks
- This means the next K1 milestone is no longer "replace all forward-mode replay". The current
  forward path already uses Slang JVP opportunistically on finite safe chunks. The remaining
  milestone is specifically "tighten vector-kernel parity further and make the JVP reliable enough
  that the replay fallback is no longer needed on the benchmark-scale safe set".

### K2 — Fused DDA Traversal Kernel

**Replaces**: `_dda_loop_body` (reflection) and `_dda_segment_loop_body_batched` (suffix).

```
Input:  ray_origin[n_rays], ray_dir[n_rays], field state, grid geometry
Output: result[n_rx]  (accumulated field per cell)
```

Native CUDA ray-march loop — no state materialization per iteration, direct grid-cell atomic
accumulation.

**Estimated savings**: eliminate ~22 state arrays × n_iterations evaluated per step.

Current status:

- The first B2 landing now covers the diffraction suffix DDA traversal through
  `witwin/channel/trace/diffraction/dda_traverse.slang`.
- This is enabled by default when Slang is available and falls back to the symbolic DrJit
  suffix traversal otherwise.
- Reflection DDA is intentionally still on the symbolic DrJit implementation because the
  A1+A2 rewrite already made it cheap on the fixed benchmark, and the remaining hotspot was the
  diffraction suffix traversal.

### K3 — Fused State Gather + Field Evaluation

**Replaces**: `_gather_state_arrays` + `_edge_state_field_to_targets` pair.

```
Input:  state_arrays[n_states], indices[n_pairs], target_pos[n_pairs]
Output: field[n_pairs], vector[n_pairs]
```

Single kernel reads state by index, computes geometry + UTD, returns field.

**Estimated savings**: eliminate ~25 gather arrays + ~40 computation intermediates.

### K4 — Fused Fresnel Integral

**Replaces**: `fresnel_integral` and the Boersma polynomial evaluation.

Compute all 11 power terms in registers, evaluate polynomial in a single pass.

**Estimated savings**: eliminate ~30 intermediate arrays per call (called 4× per coefficient).

Current status:

- The standalone Slang Fresnel bridge is no longer part of the runtime architecture.
- The Boersma / UTD helper math from that experiment now exists only as the shared
  `utd_accumulate_math.slang` include used by the K1 fused accumulation kernel.

### K5 — Fused State Concatenation

**Replaces**: `_concat_arrays` / `_concat_state_arrays`.

Single kernel copies N source arrays into destination with precomputed offsets.

**Estimated savings**: eliminate ~60 temporary arrays during state merging.

---

## Quick Wins (No Kernel Required)

| # | Change | Location | Estimated savings |
|---|---|---|---|
| Q1 | Add chunking to `_build_higher_order_state_arrays` | builders.py:309 | 0.5–2 GB at peak |
| Q2 | Add chunking to `_build_reflection_first_order_state_arrays` | builders.py:169 | 0.2–1 GB |
| Q3 | Reduce pruning-side torch temporaries and repeated dtype casts; evaluate zero-copy bridging only after that | state.py:413-426 | 0.2–0.5 GB |
| Q4 | Force `dr.eval()` per chunk boundary in accumulation loops to release intermediates | field.py, suffix.py | variable |
| Q5 | Sparse/tiled result buffers in suffix DDA (most cells never hit) | suffix.py:163-171 | 1–3 GB |

---

## Summary Priority Table

| Priority | Target | Type | Est. Memory Savings |
|---|---|---|---|
| 1 | K1: UTD accumulation kernel | GPU kernel | 3–5 GB per trace |
| 2 | K2: DDA traversal kernel | GPU kernel | 1–3 GB per trace |
| 3 | Q1+Q2: Chunking in builders | Python | 0.5–2 GB at peak |
| 4 | K3: State gather + field fused | GPU kernel | 2–4 GB per chunk |
| 5 | Q3: Pruning zero-copy fix | Python | 0.2–0.5 GB |
| 6 | Q5+K5: Suffix sparse buffers / concat | Python or kernel | 1–3 GB |

---

## Implementation Language: Slang vs Raw CUDA

### Recommendation: Use Slang

The monorepo already has production Slang kernel infrastructure across three subprojects
(maxwell/FDTD, radar/chirp, core/mesh-SDF).  The channel kernels should follow the same path.

### Why Slang

| Factor | Slang | Raw CUDA |
|---|---|---|
| **Performance** | Compiles to PTX via NVRTC — same code generation backend, same instruction set. No runtime overhead. | Identical (both produce PTX). |
| **Differentiability** | `[Differentiable]` annotation generates backward kernels automatically. Already used in `mesh_sdf.slang`. For the channel path this matters because the tracer must be end-to-end differentiable for optimization workloads. | Must hand-write every backward kernel. Error-prone and doubles maintenance. |
| **Existing infra** | `slangtorch.loadModule()`, `module_cache.py`, compilation pipeline, `[AutoPyBindCUDA]` + `[CUDAKernel]` conventions — all established. | Would need new build system integration (CMake/setuptools CUDA extension or cupy JIT). |
| **PyTorch interop** | `TensorView<float>` maps directly to PyTorch tensors. Launch via `.launchRaw(blockSize, gridSize)`. | Same via pybind11/ctypes, but requires manual binding boilerplate. |
| **Syntax** | C#/HLSL-like. Generics, interfaces, operator overloading. Complex number structs trivially expressible. | C with extensions. Verbose for complex-valued math. |
| **Debug / iterate** | Source-level errors point to `.slang` lines. Hot-reload via module cache invalidation. | nvcc errors are harder to trace. Recompilation pipeline must be built. |
| **Team familiarity** | Already writing Slang kernels in maxwell and radar. | Would fragment the codebase into two kernel languages. |

### When Raw CUDA might be needed

- **Warp-level intrinsics** (`__shfl_sync`, cooperative groups, warp-matrix ops): Slang exposes
  some but not all. If the UTD accumulation kernel needs warp-shuffle reductions for the
  `scatter_reduce` atomic-add path, raw CUDA gives full control.
- **Shared memory tiling**: Slang supports `groupshared` (equivalent to `__shared__`), so this
  is covered. But exotic shared-memory patterns (async copy, swizzled layouts) may require raw CUDA.
- **Inline PTX / `asm` blocks**: For maximum control over memory barriers or specific instruction
  selection. Extremely rare need.

In practice, none of the five proposed kernels (K1–K5) require warp-level tricks. They are
all straightforward "parallel over work items, read structured data, compute scalar math,
atomic-add to output" patterns — exactly what Slang handles well.

### DrJit ↔ Slang Bridge

The channel module currently uses DrJit arrays. The kernels would operate on PyTorch tensors
(via `TensorView`).  Bridge strategy:

1. **Input**: Convert DrJit state arrays to PyTorch tensors once before kernel launch via
   `.torch()` (zero-copy when possible via DLPack).
2. **Output**: Kernel writes into pre-allocated PyTorch output tensors.
3. **Return to DrJit**: Wrap output tensors back to DrJit via `bk.Float(tensor)` for
   downstream DrJit operations (scatter into monitor grid, etc.).

This is the same pattern used in `mesh_sdf.py` (core) where DrJit scene geometry feeds into
Slang kernels and results return to the DrJit pipeline.

### Differentiability Consideration

The channel tracer must support `dr.backward()` through the full field computation.  Options:

- **Slang `[Differentiable]`**: Automatic backward kernel generation. Best for the fused K1
  kernel once the packed-state forward path is moved out of DrJit replay and into a real kernel.
  Validated approach in mesh_sdf.slang.
- **Hand-written adjoint in Slang**: Best for K2 (DDA) where the loop structure needs explicit
  checkpointing — same approach as maxwell/adjoint.
- **`torch.autograd.Function`**: Wrap a pure Slang kernel in a custom autograd op, letting PyTorch
  handle the backward scheduling while Slang executes the actual gradient kernel. This remains
  suitable for K4-style pure function bridges such as Fresnel evaluation.
- **`dr.custom(CustomOp)`**: Best bridge for K1 because the fused accumulation kernel uses
  internal atomics and therefore cannot be exposed through `dr.wrap(...)`. The current
  totals-only prototype already validates this integration model.

Recommended: use `dr.custom(CustomOp)` as the DrJit-facing envelope for K1, with a fused
`[Differentiable]` Slang kernel behind it. Keep `torch.autograd.Function` for pure K4-style
bridges and hand-written adjoints for K2 (DDA loop).

---

## Suggested Implementation Order — One-Shot Strategy

核心发现：代码中存在非常清晰的 **路径发现 / 场合成** 分界线。

- **路径发现**（DDA 遍历、可见性检测、射线-网格求交）：全部 `dr.suspend_grad()` 或
  `mode="evaluated"` 无 AD——**不需要可微**。
- **场合成**（UTD 系数 × 相位 × 幅度 → scatter_reduce 到接收网格）：从 `tx_pos`、
  `scene geometry`、`rotation` 往下的梯度链必须保留——**必须可微**。

基于这个分界线，优化可以分为两层：**DrJit 补丁**（立即可做）和 **Slang 内核**（结构性解决）。
两层之间没有依赖关系，可以并行推进。

### Layer A — DrJit 补丁（Sionna 参考，不写 kernel）

这些改动在现有 DrJit 代码内完成，不需要 Slang 基础设施。

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| A1 | DDA 循环切 `mode="symbolic"` | `reflection/dda.py:422`, `diffraction/suffix.py:250` | **消除 Tier 2 问题**。DDA 循环体的控制流不需要 AD，只有输出（累积场）需要。symbolic 模式不物化中间 state，内存下降 1–3 GB。需验证 scatter_reduce 的 AD 路径在 symbolic 下是否正确传播。 |
| A2 | `dr.hint(exclude=[...])` | 同上 | 把 params tuple 中的常量（grid bounds, cell_size, wavelength, k, tri_data 等）排除出循环 state，减少 DrJit trace 开销。 |
| A3 | `detach_geometry()` | `reflection/field.py:276-295` | 反射路径发现后、场累积前，对中间几何（hit_p, hit_n, si.prim_index）调用 `dr.detach()`。AD 图只需从 `prev_tx → mirror → field_contrib` 这条链传播，不需要回溯到 ray-mesh intersection。 |
| A4 | Hash 去重替代 torch.unique | `reflection/paths.py` | GPU 端 FNV-1a hash 去重，消除 GPU→CPU 同步和 PyTorch 中间 tensor。 |
| A5 | Russian Roulette | `diffraction/suffix.py:383-472` | 对 suffix tracing 中后续 bounce 的低增益射线概率终止，减少活跃射线数。 |
| A6 | builders 加 chunking | `builders.py:169, 309` | 对 `n_paths × n_edges` 和 `n_prev × n_edges` Cartesian 产品加 chunk 限制。 |
| A7 | 剪裁时零拷贝 | `state.py:413-426` | `.torch()` 改用 DLPack 零拷贝，避免 DrJit + PyTorch 双分配。 |

**预期效果**：Layer A 可以解决 Tier 2（DDA 内存）、Tier 3（剪裁双分配）、Tier 4（builder 无限
Cartesian）。大致覆盖 50-60% 的内存问题。

Status update:

- `A1` and `A2` now ship together as the default implementation for the reflection DDA and
  diffraction suffix DDA hot loops.
- The winning structure is: `@dr.syntax` symbolic loop, `dr.hint(exclude=[params])` for
  constants/result buffers/non-AD traversal arrays, and explicit function arguments only for
  AD-carrying per-ray inputs.
- The reflection DDA path now keeps only the symbolic implementation; the earlier evaluated
  fallback has been removed.
- `A6` is already covered by the earlier builder chunking pass in `builders.py`; no further
  Cartesian-expansion gaps remained in the current audit.
- `A3` is now implemented in `reflection/field.py` as a selective detach path:
  - when scene triangle buffers are not gradient-enabled, hit points / normals / blocker
    distances are detached before the reflection transport and DDA accumulation path
  - when scene geometry carries gradients, the detach is skipped to preserve geometry
    optimization workflows
- `A4` is now implemented in `reflection/paths.py` with GPU torch hashing/lexsort coarse
  grouping and a device-side tolerance merge, removing the previous full `.cpu()` transfer of
  reflection-prefix candidates.
- `A7` is now implemented in `diffraction/state.py` with DLPack-backed torch views for the
  pruning hot path, eliminating the old `.torch().detach()` duplication pattern for ranking
  keys.
- `A5` is implemented behind `DiffractionExecutionConfig(suffix_russian_roulette=True)`, but it does
  not meet the fixed `tests/grad/grad_multipath.py` acceptance bar:
  - default path after `A3/A4/A7`: forward `0.635s`, backward `0.870s`, peak `2.492 GiB`
  - `A5` enabled: forward `1.282s`, backward `0.875s`, peak `2.51 GiB`
  - loss and TX gradient norm stayed aligned, so the current issue is steady-state runtime,
    not numerical correctness
  - conclusion: keep `A5` available for experiments, but disabled by default

### Layer B — Slang 内核（结构性解决 Tier 1 + Tier 5）

Layer A 之后，剩余的核心瓶颈是 **Cartesian 展开中每个 chunk 内部的 ~50 个中间数组**。
这是 DrJit 无法优化的——每个 `dr.gather` / 算术运算 / `dr.select` 都是独立 GPU 分配。
只有 fused kernel 能消除它们。

以下模块建议**直接用 Slang 重写**：

#### B1 — `utd_accumulate.slang` (最高优先级)

**替换**: `_accumulate_edge_states_to_receivers` 的整个 chunk 内循环。

```
// 输入: state arrays (SoA layout), receiver grid, k, wavelength, material
// 输出: direct_total[n_rx], multi_total[n_rx] (complex scalar + vector)
// 每个 thread 处理一个 (state, rx) pair

[Differentiable]
[AutoPyBindCUDA] [CUDAKernel]
void utd_accumulate_kernel(
    // State arrays (read-only, SoA)
    TensorView<float> edge_pos_x, edge_pos_y, edge_pos_z,
    TensorView<float> edge_dir_x, edge_dir_y, edge_dir_z,
    TensorView<float> n0_x, n0_y, n0_z,
    TensorView<float> source_pos_x, source_pos_y, source_pos_z,
    TensorView<float> wedge_n,
    TensorView<float> incident_field_real, incident_field_imag,
    // Receiver grid
    TensorView<float> rx_x, rx_y, rx_z,
    // Output (atomic add)
    TensorView<float> out_real, out_imag,
    // Params
    float k, float wavelength,
    int n_states, int n_rx
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_idx = pair_idx / n_rx;
    int rx_idx = pair_idx % n_rx;
    // ... inline: compute_edge_angles → fresnel_integral → UTD coeff → phase → atomic add
}
```

**关键设计**：
- Fresnel 积分的 Boersma 多项式全部在寄存器中完成，不创建中间数组
- 每个 (state, rx) pair 是独立 thread，read state SoA → compute → atomic add
- `[Differentiable]` 让 Slang 自动生成 backward kernel
- 消除 ~50 个中间数组 → **节省 3–5 GB**

#### B2 — `dda_traverse.slang`

**替换**: `reflection/dda.py` 和 `diffraction/suffix.py` 的 DDA 循环。

```
[AutoPyBindCUDA] [CUDAKernel]
void dda_reflection_kernel(
    // Per-ray: origin, direction, blocker_dist, reflection state
    TensorView<float> ray_ox, ray_oy, ray_dir_x, ray_dir_y,
    TensorView<float> mirror_x, mirror_y, mirror_z,
    TensorView<float> weight_real, weight_imag,
    // Grid
    TensorView<float> x_coords, y_coords,
    float x_min, float x_max, float y_min, float y_max,
    float cell_size_x, float cell_size_y, int nx, int ny,
    float wavelength, float k, float rx_z,
    // Output (atomic add)
    TensorView<float> result_real, result_imag, result_count,
    int bounce_idx
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... native loop: DDA march through grid cells, atomic add per cell
}
```

**关键设计**：
- 循环在 kernel 内部（register + local memory），不需要 DrJit 物化
- 场贡献计算是可微的部分：mirror 位置 → distance → phase → field
- Forward kernel 不需要 `[Differentiable]`（DDA 控制流不参与 AD）
- 场合成的 backward 通过 `torch.autograd.Function` 包装
- 消除每轮迭代 ~22 个 state 数组物化 → **节省 1–3 GB**

#### B3 — `fresnel.slang` (最小模块，验证管线用)

**替换**: `utd.py::fresnel_integral`

```
module utd_accumulate;
__include "utd_accumulate_math";
```

`utd_accumulate_math.slang` now owns the shared Boersma / UTD helper math used by `B1`.

**价值**：最简单的 kernel，适合作为 Slang 桥接的 proof-of-concept。验证
DrJit → PyTorch → Slang TensorView → 计算 → 返回 DrJit 的完整数据通路。

### 不需要重写的模块

| 模块 | 原因 |
|---|---|
| `los.py` | 只有 O(n_rx) 复杂度，内存压力极小 |
| `state.py::_make_state_arrays` | 一次性构造，不在热路径 |
| `state.py::_prune_state_arrays_by_budget` | Layer A7（零拷贝）即可解决 |
| `builders.py::_build_tx_first_order_state_arrays` | O(n_edges)，小规模 |
| `reflection/paths.py` | Layer A4（hash 去重）即可解决 |
| `tracer.py` 主控逻辑 | 编排层，不做重计算 |

### 实施路线图

```
Week 1-2:  Layer A (DrJit patches)
           ├─ A1: DDA symbolic mode  ← 最高优先级，最大 bang-for-buck
           ├─ A2: dr.hint(exclude)
           ├─ A3: detach_geometry
           └─ A6: builder chunking

Week 2-3:  B3: shared Fresnel / UTD math layer for K1
           ├─ 建立 Slang 编译/加载管线 (复用 module_cache.py)
           ├─ 验证 DrJit ↔ PyTorch ↔ Slang 数据通路
           └─ 验证 [Differentiable] backward 正确性

Week 3-5:  B1: utd_accumulate.slang (最大收益)
           ├─ Forward kernel
           ├─ [Differentiable] backward 验证
           └─ 替换 _accumulate_edge_states_to_receivers

Week 5-6:  B2: dda_traverse.slang
           ├─ Forward kernel (reflection + suffix 共用)
           ├─ Backward 通过 torch.autograd.Function
           └─ 替换 dda.py + suffix.py DDA 循环

Week 6+:   Layer A 剩余 (A4, A5, A7)
           ├─ Hash dedup
           ├─ Russian Roulette
           └─ 零拷贝剪裁
```

### 预期最终效果

| 场景 | 当前 peak VRAM | Layer A 后 | Layer A+B 后 |
|---|---|---|---|
| 128×128 / 5000 rays | 2.25 GiB | ~1.2 GiB | ~0.5 GiB |
| 256×256 / 5000 rays | 4.10 GiB | ~2.0 GiB | ~0.8 GiB |
| 128×128 / 20000 rays | 5.03 GiB | ~2.5 GiB | ~1.0 GiB |

---

## Reference: Sionna RT v1.2.1 Architecture Comparison

Local reference: `channel/sionna-rt-reference/` (NVIDIA, Apache-2.0).

Sionna RT is a GPU-accelerated differentiable radio propagation simulator also built on DrJit +
Mitsuba.  Comparing its architecture to witwin's reveals both useful patterns and fundamental
design divergences.

### Key Architectural Differences

| Aspect | Sionna RT | witwin.channel |
|---|---|---|
| **Loop mode** | `mode="symbolic"` (default) | `mode="evaluated"` |
| **AD through loops** | Supported for the DDA hot loops via explicit AD-carrying loop inputs plus `dr.hint(exclude=[params])` | Always evaluated, always AD-capable |
| **Diffraction order** | First-order only | Multi-order (1st, 2nd, 3rd) with mixed-path families |
| **Radio map accumulation** | Inline during SBR loop — no Cartesian product | Separate accumulation phase — Cartesian `states × receivers` |
| **Path deduplication** | FNV-1a hashing (O(1) per ray) | PyTorch `torch.unique()` on key tensors |
| **Diffraction state** | Fixed-size `WedgeGeometry` dataclass per wedge | Full state-array dict (~30 fields per state) |

### Patterns Worth Adopting

#### 1. Symbolic Loop Mode for Non-AD Paths

**Sionna pattern**: `dr.while_loop(mode="symbolic")` with `dr.hint(active, exclude=[...])`.

Symbolic mode builds a single computation graph for the entire loop body without materializing
state at each iteration.  This is the **single biggest architectural reason** Sionna uses less
memory in its SBR loops — the DDA-equivalent traversal never creates per-iteration arrays.

**Applicability**: witwin uses `mode="evaluated"` everywhere because it needs AD support.  But
many code paths (e.g., the DDA field accumulation, visibility checks, state construction) run
in the forward pass only and don't require gradients.  Switching these to symbolic mode would
eliminate the per-iteration state materialization that causes Tier 2 memory issues.

**Action**: This has now landed for the reflection DDA and diffraction suffix DDA hot loops.
The final version keeps symbolic mode on the default path, while routing AD-carrying per-ray
inputs explicitly and excluding the remaining constants/result buffers/non-AD traversal arrays
through a `params` tuple.

#### 2. `dr.hint(exclude=[...])` for Loop State Reduction

**Sionna pattern** (`sb_candidate_generator.py:344`, `radio_map_solver.py:555-561`):
```python
while dr.hint(active, label="shoot_and_bounce", exclude=[
    specular_chain_counters, paths_counter_per_source,
]):
```

Explicitly excluding non-state variables from the loop state prevents DrJit from tracing the
loop body twice and reduces the set of arrays that must be materialized each iteration.

**Action**: This has also landed. Both DDA hot loops now use `dr.hint(exclude=[params])`,
where `params` contains constants, result buffers, and non-AD traversal arrays, while only
AD-carrying per-ray inputs remain explicit helper arguments.

#### 3. `detach_geometry()` — Explicit AD Graph Trimming

**Sionna pattern** (`path_solver.py:219`):
```python
paths_buffer.detach_geometry()
```

After candidate generation (SBR + image method), Sionna explicitly detaches all path geometry
from the AD graph before computing channel coefficients.  This prevents the backward pass from
growing through the geometry discovery phase, which doesn't need to be differentiated.

**Applicability**: witwin's reflection DDA loop carries AD-attached geometry (differentiable
hit points, normals) through all bounces.  If gradients are only needed w.r.t. the final field
output and scene parameters (not the ray-tracing geometry search), detaching intermediate
geometry would significantly shrink the AD graph.

**Action**: After reflection path discovery and before field accumulation, call `dr.detach()`
on intermediate geometry arrays that don't need backward support.

#### 4. `dr.make_opaque()` for Static Data

**Sionna pattern** (`path_solver.py:191`):
```python
dr.make_opaque(src_positions, tgt_positions, src_orientations, tgt_orientations)
```

Marking position/orientation data as opaque prevents DrJit from inlining their values into
the computation graph, reducing graph compilation overhead.

**Action**: Apply `dr.make_opaque()` to tx_pos, grid coordinates, and edge geometry data
before the main tracing loops.

#### 5. Buffer Shrinking After Discovery

**Sionna pattern** (`paths_buffer.py:563-649`):
```python
paths_buffer.shrink()  # dr.reshape(..., shrink=True)
```

After SBR, the buffer is immediately resized from `max_num_paths` to the actual path count.
This frees over-allocated GPU memory before the expensive field computation phase begins.

**Applicability**: witwin's state arrays are allocated after visibility/power pruning, so this
is less critical.  But after `_concat_state_arrays` merges multiple sources, the combined
array may have significant padding.  A `shrink` step could reclaim memory before UTD evaluation.

#### 6. Inline Radio Map Accumulation (No Cartesian Product)

**Sionna pattern** (`radio_map_solver.py:555+`):

The SBR loop directly accumulates path gain into the radio map grid cells during traversal.
Each ray intersects the measurement plane and adds its contribution to the corresponding cell.
There is **no separate `paths × receivers` Cartesian expansion** — the grid hit is computed
inline and accumulated via scatter-reduce.

**Why witwin can't directly adopt this**: witwin's accumulation computes *complex coherent
fields* (amplitude + phase), not power gain.  The image-method path requires knowing the exact
source-to-receiver distance for phase coherence, which is inherently a per-path-per-receiver
computation.  Sionna's radio map computes incoherent power (`|a|^2` summed), which doesn't
require phase-coherent accumulation and can be done inline.

**Partial applicability**: For the DDA-based reflection tracing (which already traverses the
receiver grid inline), witwin already uses this pattern.  The main Cartesian bottleneck is in
the *image-method exact path accumulation* and the *UTD diffraction accumulation*, which require
per-state-per-receiver field evaluation.

#### 7. Hash-Based Deduplication

**Sionna pattern** (`sb_candidate_generator.py:296-316`):
```python
hashes = [dr.zeros(mi.UInt64, num_samples) for _ in range(num_hashes)]
specular_chain_counters = [dr.zeros(mi.UInt, spec_counter_size*num_sources)
                           for _ in range(num_hashes)]
```

FNV-1a hashing with multiple hash functions (PlaneHasher, EdgeHasher) deduplicates specular
chains in O(1) per ray during the SBR loop.  This avoids the expensive GPU→CPU transfer +
`torch.unique()` pattern used in witwin's `_collect_unique_reflection_paths`.

**Action**: Consider replacing the PyTorch unique-key deduplication in
`reflection/paths.py:_collect_unique_reflection_paths` with a GPU-side hash-counter approach.
This would eliminate the GPU-CPU sync and the intermediate PyTorch tensor allocation.

#### 8. Russian Roulette for Ray Termination

**Sionna pattern** (`radio_map_solver.py:119-148`):

Rays with low path gain are probabilistically terminated early, with the surviving rays
scaled up to maintain an unbiased estimator.  This is standard in Monte Carlo rendering.

**Applicability**: witwin's reflection suffix tracing (`_trace_reflected_suffix_from_edge_states`)
traces all rays through all bounces.  Adding Russian Roulette termination would reduce the
number of active rays in later bounces, directly cutting the DDA computation cost and
result buffer sizes.

### What Sionna Doesn't Help With

1. **Multi-order diffraction memory**: Sionna only supports first-order diffraction.  The
   combinatorial state growth in witwin's 2nd/3rd order diffraction has no Sionna counterpart.

2. **UTD intermediate bloat**: Sionna's first-order UTD is evaluated once per path, not in a
   Cartesian expansion.  The per-state-per-receiver UTD evaluation pattern is unique to witwin's
   coherent-field architecture.

3. **Backward pass memory**: Sionna punts on differentiability through loops (symbolic mode).
   Witwin's evaluated-mode AD graph size is a problem Sionna doesn't face.

### Summary: Actionable Takeaways from Sionna

| Priority | Pattern | Estimated Impact | Effort |
|---|---|---|---|
| **High** | Switch forward-only loops to `mode="symbolic"` | Eliminate per-iteration DDA state materialization (Tier 2) | Medium — requires auditing AD dependencies |
| **High** | `dr.hint(exclude=[...])` on DDA loops | Reduce loop tracing overhead + state size | Low |
| **High** | `detach_geometry()` after path discovery | Shrink AD graph significantly | Low |
| **Medium** | Hash-based path deduplication | Eliminate GPU→CPU sync in reflection paths | Medium |
| **Medium** | Russian Roulette in suffix tracing | Reduce active ray count in later bounces | Low |
| **Low** | `dr.make_opaque()` on static data | Minor graph compilation speedup | Trivial |
| **Low** | Buffer shrinking after concat | Reclaim padding memory | Low |
