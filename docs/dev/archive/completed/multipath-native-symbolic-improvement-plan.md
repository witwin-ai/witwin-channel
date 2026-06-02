# Multipath Native + Symbolic Improvement Plan

Date: 2026-03-31

## Scope

This plan targets the current multipath benchmark workload that historically gets referred to as `tests/grad/benchmark_multipath`. In the current repository layout, the canonical entrypoints are:

- `tests/support/bin/benchmark_multipath.py`
- `tests/support/bin/benchmark_multipath_ad.py`
- `tests/support/bin/_multipath_benchmark.py`

The goal is not to maximize the number of CUDA kernels. The goal is to improve end-to-end throughput and AD usability for the fixed multipath workload, while keeping the solver architecture correct, benchmarkable, and stable.

Status update:

- The historical `suffix_dda="evaluated"` path has been removed from the runtime. `DiffractionExecutionConfig.suffix_dda` is now symbolic-only, and AD-sensitive suffix workloads are expected to use the native suffix backend.

## Current Implementation Status

Status snapshot as of `2026-04-01`:

| Phase | Status | Notes |
|---|---|---|
| Phase 0 | Completed | Benchmark entrypoints, backend reporting, and timing metadata were stabilized. |
| Phase 1 | Completed | Reflection AD workloads are explicitly routed away from frozen EPC; policy and parity guards are in place. |
| Phase 2 | Completed | Benchmark-critical builder replay, compaction, and packing slices are rebaselined; state preparation now clears the original reduction target with regression coverage. |
| Phase 3 | Completed | Native UTD JVP/VJP remains the steady-state path, and pre-kernel visibility compaction now uses the shared zero-copy pair helper on the benchmark workload. |
| Phase 4 | Completed | Fresh field traces now honor the requested backend by default, while EPC is restricted to explicit replay and path-audit workflows. |
| Phase 5 | Partial | The AD benchmark now has explicit `full_field` vs `scalar_loss` workload modes, and the default path avoids building full monitor payloads for scalar-loss optimization runs; monitor chunking remains open. |
| Phase 6 | Not started | Secondary kernel revisits and final cleanup remain deferred. |

## Baseline Snapshot

Historical note:

- This section captures the pre-migration baseline used to define the plan.
- Several blockers below have since been resolved and are superseded by the phase status sections later in this document.

### Benchmark configuration

The measurements below were taken against the fixed workload assembled by `_multipath_benchmark.py`:

- Reflection enabled with `reflection_max_bounces=3`
- Reflection sampling budget `reflection_n_rays=10000`
- RD diffraction enabled
- Suffix DDA mode `symbolic`
- Field monitor resolution `256 x 256`

### Current hard blocker

The reverse benchmark path is not only slow. It is currently incomplete for the exact reflection EPC path:

- `_multipath_benchmark.py` calls `dr.backward(loss)`
- The exact reflection EPC path reaches `trace/reflection/api.py`
- The native exact reflection accumulation op in `kernels/reflection/native_impl.py` implements `eval()` and `forward()`, but not `backward()`
- The current runtime failure is `_ReflectionChunkOp.backward(): not implemented!`

This must be treated as a Phase 1 correctness blocker, not as a later optimization item.

### Measured steady-state phase split

Representative steady-state forward timings for the fixed workload:

| Stage | Time |
|---|---:|
| Reflection accumulation | ~0.14 s |
| Diffraction state preparation | ~5.37 s |
| UTD accumulation | ~5.48 s |
| Reflected suffix tracing | ~0.47 s |

Additional workload shape:

- Diffraction states: `3171`
- Candidate edges: `12`
- Receivers: `65536`

The main conclusion is simple: for this benchmark, the dominant bottleneck is diffraction, not reflection. Within diffraction, `state preparation` and `UTD accumulation` are the two large buckets and they are of similar size.

### Diffraction state-prep breakdown

Representative breakdown inside `prepare_diffraction_state_arrays(...)`:

| Sub-stage | Time |
|---|---:|
| TX first-order builder | ~0.11 s |
| Reflection-prefix first-order builder | ~0.89 s |
| Higher-order state builder | ~3.46 s |
| Inserted-reflection state builder | ~0.58 s |
| Pruning | ~0.00 s |

This benchmark runs without an active path budget bottleneck, so pruning is not a meaningful target here.

### Native microbenchmark summary

Representative results from `tests/support/bin/benchmark_native_kernels.py`:

| Kernel path | Native result |
|---|---|
| `utd` | major win, about `31.5x` faster than Dr.Jit |
| `suffix_grid_forward` | good win, about `3.05x` |
| `suffix_grid_backward` | good win, about `2.63x` |
| `reflection_grid_forward` | moderate win, about `1.78x` |
| `pruning_sort` | small win, about `1.21x` |
| `reflection` | regression, native slower than Dr.Jit |
| `cartesian_filter` | strong regression, native much slower than Dr.Jit |

The main conclusion is also simple: the raw UTD math kernel is already in the right place. The remaining work is around the symbolic scaffolding before and around that kernel, and around unfinished AD integration.

## Prioritization Summary

### Highest-priority improvement targets

| Priority | Area | Why |
|---|---|---|
| P0 | Reflection exact-path backward support or dispatch guard | Current reverse benchmark does not complete |
| P1 | Higher-order diffraction state builder | Largest symbolic hotspot in the fixed workload |
| P1 | Native UTD JVP/VJP completion | Current AD path still replays reference code |
| P2 | Reflection-prefix first-order builder | Nearly 1 second in the fixed workload |
| P2 | Inserted-reflection diffraction builder | Moderate symbolic cost and a clean native candidate |

### Lower-priority or deferred targets

| Area | Current recommendation |
|---|---|
| Exact reflection accumulation kernel expansion | Do not prioritize until correctness and policy are fixed |
| Suffix kernel expansion | Do not prioritize for this benchmark yet |
| `cartesian_filter` native path | Do not expand before resolving its regression |
| `pruning_sort` | Defer unless path budgets become active in the target workload |

## Guiding Principles

- Optimize the fixed multipath workload first, not isolated microbenchmarks in isolation.
- Do not promote a native path to default for AD workloads until forward, JVP, and backward all have parity.
- Reduce symbolic graph size and Dr.Jit compaction overhead where it is on the critical path.
- Keep GPU-first execution. Do not solve these bottlenecks by adding CPU staging or CPU fallback logic.
- Add internal timings and capability checks before large kernel migrations. Blind optimization will waste time.

## Phase Plan

## Phase 0: Benchmark and Dispatch Stabilization

### Current status (`2026-04-01`)

Completed.

What landed:

- `benchmark_multipath.py` and `benchmark_multipath_ad.py` now share a common runtime helper and report aligned package/native capability state.
- Field-trace metadata now exposes stable `runtime_backends` and `performance_timing` data for benchmark consumption.
- Benchmark runs now surface backend selection instead of silently hiding native-vs-reference dispatch.

### Goal

Make the benchmark harness trustworthy so every subsequent performance number is comparable and uses the intended package and native capability configuration.

### Work items

- Normalize benchmark entrypoints so `benchmark_multipath.py` and `benchmark_multipath_ad.py` resolve the same local package and native extension state.
- Remove or isolate import-path behavior that can silently disable native availability in one benchmark entrypoint but not the other.
- Add a benchmark capability report at startup:
  - local package path
  - native extension availability
  - selected backend path for reflection, diffraction, and suffix stages
- Add stable internal timings for:
  - reflection accumulation
  - diffraction state preparation
  - UTD accumulation
  - suffix tracing
- Fail fast when a requested native benchmark mode is not actually available.

### Main modules

- `tests/support/bin/benchmark_multipath.py`
- `tests/support/bin/benchmark_multipath_ad.py`
- `tests/support/bin/_multipath_benchmark.py`
- `witwin/channel/trace/tracer.py`
- `witwin/channel/monitors/field/trace_field.py`

### Exit criteria

- Both benchmark entrypoints report the same package root and native capability state.
- A native benchmark run cannot silently fall back without surfacing that fact.
- Internal timings are available without ad hoc local instrumentation.

## Phase 1: Unblock Reflection Reverse AD Correctness

### Current status (`2026-04-01`)

Completed, with dispatch-guard resolution instead of full exact-replay reverse support.

What landed:

- AD-sensitive fresh traces are explicitly routed away from frozen EPC.
- Reflection policy metadata is now explicit and benchmark-visible.
- Reflection scalar-field gradient regressions for geometry / rotation / TX motion were fixed.

Remaining note:

- The EPC custom op itself is still not the default AD path. The current resolution is policy-based routing, not a new exact-replay `backward()` implementation.

### Goal

Make the benchmark reverse path complete and correct before pushing more performance work.

### Work items

- Implement `backward()` for the exact reflection accumulation native op, or explicitly route AD workloads away from that path until native backward exists.
- Add parity tests for:
  - forward value
  - JVP
  - backward / VJP
- Make the dispatch policy explicit:
  - exact reflection EPC for path auditing
  - field-monitor AD-safe reflection path for optimization workloads
- Verify that `benchmark_multipath` reverse runs to completion with finite gradients.

### Main modules

- `witwin/channel/kernels/reflection/native_impl.py`
- `witwin/channel/trace/reflection/api.py`
- `tests/backend/test_native_kernel_consistency.py`
- `tests/reflection/test_symbolic_dda_toggle.py`
- `tests/main/test_multipath_main.py`

### Exit criteria

- `benchmark_multipath` reverse no longer fails on `_ReflectionChunkOp.backward(): not implemented!`
- Reflection AD path is documented and deterministic for the benchmark workload.
- Reflection forward and AD parity tests pass for the chosen dispatch policy.

## Phase 2: Cut Diffraction State-Preparation Cost

### Current status (`2026-04-01`)

Completed.

What landed:

- Per-order diffraction builder timing is now exposed through the benchmark metadata.
- Higher-order candidate dedup no longer depends on the old `torch.unique` hot path.
- `_selected_edge_bvh_mask(...)` no longer uses Torch / DLPack bridging.
- Inserted-reflection packing now has a dedicated packed-state field gather instead of relying only on broad `subset_state_arrays(...)`.
- Reflection-prefix replay has a native primal batch path for eligible no-gradient workloads.
- Higher-order state packing now reuses the same narrow packed-state gather path instead of filtering a full wide state bundle.
- Inserted-reflection processing now compacts to hit rays immediately after surface intersection, so `field_to_hit`, reflection response, visibility, and state packing no longer run over miss-side rays.
- Reflection-prefix replay no longer falls back to the Python reference implementation when gradients are active; the replay body now stays on the native forward path with direct JVP/VJP parity coverage.
- Higher-order builder visibility compaction now uses a dedicated zero-copy pair-compaction helper instead of repeated `dr.compress()` + re-gather loops.
- The same zero-copy pair-compaction helper now feeds the native UTD visibility stage, so the benchmark-critical pair-compaction slices are shared across builder and solver code paths.
- Direct backend parity coverage now includes reflection EPC JVP/VJP and explicit pair-compaction parity.

Latest steady-state benchmark snapshot (`2026-04-01` reruns):

- `diffraction_state_preparation_seconds ~= 0.114 s .. 0.126 s`
- `reflection_prefix_first_order_seconds ~= 0.013 s .. 0.016 s`
- `higher_order_seconds ~= 0.042 s .. 0.044 s`
- `inserted_reflection_seconds ~= 0.052 s .. 0.055 s`

Residual note:

- Candidate expansion and some final packing steps are still mixed symbolic/native code, but they no longer block the benchmark target and are no longer Phase 2 exit blockers.
- Inserted-reflection still spends measurable time in `state_pack_seconds` and `field_to_hit_seconds`, but the end-to-end state-preparation target is already exceeded by a wide margin.
- Further fusion of those residual slices should be treated as Phase 6 cleanup work, not as unfinished Phase 2 critical-path work.

### Goal

Reduce the largest symbolic bottleneck in the fixed benchmark: diffraction state construction.

### Work items

- Move higher-order diffraction candidate generation off the current heavily symbolic path.
- Replace repeated `dr.compress()`, `dr.width()`, `torch.unique`, and gather/pack cycles with a compact native packing pipeline.
- Kernelize or fuse the higher-order builder operations that currently dominate:
  - candidate expansion
  - visibility filtering / compaction
  - deduplication / canonical packing
- Native-ize the reflection-prefix first-order builder.
- Native-ize the inserted-reflection builder once the higher-order direct builder layout is stable.
- If needed, introduce a shared packed-state representation used across builder stages to avoid repeated concat/repack churn.

### Main modules

- `witwin/channel/trace/diffraction/builders/__init__.py`
- `witwin/channel/trace/diffraction/builders/higher.py`
- `witwin/channel/trace/diffraction/builders/prefix.py`
- Native extension code for new builder-side kernels

### Exit criteria

- `prepare_diffraction_state_arrays(...)` is reduced by at least `2x` on the fixed workload.
- The builder path preserves current path semantics and canonicalization behavior.
- Mixed-path tests continue to pass for direct, reflection-prefix, and inserted-reflection families.

## Phase 3: Complete Native UTD AD and Remove Pre-Kernel Dr.Jit Overhead

### Current status (`2026-04-01`)

Completed.

What landed:

- The steady-state native UTD path no longer relies on `_reference_forward_jvp(...)` or `_reference_geometry_backward(...)`.
- The inner UTD custom op now implements both `forward()` and `backward()`.
- `benchmark_multipath_ad.py` can complete in native mode for both JVP and VJP.
- End-to-end benchmark metadata now reports `resolved_jvp = native_inner_custom_op_forward_mode` and `resolved_backward = native_inner_custom_op_backward` for the active diffraction backend.
- Native pre-kernel visibility compaction now uses the shared zero-copy pair-compaction helper instead of `dr.compress()` + re-gather loops in the steady-state UTD path.
- Reflection-prefix replay AD parity is now covered directly, so the benchmark no longer depends on a Python reference replay fallback before the native diffraction path can run.

Latest benchmark snapshot (`2026-04-01` reruns):

- `benchmark_multipath_ad.py --modes vjp jvp --repeats 1 --warmup-runs 1` completed in native mode for both `vjp` and `jvp`.
- Representative native AD trace timings now sit around `0.215 s .. 0.243 s`, with steady-state diffraction UTD accumulation around `0.013 s .. 0.017 s`.
- Representative native AD differentiation timings now sit around `0.093 s .. 0.105 s` for VJP and `0.489 s .. 0.527 s` for JVP on the fixed benchmark workload.

Residual note:

- Pair index construction and BVH visibility queries still stay in Python / Dr.Jit because scene-query ownership remains there.
- Those remaining symbolic slices are no longer replay-based AD fallbacks and are no longer the critical-path blocker for the benchmark workload.

### Goal

Keep the existing native UTD math win, then remove the remaining AD replay and symbolic pre-kernel overhead around it.

### Work items

- Replace `_reference_forward_jvp(...)` with a real native JVP path.
- Remove or substantially reduce `_reference_geometry_backward(...)` from the steady-state native VJP path.
- Reduce Dr.Jit-side visibility filtering and compaction ahead of the UTD kernel.
- Revisit kernel boundaries so the native path owns more of:
  - visibility mask application
  - state compaction
  - per-state receiver packing
- Add dedicated JVP/VJP benchmarks for the fixed workload shape, not only for microbench kernels.

### Main modules

- `witwin/channel/kernels/utd/native_impl.py`
- Native extension code for UTD AD and packing kernels
- `tests/backend/test_native_kernel_consistency.py`
- `tests/diffraction/test_utd_angle_derivatives.py`

### Exit criteria

- The native UTD JVP/VJP path no longer replays the reference forward implementation in steady state.
- `benchmark_multipath_ad.py` completes in native mode for both JVP and VJP.
- End-to-end AD time improves materially on the fixed workload after Phase 2 is in place.

## Phase 4: Reflection Runtime Policy Rationalization

### Current status (`2026-04-01`)

Completed.

What landed:

- Reflection runtime policy is now explicit for fresh traces versus replayed `reflection_detail`.
- AD-sensitive field workloads preserve discovery gradients by staying on the requested non-frozen backend.
- Fresh field-monitor traces no longer default to EPC just because the scene is replay-eligible.
- EPC is now restricted to explicit `reflection_detail` reuse and path-audit style workflows.
- Fresh field traces now honor the requested reflection backend for both AD and non-AD workloads, so benchmark backend selection no longer depends on an accidental discovery freeze.
- Reflection policy metadata now reports whether the requested backend was honored and labels EPC as an explicit replay / path-audit role instead of a default field backend.

Latest benchmark snapshot (`2026-04-01` post-policy reruns):

- `benchmark_multipath --warmup-runs 1` resolved reflection to `native_cuda_custom_op` with `forward ~= 1.278 s` and `backward ~= 0.732 s` on the reverted `256 x 256`, `10000`-ray workload.
- `benchmark_multipath_ad --modes vjp jvp --repeats 1 --warmup-runs 0` resolved reflection to `native_cuda_custom_op` for all reported runs.
- Fresh benchmark field workloads now map directly to the requested reflection backend instead of silently switching to EPC.

Final policy split:

- Path monitors: explicit path discovery / replay workflow; EPC remains valid here as an audit-oriented path representation.
- Field monitors, fresh traces, AD workloads: always honor the requested field backend and preserve discovery gradients.
- Field monitors, fresh traces, non-AD workloads: also honor the requested field backend; EPC is no longer the default just because it is eligible.
- Field monitors with explicit `reflection_detail`: reuse the frozen discovered path set through EPC, independent of the requested field backend.

### Goal

Decide what the reflection runtime should do by default for field-monitor workloads, instead of keeping a half-native, half-symbolic policy that is hard to benchmark.

### Work items

- Reevaluate exact-path reflection accumulation after Phase 1:
  - if it reaches parity and stays stable, keep it for field AD workloads
  - if it remains slower or harder to maintain, restrict it to path/audit workflows
- Document the default reflection policy for:
  - path monitors
  - field monitors
  - AD workloads
  - non-AD workloads
- Add guardrails so benchmark mode selection maps to a predictable reflection backend.

### Main modules

- `witwin/channel/trace/reflection/api.py`
- `witwin/channel/trace/tracer.py`
- `witwin/channel/monitors/field/trace_field.py`

### Exit criteria

- Reflection backend policy is explicit and benchmarkable.
- The benchmark workload no longer depends on an accidental backend choice.
- No reflection-stage regression is introduced after Phase 1 and Phase 3 changes.

## Phase 5: Symbolic Graph and Memory Reduction for Optimization Workloads

### Current status (`2026-04-01`)

Partial.

What landed:

- `benchmark_multipath_ad.py` now records torch CUDA and Dr.Jit allocator memory snapshots during setup / trace / loss / AD phases.
- `benchmark_multipath_ad.py` now exposes explicit `--workload full_field|scalar_loss` modes, and the AD benchmark defaults to the scalar-loss workload instead of the historical full-field collection path.
- A new total-field-only trace helper now skips the full `field/vector/jones` monitor payload assembly when the benchmark only needs the scalar loss, while keeping the same runtime backend and timing metadata.
- Reflection detail assembly now has a light-weight mode that omits receiver-wide polarization/Jones payloads when downstream code only needs replay/material metadata.
- Diffraction now has a metadata-only return path for scalar-loss workloads, so the AD benchmark can keep solver metadata and performance timing without retaining the full diffraction component payload.

Latest benchmark snapshot (`2026-04-01`, `256x256`, `10000` rays, single runs):

- `tx_x vjp`: `full_field total=2.116s`, `scalar_loss total=1.809s`; Dr.Jit allocator peak dropped from `2.772 GiB` to `2.767 GiB`.
- `tx_x jvp`: `full_field total=3.748s`, `scalar_loss total=3.526s`; Dr.Jit allocator peak dropped from `2.814 GiB` to `2.767 GiB`.
- `cube1_x vjp`: `full_field total=1.480s`, `scalar_loss total=1.163s`; Dr.Jit allocator peak dropped from `2.814 GiB` to `2.782 GiB`.
- `cube1_x jvp`: `full_field total=4.099s`, `scalar_loss total=3.468s`; Dr.Jit allocator peak dropped from `2.829 GiB` to `2.782 GiB`.

Still open:

- Monitor tiling or chunked accumulation strategies have not been implemented yet.
- The benchmark still relies on allocator/memory snapshots instead of a first-class symbolic graph-size counter.

### Goal

Reduce graph size and memory pressure for large field-monitor optimization workloads once the main throughput bottlenecks are addressed.

### Work items

- Add benchmark modes that measure graph size, peak memory, and AD overhead separately from forward time.
- Investigate monitor tiling or chunked accumulation strategies that reduce graph fan-out without changing physical results.
- Reuse compact packed-state layouts from Phase 2 and Phase 3 to avoid repeated materialization of large symbolic arrays.
- Keep the scalar-loss optimization workload separate from the historical full-field collection path and compare both in reruns.

### Main modules

- `witwin/channel/monitors/field/trace_field.py`
- `witwin/channel/monitors/field/grid_diffraction.py`
- `witwin/channel/trace/tracer.py`

### Exit criteria

- Optimization workloads show lower peak memory or lower graph build cost on large monitors.
- The benchmark suite can separate throughput regressions from graph-size regressions.

## Phase 6: Secondary Kernel Work and Cleanup

### Current status (`2026-04-01`)

Not started.

Still open:

- Revisit `cartesian_filter` only after post-Phase-5 data justifies it.
- Revisit `pruning_sort` only if target workloads become budget-dominated.
- Do the final pack/unpack cleanup once the primary Phase 2 and Phase 3 migrations stop moving.

### Goal

Address remaining secondary hotspots only after Phases 0 through 5 are complete and rebenchmarked.

### Work items

- Revisit `cartesian_filter` only if a new design avoids the current regression.
- Revisit `pruning_sort` only if target workloads actually enable path budgets heavily.
- Revisit suffix bounce-loop internals only if suffix becomes material again after earlier phases.
- Collapse duplicated pack/unpack logic introduced during migration into shared runtime utilities.

### Exit criteria

- Secondary kernel work is justified by post-Phase-5 benchmark data, not by assumption.
- Cleanup reduces maintenance cost without changing solver behavior.

## Recommended Execution Order

Recommended order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Rebenchmark and re-baseline
6. Phase 4
7. Phase 5
8. Phase 6 only if new data justifies it

Do not start broad reflection-kernel expansion, suffix-kernel expansion, or `cartesian_filter` optimization before Phases 1 through 3 are complete. Those are not the critical path for this benchmark today.

## CUDA Kernel Candidates by Module

### Strong candidates

| Module | Candidate work |
|---|---|
| `witwin/channel/trace/diffraction/builders/higher.py` | Higher-order candidate generation, visibility filtering, deduplication, packing |
| `witwin/channel/trace/diffraction/builders/__init__.py` | Shared state-array assembly entrypoints and compact packed-state handoff |
| `witwin/channel/trace/diffraction/builders/prefix.py` | Reflection-prefix first-order builder |
| `witwin/channel/kernels/utd/native_impl.py` | UTD JVP/VJP integration and pre-kernel compaction ownership |

### Conditional candidates

| Module | Condition |
|---|---|
| `witwin/channel/trace/diffraction/builders/higher.py` | Inserted-reflection builder after direct higher-order layout stabilizes |
| `witwin/channel/monitors/field/grid_diffraction.py` | Only if suffix becomes material again after earlier phases |

### Not recommended right now

| Module | Reason |
|---|---|
| `witwin/channel/kernels/reflection/native_impl.py` EPC expansion | Correctness and policy are unresolved; current forward microbench is not a win |
| `cartesian_filter` native path | Current native result regresses badly |
| `pruning_sort` | Not on the critical path for the fixed workload |

## Validation Matrix

Activate the required environment first:

```bash
conda activate witwin2
```

Run benchmark commands serially, one process at a time:

```bash
python -m tests.support.bin.benchmark_multipath --warmup-runs 1
python -m tests.support.bin.benchmark_multipath_ad --modes vjp jvp --repeats 1 --warmup-runs 1
python -m tests.support.bin.benchmark_native_kernels
```

Targeted regression coverage after each phase:

```bash
python -m pytest tests/backend/test_native_kernel_consistency.py
python -m pytest tests/trace/test_chunked_cartesian_accumulation.py
python -m pytest tests/mixed/test_rd_multipath_consistency.py
python -m pytest tests/mixed/test_drd_inserted_reflection.py
python -m pytest tests/mixed/test_reflection_prefix_path_canonicalization.py
python -m pytest tests/reflection/test_symbolic_dda_toggle.py
python -m pytest tests/diffraction/test_utd_angle_derivatives.py
python -m pytest tests/main/test_multipath_main.py --gpu
```

Recommended phase-specific validation emphasis:

| Phase | Minimum validation focus |
|---|---|
| Phase 0 | Benchmark entrypoints and backend capability reporting |
| Phase 1 | Reflection forward/JVP/VJP parity and reverse completion |
| Phase 2 | Mixed-path correctness, canonicalization, inserted-reflection coverage |
| Phase 3 | UTD derivative correctness and native AD execution |
| Phase 4 | Reflection backend selection behavior |
| Phase 5 | Memory and graph-size benchmark variants |

## Success Criteria for the Full Plan

The plan should be considered successful only if all of the following are true:

- `benchmark_multipath` forward and reverse both run to completion in the intended native configuration
- The benchmark harness exposes which backend paths are actually active
- Diffraction state preparation is no longer a dominant symbolic bottleneck at the current scale
- Native UTD AD no longer depends on steady-state reference replay
- Reflection backend policy for field-monitor AD workloads is explicit and stable
- Secondary kernel work is driven by new measurements, not by outdated assumptions
