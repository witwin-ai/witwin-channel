# RayD AD Routing and EPC Performance

Status: Active
Last reviewed: 2026-05-23

## Decision

Reflection tracing stays on the existing DrJit-native RayD `trace_reflections(...)` path. Do not migrate reflection tracing to a handwritten CUDA/OptiX JVP/VJP path unless a new benchmark proves a complete custom derivative is both correct and faster on the solver workload.

Diffraction accumulation stays on RayD handwritten AD for the native OptiX/scatter hot paths. The active RayD custom-op coverage is direct diffraction, Keller cone diffraction, and suffix-reflection diffraction for order 1 plus direct, Keller, and suffix-reflection chain accumulation for orders 2 and 3.

EPC reflection field evaluation defaults to RayD native `trace_refl_epc_field(...)`. A multi-transmitter geometry bug in the Channel integration was fixed after the initial benchmark record: the merged path exporter now passes the per-lane transmitter position into RayD EPC instead of always using transmitter 0.

## Routing Matrix

| Component | Default path | Opt-in path | Current reason |
| --- | --- | --- | --- |
| `trace_reflections(...)` | DrJit-native RayD reflection trace | None | Preserves mesh-vertex gradients and symbolic/Lazy-JIT compatibility. |
| Reflection EPC field | `reflection_field_backend="native"` / RayD `trace_refl_epc_field(...)` | `reflection_field_backend="drjit"` | RayD EPC is faster after the multi-Tx geometry fix and now matches the independent geometry residual check. |
| `accumulate_reflections(...)` | RayD native AD | None | Covered by RayD AD tests; no routing change in this pass. |
| `accum_dfr_direct(...)`, order 1 | RayD handwritten JVP/VJP | None | Covers direct, Keller, suffix reflection, and suffix reflector mesh-vertex gradients on a fixed tape. |
| Higher-order `accum_dfr`, order 2/3 | RayD handwritten JVP/VJP | None | Covers direct, Keller, and suffix-reflection chain accumulation on a fixed tape. |
| Suffix reflector mesh vertices | RayD handwritten JVP/VJP through reflector triangle plane point/normal | None | Mesh-vertex motion reaches order-1 and higher-order suffix reflection accumulation through RayD's registered scene geometry inputs. |

## Evidence

| Area | Command / probe | Result |
| --- | --- | --- |
| RayD EPC correctness | `python -m unittest tests.drjit.test_reflection_epc.ReflEpcTests` in `E:\Code\RayDi` | 11 tests passed. |
| Channel EPC integration | `python -m pytest tests\deterministic\test_reflection_rayd_epc_backend.py -q` | 7 tests passed, including the multi-Tx physical geometry regression. |
| Reflection trace AD | Installed RayD probe from Channel cwd, eager and symbolic trace | Both returned `bounce_count=1`, `valid=true`, `t=1.0`, `grad_z_sum=1.0`. |
| RayD reflection accumulation AD | Targeted `ReflectionAccumulationTests` AD parity tests | 2 tests passed. |
| EPC micro-benchmark | 32,768 targets, one wall, primal/VJP | RayD EPC median: 0.468 ms primal, 1.820 ms VJP. Old Channel EPC median: 0.634 ms primal, 2.339 ms VJP. |
| Munich path EPC smoke after fix | Munich order-0 path solve, 4,096 samples, 1 bounce | RayD native median: 13.607 ms and 28 valid paths. `drjit` median: 17.479 ms and 28 valid paths. The per-pair delay sets matched. |
| Munich physical geometry residual | Same RayD native Munich result, independent of Sionna | 22 reflected paths checked; max specular-direction residual `4.6e-7`, max geometric-delay residual `3.0e-5 m`. |
| RayD diffraction AD | `benchmark_rayd_diffraction_ad_compare.py --case all --samples 65536 --warmup 1 --repeats 3 --json` | RayD custom op includes OptiX visibility/tape capture and is slower than the fixed-form DrJit micro-reference, but it avoids the old solver replay stall. Current medians: direct 0.962 ms primal, 1.734 ms JVP, 13.087 ms VJP; suffix 0.881 ms primal, 3.162 ms JVP, 3.328 ms VJP. |
| RayD higher-order diffraction AD | `python -m unittest tests.drjit.test_diffraction_accumulation` in `E:\Code\RayDi` | 23 tests passed, including order-2 VJP, order-3 JVP/VJP, order-1 suffix JVP/VJP, order-2/3 suffix-chain JVP/VJP, and suffix reflector mesh-vertex VJP finite-difference checks. |
| Channel BDPT RayD routing | `python -m pytest tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_diffraction_auto_resolves_to_rayd_for_primal_and_ad_tapes tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_bdpt_order2_rayd_optix_uses_native_direct_and_keller_chain tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_bdpt_order2_rayd_optix_ad_uses_native_suffix_chain tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_bdpt_order3_rayd_optix_uses_native_direct_and_keller_chain tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_bdpt_ad_accepts_reflection_coupled_suffix_with_rayd_native_ad tests\montecarlo\test_monte_carlo_radiomap_integrators.py::test_basic_diffraction_rayd_optix_accepts_ad_tape_collection -q` | 6 tests passed. Channel keeps RayD as the AD-capable accumulation route for Basic and BDPT diffraction, including higher-order reflection-coupled suffix accumulation. |
| Munich BDPT diffraction smoke | `benchmark_munich_performance --cases mc_bdpt_order1 --warmup 1 --repeats 2 --json` | Median 25.511 ms with RayD order-1 suffix AD enabled; no baseline gate in that smoke. |
| Munich BDPT higher-order smoke | `benchmark_munich_performance --cases mc_bdpt_order2,mc_bdpt_order3 --warmup 1 --repeats 1 --diffraction-accumulate-primal rayd_optix --json` | Both cases completed with finite maps. Order-2 median 116.651 ms, order-3 median 115.630 ms, no baseline gate in that smoke. |

## Interpretation

DrJit remains the better default for reflection trace geometry and small smooth algebra because it can preserve Lazy JIT fusion and existing mesh-gradient behavior without forcing a custom-op materialization boundary.

Handwritten RayD JVP/VJP is the right target for native OptiX/scatter/atomic hot paths where DrJit replay creates solver-scale stalls. This now applies to order-1 diffraction accumulation and order-2/3 direct, Keller, and suffix-reflection chain accumulation. The derivative contract is fixed tape: discrete visibility, edge/path selection, and suffix reflector candidate selection remain detached.

RayD EPC is the stable default after the multi-Tx fix. The earlier mismatch was not a RayD kernel physics issue: Channel's merged multi-transmitter exporter passed transmitter 0 to every RayD EPC lane. The independent wall regression catches this directly by checking mirror-source reflection points for two transmitters without using Sionna as an oracle.

## Next Gates

Keep the multi-Tx physical geometry regression in the EPC integration suite. For larger scenes, prefer geometry residual gates over Sionna-only parity: check specular direction, geometric delay, valid surface containment, and segment visibility.

Keep the suffix-chain finite-difference tests and Channel max-depth-2 suffix AD routing test as release gates. Higher-order suffix AD remains a fixed-topology derivative: it covers continuous state/material/reflector-plane changes, not discrete suffix candidate or visibility changes.

Keep the RayD-vs-DrJit AD benchmark as a performance gate when changing custom-op internals. Record forward, JVP, and VJP medians separately because RayD's fixed-tape OptiX materialization boundary changes the performance model compared with fused DrJit algebra.
