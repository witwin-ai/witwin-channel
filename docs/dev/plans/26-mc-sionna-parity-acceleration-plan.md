# Monte Carlo Sionna-Parity Acceleration Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-20

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **RayD source location (READ FIRST).** Any task in this plan that touches RayD (all of Tier 1, T2.1, T2.2, T3.2) MUST build and install from the local source tree at **`E:\Code\RayDi`**, not from PyPI. The PyPI `rayd` package lags the local tree and is missing the kernels this plan depends on. Install with `pip install -e E:\Code\RayDi --no-deps` (per [CLAUDE.md](../../CLAUDE.md) `--no-deps` rule), or follow `E:\Code\RayDi\CLAUDE.md` / `AGENTS.md` for the native CMake build. Before editing RayD code, read its in-tree `CLAUDE.md` first.

## Goal

Make `witwin.channel.montecarlo` (both the `basic` integrator and the BDPT family) **measurably faster than Sionna RT 2.0** on equivalent radio-map and path workloads, on the same GPU and the same scene/parameters. The target is not parity. The target is a clear margin that survives independent reproduction.

This plan is the concrete implementation companion to `24-realtime-rt-architecture-roadmap.md`. Plan 24 sets the strategy ("RayD-owned GAS + fused `trace_*` kernels"); this plan turns it into a sequenced, file-level work list, scoped to Monte Carlo.

## Why this plan exists separately from plan 24

Plan 24 is a strategy document — it explains *why* RayD+channel can beat Mitsuba+Sionna at tier-3 real-time digital twin, and inventories candidate fusion kernels (§7). It deliberately stops before deciding what to ship first or how to acceptance-test the results.

This plan picks up there: a tier-ordered work list, with file:line targets, exit criteria per tier, and a head-to-head benchmark protocol against the two in-tree Sionna references (2.0.0 at `channel/sionna-rt-reference-2.0.0/`, 2.0.1 at `channel/reference/sionna-rt-reference-2.0.1/`).

## Baseline: Sionna actual execution model

Sources of truth:
- **`e:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\`** — current upstream (`drjit==1.3.1`, `mitsuba==3.8.0`). Use this for any claim that ships externally.
- `channel/sionna-rt-reference-2.0.0/` — earlier snapshot kept for diff/history. Do not cite alone.
- Not the paper.

Sionna's speed comes from **three properties**, no more:

| # | Property | Location |
|---|----------|----------|
| S1 | Entire SBR is one Dr.Jit symbolic megakernel via `@dr.syntax` + `while dr.hint(active, mode="symbolic", ...)` | `path_solvers/sb_candidate_generator.py:238-534`; `radio_map_solvers/radio_map_solver.py:420-692` |
| S2 | In-kernel `dr.scatter_reduce(Add)` writes radio-map tensor directly; no intermediate buffer | `radio_map_solvers/planar_radio_map.py:302`, called from SB loop at `radio_map_solvers/radio_map_solver.py:619` |
| S3 | One importance-sampled secondary per intersection (width stays = N_S, no branching) | `path_solvers/sb_candidate_generator.py:355,531,645` |

Total OptiX launches per `RadioMapSolver(...)` call: **2–3 megakernels** (LoS + SB + optional first-order diffraction).

Sionna's **structural ceilings** (these are channel's opportunity):

- W1 (REVISED 2026-05-20, V1.1 closed against 2.0.1 source at `e:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\`): the original "symbolic mode does not support backprop" claim was **partially** stale at the Dr.Jit layer but **still holds in practice** for shipped Sionna 2.0.1. The corrected story:
   - **What changed in the substrate:** Dr.Jit 1.3.0 (2026-Q1) shipped reverse-mode AD through symbolic loops via **trajectory replay**, and Sionna 2.0.1 pins `drjit==1.3.1` ([sionna-rt-reference-2.0.1/pyproject.toml:42](../../../reference/sionna-rt-reference-2.0.1/pyproject.toml#L42)). The capability is therefore present in the runtime Sionna ships with.
   - **What Sionna 2.0.1 does with it: nothing.** `loop_mode` defaults to `"symbolic"` ([radio_map_solver.py:198](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/radio_map_solvers/radio_map_solver.py#L198), [image_method.py:55](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/image_method.py#L55), [field_calculator.py:46](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/field_calculator.py#L46)) with no grad-aware switch. Sionna's own docstrings explicitly state: "Symbolic mode is the fastest mode but does not currently support backpropagation of gradients" ([image_method.py:53-54](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/image_method.py#L53), [field_calculator.py:44-45](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/field_calculator.py#L44), [radio_map_solver.py:205-208](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/radio_map_solvers/radio_map_solver.py#L205)).
   - **`detach_geometry` is PathSolver-only, NOT RadioMapSolver.** Earlier revision claimed both paths cut geometry — that was a source-reading error, retracted. [path_solver.py:219](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/path_solver.py#L219) calls `paths_buffer.detach_geometry()` ([paths_buffer.py:873-887](../../../reference/sionna-rt-reference-2.0.1/src/sionna/rt/path_solvers/paths_buffer.py#L873)) — `dr.detach` on `_vertices_x/y/z`, `_theta_t`, `_phi_t`, `_theta_r`, `_phi_r`, `_diffracting_wedges`. RadioMapSolver scatters directly into a tensor without constructing a `PathsBuffer`, so this detach does not apply.
   - **What actually happens on RadioMapSolver (V1.2 empirical, drjit 1.2 / mitsuba 3.7.1, 3-cube scene, `samples_per_tx ∈ {2·10⁵, 10⁶}`):** see [`docs/dev/optimization/sionna-2.0.1-radiomap-grad-probe-2026-05-20.md`](../optimization/sionna-2.0.1-radiomap-grad-probe-2026-05-20.md) for full data. Summary:

       | mode | material AD (`eps_r`) | geometry AD (`vertex_positions`) |
       |---|---|---|
       | symbolic (default, fast) | broken: `grad_enabled(metric) = False`, `dr.backward` raises | chain present but all 72 components return identically zero |
       | evaluated (slow) | works: AD = 1.96e-6, FD = 1.40e-6 (within MC noise) | 24 / 72 nonzero — *exactly 1 axis per face-vertex* (8 nonzero in each of x/y/z). FD vs AD at 1 M samples is inconclusive: per-vertex signal (~1e-6) is below MC noise floor (~1e-5); top-5 sign agreement is 3/5, magnitude off by 1.6–3.8× when signs match. |

     Two readings of the geometry pattern are consistent with the evidence: (a) it is the mathematically correct face-normal-only derivative (in-plane vertex motion does not change the plane equation); (b) the AD chain is partially detached inside Mitsuba's intersection routine. Resolving this requires either much higher sample counts for FD or a small-ROI metric — tracked under V1.2 follow-up in the optimization doc.
   - **Net witwin/Sionna contrast (drjit 1.2; drjit 1.3 trajectory replay is documented but not wired up by Sionna 2.0.1):**

       | feature | Sionna 2.0.1 fast mode (symbolic) | Sionna 2.0.1 slow mode (evaluated) | witwin |
       |---|---|---|---|
       | material AD on radio map | ❌ | ✅ | ✅ in fast mode |
       | geometry AD on radio map | ❌ (silent zeros) | ⚠️ partial / unverified | ✅ in fast mode (auditable: integer-topology tape + sparse VJP) |
       | geometry AD on path solver | ❌ | ❌ (`detach_geometry`) | ✅ |
   - **W1a — Trajectory storage cost (still relevant if Sionna ever turns it on).** Dr.Jit replay allocates one entry per thread per non-invariant loop-state variable per iteration. SBR with N rays, `max_depth = 5–8`, ~10 state vars → O(N × max_iter × N_state) floats. At N = 10⁶ this can exceed VRAM on city-scale scenes. This is the upper-bound argument for why Sionna may not flip the default even when they could.
   - **W1b — `compress=True` incompatibility.** Dr.Jit caveat: "compress=True breaks the fixed iteration-to-slot mapping that trajectory replay relies on, making it incompatible with positive max_iterations values." Currently moot — Sionna 2.0.1 does not pass `compress=` in any `dr.hint(...)` call — but relevant if Sionna later adds ray-compaction to maintain GPU utilization under AD.
   - **witwin's differentiator: representation + scope.** witwin's AD path tapes only **discrete topology** (`prim_index_by_bounce`, `edge_index_by_step`, `cell_idx`, `strategy`, `order`) — integers, O(N × K) bytes. The backward pass is a bounded host-side `for` over recorded topology that emits a `dr.CustomOp` calling `SparseCoeffKernel.launch_vjp_into`, scattering sparse `(cell_idx, vertex_idx, ∂P/∂v)` triplets back into the parameter grid. This (a) avoids O(N × max_iter × N_state) float trajectory storage, and (b) **is auditable**: each triplet is finite-diff-checkable, unlike Sionna's evaluated-mode geometry AD where the structural pattern raises questions that 1 M samples cannot resolve.
- W2: first-order diffraction only, sampled with `DIFFRACTION_SAMPLING_PROBABILITY = 0.2`, and incompatible with diffuse (`sb_candidate_generator.py:403-406`).
- W3: Mitsuba `mi.Scene` encapsulates the GAS — no `optixAccelBuild(UPDATE)` partial refit, dynamic scenes must rebuild.
- W4: specular dedup is FNV-1a hash with `dr.scatter_inc`; collisions silently drop paths (`sb_candidate_generator.py:432-498`).
- W5: phenomenological materials only — no T-matrix / full-wave scattering coupling.

### W1 verification gate — required before any AD-comparison claim ships externally

Any external-facing artefact (pitch, paper, slide deck, README, benchmark report) that contrasts witwin's AD path against Sionna's MUST first land the following verification:

- [x] **V1.1 — CLOSED 2026-05-20.** Source-level read of `e:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\`. Findings: (a) `loop_mode` defaults to `"symbolic"` with no grad-aware switch; (b) Sionna docstrings explicitly disclaim symbolic-mode backprop despite drjit 1.3.1 capability; (c) `paths_buffer.detach_geometry()` cuts geometry only on the **PathSolver** path — RadioMapSolver does not call it.
- [x] **V1.2 — CLOSED 2026-05-20.** Empirical probe via [`docs/dev/optimization/sionna-2.0.1-radiomap-grad-probe-2026-05-20.md`](../optimization/sionna-2.0.1-radiomap-grad-probe-2026-05-20.md). Symbolic-mode confirmed AD-broken; evaluated-mode material AD agrees with FD within MC noise; evaluated-mode geometry AD reports a structurally suspicious "1 axis per face-vertex" pattern that 1 M samples cannot validate against FD. Witwin's contrast table above is now sourced from data, not inference.
- [ ] **V1.3** Same probe replayed against witwin's MC tape-replay path on the identical 3-cube scene. Record per-ray bytes, peak VRAM, FD-vs-AD agreement on geometry. Required before any "witwin's AD is cheaper / more correct than Sionna's" claim ships externally.
- [ ] **V1.4 (follow-up to V1.2)** Re-run V1.2 geometry probe at `samples_per_tx ≥ 10⁷` with a small-ROI metric (~10×10 cells around the strongest reflection focal point) to determine whether Sionna's evaluated-mode geometry AD is "face-normal-only but correct" (charitable) or "partially detached" (uncharitable). The verdict changes the published witwin claim by one notch — pick the version after this resolves.
- [ ] **V1.5** Update pitch artefacts (`docs/dev/witwin_vs_sionna.md` if present, slides) with V1.3 + V1.4 numbers. After V1.5, full AD claims may ship externally.

V1.1 + V1.2 are enough to retire the false "Sionna can't AD anything" framing and replace it with the measured matrix above. V1.3–V1.5 are needed to quantify the witwin-side advantage and resolve the geometry-AD interpretation.

## witwin Monte Carlo bottleneck inventory

Bottlenecks were located by reading the actual code, not by speculation. File:line references are exact.

### Basic integrator

| ID | Severity | Site | Description |
|----|----------|------|-------------|
| B1 | Critical | `witwin/channel/montecarlo/integrators/basic.py:457`; sync at `witwin/channel/montecarlo/kernels/monte_carlo/native_impl.py:40` | Python `for batch_start in range(...)` outer loop forces `dr.eval()` between batches. 256-ray batch × 39 iterations vs Sionna's single megakernel. |
| B2 | High | `witwin/channel/montecarlo/trace/diffraction.py:309-318` | Four sequential `scene.segment_visible` calls per diffraction state for source/target × base/offset shadow-boundary tests. |
| B3 | High | `witwin/channel/montecarlo/grid_ops.py:213-272` | `GridContributionStore` double-write: `dr.scatter` into 7 staging arrays inside the loop, then `scatter_into()` re-reads and re-scatters into the grid. Two GMEM passes per hit. |
| B4 | Medium | `witwin/channel/montecarlo/trace/los.py:86-99` | LoS computes power for every cell then `dr.compress` collects visible lanes. Equivalent to Sionna's launch count, but wastes BW. |
| B5 | Medium | `witwin/channel/montecarlo/integrators/basic.py:57-62` | `_MC_BATCH_ALIGN = 256` rounds down — small workloads underutilize SMs. |
| B6 | Low | `witwin/channel/montecarlo/trace/reflection.py:536-551` | No in-kernel Russian roulette. Reflection loop always runs to `max_bounces`. |

### BDPT (`bdpt.py` + `bdpt_diffraction.py`)

| ID | Severity | Site | Description |
|----|----------|------|-------------|
| P1 | Critical | `witwin/channel/montecarlo/integrators/bdpt_diffraction.py:661-681` (dispatch); call sites at `:811-815, 985-994, 1138-1142, 1267-1271, 1462-1469, 1680-1689` | 4–6 visibility queries per sample. Native dispatch gated on `2 * n_segments * n_triangles >= 4096`; below threshold falls back to Python `scene.segment_visible` per sample. |
| P2 | High | `witwin/channel/montecarlo/integrators/bdpt_diffraction.py:1130-1167` | Chain inter-edge visibility: order-3 path → 6 OptiX launches before reaching the receiver. |
| P3 | High | `witwin/channel/montecarlo/integrators/bdpt_diffraction.py:535-586,1013-1021,1284-1290,1491-1498` | MIS weights (`exterior_angle`, `integration_weight`, direct/Keller scores) recomputed per sample. Dr.Jit cannot tell they are per-edge constants. |
| P4 | High | `witwin/channel/montecarlo/trace/diffraction.py:733-743` (and re-evaluation across `bdpt_diffraction.py:817-824,1144-1150,1273-1280,1472-1480,1695-1702`) | UTD geometric coefficients re-evaluated per sample symbolically; no per-edge cache. |
| P5 | Medium | `witwin/channel/montecarlo/integrators/bdpt_diffraction.py:117-146,1334-1357,1534-1560,1776-1799` | AD tape: 10 float fields + 1 int scatter per accepted sample, serialised. |
| P6 | Medium | `witwin/channel/montecarlo/trace/diffraction.py:670-927` | `DiffractionStates` SoA ≈ 200 B/vertex including 2× `FaceMaterial`; per-sample full gather. |

### Already at parity (do not regress)

- LoS single-launch behaviour matches Sionna.
- Within-batch multi-bounce symbolic fusion in `reflection.py:536` is structurally equivalent to Sionna's SB megakernel.
- Multi-order UTD diffraction, integer-topology AD tape with sparse-VJP scatter (cheaper than Dr.Jit 1.3 trajectory replay on city-scale scenes; see W1), IAS refit for dynamic scenes, and the planned T-matrix coupling (`project_witwin_tmatrix_scattering_plan.md`) are unmatched by Sionna and must survive every refactor.

## RayD primitive availability (Stage 4 of plan 24)

The acceleration primitives that this plan depends on live in RayD
(`E:\Code\RayDi`). Their design and current status are tracked in
two RayD-side docs:

- [`rayd/docs/rf_trace_kernel_plan.md`](file:///E:/Code/RayDi/docs/rf_trace_kernel_plan.md)
  — `trace_*` kernel inventory, Phase 1 / Phase 2, backend dispatch
- [`rayd/docs/edge_bvh_optix_migration_plan.md`](file:///E:/Code/RayDi/docs/edge_bvh_optix_migration_plan.md)
  — edge BVH OptiX custom-AABB migration (delivered)

Snapshot of what is callable from channel today (cross-check before
starting any tier task):

| RayD primitive | Status | Channel routes through |
|---|---|---|
| `Scene.trace_segment_visibility(start, end, ignore_prim_ids)` | ✅ shipped (`src/multipath/segment_visibility.cu`) | `Scene._trace_segment_visibility_rayd` in [`core/scene/scene.py`](../../witwin/channel/core/scene/scene.py) |
| `Scene.trace_segment_pair_visibility(start, end_a, end_b)` | ✅ shipped (same module) | `Scene.segment_pair_visible` |
| `Scene.trace_axial_edge_visibility(source, edge, fractions)` | ✅ shipped (same module) | `Scene.axial_edge_visible` and MC shadow-boundary source-edge pre-culling |
| `Scene.nearest_edges_topk(query, k)` | ✅ shipped (edge BVH OptiX) | (used via existing `nearest_edge`-style call sites) |
| `Scene.trace_segment_chain_visibility(points, length, ignores)` | ✅ shipped (`src/multipath/segment_visibility.cu`) | `Scene.segment_chain_visible` (available for MC chain batching; not consumed by deterministic EPC in this MC-focused pass) |
| `trace_reflections` trailing fields | ✅ shipped ([rf_trace_kernel_plan §6](file:///E:/Code/RayDi/docs/rf_trace_kernel_plan.md)) | not yet consumed |
| `trace_reflections_accumulating` (T2.1) | ✅ shipped in RayD as explicit non-AD complex-polarized native fast path; channel routes through `accumulation_backend="rayd_reflection_accumulation"` | `Scene.trace_reflections_accumulating` in [`core/scene/scene.py`](../../witwin/channel/core/scene/scene.py), consumed by basic MC reflection |
| `trace_diffraction_chain_mc` (T3.2) | ❌ not shipped — magnitude reassessed | n/a |

### Backend dispatch — `RAYD_TRACE_VISIBILITY_BACKEND`

The shipped visibility kernels (`trace_segment_visibility`,
`trace_segment_pair_visibility`, `trace_axial_edge_visibility`,
`trace_segment_chain_visibility`) come
with **two implementations** and pick one automatically:

- **jit path** — inline `optixTrace` in Dr.Jit-emitted PTX; symbolic-
  loop-recordable; no kernel boundary
- **native path** — standalone `optixLaunch` with custom anyhit /
  closesthit; required when `ignore_prim_ids` is non-empty

Default dispatch picks jit when `ignore_k == 0`, native otherwise.
Override with `RAYD_TRACE_VISIBILITY_BACKEND={auto, jit, native}` for
benchmarking — full four-quadrant truth table in
[`rf_trace_kernel_plan.md` "Backend Dispatch"](file:///E:/Code/RayDi/docs/rf_trace_kernel_plan.md).

Practical implication for this plan: every channel call site that
switches to `Scene.trace_segment_visibility(...)` (T1.1, T1.3 sites,
T4.1 site) automatically gets the right backend without
caller-side selection logic. Channel does not fall back when a control
combination requires the native path but is incompatible with Dr.Jit
symbolic recording; it raises immediately.

### Honest speedup re-calibration (carry over from plan 24)

The first revision of this plan and plan 24 §7 quoted aggressive
per-kernel speedups (3–10×). A later honest audit landed at more
modest numbers; the values below are the ones this plan should use
when sizing tier exit criteria:

| Kernel | Honest re-estimate | Source |
|---|---|---|
| `trace_segment_visibility` (with ignores) | **3–8×** vs Python re-fire | rf_trace_kernel_plan §8.3 |
| `trace_segment_pair_visibility` | 1.5–2× | same |
| `trace_axial_edge_visibility` | 2–4× | same |
| `nearest_edges_topk` | 2–4× plus recall guarantee over 18-probe heuristic | same |
| `trace_segment_chain_visibility` (Phase 2) | 3–5× on EPC, 1.5–2× on BDPT | same |
| `trace_reflections_accumulating` (T2.1) | 1.5–3× over Dr.Jit symbolic loop | same |
| `trace_diffraction_chain_mc` (T3.2, MC sampled-path mode) | 2–4× over Dr.Jit symbolic | same |

End-to-end channel wall-clock after **all** of the above land: ~1.5×
once tier-0 launch storm fixes are also in. Reaching tier-3 live
digital twin (30–100 ms frame budget) requires algorithm changes on
top of fusion — out of scope for this plan; tracked in plan 24 §8.

## Tiered acceleration plan

Tiers are ordered by (speedup × tractability). Each tier has a measurable exit criterion. Do not start tier N+1 before tier N's exit criterion passes.

### Tier 0 — Eliminate the basic-integrator launch storm (no RayD change)

Goal: collapse the Python batch loop and the double-write reduction. After this tier, basic-MC OptiX launches per radio map should be in the single digits, comparable to Sionna.

- [x] **T0.1 Single symbolic batch.** Move the `samples_per_tx` iteration off Python and inside the symbolic trace. Remove the `for batch_start in range(...)` at [basic.py:457](../../witwin/channel/montecarlo/integrators/basic.py#L457). Drive batching with Dr.Jit chunking only if VRAM forces it, not as the default.
- [x] **T0.2 Direct `dr.scatter_reduce(Add)` into the grid.** Replace the `GridContributionStore` double-write at [grid_ops.py:213-272](../../witwin/channel/montecarlo/grid_ops.py#L213) with an in-loop `dr.scatter_reduce(dr.ReduceOp.Add, grid_power.array, value, cell_index, active)`. Pattern mirrors Sionna `planar_radio_map.py:302`. Delete `GridContributionStore` if no other caller remains.
- [x] **T0.3 Drop forced `dr.eval()` in `MonteCarloKernel.scatter_samples`.** [kernels/monte_carlo/native_impl.py:40](../../witwin/channel/montecarlo/kernels/monte_carlo/native_impl.py#L40) currently forces materialisation. After T0.2 the kernel is dead; remove it.
- [x] **T0.4 In-kernel Russian roulette.** Add `rr_depth` / `rr_prob` to the reflection loop at [reflection.py:536-551](../../witwin/channel/montecarlo/trace/reflection.py#L536) with the unbiased `rsqrt(p)` reweighting. Reference: `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py:673-683`.
- [x] **T0.5 Tune `_MC_BATCH_ALIGN`.** Replace the hard 256 with a min(workload, 256) policy, or remove entirely once T0.1 lands.

**Exit criterion**: On a Munich-class scene with `samples_per_tx = 10⁶` and `max_bounces = 5`, total OptiX launches per `MonteCarloSolver.solve()` ≤ 5, measured by Nsight Systems. Wallclock ≤ 1.1× Sionna's `RadioMapSolver` on the same hardware. Numerical equivalence to pre-tier baseline within Monte Carlo noise on the radio map L1 norm.

### Tier 1 — RayD primitives that subsume per-call ignore loops

Goal: deliver the two RayD changes from plan 24 §3.1/§3.2 that channel can immediately consume. Each one removes a whole class of repeated launches.

- [x] **T1.1 `trace_segment_visibility(start, end, ignore_prims)`** in RayD with `__anyhit__` calling `optixIgnoreIntersection()`. Plan 24 §7-D1. **Status: shipped.** RayD-side: `src/multipath/segment_visibility.cu` (native path) + `OptixScene::segment_hit<true>(...)` (jit path), dispatched by `use_jit_trace_visibility_path(ignore_k)` with `RAYD_TRACE_VISIBILITY_BACKEND` env-var override (see "RayD primitive availability" above). Channel-side: routed via `Scene._trace_segment_visibility_rayd` in [`core/scene/scene.py`](../../witwin/channel/core/scene/scene.py); the historical Python ignore re-fire loop is retired, and unsupported controls raise instead of falling back.
- [x] **T1.2 Thin shadow payload variant.** Equivalent capability shipped as part of T1.1 — the jit path uses `OptixScene::segment_hit<true>` which is the shadow-test thin variant (returns only the visible bit, no full Intersection). No separate `intersect_thin(ray)` API is needed: any caller that wants thin payload routes through `trace_segment_visibility` with no ignores and gets the jit-path shadow_test automatically. The original LoS at [deterministic/path/los.py:25](../../witwin/channel/deterministic/path/los.py#L25) and MC LoS at [montecarlo/trace/los.py:91](../../witwin/channel/montecarlo/trace/los.py#L91) already call `scene.ray_test` (jit-path shadow_test).
- [x] **T1.3 BDPT visibility always-native.** **Status: shipped.** The `2 * n_segments * n_triangles >= 4096` gate at the former [bdpt_diffraction.py:661-681] has been removed; `_segment_pair_visible` now unconditionally calls `scene.segment_pair_visible(...)` ([bdpt_diffraction.py:660](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L660)), which routes to `Scene.trace_segment_pair_visibility` under the dispatch rule above.

**Exit criterion**: Per-BDPT-sample OptiX launch count for visibility queries drops from 4–6 to 1–2 (one per logical visibility test, regardless of whether it has ignores). Tier 0 + Tier 1 combined: basic-MC ≤ 0.7× Sionna wallclock on the same Munich-class benchmark. *(2026-05-20 local Munich checkpoint: the basic-MC wallclock gate passes; see "Benchmark record — MC basic Munich". Still pending for formal closure: Nsight OptiX launch count, the pinned Sionna 2.0.1 dependency environment, ≥3-seed MC-noise acceptance, and BDPT visibility throughput validation.)*

### Tier 2 — Accumulating `trace_*` kernels (the head-to-head margin)

Goal: claim the structural advantage from plan 24 §4. Custom OptiX programs against RayD's GAS with in-kernel atomic accumulation into receiver bins.

- [x] **T2.1a `trace_reflections_accumulating` complex-polarized non-AD fast path.** RayD now exposes a native `optixLaunch` kernel that accumulates complex reflection field (`reflection_field_x/y/z`) plus scalar power into the receiver grid and returns a wedge event buffer. Channel integration is explicit via `accumulation_backend="rayd_reflection_accumulation"`; `auto` does not silently switch. Channel passes endpoint TX polarization to RayD, writes RayD field output into coherent/coherent-power diagnostics, and raises if the complex field payload is missing.
- [x] **T2.1b Strict AD contract for RayD accumulation.** The RayD accumulation kernel remains a native `optixLaunch` non-AD path. Channel rejects `accumulation_backend="rayd_reflection_accumulation"` as soon as AD is requested or inferred; AD users must explicitly select `accumulation_backend="native_monte_carlo"` to use the existing fixed-topology tape path. This is not a fallback.
- [x] **T2.2a Direct wedge event buffer handoff for depth-0 diffraction seeds.** Channel converts RayD `wedge_events` into `DiffractionHitStore` for the basic MC direct wedge path.
- [ ] **T2.2b Prefix wedge event handoff.** Still open. RayD reports prefix hit geometry, but channel still needs image-source/source-power/prim-history reconstruction for reflected-prefix diffraction. Explicit RayD backend raises on `collect_wedge_prefixes=True`.

**Exit criterion**: On the Munich benchmark, basic-MC wallclock ≤ 0.4× Sionna's. Nsight Compute confirms the SB-equivalent kernel is launch-bound by occupancy (not by per-launch overhead). Numerical agreement with the Tier 0 path on radio map L1 < 1% MC noise.

### Tier 3 — Diffraction chain fusion (deferred)

Deferred by the 2026-05-20 consolidation decision. Do not start Tier 3 until the Tier 2 RayD accumulation contract, AD behavior, and benchmark/parity gates below are stable.

Goal: collapse the order-N visibility chain in BDPT diffraction into a single launch. This is plan 24 §7-D2's MC sibling.

- [ ] **T3.1 Per-edge MIS / UTD cache.** New CUDA kernel that, given the active edge list, pre-computes per-edge constants (exterior angle, integration weight, UTD geometric coefficients independent of the sample direction). Loaded once per state; sample loop only gathers. Targets P3 + P4 directly.
- [ ] **T3.2 `trace_diffraction_chain_mc(rays, edges, max_order, mis_state)`** in RayD. Single launch walks `TX → e₁ → e₂ → ... → RX` per ray. Payload carries `Complex2f` field components and per-edge lineage. MC variant of plan 24 §7-D2 — does **not** require plan-22's discontinuity smoothing (that constraint applies only to the deterministic D2). **Honest expected speedup: 2–4× over Dr.Jit symbolic** (per rf_trace_kernel_plan §8.3), not the 3–10× implied by the original D2 framing — the dominant cost is per-edge UTD coefficient evaluation which fusion does not eliminate. Size the T3 exit criterion accordingly.
- [ ] **T3.3 BDPT chain rewrite.** Replace the inter-edge loop at [bdpt_diffraction.py:1130-1167](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L1130) and the suffix visibility chain at [bdpt_diffraction.py:1680-1689](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L1680) with single calls to T3.2.
- [ ] **T3.4 State pruning.** Split `DiffractionStates` so `FaceMaterial` is keyed by `edge_index` in a separate table and gathered only on accepted hits. Targets P6. Self-contained on the witwin side; no RayD dependency.
- [ ] **T3.5 AD tape packing.** Pack the 10-field per-sample tape scatter at [bdpt_diffraction.py:117-146](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L117) into a single struct, one scatter per accepted sample. Targets P5.

**Exit criterion**: BDPT diffraction-dominant benchmark wallclock ≤ 0.5× the post-Tier-2 baseline. Multi-order diffraction throughput ≥ 3× pre-Tier-3. Numerical agreement on representative paths within Monte Carlo noise.

### Tier 4 — Polish (only after Tier 3 is shown to win)

- [x] **T4.1 Source-edge axial visibility fusion.** Fold the per-fraction loop at [montecarlo/trace/postprocessing.py:140-163](../../witwin/channel/montecarlo/trace/postprocessing.py#L140) into a single launch (plan 24 §7-M5). **Status: shipped.** RayD-side `Scene.trace_axial_edge_visibility(...)` is routed through `Scene.axial_edge_visible(...)`; MC shadow-boundary source-edge pre-culling now passes the constant `(0.02, 0.25, 0.5, 0.75, 0.98)` tuple once instead of looping over `segment_visible`. Honest gain: **2–4×** (not the 3–5× originally quoted; capped by `n_samples=5`).
- [ ] **T4.2 Hash-based MC dedup.** If profiling shows duplicate specular chains hurting Tier 2 throughput on dense reflection scenes, add the dual FNV-1a + `dr.scatter_inc` pattern. Reference: `sionna-rt-reference-2.0.0/src/sionna/rt/utils/hashing.py`. Skip if not measurably needed.

## Benchmark protocol

Every tier's exit criterion uses the **same** protocol. No tier may declare exit on synthetic micro-benchmarks alone.

1. **Scene**: Munich-class building dataset shared with plan 24 §4 evaluation. Single-precision, ≈10⁵ triangles, ≈10³ edges.
2. **Parameters**: `samples_per_tx = 10⁶`, `max_bounces = 5`, 1 TX, planar receiver grid 256 × 256 cells, 2.4 GHz carrier.
3. **Hardware**: pinned to a single GPU model (RTX 4090 reference). Record driver / CUDA / OptiX versions.
4. **Sionna baseline**: in-tree **`sionna-rt-reference-2.0.1`** (drjit 1.3.1, mitsuba 3.8.0) invoked via its `RadioMapSolver` with `loop_mode="symbolic"` (the fast path). Record both wallclock and Nsight launch counts. Use 2.0.0 only when diffing across Sionna versions.
5. **Numerical equivalence**: per-tier regressions in radio-map L1 must stay within the per-run MC noise band (≥ 3 independent seeds).
6. **AD comparison**: gated by V1 (see W1 verification gate). Compare Sionna `main` (drjit 1.3.1, symbolic+replay) against witwin's integer-topology tape on the same (N, max_depth) sweep. Report peak VRAM and wallclock side-by-side. The claim under test is no longer "Sionna can't AD symbolic" but "witwin's tape-replay representation scales better in memory at city-scale N". Do not publish AD claims before V1 closes.

Benchmarks live under `tests/support/bin/` (per [CLAUDE.md](../../CLAUDE.md) "exploratory or script-style gradient workflows" rule) with a results record in `docs/dev/optimization/`.

### Benchmark record — MC basic Munich (2026-05-20)

Purpose: local checkpoint for the current `montecarlo` **basic** integrator against the Sionna RT 2.0.1 source tree. This is a solver steady-state timing record, not a formal external benchmark report.

Environment:

| Item | Value |
|---|---|
| Conda env | `witwin2` |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| Driver | 596.49 |
| Sionna source | `E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src` |
| Loaded Sionna package | `E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\__init__.py` |
| Runtime Dr.Jit / Mitsuba | `drjit==1.2.0`, `mitsuba==3.7.1`, variant `cuda_ad_mono_polarized` |
| Caveat | This uses the Sionna 2.0.1 source tree, but not the pyproject-pinned runtime (`drjit==1.3.1`, `mitsuba==3.8.0`). Re-run in a pinned Sionna 2.0.1 environment before using the numbers externally. |

Scene and solver parameters:

| Item | Value |
|---|---|
| Scene | Munich XML from Sionna RT 2.0.1, `merge_shapes=True` |
| Imported witwin scene | 11 structures, 38,936 triangles, 51,631 diffraction edges with the benchmark `all_edges + half_plane` policy |
| Frequency | 2.4 GHz for the RayD T2 stress run; older local rows below used 3.5 GHz |
| TX | `(8.5, 21.0, 27.0)` |
| Grid | plane `z=1.5`, bounds `(-120, 120) x (-120, 140)` |
| Samples | `samples_per_tx=1,000,000`; RayD T2 row uses seeds `11,17,23` |
| Interactions | LoS + specular reflection + first-order diffraction, edge diffraction on, refraction off |
| Witwin settings | `integrator="basic"`, `shadow_boundary_mode="none"`, `EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane")`; RayD T2 row uses `accumulation_backend="rayd_reflection_accumulation"` |
| Sionna settings | `RadioMapSolver`, `loop_mode="symbolic"`, `diffraction=True`, `edge_diffraction=True`, `diffraction_lit_region=True` |
| Timing method | Warmed steady-state solve time, result synchronized; scene load excluded |

Results:

| Case | Witwin median | Sionna median | Witwin / Sionna | Sionna / Witwin | Witwin sum | Sionna sum |
|---|---:|---:|---:|---:|---:|---:|
| RayD T2, corrected edge diffraction policy, `256x256`, `max_bounces=max_depth=5`, warmup 2, repeats 5, seeds `11,17,23` | 15.02 ms | 32.81 ms | 0.473x | 2.11x | not accepted | not accepted |
| `128x128`, `max_bounces=max_depth=5`, warmup 2, repeats 5 | 11.51 ms | 29.28 ms | 0.393x | 2.54x | `4.1211e-05` | `3.9258e-05` |
| `512x512`, `max_bounces=max_depth=5`, warmup 2, repeats 5 | 15.02 ms | 31.78 ms | 0.473x | 2.12x | `6.5898e-04` | `6.2868e-04` |
| `512x512`, `max_bounces=max_depth=1`, warmup 2, repeats 5 | 15.23 ms | 28.67 ms | 0.531x | 1.88x | `6.3815e-04` | `6.3731e-04` |

Interpretation:

- The RayD T2 stress run is now the maintained checkpoint: `256x256`, `1e6`, `max_depth=5`, seeds `11,17,23`, `2.4 GHz`, `accumulation_backend="rayd_reflection_accumulation"`, and Witwin edge diffraction matched to Sionna with `boundary_edge_policy="half_plane"`. Per-seed Witwin/Sionna median ratios in the corrected run were `0.473x`, `0.620x`, and `0.417x`; the median per-seed ratio is `0.473x` Sionna.
- The local wallclock checkpoint still passes the Tier 0 goal (`<=1.1x` Sionna) and the Tier 1 combined basic-MC wallclock goal (`<=0.7x` Sionna), but it does **not** close the full Tier 2 exit criterion (`<=0.4x` Sionna plus full field/polarization parity).
- The earlier RayD T2 row was misconfigured for parity: it passed `edge_diffraction=True` together with `boundary_edge_policy="exclude"`, which `EdgePolicy` resolved to closed-wedge-only diffraction. The benchmark runner now rejects that contradiction and uses half-plane boundary edges to match Sionna's `edge_diffraction=True` semantics.
- Numerical parity is reopened. Corrected half-plane edge diffraction raises Witwin diffraction strength substantially, but also exposes high-variance / outlier behavior in the current MC basic edge-diffraction estimator. The maintained stress output remains finite and shape-matched, but its sums are not accepted for parity until the half-plane diffraction estimator is audited against Sionna component-wise and across repeated processes.
- A Dr.Jit kernel-history run at `128x128`, `1e6`, `max_bounces=5` recorded 46 Witwin history records (`JIT=23`, `Reduce=12`, `Other=11`) and 33 Sionna history records (`JIT=7`, `Reduce=3`, `Other=23`). This is **not** an Nsight OptiX launch count and must not be used to close the launch-count part of T0.
- Witwin scene import was slower in this checkpoint (about 5.2 s vs Sionna about 1.3 s). The plan's current target is solver steady-state wallclock, not scene-load latency.

### Remaining gaps and temporary compromises

- **Formal benchmark protocol is partially closed.** The RayD T2 stress row now uses `256x256`, `2.4 GHz`, `samples_per_tx=1,000,000`, `max_depth=5`, and 3 seeds on RTX 5080. Still open: pinned GPU target, Sionna 2.0.1's pyproject-pinned runtime (`drjit==1.3.1`, `mitsuba==3.8.0`), Nsight launch counts, and formal MC-noise/L1 acceptance.
- **Benchmark runner is now maintained.** `tests/support/bin/benchmark_mc_basic_munich_vs_sionna.py` runs the formal Munich MC-basic comparison defaults (`256x256`, `2.4 GHz`, `samples_per_tx=1,000,000`, `max_depth=5`, seeds `11,17,23`) against Sionna RT 2.0.1 and writes machine-readable output to `docs/dev/optimization/`. The current RayD T2 stress output is `docs/dev/optimization/mc_basic_munich_rayd_accum_stress.json`.
- **T0 direct-scatter cleanup is incomplete.** Basic direct mode now scatters into the grid, but `GridContributionStore` still exists as a direct/staging abstraction and BDPT still uses it. The old double-write hot path is avoided for the measured basic path, but the cleanup note "delete `GridContributionStore` if no other caller remains" is not complete.
- **T0 launch-count gate is still open.** Dr.Jit history is useful for smoke checks, but it is not the requested Nsight Systems OptiX launch count. The `≤5` launch claim remains unclosed until measured with Nsight.
- **Basic diffraction visibility is reduced to two RayD pair queries.** `trace/diffraction.py` now uses `scene.segment_pair_visible(...)` for source base/offset and target base/offset checks, replacing the previous four `segment_visible(...)` calls. A single source+target fused query could still save one more logical call, but it would require a new RayD primitive and is not part of the current T0/T1 closure.
- **Tier 1 BDPT integration is implemented but not benchmark-closed.** Pair visibility now routes through `scene.segment_pair_visible(...)`, but BDPT-specific launch-count and throughput acceptance have not been measured on Munich.
- **Tier 2 is implemented as a non-AD RayD fast path, but not acceptance-closed.** RayD complex-polarized `trace_reflections_accumulating` is installed and channel can use it through `accumulation_backend="rayd_reflection_accumulation"`; depth-0 wedge events are handed to `DiffractionHitStore`. The corrected Munich stress run is recorded, but the result is `0.473x` Sionna rather than the Tier 2 target `<=0.4x`, and the half-plane edge-diffraction numerical parity gate is reopened. Remaining gaps: no AD support for this RayD backend by design, no prefix wedge reconstruction, no Nsight closure, no half-plane diffraction estimator audit, and no formal MC-noise/L1 parity gate.
- **Tier 3 is deferred.** Per-edge MIS/UTD caches, `trace_diffraction_chain_mc`, BDPT chain rewrite, state pruning, and AD tape packing remain open by decision; do not spend effort here before Tier 2 is stable.
- **T4.1 landed early.** Source-edge axial visibility fusion is shipped even though Tier 4 was originally "polish after Tier 3". This is a sequencing deviation, not a fallback path; keep it because it is low risk and already tested.
- **T4.2 is intentionally unstarted.** Hash-based MC dedup should stay pending until profiling shows duplicate specular chains are actually a bottleneck.
- **AD claims remain gated.** V1.3-V1.5 are still open, so no external claim about witwin's AD memory scaling versus Sionna symbolic replay should ship yet.
- **Strict RayD control limits are visible by design.** Visibility helpers raise on unsupported controls such as unsupported `ignore_structure_idx` mappings. The RayD reflection-accumulation backend also raises on AD, prefix wedges, missing RayD API, missing complex field payloads, and wedge buffer overflow. This matches the "no fallback" decision, but it is still a product limitation until RayD supports the missing control surface.

## Risk register

- **R1 — Atomic contention in T0.2 / T2.1.** Worst case is a small receiver grid where many rays converge on the same cell. Mitigation: confirm Sionna handles this with the same `dr.scatter_reduce` and accept the same cost. If contention is provably worse than Sionna's, fall back to per-block staging before final atomic add — but only if measured, not pre-emptively.
- **R2 — T2.1 payload register pressure.** Plan 24 §8 flags this as an open question. Budget the polarization payload before committing. If it spills, split into raygen + closesthit-only field accumulation, keeping polarization in GMEM keyed by ray index.
- **R3 — T3.2 register pressure (worse than T2.1).** Same mitigation. If the full chain payload does not fit, fall back to two-launch chain (odd orders / even orders) before abandoning fusion.
- **R4 — RayD bandwidth.** Tier 1 RayD work, T4.1, and Phase 2 support are shipped (`trace_segment_visibility`, `trace_segment_pair_visibility`, `trace_axial_edge_visibility`, `trace_segment_chain_visibility`, `trace_reflections` trailing fields, `trace_reflections_accumulating` complex-polarized non-AD path, depth-0 wedge event buffer, `nearest_edges_topk`, edge BVH OptiX migration). Remaining RayD-side dependencies are concentrated in prefix wedge ABI, possible future AD-capable accumulation design, and deferred Tier 3 (`trace_diffraction_chain_mc`).
- **R5 — AD path regression.** Every tier must keep the PyTorch-native AD boundary intact. Result is data-only (`feedback_result_data_type.md`). If a kernel-side optimization breaks gradient continuity, it is reverted and re-architected, not patched with autograd shims.

## Out of scope

- Deterministic-solver acceleration. Covered by plan 24 §7-D1/D2/D3 + plan 22's discontinuity work.
- T-matrix scattering integration. Tracked in `project_witwin_tmatrix_scattering_plan.md`.
- Reflection F-weight differentiability rollout. Tracked in plan 25.
- Mitsuba comparison. Settled in plan 24 §4.

## Cross-references

- **Strategy**: [`24-realtime-rt-architecture-roadmap.md`](24-realtime-rt-architecture-roadmap.md) — read §4 (Mitsuba comparison, structural-vs-patch grading) and §7 (fusion inventory, end-to-end ~1.5× honest re-calibration) before starting any RayD-side task.
- **RayD-side companion (read before any T1/T2/T3 RayD task)**: [`rayd/docs/rf_trace_kernel_plan.md`](file:///E:/Code/RayDi/docs/rf_trace_kernel_plan.md) — Phase 1 / Phase 2 kernel inventory, backend dispatch (`RAYD_TRACE_VISIBILITY_BACKEND`), honest per-kernel speedup audit in §8.3.
- **RayD edge-BVH details** (delivered): [`rayd/docs/edge_bvh_optix_migration_plan.md`](file:///E:/Code/RayDi/docs/edge_bvh_optix_migration_plan.md) — OptiX custom AABB migration; covers `nearest_edges_topk`, `set_edge_mask` semantics, refit path.
- Deterministic discontinuity prerequisite for D2 (not blocking T3.2's MC variant): [`22-deterministic-discontinuity-plan.md`](22-deterministic-discontinuity-plan.md).
- RFDT reflection differentiability (parallel work, must not regress): [`25-rfdt-reflection-f-weight-plan.md`](25-rfdt-reflection-f-weight-plan.md).
- Sionna upstream reference (use this for all external claims): `e:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\` — drjit 1.3.1, mitsuba 3.8.0. Read the source, not the paper.
- Sionna 2.0.0 snapshot (history / diff only): `channel/sionna-rt-reference-2.0.0/src/sionna/rt/`.
