# RFDT Reflection F-Weight Differentiability Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-20

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## 2026-05-20 Revision Constraints

This revision changes the plan from a direct always-on rewrite into a staged, opt-in feature rollout. The feature must be switchable at runtime, default to the current hard visibility behavior until reference and native parity pass, and preserve a fast path that avoids computing F-weights for paths whose reflection points are far from any surface boundary.

The implementation must follow the current source tree. Reflection EPC lives under `witwin/channel/deterministic/reflection/`, native reflection kernels live under `witwin/channel/deterministic/kernels/reflection/`, and Monte Carlo diffraction AD support lives in `witwin/channel/montecarlo/trace/diffraction_ad.py`.

The low-overhead design target is not unbounded per-edge branch enumeration. Reflection F-weighting should be evaluated only for candidate surface-boundary edges near each reflection point, capped by a small per-slot edge budget, and skipped entirely when the feature flag is disabled or when the point is confidently inside the current surface group.

## Goal

Replace the binary surface-boundary visibility mask on reflection paths with the physically grounded UTD transition-function weighting `E = F(x_a) E_a + F(x_b) E_b` so that the deterministic reflection EPC pipeline becomes continuously differentiable with respect to scene geometry as reflection points approach and cross wedge edges. The forward simulation must remain numerically faithful in the surface-interior limit, while gradients in the near-edge transition band must become well-defined and physically consistent.

This plan focuses on the reflection side of the contribution in MobiCom '26 paper "Physically Accurate Differentiable Inverse Rendering for Radio Frequency Digital Twin" (Chen et al., henceforth "RFDT"). The signal-domain surrogate (paper §4) and the DT optimization loop (paper §5.2) are out of scope here and tracked separately under the research feature roadmap.

## Paper Core Content (Gradient Handling)

The paper's differentiability contribution is concentrated in §3.2 and §3.3 and rests on four mechanisms.

### Reparameterized RF ray tracing (§3.2)

The classical method of images is reformulated so that the path geometry is constructed from the **infinite supporting planes** of the candidate triangles instead of from per-bounce ray-triangle hit tests. For a path with candidate triangles `△_1, …, △_m`, the reparameterized constructor `T_path` produces a complete ordered path

```
P = {p_tx, p_ref^(1), …, p_ref^(m), p_rx}
  = T_path(p_tx, p_rx, {S(△_i)}_{i=1..m})
```

where `S(△_i)` is the infinite plane supporting `△_i`. Because the construction is purely plane intersection plus mirror reflection, the resulting reflection points, path length `L_P`, amplitude `A_P`, and phase are all continuous and differentiable with respect to `p_tx`, `p_rx`, and every vertex `v_{i,j}`.

### Physically consistent edge transition (§3.3, primary visibility)

The binary in-triangle test from the classical formulation,

```
W(P) = ∏_{i=1..m} [ p_ref^(i) ∈ △_i ]                                    (Eq. 3, discontinuous)
```

is replaced by a multiplicative UTD weight

```
W_RFDT(P) = ∏_{i=1..m} F( k L_a(α_i) )                                    (Eq. 11)
```

where `F` is the UTD transition function

```
F(x) = 2 j √x e^{j x} ∫_{√x}^{∞} e^{-j τ²} dτ                             (Eq. 8)
```

with asymptotic limits `F(x → ∞) = 1` (full reflection recovered) and `F(x → 0) = 0` (suppressed near the boundary). For a single wedge interaction across an edge `E` shared by faces `△_a` and `△_b`, the field is

```
E^RFDT_E = F(x_a) E_a + F(x_b) E_b                                        (Eq. 14)
```

with `x_a = k L · a_a`, `x_b = k L · a_b`, where `L = s s' / (s + s')` is the effective distance and `a_a`, `a_b` are normalized angular squared deviations from the reflection boundary on each face.

### Secondary visibility via diffraction (§3.3)

When a specular segment is occluded by an unrelated obstacle `△_c`, the binary mask `H(γ)` is replaced by `w(γ) = F(k L_a)` where `L_a` is the effective distance from the silhouette edge `E_c` of the blocker to the receiver. The diffraction path through the blocker's silhouette edge supplies the complementary field via the standard UTD diffraction coefficient.

### Closed-form chain rule (§3.3, differentiation)

The gradient of the received field with respect to a mesh vertex `v` follows the chain rule

```
∂E/∂v = Σ_P ∂/∂v [ W(P) · A_P · e^{-j k L_P} ]                            (Eq. 10)
∂F(x)/∂v = F'(x) · ∂(k L_a)/∂v                                            (Eq. 16)
∂E_P/∂v = (∂A_P/∂v − j k A_P ∂L_P/∂v) e^{-j k L_P}                       (Eq. 17)
```

Each factor is composed of vector additions, dot products, and cross products on differentiable plane geometry, so the gradient is well-defined and computable via automatic differentiation once `H(·)` is replaced by `F(·)` in the path weight.

## Current Codebase Status

### What is already in place

| Paper element | Repo location | State |
| --- | --- | --- |
| Stage-1 candidate triangle sequence discovery | [`witwin/channel/deterministic/reflection/paths.py`](../../witwin/channel/deterministic/reflection/paths.py) `trace_paths`, `collect_prefix_paths`, `enumerate_first_bounce_surface_paths` | Complete |
| Stage-2 reparameterized image-source path construction (`T_path`, Eq. 2) | [`witwin/channel/deterministic/reflection/epc.py:269`](../../witwin/channel/deterministic/reflection/epc.py#L269) `chain_math_reference`, native `reflection_epc_targets_forward` | Complete |
| UTD transition function `F(x)` (Eq. 8) | [`witwin/channel/core/wave_math.py:156`](../../witwin/channel/core/wave_math.py#L156) `f_utd`, Boersma polynomials at [`wave_math.py:123`](../../witwin/channel/core/wave_math.py#L123) `fresnel_integral` | Complete |
| Wedge geometry (φ, φ', s, s', edge frames) | [`witwin/channel/core/diffraction_geometry.py`](../../witwin/channel/core/diffraction_geometry.py) | Complete |
| Per-vertex / per-face Jones JVP/VJP for diffraction | [`witwin/channel/montecarlo/trace/diffraction_ad.py`](../../witwin/channel/montecarlo/trace/diffraction_ad.py) `DiffractionAD` | Complete |
| Custom backward for reparameterized EPC | [`witwin/channel/deterministic/reflection/epc.py:528`](../../witwin/channel/deterministic/reflection/epc.py#L528) `EpcTargetsOp` + native CUDA forward | Complete |
| Full UTD diffraction with transition functions in the diffraction coefficient | [`witwin/channel/deterministic/diffraction/forward.py`](../../witwin/channel/deterministic/diffraction/forward.py), [`witwin/channel/montecarlo/trace/diffraction_utd.py`](../../witwin/channel/montecarlo/trace/diffraction_utd.py) | Complete |

### What is missing or divergent from the paper

| Paper element | Current behavior | Issue |
| --- | --- | --- |
| Eq. (3) to Eq. (14) replacement on **reflection** validity | [`epc.py:671-672`](../../witwin/channel/deterministic/reflection/epc.py#L671) and [`epc.py:702-703`](../../witwin/channel/deterministic/reflection/epc.py#L702) use `surface_contains(...)`; [`epc.py:728`](../../witwin/channel/deterministic/reflection/epc.py#L728) uses `vector_select(valid, chain_vector, vector_zero(...))` | Still the binary `H` mask. Reflection field is killed discontinuously when `p_ref` crosses a surface boundary. |
| Eq. (9) secondary visibility `w(γ) = F(k L_a)` | `shadow_completion_weight_*` family in [`wave_math.py:200`](../../witwin/channel/core/wave_math.py#L200), `_shadow_completion_weight_from_normalized_distance` uses `smoothstep01` + power decay | Smooth but not the UTD `F(·)`. Used only in post-processing, not woven into reflection. |
| Energy redistribution at wedge | Reflection (binary) and diffraction (full UTD) are added as independent additive components | Reflection still introduces a hard `0/1` jump that the additive diffraction term does not cancel. Behavior matches Sionna v0.19's failure mode rather than RFDT. |

The reparameterization makes intra-surface gradients smooth, but every surface-boundary crossing still produces a Heaviside step in the reflection contribution. Net result: the pipeline is **AD-differentiable in the technical sense** (DrJit computes gradients) but **not "physically consistent differentiable"** in the sense the paper defines.

## Gap Analysis

The required change is not a finalization-only edit. Current deterministic radiomap flow computes reflection visibility in `witwin/channel/deterministic/reflection/epc.py`, then passes a hard `valid_mask` into `witwin/channel/deterministic/kernels/reflection/native_impl.py` and the CUDA accumulation kernel. Removing `surface_contains(...)` without adding the F-weight in both EPC and accumulation would turn off the guard while still accumulating full-strength off-surface reflections. Concretely:

1. The reflection EPC must compute a complex transition weight for the primary surface and, only near an eligible boundary, an optional adjacent-surface residual.
2. The F-weight argument must include explicit side gating. A distance-only construction is not sufficient because it cannot distinguish the primary lit side from the adjacent/shadow side.
3. The downstream chain fold must keep the current single-branch fast path for interior reflections and cap near-boundary work by `reflection_f_weight_max_edges_per_slot`.
4. The `surface_contains` binary kill must remain active in `hard` mode and be replaced only in F-weight modes.
5. The native CUDA forward kernel, JVP path, and `EpcTargetsOp` reference backward must mirror the same math before `f_weight_native` is enabled.

The independently implemented diffraction pipeline already handles UTD diffraction terms, but it does not by itself cancel the reflection-side Heaviside step. Reflection F-weighting must produce a complex visibility weight and optional adjacent-surface contribution before final field accumulation. Secondary visibility and matched-ISB shadow-boundary post-processing are separate rollout items and must remain switchable to avoid double-counting near shadow boundaries.

## Switchable Low-Overhead Architecture

Add a new reflection transition mode with these semantics:

- `hard`: existing behavior. Keep `surface_contains(...)`, pass a binary `valid_mask`, and skip all reflection F-weight buffers. This remains the default until acceptance passes.
- `f_weight_reference`: DrJit/reference-only mode for tests and finite-difference validation. It computes near-boundary F-weights and adjacent-surface contributions without changing the native accumulation ABI.
- `f_weight_native`: production candidate. It uses an extended native CUDA accumulation ABI for transition descriptors, primary F-weights, bounded adjacent residual branches, and receiver scatter. Hard mode keeps the original ABI. The reference mode remains the validation oracle, not a near-boundary fallback for native mode.

The switch should live in deterministic solver tuning/config, not as a public scene API. A suitable internal name is `reflection_transition_mode`, with accepted values `hard`, `f_weight_reference`, and `f_weight_native`.

The performance design is:

- Precompute or gather surface-boundary candidate edges through the existing scene edge runtime (`scene._selected_edge_runtime()`, `scene.get_triangle_surface_edge_candidates(...)`, and the surface group edge buffers) instead of building a raw `tri_adjacent_prim[prim, 3]` table as the primary interface.
- For each reflection slot, compute a cheap inside/far-boundary test first. If the point is inside the surface group and farther than `reflection_f_weight_boundary_radius_wavelengths * wavelength` from every candidate boundary edge, return weight `1 + 0j` and no adjacent branch.
- For near-boundary points, evaluate at most `reflection_f_weight_max_edges_per_slot` candidate edges, default `1`. This means "nearest active boundary edge first"; increasing the cap is an acceptance/diagnostic option, not the production default.
- Do not enumerate all branch products. For multi-bounce paths, carry one primary chain plus a bounded residual adjacent contribution per near-boundary slot. If more than one slot is near a boundary, either cap to the strongest transition slot in production or run the uncapped reference only in small validation tests.
- Keep all F-weight math DrJit-native and CUDA-native. Do not add NumPy, Torch, DLPack, or CPU fallback paths.

## Revised Modification Plan

Each stage below should land as its own change with tests. Later stages depend on earlier ones.

### Stage 1: Feature switch and hard-mode parity

- [x] Add `reflection_transition_mode: Literal["hard", "f_weight_reference", "f_weight_native"] = "hard"` to deterministic tuning/config resolution.
- [x] Add `reflection_f_weight_boundary_radius_wavelengths: float = 2.0` and `reflection_f_weight_max_edges_per_slot: int = 1` as advanced tuning fields.
- [x] Thread the resolved values into reflection trace detail or the reflection runtime call path without changing behavior when mode is `hard`.
- [x] Add a config test that asserts default solves still report/use `hard` mode and that invalid mode names raise `ValueError`.
- [x] Run the existing deterministic reflection/radiomap tests in `hard` mode and require byte-for-byte or existing-tolerance parity.

### Stage 2: Surface-boundary edge support for reflection

- [x] Reuse the existing scene edge runtime instead of adding a raw triangle-only adjacency table. Audit `scene.get_triangle_surface_edge_candidates(...)`, `tri_data["surface_edge_indices"]`, and `_selected_edge_runtime()` for the data needed by reflection F-weighting.
- [x] Add a small reflection-owned helper that gathers, for a reflection primitive or surface group, the nearest eligible boundary edge to `hit_p`: edge endpoints, adjacent faces, face normals, edge direction, and a valid mask.
- [x] The helper must return empty/invalid support when mode is `hard`, when edge runtime is absent, or when the point is outside the configured radius.
- [x] Cover the helper with `tests/scene/test_reflection_surface_boundary_support.py` using a two-triangle coplanar surface, a wedge with two adjacent surfaces, and a boundary edge.

### Stage 3: DrJit reference F-weight math

- [x] Add a small pure DrJit math helper under `witwin/channel/deterministic/reflection/epc.py` or a focused sibling module if `epc.py` becomes too large.
- [x] Inputs: primary hit point, previous point, next point, primary plane, nearest boundary edge support, wave number, and material/default gain data. The pure F-weight helper owns the geometric/UTD inputs; EPC branch replay resolves the adjacent material/default gain inputs.
- [x] Outputs: `primary_weight: Complex2f`, `adjacent_weight: Complex2f`, `adjacent_plane_point`, `adjacent_plane_normal`, and `adjacent_valid`.
- [x] Define side gating explicitly: the primary side tends to `1 + 0j` deep inside the primary surface, tends to `0 + 0j` at the boundary, and is zeroed beyond the configured transition radius on the shadow side. The adjacent side is emitted only when a valid adjacent surface exists.
- [x] Use the existing `f_utd` for the forward value and mirror the safe small-`x` handling patterns in `witwin/channel/deterministic/diffraction/math.py` before relying on gradients at `x ~= 0`.
- [x] Add unit tests that sweep a single reflection point toward a boundary and assert continuity, finite gradients, and hard-mode parity away from the boundary.

### Stage 4: Reference EPC integration

- [x] Keep `Descriptor` unchanged for reference mode and gather bounded edge support, adjacent surface material, and tuning scalars in the reference finalization path. This avoids growing the native descriptor before the CUDA ABI is ready.
- [x] In `f_weight_reference` mode, apply a complex chain weight plus a bounded adjacent residual contribution around the `chain_math_reference` output.
- [x] Keep `surface_contains(...)` in `hard` mode. In F-weight mode, replace the final binary surface kill with the computed transition weight, while keeping `valid_prim`, EPC denominator checks, and segment visibility checks.
- [x] Update `return_endpoints` and `return_geometry` semantics: these continue to report the primary chain geometry. Adjacent residual geometry is not exported until a separate path-export plan defines the record format.
- [x] Add reference tests comparing `hard` and `f_weight_reference` far from boundaries, adjacent residual behavior after crossing one wedge edge, and finite-difference tests near one boundary.

### Stage 5: Native accumulation integration

- [x] Update `witwin/channel/deterministic/kernels/reflection/native_impl.py` so `hard` mode keeps the current ABI and `f_weight_native` uses an extended CUDA forward ABI.
- [x] Update `reflection_accumulate.cu`, `bind.cpp`, and the reflection headers for the extended CUDA forward ABI.
- [x] Extend `reflection_jvp.cu` / native custom AD so `f_weight_native` accumulation has CUDA-side transition JVP parity instead of relying on the reference gradient oracle.
- [x] The CUDA kernel preserves the fast interior path inside the F-weight kernel: pairs with no transition support skip F-weight and adjacent-branch evaluation and execute the hard reflection contribution.
- [x] Add native/reference parity tests with `reflection_f_weight_max_edges_per_slot=1`, including a guard that `f_weight_native` near-boundary accumulation does not call reference pair replay and an adjacent-residual parity case.
- [x] Add a benchmark guard on a three-cube radiomap: `f_weight_native` should not exceed `hard` mode by more than 50% on the sparse-boundary guard workload.

2026-05-20 update: `f_weight_native` now resolves to `native_cuda_f_weight`. It no longer replays near-boundary pairs through `_scatter_f_weight_reference_pairs`; Python packs per-pair/slot transition support and adjacent-plane/material descriptors, and CUDA evaluates the primary F-weight product, bounded adjacent residual branches, source-field scaling, and receiver scatter. Targeted tests cover reference/native parity for primary transition accumulation and adjacent residual accumulation, and `test_f_weight_native_accumulation_does_not_replay_near_boundary` guards against reintroducing reference replay. The repeatable sparse-boundary guard is:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_reflection_f_weight_transition --json --strict-gate
```

The 2026-05-20 native CUDA run on the 12 x 12 / 32-ray / first-order three-cube guard passed with `hard=0.00438s`, `f_weight_native=0.00624s`, overhead `1.422x`, and metadata `resolved_backend="native_cuda_f_weight"` against the `1.50x` gate using `reflection_f_weight_boundary_radius_wavelengths=0.01`. Native custom AD now also propagates CUDA-side forward-mode JVP through Fresnel angle response, edge-distance transition weights using boundary edge endpoints, and bounded adjacent residual branches. The maintained double-slit acceptance covers Tx position, strip position, and strip yaw against finite differences, plus three-cube native/reference parity at one, two, and three reflection orders.

### Stage 6: Shadow-boundary and secondary visibility rollout

- [x] Keep existing matched-ISB and shadow-boundary correction independent while reflection F-weighting is experimental.
- [x] Add a guard that prevents matched-ISB correction from being silently double-applied when `reflection_transition_mode != "hard"`.
- [ ] Introduce a UTD-based secondary-visibility policy only after primary reflection F-weighting passes acceptance. This should be a separate switch, not the default in the same change. **Detailed sub-plan below: [Next Step: Secondary Visibility (Eq. 9) Rollout](#next-step-secondary-visibility-eq-9-rollout).**

### Stage 7: Solver and documentation integration

- [x] Surface the tuning option in deterministic solver metadata so result payloads record `reflection_transition_mode`.
- [x] Keep Monte Carlo reflection AD out of this plan unless deterministic FD tests prove the same transition is needed there. If it is needed, create a separate Monte Carlo plan.
- [x] Update `FEATURE_LIST.md` only when a non-default user-visible mode lands.
- [ ] Move this plan to `docs/dev/archive/completed/` only after reference, native, gradient, and performance acceptance all pass.

## Next Step: Secondary Visibility (Eq. 9) Rollout

Primary reflection F-weighting (paper Eq. 14) has landed in reference, native CUDA forward, and native CUDA JVP modes. The remaining discontinuity in the deterministic reflection pipeline is **secondary visibility**: a reflection segment `(p_ref → p_rx)` blocked by an unrelated obstacle `△_b` is still gated by the binary `H(γ)` mask through `scene.segment_visible(...)` in `epc.finalize_reference_f_weight_outputs` and the analogous loop in `finalize_native_outputs`. This is the second half of paper §3.3 and the missing piece before the reflection side of RFDT can be called paper-fidelity.

### Paper recap

Paper Eq. 9 replaces `E_GO = H(γ) E_a^spec` with

```
E_total = w(γ) E_a^spec + E_diffract,        w(γ) = F(k L_a)
```

where:

- `γ` is the signed clearance from the reflection segment to the silhouette edge `E_c` of the blocker `△_b`.
- `L_a` is the UTD effective distance from `E_c` to the receiver projected along the shadow boundary direction.
- `E_diffract` is the UTD diffraction field through `p_d ∈ E_c` evaluated by paper Eq. 6.

As `γ → 0`, the specular term decays smoothly via `F(k L_a) → 0`, the diffraction term carries the field, and total field continuity is preserved.

### Reused infrastructure (no new BVH)

The implementation reuses three existing repo capabilities; no new acceleration structure is required.

1. **RayD triangle BVH** through [`scene.ray_intersect`](../../witwin/channel/core/scene/scene.py#L604) and [`segment_visible`](../../witwin/channel/core/scene/scene.py#L727) for blocker detection along reflection segments. This is the same BVH that `segment_visible` already queries; secondary visibility replaces the binary outcome of that query rather than the query itself.
2. **RayD edge BVH** through [`scene.nearest_edge`](../../witwin/channel/core/scene/scene.py#L610) for silhouette-edge lookup. This BVH is already exercised by higher-order diffraction in [`witwin/channel/deterministic/diffraction/builders.py`](../../witwin/channel/deterministic/diffraction/builders.py#L425) under the `rayd_edge_bvh` backend, so the integration pattern is established.
3. **Scene edge runtime** (`pos`, `edge_dir`, `n0`, `n_face_n`, `adjacent_face0/1`, `length`, `global_idx`) exposed by `scene._selected_edge_runtime(...)`. The blocker silhouette edge's geometry and adjacent-face data are gathered through the same buffers consumed by primary reflection F-weighting.

### Switchable mode

Add a new tuning field `reflection_secondary_visibility_mode: Literal["hard", "f_weight"] = "hard"`. The two switches are independent:

| `reflection_transition_mode` | `reflection_secondary_visibility_mode` | Behavior |
| --- | --- | --- |
| `hard` | `hard` | Current default. Both primary and segment visibility are binary. |
| `f_weight_reference` / `f_weight_native` | `hard` | Primary F-weight active; segments still use binary `segment_visible`. This is the post–Stage 5 state of plan 25. |
| any | `f_weight` | Both primary and segment F-weights active. New rollout target. |

Coupling secondary visibility to primary mode is rejected: production rollouts should be able to ship primary F-weight independently of segment F-weight, and FD parity tests must be runnable for each axis in isolation.

### Current implementation map

| Feature / concern | Code entry points | Current status |
| --- | --- | --- |
| Secondary visibility public tuning switch | [`config.py`](../../../witwin/channel/deterministic/config.py) `ReflectionSecondaryVisibilityMode`, `Tuning.reflection_secondary_visibility_mode`, `ResolvedTraceConfig.reflection_secondary_visibility_mode`, `resolve_trace_config(...)` | Complete. Default is `hard`; accepts `f_weight`; invalid literals raise. |
| Trace-detail and reflection plumbing | [`reflection/detail.py`](../../../witwin/channel/deterministic/reflection/detail.py) `TraceDetail.reflection_secondary_visibility_mode`, `build_trace_detail(...)`, `coerce_trace_detail(...)`; [`trace/reflection.py`](../../../witwin/channel/deterministic/trace/reflection.py); [`reflection/accumulation.py`](../../../witwin/channel/deterministic/reflection/accumulation.py) | Complete. The resolved secondary mode is carried from config into reflection path detail and accumulation. |
| Solver metadata | [`solver.py`](../../../witwin/channel/deterministic/solver.py) `_build_metadata(...)` under `runtime_backends.reflection_secondary_visibility` | Complete. Metadata records mode and currently resolved backend (`hard` or `reference_segment_f_weight`). |
| Blocker detection and silhouette support | [`reflection/secondary_visibility.py`](../../../witwin/channel/deterministic/reflection/secondary_visibility.py) `SecondaryVisibilitySupport`, `nearest_blocker_silhouette_edge(...)`, `_first_non_ignored_blocker(...)`, `_nearest_selected_blocker_edge(...)` | Complete for the reference path. Uses RayD triangle intersection for the first non-ignored blocker, then scans selected scene edge runtime entries for the nearest edge on the blocker surface group. This avoids accepting unselected internal mesh diagonals and only scans edges after a segment is actually occluded. |
| Segment attenuation math | [`reflection/f_weight.py`](../../../witwin/channel/deterministic/reflection/f_weight.py) `reflection_segment_attenuation(...)` | Complete for specular attenuation. Clear segments return `1 + 0j`; valid near-silhouette blockers return `_safe_transition(k * gamma^2 / effective_L)`; occluded segments without valid support return `0 + 0j`. |
| Reference EPC integration | [`reflection/epc.py`](../../../witwin/channel/deterministic/reflection/epc.py) `_reference_segment_visibility_weight(...)`, `_reference_branch_visibility(...)`, `finalize_reference_f_weight_outputs(...)`, `chain_to_target(...)` dispatch | Partially complete. Reference specular chains and adjacent-residual branch visibility now multiply the secondary segment attenuation. The Eq. 9 diffraction residual term through the blocker silhouette edge is still open. |
| Primary/secondary switch independence | [`reflection/epc.py`](../../../witwin/channel/deterministic/reflection/epc.py) `chain_to_target(...)`; [`kernels/reflection/native_impl.py`](../../../witwin/channel/deterministic/kernels/reflection/native_impl.py) `reflection_accumulate_forward(...)` | Complete for reference execution. `hard + f_weight` routes to reference EPC secondary attenuation. `f_weight_reference + hard` remains primary-only F-weight. |
| Native CUDA behavior | [`kernels/reflection/native_impl.py`](../../../witwin/channel/deterministic/kernels/reflection/native_impl.py) `reflection_accumulate_forward(...)` | Guarded, not implemented. `f_weight_native + reflection_secondary_visibility_mode="f_weight"` raises immediately so the solver does not silently fall back or run the old CUDA ABI with incorrect weights. Native secondary descriptors, kernels, and JVP remain open. |
| Matched-ISB / shadow-boundary interaction | [`config.py`](../../../witwin/channel/deterministic/config.py) `resolve_trace_config(...)`; [`standards/40-diffraction-path-taxonomy.md`](../standards/40-diffraction-path-taxonomy.md); [`wave_math.py`](../../../witwin/channel/core/wave_math.py) module docstring | Complete. `shadow_boundary_correction=True` is rejected when either primary reflection transition F-weighting or secondary visibility F-weighting is active. Legacy `shadow_completion_weight_*` is documented as not the path for new reflection secondary attenuation. |
| Tests: config, support helper, reference behavior, native no-fallback | [`test_reflection_transition_config.py`](../../../tests/deterministic/test_reflection_transition_config.py); [`test_reflection_secondary_visibility_support.py`](../../../tests/scene/test_reflection_secondary_visibility_support.py); [`test_reflection_secondary_visibility_reference.py`](../../../tests/deterministic/test_reflection_secondary_visibility_reference.py) | Complete for first landing. Covers defaults, validation, metadata, matched-ISB rejection, clear/off-edge/grazing support, reference segment softening, switch independence, and native no-fallback guard. FD/AD image-quality acceptance is still open. |
| User-visible feature documentation | [`FEATURE_LIST.md`](../../../FEATURE_LIST.md); this plan | Complete. The feature list documents the new `reflection_secondary_visibility_mode="f_weight"` reference mode and the native limitation. |
| Final acceptance | Acceptance checklist below | Open. Remaining acceptance items are FD parity, reference/native CUDA parity, sparse-blocker benchmark, double-slit reproduction, and the R->D diffraction residual. |

### Sub-stages

#### Stage A: Blocker / silhouette query helper

- [x] Add `witwin/channel/deterministic/reflection/secondary_visibility.py` with a `SecondaryVisibilitySupport` dataclass carrying `blocker_prim_idx`, `blocker_surface_group`, `silhouette_edge_idx`, edge endpoints, edge frames (`n0`, `n_face_n`), `adjacent_face0/1`, `gamma` (signed clearance), `effective_L`, and an `is_occluded` mask. The structure mirrors [`ReflectionBoundarySupport`](../../witwin/channel/deterministic/reflection/boundary.py#L19).
- [x] Implement `nearest_blocker_silhouette_edge(*, scene, hit_p, rx_pos, primary_surface_group, transition_mode, wavelength, boundary_radius_wavelengths, edge_policy=None) -> SecondaryVisibilitySupport`:
  1. Build a `rayd.Ray` from a `RAY_ORIGIN_BIAS`-offset `hit_p` toward `rx_pos`, with `tmax = dist - 2*bias`.
  2. Query `scene.ray_intersect(ray)` while ignoring the primary reflection surface group. Record `blocker_prim_idx` and `blocker_surface_group`.
  3. For pairs with `blocker_prim_idx >= 0`, query `scene.nearest_edge(ray)` to obtain the candidate silhouette edge. Map `global_edge_id` back to local edge index through the same `global_to_local_idx` table the diffraction builders use.
  4. Restrict the candidate edge to ones whose `adjacent_face0` or `adjacent_face1` lies on the blocker (drop edges that belong to other geometry incidentally hit by the probe ray).
  5. Compute `γ` as the perpendicular distance from the segment line to the silhouette edge (skew-line distance with clamping at the edge endpoints).
  6. Compute `effective_L = s · s' / (s + s')` where `s = ||hit_p − p_d||` and `s' = ||rx_pos − p_d||`, with `p_d` the closest point on the silhouette edge to the segment.
  7. Gate by `boundary_radius_wavelengths * wavelength` and by `transition_mode`; return `_empty_support` outside the radius or when mode is `hard`.
- [x] Unit-test the helper under `tests/scene/test_reflection_secondary_visibility_support.py` with three scenes: clear LOS to receiver (expect `is_occluded = False`), single planar blocker with off-edge segment (expect `γ` large, `is_occluded = True`), and segment grazing the silhouette edge (expect `γ → 0`).

2026-05-20 update: the first landing uses RayD for the blocker triangle query, then selects the nearest eligible edge from `scene._selected_edge_runtime(...)` for that blocker surface group. This avoids accepting unselected internal mesh diagonals returned by raw nearest-edge probes while keeping the scan restricted to already-selected diffraction edges and only running it on occluded secondary segments.

#### Stage B: Reference EPC integration

- [x] Extend `f_weight.py` with `reflection_segment_attenuation(*, support, wave_k) -> Complex2f` that returns `safe_transition(wave_k * γ² / effective_L)`. Reuse the existing `_safe_transition` helper so small-`x` behavior is consistent with primary weighting.
- [x] In [`epc.finalize_reference_f_weight_outputs`](../../witwin/channel/deterministic/reflection/epc.py#L873), for each pair of consecutive path points (`tx → first hit`, `hit_i → hit_{i+1}`, `last_hit → target`), call `nearest_blocker_silhouette_edge` and multiply `chain_weight` by `reflection_segment_attenuation(...)`.
- [ ] Accumulate a diffraction residual through the silhouette edge using the existing R→D path infrastructure.
- [ ] Drive the diffraction residual through the existing R-prefix–to-edge diffraction state builder rather than writing a new diffraction field. The candidate edge set is the singleton `{silhouette_edge_idx}`, so the existing `bruteforce` candidate backend is sufficient; `rayd_edge_bvh` is unnecessary at this stage since the edge is already known.
- [x] Replace the unconditional `scene.segment_visible(...)` calls inside the F-weight finalization with mode-gated calls: in `hard` segment mode keep binary, in `f_weight` segment mode use the `F`-attenuated specular weight and skip the binary visibility kill while keeping `valid_prim` and EPC denominator checks.
- [x] Add first-landing reference tests in `tests/deterministic/test_reflection_secondary_visibility_reference.py`:
  - [x] Clear LOS matches primary-only F-weight output with no secondary attenuation.
  - [x] Static blocker near the silhouette keeps a nonzero attenuated specular contribution in `f_weight` while `hard` stays binary.
  - [x] `hard + reflection_secondary_visibility_mode="f_weight"` works independently of the primary transition mode.
  - [x] `f_weight_native + reflection_secondary_visibility_mode="f_weight"` raises instead of falling back.
  - [ ] Moving receiver sweep across the shadow boundary -> power continuity across `gamma = 0`, with hard mode showing a step.
  - [ ] FD vs AD `d|E|/d(blocker_y)` at `gamma = 0.5 lambda`: SSIM >= 0.99, PSNR >= 30 dB.

#### Stage C: R→D diffraction residual wiring

- [ ] Audit `witwin/channel/deterministic/diffraction/builders.py` and `forward.py` to expose a narrow entry point that accepts a single Tx, a single intermediate `p_ref`, a known edge index, and a target `p_rx`, and returns the UTD diffraction field through that edge. The R→D family already supports this composition; the goal is to route through it without reimplementing the diffraction coefficient.
- [ ] Ensure the diffraction residual respects the same `target_adjacent_faces` / surface-group ignore rules that the reflection chain uses, so the residual diffraction does not double-count the blocker face it sourced.
- [ ] Cover the wiring with a wedge-edge test that compares the reference Eq. 9 evaluation (`w(γ) E_spec + E_diffract`) against a manual `f_utd` + UTD coefficient computation across a sweep of `γ` values.

#### Stage D: Native CUDA path

- [ ] Extend the `_pack_f_weight_transition_chunk_arrays` Python packer in [`witwin/channel/deterministic/kernels/reflection/native_impl.py`](../../witwin/channel/deterministic/kernels/reflection/native_impl.py#L448) to produce per-pair / per-segment secondary visibility descriptors: `gamma`, `effective_L`, `silhouette_edge_pos`, `silhouette_edge_dir`, blocker material data, and a `secondary_support_valid` mask.
- [ ] Extend `reflection_accumulate.cu`:
  - Inside the F-weight kernel, before the existing primary `chain_weight` product, multiply in `safe_segment_transition_weight(γ, L_a, k)` per chain segment when `secondary_support_valid[base]` is set.
  - Add a parallel diffraction residual accumulation: for valid `secondary_support_valid[base]`, evaluate the R→D field through the silhouette edge in-place (reuse the existing CUDA-side UTD diffraction coefficient utilities if exposed; otherwise stage this against a Python-side scatter for the first landing, gated by a `secondary_visibility_native` boolean).
- [ ] Extend `reflection_jvp.cu` with forward-mode JVP through `γ` and `L_a`. `γ` differentiates through standard geometry (skew-line distance against vertex perturbations); reuse the `safe_normalize_jvp` / `f3_cross_jvp` primitives already in the JVP TU.
- [ ] Add native vs reference parity tests at one and two reflection orders with one and two blockers, asserting `rtol=1e-5` on path gain.
- [ ] Extend `tests/support/bin/benchmark_reflection_f_weight_transition.py` to add a `--secondary` flag that runs a three-cube scene with `reflection_secondary_visibility_mode="f_weight"`. Gate native overhead vs primary-only F-weight at ≤ 20% on the sparse-blocker workload.

#### Stage E: Interaction with matched-ISB and shadow-boundary postprocessing

- [x] Match the existing primary-mode guard: when `reflection_secondary_visibility_mode != "hard"`, raise on enabling matched-ISB shadow-boundary correction on the same reflection pass.
- [x] Document the precedence in `docs/dev/standards/40-diffraction-path-taxonomy.md`: matched-ISB / `shadow_completion_weight_*` are post-processing on the deterministic diffraction component; secondary visibility F-weight is an in-line modification of the reflection component. They must not be simultaneously active on the same reflection-prefix family.
- [x] Mark `wave_math.shadow_completion_weight_*` as legacy in its module docstring and route new shadow-attenuation needs through `reflection_segment_attenuation`.

### Acceptance for the secondary visibility step

- [ ] Static reflector + moving blocker FD parity: SSIM ≥ 0.99 / PSNR ≥ 30 dB on `∂|E|² / ∂(blocker_position)` at `γ = 0.5 λ`, `γ = 1.0 λ`, and `γ = 2.0 λ`.
- [ ] Reference vs native CUDA parity: `rtol = 1e-5`, `atol = 1e-12` on three-cube path gain with two blockers at one and two reflection orders.
- [ ] Sparse-blocker overhead: `reflection_secondary_visibility_mode = "f_weight"` ≤ 1.20× the primary-only F-weight runtime when fewer than 5% of reflection segments hit a blocker.
- [ ] Double-slit Fig. 8 reproduction: with both primary and secondary F-weight enabled, the reflection-only field across the two parallel strips matches the FD ground truth at SSIM ≥ 0.99 / PSNR ≥ 35 dB, closing the residual gap left after primary-only F-weight.
- [x] Update `FEATURE_LIST.md` when `reflection_secondary_visibility_mode = "f_weight"` becomes a user-visible mode.
- [ ] Tick the Stage 6 third bullet only after all sub-acceptance items pass.

### Risks and open questions

1. **Multiple blockers per segment.** Stage A returns one silhouette edge per segment. When two blockers occlude a segment, the nearest-edge query picks one and the contribution from the other is dropped. Production default keeps a one-blocker-per-segment cap, matching the per-slot one-edge primary cap; widening to two is a validation/diagnostic configuration only.
2. **Silhouette edge identity.** `scene.nearest_edge` returns the closest edge to the probe ray, which may be an interior edge of the blocker rather than a true silhouette edge if the blocker is non-convex. Filtering by "edge is on the convex hull of the blocker from the segment's vantage" is a refinement; the first landing accepts the nearest-edge approximation and documents the limitation.
3. **Diffraction-component double counting.** The diffraction component already emits UTD fields through scene edges as a standalone solver output. Secondary visibility re-injects a diffraction term inside the reflection component. Stage E's guard against simultaneous matched-ISB and the precedence note in the diffraction taxonomy are the mitigations; FD parity testing on a wedge-blocker scene that has both standalone diffraction and secondary visibility active is the verification.
4. **Differentiability of the silhouette edge selection.** `nearest_edge` is a discrete BVH query whose result can change with a small geometry perturbation. The continuity argument is that as long as the selected edge changes between two candidates that both yield `γ` near the same value, `F(k γ² / L)` is continuous through the swap. The acceptance test at `γ = 0.5 λ`, `1.0 λ`, `2.0 λ` exercises this in practice; pathological cases where the swap happens at large `γ` are out of scope.

## Superseded 2026-05-19 Modification Sketch

The original sketch below is retained only as research context. Do not execute it directly: it used stale file paths, assumed raw per-triangle edge ownership, and allowed branch fan-out that is too expensive for production radiomap workloads.

### Previous Modification Plan (Do Not Execute)

Each stage below should land as its own change with tests. Later stages depend on earlier ones.

### Stage 1: Scene-side adjacent-face indexing for reflection slots

- [ ] Audit the existing edge support arrays consumed by `DiffractionAD.edge_geometry` (`edge_v0`, `edge_v1`, `face0_third`, `face1_third`, `face0_prim`, `face1_prim`) and confirm that for every reflection-candidate primitive in the scene, each of its three edges maps to either (a) a single adjacent primitive sharing that edge, or (b) a boundary marker.
- [ ] Add a per-triangle adjacency index `tri_adjacent_prim[prim_idx, edge_local_id] -> Int32` to the scene triangle runtime data (`scene._triangle_runtime()`), backed by a CPU-side build pass over the existing edge table. A `-1` value indicates a boundary edge.
- [ ] Add accessors that, given a `prim_idx` and a slot-local edge index `i ∈ {0, 1, 2}`, return the adjacent primitive's plane point and plane normal (broadcast-compatible with the existing `Descriptor` slot data layout).
- [ ] Cover the new adjacency build with a small unit test on a simple two-triangle wedge mesh (`tests/channel_scene/test_triangle_adjacency.py`).

### Stage 2: `Descriptor` extension for adjacent-face data

- [ ] Extend [`epc.py:170 Descriptor`](../../witwin/channel/deterministic/path/reflection_impl/epc.py#L170) with the per-slot adjacent-face plane fields:
  - `slot_adj_plane_point: Point3f` with stride `3 * chain_depth * n_paths`
  - `slot_adj_plane_normal: Vector3f` with the same stride
  - `slot_adj_edge_v0: Point3f`, `slot_adj_edge_v1: Point3f` for the edge endpoints used to define `α`
  - `slot_adj_valid: Float` (or `Bool` packed to Float) for boundary edges where the adjacent face is "free space"
- [ ] Update [`epc.py:203 build_descriptor`](../../witwin/channel/deterministic/path/reflection_impl/epc.py#L203) to gather the adjacent-face data for the three edges of each slot's primary primitive, using the Stage-1 adjacency index. Preserve the existing material gather logic for the primary face; gather material parameters for adjacent faces using `resolve_surface_material` (or the boundary marker, in which case `slot_adj_*` entries are populated with sentinel values that the math layer maps to zero contribution).
- [ ] Update [`epc.py:117 native_epc_eligible`](../../witwin/channel/deterministic/path/reflection_impl/epc.py#L117) so that gradient routing remains correct when adjacent-face data is part of the AD input set.

### Stage 3: Reference math layer — `chain_math_reference` rewrite

This is the central code change. The reverse-time fold currently constructs one `current_source` per slot; the new version constructs a small fixed-size set of per-slot branch states, each weighted by an F factor.

- [ ] Define a per-slot `BranchState` carrying `current_source: Point3f`, `accumulated_weight: Complex2f`, and `accumulated_polarization_vector: dict[axis, Complex2f]`.
- [ ] At each slot, iterate over the three edges of the primary primitive:
  - Compute the signed distance `α_i` from `hit_p` to edge `i` (using barycentric-derived perpendicular distance to the edge in the primary plane).
  - Compute the UTD effective distance `L = s s' / (s + s')` and the angular squared deviation `a_a = 2 cos²(β_a / 2)`, `a_b = 2 cos²((π − β_a) / 2)` where `β_a` is the angle between the reflection direction and the edge-to-reflection-point in-plane direction.
  - Compute `x_a_i = wave_k * L * a_a_i`, `x_b_i = wave_k * L * a_b_i`.
  - Multiply the running `accumulated_weight` of the primary branch by `f_utd(x_a_i)`.
  - If the adjacent face is valid, spawn (or update) the branch corresponding to edge `i`: re-fold `current_source` using the adjacent plane, weight by `f_utd(x_b_i)`.
- [ ] Bound branch fan-out by **pruning weights below a configurable threshold** (default `|F(x)| < 1e-3`). For typical interior reflections this collapses to the single primary branch with weight ≈ 1, matching the current behavior up to a tolerance. Only near edges do additional branches survive.
- [ ] Replace the existing scalar `geom_valid` boolean output of `chain_math_reference` with the complex-valued `chain_weight`. Keep the existing geometric sanity checks (`|denom| > EPS`, `t_hit ∈ (EPS, 1−EPS)`) but apply them as multiplicative `Float(0/1)` rather than as the final visibility decision.
- [ ] Provide a unit test on a manually constructed wedge that compares `chain_math_reference` output against analytical UTD `F(x_a) E_a + F(x_b) E_b` evaluation for a representative set of `p_ref` positions sweeping across the edge.

### Stage 4: Remove the binary visibility kill in `finalize_native_outputs`

- [ ] In [`epc.py:632 finalize_native_outputs`](../../witwin/channel/deterministic/path/reflection_impl/epc.py#L632), remove the `surface_hit = surface_contains(...)` line and the subsequent `valid = valid & valid_prim & surface_hit` for the reflection path. Keep `valid_prim` (it guards against `prim_idx < 0`) and keep the `scene.segment_visible(...)` checks for inter-segment occlusion (those remain handled by the diffraction-based secondary visibility logic when Stage 5 lands).
- [ ] Replace `vector_select(valid, chain_vector, vector_zero(width))` with the new `chain_vector_weighted = vector_scale(chain_vector, chain_weight)` where `chain_weight` is the Stage-3 complex weight.
- [ ] Keep a configurable hard-cutoff fallback (`reflection_validity_hard_cutoff: bool`) for benchmarking against the old behavior. Default to the new F-weighted path; the hard-cutoff branch is for parity tests only.

### Stage 5: Secondary visibility — switch from smoothstep to `F(k L_a)`

- [ ] In [`wave_math.py:200 shadow_completion_weight_from_distance`](../../witwin/channel/core/wave_math.py#L200), introduce a UTD-based alternative `shadow_completion_weight_utd(distance, wedge_n, wave_k, effective_L)` that returns `abs(f_utd(wave_k * effective_L * a_shadow))` for a properly computed `a_shadow`.
- [ ] Wire the new path through [`shadow_boundary_policy.py`](../../witwin/channel/core/shadow_boundary_policy.py) as an opt-in policy. Switch the default once Stage 4 verifications pass.

### Stage 6: Native CUDA forward kernel update

- [ ] Update `witwin/channel/deterministic/kernels/reflection/reflection_accumulate.cu` and `reflection_jvp.cu` to mirror the Stage-3 reference math: gather adjacent-face data per slot, apply F weighting per edge, return the complex `chain_weight` alongside the existing outputs.
- [ ] Update the CUDA binding layer in `witwin/channel/deterministic/kernels/reflection/bind.cpp` for the extended buffer signature.
- [ ] Update the native EPC variant in `witwin/channel/deterministic/path/reflection_impl/epc.py:493 launch_native_forward` to pass the new buffers.
- [ ] Provide a DrJit-reference vs native parity test (`tests/deterministic/test_reflection_epc_f_weight_parity.py`) over a multi-cube scene at multiple reflection orders. Tolerance: relative `1e-5` on chain vector magnitude, `1e-4` on phase.

### Stage 7: Solver-level integration

- [ ] Confirm [`witwin/channel/deterministic/path/reflection.py:13 trace`](../../witwin/channel/deterministic/path/reflection_impl/__init__.py) passes the new `Descriptor` fields through `compute_field` correctly.
- [ ] Update the Monte Carlo BDPT reflection AD path (`witwin/channel/montecarlo/trace/reflection.py`, `witwin/channel/montecarlo/integrators/bdpt_ad.py`) only if numerical FD parity testing shows the Monte Carlo flavor has the same Heaviside problem near edges; otherwise defer to a separate plan.

## Forward Correctness Implications

For the revised rollout, forward correctness is evaluated per mode:

- `hard` must remain identical to the current implementation.
- `f_weight_reference` and `f_weight_native` may differ from `hard` only inside the configured surface-boundary transition radius.
- In production settings with `reflection_f_weight_max_edges_per_slot=1`, the accepted approximation is nearest-boundary-only. Multi-edge products are reserved for small validation scenes until profiling proves they are affordable.

In the limit `p_ref` well inside `△_a`, all `f_utd(x_a_i) → 1` and all `f_utd(x_b_i) → 0`. The Stage-3 weight collapses to `chain_weight ≈ 1.0 + 0j`, and the reflection contribution matches the pre-change forward. Stage-1 candidate discovery is unchanged, so the set of paths under consideration is the same as before; only the per-path weight differs.

In the near-edge transition band (the band width is roughly the Fresnel zone, `O(√(L λ))`):

| `p_ref` location | Pre-change forward | Post-change forward | Physical interpretation |
| --- | --- | --- | --- |
| Deep inside `△_a` | `E_a` | `E_a` (F → 1, F_b → 0) | Same |
| On the edge `E` | `E_a` or `0` (mask-dependent) | `F(0) E_a + F(0) E_b ≈ 0`, with diffraction term carrying the field | Paper-correct: reflection has no canonical value on the edge; UTD diffraction provides the field |
| Just past `E` into `△_b` | `0` (current code drops it) | `E_b` (F_a → 0, F_b → 1) | Paper-correct: this is a legitimate reflection from `△_b` that the Stage-1 discovery may have associated with `△_a` |

The far-field power summed over the receiver grid is energy-conserving by the asymptotic properties of `F` (proof in paper Appendix D.3). Coverage maps in interior regions should change only at the noise floor; coverage in the near-edge transition band should become continuous instead of step-discontinuous.

Verification target: the canonical double-slit interference scene (paper Fig. 8) should reach SSIM ≥ 0.99 / PSNR ≥ 35 dB against the finite-difference ground truth after the change, up from the current ≈ 0.85 / 17 dB baseline behavior of the binary-mask formulation.

## Gradient Implications

Pre-change: `∂E_reflection / ∂v` contains a Heaviside `H(α)` step whose derivative is a Dirac `δ`. DrJit AD effectively returns `0` everywhere except at the exact boundary, where it is ill-defined. Gradients with respect to vertex positions, Tx position, or object pose vanish exactly when the optimization needs them most (when `p_ref` is near an edge that the optimizer wants to cross).

Post-change: `dE_reflection / dv = d[F(x_a) E_a + F(x_b) E_b] / dv` in F-weight modes. The implementation must use safe small-`x` handling for the transition function before relying on gradients at the boundary. Crossing an edge becomes a smooth optimization move only if side gating and the adjacent-surface residual are implemented consistently with the reference tests.

The paper validates this with three scene-parameter classes (object position, object rotation, Tx position) achieving SSIM > 0.99 against finite differences. Reproduction of those numbers on the existing double-slit verification harness is the acceptance criterion for this plan's gradient correctness.

## Verification Plan

### Forward parity

- [ ] Run the existing deterministic radiomap regression on `examples/deterministic_radiomap_single_cube.py` and confirm interior-region power maps change only within numerical noise.
- [ ] Run `examples/deterministic_radiomap_three_cubes.py` and confirm near-edge transition bands become smooth in the component plots without altering integrated power per surface.

### Gradient correctness

- [x] Build a controlled double-slit test scene (single transmitter, two parallel reflecting strips, single receiver line) as a new test fixture under `tests/deterministic/test_reflection_f_weight_gradient.py`.
- [ ] For a swept scene parameter `θ` (Tx vertical position) crossing the strip edge:
  - Compute `∂|E|² / ∂θ` via DrJit forward-mode AD.
  - Compute the same gradient via centered finite differences with `ε = λ / 1000`.
  - Assert SSIM ≥ 0.99 and PSNR ≥ 30 dB between the two gradient fields.
- [x] Repeat for `θ` representing object position, object rotation, and Tx position.

### Convergence smoke test

- [x] Implement a minimal inverse problem: given the field measurements from a fixed-geometry scene, recover Tx position from a `1 λ` perturbed initial guess using vanilla `dr.opt.Adam` over the field-magnitude MSE. Verify convergence within 200 iterations with `reflection_transition_mode="f_weight_reference"` or `"f_weight_native"`; compare against `reflection_transition_mode="hard"` as the baseline.

## Risks and Open Questions

Current revised-plan risks:

1. **Side-gated F-weight definition.** The plan must explicitly define primary-side and adjacent-side gating before implementation. A distance-squared-only `x` is insufficient because it cannot distinguish the lit side from the shadow side.

2. **Surface-group boundary ownership.** Reflection validity currently uses surface groups, so the implementation must operate on surface-boundary edges. Raw triangle-local edges are only implementation detail.

3. **Native/reference skew.** `hard` mode can keep the current ABI, but `f_weight_native` must update EPC outputs and accumulation together. A mixed state would produce incorrect forward fields.

4. **Performance cap.** Production mode defaults to one nearest boundary edge per slot and must keep an interior fast path. Wider edge caps are validation/diagnostic settings until profiling proves otherwise.

5. **Post-processing interaction.** Matched-ISB and secondary shadow-boundary smoothing can double-count transition effects. They must stay independently switchable while reflection F-weighting is experimental.

Historical risks from the superseded 2026-05-19 sketch are below for traceability.

1. **Per-slot multi-edge handling.** Paper Eq. (11) is a product over `m` path interactions with one `F` per interaction. A triangle has three edges; the cleanest mapping is to take the product over the three F factors per slot, but this needs verification against the paper's source code release and against energy conservation on a single-triangle wedge test. The Stage-3 plan defaults to the product-over-three-edges interpretation; the alternative is taking the F of the nearest edge only.

2. **Branch fan-out cost.** Naive enumeration produces `4^m` branches per `m`-bounce path. The Stage-3 pruning by weight magnitude is essential. For typical scenes (most reflection points well inside their triangles), real fan-out should average close to `1.x^m` rather than `4^m`. Profiling on the Munich scene with `max_reflections=3` is part of Stage 6 acceptance.

3. **Boundary edges (no adjacent face).** When the adjacent face is "free space", `E_b = 0` but `F(x_b)` must still be computed correctly so the primary face's contribution decays smoothly to zero into the empty half-space. Stage 1 introduces the boundary marker; Stage 3 must handle it without introducing a sign discontinuity in `a_b`.

4. **Interaction with existing shadow-boundary post-processing.** The `matched-ISB completion` pipeline currently corrects for some of the discontinuity downstream. After Stage 4 lands, that correction may double-count near edges. Stage 5 is the integrated fix, but staged rollout means an interim state where the matched-ISB layer needs a `disable_when_f_weight_active` flag.

5. **Native kernel rebuild lock.** Stage 6 requires rebuilding the relevant extension under `witwin/channel/_native/` (for this plan, `_deterministic_radiomap_native.pyd`). On Windows, this requires closing all Python and Jupyter sessions that have loaded the extension. The CMake incremental build path described in `CLAUDE.md` is the supported workflow.

6. **Out of scope here.** The signal-domain Dirichlet PSF surrogate (paper §4) and the Adam-plus-Laplacian DT optimization loop (paper §5.2) are not in this plan. They become useful only after this plan lands, because they assume the gradient pipeline already produces physically consistent values.

## Acceptance Checklist

- [x] Default `hard` mode remains behaviorally identical to current deterministic reflection output.
- [x] `reflection_transition_mode` is recorded in deterministic solver metadata.
- [x] `f_weight_reference` passes single-boundary continuity and finite-gradient tests.
- [x] `f_weight_native` preserves the interior fast path and stays within the accepted benchmark overhead when near-boundary pairs are sparse.

- [x] Double-slit gradient parity vs FD: SSIM ≥ 0.99 / PSNR ≥ 30 dB on object position, object rotation, Tx position.
- [x] Interior-region forward parity vs current code: no regression beyond `1e-5` relative on `examples/deterministic_radiomap_three_cubes.py`.
- [x] Convergence smoke test passes (Tx position recovery within 200 iterations).
- [x] DrJit-reference vs native CUDA parity: relative `1e-5` magnitude / `1e-4` phase across three reflection orders.
- [x] `FEATURE_LIST.md` updated when a non-default user-visible reflection F-weighting mode lands.
- [ ] Plan file moved from `docs/dev/plans/` to `docs/dev/archive/completed/` on landing the final acceptance criterion, keeping the `25-` prefix.
