# Channel Readability and Conciseness Follow-Up Plan

Status: Active  
Category: Plan  
Last reviewed: 2026-05-17  
Owners: channel team  

Related:

- `optimization/channel-readability-conciseness-audit-2026-05-16.md` — baseline audit (pre–MidMay-Refactor)
- `plans/14-compact-call-formatting-plan.md` — vertical kwargs / `ruff format` line-compaction (orthogonal)
- `plans/21-diffraction-shared-extraction-plan.md` — diffraction shared extraction (partially landed)
- `plans/10-scene-owned-endpoints-and-material-config-pruning.md` — endpoint ownership migration

## Purpose

Record what the **MidMay-Refactor** cleanup already achieved, what remains worth doing for readability and line/file reduction, and a phased execution order. This plan is the actionable follow-up to the 2026-05-16 conciseness audit after the first cleanup pass.

## Baseline vs current (`witwin/`)

Measured on the MidMay-Refactor branch after the diffraction/MC/det cleanup commits (plus unstaged `_runtime` → `arrays` migration when noted).

| Metric | Audit (2026-05-16) | After cleanup | Delta |
|--------|-------------------|---------------|-------|
| `.py` files | ~121 | **111** | −10 |
| Python lines | ~34,663 | **~30,908** | **−~3,755 (~11%)** |
| Files > 500 lines | 25 | **18** | −7 |
| `deterministic/path/diffraction_impl/math.py` | 1050 | **443** | −607 |
| `montecarlo/config.py` | 865 | **384** | −481 |

### Already landed (do not re-do)

- **`channel_utils/diffraction_geometry.py`** — shared wedge geometry (`wedge_exterior_mask`, `wedge_geometry`, pole masks, etc.).
- **Removed** `montecarlo/path/diffraction_support.py`, `channel_utils/scene_query.py`.
- **Unified results** — deleted thin `deterministic/result.py` and `montecarlo/result.py`; public type is `channel_utils.radiomap_result.RadioMapResult`.
- **Removed** `montecarlo/materials.py`, `montecarlo/_tensors.py` (logic in `channel_utils`).
- **MC path cleanup** — deleted `reflection_ad.py`, `tapes.py`; AD/tape paths folded into `montecarlo/path/reflection.py`.
- **Renamed** shadow-boundary modules to `postprocessing.py` (det + mc).
- **`grid.py` → `grid_ops.py`** per solver on top of `channel_utils.grid`.
- **`gather_structure_indices` delegates** removed from geometry helpers; callers use `Scene.gather_structure_indices` directly.
- **`deterministic/_runtime.py` deletion (in progress)** — `repeat_*` → `channel_utils.arrays`; `vector_power` / scatter helpers → `channel_utils.polarization` or inline at call sites; `torch_lexsort` localized to `deterministic/kernels/pruning_sort/drjit_impl.py`.
- **Unified native extension loader (P0.2)** — `witwin/channel/_native/loader.py` owns the shared `NativeExtensionLoader` class with `NativeExtensionSpec`; `witwin/channel/_native/{channel_utils,deterministic,montecarlo}.py` declare only extension specs (~25 lines vs ~175–210). The deterministic kernel call sites migrated from module-level `_extension`/`require_functions`/`has_functions` to `NativeExtension.load()/require_functions()/has_functions()`. Dead `cuda_runtime_version`/`run_cuda_noop`/`sample_add_one` forwarders dropped. `witwin.channel.deterministic` now also exports `NativeExtension` for parity with `witwin.channel.montecarlo`.
- **Unified Point3f / endpoint coercion (P0.3)** — `channel_utils/runtime.to_point3f` accepts `wt.Point3f`, duck-typed `.x/.y/.z`, length-3 sequences, and torch tensors of shape `(3,)`/`(N, 3)`; `to_vector3f` keeps the simple 3-vec coercion for polarization. `runtime._to_vec3`, `montecarlo/solver.py` inline check, and `path/solver._to_point3f` are gone — all three sites call the shared helper.
- **Wedge exterior mask dedupe (P0.5)** — `deterministic/kernels/radio_map_accumulate/native_impl.py` no longer defines `_reference_shadow_boundary_wedge_exterior_mask`; the four call sites now use `channel_utils.diffraction_geometry.wedge_exterior_mask` directly.
- **Shadow-boundary backend policy (P0.4, partial)** — `channel_utils/shadow_boundary_policy.py` owns `ShadowBoundaryBackendPolicy` with `resolve(...)` + `validate_small_workload(...)`. Det `resolve_shadow_boundary_statistics_backend` / `validate_dense_shadow_boundary_workload` and mc `ShadowBoundary._resolve_backend` now delegate to per-side policy instances. Net file deltas were small (−12/−20) because the metadata schemas remain genuinely solver-specific (det matched-ISB stats vs mc tile-shaped power smoothing), but backend resolution is no longer duplicated.
- **`rotate_vector_around_axis` shared (P1.1, partial)** — pure Rodrigues rotation moved from `deterministic/path/diffraction_impl/math.py::GeometrySupport` to `channel_utils.diffraction_geometry.rotate_vector_around_axis`. Sole caller (`GeometrySupport.first_order_diffraction_parameter`) updated. The bigger `GeometrySupport`/`channel_utils.polarization` overlap (Fresnel diagonal operators) was not collapsed: the in-file versions use inline isfinite-guards vs. the utility's `sanitize_complex`, and converging them needs a parity check on AD-sensitive paths.
- **Geo composition flattened (P1.2)** — `class Geo(GeometryDomain, GeometrySupport): pass` replaced with `class Geo(GeometrySupport)` containing the three ex-`GeometryDomain` methods inline; the empty `GeometryDomain` class is gone. All `Geo.X(...)` call sites unchanged.
- **`path_export.py` result-assembly split (P1 megamodule)** — F+G result-assembly group (torch conversion, summarize / select / chunk replay, `assemble_result_payload`) moved into sister `witwin/channel/deterministic/path/path_export_assembly.py`. `path_export.py` shrinks 1872 → 1126 (−746). The new file imports the two materializers and three payload-kind constants from `path_export`; `witwin/channel/path/solver.py` is the only external consumer and was updated to import `assemble_result_payload` from the new module.

## Remaining work (prioritized)

### P0 — High ROI, low algorithm risk

(All P0 items now landed; see "Already landed" above.)

### P1 — Megamodules and structure (readability first)

Split by **phase** (configure → trace → accumulate → package), not by copying det/mc.

| Lines | File | Split suggestion |
|------:|------|------------------|
| 2003 | `montecarlo/integrators/bdpt_diffraction.py` | sampling / state / accumulate / AD hooks |
| ~~1711~~ 1126 | `deterministic/path/path_export.py` | result-assembly half (~750 lines) extracted to sister `path_export_assembly.py`; remaining file is schema + collectors + materializers |
| 1036 | `montecarlo/path/reflection.py` | primal transport vs AD/tape (post–reflection_ad merge) |
| 1012 | `montecarlo/integrators/basic.py` | remove `zero = lambda` (line ~81); extract finalize + shadow call |
| 951 | `montecarlo/path/diffraction.py` | replace `gather = lambda` with named helpers |
| 944 | `channel_scene/scene.py` | edge API vs rayd runtime vs material binding |
| 753 | `deterministic/path/reflection_impl/epc.py` | touch only obvious dead code |

#### P1.1 Slim `math.py` further (continuing)

- **Already moved**: `rotate_vector_around_axis` → `channel_utils.diffraction_geometry`.
- **Still candidates**: `GeometrySupport.face_operator` / `surface_operator` / `diagonal_face_operator` overlap `channel_utils.polarization.fresnel_diagonal_operator` (different sanitize strategy). `source_field` / `source_field_normal_derivative` are pure FSPL+phase + normal derivative; 9 call sites across det modules would need to migrate to a `channel_utils.wave_math` location.
- **Note**: Keep det-specific `wt` aliases at call sites.

### P2 — Lower ROI (optional)

| Item | Action |
|------|--------|
| 31 files under 50 lines | Mostly kernel `__init__.py` barrels; merge only if import ergonomics improve |
| `get_coordinates` in `grid_ops.py` and `field.py` | Collapse to `@property` or shared grid adapter |
| `scene.get_edge_data` / `get_triangle_surface_edge_candidates` | Rename away from `get_` per house style |
| `basic_ad.py`, `diffraction_ad.py` | Keep separate; share dict factories from primal modules only |
| Broad `except Exception` (~20 sites) | Tighten in `postprocessing._array_grad_enabled`, `path_export._torch_tensor`, native probes |

## Explicit non-goals

- **Do not touch `reference/`** — keep the tree as-is (no archive, delete, move, or “cleanup” refactors under `reference/channel/`). Conciseness work applies to `witwin/` only.
- **Do not merge** `deterministic/path/reflection.py` and `montecarlo/path/reflection.py` (different algorithms).
- **Do not shrink** large `native_impl.py` / `.cu` bodies without a kernel-level design.
- **Do not remove** `assert_scene_materials_complete` or other domain invariants.
- **Do not break** divergent diffraction state schemas (`State` / `SK_*` vs `DiffractionStates`) per `21-diffraction-shared-extraction-plan.md`.

## Suggested execution order

1. P1 megamodule splits (continuing) — `bdpt_diffraction.py` next (sampling / state / accumulate / AD hooks). Then `montecarlo/integrators/basic.py`, `montecarlo/path/diffraction.py`.
2. P1.1 continuation — second pass on `GeometrySupport` overlaps with `channel_utils.polarization` / `wave_math` after a parity test pass.
3. P2 optional — small `__init__.py` barrels, `get_coordinates` dedupe, narrow `except Exception` sites (all under `witwin/` only).

## Acceptance criteria

- `witwin/` Python line count trends down or stays flat while file count may drop from barrel merges.
- No new parallel copies of wedge geometry or native DLL setup.
- Targeted pytest green: diffraction, path solver, det + mc radiomap smoke, shadow-boundary backend tests.
- `FEATURE_LIST.md` unchanged unless public API surface moves (e.g. exported `to_point3f`).

## Tracking

Update this plan when a P0/P1 item completes; move to `archive/completed/` when all P0 items and at least two P1 megamodule splits are done or explicitly deferred.
