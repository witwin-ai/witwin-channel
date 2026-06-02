# Trace Module Code Audit

Date: 2026-03-29
Re-checked: 2026-03-29 (current code on branch `fixing_materials`)

Scope: `witwin/channel/trace/` — architecture, cleanliness, performance (CPU/GPU copies), design patterns, naming, unnecessary fallbacks, over-defensive programming.

---

## Status Legend

- **FIXED** — issue no longer present in the current code.
- **PARTIAL** — partially addressed; improvement visible but residual issue remains.
- **OPEN** — issue still present as originally described.

---

## 1. Architecture Elegance

### P1: `compute_diffraction_field` and `compute_diffraction_order_breakdown` duplicate metadata construction — OPEN
- `api.py` still has two parallel metadata-building blocks (~40 lines each) in both public functions.
- `compute_diffraction_field` still builds an empty `path_budget_report` at L372, then overwrites at L422.

### P2: Suffix DDA duplication — FIXED
- A shared `_dda_cell_contribute_and_scatter()` helper now exists at `suffix.py:61-122`.
- Both `_dda_segment_loop_body_batched` (L125) and `_run_suffix_dda_symbolic` (L176) call it.
- The ~90-line duplication is eliminated.

### P3: `_diagonal_face_operator` defined twice — FIXED
- `state/arrays.py:8` now imports `from ..geometry import _diagonal_face_operator`.
- Only one definition remains in `geometry/fields.py:250`.

### P4: `_edge_face_reflection_operators` and `_pair_face_material_operators` overlap — OPEN
- `fields.py` still has `_edge_face_reflection_operators`.
- `field.py` still has `_pair_face_material_operators` which calls `_edge_face_reflection_operators` as fallback.
- Two entry points for the same conceptual operation remain.

---

## 2. Code Cleanliness

### P5: `common.py` is a pure re-export shim — OPEN
- `common.py` is unchanged: `from .constants import *` + `from .array_ops import *`.
- However, it now also re-exports the new `SK_*` constants and `_validate_state_arrays`, making it a more substantial aggregation point. Borderline acceptable.

### P6: Duplicate import in `field.py` — FIXED
- `field.py` now has only one `from ...config import coerce_diffraction_execution` at line 6. The duplicate is gone.

### P7: Wildcard import chain in `diffraction/__init__.py` — OPEN
- Still uses `from .api import *`, `from .field import *`, etc.
- No change.

### P8: `_resolve_incident_diffraction_state` three branch paths — FIXED
- `field.py:133-176` now assumes canonicalized Jones+basis transport state only.
- The function docstring explicitly points to `_canonicalize_transport_state` in `_make_state_arrays`.
- The old multi-format branch handling is gone.

### P9: Functions with excessive parameters (12+) — OPEN
- No change; `compute_diffraction_field` still takes 23 parameters, etc.

---

## 3. Performance (CPU/GPU Copies)

### P10: `state/arrays.py` unconditional `import torch` — PARTIAL (consistent now)
- `tracer.py` now also has `import torch` at module top level (line 5), removing the old `try/except` guard.
- Both files are now consistent. The inconsistency is resolved, though torch remains a top-level import in a DrJit-focused module.

### P11: `preload_diffraction_edges` Python per-edge loop — OPEN
- `constants.py:33-122` still uses a Python for-loop to collect edges. No batch GPU path.

### P12: `_concat_arrays` scatter implementation — OPEN
- `array_ops.py:7-21` unchanged. Still per-segment `dr.scatter`.

### P13: Suffix DDA O(n_states x n_rx) memory allocation — FIXED
- `suffix.py:292-299` now allocates result arrays of size `n_rx` (not `n_states * n_rx`).
- The `state_idx`-based flat indexing and `_reduce_state_flat_field`/`_reduce_state_flat_vector` calls are removed.
- DDA scatter now writes directly to `cell_idx` (n_rx-sized arrays). The `n_states * n_rx` explosion is gone.
- `result_count` is also removed — contributions accumulate directly.

### P14: Redundant gather in `_accumulate_edge_states_to_receivers` — OPEN
- `field.py:355-369` still gathers `state_edge_pos`, `state_adjacent_face0`, `state_adjacent_face1`, and `batch_rx_all` for visibility.
- After `dr.compress(visible)`, lines 376-381 re-gather `batch_states` and `batch_rx` from the compressed pair indices.
- The first set of gathered data is still discarded after compress.

---

## 4. Design Pattern Issues

### P15: dict as core state data structure (no type safety) — PARTIAL
- New `SK_*` string constants defined in `constants.py:240-289` centralize all state key names.
- New `STATE_STATIC_KEYS` tuple and `_validate_state_arrays()` provide runtime schema validation.
- `state/arrays.py` imports and uses `SK_*` constants.
- **Still open:** The state is still a plain `dict[str, DrJitArray]`, not a typed structure. Key constants help prevent typos in new code, but existing consumers (e.g., `field.py`, `suffix.py`) still use raw string literals (`"edge_pos"`, `"source_pos"`, etc.) rather than `SK_*` constants.
- Migration of all consumers to `SK_*` constants is incomplete.

### P16: `reflection_detail` duck typing — FIXED
- `materials.py` now defines `ReflectionTraceDetail` plus `coerce_reflection_trace_detail()`.
- Reflection detail payloads carry an explicit `detail_kind="reflection_trace_detail"` tag.
- Diffraction builders, reflection EPC, and tracer metadata now consume the normalized payload instead of inspecting dict shape.

### P17: Metadata dicts manually constructed — OPEN
- `tracer.py:449-528` and `api.py:361-401` still use string literal keys.
- No centralization.

---

## 5. Naming Style Issues

### P18: Inconsistent `_` prefix (`cot` vs `_cot`) — OPEN
- Both still exist in `utd.py`.

### P19: `r0` / `rn` semantically ambiguous — PARTIAL
- The `SK_*` constants now use `SK_R0 = "r_face0"` and `SK_RN = "r_face_n"`, improving the stored key names.
- However, local variable names `r0`/`rn` are still used in function bodies throughout `fields.py`.

### P20: `nn` as normal variable name — PARTIAL
- `SK_NN = "n_face_n"` — the stored key name is improved.
- Local variable `nn` / `nn_b` still used throughout `field.py`, `angles.py`, etc.

### P21: `bk` alias — OPEN (acceptable)
- No change; consistent project convention.

---

## 6. Unnecessary Fallbacks

### P22: `_is_torch_tensor` try/except ImportError — FIXED
- `tracer.py:5` now has `import torch` at module level.
- `_is_torch_tensor` (line 119-120) is now simply `return isinstance(value, torch.Tensor)`.
- The dead `try/except` is gone.

### P23: `_reflection_material_detail` implicit fallback — FIXED
- `fields.py` no longer has `_reflection_material_detail`.
- Material override normalization is explicit via `normalized_override_material()` / `_coerce_material_override()`.
- Wrong payload types now raise `TypeError` instead of silently falling back.

### P24: `_reflection_gain` 4-level key lookup — FIXED
- `fields.py` no longer has `_reflection_gain`.
- Reflection gain is now carried explicitly by `ReflectionTraceDetail` / `ReflectionMaterialContext` and threaded through call sites as a first-class argument.

### P25: `_edge_attr` dual-dispatches dict vs object — FIXED
- `constants.py:27-28` is now a plain `getattr(edge, name, default)`.
- The old dict-vs-object compatibility path is removed.

### P26: `_resolve_trace_monitors` `hasattr` fallback — FIXED
- `tracer.py:375` now directly calls `self.scene.resolved_monitors()` without `hasattr` guard.
- The fallback `getattr(self.scene, "monitors", ())` path is removed.

---

## 7. Over-Defensive Programming

### P27: `isnan`/`isinf` silent masks — OPEN
- `utd.py` `_cot` and `cot` still mask NaN/Inf to 0.
- `material_ops.py:45-52` still masks `isfinite`.

### P28: Excessive `+ EPS` — OPEN
- No change in `angles.py`, `fields.py`, `constants.py`.

### P29: `safe_*` double safety — OPEN
- `field.py:297-301` still has the pole_safe -> safe_phi -> slope_safe double-layer pattern.

---

## 8. Other Issues

### P30: `_reduce_state_flat_field` averages then sums — FIXED (removed)
- This function no longer appears in `field.py`. The suffix DDA rework (P13 fix) eliminated the need for the flat-field reduction path entirely.

### P31: `__all__` exports `_`-prefixed functions — PARTIAL
- `field.py:453-456` now only exports `_accumulate_edge_states_to_receivers` and `_edge_state_field_to_targets` (reduced from 4).
- Other modules still export `_`-prefixed names in `__all__`.

---

## Updated Priority Matrix

| Priority | ID | Status | Category | Description |
|----------|----|--------|----------|-------------|
| **High** | P2 | **FIXED** | Cleanliness | Suffix DDA duplication eliminated via shared helper |
| **High** | P13 | **FIXED** | Performance | Suffix DDA now O(n_rx) not O(n_states x n_rx) |
| **High** | P14 | OPEN | Performance | Redundant gather in visibility check |
| **High** | P15 | PARTIAL | Design | SK_* constants + validation added, but consumers not migrated |
| **High** | P9 | OPEN | Cleanliness | Functions with 15-23 parameters |
| Medium | P1 | OPEN | Architecture | api.py metadata duplication |
| Medium | P3 | **FIXED** | Cleanliness | `_diagonal_face_operator` now single definition |
| Medium | P8 | **FIXED** | Cleanliness | Incident state now canonicalized to one format |
| Medium | P11 | OPEN | Performance | Edge preload Python per-edge loop |
| Medium | P16 | **FIXED** | Design | `reflection_detail` normalized via explicit typed payload |
| Medium | P22 | **FIXED** | Fallback | torch import guard removed |
| Medium | P26 | **FIXED** | Fallback | `hasattr` monitor fallback removed |
| Medium | P27 | OPEN | Defensive | NaN/Inf silent masking |
| Medium | P28 | OPEN | Defensive | Excessive `+ EPS` |
| Low | P4 | OPEN | Architecture | Face operator two entry points |
| Low | P5 | OPEN | Cleanliness | common.py re-export shim |
| Low | P6 | **FIXED** | Cleanliness | Duplicate import removed |
| Low | P10 | **FIXED** | Performance | torch import now consistent |
| Low | P17 | OPEN | Design | Metadata manual dict construction |
| Low | P19-20 | PARTIAL | Naming | SK_* keys improved, local vars unchanged |
| Low | P23-25 | **FIXED** | Fallback | Reflection/detail fallback chains removed; `_edge_attr` simplified |
| Low | P30 | **FIXED** | Performance | Flat-field reduction removed |
| Low | P31 | PARTIAL | Cleanliness | `__all__` reduced but still has `_` names |

---

## Summary

**13 of 31 issues FIXED**, **4 PARTIAL**, **14 OPEN**.

Fixed items are concentrated in the highest-impact areas:
- P2 (DDA duplication) and P13 (memory explosion) were the two most critical issues and both are resolved.
- P3, P6, P8, P10, P16, P22-P26, P30 are all clean fixes.

Remaining high-priority open items:
1. **P14** — redundant gather in visibility check (easy performance win).
2. **P15** — migrate all state consumers to `SK_*` constants (mechanical but large).
3. **P9** — excessive function parameters (needs config object design).
