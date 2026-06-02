# Monitor-Solver Separation Refactor Plan

## 1. Motivation

`tracer.py` is currently 1220 lines and serves as a god-object: it holds Tracer
configuration, solver-mode guardrails, FieldMonitor-specific grid assembly and
Jones metadata, PathMonitor-specific path collection and merge orchestration,
reflection discovery caching, and the top-level `trace()` dispatch loop. Over
half the file (~600 lines) is monitor-specific orchestration that does not
belong in a general-purpose tracer.

This refactor extracts monitor-specific orchestration into a new
`trace/monitors/` package, leaving the solver modules (`los`, `reflection/`,
`diffraction/`, `path/`) untouched and the Tracer as a thin
configuration + dispatch coordinator.

---

## 2. Design Principles

1. **Solver modules stay monitor-agnostic.** `los.py`, `reflection/`,
   `diffraction/`, and `path/` do not import or reference any monitor
   type. They accept generic inputs (receiver positions, wavelength, etc.) and
   return generic outputs (DrJit arrays, torch tensors).

2. **Monitor orchestration lives in `trace/monitors/`.** Each monitor type gets
   its own module that knows how to call solvers and assemble results in the
   shape the monitor requires.

3. **Tracer becomes a thin coordinator.** It owns configuration, resolves
   monitors, manages the reflection discovery cache, and delegates to
   `trace_field_monitor()` / `trace_path_monitor()`.

4. **No logic changes.** This is a pure code-movement refactor. All numerical
   behavior, metadata structure, and public API remain identical.

5. **`path/` stays separate.** It is the DrJit-to-PyTorch conversion
   layer for path-level output, not a monitor orchestrator. Merging it into
   `monitors/path/path_monitor.py` would create another 1400-line file.

---

## 3. Current State

### tracer.py Breakdown (1220 lines)

| Section | Lines | Content |
|---|---|---|
| `__init__` + config | ~60 | Frequency, wavelength, materials, polarization |
| `_resolve_solver_controls` | ~100 | Mode guardrails (accuracy / fast_approximate) |
| `_coerce_*`, `update_scene` | ~90 | Input conversion helpers |
| `create_intersection_func` | ~30 | Differentiable intersection wrapper |
| `_resolve_trace_monitors` | ~40 | Separate monitors into plane / path lists |
| `_reflection_discovery_key_*` | ~20 | Cache key computation |
| `_jones_metadata` | ~43 | FieldMonitor-only Jones basis metadata |
| `_compute_los` | ~12 | FieldMonitor LoS thin wrapper |
| `_compute_reflection` | ~34 | FieldMonitor reflection thin wrapper |
| `_compute_diffraction` | ~30 | FieldMonitor diffraction thin wrapper |
| `_build_trace_metadata` | ~79 | FieldMonitor metadata assembly |
| `_zero_diffraction_components` | ~63 | FieldMonitor zero-diffraction stub |
| `_trace_field_monitor` | ~182 | **FieldMonitor orchestration** |
| `_path_monitor_positions` | ~3 | PathMonitor helper |
| `_path_monitor_receiver_groups` | ~11 | PathMonitor z-grouping |
| `_build_path_trace_metadata` | ~52 | PathMonitor metadata assembly |
| `_trace_path_monitor` | ~187 | **PathMonitor orchestration** |
| `_estimate_state_bytes` etc. | ~40 | Performance guardrail profiling |
| `trace()` | ~40 | Top-level dispatch + cache |

**~600 lines are monitor-specific** (everything marked FieldMonitor-only or
PathMonitor-only above).

### Solver Module Sizes (unchanged by this refactor)

| Module | Lines |
|---|---|
| `los.py` | 83 |
| `materials.py` | 272 |
| `reflection/` (7 files) | 2339 |
| `diffraction/` (17 files) | 5944 |
| `path/` (7 files) | 1034 |

---

## 4. Target Structure

```
trace/
├── __init__.py                ~15    (add monitors re-export)
├── tracer.py                 ~400    (config + dispatch + cache)
├── los.py                      83    UNCHANGED
├── materials.py               272    UNCHANGED
│
├── monitors/                         ★ NEW
│   ├── __init__.py             ~10
│   ├── common.py              ~120    shared: solver controls, discovery keys
│   ├── plane.py               ~500    FieldMonitor orchestration
│   └── path.py                ~350    PathMonitor orchestration
│
├── path/             1034    UNCHANGED
│   ├── __init__.py
│   ├── angles.py
│   ├── common.py
│   ├── diffraction.py
│   ├── los.py
│   ├── merge.py
│   ├── reflection.py
│   └── types.py
│
├── reflection/               2339    UNCHANGED
│   ├── field.py
│   ├── dda.py
│   ├── accumulation.py
│   ├── epc.py
│   ├── scatter.py
│   ├── paths.py
│   └── geometry.py
│
└── diffraction/              5944    UNCHANGED
    ├── api.py
    ├── builders/
    ├── geometry/
    ├── state/
    └── ...
```

---

## 5. File-by-File Specification

### 5.1 `trace/monitors/__init__.py`

```python
from .field.trace_field import trace_field_monitor
from .path.trace_path import trace_path_monitor
from .common import resolve_solver_controls

__all__ = [
    "trace_field_monitor",
    "trace_path_monitor",
    "resolve_solver_controls",
]
```

### 5.2 `trace/monitors/common.py` (~120 lines)

Extracted from `tracer.py` as **pure functions** (no `self`):

| Function | Source | Lines |
|---|---|---|
| `resolve_solver_controls(config)` | `Tracer._resolve_solver_controls` (218-316) | ~100 |
| `reflection_discovery_key_for_field(monitor, tx_pos)` | `Tracer._reflection_discovery_key_for_field_monitor` (430-443) | ~14 |
| `reflection_discovery_key_for_path(monitor)` | `Tracer._reflection_discovery_key_for_path_monitor` (445-448) | ~4 |

`resolve_solver_controls` currently reads `self.reflection_n_rays`,
`self.solver_mode`, etc. After extraction it receives a `TraceConfig` and
returns the same `{"selected", "requested", "effective", "changes"}` dict.

### 5.3 `trace/monitors/field/trace_field.py` (~500 lines)

**Public entry point:**

```python
def trace_field_monitor(
    tx_pos: bk.Point3f,
    monitor: FieldMonitor,
    scene: Scene,
    config: TraceConfig,
    solver_controls: dict,
    *,
    reflection_detail=None,
    verbose: bool = False,
    return_timing: bool = False,
    return_diffraction_audit: bool = False,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    """
    Run LoS + reflection + diffraction for a FieldMonitor.

    Returns (payload_dict, reflection_detail). The payload dict is the
    same structure consumed by MonitorResult.from_payload().
    """
```

**Module-private functions** (moved from tracer.py, `self` replaced by
explicit parameters):

| Function | Source | Purpose |
|---|---|---|
| `_compute_los(scene, rx_positions, axis, tx_pos, config, active_rx_pol)` | tracer.py:547-558 | LoS field + polarization vector |
| `_compute_reflection(field, height, tx_pos, scene, config, effective, monitor, coords, reflection_detail)` | tracer.py:560-593 | Reflection field via `compute_reflection_field` |
| `_diffraction_edge_anchor(field, tx_pos)` | tracer.py:595-599 | Edge anchor coordinate |
| `_compute_diffraction(X, Y, height, tx_pos, scene, config, effective, monitor, field, coords, reflection_detail, return_audit)` | tracer.py:601-630 | Diffraction field via `compute_diffraction_field` |
| `_jones_metadata(axis, explicit_receiver_projection)` | tracer.py:503-545 | Jones basis metadata strings |
| `_build_trace_metadata(dif_components, reflection_detail, solver_controls, monitor, scene, config, grid_shape, effective_resolution, calculation_height)` | tracer.py:632-710 | Full metadata dict assembly |
| `_zero_diffraction_components(field, scene, config, effective, reason, return_audit)` | tracer.py:712-774 | Zero stub when diffraction disabled |
| `_estimate_state_bytes(history_size)` | tracer.py:319-330 | Byte estimation for performance guardrails |
| `_build_performance_guardrails(solver_controls, dif_components, state_bytes_fn)` | tracer.py:332-361 | Profiling metadata |

The main `trace_field_monitor` body is the current `_trace_field_monitor`
(tracer.py:776-957), with `self.xxx` references replaced by `config.xxx` or
function parameters.

### 5.4 `trace/monitors/path/trace_path.py` (~350 lines)

**Public entry point:**

```python
def trace_path_monitor(
    tx_pos: bk.Point3f,
    monitor: PathMonitor,
    scene: Scene,
    config: TraceConfig,
    solver_controls: dict,
    *,
    reflection_detail=None,
    verbose: bool = False,
    return_timing: bool = False,
) -> tuple[PathResult, Mapping[str, object] | None]:
    """
    Run LoS + reflection + diffraction for a PathMonitor.

    Returns (PathResult, reflection_detail).
    """
```

**Module-private functions:**

| Function | Source | Purpose |
|---|---|---|
| `_path_monitor_positions(monitor, device)` | tracer.py:410-412 | Ensure positions on correct device |
| `_receiver_groups(rx_positions)` | tracer.py:414-424 | Group receivers by z-coordinate |
| `_build_path_trace_metadata(monitor, solver_controls, reflection_detail, diffraction_groups, rx_positions, config, return_timing, timing, path_counts)` | tracer.py:450-501 | PathResult metadata assembly |

The main `trace_path_monitor` body is the current `_trace_path_monitor`
(tracer.py:959-1145).

### 5.5 `trace/tracer.py` After Refactor (~400 lines)

**Retained content:**

| Content | Lines | Notes |
|---|---|---|
| Imports + constants | ~30 | Remove monitor-orchestration imports, add `from .monitors import ...` |
| `Tracer.__init__` | ~60 | Unchanged |
| `_resolve_material` | ~10 | Unchanged |
| `_coerce_tx_pos` | ~6 | Unchanged |
| `_coerce_vertices` / `_coerce_faces` | ~40 | Unchanged |
| `update_scene` | ~6 | Unchanged |
| `create_intersection_func` | ~30 | Unchanged |
| `_resolve_trace_monitors` | ~40 | Unchanged |
| `_normalized_monitor_bounds` | ~3 | Unchanged |
| `trace()` | ~60 | Rewritten to call `trace_field_monitor` / `trace_path_monitor` |

**New `trace()` implementation:**

```python
def trace(self, tx_pos, *, monitor=None, verbose=False,
          return_timing=False, return_diffraction_audit=False):
    tx_pos = self._coerce_tx_pos(tx_pos)
    plane_monitors, path_monitors = self._resolve_trace_monitors(monitor=monitor)
    solver_controls = resolve_solver_controls(self.config.trace)

    monitor_payloads = {}
    reflection_detail_cache = {}

    for m in plane_monitors:
        key = reflection_discovery_key_for_field(m, tx_pos)
        payload, rd = trace_field_monitor(
            tx_pos, m, self.scene, self.config.trace, solver_controls,
            reflection_detail=reflection_detail_cache.get(key),
            verbose=verbose,
            return_timing=return_timing,
            return_diffraction_audit=return_diffraction_audit,
        )
        monitor_payloads[m.name] = payload
        if rd is not None:
            reflection_detail_cache[key] = rd

    path_payloads = {}
    for m in path_monitors:
        key = reflection_discovery_key_for_path(m)
        pr, rd = trace_path_monitor(
            tx_pos, m, self.scene, self.config.trace, solver_controls,
            reflection_detail=reflection_detail_cache.get(key),
            verbose=verbose,
            return_timing=return_timing,
        )
        path_payloads[m.name] = pr
        if rd is not None:
            reflection_detail_cache[key] = rd

    return Result(
        scene=self.scene,
        monitors=monitor_payloads,
        path_monitors=path_payloads,
        primary_monitor_name=plane_monitors[0].name if plane_monitors else None,
    )
```

### 5.6 `trace/__init__.py`

```python
"""Trace-layer package exports."""

from .los import compute_los_field, los_blocked
from .reflection import compute_reflection_field
from .tracer import Tracer

__all__ = [
    "Tracer",
    "compute_los_field",
    "compute_reflection_field",
    "los_blocked",
]
```

No change to public exports. The `monitors` package is internal to `trace/`.

---

## 6. Parameter Threading: How `self.xxx` Becomes Function Parameters

The main challenge is that `_trace_field_monitor` and `_trace_path_monitor`
currently read ~20 attributes from `self`. After extraction, these come from
two sources:

### From `TraceConfig` (passed as `config`):

- `config.tx_polarization`
- `config.rx_polarization`
- `config.reflection_coef`
- `config.min_ray_contribution_threshold`
- `config.reflection_relative_permittivity`
- `config.reflection_conductivity`
- `config.reflection_material` → resolved via `_resolve_material`
- `config.diffraction_material` → resolved via `_resolve_material`
- `config.use_scene_materials_for_reflection`
- `config.use_scene_materials_for_diffraction`
- `config.enable_rd_diffraction`
- `config.resolution_wavelength`
- `config.diffraction_execution`

### Derived values (computed from `config` + `frequency`):

- `wavelength = C / frequency`
- `k = 2 * pi / wavelength`
- `reflection_material` (resolved dict)
- `diffraction_material` (resolved dict)

**Solution:** The monitor functions accept `config: TraceConfig` plus
`scene: Scene`, and derive `wavelength` / `k` from `scene.frequency` or
receive them as explicit parameters. The material resolution logic
(`_resolve_material`) moves to `common.py` as a pure function and is called
by each monitor module.

Alternatively, `Tracer.__init__` can build a frozen `ResolvedTraceConfig`
dataclass that holds all derived values, and pass that to the monitor
functions. This avoids re-deriving wavelength/k/materials on each call:

```python
@dataclass(frozen=True)
class ResolvedTraceConfig:
    """Pre-resolved trace parameters ready for monitor functions."""
    frequency: float
    wavelength: float
    k: float
    tx_polarization: tuple
    rx_polarization: tuple | None
    reflection_coef: float
    reflection_material: dict | None      # already resolved
    diffraction_material: dict | None     # already resolved
    min_ray_contribution_threshold: float
    reflection_relative_permittivity: float
    reflection_conductivity: float
    use_scene_materials_for_reflection: bool
    use_scene_materials_for_diffraction: bool
    enable_rd_diffraction: bool
    resolution_wavelength: float
    diffraction_execution: DiffractionExecution
```

`Tracer.__init__` builds this once; `trace()` passes it to monitor functions.
This is the cleanest option and avoids re-resolving materials per call.

---

## 7. Modules That Do NOT Move

| Module | Lines | Reason |
|---|---|---|
| `los.py` | 83 | Pure function, monitor-agnostic |
| `materials.py` | 272 | Data conversion, monitor-agnostic |
| `reflection/` | 2339 | `discover_reflection_paths` + `compute_reflection_field` already decoupled |
| `diffraction/` | 5944 | State enumeration + UTD evaluation, monitor-agnostic |
| `path/` | 1034 | DrJit→PyTorch conversion layer for path output |
| `result.py` | 492 | Pure data classes, no trace logic |
| `monitors.py` (scene-level) | 258 | Declarative monitor definitions, belongs to scene layer |

---

## 8. Execution Plan

| Step | Action | Risk | Test |
|---|---|---|---|
| 1 | Create `trace/monitors/__init__.py` (empty) | None | Import check |
| 2 | Create `trace/monitors/common.py`: extract `resolve_solver_controls` + discovery key functions as pure functions | Low | Unit test solver controls with same inputs → same outputs |
| 3 | Create `trace/monitors/field/trace_field.py`: move all FieldMonitor orchestration from tracer.py; replace `self.xxx` with config/scene params | Medium | `pytest tests/trace/test_field_monitor_phase4a.py --gpu` |
| 4 | Create `trace/monitors/path/trace_path.py`: move all PathMonitor orchestration from tracer.py | Low | `pytest tests/trace/test_path_monitor.py --gpu` |
| 5 | Slim tracer.py: remove moved code, update `trace()` to call new modules | Low | All tests pass |
| 6 | Update `trace/__init__.py` if needed | None | Import check |
| 7 | Full test suite | Validation | `pytest tests --gpu` |

**Step 3 is the highest-risk step** because FieldMonitor orchestration has the
most `self.xxx` references (~20 attributes) and touches the most solver APIs.
Recommend doing step 3 first, running FieldMonitor tests, then proceeding to
step 4.

---

## 9. Final Line Count Summary

| File | Before | After | Delta |
|---|---|---|---|
| `tracer.py` | 1220 | ~400 | -820 |
| `monitors/__init__.py` | — | ~10 | +10 |
| `monitors/common.py` | — | ~120 | +120 |
| `monitors/field/trace_field.py` | — | ~500 | +500 |
| `monitors/path/trace_path.py` | — | ~350 | +350 |
| All other files | 9632 | 9632 | 0 |
| **Total** | **10852** | **~11012** | **+160** |

Net change is ~160 lines (mostly the new function signatures replacing `self`
access, plus module boilerplate). No logic duplication introduced.

---

## 10. Dependency Diagram After Refactor

```
tracer.py (Tracer)
    │
    ├── resolve_solver_controls()          monitors/common.py
    ├── reflection_discovery_key_*()       monitors/common.py
    │
    ├── trace_field_monitor()              monitors/field/trace_field.py
    │   ├── compute_los_field              los.py
    │   ├── compute_reflection_field       reflection/field.py
    │   └── compute_diffraction_field      diffraction/api.py
    │
    └── trace_path_monitor()               monitors/path/trace_path.py
        ├── collect_los_paths              path/los.py
        ├── collect_reflection_paths       path/reflection.py
        │   └── discover_reflection_paths  reflection/field.py
        ├── collect_diffraction_state_paths path/diffraction.py
        │   └── _prepare_diffraction_state_arrays
        │                                  diffraction/builders/
        └── merge_paths                    path/merge.py
```

All arrows point downward. No circular dependencies. Solver modules (bottom
row) have no knowledge of monitors.
