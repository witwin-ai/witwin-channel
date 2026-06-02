# Channel Core Scene Migration Plan

## Goal

Replace the legacy RFDT scene container with a declarative scene model built on `witwin.core`, following the same integration pattern already used by `radar` and `maxwell`.

## Current State

- The old `rfdt.Scene` owned raw `vertices/faces`, Mitsuba scene state, edge caches, and diffraction preprocessing in one class.
- Most propagation code already depended only on runtime-facing properties such as `scene.vertices`, `scene.faces`, `scene.mi_scene`, `scene.tri_data_gpu`, and `scene.get_edge_data(...)`.
- Tests existed, but the suite mixed proper pytest tests with script-style smoke checks, lacked suite-level pytest configuration, and had no explicit acceptance gate.

## Target Architecture

### Public scene layer

- `Scene` extends `witwin.core.SceneBase`.
- Scene inputs are declarative `witwin.core.Structure` objects that wrap `Material` and `GeometryBase` instances.
- Public construction paths:
  - `Scene(structures=[...])`
  - `Scene.add_structure(...)`
  - `Scene.add_mesh(...)`
  - explicit declarative geometry objects such as `witwin.core.Mesh` and `witwin.channel.DrJitMesh`

### Runtime/compiler layer

- Scene structures are compiled into one merged runtime mesh for RFDT.
- The runtime mesh remains responsible for:
  - Mitsuba scene creation
  - triangle data preload
  - vertical/generic edge selection
  - wedge geometry cache
  - triangle-to-edge lookup tables
  - height-specific edge/corner projection cache

### Compatibility boundary

- Tracer and diffraction code continue to use the existing runtime contract:
  - `scene.vertices`
  - `scene.faces`
  - `scene.mi_scene`
  - `scene.tri_data_gpu`
  - `scene.get_edge_data(...)`
  - `scene.get_diffraction_edge_data(...)`
  - `scene.get_adjacent_diffraction_edge_indices_for_triangle(...)`

## Implementation Workstreams

### 1. Scene model migration

- Refactor `rfdt.Scene` to inherit from `witwin.core.SceneBase`.
- Keep all scene inputs declarative and wrap DrJit-native mesh buffers in a `GeometryBase` implementation instead of accepting raw scene constructor tuples.
- Compile enabled structures into one merged runtime mesh.
- Rebuild Mitsuba state from the compiled runtime mesh.
- Preserve `Scene.update_vertices(...)` for differentiable runtime updates without topology rebuild.

### 2. Public API alignment

- Re-export `Material`, `Structure`, `Mesh`, `GeometryBase`, and shared geometry primitives through `witwin.channel`.
- Expose a declarative `DrJitMesh` geometry wrapper for low-level DrJit mesh generators.

### 3. Test-system formalization

- Add suite-level `conftest.py`.
- Add `pyproject.toml` pytest configuration for the subproject.
- Introduce standard markers:
  - `gpu`
  - `acceptance`
  - `validation`
- Convert script-style smoke checks into pytest assertions where they are part of the maintained suite.

### 4. Acceptance coverage

- Add acceptance tests that exercise the new core-based scene path directly.
- Add parity coverage between:
  - DrJit-backed declarative geometry path
  - declarative `Structure/Mesh` scene path
- Keep reference-based validation tests behind the `--acceptance` gate.

## Test Matrix

| Layer | Purpose | Representative tests | Gate |
|------|---------|----------------------|------|
| Smoke | Import and initialization correctness | `test_tracer_init.py` | `--gpu` |
| Scene migration | Declarative-scene compilation and runtime cache refresh | `test_core_scene_migration.py` | `--gpu --acceptance` |
| Regression | Reflection, diffraction, mixed-path bookkeeping | existing RFDT regression tests | `--gpu` |
| Validation | Closed-form/reference comparisons and audit exports | `test_validation_references.py`, `test_validation_state_audit.py` | `--gpu --acceptance` |

## Acceptance Criteria

- `Scene` can be constructed directly from `witwin.core.Structure` objects.
- Declarative core scenes and declarative DrJit-mesh scenes compile to equivalent RFDT runtime topology.
- `Scene.update_vertices(...)` continues to refresh runtime data without breaking edge caches or triangle data.
- Tracer output for a declarative core scene matches the equivalent declarative DrJit-mesh scene on a small regression case.
- Reference-validation tests still pass behind the acceptance gate.
- `channel/AGENTS.md`, `channel/CLAUDE.md`, and `channel/FEATURE_LIST.md` are present and aligned with the other subprojects.

## Recommended Commands

```bash
conda activate witwin2
cd channel
python -m pytest tests/test_tracer_init.py --gpu
python -m pytest tests/test_core_scene_migration.py --gpu --acceptance
python -m pytest tests/test_validation_references.py --gpu --acceptance
python -m pytest tests/test_validation_state_audit.py --gpu --acceptance
```

## Risks To Watch

- Mitsuba parameter-key drift for the compiled runtime mesh shape name.
- Geometry conversion paths that accidentally force unnecessary GPU-CPU-GPU copies.
- Stale triangle-edge caches after topology rebuild or runtime vertex updates.
- Validation thresholds that are too strict after scene-construction refactoring.
