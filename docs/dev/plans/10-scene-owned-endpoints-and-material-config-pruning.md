# Scene-Owned Endpoints And Material Config Pruning Implementation Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-16

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move transmitter, receiver, receiver-grid, power, polarization, and material ownership into `Scene` objects and reduce solver config classes to algorithm controls only.

**Architecture:** `Scene` owns physical setup. Solvers consume a resolved runtime bundle containing `Transmitter`, `Receiver` or `ReceiverGrid`, wave parameters, and scene material queries. Solver `Config` classes keep only algorithm and execution policy fields such as ray counts, bounce limits, backend selection, sampling, quadrature, AD, memory, and result metric.

**Tech Stack:** Python dataclasses, DrJit runtime arrays, `witwin.core.Structure` and `Material`, `witwin.channel.core.scene.Scene`, standalone `witwin.channel.deterministic`, `witwin.channel.montecarlo`, and `witwin.channel.path` packages.

---

## File Structure

- Create `witwin/channel/core/scene/endpoints.py`: public scene-owned `Transmitter`, `Receiver`, and `ReceiverGrid` types plus lookup helpers.
- Modify `witwin/channel/core/scene/scene.py`: accept scene endpoints, reject implicit material defaults, and expose endpoint resolution methods.
- Modify `witwin/channel/core/scene/__init__.py`: export endpoint classes.
- Modify `witwin/channel/deterministic/solver.py`: allow solving from scene endpoint names or objects and remove solver-side endpoint setup from the main path.
- Modify `witwin/channel/deterministic/config.py`: remove physical setup fields after endpoint path is live: `tx_power`, `noise_power`, material coefficients, and endpoint polarization.
- Modify `witwin/channel/deterministic/runtime.py`: build `TraceContext` from scene-owned endpoint objects rather than config-owned endpoint/material fields.
- Modify `witwin/channel/montecarlo/solver.py`, `witwin/channel/montecarlo/config.py`, and Monte Carlo material call sites: consume scene-owned endpoints and scene material only.
- Modify `witwin/channel/path/solver.py`, `witwin/channel/path/config.py`, and `witwin/channel/path/endpoints.py`: merge path endpoint specs with scene endpoint objects and remove config-owned material and polarization setup.
- Modify `witwin/channel/core/runtime.py`: keep the strict per-triangle material assertion and remove whole-scene material fallback for non-empty triangle runtimes.
- Modify tests under `tests/integration/`, `tests/scene/`, `tests/utils/`, and examples under `examples/` to use scene-owned endpoints and per-structure material.
- Modify `FEATURE_LIST.md`: document the user-visible scene-owned endpoint/config-pruning API.

## Invariants

- Missing material is an error. There is no solver fallback material path.
- A non-empty runtime triangle table must have `material_specified=True` for every triangle.
- `Structure(material=None)` already raises in `witwin.core`; channel-side helpers must not recreate implicit `Material()` defaults.
- `Transmitter.power` owns RSS scaling.
- `Transmitter.polarization` and `Receiver` or `ReceiverGrid.polarization` own polarization transport. If receiver polarization is `None`, receive polarization defaults to the transmitter polarization.
- `Scene` owns endpoint lookup by name. Solvers may accept endpoint objects directly for tests, but config must not own endpoint state.
- New public scene APIs use `witwin.core` structures, materials, and geometry objects. No raw `vertices/faces` scene constructors are added.
- Runtime internals remain DrJit-native. No NumPy, Torch, or DLPack bridges are introduced in hot paths.

## Task 1: Strict Material Ownership

**Files:**
- Modify: `witwin/channel/core/scene/scene.py`
- Test: `tests/scene/test_core_scene_migration.py`
- Test: `tests/utils/test_channel_utils_runtime.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove channel scene helpers cannot silently create a default material:

```python
import pytest
from witwin.channel.core.scene import Scene
from witwin.core import Box


def test_add_mesh_requires_explicit_material():
    scene = Scene(structures=[], device="cpu")
    with pytest.raises(ValueError, match="material"):
        scene.add_mesh(name="box", geometry=Box(center=(0.0, 0.0, 0.0), size=1.0))
```

Keep existing `assert_scene_materials_complete` tests that verify incomplete per-triangle material flags raise.

- [ ] **Step 2: Run failing tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\scene\test_core_scene_migration.py::test_add_mesh_requires_explicit_material tests\utils\test_channel_utils_runtime.py -q
```

Expected before implementation: `test_add_mesh_requires_explicit_material` fails because `Scene.add_mesh()` still uses `material or Material()`.

- [x] **Step 3: Implement strict add_mesh**

Change `Scene.add_mesh(...)` to reject `material is None`:

```python
if material is None:
    raise ValueError("Scene.add_mesh requires an explicit witwin.core.Material.")
return self.add_structure(
    Structure(geometry=geometry, material=material, name=name, metadata=metadata)
)
```

- [ ] **Step 4: Run tests**

Run the command from Step 2. Expected: all selected tests pass.

## Task 2: Scene Endpoint Types

**Files:**
- Create: `witwin/channel/core/scene/endpoints.py`
- Modify: `witwin/channel/core/scene/__init__.py`
- Modify: `witwin/channel/core/scene/scene.py`
- Test: `tests/scene/test_core_scene_migration.py`

- [ ] **Step 1: Write failing endpoint tests**

Add tests for construction and lookup:

```python
from witwin.channel.core.scene import Receiver, ReceiverGrid, Scene, Transmitter


def test_scene_stores_named_transmitter_and_receiver_grid():
    tx = Transmitter(name="tx", position=(0.0, 0.0, 2.0), polarization=(0.0, 1.0, 0.0), power=2.5)
    grid = ReceiverGrid(
        name="map",
        axis="z",
        position=1.0,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_shape=(4, 4),
        polarization=None,
    )
    scene = Scene(structures=[], transmitters=[tx], receivers=[grid], device="cpu")
    assert scene.transmitter("tx") is tx
    assert scene.receiver("map") is grid
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\scene\test_core_scene_migration.py::test_scene_stores_named_transmitter_and_receiver_grid -q
```

Expected before implementation: import or constructor failure.

- [x] **Step 3: Implement endpoint classes**

Create dataclasses:

```python
@dataclass(frozen=True, slots=True)
class Transmitter:
    name: str
    position: object
    polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    power: float = 1.0
    orientation: tuple[float, float, float] | None = None
```

`Receiver` has `name`, `position`, `polarization`, and `orientation`.

`ReceiverGrid` has `name`, `axis`, `position`, `bounds`, exactly one of `grid_shape` or `cell_size`, `polarization`, and `orientation`. It should be coercible to the existing shared `GridSpec` contract by exposing matching attributes.

- [x] **Step 4: Add Scene endpoint storage and lookup**

`Scene.__init__` accepts `transmitters=None` and `receivers=None`, stores them, and rejects duplicate names. Add:

```python
def transmitter(self, endpoint="tx"):
    ...

def receiver(self, endpoint):
    ...
```

Lookups accept object instances directly and return them unchanged.

- [ ] **Step 5: Run tests**

Run the command from Step 2. Expected: pass.

## Task 3: Deterministic Solver Endpoint Runtime

**Files:**
- Modify: `witwin/channel/deterministic/solver.py`
- Modify: `witwin/channel/deterministic/runtime.py`
- Modify: `witwin/channel/deterministic/config.py`
- Test: `tests/integration/test_deterministic_radiomap_package.py`

- [ ] **Step 1: Write failing deterministic endpoint test**

Add a test that calls deterministic solve without `tx_pos` and `grid`, using named scene endpoints:

```python
from witwin.channel.core.scene import ReceiverGrid, Transmitter


def test_deterministic_solve_uses_scene_endpoints():
    scene = _build_channel_scene()
    scene.add_transmitter(Transmitter(name="tx", position=(-3.0, 0.0, 1.0), power=2.0))
    scene.add_receiver(ReceiverGrid(name="map", axis="z", position=1.0, bounds=((-2.0, 2.0), (-2.0, 2.0)), grid_shape=(4, 4)))
    result = deterministic.solve(scene=scene, frequency=3.5e9, transmitter="tx", receiver="map")
    assert result.tx_power == 2.0
    assert result.grid_shape == (4, 4)
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\integration\test_deterministic_radiomap_package.py::test_deterministic_solve_uses_scene_endpoints -q
```

Expected before implementation: `solve()` rejects `transmitter` and `receiver`.

- [x] **Step 3: Add endpoint-aware solve signature**

Allow:

```python
def solve(*, scene, frequency, tx_pos=None, grid=None, transmitter=None, receiver=None, config=None):
```

Resolve `tx_pos` and `grid` from endpoints when explicit legacy args are absent. For this migration, explicit `tx_pos` or `grid` can still be accepted while tests and examples move to endpoints.

- [x] **Step 4: Use endpoint power and polarization**

Create runtime `Tx` from `Transmitter.position` and `Transmitter.polarization`; compute RSS with `Transmitter.power`. Use `ReceiverGrid.polarization` in `runtime.with_rx(...)`.

- [x] **Step 5: Remove deterministic physical config fields**

After endpoint tests pass, remove from deterministic `Config` and `ResolvedTraceConfig`:

```text
tx_power
noise_power
reflection_coef
reflection_relative_permittivity
reflection_conductivity
tx_polarization
rx_polarization
```

Replace material coefficient reads with scene material queries. Keep `noise_power` only as a scene-level optional attribute or a future environment object, not solver config.

- [ ] **Step 6: Run deterministic package tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\integration\test_deterministic_radiomap_package.py tests\integration\test_deterministic_material_gradients.py -q
```

Expected: pass after test updates.

## Task 4: Path Solver Endpoint Runtime

**Files:**
- Modify: `witwin/channel/path/solver.py`
- Modify: `witwin/channel/path/config.py`
- Modify: `witwin/channel/path/endpoints.py`
- Test: `tests/integration/test_path_solver_package.py`

- [x] **Step 1: Add scene endpoint path test**

Add a path test using `Transmitter` plus a list of `Receiver` objects stored on the scene. The result metadata must report endpoint polarization from endpoint objects, not config.

- [x] **Step 2: Remove path config physical fields**

Remove material override, `use_scene_materials_*`, and polarization fields from `witwin.channel.path.Config`. Path solver always asserts scene materials before reflection or diffraction work.

- [x] **Step 3: Update path collection calls**

Pass `tx.polarization` and `rx.effective_polarization(tx)` from resolved endpoint objects into existing low-level path collection functions until those functions are also migrated to runtime bundles.

- [ ] **Step 4: Run path tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\integration\test_path_solver_package.py tests\trace\test_sionna_path_solver_parity.py -q
```

Expected: pass after tests are updated to scene endpoints.

## Task 5: Monte Carlo Endpoint And Material Pruning

**Files:**
- Modify: `witwin/channel/montecarlo/solver.py`
- Modify: `witwin/channel/montecarlo/config.py`
- Modify: `witwin/channel/montecarlo/materials.py`
- Modify: `witwin/channel/montecarlo/path/reflection.py`
- Modify: `witwin/channel/montecarlo/path/diffraction.py`
- Test: `tests/integration/test_monte_carlo_radiomap_package.py`
- Test: `tests/integration/test_monte_carlo_radiomap_integrators.py`

- [x] **Step 1: Add scene endpoint Monte Carlo test**

Add a test solving with `transmitter="tx"` and `receiver="map"` on the scene.

- [x] **Step 2: Remove Monte Carlo physical config fields**

Remove material coefficient fields, material override mappings, endpoint polarization, and `tx_power` from `TraceConfig` and public `Config`.

- [x] **Step 3: Convert material helpers to scene-only lookup**

Delete `normalize_override(...)`. `materials.edge_faces(...)` accepts `scene`, adjacent face indices, and `default_gain=1.0` only for inactive lanes; active lanes must gather per-face scene material.

- [ ] **Step 4: Run Monte Carlo tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\integration\test_monte_carlo_radiomap_package.py tests\integration\test_monte_carlo_radiomap_integrators.py -q
```

Expected: pass after tests are updated to scene endpoints and material.

## Task 6: Public Cleanup And Documentation

**Files:**
- Modify: `FEATURE_LIST.md`
- Modify: `examples/*.py`
- Modify: `docs/dev/README.md`

- [ ] **Step 1: Update examples**

Replace examples that pass `tx_pos`, `GridSpec`, endpoint polarization, or material coefficients through config with scene-owned `Transmitter`, `ReceiverGrid`, and per-structure `Material`.

- [ ] **Step 2: Update feature list**

Add a concise entry that scene endpoints and per-structure material are now the public setup model for deterministic, Monte Carlo, and path solvers.

- [ ] **Step 3: Run smoke tests**

Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\scene tests\integration\test_deterministic_radiomap_package.py tests\integration\test_path_solver_package.py -q
```

Expected: pass.

## Final Acceptance

Run the targeted package acceptance set:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\utils\test_channel_utils_runtime.py tests\scene\test_core_scene_migration.py tests\integration\test_deterministic_radiomap_package.py tests\integration\test_path_solver_package.py tests\integration\test_monte_carlo_radiomap_package.py -q
```

Expected: pass.

Run GPU-focused regression where available:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\main\test_position_rotation_tx.py tests\main\test_material.py -q --gpu
```

Expected: pass or known GPU-environment skip only.

## Self-Review

- Spec coverage: the plan covers scene-owned endpoints, strict material ownership, deterministic solver migration, path solver migration, Monte Carlo migration, config pruning, docs, and tests.
- Placeholder scan: no task relies on unspecified future code; each migration task identifies concrete files and validation commands.
- Type consistency: endpoint types are named `Transmitter`, `Receiver`, and `ReceiverGrid`; solver parameters are `transmitter` and `receiver`; existing `GridSpec` remains the internal grid contract during migration.
