# Unified Radiomap Grid Result Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a shared radiomap result contract used by deterministic and Monte Carlo solvers without erasing their coherent-field versus sampled-power payload differences.

**Architecture:** Put common result dataclasses and coordinate shaping helpers in `witwin.channel.core.radiomap_result` so solver packages do not import each other. Return the same shared `RadioMapResult` from deterministic and Monte Carlo solvers, with `witwin.channel.deterministic.Result` and `witwin.channel.montecarlo.Result` as public aliases rather than package-owned wrapper dataclasses.

**Tech Stack:** Python dataclasses, DrJit-backed Witwin tensors, existing `witwin.channel.core.tensors` conversion helpers, pytest through the `witwin2` conda environment.

---

### Task 1: Add shared radiomap result primitives

**Files:**
- Create: `witwin/channel/core/radiomap_result.py`
- Modify: `witwin/channel/core/__init__.py`
- Test: `tests/integration/test_unified_radiomap_result_contract.py`

- [x] **Step 1: Write the failing test for common imports**

Add this test file:

```python
from witwin.channel.core.radiomap_result import (
    RadioMapCoordinates,
    RadioMapFieldPayload,
    RadioMapPowerPayload,
    RadioMapResult,
)


def test_shared_radiomap_result_types_are_importable():
    assert RadioMapCoordinates.__name__ == "RadioMapCoordinates"
    assert RadioMapResult.__name__ == "RadioMapResult"
    assert RadioMapFieldPayload.__name__ == "RadioMapFieldPayload"
    assert RadioMapPowerPayload.__name__ == "RadioMapPowerPayload"
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py::test_shared_radiomap_result_types_are_importable -q
```

Expected: fail with `ModuleNotFoundError: No module named 'witwin.channel.core.radiomap_result'`.

- [x] **Step 3: Implement the shared dataclasses**

Create `witwin/channel/core/radiomap_result.py` with `RadioMapCoordinates`, `RadioMapResult`, `RadioMapFieldPayload`, `RadioMapPowerPayload`, and `coordinates_from_grid(grid, sample_positions=())`.

- [x] **Step 4: Export the shared types**

Update `witwin/channel/core/__init__.py` to export the new symbols without importing solver packages.

- [x] **Step 5: Run the import test and verify it passes**

Run the same `pytest` command. Expected: one passing test.

### Task 2: Move deterministic result onto the shared dataclass

**Files:**
- Delete: `witwin/channel/deterministic/result.py`
- Modify: `witwin/channel/deterministic/__init__.py`
- Modify: `witwin/channel/deterministic/solver.py`
- Test: `tests/integration/test_unified_radiomap_result_contract.py`

- [x] **Step 1: Add failing deterministic result contract assertions**

Extend the test file with a small scene test that runs `deterministic.solve(...)` with a scene-owned `ReceiverGrid` and asserts:

```python
assert det.solver == "deterministic"
assert det.field is not None
assert det.power is None
assert det.coords.sample_positions
assert det.path_gain.shape == det.rss.shape == det.sinr.shape
```

- [x] **Step 2: Run the deterministic contract test and verify it fails**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py::test_deterministic_result_uses_shared_contract -q
```

Expected: fail because `solver`, `field`, or `power` does not exist on deterministic `Result`.

- [x] **Step 3: Update deterministic result type**

Make `witwin.channel.deterministic.Result` a direct alias of `RadioMapResult`, delete the package-local `result.py`, and return `RadioMapResult(...)` from the solver with:

```python
field: RadioMapFieldPayload | None = None
power: RadioMapPowerPayload | None = None
components: Mapping[str, FloatTensor] = field(default_factory=dict)
```

Keep the public import path `witwin.channel.deterministic.Result`, but make it point to the shared dataclass.

- [x] **Step 4: Update deterministic result construction**

In `witwin/channel/deterministic/solver.py`, use `coordinates_from_grid(...)`, set `solver="deterministic"`, populate `field=RadioMapFieldPayload(...)`, and leave `power=None`. Preserve the existing `components` mapping.

- [x] **Step 5: Run deterministic focused tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py::test_deterministic_result_uses_shared_contract tests/integration/test_deterministic_radiomap_package.py -q
```

Expected: all selected tests pass.

### Task 3: Move Monte Carlo result onto the shared base

**Files:**
- Delete: `witwin/channel/montecarlo/result.py`
- Modify: `witwin/channel/montecarlo/__init__.py`
- Modify: `witwin/channel/montecarlo/solver.py`
- Modify: `witwin/channel/montecarlo/integrators/basic.py`
- Test: `tests/integration/test_unified_radiomap_result_contract.py`

- [x] **Step 1: Add failing Monte Carlo result contract assertions**

Extend the shared test file with a scene-owned `ReceiverGrid` Monte Carlo solve and assert:

```python
assert mc.solver == "montecarlo"
assert mc.field is None
assert mc.power is not None
assert mc.power.incoherent is mc.incoherent
assert mc.coords.sample_positions == ()
assert mc.path_gain.shape == mc.rss.shape == mc.sinr.shape
```

- [x] **Step 2: Run the Monte Carlo contract test and verify it fails**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py::test_monte_carlo_result_uses_shared_contract -q
```

Expected: fail because `solver`, `field`, `power`, or `coords.sample_positions` does not exist.

- [x] **Step 3: Update Monte Carlo result type**

Make `witwin.channel.montecarlo.Result` a direct alias of `RadioMapResult`, delete the package-local `result.py`, and return `RadioMapResult(...)` from the integrator with Monte Carlo-specific fields: `combine_mode`, `receiver_model`, `tx_association_map`, `coherent`, `incoherent`, `coherent_power`, `power`, and `timing`.

- [x] **Step 4: Update Monte Carlo result construction**

In `witwin/channel/montecarlo/integrators/basic.py`, use `coordinates_from_grid(...)`, set `solver="montecarlo"`, set `field=None`, and populate `power=RadioMapPowerPayload(...)`.

- [x] **Step 5: Run Monte Carlo focused tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py::test_monte_carlo_result_uses_shared_contract tests/integration/test_monte_carlo_radiomap_package.py -q
```

Expected: all selected tests pass.

### Task 4: Verify unified scene-grid behavior and docs

**Files:**
- Modify: `FEATURE_LIST.md`
- Test: `tests/integration/test_unified_radiomap_result_contract.py`

- [x] **Step 1: Add cross-solver shared-grid assertions**

Add one test that solves deterministic and Monte Carlo with the same scene receiver name and asserts equal `grid_shape`, `cell_size`, `surface["axis"]`, `surface["bounds"]`, and coordinate tensor shapes.

- [x] **Step 2: Run the cross-solver test and verify it passes**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py -q
```

Expected: all tests in the new file pass.

- [x] **Step 3: Update the feature list**

Add one concise bullet to `FEATURE_LIST.md` describing the shared scene-owned grid and unified result contract, including the deterministic `field` versus Monte Carlo `power` payload distinction.

- [x] **Step 4: Run final verification**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/integration/test_unified_radiomap_result_contract.py tests/integration/test_deterministic_radiomap_package.py tests/integration/test_monte_carlo_radiomap_package.py -q
```

Expected: all selected tests pass.

Observed during implementation: the full command was run through the `witwin2`
environment Python with `--gpu`. The new unified contract tests and many package
tests passed, but 11 existing Monte Carlo ThreeCubeExperiment tests failed before
result construction because `examples/monte_carlo_radiomap_three_cubes.py`
still passes the unsupported `Config(reflection_coef=0.8)` argument. The focused
contract and scene-endpoint regression tests passed separately.
