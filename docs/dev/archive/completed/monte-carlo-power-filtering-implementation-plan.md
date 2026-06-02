Status: Completed
Category: Plan
Last reviewed: 2026-04-24

# Monte Carlo Power Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement opt-in differentiable incoherent-power filtering for standalone Monte Carlo radio maps.

**Architecture:** Add public filter config types in `witwin.channel.montecarlo.config`, DrJit-native filtering operators in a new `witwin.channel.montecarlo.filtering` module, and one shared finalization hook used by Basic, BDPT, and AD result assembly. Filtering stays disabled by default and transforms `reflection` and `diffraction` power maps before totals are finalized. Diffraction filtering also transforms the shadow-boundary diffraction transition-power auxiliaries with the same operator so completion algebra remains consistent with the filtered diffraction power domain.

**Tech Stack:** Python dataclasses, DrJit CUDA AD arrays, pytest under the `witwin2` conda environment.

---

### Task 1: Public Config Contract

**Files:**
- Modify: `witwin/channel/montecarlo/config.py`
- Modify: `witwin/channel/montecarlo/__init__.py`
- Test: `tests/integration/test_monte_carlo_radiomap_integrators.py`

- [x] **Step 1: Write failing tests**

Add tests that import `FilterConfig` and `ComponentFilterConfig`, verify default disabled filtering, validate dict coercion, and reject invalid methods/radii/sigmas/blends.

- [x] **Step 2: Run tests to verify red**

Run: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/integration/test_monte_carlo_radiomap_integrators.py -q`

Expected: import failure for the new filter config types.

- [x] **Step 3: Implement config types**

Add `FilterMethod`, `ComponentFilterConfig`, and `FilterConfig`. Coerce mappings in `Config.__post_init__`, expose `Config.filtering`, include `filtering` in `Config.to_trace_config()` only if needed by internals, and export public config types from the package root.

- [x] **Step 4: Run tests to verify green**

Run the same pytest command. Expected: config tests pass.

### Task 2: DrJit Power Filter Operators

**Files:**
- Create: `witwin/channel/montecarlo/filtering.py`
- Test: `tests/integration/test_monte_carlo_radiomap_integrators.py`

- [x] **Step 1: Write failing operator tests**

Add tests for Gaussian constant preservation, impulse spreading, bilateral step preservation, component-specific application, disabled pass-through, and `dr.forward_to` through a filtered map.

- [x] **Step 2: Run tests to verify red**

Run: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/integration/test_monte_carlo_radiomap_integrators.py -q`

Expected: module/function import failure for `witwin.channel.montecarlo.filtering`.

- [x] **Step 3: Implement operators**

Implement DrJit-native normalized gather filters using grid shape, fixed spatial weights, optional bilateral range weights, and blend.

- [x] **Step 4: Run tests to verify green**

Run the same pytest command. Expected: operator tests pass.

### Task 3: Solver Integration And Metadata

**Files:**
- Modify: `witwin/channel/montecarlo/integrators/basic.py`
- Modify: `witwin/channel/montecarlo/integrators/basic_ad.py`
- Modify: `witwin/channel/montecarlo/integrators/bdpt.py`
- Modify: `witwin/channel/montecarlo/integrators/metadata.py`
- Test: `tests/integration/test_monte_carlo_radiomap_integrators.py`

- [x] **Step 1: Write failing integration tests**

Add tests that finalize a small diagnostics payload with filtering enabled, verify only configured components change, verify `path_gain == incoherent["total"]`, and verify metadata reports disabled/enabled filtering.

- [x] **Step 2: Run tests to verify red**

Run the same pytest command. Expected: missing finalization hook or metadata key failure.

- [x] **Step 3: Integrate filtering**

Add an `apply_filtering` parameter to finalization/result assembly, call it before `_finalize_component_totals`, and pass `mc_config.filtering` from Basic, BDPT, and AD result construction.

- [x] **Step 4: Run tests to verify green**

Run the same pytest command. Expected: integration tests pass.

### Task 4: User-Facing Feature List

**Files:**
- Modify: `FEATURE_LIST.md`

- [x] **Step 1: Update feature list**

Record the new opt-in differentiable Monte Carlo power filtering capability.

### Task 5: Verification

**Files:**
- All changed files

- [x] **Step 1: Run targeted tests**

Run: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/integration/test_monte_carlo_radiomap_integrators.py -q`

- [x] **Step 2: Run focused package tests**

Run: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/diffraction/test_monte_carlo_shadow_boundary_smoothing.py tests/integration/test_monte_carlo_radiomap_integrators.py -q`

- [x] **Step 3: Inspect diff**

Run: `git diff --check` and `git diff --stat`.
