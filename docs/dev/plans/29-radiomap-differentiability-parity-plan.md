# Radiomap Differentiability FD-Parity Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-22

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 2-4 touch native CUDA/C++; tasks that edit RayD must build the local source tree at `E:\Code\RayDi` (not the PyPI `rayd`). Use the `witwin2` environment and the CMake build/install flow in `AGENTS.md`.

**Goal:** Make channel radiomap gradients trustworthy. Today the deterministic and Monte Carlo radiomap solvers only finite-difference-validate for smooth dependencies with reflection off; reflection, geometry motion, and reverse-mode-through-reflection do not validate or are unimplemented. This plan first makes the gap measurable as a CI contract, then closes it solver-path by solver-path, gated by FD parity.

This is the foundation for the Tier P0 inverse-problem and calibration toolkit in `plans/00-research-feature-roadmap.md`. That toolkit is meaningless until gradients FD-match, so this plan precedes it.

## Background (grounding)

Two findings establish the scope:

1. **Reverse-mode reflection is structurally absent, not buggy.** `ReflectionAccumulateOp` and `FWeightReflectionAccumulateOp` in `witwin/channel/deterministic/kernels/reflection/bind.cpp` override only `forward()` (JVP). Neither overrides `backward()`, so `dr.backward(...)` through a radiomap reflection solve hits DrJit's default and raises `ReflectionAccumulate::backward(): operation is unimplemented!`. The forward JVP kernel `reflection_accumulate_jvp` already exists; the reflection contribution is linear in its slot/geometry/rx inputs, so the VJP is the transpose of the existing JVP.

2. **The path solver is a working FD-validated reference.** `tests/path/test_path_solver_ad.py` validates reflection gradients (tx position, material `eps_r`, surface geometry `y`) against central finite differences at `rel=5e-3` using `dr.backward`. The path solver computes reflection field on the DrJit-native path; radiomap routes reflection through the native forward-only `ReflectionAccumulateOp` for performance. The gap is therefore "the fast radiomap reflection kernel has no reverse", not a physics error.

The existing radiomap gradient tests (`tests/deterministic/test_deterministic_material_gradients.py`, the example notebooks) only assert `isfinite`, `abs>0`, and JVP-sum == VJP self-consistency. They never compare against finite differences and they run at `max_bounces=0` on 1x1 grids. They cannot catch the gaps above.

## Scope

In scope:

- A reusable FD-vs-AD parity harness for `witwin.channel.deterministic.solve` and `witwin.channel.montecarlo.solve`.
- Reverse-mode (VJP) reflection through the native reflection CustomOps.
- Geometry-motion gradients (vertex / object position) for radiomap reflection and diffraction.
- Native AD through the RayD OptiX diffraction kernels added in `plans/28-rayd-optix-diffraction-kernel-plan.md`.

Out of scope:

- New public scene/material/result API shapes.
- New CPU fallback paths.
- Heuristic gradient smoothing or soft-visibility hacks (per channel `CLAUDE.md`).
- The inverse-problem toolkit itself (separate follow-up once parity passes).

## Parity Matrix

The harness must cover this matrix for both deterministic and Monte Carlo radiomap solves, comparing AD against central finite differences on `result.path_gain[tx=0]` (raw, never `squeeze_tx()` which detaches AD):

| Parameter | Component | JVP | VJP |
| --- | --- | --- | --- |
| tx position `x` | LoS (`max_bounces=0, max_diffraction_order=0`) | expect pass | expect pass |
| material `eps_r` | diffraction (`max_bounces=0, max_diffraction_order=1`) | expect pass | expect pass |
| tx position `x` | reflection (`max_bounces>=1`) | known fail | known fail (raises) |
| tx position `x` | diffraction (`max_diffraction_order>=1`) | known fail | known fail |
| vertex / object position | reflection + diffraction | known fail (MC JVP~0, det uncorrelated) | known fail |

"known fail" cells are encoded as `xfail(strict=True)` so they are tracked, and flip to `xpass`/pass as tasks 2-4 land.

## Tasks

### Task 0: Land the completed RayD diffraction branch

- [ ] Merge `codex/rayd-diffraction-stage1` into `main` (plan 28 tasks are all complete).
- [ ] Move `plans/28-rayd-optix-diffraction-kernel-plan.md` to `archive/completed/` keeping its numeric prefix, and remove its row from `docs/dev/README.md`.

### Task 1: FD-vs-AD parity harness and gate

**Files:**

- Create: `tests/grad/test_radiomap_fd_parity.py`
- Optional helper: `tests/support/` shared FD utility if it does not duplicate an existing one.

- [ ] Add a central-difference helper that perturbs a scalar scene parameter and re-solves (`(f(+h) - f(-h)) / 2h`) on `result.path_gain[0]`, mirroring `tests/path/test_path_solver_ad.py`.
- [ ] Add JVP cells via `dr.forward_to(result.path_gain, ...)[0]` and VJP cells via `dr.backward(dr.sum(result.path_gain[0]), ...)`.
- [ ] Implement the full parity matrix above for deterministic and Monte Carlo solves on a small fixed-seed scene (single cube, small grid).
- [ ] Mark known-failing cells `pytest.mark.xfail(strict=True)` with a reason string referencing this plan.
- [ ] Run:

```powershell
conda run -n witwin2 python -m pytest tests\grad\test_radiomap_fd_parity.py -q --gpu
```

Expected: smooth cells pass; reflection/geometry/VJP cells xfail.

### Task 2: Reverse-mode reflection (VJP)

**Files:**

- Modify: `E:\Code\RayDi\...` reflection accumulation native sources (VJP kernel) if the kernel lives in RayD, else `witwin/channel/deterministic/kernels/reflection/`.
- Modify: `witwin/channel/deterministic/kernels/reflection/bind.cpp` (`ReflectionAccumulateOp::backward`, `FWeightReflectionAccumulateOp::backward`).
- Test: `tests/grad/test_radiomap_fd_parity.py`

- [ ] Add a `reflection_accumulate_vjp` native launcher (transpose of the existing JVP).
- [ ] Override `backward()` on both reflection CustomOps to call it and accumulate input grads.
- [ ] Build and install the native extension into `witwin2`.
- [ ] Flip the reflection VJP parity cell from xfail to pass.

### Task 3: Geometry-motion gradients

**Files:**

- Modify: reflection/diffraction state construction so vertex positions stay on the AD path (MC side OptiX BVH currently detaches vertices).
- Modify: deterministic geometry-motion boundary term (currently non-zero but uncorrelated with FD).
- Test: `tests/grad/test_radiomap_fd_parity.py`

- [ ] Reproduce and isolate the deterministic geometry-motion FD mismatch (boundary/Reynolds-transport term) and the MC `JVP~0` vertex detachment.
- [ ] Implement the correct boundary contribution without heuristic smoothing.
- [ ] Flip the geometry parity cells from xfail to pass.

### Task 4: Native AD through RayD diffraction kernels

**Files:**

- Modify: RayD diffraction kernels from plan 28 (`accumulate_diffraction_order1`, chains) to emit JVP/VJP, or route AD-sensitive diffraction back through the validated DrJit path under one explicit contract.
- Modify: `witwin/channel/montecarlo/integrators/`, `witwin/channel/deterministic/trace/diffraction/`.
- Test: `tests/grad/test_radiomap_fd_parity.py`

- [ ] Decide per plan 28 open question: native AD in the diffraction kernel vs explicit DrJit AD fallback for grad-sensitive inputs.
- [ ] Make the fast primal path and the gradient path consistent (no silent divergence).
- [ ] Flip the diffraction parity cells from xfail to pass.

## Verification Commands

```powershell
conda run -n witwin2 python -m pytest tests\grad\test_radiomap_fd_parity.py -q --gpu
conda run -n witwin2 python -m pytest tests\path\test_path_solver_ad.py -q --gpu
conda run -n witwin2 python -m pytest tests\deterministic\test_deterministic_material_gradients.py -q --gpu
```

## Rollout Policy

- Task 1 ships first as a tracked contract; it must not block the build.
- Each of tasks 2-4 flips its parity cells from xfail to pass in the same change that implements it.
- Do not claim FD-validated "move-the-cube" or reflection radiomap gradients in demos/docs until the corresponding cell is green.
- The inverse-problem/calibration toolkit (roadmap Tier P0) starts only after the smooth + reflection cells are green.

## Open Questions

- Whether reflection VJP should live in RayD or in the channel native extension (depends on where the forward accumulation kernel is owned).
- Whether geometry-motion boundary terms need a dedicated discontinuity estimator or can reuse the path solver's geometry AD approach.
- Whether RayD diffraction kernels get native AD now or defer to an explicit DrJit AD fallback until parity is proven.
