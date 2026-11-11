# ADR-0xx: One grid axis order for every solver (Core row-major)

- **Status:** Accepted (owner decision for 0.6.0; MC budget change accepted).
- **Date:** 2026-11-10.
- **Decision ID:** `grid-axis-order`.
- **Scope:** result layout of `deterministic`, `montecarlo.basic` and
  `montecarlo.bdpt` for a scene with exactly one `ReceiverGrid`. Layout change, no
  physics change; no launch added (MC Basic deposits natively in row-major order).

## Context

Core documents `ReceiverGrid(shape=(rows, cols))` as row-major: sample
`(row, column)` is at `origin + row * spacing[0] * x_axis + column * spacing[1] *
y_axis`. Channel's single-grid results did not follow it, and did not agree with
each other:

| Solver | `path_gain` for `shape=(4, 2)` (before) |
| --- | --- |
| Deterministic | `(1, 2, 4)` = `[T, cols, rows]` (a transposed copy of the row-major accumulation) |
| BDPT | `(1, 2, 4)` = `[T, cols, rows]` |
| MC Basic | `(1, 8)`, flattened in `(col, row)` order |
| any solver, other layouts | `[T, R]`, receiver row-major |

So `path_gain[:, k]` named different receivers in different solvers, a grid
cell and the point receiver at the same position appeared at different indices,
and MC Basic alone returned a rank-2 tensor for a single grid.

## Decision

For exactly one `ReceiverGrid` every solver returns `[T, rows, cols]` with
entry `[t, i, j]` at grid sample `(row i, column j)`; every other layout keeps
`[T, R]` in receiver row-major order. `path_gain[t].reshape(-1)` is therefore the
same receiver order as the point layout in every solver. The convention is
documented in `docs/api-reference/index.md` ("Receiver layout and grid axis
order").

- **Deterministic:** the flat accumulation is already row-major, so the grid view
  is a reshape. The per-tensor transpose copies (`path_gain`, `field`, each
  component power and field) are gone.
- **MC Basic:** every grid producer writes receiver row-major maps natively: the
  matrix-to-map layout kernel (LoS, transmission, scattering) is now the identity
  layout, the reflection deposit indexes `i0 * cols + i1` (coord0 runs along the
  grid rows), and the diffraction tape accumulation maps the RayD tape cell
  `i1 * rows + i0` to the same row-major element; each AD companion (map adjoint,
  reflection VJP, diffraction VJP/JVP) reads and writes the same order. `path_gain`
  from the elementwise finalize is therefore already row-major and is a view;
  multi-receiver grid blocks need no gather either.
- **BDPT:** its grid maps come from the same matrix-to-map kernel, so they are row
  major too; no change in BDPT itself.

All grid outputs are contiguous. A first version of this record kept MC's native
`[T, cols, rows]` maps and gathered `path_gain` (one extra launch, accepted by the
owner) with transposed views elsewhere; the owner then asked for the native
layout instead, which removes that launch.

## Budget

MC Basic kernels are unchanged from the release baseline (11 on the budget cases;
the multi-receiver grid-block gathers are gone as well). Deterministic grid cases
tighten (measured 192 for g64 and 729 for g128.c32, after this change and the D4
diffraction change; bounds 212 and 802). BDPT counts are unchanged.

## Numerical identity

`golden_mc.py` (MC Basic and BDPT on five grid scenes: reflection, diffraction,
transmission, scattering, a grid plus a point; 51 tensors, all in `[T, rows, cols]`)
compared before and after the native change: LoS, transmission, scattering, BDPT
and every component power are bit-identical; reflection and diffraction maps differ
only in the last bits where two runs of the unchanged build also differ (float
atomic order of the deposits, up to 7e-15 absolute on 1e-8 values).

## Evidence

- `tests/accuracy/test_consistency.py` four-solver agreement now indexes cells in
  Core order for all solvers and passes.
- Layout tests that pinned `[T, cols, rows]` were updated to `[T, rows, cols]`:
  deterministic `test_component_layout`, `test_grid_radiomap`,
  `test_field_continuity` (its helper transposes to its (ny, nx) map), MC Basic
  `test_basic_component_maps`, `test_basic_receiver_layouts` (now also checks a
  single grid's `path_gain` equals its columns in a two-grid solve), BDPT
  `test_component_layout`, `test_variance`, and `test_endpoints_compat`.
- `tests/ad/test_mc_basic_ad.py` and `test_mc_basic_scattering_ad.py` (finite
  differences and forward/reverse agreement through the row-major deposits) pass.
