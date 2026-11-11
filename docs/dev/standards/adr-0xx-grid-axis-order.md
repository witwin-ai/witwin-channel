# ADR-0xx: One grid axis order for every solver (Core row-major)

- **Status:** Accepted (owner decision for 0.6.0; MC budget change accepted).
- **Date:** 2026-11-10.
- **Decision ID:** `grid-axis-order`.
- **Scope:** result layout of `deterministic`, `montecarlo.basic` and
  `montecarlo.bdpt` for a scene with exactly one `ReceiverGrid`. Layout change, no
  physics change; `tests/budgets` MC kernel bound 11 -> 12.

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
- **MC Basic:** the native deposit maps stay `[T, cols, rows]`. `path_gain` is
  gathered once into row-major order with the existing native
  `mc_grid_maps_to_matrix` (AD-capable; its backward makes the incoming gradient
  contiguous) and viewed as `[T, rows, cols]`; `component_maps` are returned as
  transposed views. This is the one extra launch.
- **BDPT:** `path_gain`, `component_maps` and `variance` are returned as
  transposed views of the native maps; no launch is added.

MC Basic `component_maps` and the BDPT grid tensors are therefore
non-contiguous views; Deterministic and MC Basic `path_gain` are contiguous.
Making MC deposit natively in row-major order, or dropping BDPT's matrix-to-map
conversion (which would leave a native binding without a caller), would remove
the gather and the views; both are native changes left out of this record.

## Budget

`tests/budgets/launch_budgets.json`: the three MC Basic cases go from 11 to 12
kernels (measured 12; syncs, copies and bytes unchanged). The deterministic grid
cases tighten (measured 192 for g64 and 729 for g128.c32, after this change and
the D4 diffraction change; bounds 212 and 802). BDPT counts are unchanged
(measured 149/129/185).

## Evidence

- `tests/accuracy/test_consistency.py` four-solver agreement now indexes cells in
  Core order for all solvers and passes.
- Layout tests that pinned `[T, cols, rows]` were updated to `[T, rows, cols]`:
  deterministic `test_component_layout`, `test_grid_radiomap`,
  `test_field_continuity` (its helper transposes to its (ny, nx) map), MC Basic
  `test_basic_component_maps`, `test_basic_receiver_layouts` (now also checks a
  single grid's `path_gain` equals its columns in a two-grid solve), BDPT
  `test_component_layout`, `test_variance`, and `test_endpoints_compat`.
- `tests/ad/test_mc_basic_ad.py` and `test_mc_basic_scattering_ad.py` pass through
  the gathered `path_gain` (they failed until the gather's backward made its
  incoming gradient contiguous).
