# Old `witwin.channel` Test Suite (Reference)

This directory holds the test suite that targeted the pre-refactor monolithic
`witwin.channel` package. It was moved here on 2026-05-16 instead of being
deleted, so it can be consulted when building out test coverage for the
successor packages.

## Companion source

The matching pre-refactor package source lives in [`../channel/`](../channel/).
Together they form a frozen snapshot of the old design — solver source plus
the tests that exercised it.

## Why these are not in `tests/`

The successor packages replaced `witwin.channel` with:

- `witwin.channel_scene` — scene / endpoints / mesh adapters
- `witwin.channel_utils` — shared geometry, grids, materials, raygen
- `witwin.deterministic` — deterministic radiomap and path solvers
- `witwin.montecarlo` — Monte Carlo radiomap solver
- `witwin.path` — Sionna-style path solver

Every file in this directory imports `witwin.channel.*` (the old top-level
package, no longer in the source tree) — either directly, via the parity tests
that compared old vs new, or transitively through helpers such as
`_scene_helpers.py` and `tests.main.plot_*`. They cannot run in their current
form and must be ported, not just re-enabled.

## Layout

The original `tests/<category>/` structure is preserved exactly:

| Subdirectory | Files | Original purpose |
| --- | ---: | --- |
| `backend/` | 7 | Native extension, runtime backend switch, query backend gradients |
| `diffraction/` | 14 | UTD/Fresnel diffraction, polarization, shadow boundary |
| `integration/` | 5 | Step-by-step parity + old-vs-new package parity harnesses |
| `main/` | 25 | End-to-end visual `plot_*` / `test_*_main` pairs (three-cube radiomap, Munich, Sionna compare) |
| `mixed/` | 8 | Reflection-prefix diffraction, alternating mixed chains |
| `reflection/` | 7 | Reflection 3D phases, material response, symbolic-DDA toggle |
| `scene/` | 4 | Core-scene migration, field/radio-map monitor wiring |
| `support/bin/` | 57 | Benchmark, profile, gradient-diagnostic, optimization demo scripts |
| `trace/` | 11 | Tracer init, path/field monitors, Sionna parity, exports |
| `validation/` | 2 | Reference / state-audit validation harness |
| `wedge/` | 2 | Wedge runtime and RayD adapter |

## How to use as coverage reference

When adding tests for a new-package feature, search this directory for the
analogous old-package test. Useful starting points:

- **Field/radio-map parity** → `integration/test_sim_step_by_step.py`,
  `main/test_radiomap_*_main.py`
- **Diffraction physics** → `diffraction/test_utd_angle_derivatives.py`,
  `diffraction/test_shadow_boundary_treatment.py`,
  `diffraction/test_diffraction_order_breakdown.py`
- **Reflection physics** → `reflection/test_reflection_material_response.py`,
  `reflection/test_reflection_3d_phase1.py`
- **Mixed reflection + diffraction chains** → `mixed/test_*.py`
- **Backend / native kernel parity** → `backend/test_native_kernel_consistency.py`,
  `backend/test_query_backend_*.py`
- **Path-solver / Sionna parity** → `trace/test_sionna_path_solver_parity.py`,
  `trace/test_default_trace_repeatability.py`
- **End-to-end visual artifacts** → `main/plot_*.py` (paired with their
  `test_*_main.py` harness)

The fastest port pattern is: copy the old test, replace `witwin.channel.*`
imports with the new package equivalents, and adapt scene construction to use
`witwin.channel_scene.Scene` (which now owns transmitter/receiver/grid).

## Policy

- Do not import anything from this directory into the live `tests/` tree.
- Do not modify files here to "make them run" — port the test into `tests/`
  using the current public API instead.
- Once an old test has been ported (or judged obsolete), it can be deleted
  from this directory.
