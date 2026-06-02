# Munich Performance Regression Gates

Date: 2026-05-22

## Purpose

Munich-scale performance checks now have one Witwin-side benchmark entrypoint for the
standalone path solver, Monte Carlo basic, and Monte Carlo BDPT. Deterministic Munich
gates are intentionally deferred.

The benchmark is opt-in because the default workload is a GPU stress profile. The
normal pytest suite covers only parser and regression-gate logic, including the rule
that timing comparisons are valid only when the workload key matches exactly.

## Commands

Fast logic coverage, suitable for normal test runs:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\test_munich_performance_benchmark.py -q
```

Opt-in Munich smoke coverage:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\performance\test_munich_performance_smoke.py --run-optimize -q
```

Full benchmark report:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_munich_performance --json --output docs\dev\optimization\munich_solver_performance.json
```

Strict regression gate against a saved baseline:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.benchmark_munich_performance --baseline-json docs\dev\optimization\munich_solver_performance_baseline.json --strict-gates --max-regression-factor 2.0 --json
```

## Default Workload

- Path: order-1 and order-2 diffraction, `1e6` samples, first-order reflection, `max_num_paths=256`.
- Monte Carlo basic: order-1 diffraction, `256 x 256` receiver grid, `1e6` samples per TX.
- Monte Carlo BDPT: order-1 and order-2 diffraction, `256 x 256` receiver grid, `1e6` samples per TX.
- RayD-oriented defaults are exercised through `accumulation_backend="auto"` and `accumulate_primal="auto"`.

Additional available cases include `path_order0`, `path_order3`, `mc_basic_order0`,
`mc_bdpt_order0`, and `mc_bdpt_order3`; use `--cases all` to run every supported case.

## Gate Semantics

Each case records a `workload_key` computed from scene path, Sionna source root, solver,
diffraction order, sample count, grid size, max-bounce settings, seed, and backend
controls. A baseline comparison fails as `setup_mismatch` if the workload key differs,
rather than comparing unrelated timings.

When the setup matches, the gate compares current median runtime against baseline median
runtime. `--max-regression-factor 2.0` means a case fails once it is more than 2x slower
than the saved baseline.
