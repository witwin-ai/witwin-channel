# RayD Diffraction Task 9 Benchmark Record - 2026-05-22

This records the smoke-scale benchmark gates added for
`docs/dev/plans/28-rayd-optix-diffraction-kernel-plan.md` Tasks 8-9.

Environment reported by the benchmark helper:

- Python: 3.11.14
- DrJit: 1.2.0
- CUDA backend: cuda
- Platform: Windows-10-10.0.26100-SP0
- RayD package version: unknown

All commands used `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`.
Several runs printed `jitc_llvm_init(): LLVM API initialization failed ..` to
stderr but exited with status 0.

## Strict Gate Runs

| Workload | Command suffix | DrJit median ms | RayD median ms | Gate |
| --- | --- | ---: | ---: | --- |
| Wall grid order-1 basic | `--mode basic-rayd-diffraction --scene wall --grid-size 4 --samples-per-tx 64 --max-diffractions 1 --warmup 0 --repeats 1 --strict-gates --json` | 4228.999 | 345.662 | 12.23x, pass |
| Wall grid order-2 BDPT direct/Keller chain | `--mode bdpt-rayd-diffraction --scene wall --grid-size 4 --samples-per-tx 64 --max-diffractions 2 --warmup 0 --repeats 1 --strict-gates --json` | 4206.166 | 690.099 | 6.10x, pass |
| Wall path export smoke | `--mode path-rayd-diffraction --scene wall --grid-size 4 --samples-per-tx 64 --max-diffractions 1 --warmup 0 --repeats 1 --strict-gates --json` | 218.067 | 108.754 | path count 5/5, pass |
| Three-cube grid order-1 basic | `--mode basic-rayd-diffraction --scene three_cubes --grid-size 4 --samples-per-tx 64 --max-diffractions 1 --warmup 0 --repeats 1 --strict-gates --json` | 913.089 | 266.787 | 3.42x, pass |

The wall workload is the current Munich-style planar receiver-grid smoke
coverage for the new CLI modes. It is not a formal large Munich XML timing.

## Supplemental Three-Cube BDPT Timing

| Workload | Command suffix | DrJit median ms | RayD median ms | Result |
| --- | --- | ---: | ---: | --- |
| Three-cube grid order-2 BDPT, 64 samples | `--mode bdpt-rayd-diffraction --scene three_cubes --grid-size 4 --samples-per-tx 64 --max-diffractions 2 --warmup 0 --repeats 1 --json` | 1942.263 | 3648.099 | 0.53x, fixed overhead dominates |
| Three-cube grid order-2 BDPT, 256 samples | `--mode bdpt-rayd-diffraction --scene three_cubes --grid-size 4 --samples-per-tx 256 --max-diffractions 2 --warmup 0 --repeats 1 --json` | 22081.283 | 3704.403 | 5.96x, pass by value |

The strict 2x BDPT gate is maintained in the benchmark mode and passed on the
wall/Munich-style order-2 smoke workload. Three-cube BDPT remains volatile at
very small sample counts because the native table and launch overhead dominate
the sampled chain work.
