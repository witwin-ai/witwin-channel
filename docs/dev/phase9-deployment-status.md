# Phase 9 deployment evidence status

Phase 9 distinguishes declarations and executable checks from verified deployment
evidence:

- `benchmarks.harness.measure_cold_import()` measures a fresh interpreter against
  the source tree/build directory. It is not wheel evidence.
- `ci/wheel_smoke.py <wheel.whl>` installs one already-built wheel into a temporary
  target, starts Python with isolated-path mode, imports Channel, and calls
  the native `build_info()` ABI. The wheel gate remains `not_run`/false until that
  command succeeds for an artifact.
- `pipeline_cache_key()` defines only the versioned invalidation-key ABI; Channel
  persists no pipeline itself (the driver caches compiled OptiX programs on disk).
- `runtime_diagnostics()` derives the supported SM set from the running build's
  `build_info()["cuda_architectures"]`. Runtime-verified architectures and their
  evidence are listed only in `ci/support-matrix.toml` `[[verified]]`.

The reduced PowerShell gate accepts `-Wheel <built-wheel.whl>`. Omitting it emits a
warning and must not be reported as a passed wheel gate.
