# Phase 9 deployment evidence status

Phase 9 distinguishes declarations and executable checks from verified deployment
evidence:

- `benchmarks.harness.measure_cold_import()` measures a fresh interpreter against
  the source tree/build directory. It is not wheel evidence.
- `ci/wheel_smoke.py <wheel.whl>` installs one already-built wheel into a temporary
  target, starts Python with isolated-path mode, imports Channel Native, and calls
  the native `build_info()` ABI. The wheel gate remains `not_run`/false until that
  command succeeds for an artifact.
- `pipeline_cache_key()` defines only the versioned invalidation-key ABI.
  `PIPELINE_CACHE_IMPLEMENTED` is false; no persisted or in-memory pipeline cache
  is claimed.
- the committed SM matrix mirrors CMake code-generation declarations. Every row is
  `declared_unverified` with an empty evidence list until the wheel is executed on
  that architecture and the resulting artifact is retained.

The reduced PowerShell gate accepts `-Wheel <built-wheel.whl>`. Omitting it emits a
warning and must not be reported as a passed wheel gate.
