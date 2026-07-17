# Phase 0 Refactor Baseline

This directory freezes the modular-architecture baseline for commit
`0892d855b27ee851521a181f5158b0bf41091eda`. It is append-only: later
architecture work must compare against these artifacts rather than overwrite
them.

## Environment and build

- Python 3.11.14, Torch 2.10.0, Torch CUDA 12.8
- NVIDIA SM 120; driver and compiler details are in `static/environment.json`
  and `static/build.json`
- RayD `6047089cc7a41661402a02d40c96b9117e03a135`, clean at capture time
- Native binary and CMake cache fingerprints are recorded in
  `static/build.json`
- The only worktree dirt recorded by the collector is the preserved private
  `.claude/` root; its contents are neither captured nor modified

## Verification summary

- Full suite: 965 passed, 1 skipped, 1 xfailed
- AD suite: 199 passed, 1 xfailed
- Acceptance and performance tests: 74 passed
- Coverage run: 937 passed, 1 skipped, 1 xfailed
- Production dependency contract: passed
- Production statement coverage: 82.96%; branch coverage: 64.45%
- Ruff baseline debt: 38 errors, confined to tests and benchmarks; production
  `src/` is clean. Phase 11 must remove this debt without lowering coverage.

The expected xfail is the documented coupled R-D mesh-vertex AD gap in
`tests/ad/test_solver_geometry_ad.py`. The expected skip verifies the non-CUDA
failure path and is not observable on this CUDA host.

## Runtime baseline

The reduced runtime matrix covers Path, Deterministic, MC Basic, and BDPT on
empty-LoS and single-reflection scenes. Every case uses two independent
processes, one warmup per process, and seven steady measurements per process.
All eight cases produced identical semantic result, path-identity, and stable
metadata fingerprints across processes.

The current backend exposes aggregate launch/tape accounting but not a
per-launch name/grid/block/stream ledger. Missing per-launch fields are marked
`unavailable` in `runtime/launch-ledger.json`; they are never represented by
invented zero values.

## Artifact layout

- `static/`: API/schema, binding, import graph, source-body, environment,
  build, pytest, and attached runtime manifests
- `runtime/`: exact solver results, aggregate launch ledger, and 2x7 timing and
  memory distributions
- `coverage.json`: Coverage.py statement and branch baseline
- `validation.json`: machine-readable command outcomes and known debt

The static manifest is complete and includes SHA-256 checksums for every
component. The runtime profile is explicitly reduced; the complete AD,
Munich, propagation-component, wheel, and release matrices remain mandatory
for later phase and release gates.
