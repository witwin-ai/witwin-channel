# ADR-002: RayD dependency lock

Status: Accepted; package-source discovery amendment accepted 2026-07-22.

## Decision

Local builds may point at a working tree through `RAYD_SOURCE_DIR`. A non-empty
explicit path always wins and an invalid path fails without fallback. When the
path is absent, CMake may locate the selected Python interpreter's unique
`rayd-torch` distribution and consume its passive source-bundle metadata.

Both modes validate the fixed repository, commit, integration API/identity and
header SHA. Package mode additionally requires exact distribution/version and
RECORD ownership, rejects dirty or ambiguous metadata, and recomputes every
file in the lock-pinned full-source manifest. It never imports `rayd.torch` or
scans a conda prefix, site-packages directories, global CMake registries, or
uncontrolled installations. The source is still compiled in Channel's own
CMake/Torch/CUDA graph.

## Context

RayD is a source-linked native dependency, so developers need a fast local
override, but reproducible artifacts require an exact, verified revision. The
lock file records the pinned commit and its expected build identity, and the
CMake configure step plus the lock test reject any drift between the checked-out
RayD and the lock. A normal wheel has no Git metadata, so commit strings and the
integration header alone cannot prove the rest of its `.cu/.cpp` inputs. Lock
schema 2 therefore pins the complete package source-manifest SHA in addition to
the existing Git and ABI identity. This keeps developer flexibility without
letting an unverified RayD reach a wheel.

## Implementing artifact

`dependencies/rayd.lock.json` plus the CMake lock validation, gated by
`cmake/resolve_rayd_source.py`, `tests/kernels/test_rayd_lock_cmake.py`, and
`tests/kernels/test_rayd_package_discovery.py`.
