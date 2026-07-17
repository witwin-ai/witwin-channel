# ADR-002: RayD dependency lock

Status: Accepted for the modular hardening migration.

## Decision

Local builds may point at a working tree through `RAYD_SOURCE_DIR`, while CI and
release builds validate a fixed RayD commit and ABI fingerprint. Whether RayD
becomes a git submodule is decided separately.

## Context

RayD is a source-linked native dependency, so developers need a fast local
override, but reproducible artifacts require an exact, verified revision. The
lock file records the pinned commit and its expected build identity, and the
CMake configure step plus the lock test reject any drift between the checked-out
RayD and the lock. This keeps developer flexibility without letting an
unverified RayD reach a wheel.

## Implementing artifact

`dependencies/rayd.lock.json` plus the CMake lock validation, gated by
`tests/kernels/test_rayd_lock_cmake.py`.
