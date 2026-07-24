# ADR-035: RayD native trace backend selection

- **Status:** Accepted (2026-07-23)
- **Date:** 2026-07-23
- **Kind:** Native trace backend ownership, capability failure, and staged
  acceptance.
- **Related:** ADR-001 (native dispatch), ADR-006 (developer override),
  ADR-009 (native fusion ownership), ADR-023 (direct RayD typed integration),
  ADR-032 (controlled compact cardinality), and
  [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md).

## Context

RayD 0.7.0 keeps the stable `rayd/torch/integration.h` identity, API 6, and
typed `SceneResource` boundary, but its implementation adds a pure-CUDA
tracing path. `rayd::torch::create_scene` constructs the RayD scene with
`TraceBackend::Auto`: RayD selects OptiX when it can create the required
OptiX context and selects its native pure-CUDA trace implementation otherwise.

The prior Channel rule treated missing OptiX as a terminal missing capability.
That rule would reject a complete RayD-owned CUDA implementation even though it
does not cross into Torch, CPU, Dr.Jit, Python, reduced physics, or a second
runtime owner. It also conflicts with the immutable RayD 0.7.0 typed API,
which deliberately keeps backend selection internal to RayD.

## Decision

### RayD remains the sole native tracing owner

Channel accepts RayD `TraceBackend::Auto` for the locked RayD 0.7.0
dependency. RayD remains the only owner of generic scene acceleration
structures, intersection, visibility, reflection/diffraction tracing, and the
native trace implementations selected behind its typed integration boundary.
Channel must not inspect RayD internals, duplicate either implementation,
introduce a second trace registry, or select a backend through an undocumented
environment variable or private symbol.

OptiX is the preferred performance path. When OptiX is unavailable, RayD may
select its full-result pure-CUDA path. Selection occurs during RayD scene
construction and remains owned by that resource; Channel does not switch
backend in response to a later operation failure.

### Native selection is not a forbidden fallback

The accepted selection is between two RayD-owned native CUDA implementations
of the same typed operation and result contract. It does not authorize:

- Torch, CPU, NumPy, Python, Dr.Jit, or finite-difference production compute;
- reduced physics, reduced accuracy, empty success, zero substitution, silent
  truncation, or partial results;
- retrying a failed operation on another backend;
- a second Python extension, dispatcher, scene, or numerical owner; or
- changing ADR-032 compact `O(K)` ownership, row order, or controlled count
  observation.

An operation unsupported by the selected RayD backend must fail its typed
capability validation before that operation launches numerical work or exposes
any output. Channel must propagate the error once and may not retry, catch it
as success, or continue with a partial solver result.

### Evidence is staged at large-module checkpoints

Phase 0A accepts only the immutable dependency and static compatibility
baseline: final tag/SHA, distribution version, source manifest, stable typed
header, build-fingerprint inputs, workflow pins, product identity, and compact
owner continuity. It does not claim full numerical certification of the new
pure-CUDA path.

The Phase 2 large-module checkpoint must cover every Channel solver operation
reachable through each selected native trace path, fail-loud unsupported
capabilities, result schema, exactness/tolerance policy, row identity/order,
primal/JVP/VJP support, stream behavior, launches, transfers, and peak memory.
The Phase 3 Stage-I release checkpoint must cover end-to-end latency,
throughput, wheel/fingerprint, supported platform/SM rows, and the complete
four-solver acceptance matrix. OptiX remains the reference performance path;
pure CUDA needs its own measured row and may not inherit an OptiX result.

## Consequences

- Channel can bind the final RayD 0.7.0 release without creating an
  out-of-tree backend selector.
- Missing OptiX alone is no longer a Channel capability failure.
- Missing support for a requested operation remains a loud typed failure and
  never authorizes a runtime retry or reduced result.
- The stable integration header hash cannot prove implementation equivalence;
  dependency evidence must record the RayD implementation delta explicitly.
- Phase 0A completion does not weaken or pre-run the concentrated Phase 2 and
  Phase 3 numerical, AD, performance, and release gates.
