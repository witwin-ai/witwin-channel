# Scene owner

## Ownership

`scene` owns scene/endpoint/structure models, Mitsuba loading, compilation,
canonical geometry/material/assignment stores, cache invalidation, lazy
scattering resources, and typed RayD resource lifetime. `core.scene` retains
the stable public scene identity; the RayD lifecycle owner is exclusively
`scene.kernels.rayd_scene` and has no compatibility re-export.

## Public entry points

`scene.__init__` intentionally exports nothing. The root package exports
`Scene`, `Structure`, `Transmitter`, `ReceiverGrid`, and `ReceiverPoint`. The
frozen `Scene` target remains `core.scene.Scene`, the same class object defined
by `scene.models`. Compile, stores, and kernels are internal.

## Dependency rules

Scene may depend on runtime resources, material encoding, scattering resource
types, and narrow topology/geometry primitives needed during compilation. It
must not import a solver or solver pipeline. Typed resources and mutable caches
remain private; propagation topology and geometry kernels may not import scene
back.

## Numerical and AD contract

Compilation and cache invalidation must preserve tensor storage, device,
ordering, material ABI, geometry identity, and RayD resource lifetime. SI units,
endpoint ordering, face/edge IDs, winding, and UV conventions are contractual.
Boundary moves may not add copies or launches.

### AD contract

Compilation freezes discrete topology and material records at the primal scene
state. Continuous tensor leaves needed by fixed-topology AD retain identity and
their graph; topology-only scalarization is explicit and detached.
Frequency-dependent compiled records reject unsupported frequency AD.

Phase-screen resources are CompiledScene-owned and remain lazy: scenes that
never enter a phase-screen consumer perform no height allocation or validation.
The first consumer atomically caches a typed immutable resource per structure:
mode exclusivity, host structure/material ids, face range and first face,
resident scaled heights, checked UV tensors/triangles, face areas, static
UV-to-world scale, and the RMS-slope applicability guard. Geometry, material,
assignment, and mutation-aware height identity participate in invalidation.
Frequency-, endpoint-, and solver-config-dependent patch subdivision and
visibility remain solve-plan state and are never presented as compile resources.

## Forbidden fallback

Production scene construction may not fall back to CPU geometry, a Python ray
tracer, reconstructed material tensors, or a global RayD extension. Missing or
stale native resources fail before solver execution.

## Maintenance

Changing root scene exports or targets requires `ci/public-api-snapshot.json`,
contract tests, and a migration note. The completed scene-kernel migration
ledger is archived at `docs/dev/audit/phase12-ops-migration-ledger.json`.
Store/material ABI changes require a versioned migration; dependencies must
satisfy the import graph.
