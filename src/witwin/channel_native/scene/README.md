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
The first consumer builds a typed immutable resource per structure and publishes
the cache only after the complete build succeeds; this is not a multi-thread
locking guarantee. The resource retains mode exclusivity, true host UV-presence,
host structure/material ids, face range and first face, resident scaled heights,
checked UV tensors/triangles, face areas, static UV-to-world scale, and the
RMS-slope applicability guard. Geometry, material, assignment, RayD wrapper,
and mutation-aware height identity participate in invalidation.
The numeric `id(RayDSceneResource)` key component is only Python wrapper
identity while CompiledScene owns the wrapper; it is not a native pointer,
scene handle, or ABI argument.

This lazy builder caches the same scene-static UV/area/scale/slope calculation
that the realization consumer previously repeated, with the same exception and
numerical order. It is retained Plan-13 resource construction, not a Torch
production-physics owner or fallback. Native ownership of that static build,
including removal of its remaining scalar device read, requires a separate ADR.
Frequency-, endpoint-, and solver-config-dependent patch subdivision and
visibility remain solve-plan state and are never presented as compile resources.

ADR-027 also freezes two scalar penetration policy inputs at compile time.
`CompiledScene.enumerated_penetration_scene_diagonal_m` is the L2 diagonal of
the RayD edge-record vertex bounding box; the Monte Carlo counterpart is the
L2 diagonal of the union of structure bounding boxes. Empty scenes produce
zero. Solver execution consumes these cached host metadata values and must not
repeat scene traversal, read a CUDA scalar, or substitute one policy's
diagonal for the other.

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
