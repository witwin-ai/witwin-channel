# Scene owner

## Ownership

`scene` owns scene/endpoint/structure models, Mitsuba loading, compilation,
canonical geometry/material/assignment stores, cache invalidation, lazy
scattering resources, and RayD lifetime. `core.scene` and `core.runtime` are
identity-preserving compatibility facades only.

## Public entry points

`scene.__init__` intentionally exports nothing. The root package exports
`Scene`, `Structure`, `Transmitter`, `ReceiverGrid`, and `ReceiverPoint`. The
frozen `Scene` target remains `core.scene.Scene`, the same class object defined
by `scene.models`. Compile, stores, and kernels are internal.

## Dependency rules

Scene may depend on runtime resources, material encoding, scattering resource
types, and narrow topology/geometry primitives needed during compilation. It
must not import a solver or solver pipeline. Mutable handles/caches remain
private; propagation topology and geometry kernels may not import scene back.

## Numerical and AD contract

Compilation and cache invalidation must preserve tensor storage, device,
ordering, material ABI, geometry identity, and RayD handle lifetime. SI units,
endpoint ordering, face/edge IDs, winding, and UV conventions are contractual.
Boundary moves may not add copies or launches.

### AD contract

Compilation freezes discrete topology and material records at the primal scene
state. Continuous tensor leaves needed by fixed-topology AD retain identity and
their graph; topology-only scalarization is explicit and detached.
Frequency-dependent compiled records reject unsupported frequency AD.

## Forbidden fallback

Production scene construction may not fall back to CPU geometry, a Python ray
tracer, reconstructed material tensors, or a global RayD extension. Missing or
stale native resources fail before solver execution.

## Maintenance

Changing root scene exports or targets requires `ci/public-api-snapshot.json`,
same-object compatibility tests, and a migration note. Frozen scene-kernel
moves update `ci/ops_migration_manifest.json` while preserving the contract.
Store/material ABI changes require a versioned migration; dependencies must
satisfy the import graph.
