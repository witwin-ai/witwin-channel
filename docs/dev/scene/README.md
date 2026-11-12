# Scene owner

## Ownership

`scene` owns scene/endpoint/structure models, Mitsuba loading, compilation,
canonical geometry/material/assignment stores, cache invalidation, lazy
scattering resources, and typed RayD resource lifetime. `core.scene` retains
the stable public scene identity.

The package is three modules, one per artifact category, and each is the single
owner of what it holds:

| Module | Owns |
|---|---|
| `scene.compiler` | `compile`, the compile registry, `CompiledScene`, the geometry/material/assignment stores and their tensor validation, and the endpoint tensor exports every solver reads off a bound scene |
| `scene.endpoints` | the endpoint views over one Core antenna state, `SolverScene` and `bind_solver_scene`, antenna pattern/array/weight response, axis-aligned receiver-grid geometry, the `transmitter_polarizations_f32` / `receiver_polarizations_f32` endpoint tensors, and the plan 07 AD-2 scene-leaf geometry seam |
| `scene.resources` | the typed RayD scene lifetime (`RayDSceneResource`, `build_scene_from_structures`, and their two native facades), the diffraction `EdgePolicy` and the scene-policy edge refinement, the lazy Kirchhoff and phase-screen resources, and the compile-time construction of the `KirchhoffTable` and `PhaseScreenRuntime` those resources are made of |

The RayD lifecycle owner is exclusively `scene.resources` and has no
compatibility re-export.

## Public entry points

`scene.__init__` exports `compile`, `clear_compile_cache`, and `CompiledScene`.
The logical world model is owned by `witwin.core` and is not re-exported here.
Stores, endpoint views, and native resources are internal.

## Dependency rules

Scene may depend on runtime resources, material encoding, the layer-stack
evaluation its compile-time Kirchhoff build calls, and narrow topology/geometry
primitives needed during compilation. It
must not import a solver or solver pipeline. Typed resources and mutable caches
remain private; propagation topology and geometry kernels may not import scene
back.

Inside the package the dependency runs one way: `compiler` imports `endpoints`
and `resources`, and neither imports `compiler` at module scope. That is load
bearing, not cosmetic. An edge from `endpoints` back to `compiler` would pull
the whole compile-time dependency set - materials, topology kernels,
penetration - into a cold import of every consumer of the endpoint views, the
solver-neutral propagation consumer among them. `require_compiled` resolves
`CompiledScene` inside the call, and `resources` annotates the stores under
`TYPE_CHECKING`, for that reason.

## Endpoint store

`bind_solver_scene` is host-only: it validates the endpoints and returns a `SolverScene`
whose one `EndpointStore` is built by `endpoint_store(scene)` on first use inside the
solve, so a solve refuses its memory budget before any CUDA or tensor work. The store holds
every endpoint tensor a solve reads, on the CUDA device: transmitter positions,
powers and world-frame polarizations, flattened receiver positions (a
`ReceiverGrid` through the native grid generator), per-receiver polarizations
and a per-row polarization view in which a grid is one broadcast row. The store
is built in batched operations - one stack per field, one batched orientation
and normalization per (device, dtype, orientation kind) group, one host-to-device
copy of a packed float32 buffer - and cached on the `CompiledScene` under
`endpoint_token(source)`. The token is host-only: per endpoint its identity and,
for each tensor leaf, the tensor identity, `_version` and `requires_grad`
(snapshot rigid motions likewise). It reads no tensor data and is meant to key
any endpoint-only cache. A binding whose endpoint leaves participate in AD
(reverse `requires_grad` or a forward-mode tangent) builds the store per solve
and does not cache it, so the leaves stay on the graph through one stack.
Validation of endpoint values runs when a store is built, not on every solve.
It refuses non-finite values and coincident endpoints: a point receiver or any
`ReceiverGrid` sample within `coincidence_tolerance_m` of a transmitter (the
larger of wavelength/(4*pi) at the compile frequency and `2^-20` times the
coordinate magnitude), named by receiver, grid cell and transmitter. The grid
test is analytic (an index box around the transmitter's in-plane grid
coordinates) and never materializes the grid. CUDA-resident coordinates are
not read. See `docs/dev/standards/adr-067-coincident-endpoints.md`.

The batched polarization reproduces the per-endpoint `rotation @ unit`
product bit for bit: the 3x3 matrix-vector product sums `(r1*u1 + r2*u2) + r0*u0`
on the host BLAS and `(r0*u0 + r2*u2) + r1*u1` on CUDA, and the batched code
uses that per-device order.

`endpoint_store(scene)` returns the bound store (building it once per `SolverScene`), or
builds an uncached one when a caller rebinds endpoints (explicit-array expansion binds its own; a BDPT
sub-scene over one grid builds one per call). The module-level helpers
(`transmitter_positions`, `receiver_positions`,
`transmitter_polarizations_as_stored` in `compiler`;
`transmitter_polarizations_f32`, `receiver_polarizations_f32`,
`receiver_polarization_rows`, `transmitter_positions_ad`,
`receiver_positions_ad` in `endpoints`) all read the store.

The two same-named `transmitter_polarizations` owners the audit reported are
resolved by name rather than by unification.
`scene.endpoints.transmitter_polarizations_f32` is the float32 store field;
`scene.compiler.transmitter_polarizations_as_stored` keeps the dtype the scene
stored. Both are live and each has its own callers.

`scene.resources` also holds the compile-time Kirchhoff/phase-screen
construction that used to sit in a root `scattering` module. That module read
like the scattering owner while `interactions.scattering` owns per-solve path
evaluation, and everything it held was compile-time resource construction this
module already cached. The float64 NumPy build sits behind an explicit banner
inside the file: the CPU-compute policy line is drawn there, and nothing below
it may grow into per-solve physics.

## Compile cache

`compile` keys a runtime on Channel's own state of the source world, not on the
Core version domains, because those also count the scene's own antennas and no
compiled resource depends on them (solvers read endpoints from the source when
they bind it; the consumer takes them from its endpoint batches). The state is
read from host attributes only: object identity, scalar fields, and per tensor
its identity, `_version`, storage address and `requires_grad`, split into the
same four domains as Core (topology, geometry, material, assignment) minus the
endpoints. A warm hit therefore costs one walk of those attributes and never
hashes the world. The Core versions are still read when a new runtime is built,
for provenance.

- The RayD scene, geometry tables, penetration diagonals and runtime structures
  are shared by every runtime whose topology and geometry state are equal;
  materials by material state and frequency; assignments by topology,
  assignment and material IDs.
- The registry holds one runtime per (source, frequency, device, trace
  backend). A recompile of an edited source replaces all of its runtimes at
  once. Least recently used runtimes are dropped above `_CACHE_BYTE_LIMIT`
  (1 GiB of device memory, each shared resource counted once; the newest
  runtime is always kept) or `_CACHE_ENTRY_LIMIT` (64) runtimes.
- When only vertex values change on the same geometry objects with the same
  faces and UVs, the first such compile rebuilds the RayD scene with the
  changed meshes created refittable and later compiles refit them in place
  through RayD's typed `update_mesh_vertices` + `sync_scene`. Building
  refittable on the first change, not on the second, costs nothing extra (that
  compile rebuilds anyway) and confines the uncompacted refittable GAS to the
  meshes that actually moved. Every runtime on a refit scene came from the same,
  now edited, objects and is dropped; `RayDSceneResource.generation` lets a held
  one report itself stale. A refit scene traces bitwise like a fresh build
  (RayD parity test); only the primitive reported for an exact tie between
  coincident triangles may depend on the BVH. Snapshots with per-instant
  motion or deformation objects rebuild instead of refitting.
- `CompiledScene.stale_domain()` names the domain that changed since compile
  (or `geometry_version` after a refit); the consumer's freshness check and
  `rediscovery_required(revalidate_source=True)` use it.

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
ledger is archived in Git history (`docs/dev/archive.md`).
Store/material ABI changes require a versioned migration; dependencies must
satisfy the import graph.
