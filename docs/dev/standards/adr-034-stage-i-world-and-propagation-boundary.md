# ADR-034: Stage-I world and propagation boundary

- **Status:** Accepted and active for Stage I; Phase 0A locked RayD 0.7.0,
  Phase 1 established Core world contracts, Phase 2 switched Channel owners,
  and Phase 3 activates consumer contract version 1
- **Date:** 2026-07-23
- **Kind:** Ownership, public API, frequency, material, invalidation,
  propagation convention, AD, cardinality, and release decision
- **Related:** ADR-003, ADR-007, ADR-008, ADR-023 through ADR-028,
  ADR-032, ADR-033, and the Channel/Radar architecture plan

## Context

Channel already has a production compiler, `CompiledScene`, typed
`RayDSceneResource`, RayD scene/BVH ownership, geometry/material/assignment
stores, lazy scattering resources, four solver callers, and an accepted native
compact `O(K)` boundary. The missing architecture is not another propagation
runtime. It is a shared logical world contract, a stable compile facade, and a
solver-neutral propagation consumer boundary.

The current logical Channel `Scene` also owns frequency, compiler caches, and
runtime construction. Core has reusable geometry and basic material/structure
values, but no `Scene`, snapshot, stable IDs, dynamics, antenna state, or
granular version contract. Keeping these owners in parallel would force Radar
and future solvers to maintain adapters and duplicated physical-material
models.

This ADR freezes the Stage-I target before implementation. Phase 0A has pinned
the final clean RayD/`rayd-torch` 0.7.0 release SHA and artifacts. RayD 0.7.0
is a dependency baseline, not a reopening of completed Plan 13 architecture.

## Decision

### Logical world and runtime ownership

`witwin.core` is the only owner of:

- `Scene`, `SceneSnapshot`, `Structure`, and stable structure/object/primitive
  identity;
- solver-neutral geometry and physical-material specifications;
- logical material and surface-assignment identity;
- logical phase-screen descriptors and height/correlation state;
- logical transmitter, receiver, and antenna state;
- trajectory, rigid motion, deformation state, and
  `DynamicScene.at(time_s)`;
- topology, geometry, material, and assignment version tokens.

Core world-contract modules do not own or import:

- carrier or reference frequency;
- Channel, Radar, or RayD;
- BVHs, native resources, native handles, stores, compiled records, or caches;
- propagation results, solver tapes, or failure transactions;
- Radar target or RCS properties.

`witwin.channel` remains the only owner of:

- `compile(scene_or_snapshot, *, reference_frequency_hz=...)`;
- `CompiledScene` and its cache registry;
- RayD scene/BVH construction and the typed `RayDSceneResource` lifetime;
- geometry, material, and assignment GPU stores;
- material ABI encoding, CSR layout, resident scattering resources, and the
  numerical facade;
- propagation topology discovery, field transport, compact selection and
  packing, error observation, solver accumulation, and public results.

RayD remains the only numerical source owner selected by ADR-023 through
ADR-028 for shared complex, Fresnel, layer-stack, Jones, UTD, scattering, and
generic geometry runtime families. Stage I does not create a shared
RF/geometry static library and does not move numerical source into Core or
Radar.

### Public API migration

The owner move is an intentional breaking migration:

- Channel root `Scene`, `Structure`, logical material, and logical endpoint
  exports must resolve directly to approved `witwin.core` contracts.
- The old Channel logical implementations and compatibility modules are
  deleted in the same Phase-2 change as the four solver caller switch.
- No compatibility facade, duplicate implementation, pickle-only module-name
  rewrite, or fallback route is retained.
- `CompiledScene` remains a Channel runtime type and is never re-exported by
  Core.
- Internal `EvaluatedPaths` remains internal under ADR-007. The new consumer
  interface does not expose or alias its defining modules.

ADR-003 and `ci/public-api-snapshot.json` are updated atomically during the
Phase-2 owner move and again during the Phase-3 consumer publication. A public
snapshot change before the corresponding implementation switch is forbidden.

### Frequency and compiled-material contract

Core `Scene` and `SceneSnapshot` contain no frequency.

Channel compilation has the single production signature:

```python
witwin.channel.scene.compile(
    scene_or_snapshot,
    *,
    reference_frequency_hz,
) -> CompiledScene
```

The reference frequency is a finite positive scalar and is part of the
Channel cache key together with Core topology, geometry, material, and
assignment version tokens and the relevant runtime/device identity.
`CompiledScene` freezes:

- `reference_frequency_hz`;
- the material ABI version;
- all frequency-evaluated material records;
- the set of records whose law is frequency-dependent.

`PropagationRequest.reference_frequency_hz` must exactly match the compiled
reference-frequency identity before any native compute, allocation, partial
result, or tape creation. A mismatch fails loudly. Stage I does not silently
recompile and does not replay material laws on the host.

`frequency_offsets_hz` is accepted only by a capability that explicitly
supports fixed-topology offset evaluation. For such a nondispersive/frozen
record capability, the reference coefficient is advanced with the documented
delay convention. A frequency-dependent material record or unsupported
frequency tangent fails before compute.

The current `+0.1%` probe used to infer frequency dependence is baseline
behavior, not the final logical contract. Phase 2 must replace inference with
the explicit Core dispersion capability when available; it may retain a probe
only as a validation assertion, never as the owner of material semantics.

### Physical-material and assignment contract

Core `PhysicalMaterial` expresses the complete logical input needed by current
Channel behavior:

- relative permittivity and permeability;
- electric conductivity;
- thickness;
- ordered material layers;
- front and back roughness;
- dispersion law and its logical parameters;
- gain;
- scattering coefficient;
- cross-polarization discrimination coefficient;
- stable material identity and material version.

Core also owns stable logical assignment identity and the logical phase-screen
descriptor, including height, scaling/offset, realization identity, mode,
quadrature tolerance, and correlation parameters.

Channel ABI v3 remains the compiled owner. The Phase-2 mapping is one-way:

```text
Core PhysicalMaterial / assignment
        -> Channel ABI v3 records and stores
        -> Channel/RayD numerical facade
```

There is no reverse adapter and no second Channel logical material model.
Continuous tensor leaves are not converted to Python scalars, cloned,
detached, or moved to CPU by Core.

Current Channel ABI v3 consumes front-side roughness only. Core nevertheless
preserves both sides. Until a separately accepted ABI/numerical change adds
back-side transport, Channel compilation must fail loudly when a distinct
back-side roughness would be required; it must not silently discard or
silently activate it.

### Stable identity and invalidation

Core identity is stable within the authored world and survives snapshots.
Container order is not identity. Channel must use Core IDs when constructing
structure, primitive, face, material, surface, and assignment mappings.

Core exposes four monotonically changing logical version domains:

| Version | Changes represented |
| --- | --- |
| topology | structure/primitive/face membership or connectivity |
| geometry | continuous geometry, pose, or deformation |
| material | physical-material specification or tensor state |
| assignment | material/surface/phase-screen association |

The minimum Channel invalidation matrix is:

| Version change | Required invalidation |
| --- | --- |
| topology | RayD scene/BVH, geometry store, assignment store, dependent caches |
| geometry | RayD scene/BVH, geometry store, scene diagonals, geometry-dependent caches |
| material | material store and material/scattering caches |
| assignment | assignment store and assignment/phase-screen caches |

A dependency may widen invalidation only when a contract test documents why.
It may not omit a required invalidation. The same snapshot/version/frequency
key maps to one reusable production `CompiledScene`; caches are Channel-owned
and are never written into Core objects.

Continuous tensor mutation participates in version identity through stable
tensor identity and mutation state. Core must not observe values through
`.item()`, NumPy, DLPack, or CPU staging to create a version token.

### Propagation convention

The stable consumer uses SI units:

- positions and path length: metres;
- delay: seconds, `delay_s = path_length_m / c`;
- frequency: hertz;
- direction and polarization bases: unitless world-Cartesian vectors;
- complex values: `torch.complex64` on the same CUDA device as the row batch.

The phasor convention is:

```text
time dependence: exp(+j 2*pi*f*t)
propagation phase: exp(-j k d)
```

For a path `p`, the reference-frequency scalar coefficient `C_p(f_ref)`
includes free-space attenuation, propagation phase at `f_ref`, interaction
coefficients, and source-to-sink polarization projection for unit source
amplitude. A transmitted field value may additionally include source
amplitude. Power/gain values are squared magnitudes of the corresponding
field quantity; they are not interchangeable with complex coefficients.

For an explicitly supported frozen nondispersive offset:

```text
C_p(f_ref + delta_f)
    = C_p(f_ref) * exp(-j 2*pi*delta_f*delay_s[p])
```

This relation is not permission to ignore frequency-dependent material or
antenna behavior.

`Complex3` is a world-Cartesian complex electric-field vector plus propagation
direction. Jones transport is a complete complex `2 x 2` linear operator from
an explicit source transverse basis to an explicit sink transverse basis. A
two-component field state, sidecar, or already receiver-projected scalar is not
Jones transport.

### Consumer cardinality, segmentation, and failure boundary

The Phase-3 consumer result contains actual compact `K` rows. Its canonical row
order is the owning native compact stage order. It provides stable pair
segmentation through native-produced `pair_index` and `pair_offsets`; a Python
or Torch Boolean gather is not an implementation.

The existing ADR-032 cardinality observation is reused. Consumer projection
must not add a second count D2H transfer or synchronization. Equivalent fields
may alias existing compact storage; new fields are produced once by the owning
native producer.

The allowed D2H is limited to small integer metadata needed for
shape/allocation/control at the accepted compact boundary. It may not transfer
physics payload, run CPU physics, introduce a fallback, or propagate
capacity-sized inactive rows.

Channel observes and terminates native/fixed-capacity failure transactions
inside its boundary. A consumer result never exposes a failure state, raw bit,
observer, native handle, lease, scene resource, cache, or solver tape. Failure
publishes no usable partial result.

### AD boundary

Derivatives are defined only for explicitly supported continuous inputs under:

- fixed topology;
- fixed compact row identity and order;
- fixed winner/selection;
- fixed pair segmentation;
- fixed material capability.

Primal, JVP, and VJP consume the same compact rows. Backward/JVP must not rerun
topology discovery, compaction, winner selection, or material compilation.
Discrete changes and unsupported frequency/material/Jones combinations fail
before partial results or tape creation. Native numerical owners retain their
existing tape ownership.

### Runtime and release matrix

The first integrated Stage-I release locks Torch to 2.10.0. Python 3.10 through
3.14 are candidates, not claims, until each row has clean build, import,
native-load, and wheel evidence with the final RayD 0.7.0 artifacts.

The activated Channel matrix remains CPython 3.11 and Torch 2.10.0 only.
`_channel` uses the versioned LibTorch/Python extension ABI and is not a
LibTorch Stable ABI binary. The repository-wide Stable ABI floor of Torch 2.10
applies only to artifacts that actually implement and test that boundary; it
does not turn Channel into one. Core metadata likewise advertises only the
Python row verified for this Stage-I release.

The release matrix must also satisfy the repository-wide wheel policy:

- Linux wheels are built inside a real `manylinux_2_28` environment;
- native SM87 SASS is present;
- declared architectures are distinguished from runtime-verified GPUs;
- the packaged extension fingerprint binds compiler, CUDA, Torch, RayD lock,
  integration header, ABI/binding manifest, and source manifest.

An infeasible Python row requires an explicit breaking support decision before
Phase 1 is released. Metadata must not claim an unverified row. Phase 2 adds an
exact dependency on the Phase-1 Core distribution (`witwin==<released
version>`), not only a namespace import.

### Phase-3 activation record

The stable owner is `witwin.channel.propagation.consumer`, contract version 1.
The public module remains solver-neutral and Radar-neutral. It publishes typed
endpoint/request/path/convention/capability/evaluation contracts, scalar,
Complex3, and complete source-basis-to-sink-basis Jones response contracts,
plus fixed-topology reevaluation. Unsupported component, response, offset, or
AD combinations fail before partial output.

Contract version 1 advertises only LoS, reflection, transmission, and
diffraction. Scattering is rejected before compute because the current
enumerated scattering representation is incoherent power-domain output and
does not satisfy the consumer's coherent transport or canonical pair-major row
contracts.

The consumer reuses the ADR-032 compact owner. Its count-observation and
synchronization delta relative to the corresponding internal evaluation is
zero. Channel terminates failures before constructing the public evaluation;
the public result cannot contain a failure state, observer, native handle,
resource, cache, or tape. ADR-029 remains Superseded, ADR-030 remains Dormant,
and ADR-031 remains Rejected; none gains a consumer caller, capability,
binding-preservation requirement, or release obligation.

### Validation cadence

Small implementation steps run targeted contract and static gates. Full
adversarial and module acceptance runs once after each large module:

1. Phase 1: Core world contracts;
2. Phase 2: Channel compiler and four solver caller switch;
3. Phase 3: propagation consumer and Stage-I release.

Each phase ends in one independent, reviewable final commit. Full
quick/CUDA/nightly/release matrices are not repeated for every edit.

## Consequences

Core becomes a lightweight logical-world distribution while Channel keeps the
heavy propagation runtime. Radar can later consume one shared snapshot and one
Channel-compiled runtime without owning a second BVH or material system.

The migration intentionally removes old Channel logical identities. Existing
Channel users must move construction imports to `witwin.core`; migration notes
and the public snapshot make this break explicit.

Back-side roughness, wideband dispersive offsets, and unsupported AD modes fail
loudly rather than being silently ignored or approximated. Adding those
capabilities requires a separately measured numerical decision.

## Stop conditions

Stop the active phase if it:

- proceeds without the final clean RayD 0.7.0 baseline;
- introduces a second logical Scene/material owner or compatibility route;
- moves native resources, caches, frequency, or Radar RCS into Core;
- changes a RayD/Channel numerical owner without a separate ADR;
- adds CPU physics, payload D2H, a fallback, or a second cardinality
  observation;
- changes compact row identity/order or returns partial results;
- claims an unverified runtime/wheel row; or
- weakens exactness, AD, performance, memory, ABI, or packaging gates to pass.
