# Stage-I Phase 0 Channel scene maturity audit

Status: complete against Channel commit `fc1680f`; activated by the accepted
RayD 0.7.0 Phase-0A baseline.

## Conclusion

Channel already has the production runtime needed by Stage I. It has one
canonical compile implementation, a typed RayD scene holder, a compiled scene
with owned stores and lazy caches, four real solver callers, and the accepted
native compact `O(K)` path. Phase 2 is an atomic logical-owner, frequency,
cache-key, stable-ID, and caller migration. It is not a rewrite of RayD, BVH
construction, field transport, or compact selection.

The authoritative Channel dependency baseline is:

| Field | Current value |
| --- | --- |
| RayD commit | `49c58c4cb8212f6babb920cc88fb937509826cc5` |
| distribution | `rayd-torch` 0.7.0 |
| integration API | 6 |
| integration identity | `rayd.torch.integration` |
| header SHA-256 | `57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14` |
| source manifest | `e2eb1a7577f906b3ab52e6345b039837228771c8f1582c9f821d0f2bb07d41b4` |
| native trace selection | RayD `Auto`: OptiX preferred, pure CUDA accepted |

The accepted evidence is recorded in
`docs/dev/audit/stage1-phase0a-rayd-0.7.0-dependency-baseline.json`, and
ADR-035 defines the RayD-owned backend-selection boundary. Stage-I
implementation is no longer blocked on the dependency baseline.

## Runtime owners

| Capability | Current owner | Stage-I treatment |
| --- | --- | --- |
| canonical scene compilation | `scene/compile.py::compile_scene` | retain implementation, publish new facade |
| compiled lifetime | `scene/compiled.py::CompiledScene` | retain in Channel |
| RayD scene/BVH | `scene/kernels/rayd_scene.py::RayDSceneResource` | retain typed Channel facade/RayD resource |
| geometry store | `scene/stores/geometry.py` | retain, remap from Core IDs |
| material ABI v3/CSR store | `materials` + `scene/stores/materials.py` | retain, map from Core specification |
| assignment store | `scene/stores/assignments.py` | retain, map from Core assignments |
| Kirchhoff/phase-screen resources | `scene/scattering_resources.py` | retain lazy CompiledScene ownership |
| logical Scene/Structure/endpoints | Channel `scene/models.py` | replace with Core contracts in Phase 2 |
| compile/RayD caches | mutable Channel logical Scene | move to Channel cache registry |
| compact path selection/packing | native `path_compaction.cu` lineage | retain and extend once in Phase 3 |

## Current compiler behavior

`compile_scene` currently:

1. reads frequency from the mutable Channel `Scene`;
2. computes geometry, material, and assignment versions;
3. evaluates material records and a JSON material cache token;
4. reuses or creates a typed `RayDSceneResource`;
5. compiles geometry, material, and assignment stores;
6. stores runtime caches back on the logical `Scene`;
7. returns a `CompiledScene` with owned versions and scene diagonals.

This pipeline is mature, but its logical/runtime ownership is wrong for a
shared world contract. The Phase-2 facade must accept Core Scene/Snapshot plus
explicit `reference_frequency_hz`, and the cache registry must be Channel
owned.

## Gaps to close

### Identity and invalidation

The current scene exposes geometry/material/assignment integers, but has no
topology version, stable structure/object/primitive identity, snapshot, or
dynamics token. Some invalidation is manually bumped and assignment mutation
has no complete owner API.

Phase 2 must consume Core's four version domains and stable IDs. Tests must
cover both required invalidation and forbidden unnecessary rebuilding.

### Frequency

Frequency currently lives on Channel `Scene`, is read at compile time, and is
also read independently by solver pipelines. Frequency-dependent material
records are inferred by a nearby-frequency probe. This creates multiple
observation points and an ambiguous cache contract.

The final boundary is:

```text
Core Scene/Snapshot (no frequency)
    -> Channel compile(reference_frequency_hz)
    -> CompiledScene(reference_frequency_hz, frozen material records)
    -> matching PropagationRequest
```

Every request/compiled mismatch must fail before native compute.

### Material mapping

The current logical model covers layer stacks, geometry mode, front/back
roughness, dispersive and ITU laws, PEC, gain, scattering coefficient, XPD,
phase screens, and surface assignment. ABI v3 currently encodes one roughness
record sourced from the front side. Phase 1 must preserve the complete logical
specification in Core; Phase 2 must use one documented mapping and fail loudly
for a distinct unsupported back-side numerical request.

### Existing host work

The compiler currently creates several ID stores from Python lists/CPU
`torch.tensor` construction and performs controlled compile-time host
observations for scene diagonals and scalar frequency. Phase-2's no-staging
gate means the Core migration may not add NumPy, DLPack, payload D2H, or CPU
physics. Removing all pre-existing metadata construction is a separate
measured optimization unless the new stable-ID mapping naturally replaces it.
The report must not claim these existing paths are already zero.

### Public and compatibility debt

The package root and API snapshot point at old Channel logical types.
Compatibility modules and `__module__` rewrites preserve earlier internal
locations. Phase 2 deletes these atomically with the owner switch and updates
ADR-003, the public snapshot, import graph, migration notes, and feature list.
No compatibility facade remains.

## Real callers

All current production solvers consume the compiled runtime:

- enumerated propagation engine;
- Path;
- Deterministic;
- Monte Carlo Basic;
- Monte Carlo BDPT.

The Phase-2 switch must update all of them together. Path and Deterministic keep
their internal result path; they are not routed through the new public
consumer API. MC/BDPT continue to use only the ADR-008-approved boundary.

## Compact and consumer readiness

Internal `EvaluatedPaths` already provides row-aligned topology, geometry, and
field tensors. It remains internal. The legacy Path projection performs Torch
`nonzero`, `index_select`, and `contiguous` compaction and therefore cannot
become the public consumer implementation.

The native compact owner is mature and remains authoritative. Phase 3 must
extend that owner to produce the public compact view, stable `pair_index`, and
`pair_offsets` without a second count observation. A current two-component
`JonesState` is a field state, not a source-to-sink `2 x 2` Jones transport
operator.

## Runtime matrix decision

The current verified Channel row is CPython 3.11, Torch 2.10, Torch CUDA 12.8,
native CUDA 12.9, runtime SM120. Stage I keeps Torch fixed at 2.10.0.
CPython 3.10 through 3.14 remain candidate rows until each row passes the
Phase-3 clean build/import/native-load/wheel smoke.

Therefore the Phase-0 matrix decision is:

- **go** for architecture implementation against Torch 2.10.0;
- **hold** public Python-range expansion until 0.7.0 artifact evidence;
- **no-go** for claiming any unverified row in package metadata;
- require real manylinux_2_28 Linux builds and native SM87 SASS before the
  Stage-I release.

## Phase-2 migration checklist

- add an exact released `witwin` Core dependency;
- publish the single explicit-frequency Channel compile facade;
- move caches out of Core/logical objects;
- remap RayD/stores from stable Core IDs;
- implement the documented Core material-to-ABI-v3 mapping;
- implement four-token invalidation and reuse;
- switch every solver caller atomically;
- delete duplicate logical owners and compatibility modules;
- update root identity, API snapshot, import graph, migration notes, and
  feature list;
- run one concentrated compiler/four-solver/AD/performance/wheel acceptance.

## Evidence reuse

No-production-delta facts reuse their accepted commit evidence:

- Plan 13 direct typed RayD integration and numerical owners;
- ADR-032 compact recovery, row order, count-transfer ledger, performance,
  memory, and failure behavior;
- current public API snapshot and import graph;
- current four-solver compile/runtime coverage.

Fresh full CUDA/nightly/release evidence is intentionally deferred to the
large Phase-2 and Phase-3 checkpoints.
