# ADR-040: World provenance and fixed-topology staleness

- **Status:** Accepted
- **Date:** 2026-07-26
- **Kind:** Additive contract change. One new value type, one new host-only
  function, one new request field with a safe default, one new field on
  `CompiledScene`. No native code, no ABI symbol, no kernel, no launch, no
  device work, no allocation, no change to any published number.
- **Related:** ADR-032 (the compact cardinality budget this must not touch),
  ADR-034 (`witwin.core` owns the world and its four version domains, Channel
  owns `compile` and `CompiledScene`), ADR-036, ADR-037 (`row_valid` is the
  sole authority and an invalid row is inert at the input), ADR-038, ADR-039
- **Closes:** Phase-7 work items 1 (version-token half) and 7

## Context

`propagation.consumer` lets a caller discover a topology once and replay it at
new endpoint positions for the rest of a frame, a pulse train, or a symbol
block. `_preflight_reevaluate` checked the request type, the device agreement,
the reference frequency, the fixed-topology capability, the interaction width,
the polarimetric basis, and the AD-frozen inputs.

Nothing compared the compiled scene to the world.

Measured on a one-wall world with a transmitter at the origin and a sink 15 cm
along `+x`, components `{los, reflection}`, depth 1, `f_ref = 77 GHz`:

```text
A. discovery on the static scene
   K=2  depth=[0,1]  delay=[5.003462e-10, 2.618478e-08]
C. structure moved +5 m in y, recompiled
   c0.geometry_version == c1.geometry_version : False
   c0.topology_version == c1.topology_version : True
D. replay the frozen topology on the NEW compiled scene
   K=2  row_valid=[True, False]   reflection delay and |c| exactly 0
E. replay the frozen topology on the STALE compiled scene c0
   K=2  row_valid=[True, True]    full-strength OLD answer, no warning
F. fresh discovery at t=1
   K=1  (the reflection row is genuinely gone)
```

Case D is already correct and is a legitimate production case: the reflection
re-solve reads vertices from the *passed* compiled scene, rigid motion and
deformation preserve face indexing, so the frozen winner label still means
what it meant and the row that stopped existing is published through
`row_valid`. Case E is a silent wrong answer, and the caller has no way to
detect it.

Two facts shape the decision:

1. The four version domains are content hashes over the world tensors, so
   equal versions mean equal world content, which is precisely the condition
   under which a frozen replay is numerically meaningful. Object identity is
   the wrong guard; content equality is the right one.
2. A geometry-version mismatch is not automatically an error (case D), while a
   topology, material, or assignment mismatch always is: those respecify the
   labels - face indices, primitive sequences, material rows - that the frozen
   rows carry.

## Decision

### 1. `CompiledScene.time_s`

`CompiledScene` gains `time_s: float | torch.Tensor | None = None`, set from
`SceneSnapshot.time_s` at compile and `None` for a plain `Scene`. It is
recorded verbatim, so a tensor time keeps its identity and costs no host read.

It is reporting and cross-consumer correlation metadata **only**. It is never
compared and never gates a call, because two instants of a static world are the
same world. It is what a cross-consumer acceptance criterion asserts against
when it needs to say that two results describe the same world state.

### 2. `WorldProvenance`

A new frozen value type in `propagation/consumer/contracts.py`:

```python
WorldProvenance(
    topology_version, geometry_version, material_version, assignment_version,
    time_s=None,
)
WorldProvenance.of(compiled) -> WorldProvenance
provenance.moved_domain(current, *, allow_geometry=False) -> str | None
```

`moved_domain` reports the first differing domain in the fixed order

```text
topology_version, material_version, assignment_version, geometry_version
```

published as `capabilities().world_version_domains`. Geometry is last because
it is the only domain a caller can declare tolerable.

### 3. The token rides the topology

`evaluate` stamps `PropagationTopology.provenance`.
`prepare_fixed_topology` forwards it verbatim through
`PreparedFixedTopology.provenance`. **No call signature changes anywhere**,
which is the whole reason the token rides the topology rather than the prepare
call: an existing caller keeps compiling and keeps working.

`provenance is None` on a hand-built topology, which has no world to be stale
against.

### 4. The staleness rule

`FixedTopologyRequest` gains

```python
world_motion: Literal["frozen_world", "fixed_winner_replay"] = "frozen_world"
```

and `_preflight_reevaluate` applies, before any native work:

| condition | result |
|---|---|
| `provenance is None` | proceed |
| `topology_version` / `material_version` / `assignment_version` mismatch | `ValueError` naming the domain, always |
| `geometry_version` mismatch, `world_motion="frozen_world"` | `ValueError` naming the domain |
| `geometry_version` mismatch, `world_motion="fixed_winner_replay"` | proceed |

Four host integer comparisons. Zero device-to-host copies, zero kernels, zero
synchronizations, zero allocations, no ADR-032 budget impact. This is a
contract failure, so it raises before a result exists; a dead row remains data
published through `row_valid`. The two channels are never mixed.

`"fixed_winner_replay"` is the explicit caller statement that the discrete
winner set is deliberately held fixed while the geometry moves. It is exactly
case D above, which is already numerically correct.

### 5. Explicit rediscovery

```python
consumer.rediscovery_required(compiled, prepared) -> str | None
```

Names the version domain that moved, or `None`. Four host integer comparisons,
no device work, no allocation, callable every frame. `"geometry_version"` is
reported like any other domain; a caller replaying under
`fixed_winner_replay` deliberately ignores that one. This is the explicit
rediscovery half of Phase-7 item 7: poll it, and call `evaluate` plus
`prepare_fixed_topology` again when it fires.

### 6. `CONTRACT_VERSION` 3 to 4

Two new exports (`WorldProvenance`, `rediscovery_required`), one new topology
field, one new request field, two new capability fields, and a call that used
to succeed can now raise. The bump happens once for the whole of Phase 7.

## Limitations, stated and pinned

### A replay is subtractive: rows die, rows are never born

A frozen row that stops existing is published as `row_valid=False`. A path
that comes into existence at the new endpoint or world state is **not**
discovered by a replay and is silently absent. Measured with a wall arriving
from 40 m:

```text
wall far  : K=4, all line-of-sight
wall near : K=3, two line-of-sight rows die, one reflection row is born
replay the far-frozen topology on the near scene:
            K=4, row_valid=[True, False, True, False]
            the reflection row that now exists is simply not present
```

The rows that are published are exactly correct and the batch under-reports.

Phase 7 does not add a birth detector. Every candidate is either a full
discovery (measured 9-40 ms against a 2.3-2.9 ms replay) or a device reduction
plus a host read the ADR-032 budget does not have. Following the precedent
ADR-037 set for the line-of-sight limitation, the limitation is documented in
the `FixedTopologyEvaluation` docstring, in `consumer/README.md`, and here, and
is pinned by
`tests/propagation/consumer/test_phase7_world_provenance.py::test_a_born_row_is_absent_from_a_replay`.

The mitigation is caller-owned cadence: poll `rediscovery_required` for a
changed world, and rediscover on a motion-event cadence when the scene can gain
paths.

### An old compiled scene of an unmutated world is not detectable

A `CompiledScene` and the rows discovered on it always agree with each other.
Driving motion through `DynamicScene.at(t)` leaves the source `Scene`
untouched, so holding the old compiled scene and never compiling the new
snapshot leaves a complete, self-consistent world, and nothing in the request
names the instant the caller meant. Channel is never told the caller moved on.

The signal is the new compiled scene: the moment the caller compiles the new
snapshot, the default rule refuses the frozen replay by name. The two
realistic production hazards are therefore covered:

- recompile but reuse the old frozen topology, refused by rule 4;
- mutate the live world in place and forget to recompile, reported by
  `rediscovery_required(..., revalidate_source=True)`.

`revalidate_source=True` recomputes the four domains from the live
`witwin.core` world the compiled scene was built from. That walks and hashes
the world, so it is `O(scene)` host work and is deliberately opt-in and
default-off: putting it in `_preflight_reevaluate` would tax the per-frame
replay loop with Python proportional to scene size, which is exactly the cost
the fixed-topology capability exists to avoid.

## Recorded, not patched

These are `witwin.core` observations found while implementing this decision.
Core is read-only here; they are recorded with repros and not patched.

- **C1, endpoint motion over-invalidates `geometry_version`.**
  `core/witwin/core/scene.py:366-368` folds endpoint states into the dynamic
  geometry hash, so a `DynamicScene` whose structures are completely static but
  whose endpoints move produces a new `geometry_version` per instant and forces
  a full RayD scene and BVH rebuild that `compile` never needed: `compile`
  consumes `SceneSnapshot.structures` only. Measured 2.41 ms per frame of pure
  waste on a two-triangle wall. The Channel-side design routes around it,
  because endpoint motion goes into `EndpointBatch` tensors and never triggers
  a recompile, so it is a cost rather than a correctness issue.
- **C2, `DeformationState` has no velocity descriptor**
  (`core/witwin/core/dynamics.py:116-147`), so deformation velocity has no
  analytic source in Core and production finite differences are forbidden.
- **C3, a scalar material edit does not move `material_version`.**
  `Scene.material_version` hashes `(material_id, material.version)` plus the
  material tensor fields, and `PhysicalMaterial.eps_r` is a Python float on the
  default path. Rebuilding a scene with `eps_r=5.24` changed to `4.0` and an
  unchanged `version` therefore leaves `material_version` identical, so both
  the Channel compile cache and this freshness check regard it as the same
  world. `PhysicalMaterial(version=...)` is the Core-declared way to announce a
  scalar respecification, and the ADR-040 test uses it.

## Evidence

`tests/propagation/consumer/test_phase7_world_provenance.py`, 14 tests:

- a fresh replay passes and `rediscovery_required` returns `None`;
- the topology carries the world it was discovered against, and
  `prepare_fixed_topology` forwards the same object;
- a moved structure raises by default, naming `geometry_version`;
- a declared `fixed_winner_replay` is allowed and correct: `row_valid` is
  `[True, False]` and the dead row publishes exactly zero delay, path length,
  and coefficient magnitude, compared with `torch.equal`;
- a material or assignment mismatch raises under both motion modes;
- an unstamped topology replays, pinning the documented escape;
- `rediscovery_required` performs zero device reads, measured by wrapping
  `Tensor.item`, `tolist`, `cpu`, `numpy`, and `__bool__`;
- the ADR-032 budget is unchanged: `validation_d2h_copies == 1`,
  `validation_d2h_bytes == 4`, `validation_sync_count == 1`,
  `compact_count_d2h_copies == 0`;
- a born row is absent from a replay;
- `time_s` round-trips from snapshot to compiled scene, is `None` for a plain
  `Scene`, preserves a tensor by identity, and gates nothing;
- `compile` applies `RigidMotion` and `DeformationState` from a `DynamicScene`,
  the first Channel test that ever did, closing the untested cross-repo
  coupling between the `("dynamic-source", id(source_scene))` compile-cache
  short circuit and the Core dynamic geometry hash;
- the capability record publishes the world-motion vocabulary and an unknown
  value is refused at request construction.

## Documentation drift corrected in the same change

- The `FixedTopologyEvaluation` docstring said a frozen line-of-sight row is
  never re-tested for visibility. Commit `fb23078` made it re-tested; the
  consumer README already said so.
- `consumer/README.md` said a prepared reflection call re-stages scene-static
  material and face tables host-to-device on every call. `CompiledScene`
  caches them since `fb23078`, and bypasses the cache only when a table
  participates in autograd.

Both are exactly the sentences a reader consults when choosing a rediscovery
policy.
