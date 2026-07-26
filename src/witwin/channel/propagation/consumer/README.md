# Propagation consumer boundary

## Ownership

`propagation.consumer` is the stable, solver-neutral contract that packages
outside Channel use to obtain propagation paths. It owns the request and result
value types, the declared conventions, the capability record, and the
orchestration that binds an explicit endpoint batch to the canonical enumerated
and compact owners.

It owns no physics. Discovery, geometry, field evaluation, and compaction stay
with `propagation.enumerated`, `propagation.geometry`, `propagation.fields`,
and `propagation.topology`. The consumer adds no second compaction, no count
observation, and no Torch gather; it reuses the ADR-032 native compact stage
and reports what that stage cost under `PropagationDiagnostics`.

The consumer is solver-neutral in both directions: it never imports `path`,
`deterministic`, `montecarlo.basic`, or `montecarlo.bdpt`, and those solvers
continue to use the internal contracts directly rather than routing through
here.

## Vocabulary

`contracts.py` is the single source of truth for the accepted values. Each
dimension is a `Literal` alias with a matching frozen set:

| Dimension | Alias | Values |
|---|---|---|
| Component | `PropagationComponent` | `los`, `reflection`, `transmission`, `diffraction` |
| Response | `PropagationResponse` | `scalar_transport`, `complex3_transport`, `polarimetric_transport` |
| Topology | `PropagationTopologyMode` | `discover` |
| AD | `PropagationAdMode` | `none`, `jvp`, `vjp` |

Scattering is deliberately absent. The existing scattering result is incoherent
power appended out of canonical order; publishing it here would misrepresent it
as field transport. It can only join once it has a coherent contract, a single
compact owner, and AD evidence.

## Discovering what is supported

```python
from witwin.channel.propagation import consumer

caps = consumer.capabilities()
assert "reflection" in caps.components_for("complex3_transport")
assert "vjp" in caps.ad_modes_for("scalar_transport")
```

`capabilities()` returns a frozen record and never runs a solve, so a caller can
check a component/response/AD combination before building a request rather than
discovering the answer from a rejected call.

The record splits discovery from reevaluation on purpose.
`response_components` and `response_ad_modes` describe `evaluate`, where
`polarimetric_transport` is still line-of-sight only. What a frozen topology
can be reevaluated into is `fixed_topology_components` crossed with
`fixed_topology_responses`, which is where reflection and the polarimetric
operator are available.

## Reevaluating a frozen topology

`reevaluate` takes either form of frozen rows.

A raw `PropagationTopology` is the zero-interaction fast path: it keeps the
fused native gather, its all-or-nothing validation, and its
one-copy/four-byte/one-synchronization budget. It serves `scalar_transport` and
`complex3_transport`.

Anything that carries interactions goes through `prepare_fixed_topology` first:

```python
prepared = consumer.prepare_fixed_topology(discovered.paths.topology)
for frame in frames:                       # new endpoint positions per frame
    result = consumer.reevaluate(compiled, consumer.FixedTopologyRequest(
        sources=frame.sources, sinks=frame.sinks,
        reference_frequency_hz=f_ref, topology=prepared,
        response="scalar_transport", ad_mode="jvp"))
```

Reflection field transport takes one uniform interaction depth per native
launch, so the frozen rows have to be partitioned by `(component, depth)`.
That partition depends only on the frozen topology, so it is computed once
here and reused. **`prepare_fixed_topology` synchronizes.** Its cost is
recorded on the handle as `prepare_d2h_copies` / `prepare_d2h_bytes` /
`prepare_synchronizations` and is deliberately kept off the per-call
diagnostics: calling it every frame gives up the entire point of holding a
topology fixed.

Preparation is also where `fixed_topology_components` is enforced. A frozen
component with no fixed-topology owner is rejected there, by name, before any
solve.

## Per-row validity

A frozen reflection row is a face sequence, not a fixed point in space. At new
endpoint positions its stationary point moves and can leave its facet or become
occluded. That is a correct, complete answer - the path does not exist at these
endpoints - so it is published per row rather than failing the batch:

```python
alive = result.row_valid            # bool[K] on device, frozen row order
```

`row_valid` is `None` when every published row is valid by construction, which
is the case for the line-of-sight route. When present it is **the sole
authority for the components it covers**: a geometrically valid row may
legitimately carry a zero coefficient (a cross-polar or grazing null), so
validity can never be inferred from the payload. An invalid row carries exact
zeros in its transport, path length, delay, direction, and interaction tables,
and contributes exactly zero to every gradient.

It covers exactly `capabilities().fixed_topology_row_validity_components`,
today `{"los", "reflection"}`. A frozen reflection row dies when its
stationary point leaves the facet or a leg becomes occluded. A frozen
line-of-sight row is re-tested with the same native visibility gate discovery
applies to LoS candidates, so a sink that moves behind a wall publishes
`row_valid=False` and exact zeros rather than a stale full-strength answer.
The raw zero-interaction route remains version-1 surface and does not carry a
mask; on that route, rediscover when blockage matters.

This does not weaken the all-or-nothing rule. Capacity, ABI, contract, and
device failures still raise before a result exists; only geometric
non-existence is published as data.

### Replay is subtractive: rows die, rows are never born

A frozen row that stops existing is published as `row_valid=False`. A path that
comes into existence at the new endpoint or world state is **not** discovered
by a replay and is silently absent from the batch. The published rows are
exactly correct and the batch under-reports.

There is no birth signal, by design: every candidate detector costs either a
full discovery or a device reduction plus a host read the ADR-032 budget does
not have. A caller whose scene can gain paths owns the rediscovery cadence.
ADR-040 records the limitation and a test pins it.

### Forward mode

`ad_mode="jvp"` on the prepared route requires the endpoint position tensors to
carry only a forward tangent; `requires_grad` is not required:

```python
with torch.autograd.forward_ad.dual_level():
    dual = torch.autograd.forward_ad.make_dual(positions, velocity)
```

Geometry liveness is decided at the caller-facing wrapper, where a forward
dual is still visible, and passed into the shared field companions explicitly
(ADR-038). `path_length_m`, `delay_s`, and the transport all carry tangents
from a forward-only dual, and the `delay_s` tangent is what a Doppler
`delay_rate` reader consumes. The older requires_grad-plus-dual convention
keeps working unchanged.

### Per-call cost beyond the published budget

`validation_d2h_copies` counts device-to-host reads and is one. The
scene-static vertex and material tables a prepared reflection call needs are
staged once per `CompiledScene` and cached on it
(`CompiledScene.fixed_reevaluation_tables`), so a primal per-frame replay pays
the staging once rather than per call. The cache is bypassed whenever a table
tensor participates in autograd, because a cached graph node would be freed by
the first backward; a differentiable-material or differentiable-mesh loop
therefore pays the staging every call, at parity with what a discovery solve
pays per solve. ADR-037 names and measures the uncached cost.

## World provenance and staleness

`evaluate` stamps the four `witwin.core` version domains of the compiled scene
onto the topology it publishes, `prepare_fixed_topology` forwards the stamp
verbatim, and `reevaluate` compares it before any native work: four host
integer comparisons, no device work, no allocation, no synchronization.

```python
provenance = discovered.paths.topology.provenance   # WorldProvenance | None
moved = consumer.rediscovery_required(compiled, prepared)   # str | None
if moved is not None:
    prepared = consumer.prepare_fixed_topology(
        consumer.evaluate(compiled, request).paths.topology
    )
```

A mismatched `topology_version`, `material_version`, or `assignment_version`
always raises: those respecify the very labels the frozen rows carry, so no
replay of them can mean anything. A mismatched `geometry_version` raises too,
unless the request declares `world_motion="fixed_winner_replay"`. That
declaration is the caller's statement that the discrete winner set is
deliberately held fixed while the geometry moves, which is a legitimate and
already-correct case: a rigid motion or a deformation preserves face indexing,
the reflection re-solve reads vertices from the passed compiled scene, and a
row that stops existing is published through `row_valid`.

A topology with `provenance is None` is hand-built and has no world to be
stale against, so it replays unchecked. That escape is pinned by test so it
cannot widen to cover a discovery-produced topology.

`CompiledScene.time_s` carries the `SceneSnapshot` instant a scene was
compiled from, or `None` for a plain `Scene`. It is reporting and
cross-consumer correlation only. It is never compared and never gates a call,
because two instants of a static world are the same world.

### The one staleness class the versions cannot see

A compiled scene and the rows discovered on it always agree with each other, so
mutating the live world in place after compilation leaves the pair internally
consistent and the freshness check silent. Pass `revalidate_source=True` to
`rediscovery_required` to recompute the four domains from the live
`witwin.core` world and catch it. That walks and hashes the world, so it is
`O(scene)` host work: poll it on a motion-event cadence, never inside a replay
loop.

## Slot batching: a whole frame in one call

A caller who needs the same frozen rows at `T` world instants declares
`slot_count=T` instead of calling `reevaluate` `T` times. The frozen rows and
both endpoint batches are then `T` slots stacked slot-major, and the pairing
law becomes block diagonal (`PropagationConvention.slot_pair_layout`):

```python
replicated = consumer.replicate_over_slots(
    prepared, slot_count, source_count=S, sink_count=K   # PER-SLOT counts
)
result = consumer.reevaluate(compiled, consumer.FixedTopologyRequest(
    sources=stacked_sources, sinks=stacked_sinks,        # [T*S, 3], [T*K, 3]
    reference_frequency_hz=f_ref, topology=replicated,
    response="scalar_transport", ad_mode="jvp", slot_count=slot_count))
```

Slots never cross-pair, so `pair_count = slot_count * S * K` is linear in the
slot count rather than the `(T*S) x (T*K)` outer product; inside one slot the
sink-major layout above is preserved exactly. One launch per bucket, one
four-byte validation copy, and one synchronization cover the whole set,
whatever `T` is - which is the point, because a Python loop over instants keeps
every call inside its own ADR-032 budget while multiplying the budget of the
frame by `T`.

`source_count` and `sink_count` are the per-slot endpoint counts and are
required, not inferred: an endpoint that publishes no frozen row never appears
in `source_index`, so the largest index a topology carries is not an endpoint
count.

`slot_count > 1` requires a `PreparedFixedTopology`. The raw zero-interaction
route builds its pair segmentation inside the native gather over the full outer
product and refuses slot batching by name rather than approximating it; a
LoS-only prepared topology reaches the same field owner and additionally
re-tests visibility.

All slots share one `CompiledScene`, so a slot set is one structure-geometry
epoch - one frame, one pulse train, one symbol block. Structure motion is a new
call with a new compiled scene, not another slot.

## Time-varying impulse response

`delay_s` plus the transport plus `pair_offsets` plus `row_valid` already IS an
impulse response per pair. `evaluate_time_varying` is the time axis over it,
and nothing else:

```python
cir = consumer.evaluate_time_varying(compiled, consumer.TimeVaryingRequest(
    sources=stacked_sources, sinks=stacked_sinks,
    reference_frequency_hz=f_ref, topology=prepared,   # PER-SLOT topology
    times_s=torch.arange(T, dtype=torch.float64) * dt,
    response="scalar_transport", ad_mode="jvp"))

cir.delay_s                    # [T, K]
cir.transport.coefficient      # [T, K]
cir.row_valid                  # [T, K], still the sole authority
cir.pair_offsets               # per slot, frozen: the same segmentation every T
```

Every published tensor is a view over the storage one replay produced. The
result carries the diagnostics of that one call, so the ADR-032 budget of a
whole `T`-instant frame is readable in one place.

`times_s` labels the slots. It is never differenced or integrated: a delay RATE
is the ADR-038 forward tangent on the endpoint positions, and a production
finite difference is forbidden.

Being labels, they are also never reconciled against the world. Endpoint motion
legitimately runs many instants against one compiled scene, so a gate on
`times_s == compiled.time_s` would refuse the normal case; the consequence is
that labelling slots `t = 1, 2` while the structures still stand at their
`t = 0` pose proceeds silently, with every row valid. Keeping the labels inside
one structure-geometry epoch is the caller's obligation, and
`CompiledScene.time_s` is published so that obligation is checkable rather than
assumed.

`evaluate_time_varying` does not compile scenes. The caller passes one
`CompiledScene` per structure-geometry epoch, which keeps scene lifecycle out
of the consumer and keeps a moving-structure sequence honest: `T` epochs are
`T` compiles and `T` calls.

## Rediscovery cadence

| Tier | What runs | Cadence |
|---|---|---|
| 0 - session freeze | `compile` + `evaluate` + `prepare_fixed_topology` | once per topology epoch |
| 1 - motion event | `evaluate` + `prepare_fixed_topology` when `rediscovery_required` fires, or on a declared cadence for the birth gap | per motion event, never per pulse |
| 2 - inner loop | one slot-batched `reevaluate` / `evaluate_time_varying` per frame | per frame, pulse train, or symbol block |

Measured on the fixture geometry: discovery is 9.1 ms at 2 x 2 pairs, 14.0 ms
at 4 x 16 and 39.6 ms at 16 x 128, and it grows with the endpoint count.
`prepare_fixed_topology` is 0.7-2.5 ms and synchronizes. A batched replay is
~2.2 ms for the whole frame and is flat in the slot count: 128 rows and 131072
rows cost the same, because replay is launch-bound rather than work-bound. That
asymmetry is the whole argument - tier 1 is affordable per motion event and
never per pulse, and tier 2 is never a Python loop.

## Source amplitude

`sources.powers_w` is the declared transmit power of each source endpoint, and
under ADR-039 it reaches what you read. `ScalarTransport.coefficient` and
`Complex3Transport.field` carry `sqrt(powers_w)` of the row's own transmitting
source, so they are transported field values: projecting the complex3 field
onto the receive polarization reproduces the scalar coefficient, and a power or
gain is the squared magnitude of either.

`JonesTransport` does not. A complex `2 x 2` polarization-basis map is not a
transported field; the fused native LoS Jones owner takes no power input at
all. Apply the amplitude to your own source-basis excitation instead.

The amplitude is applied by a native owner
(`field_source_amplitude_scale`, using the same expression the transport
kernels use for `path_field`). The scalar response reads the excited output
the transport launch already produced, so it costs nothing; the complex3
response pays one elementwise launch.

## Polarimetric transport

`polarimetric_transport` publishes the complete complex 2 x 2 operator from the
source transverse basis to the sink transverse basis, with
`matrix[k, i, j]` the response of sink component `i` to source component `j`.

Through a reflection path the operator is composed, not fused: the native
transport is linear in the transmit polarization and in the receive
polarization, so exciting it once per source basis vector and projecting each
response onto both sink basis vectors recovers all four entries exactly. Both
transverse bases come from the native endpoint-basis owner, because a
reflection row launches on one direction and arrives on another and a basis
that is not transverse to its own leg is silently shortened by the projection.

Both bases are primal-only by contract: they reach the native companions as the
transmit and receive polarization, which reject gradients on them. Endpoint
positions, interaction geometry, materials, and frequency are differentiable.
`ad_mode="none"` on `evaluate` uses the single-launch fused native operator;
an AD mode uses the composed route, and the two agree bit for bit.

## Where validation happens

`PropagationRequest` and `FixedTopologyRequest` validate their own structure at
construction: endpoint roles and device agreement, component membership, the
three vocabulary fields, `max_depth` range, and `max_paths` positivity.

Checks that need the compiled scene stay in `evaluate` and `reevaluate`:
reference-frequency match, response/component and response/AD compatibility,
polarimetric basis presence and its primal-only contract, fixed-topology
interaction width, the smooth-scene requirement of the reflection route, and
fixed-topology AD input restrictions. All of them raise before any native work
runs; there is no partial result.

## Result shape

`PropagationPathBatch` publishes actual compact `K` rows, never provisioned
capacity. `pair_index` and `pair_offsets` are the CSR-style segmentation that
the owning native stage produces, in sink-major/source-minor order.
`path_count` is the host-visible `K`.

`PropagationGeometry` carries one interaction tensor pair,
`interaction_positions_m` and `interaction_normals`, both shaped
`(K, depth, 3)`. There is no separate first-interaction field; slice column 0.

## Conventions

`PropagationConvention` is returned with every result and states the pair
layout, units, phasor, time dependence, coefficient reference, Complex3 basis,
and Jones mapping. Its phasor and time-dependence strings come from
`witwin.channel.constants`, which is the package-wide owner of that convention.

## Frequency offsets

There is no frequency-offset input, and a coefficient is always reported at the
compiled reference frequency. Shifting off that frequency under the narrowband
approximation is a post-multiply the caller owns:

```python
# convention.narrowband_frequency_offset_law
#   H(f_ref+df) = C(f_ref)*exp(-j*2*pi*df*delay_s)
shifted = coefficient * torch.exp(
    -2j * torch.pi * offset_hz * paths.geometry.delay_s
)
```

`delay_s` is published per row for exactly this, and
`coefficient_reference` confirms the coefficient already carries the
reference-frequency phase. The law is stated on `PropagationConvention` rather
than left implicit because its sign follows the frozen phasor and
time-dependence, which is easy to get wrong when rederived.

This holds only while the coefficient is constant across the offset.
Re-evaluating dispersive material response per frequency point is a different
operation - N field evaluations rather than a post-multiply - and would arrive
as its own capability with a `CONTRACT_VERSION` bump, not as a parameter on
this one.
