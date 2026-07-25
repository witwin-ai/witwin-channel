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
today `{"reflection"}`. **A frozen line-of-sight row is never invalidated.** It
is replayed as pure free-space transport and is not re-tested for visibility,
so a sink that moves behind a wall still reports `row_valid=True` and a
full-strength field even though fresh discovery would drop that row. This is
the shipped version-1 line-of-sight behavior, unchanged. If your scene has
blockers and you need blockage on the direct path, rediscover; the mask will
not tell you.

This does not weaken the all-or-nothing rule. Capacity, ABI, contract, and
device failures still raise before a result exists; only geometric
non-existence is published as data.

### Forward mode

`ad_mode="jvp"` on the prepared route requires the endpoint position tensors to
carry `requires_grad` **in addition to** their forward tangent:

```python
positions = base.clone().requires_grad_()
with torch.autograd.forward_ad.dual_level():
    dual = torch.autograd.forward_ad.make_dual(positions, velocity)
```

Without it the shared native field companions cannot observe a forward-only
tangent and publish `path_length_m` and `delay_s` without one, which would make
a delay-rate reader silently read zero. The route raises rather than answering
partially. See ADR-037 section 8.

### Per-call cost beyond the published budget

`validation_d2h_copies` counts device-to-host reads and is one. It is not the
whole per-call cost: a prepared reflection call also re-stages scene-static
material and face tables host-to-device on every call, at parity with what a
discovery solve pays per solve. ADR-037 names and measures this.

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
