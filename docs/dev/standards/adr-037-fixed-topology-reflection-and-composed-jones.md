# ADR-037: Fixed-topology reflection reevaluation with per-row geometric validity, and composed source-to-sink Jones transport

- **Status:** Accepted (2026-07-24, merged to main by owner decision)
- **Date:** 2026-07-24
- **Kind:** Consumer contract, result schema, and AD-boundary decision
- **Related:** ADR-003, ADR-007, ADR-024, ADR-026, ADR-032, ADR-034, ADR-036

## Context

`propagation.consumer` publishes the solver-neutral propagation contract. Two
declared capabilities were narrower than their names suggested.

`polarimetric_transport` accepted only the `los` component and only
`ad_mode="none"`. It was served by one fused native operator,
`consumer_los_jones`, which evaluates free-space transport between two endpoint
transverse bases and rejects any differentiable input. A caller that needed a
source-basis to sink-basis operator through a reflection path, or needed the
derivative of one, had no route at all.

`reevaluate` accepted only a zero-interaction line-of-sight topology. The real
gate was a shape check in the service layer:

```python
if request.topology.primitive_sequence.shape[1] != 0:
    raise NotImplementedError(...)
```

`fixed_topology_components` declared `{"los"}` but was never read by any code
path, which is exactly the declared-but-unenforced pattern ADR-036 rejects.

A caller that wants to hold a discovered topology fixed and reevaluate it at
new endpoint positions every frame - to obtain endpoint-motion derivatives
without re-running discovery - could therefore only do so for line-of-sight.

## Decision

### 1. Both capabilities are added, and nothing existing is relaxed

This ADR adds two routes. Every operation that exists today keeps its exact
contract, including `consumer_fixed_los_gather`'s five-bit all-or-nothing
validation, its one-copy/four-byte/one-synchronization budget, the ADR-032
compact budget, and the enumerated fixed-winner re-solve's requirement that
every row reproduce.

### 2. A frozen topology that carries interactions must be prepared

Reflection field transport takes one uniform interaction depth per native
launch, so a mixed-depth frozen batch cannot be replayed in one call. The
partition into `(component, depth)` buckets is a property of the frozen
topology alone, so it is computed once by `prepare_fixed_topology` and reused
by every later `reevaluate`.

`prepare_fixed_topology` is the single place the consumer observes a frozen
topology on the host. It validates every row's depth and interaction-sequence
padding, rejects any component outside
`capabilities().fixed_topology_components`, and records what it cost. It
synchronizes; calling it per frame gives up the capability, and its recorded
counters are deliberately kept off `PropagationDiagnostics` so a per-frame call
cannot be mistaken for a free one.

This makes `fixed_topology_components` an enforced contract rather than an
advisory string, satisfying ADR-036.

The raw-`PropagationTopology` form of `FixedTopologyRequest` remains the
zero-interaction scalar and complex3 fast path on its unchanged native gather.
A raw topology carrying interactions, and any raw-topology polarimetric
request, fail loudly and name `prepare_fixed_topology`.

### 3. Per-row geometric validity is an answer, not a failure

A frozen reflection row is a face sequence, not a fixed point in space. At new
endpoint positions its stationary point moves, and it can leave its facet or
become occluded. Three options were considered:

1. fail the whole batch, matching today's all-or-nothing rule;
2. publish a per-row validity mask;
3. publish nothing and let the caller guarantee continuity.

Option 1 forces a caller back to full discovery the first time a single path
dies, which defeats the capability. Option 3 silently publishes meaningless
numbers. This ADR accepts option 2.

`FixedTopologyEvaluation` gains `row_valid`: `None` when every published row is
valid by construction, otherwise a CUDA `bool` mask over the frozen rows in
frozen row order. It is never read to the host by the consumer, adds no
device-to-host copy, and triggers no compaction. The published `K`, the row
order, and the pair segmentation are unchanged by validity.

`row_valid` is the SOLE authority for the components it covers, and it covers
exactly `capabilities().fixed_topology_row_validity_components`, today
`{"reflection"}`. A geometrically valid row may legitimately carry a zero
coefficient - a cross-polar null or a grazing null - so validity can never be
inferred from the payload.

**What the mask does not cover, stated plainly.** A frozen line-of-sight row is
replayed as pure free-space transport and is not re-tested for visibility, so a
sink that moves behind a wall still publishes `row_valid=True` and a
full-strength field, while fresh discovery would drop that row entirely. This is
inherited from the shipped version-1 line-of-sight route, which has the same
behavior and no mask at all. Adding line-of-sight validity means a per-row
visibility trace on the LoS bucket and a semantic change to a version-1 route,
which is a separate decision. A caller whose scene has blockers and needs
blockage on the direct path must rediscover; the mask will not tell it. A test
pins this so it cannot drift into a silent surprise.

**Reconciliation with ADR-034's "failure publishes no usable partial result".**
That clause governs failure transactions: capacity overflow, ABI or contract
violation, device fault. Those still raise before a result exists and are
untouched here. Geometric non-existence of a frozen path at new endpoints is a
different thing: it is the correct, complete answer to the question that was
asked. The two channels are never mixed - a contract violation still fails the
whole batch, and a dead path never raises.

**Known limitation: no invalidity attribution.**
`rayd::torch::ReflectionEpcResult` publishes `valid` but not the blocking
attribution that exists inside its parameters, so a caller cannot distinguish
facet-exit from occlusion. Exposing it is a RayD `integration.h` result
extension and an API-version bump, and is out of scope. Both branches are
covered by tests, one per branch, with a control that isolates the occluder.

**Known limitation: the frozen primitive labels can go stale within a coplanar
group.** The native re-solve compares the resolved primitive against the frozen
sequence under coplanar-group equivalence, so when the stationary point slides
onto a coplanar twin triangle the row stays valid and its published numbers are
exactly what fresh discovery would produce. The resolved ids are not
republished, so `topology.primitive_sequence` keeps the label the row was
discovered with. Material divergence within a group is impossible today because
the group key includes `face_surface_id` and material is uniform per surface, so
this is discrete-metadata staleness only. A caller that keys per-frame state on
primitive identity rather than on row identity has to be aware of it.

### 4. An invalid row is made inert at the input, not patched at the output

The transmit polarization of an invalid row is replaced by the zero vector
before the native transport runs. `project_to_wedge_plane(0, d)` is exactly
zero, a Fresnel bounce is linear in the field, and the trailing free-space
factor is a scalar, so `field_vector`, `coefficient`, `path_field`, and
`path_gain` all come out as exact zeros from the kernel that owns them. RayD's
own invalid-row store already zeroes the hit positions and normals.

Only the scalar path geometry - `path_length_m`, `delay_s`, and `direction` -
has no inert excitation, and it is selected against the mask afterwards. That
selection is a choice between a computed row and a constant, never a numerical
blend: it computes no physics, and its derivative gives an invalid row exactly
zero rather than a small wrong number.

### 5. Reflection Jones is composed, not a new native operator

The native reflection transport is linear in the transmit polarization and
linear in the receive polarization. `project_to_wedge_plane(v, e)` is linear in
`v`; a bounce scales the s and p components by coefficients that depend on the
incidence frame and the material, never on the field; the trailing propagation
factor is a complex scalar; and `project_receiver(E, d, p)` is linear in both
`E` and `p`. The map is therefore bilinear, and the operator is recovered
exactly by exciting the SAME production transport once per source basis vector
and projecting each response onto both sink basis vectors.

The composition is fixed and is structural packing, not new physics:

1. build the source transverse basis on the launch leg;
2. evaluate the native transport twice, once per source basis vector;
3. build the sink transverse basis on the arrival leg;
4. project four times through `field_project_complex3`;
5. stack into `(K, 2, 2)`.

`matrix[k, i, j]` is the response of sink basis vector `i` to source basis
vector `j`, which is the index convention the fused native LoS operator already
publishes.

**Both bases come from the native endpoint-basis owner.** A reflection row has
two different directions - the launch direction toward its first interaction
and the arrival direction from its last one - and a basis that is not
transverse to its own leg is silently shortened by the native projection, which
would make the published operator stop being the operator in the published
basis with no exception anywhere. Rather than restate the projection and
orthonormalization in Torch, each basis is obtained by handing that leg's two
endpoints to `consumer_los_jones` with per-row endpoint tables and the diagonal
pair index, so the direction is recomputed by the same native `safe_normalize`
the field kernel uses.

**Accepted cost.** Per bucket the composition issues nine launches - one source
basis, two transports, one sink basis, four projections, and for a reflection
bucket the shared EPC re-solve - against one launch for the fused primal LoS
operator, and roughly twice the Fresnel work of a single-polarization solve.
Both transports alias the same interaction tensors, so tape storage is not
duplicated. A native operator carrying a 2x2 through each interaction would be
one launch at one times the cost, but it moves a fusion boundary owned by RayD
under ADR-024 and ADR-026 and is explicitly out of scope.

**Rejected in review.** A Torch normalize, a Torch cross product or
Gram-Schmidt, a Torch dot in place of `field_project_complex3`, and
reconstructing the second matrix row by rotating the first.

### 6. Two Jones routes, held to exact agreement

`evaluate` keeps `consumer_los_jones` as the fused primal-only operator: one
launch for the whole batch, and it rejects differentiable inputs by contract.
When an AD mode is requested it uses the composed route instead. Discovery
restricts this response to line-of-sight rows, so the composed route sees one
uniform leg and needs no bucketing there.

The two routes evaluate the identical native expressions on the identical
inputs, and a test asserts bit-identical agreement of the matrix and both
bases. That agreement is what justifies keeping both: the fused route is a
launch-count optimization of the composed one, not a second implementation of
the physics.

### 7. AD contract

Differentiable: `source`, `target`, `interaction_positions`,
`interaction_normals`, `eps_r`, `sigma_e`, `gain`, `thickness`, `frequency`,
and the mesh vertices the fixed-winner re-solve chains through.

Frozen, and rejected before any native work: `tx_power`, `mu_r`, the endpoint
polarizations, and both polarization bases. The composition feeds the bases to
the native companions as `tx_polarization` and `rx_polarization`, which those
companions reject by contract, so a differentiable basis could only ever yield
a silently incomplete derivative.

This is enforced at BOTH entry points. `reevaluate` already checked all five;
`evaluate` checked only the two bases and silently accepted a differentiable
`powers_w` or endpoint polarization, returning a result whose gradient with
respect to them is simply absent - the declared-but-unenforced pattern ADR-036
rejects. `evaluate` now refuses them by name, and a test covers each declared
input.

Two consequences are stated rather than left implicit.

`direction` is a dead AD edge and that is exact. The native companions mark it
non-differentiable, so it reaches the projection as a constant. The dropped
term carries the factors `sink_basis . direction` and `field . direction`, and
a correctly built transverse basis makes the first exactly zero. This holds
only under the transversality invariant above, so the two are asserted in one
test.

The derivative is `dM/dtheta` with world-referenced bases held fixed, not the
derivative in a co-rotating transverse frame. The two agree at the
linearization point and diverge at second order, which is the right contract
for a first-order endpoint-motion derivative.

### 8. Deferred, with reasons

- **Reflection Jones in the discover path.** A compact discovered batch is
  mixed-depth, and the only sync-free way to bucket it would be device-resident
  depth segmentation published by the compact finalizer, or hoisting the
  composition into the field owner's existing per-depth block loop. Both are
  separate decisions. A caller that wants a reflection operator from a fresh
  discovery calls `evaluate` and then `reevaluate` on the returned topology.
- **Rough-reflection scaling in the fixed route.** The coherent rough factor
  and the realization phase-screen delta replacement are gated on host material
  state inside the discovery field loop. Reproducing that gate here would
  duplicate another owner's policy, and silently disagreeing with `evaluate` on
  a rough scene is worse than refusing, so the route rejects a rough scene
  before any native work. The lift is to give the existing helper a row index
  set instead of a discovery topology.
- **Diffraction and transmission in a frozen topology.** This ADR scopes itself
  to component ids 0 and 1 explicitly, so admitting another component later is
  a visible contract change rather than a silent one.
- **Forward-mode `delay_s` and `path_length_m` tangents.** The existing field AD
  companions decide their conditional differentiability inside `setup_context`.
  `torch.autograd.Function.apply` unpacks a dual before `setup_context` runs -
  verified directly: a probe Function sees `unpack_dual(x).tangent is None` for
  an input that carries a tangent - so `_ad_geometry_live` is structurally blind
  to a forward-only request and marks those two outputs non-differentiable. The
  result is then partially dual: the transport carries a tangent and the two
  geometry outputs silently do not.

  This is pre-existing, lives in `witwin.channel.runtime.autograd_contracts` and
  `propagation.fields.kernels.autograd`, affects the shipped line-of-sight route
  and all four solvers identically, and fixing it means changing the
  differentiability marking of a shared native AD facade - a `propagation.fields`
  AD-boundary change that needs its own ADR and its own evidence. Reverse mode is
  unaffected.

  What this ADR does instead is refuse to answer partially. Marking the endpoint
  positions `requires_grad` in addition to making them dual makes the same
  companions publish both tangents, correctly: the measured `delay_s` tangent is
  `3.3356e-9` s/m, exactly `1/c`, and the `path_length_m` tangent matches central
  differences. The prepared route therefore requires that convention and raises
  `NotImplementedError` on a forward-only dual. The check is scoped to the
  prepared route; the raw zero-interaction route is version-1 surface with
  shipped callers and keeps its acceptance rules unchanged.

### 9. `CONTRACT_VERSION` goes to 2

Widening `fixed_topology_components` and `fixed_topology_responses` alone would
NOT have required a bump: a caller discovers both through `capabilities()`, and
adding a member to a set it already inspects breaks nothing. The bump is driven
by three other things.

`FixedTopologyEvaluation` grows `row_valid`, a public result-schema addition a
caller must be able to detect without introspecting the dataclass.
`PropagationCapabilities` grows two fields and the package grows three exports,
which move `contract_sha256` and the public snapshot. And the shipped version-1
documentation states that `polarimetric_transport` is primal-only; lifting that
while remaining at version 1 would make the contract self-contradictory.

## Consequences

- `prepare_fixed_topology`, `PreparedFixedTopology`, and `FixedTopologyBucket`
  join the public surface; the snapshot, the contract-coverage manifest, and
  its expected export count move with them.
- No native symbol is added, so the binding manifest is unchanged. The
  composition and the frozen replay are built from already-owned operators.
- `propagation.geometry.reevaluate` publishes `reflection_epc_paths`, the one
  implementation of the fixed-winner re-solve. The enumerated caller keeps its
  all-or-nothing raise on top of it; the consumer publishes the mask. One
  implementation, two policies.
- Row selection and pair segmentation for a prepared topology are structural
  integer work on caller-owned tensors, validated by one device bitmask read
  under the same one-copy/four-byte/one-synchronization budget the native LoS
  gather uses. That budget is a self-reported constant, so a companion test
  measures the real thing: it counts every host read of a CUDA tensor inside one
  warm reevaluation and requires exactly one, the contract bitmask `.item()`.

## Named residual cost

The published validation budget covers device-to-host reads. It is not the whole
per-call cost, and the difference is named here rather than left to be
rediscovered.

A prepared reflection reevaluation re-stages scene-static host tensors on every
call: `_scene_tables` re-runs `face_material_field_bundle`, which uploads the
host `MaterialStore` tensors and gathers them per face, and `reflection_epc_paths`
re-runs the face anchor and normal construction per bucket. Only the coplanar
union-find is cached. Measured with `torch.cuda.set_sync_debug_mode`, a warm
prepared reflection call reports on the order of twenty synchronizing torch
operations, nearly all of them these host-to-device staging copies, against one
for a prepared line-of-sight call.

This is parity with what a discovery solve already pays per solve, it is data
staging rather than physics, and it is not a fallback - but it is real per-frame
work on a capability whose whole point is per-frame use. Hoisting it into a
scene-owned lazy cache is a `scene`-owned change with its own resource and
lifetime questions, in the same family as the Plan-13 phase-screen resources, and
is deliberately not bundled into this contract change.

## Accepted numerical edge

If a sink coincides exactly with the last interaction point, the arrival leg has
zero length. The sink basis then falls back to the native `safe_normalize`
constant `(0, 0, 1)` while the field kernel's own projection direction falls back
to the last reflected direction, so the two disagree. This requires a receiver
placed exactly on the reflecting surface and is not reachable from any physical
request; it is recorded so the disagreement is a known constant rather than a
surprise.
