# ADR-043: The propagation-consumer AD capability matrix

- **Status:** Accepted
- **Date:** 2026-07-26
- **Kind:** One breaking contract-version bump (`5 -> 6`) carrying: six new
  capability-record fields, one narrowed capability-record field, two new
  diagnostics fields, one new native cotangent input and one new native tangent
  output on each of the two Channel-owned field transports, two new
  pre-compute refusals, and one first-order guard on every registered backward.
  No published primal number changes. No new production Torch physics, no
  fallback, no new host/device transfer, no new synchronization, no change to
  reduction order, and no additional kernel launch on any route.
- **Related:** ADR-036 (a declared limit must be discoverable, not learned from
  a rejection), ADR-037 (the fixed-topology reflection route and the composed
  Jones operator), ADR-038 (wrapper-level forward-AD liveness, whose decision
  shape this reuses), ADR-039 (the source amplitude companion the ledger now
  counts), ADR-041 (slot batching and the time-varying CIR), ADR-042 (wideband
  columns and `frequency_column_count`)
- **Closes:** Phase-9 work item 1, the Channel half

## Context

An AD census of `witwin.channel.propagation.consumer` enumerated 295 cells over
`(route, leaf, ad_mode)`. The result was not "AD is broken". 122 cells produced
a genuine derivative and most of them were correct. The result was that the
contract could not say which.

Four specific failures made that concrete.

**1. `capabilities()` could not express any of the real structure.**
`component_ad_modes` was built as `tuple((component, AD_MODES) for component in
sorted(COMPONENTS))`: every component advertised every mode, unconditionally.
Nothing in the record was component-specific, leaf-specific, or route-specific,
so every gap below was invisible to a caller who asked before calling.

**2. A published continuous quantity returned a silent zero.**
`PropagationGeometry.field_direction` was `mark_non_differentiable` in both
branches of both Channel-owned field companions, on every route and in every AD
mode. A Radar angle-of-arrival or beam-steering loss differentiated through it
got an exact zero with no error and no warning. The same was true of
`interaction_positions_m` on the discovery route only, while the identical field
name was live on the reevaluate route: one field name, two silently different
semantics.

**3. Unsupported reverse-mode requests failed after publishing a result.**
`sources.powers_w`, the endpoint polarizations, and `mu_r` are rejected by the
native companions by name - correctly and loudly - but on the discovery route
that rejection fires from inside `backward()`, after `evaluate` has already
returned a complete `PropagationEvaluation`. A partial result for an
unsupported request is exactly what the compute policy forbids.

**4. A second-order request read as an exact zero.** A reverse gradient taken
inside a `dual_level` came back with the correct first-order value and
`unpack_dual(grad).tangent is None`. No error anywhere. Reverse-over-reverse was
almost as bad: `create_graph=True` returned a silently detached first gradient
and failed one step later with a generic Torch message that named Torch rather
than the owner that could not answer.

Separately, `diffraction` advertised all three AD modes while
`consumer.evaluate(components={"diffraction"})` raised `IndexError` at every
mode including `none`, so an entire advertised AD column was fictional.

## Decision

### The supported differentiable route is fixed topology

Every decision below follows from taking that literally. The fixed-topology
reevaluation route is widened and proved; the discovery route's differentiable
surface is *declared and narrowed* rather than expanded; everything else becomes
an explicit refusal.

### Four target states, and "silent" is not one of them

`SUP`, `ZERO`, `REF`, `DECL`, defined with their required evidence in
`docs/dev/propagation-ad-capability-matrix.md`. `DECL` is legal for published
outputs only, never for inputs, and only with a named deferral. A cell with no
declaration is a defect.

### `field_direction` becomes a real derivative on `{los, reflection}`

The adjoint and the dual already existed inside the two Channel-owned kernels
and were dropped at the ABI boundary. The `field_transport_free_space.cu`
provenance section in `fields.cu` already
built an internal `g_direction` accumulator and pushed it through
`adj_v3_safe_normalize`; the `field_transport_reflection.cu` provenance
section in `fields.cu` already built
`g_final_direction` and `DualF3 final_direction`. This ADR adds an optional
`grad_direction` cotangent input that seeds those accumulators, and a
`direction` tangent output that publishes the dual that was already computed.
Four kernels, four binding signatures, no new math, no new launch, and no new
allocation on any path that does not ask for it.

The reflection seeding deserves a note, because it is the one place an external
seed can interact with existing state. `adj_safe_normalize(final_offset,
outgoing_last, g_final_direction, ...)` splits the direction cotangent over the
segment *and* over the alternate branch `outgoing_last`, which is the last
bounce's reflected direction. Seeding `g_final_direction` therefore correctly
carries a direction cotangent back into the bounce chain rather than stopping at
the final segment. That is a property of the existing adjoint, not something
added here, and it is what makes the reflection cell exact against finite
differences.

**Transmission, the wedge family, and the coupled RD/DD family are excluded and
the reason is hard**: their backward and jvp entry points forward directly to
`rayd::torch::*`, and RayD owns their direction seam. Those cells are `DECL`
with a named deferral to a RayD ADR.

### Direction liveness is one decision per result

`direction_differentiable_components = {"los", "reflection"}` is published, and
liveness is decided **once for the whole result** at the wrapper level, in the
same place ADR-038 decides geometry liveness, from the host-known component set
of the frozen batch. A batch that carries any other component publishes a fully
detached `field_direction` for every row - today's behaviour, no regression -
and the record says why.

There is never a result whose `field_direction` is live for some rows and
silently dead for others. Under fixed topology this costs nothing:
`fixed_topology_components` is already `{los, reflection}`, so the inner loop a
per-frame consumer runs is always fully live. Direction liveness is additionally
conjoined with the ADR-038 geometry decision, because a direction derivative is
a geometry derivative and cannot be live where the geometry is not.

### The discovery route declares its geometry rather than detaching it

`differentiable_geometry_outputs` publishes, per route, which of
`path_length_m`, `delay_s`, `interaction_positions_m`, and `field_direction`
carry a derivative: `discovery -> {path_length_m, delay_s}` and
`fixed_topology -> ` all four. Discovery re-solves the topology, so the
derivative of a discovery result is only defined between selection boundaries
and Channel deliberately publishes no subgradient at one. Stating that is the
fix; making it live is a separate decision with no consumer and no accepted
answer to the selection-boundary question.

### Material leaf x component is published, not inferred from a zero

`component_material_leaves` publishes which compiled-material tensors each
component reads. Marking the wrong one stays a `ZERO`, because the zero is the
true and complete answer and refusing would break a legitimate mixed-component
request where one tensor is live for one component and not another. What
changes is that the zero is discoverable before the call instead of being
indistinguishable from "AD is broken here".

### Every primal-only input is refused before any native work

`primal_only_ad_inputs` publishes the union - the three endpoint constants, the
two polarization bases, and the two relative permeabilities - and
`_require_primal_only_ad_inputs` runs on the pre-flight of **every** response
and **every** route, driven by that record rather than by a list duplicated per
route. The refusal that used to arrive from inside `backward()` now arrives
before a result object exists.

The same commit corrects the message that claimed "only endpoint positions and
`reference_frequency_hz` support AD", which was false for the reflection
reevaluate route where vertices, `eps_r`, `sigma_e`, `thickness_m`, and `gain`
all produce measured nonzero derivatives.

### `diffraction` advertises only the primal

`component_ad_modes["diffraction"]` narrows to `{"none"}`. The existing
`unsupported_ad` branch of `_preflight_evaluate` then refuses `jvp` and `vjp`
before any native work, with no new refusal code.

The primal `IndexError` is **recorded, not fixed**. It is a primal reachability
defect (`_solver_scene` builds `SolverScene(transmitters=(), receivers=())`
while `enumerated/diffraction.py:177` indexes `tx_polarizations[tx_index]`),
it is out of this ADR's scope, and fixing it would silently re-open an AD column
nobody has validated. A regression test pins the failure by exception type and
site so a future fix is a deliberate decision. No polarization is fabricated to
make the route run.

### No second-order AD, refused before any partial second-order result

`supports_higher_order_ad = False`, enforced by two mechanisms that catch
different compositions:

- **Reverse-over-reverse.** `_ad_first_order_only` decorates every registered
  backward in the package (46 sites) and raises when `torch.is_grad_enabled()`,
  which is precisely the state `create_graph=True` establishes. It fires before
  any native launch and names the owner.
  `torch.autograd.function.once_differentiable` stays underneath as defence in
  depth; it cannot replace this check, because it runs the backward body inside
  `torch.no_grad()` and only fails when the detached gradient is later used.
- **Forward-over-reverse.** `ad_mode="vjp"` with a forward dual on any input a
  caller can seed is refused on the pre-flight of both routes.

The symmetric rule - `ad_mode="jvp"` with a `requires_grad` input - was checked
against the existing tests and **deliberately not enforced**. ADR-038's declared
convention is that a forward-only dual and a `requires_grad` leaf agree bit for
bit, `test_forward_mode_publishes_geometry_tangents_under_the_declared_convention`
legitimately builds a dual on a `requires_grad` primal under `ad_mode="jvp"`,
and the field facades run one `Function` for both modes, so such a request is a
legitimate first-order one. The narrower "a dual and `requires_grad` on the same
tensor" variant would have broken the same test. The composition that rule was
meant to catch is caught instead where it becomes wrong: inside the backward.

Nested forward levels stay Torch-owned. Torch raises `Nested forward mode AD is
not supported`, that message is pinned by a test, and Channel does not wrap it.

### AD accounting is published on both routes

`PropagationDiagnostics` gains `ad_companion_launches` and `ad_tape_bytes`.
The discovery route already built an `AdLaunchLedger` inside its field loop and
`_diagnostics` threw it away; it is now published. The fixed-topology route -
the inner loop a per-frame consumer runs - built no ledger at all; it now builds
exactly one per call, at the one place that owns the whole call.

`ad_tape_bytes` reproduces the reverse-only gate the solver metadata layer
applies (`deterministic/pipeline.py`): forward mode retains nothing past the
solve, so a `jvp` call reports zero tape however many companions it launched.
Forwarding the raw counter would have reported retained tape for a forward solve
and contradicted the ledger's own contract.

## Consequences

- `CONTRACT_VERSION` moves `5 -> 6`. A consumer that pins by equality moves its
  pin. `ci/public-api-snapshot.json` records the two changed contract hashes.
- A caller who was relying on `field_direction` being detached now receives a
  graph-bearing tensor on the fixed-topology route. That is the point, and it is
  additive: the primal values are bit-for-bit unchanged.
- A caller who was relying on a discovery `.backward()` raising late now gets
  the same refusal earlier, from `evaluate` rather than from `backward`.
- A caller who was relying on `create_graph=True` silently returning a detached
  gradient now gets a `NotImplementedError` naming the owner. There was no
  correct second-order answer behind that silence.
- `diffraction` with `ad_mode` in `{jvp, vjp}` now raises at the pre-flight
  instead of raising an `IndexError` deeper in. The primal behaviour is
  unchanged and still broken, on purpose, with a pinned regression test.

## Alternatives rejected

**Make discovery geometry live too.** Rejected: the topology-selection
derivative has no accepted answer, no consumer asks for it, and it would need
new cotangent seeding through `evaluated_paths_compact_finalize_backward`. A
declaration is the honest answer this phase.

**Refuse a material leaf that a component does not read.** Rejected: it would
break a legitimate mixed-component request, and the zero is genuinely the
correct derivative. Publishing the split makes it discoverable without making a
true answer into an error.

**Fix the diffraction primal here.** Rejected: out of scope, and it would
re-open an unvalidated AD column as a side effect of a capability change.

**Decide `field_direction` liveness per row.** Rejected outright. A result whose
rows disagree about whether a published field carries a derivative is the exact
defect class this ADR exists to remove.

## Evidence

- `tests/propagation/consumer/test_phase9_ad_matrix.py` - every decided cell at
  the consumer boundary, with finite differences, analytic closed forms, and
  exact-zero assertions as oracles.
- `tests/propagation/consumer/test_ad_capability_matrix.py` - the matrix
  document is parsed, its vocabulary is closed, every cited test resolves, and
  the document is pinned against the live capability record.
- `tests/ad/test_field_em_ad.py` - direct contract tests for the new native
  cotangent input and tangent output, validated by the adjoint identity
  `<w, J v> == <J^T w, v>`, plus the wrong-shape refusal and the
  seed-alone launch.
- `docs/dev/propagation-ad-capability-matrix.md` - the matrix, the tape ledger,
  and the named deferrals.
