# ADR-018: BDPT standalone diffraction realignment

Status: Superseded by Plan 13 Phase 4 for native binding disposition; the
standalone-diffraction numerical decision remains accepted.

## Supersession note

Plan 13 Phase 4 performed the follow-up governance work that this ADR deferred.
Its four-axis reachability audit covered static production callers, dynamic
binding lookup, public imports, and real BDPT end-to-end execution. Both crude
bindings, `bdpt_diffraction_point_connection_samples` and
`bdpt_diffraction_connection_samples_from_tape`, failed all four axes and were
deleted together with their two kernels and the now-dead
`bdpt_diffraction_contribution` helper. The evidence is recorded in
`docs/dev/audit/phase13-phase4-dead-binding-reachability.json`.

This supersession changes only the later native binding disposition. The
historical reasoning below is retained verbatim as the record of why ADR-018 did
not perform that native surgery in its original commit. In particular, the
frozen Phase 9 owner inventory remains an immutable historical artifact; Phase
4 records its approved current-state delta separately.

## Context

The WS1 alignment audit measured the BDPT solver's standalone (first-order,
non-coupled) diffraction component against the deterministic reference on the
shared WS1 wedge fixture (`tests/support/scenes.py::wedge_diffraction_scene`,
grid receivers at `x=3.0` and the point-receiver variant). On the same scene the
three solvers reported:

- deterministic diffraction (grid): `1.9039e-06`
- montecarlo.basic diffraction (grid): `2.6567e-06` (within ~1.4x of the
  reference)
- montecarlo.bdpt diffraction (grid): `8.2547e-04`

BDPT was therefore `8.2547e-04 / 1.9039e-06 = 434x` above the deterministic
reference on the grid fixture (the audit reports 430x grid / 2175x point).

Root cause: BDPT's standalone diffraction did not evaluate UTD/Fresnel edge
physics. It called a crude native power heuristic,
`bdpt_diffraction_contribution` in
`native/channel/kernels/bdpt_connect_common.cuh:226-243`, which forms the
per-connection power as

```
src_power * material_gain * (wavelength / 4*pi)^2 * edge_measure_weight *
    grid_cell_area * wedge_scale / (source_distance^2 * target_distance^2)
```

This is a `material_gain`-only geometric fall-off with a `wedge_scale` clamp of
the exterior angle. It has no diffraction coefficient (no UTD `D`), no Fresnel
term, no phase, and no dependence on the incidence/diffraction geometry beyond
the two path-leg distances. The native point sampler
(`bdpt_diffraction_point_connection_samples`) and its grid tape sibling
(`bdpt_diffraction_connection_samples_from_tape`) both build their per-row
contribution from this helper, then apply a Monte Carlo direct/Keller sampling
split with MIS. The estimator was self-consistent (seed-stable, additive over
disjoint wedges) but numerically unrelated to the UTD field the deterministic
and Path solvers evaluate.

By contrast, BDPT already produces bit-identical delta-specular *reflection* and
coupled reflection-diffraction contributions by routing those path classes
through the shared enumerated propagation engine as an opaque discrete-path
oracle (ADR-008):

- reflection: `montecarlo.bdpt.pipeline._reflection_discrete_connection_samples`
  -> `evaluate_enumerated_paths({"reflection"})`, selecting `component_id == 1`.
- coupled: `montecarlo.bdpt.pipeline._coupled_discrete_connection_samples`
  -> `evaluate_enumerated_paths({"reflection", "diffraction"}, coupled_paths=True)`,
  selecting `component_id >= 3`.

Standalone diffraction is a delta-like discrete path in exactly the same sense:
first-order UTD produces one deterministic edge-diffraction connection per
(transmitter, edge, receiver) triple, with the same field the deterministic
solver evaluates. There is no reason for BDPT to estimate it stochastically with
a separate, physically wrong heuristic.

## Decision

Option B. Route BDPT standalone diffraction through the shared enumerated
continuity op, mirroring the reflection precedent exactly.

`montecarlo.bdpt.pipeline` gains `_diffraction_discrete_connection_samples`,
which calls the public `evaluate_enumerated_paths({"diffraction"})` read-only
(ADR-008), with `max_diffraction_order` left at its default of 1 (first-order
UTD; the enumerated engine only implements `_diffraction_topology_order1`),
selects the rows with `component_id == 2`, and packs them with
`_evaluated_connection_samples(..., component_out=2)`. Each selected row becomes a
single discrete connection with unit forward/reverse mass (`pdf = 1`,
`mis_weight = 1`), identical to the reflection and coupled discrete-connection
blocks. The `_collect_connection_samples` diffraction branch appends this block
to `sample_blocks` and increments `launch_count` by 1, replacing the native
point-sampler dispatch and its grid streaming accumulator.

The crude native heuristic is removed from BDPT dispatch. The obsolete Python
connection builder `_native_diffraction_point_connection_samples` and its private
helper `_diffraction_strategy_count` are deleted from
`montecarlo.bdpt.connections`, together with the sampler-only helper
`_diffraction_sample_split` (and its re-export in `montecarlo.bdpt.solver` and
its unit test).

### Native kernel and binding disposition

The crude device helper `bdpt_diffraction_contribution` is NOT removed, and the
native ABI symbols `bdpt_diffraction_point_connection_samples` and
`bdpt_diffraction_connection_samples_from_tape` are NOT removed. Reasons:

1. `bdpt_diffraction_contribution` does not become fully dead. Both the point
   kernel (`bdpt_diffraction_point_connection_samples_kernel`) and the grid tape
   kernel (`bdpt_diffraction_connection_samples_from_tape_kernel`) call it. This
   ADR retires only the point-sampler production dispatch; the grid tape variant
   is out of scope, so the helper still has a live in-tree caller. The task's
   native-removal branch is gated on this helper being fully dead, and it is not.

2. Removing the `channel_bdpt_diffraction_point_connection_samples_cuda` ABI symbol is
   a native-surgery + governance change of a different class: it would edit the
   frozen phase-9 owner inventory
   (`docs/dev/audit/phase9-native-owner-inventory.json`) `abi_owner` set and its
   sha256 digest, the frozen `cpp_body_hash_multiset`, the owner-inventory unit
   test (`tests/test_native_owner_inventory.py`
   `test_bdpt_abi_owners_are_complete_and_frozen`), the native binding count
   (`EXPECTED_NATIVE_BINDING_COUNT` 193), the contract-coverage manifest, the
   maintenance-budget entry, the negative/boundary tests, and it would require a
   full native rebuild with rebuild evidence. Per the change-together and
   fusion-ownership rules, that belongs in its own governance-complete commit,
   not in this Python realignment.

3. Precedent: the sibling `bdpt_diffraction_connection_samples_from_tape` binding
   already exists as a contract-tested, E2E-name-tagged native facade with no
   production caller, and CI is green with it. After this change the point
   sampler is in the same state (owned by
   `montecarlo.bdpt.kernels.paths.bdpt_diffraction_point_connection_samples`,
   contract-tested by `tests/kernels/test_bdpt_ops_facade.py`, E2E scenario tag
   `bdpt-diffraction`). CI reachability is name-based, not runtime call-graph, so
   the coverage manifest stays valid.

A follow-up native-cleanup ADR may retire both crude diffraction kernels, the
`bdpt_diffraction_contribution` helper, and the point/tape ABI bindings together
with the full governance choreography and a rebuild. That is explicitly out of
scope here.

## MIS implications

The 3-way diffraction sampling split in
`montecarlo.bdpt.connections._diffraction_sample_split`
(`direct = ceil(n/3)`, `keller = ceil((n+1)/3)`, `suffix = 0`) and the paired
`_diffraction_strategy_count` existed only to partition the stochastic
direct-shooting and Keller-cone proposals for the crude estimator and to feed
`bdpt_diffraction_strategy_mis_weight` (balance/power-heuristic weighting across
the two proposals). Those samplers are removed, so the split disappears with
them.

The enumerated discrete diffraction connection is not a sampled strategy. It is a
single deterministic UTD path per (tx, edge, rx), packed with `pdf = 1` and
`mis_weight = 1` exactly as the reflection and coupled discrete blocks already
are. Consequently:

- Standalone diffraction contributes a unit-mass discrete connection with no MIS
  down-weighting, consistent with reflection and coupled.
- There is no double count between standalone diffraction (`component_id == 2`,
  first-order UTD) and coupled reflection-diffraction (`component_id >= 3`,
  order-2 mixed) because they are disjoint path classes; both accumulate into the
  diffraction output bucket (`component_out=2`) additively, unchanged from the
  prior coupled behavior.
- The estimator's other stochastic strategies (LoS, sampled
  reflection+transmission chains, Kirchhoff scattering NEE) keep their own MIS
  weights untouched; standalone diffraction never shared a MIS strategy group
  with them (it was accumulated as its own component block), so removing the
  direct/Keller split cannot change their weights.
- `mis="none"`, `mis="balance"`, and `mis="power_heuristic"` now all yield the
  same standalone-diffraction estimate, because a unit-mass single-strategy
  connection is MIS-invariant. This matches how reflection already behaves under
  the three MIS modes.

## Acceptance gates

- BDPT standalone diffraction component power within `[0.5x, 2x]` of the
  deterministic reference on the WS1 wedge fixtures (grid and point), versus 434x
  (grid) / 2175x (point) before. In practice the enumerated route reproduces the
  deterministic value to within tight tolerance because both consume the same
  enumerated field evaluation. Guarded by
  `tests/montecarlo/bdpt/test_diffraction_parity.py`.
- Existing BDPT suites stay green. The tests that encoded the retired crude
  estimator's behavior are updated (listed below), not weakened.
- Sample-count and BDPT runtime stay within 1.5x of the pre-change budget. The
  enumerated diffraction block emits far fewer connection rows (edges x
  receivers) than the retired sampler (samples x receivers), so both the row
  count and the runtime decrease; the workspace estimate in
  `_estimate_workspace_bytes` is left as a conservative upper bound, matching the
  existing enumerated-reflection estimate.

### Updated tests (encoded the old crude behavior)

- `test_bdpt_single_wedge_diffraction_fixed_seed_is_stable`: the crude estimator
  was seed-dependent, so the test asserted `seed=8 != seed=7`. Enumerated
  diffraction is deterministic; the seed-difference assertion is replaced by a
  seed-invariance assertion.
- `test_bdpt_single_wedge_diffraction_uses_original_direct_keller_split` and the
  `_diffraction_sample_split` helper it exercised are removed: the direct/Keller
  split no longer exists.
- `test_bdpt_single_wedge_point_diffraction_converges_to_maintained_reference`:
  the maintained reference `4.66e-05` fossilized the crude point estimate. It is
  replaced by a deterministic-parity assertion (the enumerated route is not a
  convergence-to-magic-number estimator).

## Enforcement

The BDPT -> enumerated import edge already exists for reflection and coupled and
is admitted by `mc-enum-001` in `ci/import_graph_allowlist.json` (ADR-008).
Standalone diffraction reuses the same public `evaluate_enumerated_paths` entry
imported on the same line (`pipeline.py:23`) from the same module, so no new
import edge is created and the allowlist is unchanged. The allowlist baseline is
frozen by a sha256 digest that rejects rewriting an existing entry, so the
`mc-enum-001` justification text is intentionally left as-is; this ADR is the
record that the admitted edge now also carries standalone diffraction.

## Revisit condition

Re-evaluate when the follow-up native-cleanup ADR retires the crude diffraction
kernels and their ABI bindings, or if first-order UTD diffraction ever needs a
stochastic estimator in BDPT that the deterministic enumerated route cannot
express.
