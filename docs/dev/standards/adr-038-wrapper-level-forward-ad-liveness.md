# ADR-038: Forward-mode geometry liveness is decided at the wrapper boundary

- **Status:** Accepted
- **Date:** 2026-07-25
- **Kind:** AD dispatch-contract decision. No numerical change: every native
  primal/JVP/VJP companion computes exactly what it computed before.
- **Related:** ADR-034, ADR-037 (records the defect and its interim refusal)

## Context

The conditionally differentiable geometry outputs - `path_length_m` and
`delay_s` - are marked live only when a geometry input participates in AD.
That policy is deliberate (plan 07 AD-2): a materials-only graph keeps them
detached so it never pays for geometry adjoints it did not request.

The liveness test, `_ad_geometry_live`, checks both `requires_grad` and a
forward-mode dual tangent. The test itself was correct. It was being asked the
question in the wrong place: inside `setup_context`, after
`torch.autograd.Function.apply` has already unpacked forward duals into
separate primal and tangent streams. From there a forward-only request is
indistinguishable from no request. The Function's `jvp` hook then computed the
geometry tangents natively and discarded them on return, and
`mark_non_differentiable` pinned the outputs besides.

The observable defect: a dual-without-`requires_grad` request received a
partially differentiated answer - transport tangent present, `delay_s` tangent
silently absent. `delay_s` is precisely what a Radar Doppler `delay_rate`
consumer reads. ADR-037 refused such requests on the prepared fixed-topology
route rather than answering partially; the raw LoS route and the four solvers
kept the silent drop.

## Decision

Liveness is computed once, in the caller-facing wrapper, where forward duals
are still visible, and passed into the Function as an explicit trailing
`geometry_live` input. `setup_context` stores the flag; it no longer infers.

Three Functions carry the flag - the only ones whose differentiability is
conditional on geometry liveness:

| Function | wrapper | liveness inputs |
|---|---|---|
| `_FieldFreeSpaceAdFunction` | `field_free_space_ad` | `source`, `target` |
| `_FieldReflectionSequenceAdFunction` | `field_reflection_sequence_ad` | `source`, `target`, `interaction_positions`, `interaction_normals` |
| `_FieldRoughReflectionScaleAdFunction` | `field_rough_reflection_scale_ad` | `positions`, `normals`, `source` |

The other field Functions (transmission, wedge, coupled) have no conditional
geometry marking and are unchanged. The BDPT pipeline already called
`_ad_geometry_live` at pipeline level, where duals are visible; unchanged.

The ADR-037 refusal on the prepared route is deleted. A forward-only dual now
receives the complete derivative on every route: prepared fixed-topology, raw
LoS, and the four solvers.

## What does not change

- **Numerics.** The native JVP companions were already invoked and already
  computed these tangents; the change is whether the caller receives them.
  The tangent values are the same tensors the companions always produced.
- **Reverse mode.** `requires_grad` is equally visible at the wrapper and at
  `setup_context`; every reverse-mode request resolves to the same liveness as
  before.
- **The materials-only detachment contract.** With no geometry gradient or
  tangent requested, `geometry_live` is `False` exactly as before and the
  outputs stay detached (AD-1 exactness).
- **The requires_grad-plus-dual convention.** Requests written against the
  ADR-037 interim rule resolve to `geometry_live=True` through either signal.

## Evidence

- The consumer test that previously asserted the refusal now asserts the
  positive contract: a forward-only dual on the prepared reflection route
  yields a `path_length_m` tangent matching central finite differences and a
  `delay_s` tangent equal to it divided by c
  (`tests/propagation/consumer/test_fixed_reflection.py`).
- Direct facade coverage for the same contract on free space
  (`tests/ad/test_forward_mode_liveness.py`), including the negative control:
  a materials-only request keeps `path_length_m` and `delay_s` detached.
- The full suite passes unchanged everywhere else; no tolerance, manifest, or
  budget moved.
