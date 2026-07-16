# ADR-009: Native fusion and translation-unit ownership

Status: accepted for Phase 9 of the modular architecture hardening migration.

## Context

Python packages express user-facing semantics, but they are not a safe template
for CUDA translation units. A native operation may intentionally fuse validation,
geometry, field evaluation, AD recomputation, reduction, and packing. Mirroring a
Python directory can therefore add kernel launches, synchronization, tensor
materialization, tape storage, or a different floating-point evaluation order.

The pre-migration evidence is frozen in
`docs/dev/audit/phase9-native-owner-inventory.json`. It records each owner's ABI
entries, semantic surface, tape lifetime, source-level launches, intermediate
storage, fusion boundary, non-split reason, expiry condition, and normalized C++
function-body hash multiset. Paths and line numbers are excluded from body-hash
identity so a complete function may move without weakening the exact gate. The
Phase 0 baseline and its digest remain immutable.

## Decision

Native ownership follows this priority order:

1. ABI operation and result schema.
2. Kernel fusion and launch/synchronization contract.
3. Tape lifetime and row identity.
4. Device primitive and numerical evaluation order.
5. Compile dependency and physical translation unit.

The initial owners are:

| Baseline translation unit | Owners |
| --- | --- |
| `kernels/path_trace.cu` | `path.compaction`, `path.topology`, `path.core` |
| `kernels/field_transport_ad.cu` | `field_transport.free_space`, `field_transport.reflection_sequence`, `field_transport.transmission_sequence` |
| `kernels/field_wedge_ad.cu` | `field_wedge.diffraction`, `field_wedge.coupled_rd`, `field_wedge.project_complex3`, `field_wedge.coupled_prepare` |
| `kernels/bdpt_connect.cu` | `bdpt.mis`, `bdpt.endpoint_connection`, `bdpt.diffraction_connection`, `bdpt.connection_storage` |
| `field_transport.cuh` / `field_transport_ad.cuh` | `legacy_slab.primal`, `legacy_slab.dual` |

Host validation, launch, and packing may be extracted first. A migration moves
complete functions, keeps each ABI schema and tape row identity, and preserves
launch configuration, explicit synchronization, compiler attributes/macros, and
the normalized signature/body hash tuple. It must not split a fused row operation
when that would introduce another launch, materialized tensor, persistent tape,
or reduction-order change.

The source snapshot contains 51/11 launch/sync sites in `path_trace.cu`, 9/0 in
`field_transport_ad.cu`, 9/0 in `field_wedge_ad.cu`, and 17/2 in
`bdpt_connect.cu`. These are migration evidence, not a license to add launches;
runtime launch baselines remain the acceptance authority.

## Translation-unit budget

Native `.cpp` and `.cu` translation units have a 3,000-line hard limit and a
2,000-line recommendation. At this decision point only `path_trace.cu` exceeds
the hard limit (4,270 lines). The four planned Phase 9 units exceed the
recommendation at 4,270, 2,496, 2,473, and 2,356 lines respectively.

These debt sets are exact allowlists with current-size caps. Tests require an
allowlist to equal the current violation set, so it must shrink as soon as a unit
moves below its threshold. A new violation cannot inherit a generic exception.

## Numerical duplication and LegacySlab

Host checks and purely discrete packing may be shared after exact validation.
Numerical expressions in primal/JVP/VJP paths remain duplicated by default.
They may be deduplicated only in a separate numerical change that proves output
exactness, unchanged evaluation order and inline attributes, relevant PTX/SASS
parity, and no performance regression.

`LegacySlabComplex` primal math and its `DualLC` derivative mirror remain
separate lockstep owners. The word "legacy" is not deletion evidence. Their
floor, clamp, branch, complex-square-root, and phase evaluation order must not be
replaced or templated during a translation-unit move. The inventory's primal and
dual hash multisets plus dedicated runtime primal/dual exact and duality tests are
the lockstep gate. Templating is permitted only after those tests, SASS evidence,
and performance gates demonstrate equivalence.

## Expiry and acceptance

Every retained fusion in the inventory has a concrete expiry condition. "File is
large" is not sufficient to override it. Each moved owner must satisfy the
Phase 0 semantic/schema/runtime baselines, normalized body hashes, launch and
memory gates, AD exact/duality tests where applicable, and production static
gates. If a safe split would add launches or materialization, the owner remains
fused and this ADR is the explicit exception rather than a mechanical file split.
