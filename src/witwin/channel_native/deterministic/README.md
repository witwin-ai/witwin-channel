# Deterministic solver owner

## Ownership

`deterministic` owns deterministic solver configuration, orchestration,
coherent accumulation, result assembly, and deterministic scattering-path
integration. Shared topology, geometry, and fields remain in `propagation`;
scene compilation and material encoding remain outside this package.

## Public entry points

`witwin.channel_native.deterministic` exports exactly `Config`, `PathTable`,
`Result`, and `solve`. Kernel, pipeline, accumulation, field, and scattering
modules are internal.

## Dependency rules

The solver consumes scene, material, and typed propagation contracts. It may
select solver policy but cannot redefine shared domain owners or import another
solver. Native calls belong in deterministic or shared kernel facades, never in
result models.

## Numerical and AD contract

Contributions are coherent complex fields before power projection. Component
IDs, path depth, row order, visibility, phase convention, and accumulation
order are contractual. Expression-order or tolerance changes are separate
numerical changes, not architecture cleanup.

### AD contract

`Config.ad_mode` accepts `none`, `jvp`, and `vjp`. AD uses fixed topology and
winners: continuous geometry/material/frequency/field leaves use registered
native companions while integer topology stays frozen. Scattering with AD,
unsupported tangents, double backward, and unsupported dispersive-frequency
seams fail loudly rather than detach.

## Forbidden fallback

Missing CUDA/RayD/native companions cannot trigger CPU enumeration, PyTorch RF
recomputation, zero fields, reduced depth, or another solver. Capability and
memory-budget failures occur before partial execution.

## Maintenance

Public changes require `ci/public-api-snapshot.json` and a migration note.
Moving frozen kernel bodies updates the canonical owner in
`ci/ops_migration_manifest.json` without changing the contract. New
dependencies must satisfy the import-graph manifest.
