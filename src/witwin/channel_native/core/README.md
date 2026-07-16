# Core value-contract boundary

## Ownership

`core` owns small cross-domain value contracts: antenna definitions,
component/edge policy, complex field state, memory budgets, and shared
metadata. Domain implementations live in their canonical top-level owners.
The retired `core.kernels.ops` and `core.path_topology` modules do not exist.

## Public entry points

The root package exports `AntennaArray`, `AntennaPattern`, `Complex3State`, and
`JonesState` from core owners. Other core modules are internal contracts.
Native operations are imported from their scene, material, propagation,
solver, scattering, or runtime owner.

## Dependency rules

Value-contract modules remain solver-independent. Code imports canonical
domains directly; no compatibility facade may route around those boundaries.

## Numerical and AD contract

Field state follows the package `e^{+j wt}` / `e^{-j k r}` convention. Tensor
contracts preserve dtype, device, shape, row order, and storage identity unless
conversion is explicit. Topology IDs, component IDs, and winners are discrete
integer metadata.

### AD contract

Core defines the shared `none`, `jvp`, and `vjp` vocabulary and fixed-topology
metadata. Discrete topology is non-differentiable; continuous leaves remain
live only through their owning native AD seam.

## Forbidden fallback

Core cannot provide CPU/Torch substitutes for missing native kernels,
manufacture empty success results, load an unvalidated extension, or silently
change component semantics. Required capabilities fail before computation.

## Maintenance

Root export changes require `ci/public-api-snapshot.json` and a migration note.
The final historical ops ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`. Cross-domain changes must
pass the import-graph manifest rather than add debt.
