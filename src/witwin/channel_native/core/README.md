# Core contract and compatibility boundary

## Ownership

`core` owns small cross-domain value contracts: antenna definitions,
component/edge policy, complex field state, memory budgets, and shared
metadata. `core.scene`, `core.runtime`, and `core.kernels.ops` are
identity-preserving compatibility facades for canonical domain owners, not a
place for new implementation bodies.

## Public entry points

The root package exports `AntennaArray`, `AntennaPattern`, `Complex3State`, and
`JonesState` from core owners. Other core modules are internal contracts.
`core.kernels.ops` is a bounded re-export facade retained for compatibility,
not a supported extension point.

## Dependency rules

Value-contract modules remain solver-independent. Compatibility modules import
canonical owners only to re-export the same object. New code imports the
canonical domain directly and does not route around boundaries through
`core.kernels.ops`.

## Numerical and AD contract

Field state follows the package `e^{+j wt}` / `e^{-j k r}` convention. Tensor
contracts preserve dtype, device, shape, row order, and storage identity unless
conversion is explicit. Topology IDs, component IDs, and winners are discrete
integer metadata.

### AD contract

Core defines the shared `none`, `jvp`, and `vjp` vocabulary and fixed-topology
metadata. Discrete topology is non-differentiable; continuous leaves remain
live only through their owning native AD seam. Compatibility facades must not
detach, copy, or wrap canonical objects.

## Forbidden fallback

Core cannot provide CPU/Torch substitutes for missing native kernels,
manufacture empty success results, load an unvalidated extension, or silently
change component semantics. Required capabilities fail before computation.

## Maintenance

Root export changes require `ci/public-api-snapshot.json` and a migration note.
Moving an ops facade body requires a canonical-owner update in
`ci/ops_migration_manifest.json` with frozen signature/body/AST preserved.
Cross-domain changes must pass the import-graph manifest rather than add debt.
