# Runtime owner

## Ownership

`runtime` owns compiled-extension selection, native symbol/bootstrap validation,
and immutable build identity. It does not own solver policy, scene construction,
materials, propagation algorithms, or numerical kernels.

## Public entry points

The stable top-level entry is `witwin.channel_native.build_info`. Direct imports
from `runtime` are internal during the migration and must not widen `__all__`
without updating the public API snapshot.

## Dependency rules

Runtime code may depend on the standard library and the supported Torch runtime
only for ABI inspection. It must not import solver, scene, propagation, or
scattering modules. Higher layers may depend on runtime, never the reverse.

## Numerical and AD contract

This package performs no RF computation and creates no result tensors or AD
tapes. ABI and required-symbol validation must finish before any native
computation begins.

## Forbidden fallback

Implicit global extension loading, artifact-directory search, CPU/Torch
recomputation, and silent ABI downgrade are forbidden. Developer loading must
be explicit and validate the complete declared build identity.
