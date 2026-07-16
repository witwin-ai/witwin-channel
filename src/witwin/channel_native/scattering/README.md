# Scattering owner

## Ownership

`scattering` owns physical scattering models and their `eval`, `sample`, and
`pdf` semantics, including resident lookup tables and phase-screen behavior. It
does not own path enumeration, solver accumulation, scene lifetime, or raw
native tuple layouts.

## Public entry points

`witwin.channel_native.scattering` exposes the names in its `__all__`, including
Kirchhoff table/evaluation/sampling and phase-screen runtime helpers. These are
domain APIs, not root exports. Kernel modules and energy helpers are internal.

## Dependency rules

Scattering may depend on material value contracts, physics utilities, and
domain kernel facades. It must not import a solver or acquire a mutable scene or
native handle. Solvers consume scattering contracts through their owning
pipeline stage.

## Numerical and AD contract

Directions point away from the mean surface; roughness axes and SI units follow
the material record. Kirchhoff BSDF values are power per steradian and PDFs
normalize over the outgoing hemisphere. Phase screens change the complex
phasor, never geometry or averaged height. Evaluation, sampling, seed
consumption, dtype/device, and aliasing stay exact during owner moves.

### AD contract

Current scattering evaluation/sampling is primal. Solvers reject scattering AD
before launch unless a path has registered native derivative companions and
documented gradient semantics. No detach or surrogate gradient may make an
unsupported request appear successful.

## Forbidden fallback

Missing native operations must fail loudly. CPU/PyTorch reference
recomputation, zero-result substitution, and silent model downgrade are
forbidden in production paths.

## Maintenance

Export changes require a migration note and, if promoted to a curated public
module, `ci/public-api-snapshot.json`. Frozen kernel moves update
`ci/ops_migration_manifest.json` without altering the contract. Dependencies
must pass the import graph; numerical convention changes require separate
oracle evidence.
