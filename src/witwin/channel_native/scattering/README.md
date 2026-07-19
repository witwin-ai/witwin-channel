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
pipeline stage. After Plan 13 Phase 6A, retained scattering and chain kernels
consume RayD's public shared RF device headers. ADR-026 accepts RayD as the
final numerical owner of 17 generic resident scattering runtime contracts and
the seven scattering-table helpers, but Channel remains their production owner
until the corresponding Phase 10A/10B atomic pin/switch/delete activation.

The accepted boundary is deliberately narrow: RayD consumes caller-owned
resident table/height/geometry tensors and owns complete native operation
families. Channel continues to own table construction/cache/versioning,
`KirchhoffRuntimeResources`, `PhaseScreenRuntime`, seeds, topology and C1/C2
packing, `scattering_event_probabilities`, solver accumulation, RNG/MIS/event
policy, and result/metadata assembly. The chain families retain their as-built
AD difference: ensemble geometry is JVP-only and rejects reverse mode;
realization geometry keeps both VJP and JVP.

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
module, `ci/public-api-snapshot.json`. The completed kernel migration ledger is
archived at `docs/dev/audit/phase12-ops-migration-ledger.json`. Dependencies
must pass the import graph; numerical convention changes require separate
oracle evidence.
