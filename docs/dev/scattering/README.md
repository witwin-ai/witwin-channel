# Scattering owner

## Ownership

Scattering is a subject with three owners, split on when the work runs. There is
no `witwin.channel.scattering` module and there must not be one again: it read
like "the scattering owner" while holding only the compile-time half.

| Owner | Runs | Holds |
|---|---|---|
| `scene.resources` | compile time | `KirchhoffTable`, `build_kirchhoff_table`, `eval_bsdf`, `sample_directions`, `pdf`, `pdf_reverse`, `PhaseScreenRuntime`, `generate_gaussian_realization`, `realization_seed`, `patch_phase_integral`, the table grid constants, and the lazy per-scene resources built from them |
| `kernels.scattering` | per launch | every native facade the three consumers dispatch |
| `interactions.scattering` | per solve | path evaluation, event policy, topology and packing |

Together they own physical scattering models and their `eval`, `sample`, and
`pdf` semantics, including resident lookup tables and phase-screen behavior.
None of them owns path enumeration outside `interactions`, solver accumulation,
or raw native tuple layouts.

The compile-time owner keeps its float64 NumPy build behind an explicit banner
inside `scene/resources.py`. That banner is the CPU-compute policy line: the
offline table construction above it is sanctioned by CLAUDE.md, and nothing
below it may grow into production hot-path physics.

## Public entry points

These are domain APIs, not root exports. Kernel modules are internal.

## Dependency rules

Scattering may depend on material value contracts, physics utilities, and
domain kernel facades. No owner may import a solver. `kernels.scattering` and
`interactions.scattering` must not acquire a mutable scene or a native handle
either; `scene.resources` holds the typed RayD lifetime because that is its
whole job, and the compile-time table build reaches only `constants` and
`materials`. Solvers consume scattering contracts through their owning
pipeline stage. After Plan 13 Phase 10A/10B, RayD is the active numerical owner
of all 17 table evaluation/sampling, single-bounce ensemble, patch-integral, and
fused ensemble/realization chain contracts plus the seven scattering-table
helpers. Channel retains the stable extension ABI and typed domain/autograd
facades, table/phase-screen lifecycles, event policy, topology/packing, RNG/MIS,
accumulation, and result ownership. Ensemble chain geometry remains JVP-only
with fail-loud VJP; realization chain geometry supports VJP and JVP.

The accepted boundary is deliberately narrow: RayD consumes caller-owned
resident table/height/geometry tensors and owns complete native operation
families. Channel continues to own table construction/cache/versioning,
`KirchhoffRuntimeResources`, `PhaseScreenRuntime`, seeds, topology and C1/C2
packing, `scattering_event_probabilities`, solver accumulation, RNG/MIS/event
policy, and result/metadata assembly. The chain families retain their as-built
AD difference: ensemble geometry is JVP-only and rejects reverse mode;
realization geometry keeps both VJP and JVP.

`PhaseScreenRuntime` retains height scaling, sampling, seed, and realization
semantics. Scene-static bindings and resident tensors are assembled once by the
lazy typed CompiledScene resource; scattering consumers read that resource and
do not reconstruct UV scale, face ownership, or RMS-slope state per solve.
The builder merely caches the same static calculation in its original exception
and numerical order; it does not add a Torch physics backend, fallback, or new
RNG behavior. Moving this retained resource construction into native code is a
separate ownership decision requiring its own ADR.
Patch subdivision and visibility stay in the solve plan because they depend on
frequency, endpoints, and solver configuration.

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
