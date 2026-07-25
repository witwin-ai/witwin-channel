# ADR-033: Channel replacement product identity

- **Status:** Accepted by owner direction (2026-07-22); product-identity and
  repository-history migration complete, version-1 consumer surface activated
- **Kind:** Breaking package, module, native-extension, build-identity, and
  operational-name migration
- **Related:** ADR-003 (public/internal API), ADR-006 (developer override),
  ADR-023 (single typed RayD integration), and the replacement migration
  record

## Context

The Torch/CUDA implementation is preparing to replace the existing Channel
distribution rather than ship as a permanently parallel product. Keeping a
native-suffixed distribution, Python namespace, extension, and operational
identity would force downstream consumers to choose between two public
products and would prevent an installation from replacing the existing
Channel package atomically.

The existing distribution is `witwin-channel==0.3.0` and owns the
`witwin.channel` namespace. Its public API differs from this implementation.
This migration therefore cannot be represented as a compatibility-preserving
alias. The replacement API remains the curated root plus the Path,
Deterministic, Monte Carlo Basic, and Monte Carlo BDPT solver entry points
already frozen by ADR-003.

## Decision

The replacement has one product identity:

| Boundary | Final identity |
| --- | --- |
| Distribution | `witwin-channel` |
| Python package | `witwin.channel` |
| Native Python extension | `witwin.channel._channel` |
| Native module init | `PyInit__channel` |
| CMake project/target prefix | `channel` / `CHANNEL_*` |
| C++ owner namespace | `channel` |
| Developer/release environment prefix | `WITWIN_CHANNEL_*` |
| Build metadata backend | `channel` |
| Source directories inside the repository | `src/witwin/channel` and `native/channel` |

The repository checkout directory may remain `channel_native/`; it is a source
location, not an installed product identifier. Git remote URLs and immutable
historical evidence may retain the former repository/product spelling when
they identify an actual past artifact. A zero-reference CI gate must reject
the former identity everywhere else, including production source, tests,
build rules, current documentation, wheel members, environment variables, and
generated build identity.

There is no compatibility package, import alias, extension alias, duplicated
DSO, environment-variable alias, fallback loader, or dual distribution. A
process or environment containing files from both implementations is invalid;
wheel smoke must prove the replacement wheel contains only the final package
and extension paths.

The first replacement release is `0.4.0`, which is intentionally newer than
the existing `0.3.0` distribution and therefore supports an ordinary package
upgrade. Repository-owned consumers must select `witwin-channel>=0.4,<0.5`
atomically with their API migration. External consumer rollout remains a
separate release action and does not authorize a compatibility shim.

Stage-I Phase 3 publishes the solver-neutral consumer only at
`witwin.channel.propagation.consumer`. It does not introduce a second
distribution, top-level package, extension, or package-root alias. Radar
adoption remains a later consumer rollout and is not claimed by the Channel
release.

## Compatibility and migration consequences

- Imports move atomically to `witwin.channel`; downstream code must update in
  the same rollout.
- The public export set and callable semantics do not otherwise change in this
  rename.
- Pickle/module-qualified class identity, deployment ABI strings, pipeline
  cache keys, native build fingerprints, and developer override variables use
  the final identity. Old caches are intentionally invalid.
- The public API snapshot changes module and target paths but not its export
  inventory or callable signatures.
- Native binding names for RF operations do not change merely because their
  owning extension and C++ namespace change.
- Numerical order, CUDA launches, RayD ownership, result schemas, allocation,
  synchronization, AD, RNG, and solver behavior are out of scope and must
  remain unchanged.

## Acceptance

1. The installed wheel exposes `witwin.channel` and `_channel`, and exposes no
   parallel native-suffixed package or DSO.
2. Public snapshot, binding/coverage manifests, import graph, deployment
   identity, wheel/PE smoke, and release workflows agree on the final names.
3. A repository identity scan reports no former product token outside the
   explicit immutable-history and ADR migration exceptions.
4. Quick and CUDA tiers pass without a compatibility shim or fallback.
5. Current-platform wheel smoke validates distribution name, package origin,
   native module, fingerprint, exports, and RayD lock identity.

