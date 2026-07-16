# Runtime owner

## Ownership

`runtime` owns extension selection, symbol/bootstrap validation, immutable build
identity, tensor/AD call contracts, native buffers, and pure-stdlib native
handle normalization. It does not own solver policy, scene construction,
materials, propagation algorithms, or RF numerical kernels.

## Public entry points

The stable root entry is `witwin.channel_native.build_info`. `runtime.__all__`
also exposes internal loader/symbol APIs for domain facades; these are not root
public promises. `runtime.native_handles._raydn_scene_handle_id` is the unique
owner and scene/core modules compatibility-re-export the same object.

## Dependency rules

Runtime code may depend on the standard library and the supported Torch runtime
only for ABI inspection. It must not import solver, scene, propagation, or
scattering modules. Higher layers may depend on runtime, never the reverse.

## Numerical and AD contract

Runtime buffers preserve declared dtype, device, shape, and contiguity. Runtime
defines no RF phase, material, or solver numerical policy. ABI and required
symbol validation finish before native computation.

### AD contract

`autograd_contracts` and `torch_compat` validate dispatch state, tangents, and
native tape boundaries but do not invent derivatives. Unsupported active
inputs, missing companions, and higher-order requests fail explicitly.

## Forbidden fallback

Implicit global extension loading, artifact-directory search, CPU/Torch
recomputation, and silent ABI downgrade are forbidden. Developer loading must
be explicit and validate the complete declared build identity.

## Maintenance

Root exports update `ci/public-api-snapshot.json` and a migration note. A
frozen helper/facade move updates its canonical owner in
`ci/ops_migration_manifest.json` with signature/body/AST preserved. Runtime
stays below scene and solver domains in the import graph; undocumented debt is
not permitted.
