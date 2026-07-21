# Runtime owner

## Ownership

`runtime` owns extension selection, symbol/bootstrap validation, immutable build
identity, tensor/AD call contracts, native buffers, and pure-stdlib native
handle normalization. It does not own solver policy, scene construction,
materials, propagation algorithms, or RF numerical kernels.

`runtime.capacity.CapacityFailureState` owns the ADR-029 transaction failure
protocol. Native creation asynchronously zeros one contiguous CUDA `int32[1]`
bitmask on the caller's current stream. Capacity producers receive and retain
the same typed object/storage, atomically OR owner-specific bits, and never read
the state on the host. Intermediates do not trap; the solve/result boundary owns
the single `capacity_failure_terminal_check` observation. That runtime-owned
native operation launches on the caller's current stream, leaves the bitmask
unchanged, does nothing when it is zero, and raises an asynchronous device
failure when any bit is set. It performs no host read, synchronization, scalar
extraction, result allocation, or payload sanitization; every producer must
publish its canonical inert result before the terminal launch.

ADR-027 penetration failure owns the stable transaction bit
`SEGMENT_PENETRATION_FAILURE = 1 << 7`. It covers overflow, request/device-mask
contract contradiction, and non-finite penetration state. The dormant RayD penetration family
and Channel transmission-topology pack receive the exact same
`CapacityFailureState` object/storage; neither may clear, replace, observe, or
trap it. Their future solver switch does not create a penetration-local
failure observer.

`runtime.profiling` owns the closed semantic NVTX annotation vocabulary used
by performance evidence. It may emit only balanced ranges and point marks: it
does not evaluate tensors, allocate results, launch CUDA work, synchronize,
copy data, or affect numerical/error behavior. Payloads describe stable domain
operations and must not encode a plan phase, candidate identity, or temporary
generation name.

## Public entry points

The stable root entry is `witwin.channel_native.build_info`. `runtime.__all__`
also exposes internal loader/symbol APIs for domain facades; these are not root
public promises. `runtime.native_resources._rayd_scene_resource` is the unique
owner and scene/core modules compatibility-re-export the same object.

## Dependency rules

Runtime code may depend on the standard library and the supported Torch runtime
only for ABI inspection and non-numerical NVTX annotation. Profiling code may
not evaluate tensors, launch kernels, synchronize, or copy data. Runtime must
not import solver, scene, propagation, or scattering modules. Higher layers may
depend on runtime, never the reverse.

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

Root exports update `ci/public-api-snapshot.json` and a migration note. The
completed helper migration ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`. Runtime stays below scene
and solver domains in the import graph; undocumented debt is not permitted.
