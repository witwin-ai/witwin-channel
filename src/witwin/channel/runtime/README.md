# Runtime owner

## Ownership

`runtime` owns extension selection, symbol/bootstrap validation, immutable build
identity, tensor/AD call contracts, native buffers, and pure-stdlib native
handle normalization. It does not own solver policy, scene construction,
materials, propagation algorithms, or RF numerical kernels.

`runtime.capacity.CapacityFailureState` owns the shared failure protocol used
by accepted genuinely fixed-capacity operations and retained caller-free
experiments. Native creation asynchronously zeros one contiguous CUDA
`int32[1]` bitmask on the caller's current stream. Participants receive and
retain the same typed object/storage, atomically OR owner-specific bits, and
never read the state on the host. Intermediates do not trap; a participating
solve/result boundary owns the single `capacity_failure_terminal_check`
observation. That runtime-owned
native operation launches on the caller's current stream, leaves the bitmask
unchanged, does nothing when it is zero, and raises an asynchronous device
failure when any bit is set. It performs no host read, synchronization, scalar
extraction, result allocation, or payload sanitization; every producer must
publish its canonical inert result before the terminal launch.

ADR-032 restores the enumerated `O(K)` compact production result. Its sole new
accepted count D2H/synchronization allocation boundary is owned by propagation,
not runtime, and does not weaken this failure protocol. This does not imply a
whole solve has only one transfer or synchronization: pre-existing observed
boundaries remain named measurement debt. ADR-029 capacity-result operations
and ADR-031/030 candidates remain caller-free; runtime must not make them
reachable through a capability, loader choice, or fallback.

ADR-027 penetration failure owns the stable transaction bit
`SEGMENT_PENETRATION_FAILURE = 1 << 7`. It covers overflow, request/device-mask
contract contradiction, and non-finite penetration state. The live RayD
penetration family, Channel transmission-topology pack, and Monte Carlo Basic
wall-product estimator receive the exact same
`CapacityFailureState` object/storage; none may clear, replace, observe, or
trap it. The completed ADR-027 Phase P/E/M switches do not create a
penetration-local failure observer.

`runtime.profiling` owns the closed semantic NVTX annotation vocabulary used
by performance evidence. It may emit only balanced ranges and point marks: it
does not evaluate tensors, allocate results, launch CUDA work, synchronize,
copy data, or affect numerical/error behavior. Payloads describe stable domain
operations and must not encode a plan phase, candidate identity, or temporary
generation name.

## Public entry points

The stable root entry is `witwin.channel.build_info`. `runtime.__all__`
also exposes internal loader/symbol APIs for domain facades; these are not root
public promises. `runtime.native_resources._rayd_scene_resource` is the unique
owner and scene/core modules compatibility-re-export the same object.

Build-time RayD source selection is not a runtime backend choice. The compiled
identity records `rayd_source_kind` and the lock-pinned full-source manifest
SHA, but never the source's absolute path. Explicit Git checkout and validated
`rayd-torch` package source both produce the same single `_channel_native`
runtime boundary; package discovery does not import or dispatch through RayD.

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
