# Deterministic solver owner

## Ownership

`deterministic` owns deterministic solver configuration, orchestration,
coherent accumulation, result assembly, and deterministic scattering-path
integration. Shared topology, geometry, and fields remain in `propagation`;
scene compilation and material encoding remain outside this package.

## Public entry points

`witwin.channel_native.deterministic` exports exactly `Config`, `PathTable`,
`Result`, and `solve`. Kernel, pipeline, accumulation, field, and scattering
modules are internal.

## Dependency rules

The solver consumes scene, material, and typed propagation contracts. It may
select solver policy but cannot redefine shared domain owners or import another
solver. Native calls belong in deterministic or shared kernel facades, never in
result models.

## Numerical and AD contract

Contributions are coherent complex fields before power projection. Component
IDs, path depth, row order, visibility, phase convention, and accumulation
order are contractual. Expression-order or tolerance changes are separate
numerical changes, not architecture cleanup.

Under ADR-029, exported `PathTable` storage is endpoint-pair-major with exactly
`path_capacity_per_pair` rows per pair. Its row length is capacity, not actual
cardinality; CUDA Boolean `valid` and CUDA contiguous `int32 num_paths` carry
actual validity/counts. Native primal/backward/JVP accumulation skips invalid
rows before reading IDs or numerical fields. Diffraction exporter work uses the
explicit device-selected `diffraction_state_capacity`, and capacity overflow
makes the entire result inert before surfacing a standard asynchronous CUDA
error; it never returns a usable partial result or synchronizes to raise early.

`deterministic.capacity.deterministic_path_table_capacity_pack` is the dormant
fixed-capacity exporter. It consumes the shared `CapacityFailureState` carried
by `CapacityPathLayout`, preserves the exact flat `pair * C + slot` row map,
and returns a strictly internal typed bundle containing every existing
`PathTable` field plus CUDA `valid`, CUDA int32 `num_paths`, and host-known
pair/capacity metadata. This bundle is not a `Result.paths` type and is not
exported from the package root; the later atomic solver switch must merge its
capacity metadata into stable `PathTable` and delete the internal bundle.
Failure bits and local
overflow are tested before any payload or ID read; every failed or invalid row
is canonical inert. The constructor validates metadata only and never reduces
device counts. Valid rows are bitwise identical to `build_path_table` for both
`include_fields` modes. `phase_rad` deliberately preserves the current native
export contract and is non-differentiable; only the existing eleven continuous
evaluated-path inputs have native backward/JVP companions.

ADR-027 straight-transmission discovery is shared with Path and the ADR-008
BDPT oracle through `evaluate_enumerated_paths`. The engine owns one pair-major
`EnumeratedFullDistance` RayD batch, the compile-cached policy diagonal, the
device actual-count sidecar, and the typed solve transaction. This solver
sanitizes after scattering, completes accumulation, optional PathTable export,
and `Result` construction, then enqueues the runtime-owned terminal observer
exactly once. It does not reconstruct or specialize the penetration route.

Under ADR-030, `deterministic.kernels.diffraction_pair` owns the dormant native
source-lane pair-reduction primal/VJP/JVP family. One warp owns each endpoint
pair, lane 0 adds valid source states strictly in ascending ordinal order, and
an in-stream status kernel compares RayD's contiguous CUDA `int32[1]` reported
count with the source-lane validity population. A mismatch ORs the shared
`DIFFRACTION_PATH_CONTRACT_ERROR` bit before the reducer makes every pair
inert; the shared `CapacityFailureState` is checked before validity or field
payload.
The result is a typed CUDA complex64 `[pair_count, 3]` field plus float32
`[pair_count]` power. This family has no live solver caller yet: the current
ReceiverGrid Torch `index_add_` route remains authoritative until the separate
atomic RayD pin/switch/delete commit.

### AD contract

`Config.ad_mode` accepts `none`, `jvp`, and `vjp`. AD uses fixed topology and
winners: continuous geometry/material/frequency/field leaves use registered
native companions while integer topology stays frozen. Scattering with AD,
unsupported tangents, double backward, and unsupported dispersive-frequency
seams fail loudly rather than detach.

## Forbidden fallback

Missing CUDA/RayD/native companions cannot trigger CPU enumeration, PyTorch RF
recomputation, zero fields, reduced depth, or another solver. Capability and
memory-budget failures occur before partial execution.

## Maintenance

Public changes require `ci/public-api-snapshot.json` and a migration note. The
completed kernel migration ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`. New dependencies must
satisfy the import-graph manifest.
