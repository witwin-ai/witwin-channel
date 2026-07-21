# ADR-029: Device-resident capacity results

- **Status:** Accepted (2026-07-20); implementation pending
- **Date:** 2026-07-20
- **Kind:** Public result, dynamic-cardinality, native selection, AD, and
  launch-contract decision.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-007 (propagation data ownership), ADR-009 (native fusion ownership),
  ADR-023 (direct typed RayD integration), ADR-025 (diffraction operation-family
  ownership), and ADR-028 (device-resident diffraction state selection).

## Context

RayD order-1 diffraction export already returns a fixed-capacity result with a
CUDA validity mask and CUDA count. Channel currently destroys that contract in
`deterministic_diffraction_order1_compact`: it copies the count to the host,
synchronizes the active stream, allocates tensors with dynamic row count `K`,
and then assumes every row is valid. This makes the public result shape depend
on a device-selected cardinality and leaves invalid-row behavior undefined in
native vector accumulation, topology packing, sorting, and AD.

ADR-028 intentionally retained the full transmitter-visible state capacity
`N`. That removed the earlier transmitter-side synchronization but amplified
RayD exporter work in sparse scenes. The frozen Munich Phase 12 case has
`N = 51,640`, active count `K = 5,682`, `R = 1,024` receivers, and pair budget
`P = 4,194,304`. The historical compact planner used
`ceil(R / floor(P / K)) = 2` exporter chunks. Planning directly from `N` uses
`ceil(R / floor(P / N)) = 13` chunks. The final design must keep cardinality
device-resident while restoring exporter planning close to the historical two
chunks.

A CUDA-selected `K` cannot determine an ATen allocation shape without exposing
that count to the host. Exact dynamic shape and a zero-host-sync production
boundary therefore cannot both be retained. The stable result must expose a
host-known capacity and keep the actual count on the device.

## Decision

### Public capacity contract

Path and Deterministic configuration gain the stable, non-versioned field
`path_capacity_per_pair`. It is a host integer storage capacity `C`, not a
computed count and not a truncation policy. It is required for `Path.solve` and
for `Deterministic.solve` when `export_paths=True`; `None` is accepted only at
configuration construction and fails loudly before enumeration when a
capacity-backed public result is requested. `C >= 0` is valid, including an
explicit zero-capacity empty result.

The existing `max_paths` and `max_paths_scope` fields keep their algorithmic
selection semantics. They may intentionally select or truncate paths before
result packing. `path_capacity_per_pair` never silently truncates. If the
selected valid rows for any endpoint pair exceed `C`, the result-capacity
overflow contract below applies. When a capacity-backed public result is
requested, a per-pair `max_paths` value must not exceed `C` and a global limit
must not exceed the host-known total capacity formed from the endpoint-pair
count and `C`. The canonical selector itself is independent of `C`, so
non-export enumeration and the ADR-008 BDPT oracle do not require public result
storage.

For `PathResult`, the path axis is exactly `C`; consequently
`PathResult.max_num_paths` means capacity, not the maximum actual valid count.
`valid` is the sole row-validity truth and `num_paths` is a CUDA contiguous
`int32` tensor with the endpoint-pair shape whose values equal the device sum of
`valid` for each pair. No constructor or solver may infer `C` with `.item()`, a
device-to-host copy, Boolean compaction, or an implicit synchronization.

When Deterministic exports paths, `PathTable` stores exactly
`endpoint_pair_count * C` rows in stable endpoint-pair-major order, carries the
row-aligned CUDA Boolean `valid`, and adds a CUDA contiguous `int32`
`num_paths` tensor with the endpoint-pair shape. Its flat row dimension is a
capacity. Host metadata may report configured/effective capacities but must not
claim a device-selected actual count. A caller that needs actual counts reads
the device tensor and knowingly chooses its own synchronization boundary.

Before the atomic public solver switch, the dormant Channel owner
`deterministic_path_table_capacity_pack` may expose this representation only
through a strictly internal typed bundle. It consumes the exact shared
`CapacityFailureState` carried by its `CapacityEvaluatedPaths` layout, keeps
the flat `pair * C + slot` mapping, and never changes live `Result.paths`.
Failure bits, local overflow, and row validity are checked in that order before
any identifier or payload read. Its native VJP/JVP companions consume only the
canonical output validity, cover the existing eleven continuous evaluated-path
inputs, and return exact positive zero for inactive derivatives. `phase_rad`
retains the live native export formula bit-for-bit and remains
non-differentiable. The later atomic switch must merge the count/host-capacity
metadata into stable `PathTable` and delete the internal bundle without a shim.

`RaggedPathSoA` remains an explicit host-structural/import representation; it is
not an allowed production solver boundary for deriving a result shape. Public
filtering and array packing preserve `C`, row order, and device count tensors.
They may change `valid` and recompute `num_paths` on the device but may not
shrink the path axis.

Every invalid result row has one canonical inert representation:

- `valid=False`, coefficient/field/power/vector/position/normal values zero;
- delay and path length `-1`, endpoint angles zero;
- interaction type `NONE` (`0`); and
- primitive, material, edge, component, transmitter, and receiver identifiers
  `-1` where the field admits a sentinel.

Native topology concatenation, deterministic ordering, row selection, vector
field evaluation, complex accumulation, and result packing treat invalid rows
as nonparticipants. They must test validity before reading numerical values or
identifiers. Invalid rows containing NaN, infinity, or out-of-range identifiers
must not affect an output, winner, reduction, launch address, or error path.

### Diffraction state capacity

Path and Deterministic configuration also gain the stable field
`diffraction_state_capacity`, written as `M` in this ADR. It is a non-negative
host-known upper bound on active order-1 diffraction states per transmitter.
It must be specified whenever order-1 diffraction is requested; `None` fails
loudly before the selector or exporter launches. The effective state capacity
is `min(M, N)`, so a capacity chosen for a large scene remains valid for a
smaller scene without padding beyond the available state table.
Below, `M` denotes this effective capacity unless the configured upper bound is
explicitly distinguished.

The Channel-owned selector consumes the ADR-028 twelve tensor aliases and
CUDA `active[N]` mask. It performs stable device selection into a fixed
capacity state block, preserving the original state order. The selected-state
contract contains all twelve row-aligned CUDA outputs at effective capacity
`M`, CUDA Boolean `valid[M]`, CUDA contiguous `int32[1]` actual count, and CUDA
Boolean overflow state. The selected block is a new downstream contract and
does not weaken ADR-028's alias requirement for the planner result. RayD
consumes the capacity block through the stable `rayd/torch/integration.h` typed
API; the API identity and target names remain unchanged and only the
independent numeric API version may advance.

RayD exporter chunk planning uses effective `M`, never the device count `K` and
never the original `N`. For the frozen Munich case, `M = 8,192` gives
`ceil(1,024 / floor(4,194,304 / 8,192)) = 2` exporter chunks while covering the
recorded `K = 5,682`. This restores the exporter launch count close to the old
compact behavior without exposing `K` to the host. The selector launches and
scratch storage remain explicit Phase 12 costs and must be included in the
performance and memory evidence.

### Shared transaction failure state and fail-loud behavior

Selection and result packing share one solve-owned `CapacityFailureState`: a
runtime-created contiguous CUDA `int32[1]` bitmask asynchronously zeroed on the
caller's current stream. Every capacity producer receives and retains that same
typed object and storage. A producer atomically ORs its owner-specific failure
bit; after any bit is set it must not consume upstream payload, and it publishes
only the canonical inert representation: all selected-state and result-capacity
rows inert, every `valid` bit false, every device count zero, and no vector/grid
accumulation. Typed per-operation overflow/count outputs remain available for
direct contract diagnostics when that operation owns a capacity boundary, but
they do not replace the shared transaction state. The canonical selector owns
no public pair-capacity boundary and therefore returns no overflow tensor.

Intermediate capacity operations never trap. The solve/result boundary reads
the shared state on the same ordered CUDA stream and owns exactly one terminal
native asynchronous failure operation after all device outputs are inert. This
permits multiple producer failures to accumulate without poisoning subsequent
initialization kernels or exposing a partial result.

The dormant runtime owner is named `capacity_failure_terminal_check`. It
consumes the typed state's exact CUDA `int32[1]` storage, enqueues one 1x1
kernel on the caller's current stream, leaves the bitmask unchanged, and does
nothing for zero bits. Any nonzero bit executes the one production device trap;
the operation has no output, allocation, payload read, sanitizer, host copy, or
synchronization. It has no live solver caller until the atomic capacity-result
switch installs exactly one call after all result sanitizers.
Its frozen launch/memory budget and uniqueness audit are recorded in the
[terminal resource ledger](../audit/phase13-adr029-capacity-terminal-resource-ledger.json)
and [terminal duplication ledger](../audit/phase13-adr029-capacity-terminal-duplication-ledger.json).

This error is fail-loud at the next normal CUDA synchronization boundary. The
solver call must not return a usable partial numerical result. It is expressly
forbidden to copy the overflow flag or count to the host, call
`cudaStreamSynchronize`, extract a scalar, or introduce an implicit sync merely
to raise a host exception before `solve` returns. The current stream and normal
CUDA error propagation define the asynchronous failure boundary.

`deterministic_diffraction_order1_compact` is deleted atomically when this
contract activates. Its replacement uses an owner-based stable name such as
`deterministic_diffraction_order1_capacity_block`; there is no compatibility
alias, forwarding façade, alternate dispatcher, or dynamically compact result.
No symbol, type, file, target, header, schema, or API identity introduced by
this work may use `_v2`, `v2`, `WIP`, `next`, or another temporary generation
suffix. Numeric API/schema versions may advance independently of names.

### AD and numerical invariants

Validity, selected indices, counts, capacities, and overflow are discrete and
have no tangent or cotangent. Primal, float64 reference accumulation, backward/
VJP, and JVP companions accept the same validity contract. They skip invalid
rows before every read, contribute exactly zero for those rows, initialize
invalid input gradients/tangents to zero, and preserve the existing evaluation
and reduction order of valid rows. No derivative path may reconstruct the
operation in Torch, use finite differences, or detach a supported gradient.

This ADR changes storage, launch planning, and public shape semantics. It does
not authorize a physics, fast-math, reduction-order, RNG, winner, tolerance, or
valid-row numerical change. Any proposed fusion beyond the capacity selector,
capacity RayD export, inert native packing, and validity-gated accumulation needs
separate profiler and exactness evidence.

## Implementation sequence

1. Freeze this ADR and the public migration/API snapshot delta.
2. Add any dormant RayD capacity-state contract changes and direct
   current-stream, invalid-lane, zero-capacity, and overflow tests without
   changing the live Channel lock.
3. Add dormant Channel selector/capacity-block operations, binding/coverage
   manifests, native owner inventory, and direct contract tests.
   The dormant Path base-result portion is implemented by the stable
   `path_result_capacity_pack` primal/backward/JVP family. It inherits the
   shared failure state carried by `CapacityEvaluatedPaths`, gates both that
   state and the upstream overflow tensor, produces the exact
   `(rx, 1, tx, 1, C, ...)` base layout, and introduces no Ragged boundary,
   host cardinality read, live solver caller, or intermediate trap.
   `PathResult.__post_init__` now validates only host-visible tensor metadata;
   it no longer recomputes `num_paths` from CUDA `valid` or launches an
   asynchronous Torch assertion. The native capacity owner is the sole source
   of the count relationship, with a direct constructor contract test.
4. Change Path and Deterministic configs/results atomically with the public API
   snapshot and migration note; propagate validity through topology,
   accumulation, backward, and JVP.
5. Pin RayD, switch both solvers, delete compact-shape code and obsolete
   symbols without aliases, then run exactness and synchronization audits.
6. Record independent-process Munich and non-target performance evidence and
   activate only if all gates below pass.

The dormant canonical selector in step 3 uses the complete host-known candidate
capacity `N` for its internal output, with selected rows in a CUDA-valid compact
prefix, a CUDA `int32[1]` selected count, and CUDA `int32[P]` per-pair counts.
This is not the public `P*C` layout. The selector has no
`path_capacity_per_pair` input, does not test `PAIR_CAPACITY_OVERFLOW`, and
returns no local overflow output. It preserves live compact valid-row ordinals
until deterministic accumulation; public pair-major padding and capacity
failure remain later result-packing operations. The selector reproduces the
existing stable topology sort, canonical event/object deduplication,
shortest-path winner, and `max_paths` policy before scattering append. Its
decisions are frozen and non-differentiable; continuous AD belongs to the
subsequent native gather.
The internal `CanonicalPathSelection` also retains host-known `num_tx` and
`num_rx`, with `pair_count == num_tx * num_rx`, so that gather validation can
prove every reported per-pair count without reading a device scalar. The
dormant `evaluated_paths_canonical_capacity_gather` primal/VJP/JVP family
creates a new sanitized selection and fixed candidate-capacity
`CanonicalEvaluatedPaths`: it validates compact-prefix validity, source-index
range and uniqueness, source validity, endpoint IDs, selected count, and every
pair count before payload reads. A device bitset of `ceil(N / 32)` words makes
the uniqueness proof deterministic without a floating-point atomic or
quadratic scan. Failure makes the new index, validity, counts, evaluated rows,
VJP, and JVP wholly inert. This stage is not the later public pair-major
capacity pack and has no live caller before atomic activation.
The dormant launch, synchronization, copy, output, and CUB scratch formulas are
recorded in
[`phase13-adr029-canonical-selector-resource-ledger.json`](../audit/phase13-adr029-canonical-selector-resource-ledger.json);
activation evidence must replace its formulas with measured peak bytes and
timings without widening the recorded work.

Each implementation step is a separately reviewable commit. A dormant producer
must precede a consumer switch, and deletion occurs in the same commit as the
switch that makes the old path unreachable.

## Acceptance

- contract tests cover `N=0`, `M=0`, `K=0`, `K=1`, `K=M`, `K=M+1`, all-invalid,
  sparse, dense, all-valid, multi-transmitter/receiver/chunk, and non-default
  CUDA-stream cases;
- overflow makes all twelve selected outputs, every solver result row, every
  `valid` bit, every device count, and every accumulation inert before the
  asynchronous device error is observed; no usable partial result exists;
- invalid rows containing NaN, infinity, and invalid IDs are never read and
  produce bitwise-zero primal/JVP/VJP/backward contributions;
- stable active row order and all valid-row hashes match the frozen baseline;
- static and timeline audits prove the replacement path contains no count or
  overflow D2H copy, `.item()`, host Boolean compaction, implicit sync, or
  `cudaStreamSynchronize`;
- the Munich record includes `N=51,640`, `K=5,682`, `R=1,024`,
  `P=4,194,304`, old compact `2` chunks, ADR-028 capacity `13` chunks, and the
  accepted `M=8,192` target of `2` exporter chunks;
- the accepted ADR-030 comparable-baseline protocol uses five independent A/B
  process pairs with AB/BA order, one warmup, and seven steady observations;
  pooled paired target-stage and end-to-end medians improve by at least 10%
  and 5%, pooled non-target median/p95 regression is no worse than 5%/10%, and
  output hashes are identical; per-process medians remain diagnostics and a
  deterministic 100,000-resample paired 95% bootstrap improvement lower bound
  must be above zero;
- selector scratch, exporter workspace, launch/sync/copy counts, and peak
  temporary bytes remain within an explicitly frozen budget;
- public API snapshot, migration note, binding/coverage manifests, owner
  inventory, no-fallback tests, AD tests, and current synchronization evidence
  change with their owning implementation commits; and
- `quick`, full `cuda`, relevant `nightly`, clean multi-architecture wheel,
  ABI/PE/DSO/fingerprint checks, and final `release` pass from locked clean
  checkouts of both repositories.

## Stop conditions

Stop rather than activate if implementation requires a host-visible selected
count, dynamic ATen result shape, hidden synchronization, silent truncation,
partial output on overflow, invalid-row numerical reads, unstable valid-row
order, a Torch/CPU fallback, a second numerical owner, a compatibility shim, a
generation-suffixed name, changed valid-row hashes, an unfrozen memory increase,
or performance outside the accepted gates.

## Consequences

Public shapes become explicit capacity contracts and actual cardinality remains
device resident. Callers that previously treated `max_num_paths` or a flat
table length as an actual count must instead use `valid`/`num_paths`. Capacity
must be provisioned explicitly; undersizing fails asynchronously and cannot
return partial physics. In exchange, the production pipeline no longer needs a
device count to allocate or pack results, sparse diffraction regains bounded
exporter work through `M`, and invalid rows have one enforceable primal/AD
meaning across both solvers.
