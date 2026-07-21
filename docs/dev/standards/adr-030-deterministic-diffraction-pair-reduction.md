# ADR-030: Deterministic diffraction pair reduction

- **Status:** Accepted (2026-07-20); implementation pending
- **Date:** 2026-07-20
- **Kind:** Numerical-order, cross-repository typed storage, native reduction,
  AD, and performance decision.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-007 (propagation data ownership), ADR-009 (native fusion ownership),
  ADR-023 (direct typed RayD integration), ADR-025 (diffraction
  operation-family ownership), ADR-028 (device-resident diffraction state
  selection), and ADR-029 (device-resident capacity results).

## Context

RayD enumerates an order-1 diffraction request with the logical lane mapping

```text
lane = ((tx_index * rx_count + rx_index) * state_limit) + state_index
```

where `state_index` is the fastest-varying coordinate. The current exporter
does not preserve that identity in storage: each successful lane reserves a
compact output row through an atomic device counter. Channel then compacts the
result and coherently accumulates the six real/imaginary vector components with
three Torch CUDA `index_add_` calls. Successful exporter row order and floating
point atomic-add order therefore depend on GPU scheduling.

For a `ReceiverGrid`, Deterministic additionally computes
`abs(vector_field).square().sum(-1)` in Torch and replaces the diffraction
component power with that value. This sidecar intentionally includes every
valid order-1 exporter contribution before public `max_paths` selection. It is
not equivalent to reducing only the paths that survive public result packing.

ADR-029 removes host-shaped compaction and requires capacity-plus-valid
storage, but it deliberately did not authorize a reduction-order change. A
stable device capacity alone does not make the existing atomic reduction
deterministic. Reordering compact rows by their atomic reservation ordinal also
cannot establish a canonical numerical order. A separate accepted numerical
decision is required.

## Decision

### Ownership and operation boundary

RayD remains the sole owner of order-1 ray traversal, visibility, stationary
point selection, generic UTD evaluation, and the six exported field
components. Channel remains the sole owner of deterministic solver pair
reduction, grid power, result assembly, and policy.

RayD adds a stable source-lane output layout to its existing typed
`DiffractionPathConfig`; Channel requests that layout through
`rayd/torch/integration.h`. Channel implements the complete native
`deterministic_diffraction_pair_reduce` primal/VJP/JVP family. The reduction is
not moved into RayD and UTD physics is not copied into Channel.

The stable names, header identity, library targets, Python facade, native
symbols, result types, and files introduced by this decision use owner and
operation semantics only. Generation-suffixed or provisional identities are
forbidden. The independent numeric integration API version may advance.

### RayD source-lane layout

The typed RayD request gains a layout choice with stable values equivalent to
`Compact` and `SourceLane`. Existing RayD callers retain `Compact` as their
default. The locked Channel direct-integration caller explicitly requests
`SourceLane`.

For `SourceLane`, let:

- `T = tx_count`;
- `R = rx_count` for the current exporter receiver chunk; and
- `M = state_limit`, which is the ADR-029 effective
  `diffraction_state_capacity` for the request.

The host-known output capacity covers `T * R * M` rows. Row

```text
((tx * R + rx) * M) + state
```

is the sole storage row for that logical lane. A successful lane writes that
row and sets its CUDA Boolean `valid` bit. A rejected or inactive lane leaves
the row in the canonical inert representation. The CUDA contiguous
`int32[1]` count remains the actual number of valid rows; it does not select a
shape, storage ordinal, or numerical order. Integer atomics may maintain this
diagnostic count, but no floating point result may be atomically reduced.

The source-lane layout must be implemented in the same RayD exporter operation
and shared UTD/traversal body as the compact layout. It is not a second physics
backend or a copied exporter. Compact callers retain their existing contract;
Channel does not keep or add a compact-layout compatibility path after its
atomic switch.

The RayD result is initialized on the caller's current CUDA stream before any
export. Capacity, device, dtype, ABI, state-limit, and layout violations fail
loudly. An invalid lane is never allowed to load inactive state payload merely
because its fixed row exists.

### Channel pair-reduction contract

The stable Channel family is:

```text
deterministic_diffraction_pair_reduce
deterministic_diffraction_pair_reduce_backward
deterministic_diffraction_pair_reduce_jvp
```

Its typed primal input contains the shared `CapacityFailureState`, CUDA Boolean
source-lane validity, the six CUDA float32 field component tensors, and the
host-known `pair_count` and `state_capacity=M`. All row-aligned tensors have
capacity `pair_count * M`. Its named result contains:

```text
field_xyz : CUDA complex64 [pair_count, 3]
power     : CUDA float32   [pair_count]
```

The pair index is `tx * R + rx` within a RayD request. Receiver chunks must
partition endpoint pairs: one pair belongs to exactly one chunk and may not be
split across chunks. Native structural assembly writes each reduced pair once
to its final transmitter/receiver position without arithmetic. Changing a
receiver chunk boundary therefore cannot change a pair's bits.

The reducer consumes every valid order-1 contribution before global or
per-pair public `max_paths` selection. Public path capacity, exported path
selection, and whether path export is requested do not change the exact
diffraction grid power. Reducing only final `PathTable` or `PathResult` rows is
forbidden because it would make the radio map depend on path-result capacity or
selection policy.

### Pair-serial CUDA algorithm

One CUDA warp owns one endpoint pair. Warp lanes load one consecutive group of
up to 32 state rows in parallel. Each lane checks `valid` before reading any of
the six numerical components; an invalid lane contributes six positive zeros
and its payload may contain poison.

Only lane 0 owns the six float32 accumulators. For each group, lane 0 obtains
the six values from source lanes `0, 1, ..., 31` through fixed-order warp
shuffles and performs the additions in that order. Groups are processed in
ascending order until `state=M-1`. Parallel loads are permitted; a parallel
tree sum, block reduction, cooperative partial sum, floating point atomic, or
order selected by validity/arrival is not.

For pair `p`, the exact component order is:

```text
for state in 0 .. M-1:
    if valid[p*M + state]:
        sx_re = sx_re + x_re[p*M + state]
        sx_im = sx_im + x_im[p*M + state]
        sy_re = sy_re + y_re[p*M + state]
        sy_im = sy_im + y_im[p*M + state]
        sz_re = sz_re + z_re[p*M + state]
        sz_im = sz_im + z_im[p*M + state]
```

Every accumulator starts at positive binary32 zero. Each multiplication and
addition in power finalization rounds separately to binary32 round-to-nearest-
even. Contraction is disabled for the reducer translation unit. Power is:

```text
px = sx_re * sx_re + sx_im * sx_im
py = sy_re * sy_re + sy_im * sy_im
pz = sz_re * sz_re + sz_im * sz_im
power = (px + py) + pz
```

No fast-math flag from RayD pure-wedge code may spread into the Channel
reducer. The reducer's compiler flags, launch geometry, shuffle order, and
power parentheses are part of the frozen numerical contract.

### Capacity, validity, and transaction failure

The reducer participates in the exact ADR-029 solve-owned
`CapacityFailureState`. A device status operation validates the RayD count and
source-lane capacity contract before numerical reduction without copying a
count or flag to the host. Failure bits, upstream overflow, and validity are
checked in that order before any payload or identifier read.

If a state-selection overflow, path-capacity overflow, RayD count/layout
contract error, or any earlier transaction failure is present, all pair fields
and powers are bitwise zero. No pair accumulated before the failure may remain
usable. Because a later public result-capacity producer can fail after pair
reduction, the transaction's final native sanitizer re-gates pair field and
power together with every other result before terminal failure. Intermediate
operations do not trap. After every device result is inert, the solve/result
boundary owns the single terminal asynchronous CUDA failure. It is forbidden
to add a device-to-host count/overflow copy, scalar extraction, implicit
synchronization, or `cudaStreamSynchronize` to raise earlier.

`M=0`, zero pairs, all-invalid input, and empty receiver chunks are valid
capacity cases and return correctly shaped inert outputs. Capacity is never a
silent truncation policy.

### VJP and JVP

Validity, layout, state ordinal, pair ordinal, capacities, device counts, and
failure bits are discrete and have no tangent or cotangent.

The native VJP consumes optional complex field cotangent and float32 power
cotangent. For each valid source lane of a successful pair it writes:

```text
grad_x_re = grad_field_x.real + 2 * sx_re * grad_power
grad_x_im = grad_field_x.imag + 2 * sx_im * grad_power
```

with the corresponding formulas for `y` and `z`. Each input row has one output
gradient row, so no floating point atomic or reduction is required. Missing
cotangents, invalid rows, and failed transactions produce exact positive zero.

The native JVP performs the same warp-per-pair, lane-0 state-serial reduction
over the six tangent tensors. The field tangent is the six tangent sums. Its
power tangent uses frozen separate multiply/add evaluation:

```text
t0 = sx_re * dsx_re
t1 = sx_im * dsx_im
t2 = sy_re * dsy_re
t3 = sy_im * dsy_im
t4 = sz_re * dsz_re
t5 = sz_im * dsz_im
dot = (((((t0 + t1) + t2) + t3) + t4) + t5)
dot_power = 2.0f * dot
```

Production Torch autograd may dispatch these companions but may not reconstruct
either derivative numerically.

### ReceiverGrid AD boundary

The current exact ReceiverGrid diffraction sidecar is produced by RayD's
detached order-1 path exporter. Native reducer derivatives with respect to its
six field inputs do not create missing derivatives of traversal, UTD,
polarization, material, endpoint, geometry, or frequency inputs.

Until a separate accepted ADR defines and implements a complete RayD
source-lane/fixed-valid exporter primal/backward/JVP family with real
transmitter polarization and all advertised continuous inputs,
`ReceiverGrid + diffraction + ad_mode != "none"` must fail loudly during solve
planning, before partial numerical computation. It may not return the current
detached exact power, fall back to selected public paths, reuse a wedge
operation with a fabricated polarization, or silently omit a supported
gradient.

This restriction does not remove the existing fixed-topology per-path
pure-wedge AD contract for supported non-grid Path/Deterministic results. A
future exporter-AD decision must be separately accepted; it is not hidden in
this reduction change.

### Numerical baseline and ADR-029 override

This ADR intentionally supersedes ADR-029 only where ADR-029 required the old
valid-row diffraction pair reduction order and old ReceiverGrid diffraction
map hash to remain exact. The old order was selected by exporter reservation
and floating point atomics and is not a stable reference.

The new authoritative baseline is the source-state-ascending pair-serial
result defined here. Repeated launches, receiver chunkings, supported current
streams, and independent processes using the same locked build must produce
bitwise-identical pair field and power outputs. Primal-under-AD for the direct
reducer must be bitwise identical to its plain primal.

Pairs with multiple valid contributions may change in their least significant
bits and cancellation-sensitive pairs may show a larger relative difference
from the old atomic result. The frozen migration evidence must record absolute,
relative, and ULP differences against both the old result distribution and a
float64 state-ordered oracle; it must not widen an unrelated tolerance.

RayD per-path physics payload, visibility, winners, state selection, public
path topology/order, path fields, ReceiverPoint results, Path solver results,
non-diffraction components, RNG, and metadata remain exact unless another
accepted decision says otherwise. In Deterministic ReceiverGrid results, the
intended public byte changes are limited to diffraction component power and,
for incoherent accumulation, the total power that incorporates that component.

ADR-029 performance evidence that required the old target map hash is replaced
for this target by exact matching to the newly frozen deterministic hash. All
unaffected hashes remain exact.

## Implementation sequence

1. Freeze this ADR and update Plan 13 and repository architecture summaries in
   one documentation commit.
2. In RayD, add the dormant typed source-lane layout, keep compact behavior as
   the default for existing RayD consumers, and add direct current-stream,
   poison-invalid, empty, multi-pair, and layout tests. Commit and push a fixed
   RayD revision before Channel changes its lock.
3. In Channel, add the dormant `deterministic_diffraction_pair_reduce`
   primal/VJP/JVP family, typed facade, binding/coverage manifests, owner
   inventory, direct numerical/AD tests, and compiler-flag checks. It has no
   live solver caller yet.
4. In one Channel pin/switch/delete commit, lock the accepted RayD revision,
   request source-lane storage, switch the ReceiverGrid sidecar to the native
   reducer, add the AD fail-loud gate, and delete the old Channel Torch
   numerical route without aliases.
5. Record independent-process bitwise, float64-oracle, Nsight Systems,
   launch/copy/sync, memory, Munich target, non-target, clean wheel,
   fingerprint, nightly, and release evidence.

Each step is independently reviewable and bisectable. A dormant producer
precedes its consumer. The live switch and deletion are atomic. Channel never
pins an uncommitted, dirty, branch-only, or unpushed RayD state.

## Migration and deletion

The Channel switch deletes:

- the ReceiverGrid diffraction `torch.zeros` plus three-axis `index_add_`
  numerical accumulation;
- Torch complex construction used only by that pair sidecar;
- the Torch `abs().square().sum(-1)` power computation;
- the untyped `diffraction_vector_field` sidecar representation and tests that
  require the old route; and
- together with the ADR-029 switch, the live
  `deterministic_diffraction_order1_compact` path.

There is no compatibility facade, alternate dispatcher, environment switch,
or fallback to the deleted Channel path. RayD may retain its compact layout for
its existing non-Channel public consumers, but Channel has exactly one live
source-lane integration.

The per-path `deterministic_diffraction_vector_field` operation is a different
complex3-to-scalar path-field operation and is not deleted merely because its
historical name contains "vector field". MC Sionna diffraction, coupled RD/DD,
pure-wedge physics, and unrelated deterministic accumulation remain with their
accepted owners and boundaries.

## Acceptance

### Direct correctness and AD

- RayD proves the exact source-lane index formula for multi-transmitter,
  multi-receiver, multi-state, one- and two-phase visibility paths while its
  default compact contract remains unchanged.
- Tests cover `M=0,1,31,32,33`, zero pairs, all-invalid, all-valid, sparse,
  non-default streams, receiver chunks, and multiple transmitter/receiver
  pairs.
- Invalid rows containing NaN, infinity, and invalid identifiers are not read
  and contribute bitwise-zero primal, VJP, and JVP values.
- A non-associative fixture such as `[2^24, 1, -2^24]` pins source-state order
  across a warp boundary and distinguishes the accepted result from another
  legal mathematical ordering.
- VJP/JVP pass direct formulas, adjoint dot-product tests, missing-cotangent
  tests, invalid/failure zero tests, and primal-under-AD bitwise parity.
- `ReceiverGrid + diffraction + ad_mode != "none"` fails before exporter or
  reducer work until the separate exporter-AD decision is accepted.

### End-to-end and failure

- ReceiverGrid diffraction results are bitwise stable across repeated launches,
  receiver chunk sizes, and at least five independent processes.
- Changing public `max_paths`, path export, or path-result capacity does not
  change pair power when the solve does not overflow.
- State, path, or result-capacity overflow makes every selected state, path
  result, pair field, pair power, device count, and validity bit inert before
  the asynchronous device failure is observed. No partial map is usable.
- ReceiverPoint, Path, exported per-path, non-diffraction, coherent total-field,
  winner, topology, and metadata hashes remain at their frozen values.
- Static and timeline audits find no target-path Torch numerical operation,
  D2H count/overflow transfer, `.item()`, Boolean host compaction, implicit
  synchronization, or `cudaStreamSynchronize`.

### Performance and resources

The frozen Munich case uses `M=8,192`, `R=1,024`, and 8,388,608 logical source
lanes across its two exporter chunks. Six binary32 field components occupy
192 MiB at that full logical capacity and the Boolean validity mask occupies
8 MiB. The implementation must not allocate a second full six-component lane
workspace merely to restore ordering. Source-lane exporter storage is the
reducer input.

Evidence records exporter, capacity/status, reducer, topology packing, total
diffraction stage, and end-to-end timings; launch/synchronization/copy counts;
temporary and peak bytes; register/shared-memory/spill data; and the
capacity/active ratio. It compares the accepted warp-per-pair implementation
with rejected single-thread-per-pair and parallel-tree candidates.

The Phase 12 targets remain at least 10% median improvement for the target
stage and at least 5% end-to-end median improvement, with non-target median and
p95 regressions no worse than 5% and 10%. Borderline results use five processes
and a paired 95% bootstrap improvement lower bound above zero. Because the new
deterministic numerical contract is mandatory correctness rather than an
optional old-hash optimization, failure to improve does not authorize the old
atomic path. Phase 12 cannot close until the improvement gates pass; exceeding
the regression or memory budgets is an immediate stop condition.

### Repository gates

- RayD direct/integration suites and Channel `quick`, full `cuda`, relevant
  `nightly`, clean multi-architecture wheel, ABI/PE/DSO/fingerprint, and final
  `release` pass from locked clean checkouts.
- Native binding and coverage manifests, current owner inventory, duplication
  and resource ledgers, no-fallback tests, migration notes, Plan 13, public API
  snapshot when applicable, and RayD lock/build fingerprint change with their
  owning commits.
- Static scans find no generation-suffixed identity and no deleted Channel
  reducer reachability.

## Stop conditions

Stop rather than activate if implementation requires a host-visible device
count, dynamic ATen result shape, pair splitting across chunks, a floating
point atomic/tree reduction, invalid-row payload reads, partial output on
failure, a Torch/CPU/fallback numerical route, fabricated polarization,
detached ReceiverGrid gradients, a second physics owner, a compatibility shim,
an unfrozen compiler/evaluation order, a second full lane-field workspace, or
performance/resource results outside the accepted regression budgets.

## Consequences

ReceiverGrid diffraction power gains a canonical, reproducible numerical
meaning independent of GPU scheduling and exporter compaction order. RayD pays
only a storage-layout contract change and continues to own the generic
diffraction computation; Channel owns one explicit native solver reduction
family. The source-state-serial order costs less arithmetic parallelism than a
tree reduction, while the warp load pattern retains coalesced memory access and
enough resident warps for large grids.

The old atomic map bytes cease to be a baseline. Some accepted public grid
values change at float32 rounding level, and cancellation-sensitive values may
move more. In return, new baseline hashes are stable and auditable. Exact grid
AD is explicitly unavailable rather than silently detached until its larger
RayD exporter operation family is designed and accepted independently.
