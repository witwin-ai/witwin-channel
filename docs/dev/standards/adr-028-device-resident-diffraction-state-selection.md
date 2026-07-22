# ADR-028: Device-resident diffraction state selection

- **Status:** Accepted (2026-07-20)
- **Date:** 2026-07-20
- **Kind:** Native fusion, dynamic-cardinality, storage, and launch-contract
  decision.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-009 (native fusion ownership), ADR-023 (direct typed RayD integration),
  ADR-025 (diffraction operation-family ownership), and accepted ADR-027
  (batched segment-penetration geometry and Monte Carlo native glue).

## Context

The live transmitter-side diffraction prefilter evaluates four points per edge,
calls RayD visibility, reduces the results, extracts a host Boolean, and uses
Torch Boolean indexing to compact twelve row-aligned state tensors. A CUDA
selected count cannot determine the first dimension of newly allocated ATen
tensors without a host-visible synchronization. Keeping compact `K`-row output
would therefore preserve the very host dependency that Phase 8B must remove.

The twelve inputs already form the complete order-1 state capacity, in this
order: `edge_index`, `edge_position`, `edge_direction`, `edge_t_min`,
`edge_t_max`, `n0`, `n1`, `prim0`, `prim1`, `exterior_angle`, `source`, and
`source_power`. Downstream RayD order-1 export already accepts a device Boolean
`active` mask and checks it before loading a state or evaluating UTD/visibility.

## Decision

Phase 8B uses a capacity-plus-device-mask contract. The Channel-owned composed
operation is named `diffraction_tx_visible_state_plan` and returns a named
`DiffractionVisibleStatePlan` containing:

1. the original twelve tensors as exact object/storage aliases, preserving
   order, stride, dtype, device, gradient state, and capacity `N`; and
2. one contiguous CUDA `bool[N]` active mask whose `i`th bit is true exactly
   when any of the four ordered edge samples for state `i` is visible from the
   separately supplied transmitter tensor `tx`.

The four fractions remain the binary32 values with bit patterns
`(0x3ca3d70a, 0x3eaaaaab, 0x3f2aaaab, 0x3f7ae148)`, corresponding to
`(0.02, 1/3, 2/3, 0.98)`. Evaluation order is `span = t_max - t_min`,
`t = t_min + fraction * span`, then
`point = edge_position + t * edge_direction`; Phase 8B direct tests lock the
resulting point/mask behavior against the frozen implementation at occlusion
boundaries. Every subtraction, multiplication, and addition rounds separately
to binary32 round-to-nearest-even before the next operation; contraction/FMA is
forbidden for this point construction. `tx` is broadcast as the visibility
start for every state;
`state.source` is not consulted by the planner and is only returned unchanged
for the exporter. A mismatched `tx`/`state.source` case is mandatory contract
coverage. No face is ignored. The mask is the sole validity truth; there is no
host or device selected-count field because no consumer needs one. `N == 0`
returns the input aliases plus an empty CUDA mask and launches nothing.

`tx` must be a contiguous CUDA float32 tensor of shape `(3,)`.
`edge_position` and `edge_direction` must be contiguous CUDA float32 tensors of
shape `(N, 3)`; `edge_t_min` and `edge_t_max` must be contiguous CUDA float32
tensors of shape `(N,)`. The planner rejects a non-contiguous view for any of
these five numerical inputs rather than inserting a copy. The other eight
state fields are passthrough aliases: the planner validates their frozen
dtype/device/row-shape contracts but neither reads nor restrides them. Their
original stride is preserved exactly. Tests cover both accepted passthrough
views and rejected non-contiguous numerical inputs.

The caller passes the mask directly to the existing RayD order-1 exporter with
`state_limit == N`. Python performs validation and named-contract assembly only.
It may not construct sample geometry, loop over fractions, reduce visibility,
extract a scalar, compact rows, or provide a Torch/CPU/fallback implementation.
The operation uses the caller's active CUDA stream and fails loudly on missing
capability, ABI, dtype, shape, device, or native exceptions.

The supported per-transmitter state capacity is `0 <= N <= P`, where the
existing pair budget `P` is exactly `4,194,304`. `N > P` fails loudly before a
visibility or exporter launch. This accepted limit is required for the peak
workspace bound below; supporting larger state sets requires native state
tiling and a follow-up contract rather than silently allocating beyond `P`.

This contract intentionally changes resource behavior relative to `K`-row
compaction: partially visible inputs may launch inactive lanes; receiver chunk
planning uses capacity `N`; and a nonempty all-invisible input may enqueue a
bounded exporter that produces zero valid rows. Peak pair workspace remains
bounded by the existing capacity budget because receiver chunk size varies
inversely with `N`. For `0 < K <= N` and `N > 0`, with pair budget
`P = 4,194,304`, compact planning used `ceil(R / floor(P / K))` chunks and
approximately `R*K` lanes, whereas this contract uses
`ceil(R / floor(P / N))` chunks and approximately `R*N` lanes.
Therefore sparse work can approach an `N/K` lane amplification and can cross a
chunk boundary even when only one row was filtered; `K == 0` no longer implies
zero downstream launches. The frozen exporter allocation remains approximately
`4 + 89 * capacity` bytes per chunk and is bounded near 356 MiB at full `P`.
Output row order and every visible numerical row remain exact. These resource
changes are knowingly accepted in exchange for a device-resident,
zero-host-sync boundary and must be measured in Phase 12.

The misleading sample-tape producer is renamed, without an alias, from
`bdpt_diffraction_accumulation_forward` to
`rayd_diffraction_sample_tape_forward`. This rename does not change its tape,
RNG, launch, output, or numerical contract.

## Phase 12 optimization boundary

Phase 8B may initially compose the stable typed RayD visibility primitive while
removing Python/Torch computation and synchronization. A follow-up single-launch
edge-visibility implementation is accepted only as a separate Phase 12 commit
after a reproducible baseline and Nsight Systems evidence show the visibility
launch sequence is material. It must be a pure native typed RayD operation,
preserve the exact mask, fractions, stream, and error contract, and introduce no
ATen numerical reconstruction or hidden synchronization.

The downstream `deterministic_diffraction_order1_compact` host count copies and
`cudaStreamSynchronize` are part of the same device-residency closure. Phase 12
must replace that compact-shape boundary with a capacity-plus-valid contract
before final acceptance. Invalid capacity rows must remain inert through native
vector accumulation and topology packing; Python/Torch must not index, reduce,
or reconstruct their numerical fields. The final public/solver result may be
assembled only by an accepted device-resident contract. Phase 8B evidence must
name this pre-existing downstream debt and may not claim full-pipeline zero-sync
until the Phase 12 replacement is active.

Performance acceptance uses independent-process steady-state samples rather
than the broad Phase-E disaster budget: two processes with one warmup and seven
steady samples each, expanded to five processes when dispersion is material.
The comparison records CUDA time, launch count, synchronization, device-copy
events, peak temporary bytes, output hashes, and capacity/active ratios for
empty, all-visible, all-invisible, sparse, and dense single-wedge cases.
Activation requires every independent-process target-stage median to improve
by at least 10%, end-to-end median to improve by at least 5%, non-target median
regression no worse than 5%, p95 regression no worse than 10%, and identical
output hashes. Borderline results use five processes and a paired 95% bootstrap
confidence interval whose improvement bound must remain above zero.

## Acceptance

- direct native contract tests cover empty, all-visible, all-invisible, sparse,
  dense, mismatched `tx`/`state.source`, non-contiguous, invalid input, and
  active-stream cases;
- the twelve returned state fields preserve exact tensor identity, storage,
  stride, dtype, device, row order, and gradient state;
- device masks exactly match the frozen four-fraction reference and Path,
  Deterministic, and ADR-008 BDPT end-to-end outputs remain exact;
- manifests, ownership inventories, no-fallback tests, static transfer/sync
  audits, launch/resource records, and the sample-tape rename change together;
- `N == P` remains within the frozen workspace budget and `N > P` fails before
  partial computation;
- no old symbol, compatibility re-export, `integration_v2` WIP boundary name,
  or production Python visibility geometry remains; and
- `quick`, targeted CUDA/AD, full `cuda`, relevant `nightly`, packaging, and
  final `release` gates pass from clean, locked checkouts.

## Stop conditions

Stop instead of activating if any implementation requires a host-visible
selected count, scalar extraction, implicit synchronization, CPU/Torch
numerical fallback, unstable row order, copied state tensors, changed fractions,
changed RNG/numerics, an unbounded workspace, an unowned native symbol, or a
second compiled owner. Stop a Phase 12 optimization if output hashes differ,
launch/sync/copy behavior regresses, peak temporary memory exceeds the frozen
capacity model, or repeated independent-process timing does not demonstrate a
material improvement outside observed variance.

## Consequences

Dynamic diffraction validity remains entirely device resident and all
row-aligned state storage retains a single owner. Sparse cases trade compaction
for inactive capacity lanes; this cost is explicit, bounded, and observable.
Future compact dynamic output would require a different asynchronous shape/
consumer contract and a new ADR rather than a hidden count synchronization.
