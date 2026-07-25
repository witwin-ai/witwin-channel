# ADR-031: Per-pair raw reflection EPC capacity

- **Status:** Rejected (2026-07-21); ADR-032 is authoritative, retained for
  historical evidence only
- **Date:** 2026-07-21
- **Kind:** Experimental planning-capacity, device-cardinality, enumerated
  reflection, cross-solver, and failure-transaction proposal.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-008 (enumerated propagation as the BDPT discrete-path oracle), ADR-009
  (native fusion ownership), ADR-023 (direct typed RayD integration), ADR-029
  (superseded capacity-result experiment), and ADR-032 (controlled compact
  production boundary).

## Proposal boundary

This record preserves the ADR-031 design reviewed at commit `4fcd9b6` for
historical audit. It is not accepted production architecture and creates no
implementation, test, manifest, release, or preservation requirement. The
public field `reflection_candidate_capacity_per_pair` (`Qr`) must not appear in
the formal public API, production Config, capability manifest, solver planning,
or ADR-008 BDPT adapter. No production caller may reach the per-pair capacity
producer. Wording below describes the proposal, not current behavior.

ADR-032 restores the `O(K)` compact caller. Reconsidering this proposal requires
new evidence that passes E2E latency, peak memory, steady throughput,
capacity/active ratio, exactness, and concurrency gates together. Eliminating a
D2H count transfer is not independent evidence of benefit.

## Context

The ADR-029 experiment removed host-visible reflection counts and activated a
post-EPC block sized from the complete host-known theoretical RayD EPC batch row
count. This was correct but extremely large: a city-scale request can contain
millions of theoretical rows while only a small number of visible reflection
rows exist for each transmitter/receiver pair. The theoretical batch capacity
can exhaust device memory before canonical selection or public result capacity
is relevant.

The CUDA `visible` mask cannot determine an ATen allocation shape without a
device-to-host count transfer or synchronization. Reusing the proposed public
`path_capacity_per_pair=C` would also be wrong: `C` belongs after canonical
selection, while reflection EPC rows are raw pre-canonical candidates.
`max_paths` is an intentional selection policy and the proposed
`diffraction_state_capacity=M` belongs to diffraction. None can silently
truncate raw reflection work.

Path and Deterministic consume the shared enumerated engine. Under ADR-008,
BDPT may consume it read-only as a discrete reflection oracle. Any future raw
reflection planning contract would therefore have three callers and could not
be a hidden solver-specific option.

## Proposed design

### Public host-known capacity

The rejected proposal would add
`reflection_candidate_capacity_per_pair=Qr` to Path, Deterministic, and
`montecarlo.bdpt` configurations. It would be a non-negative host integer giving
the maximum number of raw RayD-visible EPC rows retained for one endpoint pair
in one enumerated reflection solve. Zero would mean explicit empty capacity;
Boolean values would not qualify as integers.

The proposal allowed `None` during construction but required a reflection solve
with `max_depth >= 1` to fail during planning unless `Qr` was explicit. The BDPT
adapter would forward it through private topology options to the unchanged
public enumerated entry without importing enumerated internals. None of these
API or forwarding changes is authorized under the current Proposed status.

### Raw EPC capacity and original order

For one RayD EPC request, let `N` be theoretical input rows and `P` be endpoint
pairs. The originally proposed checked host output capacity was:

```text
Q = P * Qr
```

The producer would preserve pair order and original EPC row order within each
pair. Row `pair * Qr + slot` would hold the `slot`-th visible raw EPC row;
unused slots would be canonical inert. It would not sort, deduplicate, choose
winners, or apply `max_paths`.

One solve-owned CUDA `int32[pair_count]` count state was intended to accumulate
visible rows across all order-1 and multibounce requests, with overflow joining
the exact solve-owned `CapacityFailureState`. Overflow was specified to clear
every validity bit and device count, publish only inert output, and fail at the
single terminal observer. No count/overflow D2H, dynamic allocation, partial
result, intermediate trap, or silent truncation was allowed.

The production candidate did not realize the assumed solve-wide storage reuse:
each of the three EPC requests allocated `P * Qr` rows. Its actual reflection
storage was therefore `P * Qr * H` for `H=3`, while later capacity-sized stages
remained unchanged. This implementation fact is part of the rejection evidence,
not a reason to retrofit production with another large feature.

### Independence from downstream policy

The proposal required `Qr` to remain upstream of the canonical selector and
independent of event/object keys, topology order, shortest-path winners,
`max_paths`, public result capacity, diffraction capacity, path export, and
coupled candidate policy. A sufficient `Qr` was intended to preserve every
valid reflection row bitwise. These are prerequisites for any future
experiment, but do not authorize the field or its caller now.

## Original implementation sequence

The rejected sequence was to freeze the public contract, add three Config
fields/forwarding paths, atomically change the shared reflection producer and
all reflection-depth callers, update public/native manifests and resource
ledgers, then run the full Plan 13 evidence. It prohibited retaining the
theoretical-batch path as fallback.

That sequence is cancelled for stable recovery. Existing experimental native
operations and tests may remain caller-free. No Config/API/snapshot change,
production switch, compatibility alias, or fallback is permitted.

## Original correctness requirements

Any future reconsideration must still prove:

- strict validation of `None`, zero, positive, negative, Boolean, and
  non-integer inputs in an intentionally accepted public contract;
- exact order-1 and multibounce pair/original-row order;
- direct coverage of zero pairs, no-visible, exactly `Qr`, `Qr+1`, poisoned
  invalid rows, non-default streams, and checked multiplication overflow;
- complete inert output and one loud terminal failure for insufficient `Qr`;
- unchanged canonical winners and valid-row hashes for multiple sufficient
  values; and
- no partial result, silent truncation, Torch/CPU fallback, early max-path
  policy, or second enumerated implementation.

These correctness properties are necessary but not sufficient. The measured
candidate met logical PathTable exactness for sufficient capacity yet failed
the production resource and E2E gates.

## Measured rejection evidence

The same Munich/RayD/toolchain/GPU protocol compared compact A (`e7d82d2`),
ADR-029 B (`768718e`), and ADR-031 C (`4fcd9b6`). Reflection had
`N=4,552,704`, `P=1,024`, and `K=2,629`; per-pair visible maximum/p50/p95/p99
were 20/0/13/17.

| Variant | Reflection capacity/active | Peak allocated | Median solve | Throughput |
|---|---:|---:|---:|---:|
| compact A | 1.000x | 1,071,493,120 B | 66.467 ms | 15.374 solve/s |
| ADR-029 B | 1,731.725x | 17,854,649,344 B | 1,009.587 ms | 0.793 solve/s |
| ADR-031 C, `Qr=128` | 149.569x | 12,171,594,752 B | 232.748 ms | 4.285 solve/s |
| ADR-031 C, sufficient `Qr=20` | 23.370x | 11,671,543,808 B | 226.429 ms | 4.410 solve/s |

Minimum-sufficient C still used 10.893 times A's peak allocated memory, took
3.407 times as long, and lost 71.32% of throughput. Lowering `Qr` from 128 to
20 recovered only about 477 MiB and 2.7% latency because diffraction,
canonical, gather, field, and later capacity work dominated. A's six scalar
D2H copies totalled only 24 bytes and approximately 0.152 ms of copy/immediate
synchronization time.

All 24 logical PathTable field hashes matched for sufficient capacity. This
does not rescue C: exactness is one gate, not a substitute for memory, E2E, or
throughput. Whole-map bitwise equality was not established because the legacy
atomic map is nondeterministic even between independent A processes.

## Stop conditions

Do not activate this proposal if it requires a host-visible count, dynamic
result shape, silent truncation, changed EPC order, early canonical/max-path
policy, reuse of an unrelated capacity, partial output, intermediate trap,
Torch/CPU fallback, compatibility path, or any regression against ADR-032's
latency, peak-memory, throughput, capacity-utilization, exactness, D2H budget,
or device-headroom gates.

## Consequences

There is no current public or production consequence: `Qr` remains absent and
the compact ADR-032 route is authoritative. Dormant implementation artifacts
may be retained for audit but cannot create a second production route.

A future per-pair capacity experiment would need to solve the complete
capacity-amplification problem, not merely reduce one raw reflection buffer.
That work is intentionally outside stable recovery and requires a new accepted
decision based on measured end-to-end benefit.
