# ADR-032: Controlled compact-cardinality boundary and stable recovery

- **Status:** Accepted (2026-07-21)
- **Date:** 2026-07-21
- **Kind:** Production cardinality, synchronization exception, performance
  recovery, public-result, and fail-loud decision.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-007 (propagation data ownership), ADR-009 (native fusion ownership),
  ADR-027 (batched segment penetration), ADR-028 (device-resident diffraction
  state selection), ADR-029 (superseded capacity-result experiment), ADR-030
  (dormant deterministic diffraction reducer), and ADR-031 (rejected raw
  reflection capacity experiment).

## Context

ADR-029 treated elimination of a device-selected count transfer as a primary
architecture objective. Its production candidate replaced an `O(K)` compact
boundary, where `K` is the actual visible row count, with storage and downstream
work proportional to a host-known theoretical capacity. ADR-031 attempted to
reduce reflection capacity with a per-pair `Qr`, but each reflection-depth EPC
request allocated its own `P * Qr` block and the later diffraction, canonical,
gather, field, and public-capacity stages still processed capacity-sized rows.

This was a performance regression, not an architecture improvement. Avoiding a
small, explicit cardinality synchronization is not a goal when it increases
end-to-end latency, peak CUDA memory, or decreases steady-state throughput.
Correctness and fail-loud behavior remain mandatory, but device residency is an
optimization constraint rather than a reason to propagate inactive capacity
through the whole solve.

The user selected recovery to the last compact production baseline and accepted
one explicit, auditable count transfer/synchronization boundary. No new large
feature is authorized by this decision.

## Frozen A/B/C evidence

All variants used the same Munich scene and workload, locked RayD commit
`474c122aa3cd6b6d098675e076a73e6f485bd6be`, integration header SHA-256
`57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`,
MSVC 19.44, CUDA compiler 12.9.41, CUDA architecture 120, Torch 2.10, and an
RTX 5080 with 17,094,475,776 bytes of CUDA memory. The workload used one
transmitter at `[8.5, 21, 27]`, a 32 by 32 receiver grid, 2.4 GHz,
`max_depth=3`, `max_diffraction_order=1`, LOS/reflection/diffraction, path
export, `C=256`, and `M=8192` for the experimental capacity builds.

The three reflection EPC requests contained:

- theoretical candidate rows `N = 4,552,704`;
- endpoint pairs `P = 1,024`;
- actual visible rows `K = 2,629`;
- request-level `(N, K)` values `(874,496, 730)`, `(1,666,048, 900)`, and
  `(2,012,160, 999)` for depths 1, 2, and 3; and
- cumulative visible rows per pair: maximum 20, p50 0, p95 13, and p99 17.

The complete fixed candidate output consumes `65 + 48D` bytes per slot at
depth `D`: 113, 161, and 209 bytes for the three measured depths. Those bytes
exclude later canonical-selector/gather, field, topology, CUB scratch, RayD,
OptiX, and allocator overhead. The B capacity-pack vicinity alone scales as
approximately `106 + 108D` bytes per theoretical row, or 430 bytes at depth 3;
the measured canonical gather can require approximately 467.125 bytes per
capacity row for the measured width. These formulas explain why raw `Qr`
reduction does not remove the downstream capacity amplification.

| Variant | Channel commit | Reflection storage rows | Capacity/active | Peak allocated | Peak reserved | Median solve | Steady throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| A: compact `O(K)` | `e7d82d2d1d290bbc106ef68410ebf88aeb1c99e9` | 2,629 | 1.000x | 1,071,493,120 B (0.998 GiB) | 1,413,480,448 B (1.316 GiB) | 66.467 ms | 15.374 solve/s |
| B: ADR-029 theoretical `O(N)` | `768718eefa281cbef5123e7f2ca8c902fbd667c8` | 4,552,704 | 1,731.725x | 17,854,649,344 B (16.628 GiB) | 19,727,908,864 B (18.373 GiB) | 1,009.587 ms | 0.793 solve/s |
| C: ADR-031 `Qr=128` | `4fcd9b67a7b76b5f70f31648c956aafe5ad33fa1` | 393,216 (`P*Qr*3`) | 149.569x | 12,171,594,752 B (11.337 GiB) | 13,591,642,112 B (12.658 GiB) | 232.748 ms | 4.285 solve/s |
| C: minimum sufficient `Qr=20` | `4fcd9b67a7b76b5f70f31648c956aafe5ad33fa1` | 61,440 (`P*Qr*3`) | 23.370x | 11,671,543,808 B (10.870 GiB) | 13,283,360,768 B (12.371 GiB) | 226.429 ms | 4.410 solve/s |

Against A, B used 16.663 times peak allocated memory, took 15.189 times as
long, and lost 94.84% of throughput. Even minimum-sufficient C used 10.893
times peak allocated memory, took 3.407 times as long, and lost 71.32% of
throughput. B's reserved memory exceeded physical CUDA memory and relied on
WDDM oversubscription; it is not an acceptable successful production result.
Reducing `Qr` from 128 to 20 recovered only about 477 MiB and 2.7% latency,
showing that the dominant cost is the remaining capacity-sized pipeline.

Stage instrumentation measured A reflection EPC, compact candidate pack,
legacy canonical selection, and field work at 10.707, 1.444, 6.888, and
0.182 ms respectively. B measured 15.017 ms EPC, 1.428 ms candidate pack,
45.846 ms canonical selection, 7.262 ms canonical gather, 0.678 ms field,
5.041 ms capacity pack, and 0.121 ms PathTable pack. These are diagnostic
stage timings; the table's synchronized end-to-end measurement is the
acceptance truth.

The reflection compact owner performed six 4-byte device-to-host count
transfers per solve, 24 bytes in total. Nsight recorded about 0.152 ms for the
copy APIs and their immediate post-copy synchronization. This is not the whole
solve total: the same A Nsight audit contained approximately 66 D2H transfers
totalling 282 bytes and 125 stream synchronizations with about 18.830 ms of
aggregate CPU API time, including metadata and historical structural
boundaries. A separate approximately 5.162 ms wait belonged to the existing
Thrust scan boundary and must not be attributed to those 24 bytes. B and C
eliminated the reflection count transfers but failed the end-to-end, memory,
and throughput gates by large margins.

For sufficient capacity, all 24 logical PathTable field hashes matched across
A, B, and C. The final reflection/diffraction maps have pre-existing atomic
nondeterminism even across independent A processes, so a whole-map bitwise
claim is not supported by this experiment. An insufficient `Qr` must continue
to produce only inert device outputs and a loud terminal failure; it may never
be interpreted as truncation or partial success. Concurrent-solve capacity was
not independently load-tested, so no fabricated concurrency count is accepted;
B already fails single-solve physical-memory headroom and C leaves insufficient
headroom for a stable concurrent-service claim.

## Decision

### Restore the compact production boundary

The authoritative production enumerated reflection route is the A lineage's
stable `O(K)` compact route. A CUDA producer determines visible cardinality,
the single owning compact boundary copies only the required integer count
metadata to the host, explicitly synchronizes the caller's current stream,
allocates the exact compact output, and performs structural device packing in
the original stable row order.

This is the only accepted production exception to the general no-hot-path-D2H
rule. It is an allocation-shape boundary, not CPU physics, numerical selection,
or a fallback. It must remain visible in source, ownership records, and timeline
evidence; it may not be hidden behind an ATen allocation, scalar extraction,
Boolean indexing, helper with an unrelated name, or implicit synchronization.

For the frozen depth-3 Munich solve, the complete reflection compact boundary
budget is at most six scalar D2H copies totaling 24 bytes. A future reduction
in copies is welcome only when the same row order, values, E2E performance, and
resource gates pass. Adding copies or synchronization requires a separate
accepted decision with measurement.

Public Path and Deterministic results recover their compact semantics. A path
axis or flat PathTable row count represents actual compact rows, and
`max_num_paths` represents the maximum actual path count of the compact result,
not a provisioned storage capacity. The ADR-029 public
`path_capacity_per_pair`, public `diffraction_state_capacity`, capacity-shaped
`valid`/`num_paths`, and ADR-031 public `Qr` contracts are not production API
requirements under this decision. Intermediate builds that exposed those
fields must remove them together with the public API snapshot and migration
note; no compatibility alias or ignored WIP field is retained.

`e7d82d2` is the numerical/caller recovery base, not the final public-schema
commit: it still contains construction-only `path_capacity_per_pair` and
`diffraction_state_capacity` silent no-op fields. Recovery phase R2 removes
those fields atomically with the public snapshot; this ADR does not claim they
were absent from the historical base.

### Preserve fail-loud safety without propagating capacity

This recovery does not weaken failure behavior. Native contract errors,
overflow in genuinely fixed-capacity operations, missing native capability,
allocation failure, and CUDA failure remain loud. No operation may silently
truncate, return a partial result, substitute an empty success, or continue
with poisoned identifiers or payload.

The typed `CapacityFailureState`, inert-output sanitizers, and unique terminal
observer remain valid for already accepted fixed-capacity transactions such as
ADR-027 penetration and for dormant experiments. Their existence does not
require reflection, canonical selection, field evaluation, or public results
to remain `O(capacity)`. The compact boundary must publish either the complete
`K` rows in stable order or no usable result.

### Retire experiments from production policy

ADR-029/030 implementation artifacts may remain temporarily as caller-free
cleanup debt, but they are not supported features and create no implementation,
direct-test, manifest, release, or preservation requirements. ADR-031 is
Rejected; `Qr` has no production caller and is not part of the formal public
API. ADR-030's `SourceLane` activation remains dormant because it inherits the
same `O(P*M)` lane storage amplification and does not repair ADR-029's
regression. None of these artifacts may be reached through a production solver,
capability claim, default, fallback, or compatibility mode.

No GPU tiled EPC/incremental merge, new exporter AD family, new public
capacity, CUDA Graph, or other large optimization is part of stable recovery.
Such work needs a future, separately measured and accepted ADR.

## Recovery commit sequence

1. Accept this ADR and mark ADR-029 Superseded, ADR-031 Rejected, and ADR-030
   Dormant.
2. Establish `e7d82d2d1d290bbc106ef68410ebf88aeb1c99e9` as the compact production
   recovery base with the same locked RayD revision and header.
3. Remove capacity-result and `Qr` public configuration/schema changes and
   production caller reachability without deleting dormant native owners.
4. Synchronize public snapshot, migration docs, manifests, caller inventory,
   and no-fallback/no-partial-result coverage.
5. Run targeted compact-boundary tests, `quick`, full `cuda`, clean-checkout
   `nightly`/`release`, wheel/fingerprint, and the frozen Munich performance
   evidence. Stop at the first failed gate; do not add a large feature to make
   recovery pass.

Each phase is a separate reviewable commit. Documentation acceptance precedes
the production restore. The public/API restore and caller switch are atomic;
there is no supported dual production route.

## Acceptance

- Munich median single-solve latency is no greater than 76 ms;
- Munich steady-state throughput is at least 13.8 solve/s;
- peak allocated CUDA memory is no greater than 1.25 GiB and the run retains at
  least 1 GiB physical device headroom without WDDM oversubscription;
- post-EPC reflection storage has capacity/active ratio exactly 1 for successful
  solves, apart from documented allocator granularity and transient scan
  scratch that do not propagate row capacity;
- the frozen depth-3 workload has no more than six D2H count copies totaling
  24 bytes, and records their actual API/synchronization time;
- logical PathTable topology, geometry, field, winner, order, and valid-row
  hashes are bitwise equal to A; atomic whole-map nondeterminism is reported
  honestly and is not hidden by a weaker tolerance;
- invalid/count-contract/allocation/CUDA failures expose no partial result, and
  every fixed-capacity overflow path remains inert before its loud terminal
  observation;
- production source and public API contain no required `Qr`, capacity-shaped
  Path/PathTable result, ignored capacity compatibility field, or reachable
  ADR-029/030/031 experimental caller;
- no Torch/CPU numerical implementation, legacy backend, runtime fallback, or
  reduced physics is introduced; and
- `quick`, full `cuda`, relevant clean `nightly`, clean `release`, wheel,
  ABI/PE/DSO/fingerprint, manifest, public-snapshot, and migration gates pass
  without weakening a tolerance, allowlist, or resource budget.

## Stop conditions

Stop recovery if it changes valid-row order or physics, introduces partial
results or silent truncation, exceeds any memory/latency/throughput/copy gate,
requires a second production route, changes RayD ownership, or expands into a
new feature. Do not trade a local copy-count improvement for worse end-to-end
latency, peak memory, steady throughput, capacity/active ratio, exactness, or
concurrent-service headroom.

## Consequences

Production accepts a small, named, measurable synchronization in exchange for
bounded `O(K)` storage and substantially better end-to-end behavior. Actual
cardinality is again represented by compact result shape at this boundary.
Device residence remains the default everywhere else, and all numerical work
remains in native CUDA/RayD.

ADR-029 remains a historical record of a superseded production experiment.
ADR-031 is a rejected proposal, and ADR-030 is a dormant numerical candidate.
None creates an active implementation or release requirement. Future
reconsideration requires a new accepted ADR and must beat this compact baseline
on E2E latency, peak memory, steady throughput, capacity/active ratio, exactness,
and concurrency together; eliminating D2H is not an independent acceptance
criterion.

Final stable-recovery implementation, package, validation, and known-debt
evidence is recorded in
[`phase13-adr032-stable-recovery-final-report.md`](../audit/phase13-adr032-stable-recovery-final-report.md).
