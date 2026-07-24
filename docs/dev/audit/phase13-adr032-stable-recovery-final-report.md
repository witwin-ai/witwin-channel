# Plan 13 ADR-032 stable-recovery final report

## Outcome

Plan 13 stable recovery is accepted at the compact `O(K)` production boundary.
ADR-029 is Superseded, ADR-031 is Rejected without public API or a production
caller, and ADR-030 remains Dormant. The already accepted
RayD ownership migration is preserved at RayD
`474c122aa3cd6b6d098675e076a73e6f485bd6be`; stable recovery required no new
RayD change.

The production Reflection and Diffraction solvers call the compact owners.
ADR-029/030/031 capacity owners remain direct-testable but caller-free. Public
Path and Deterministic configs contain no `Qr`, `path_capacity_per_pair`,
`diffraction_state_capacity`, or ignored compatibility field. Failures remain
all-or-nothing and loud; no physics, fallback, tolerance, resource budget, or
result exactness rule was weakened.

The measured native/runtime boundary is `2fec08b7b620c66fd29f4bdc768d8fa7f0580c84`.
The reviewed implementation boundary is
`966a4498b501980dee38f08616adde60f00c59a6`; the later delta is tests,
duplication-ledger maintenance, and comments only. The report commit is
intentionally not self-referential. Its final all-architecture package rebind
is performed after the report commit and recorded in the release handoff.

## Munich decision evidence

All A/B/C variants use the same RTX 5080, RayD commit, SM120 toolchain, Munich
32 by 32 receiver grid, depth 3, and LoS/reflection/diffraction configuration.
The workload has `N=4,552,704` theoretical reflection candidates, `P=1,024`
endpoint pairs, and `K=2,629` visible rows. Per-pair visible counts have
max/p50/p95/p99 `20/0/13/17`.

| Route | Capacity/active | Peak allocated | Median | Throughput |
|---|---:|---:|---:|---:|
| A compact `O(K)` | 1.000x | 1,071,493,120 B | 66.467 ms | 15.374 solve/s |
| B ADR-029 `O(N)` | 1,731.725x | 17,854,649,344 B | 1,009.587 ms | 0.793 solve/s |
| C ADR-031 `Qr=20` | 23.370x | 11,671,543,808 B | 226.429 ms | 4.410 solve/s |
| C ADR-031 `Qr=128` | 149.569x | 12,171,594,752 B | 232.748 ms | 4.285 solve/s |

The full candidate slot is `65 + 48D` bytes: 113, 161, and 209 bytes at
depths 1, 2, and 3, before downstream canonical/gather/field/topology, CUB,
RayD, OptiX, and allocator overhead. B exceeds physical device memory even for
one solve. Sufficient-capacity A/B/C results match bitwise for all 24 logical
PathTable fields. Whole-map bitwise equality is not claimed because the
existing atomic accumulation is non-deterministic across independent runs.

The current-head replay improved to 52.235 ms median and 18.877 solve/s while
preserving `N/P/K`, capacity/active 1, a 1,071,486,464-byte cold peak, and the
frozen logical path hashes.

## D2H and synchronization self-audit

The accepted reflection compact owner performs six 4-byte count copies per
solve, 24 bytes total, with about 0.152 ms in the copy APIs and their immediate
post-copy synchronization. This is a local owner metric, not a whole-solve
claim.

The same A Nsight capture contains approximately 66 D2H transfers totalling
282 bytes and 125 stream synchronizations with about 18.830 ms of aggregate
CPU API time. That total includes metadata and historical structural
boundaries. A separate approximately 5.162 ms wait belongs to the existing
Thrust scan. Those remaining boundaries are disclosed audit debt; ADR-032 does
not call them free or claim zero D2H.

## Acceptance results

- Quick passed.
- Full CUDA passed: 2,019 passed and 3 skipped; four-solver smoke 14 passed,
  no-fallback 26 passed, and AD core 117 passed.
- Full AD passed: 301 passed and 1 expected failure.
- Munich parity passed: 17 grouped tests plus 9 deterministic tests.
- Full statistics passed all 3 cases over 16 seeds.
- Coverage policy passed after separately retrying one TEMP/DrJit environment
  failure; no product failure was suppressed.
- Duplication passed at 170 regions and 10.031482%, below the unchanged
  10.211512% budget, with zero stale and zero unclassified entries.
- Full Phase-E release passed all 385 checks against the frozen Munich and San
  Francisco asset hashes.
- Peak-memory/preflight, four-solver cold-start, four-solver scaling, RayD
  lock/build identity, all-architecture wheel, isolated wheel loading, and
  PE/DSO audit passed.

The validated wheel at the measured boundary contains SM75/80/86/89/120 plus
SM120 PTX, is 35,071,560 bytes, and has SHA-256
`ccb8f8a9fd17697d918e95f543b94fd4995ce3ae034e1a2a073c7e436ee047b1`.
The packaged extension has SHA-256
`825c01201834aab4de1a5efaf44c133095e11470601cc000f9a6a55d0c8f03f6`,
78 exports, and passed isolated loading without a RayD Python extension.

## Disclosed debt and limits

An independent native-vs-old-Channel BDPT steady-state comparison remains a
real pre-existing performance debt: native speedup is only 0.160x, 0.166x,
and 0.179x for the three measured configurations. Cold-start and absolute
release thresholds do not supersede that result. It is not repaired here
because the user requested a fast stable recovery without another large
feature.

Concurrent solves were not load-tested, so this report makes no concurrency
count claim. The whole-solve D2H/sync boundaries and atomic whole-map
non-determinism remain explicit. Raw A/B/C profiler files remain local ignored
artifacts; the tracked gate preserves their hashes and summarized facts.

The machine-readable companion is
[`phase13-adr032-stable-recovery-final-report.json`](phase13-adr032-stable-recovery-final-report.json).
