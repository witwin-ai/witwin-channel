# ADR-0xx: Reflection candidate generation v2: image tree, receiver test, one validator

- **Status:** Accepted.
- **Date:** 2026-11-11.
- **Decision ID:** `reflection-candidate-generation-v2`.
- **Scope:** reflection discovery of the enumerated engine (`interactions/reflection.py`, the
  native `kernels/discovery.cu`), the specular legs of scatter chains, and the `discovery`
  option and `complete` metadata of Path and Deterministic. Plan items M3-3, M3-4, M3-5 and the
  owner decision of 2026-11-15 on `discovery=` / `complete`. Numerical change. Supersedes the
  undocumented 0.5.0 planning rule (exhaustive plane-group sequences when every requested depth
  has at most 100,000 sequences, ray discovery for all depths otherwise).

## Context

Findings P-3, P-4 and C-1. The 0.5.0 planner made one all-or-nothing choice for every depth
from the sequence count alone, so the 4x4 city at depth 3 (33 plane groups, 33,792 sequences)
enumerated every sequence against every receiver: 4,096 receivers meant 138 million EPC
candidates, 418 ms and 2.15 GB, with EPC launches of several hundred milliseconds, while depth 4
(over the limit) used ray discovery and took 22.8 ms. The two schemes returned different rows
(C-1), so raising `max_depth` changed the rows of lower depths. Every candidate was materialized
(443-680 bytes each) in fixed 4,194,304-candidate chunks regardless of free memory.

## Decision

**Two candidate sources, one validator.** Every candidate is a plane-group sequence with a
transmitter and a receiver, and the RayD reflection EPC with the path-validity rules of the M3-1
record decides it. A complete source yields exactly the rows the validator accepts among all
sequences; ray discovery yields a subset of them. Row order is the same for every source (per
transmitter, sequences in lexicographic group order, then receivers), so equal sets give equal
rows.

**Source A, the image tree (complete).** Depth 1 is every plane group; depth `k` appends to each
node every group except its last one, keeping a child unless the beam provably misses it. The
beam of a node with image `I` and last group `A` is the set of rays from `I` through `A`'s
polygon, beyond `A`'s plane. A child group `B` is dropped only when none of `B`'s occupied macro
cells (below) has a corner beyond `A`'s plane, or none of their bounding spheres meets the cone
from `I` around the bounding sphere of one of `A`'s occupied macro cells, with tolerances of
`1e-4` of the scene diagonal and `1e-3` rad. Both tests contain the beam, so pruning never drops a
valid path. The tree is expanded on the GPU breadth-first with one flag/scan/compact launch
group per depth (chunked to `2^25` pair tests per launch), keeping lexicographic order.
`discovery="exhaustive"` is the same tree without pruning.

**Receiver test (M3-4).** For each `(sequence, receiver)` pair the image method back-traces the
reflection points from the receiver through the mirrored transmitter images (one block per
sequence and 256-receiver tile; the mirrored chain is computed once per block in shared memory)
and drops the pair when a back-trace parameter leaves `[-1e-3, 1 + 1e-3]`, a reflection point
lies in no occupied cell of its group, or a segment is shorter than half the validity threshold.
Each group carries a 32 x 32 occupancy grid over its bounds in the plane, a cell set when a member
triangle's bounds (grown by `1e-4` of the diagonal) overlap it; four by four cells form a macro
cell. A count pass and a write pass produce exact `(sequence, receiver)` lists in
`(sequence, receiver)` order with one count copy per chunk. The EPC input builder reads these
explicit lists (implicit indices into the transmitter, receiver and sequence tables), so only
surviving candidates are materialized. Small products (`<= 2^20` pairs) whose groups all have at
most 64 members skip the test and go to the validator whole, which keeps small depth-1 solves on
their previous launch and sync counts.

**Source B, ray discovery (never complete).** Unchanged tracing (262,144 Fibonacci directions per
transmitter, one trace for all depths), now dropping chains that repeat a plane group in
consecutive bounces.

**Per-depth choice (M3-3).** Under `discovery="auto"` each depth uses the tree unless its
expansion (`nodes x groups`) or its receiver test (`sequences x receivers`) exceeds `2^31` pair
tests, or its frontier exceeds an eighth of free device memory; that depth and every deeper one
then use ray discovery. The decision for depth `d` reads only depth `d`'s own sizes, so it does
not depend on `max_depth`. `discovery="rays"` uses ray discovery at every depth.
`discovery="exhaustive"` never switches and raises `ValueError` before any EPC work when its
frontier does not fit.

**Memory and launch length.** EPC chunks are sized from free memory, at most 512 MiB and
`2^20` candidates per launch; tree and receiver-test launches are at most `2^25` pair tests.

**Metadata.** Path and Deterministic `Config.discovery` (default `"auto"`). Results carry
`metadata["discovery"] = {"requested", "sources": {depth: "image_tree" | "exhaustive" | "rays"},
"complete"}` and `metadata["complete"]`, true only when every reflection depth came from a
complete source. Components other than reflection enumerate exhaustively and do not affect it.

**RayD typed API for implicit candidates: not added.** P-4 proposed a RayD request taking the
tables plus implicit indices. With the receiver test the EPC sees few candidates (340,465 at depth 3 in the case below instead of
138 million), so the remaining per-candidate copies are small; the Channel EPC input builder now takes the
implicit indices instead. A table-plus-index RayD request remains possible if a profile shows the
copies matter.

## Evidence

All on the 4x4 synthetic city (`benchmarks/synthetic_city.py`, 33 plane groups), one transmitter,
RTX A6000 GPU 1 shared with other jobs (labelled contended), minimum of 6 warm solves, peak
allocation above the pre-solve baseline, LoS + reflection, `discovery="auto"`:

| Case | 0.5.0 (report) | This change |
| --- | ---: | ---: |
| Depth 3, 64 x 64 grid | 418 ms / 2,155 MiB | 13.9 ms / 97 MiB, longest kernel 0.8 ms |
| Depth 2, 64 x 64 grid | 21.9 ms / 1,301 MiB | 9.3 ms / 146 MiB |
| Depth 4, 64 x 64 grid | 22.8 ms / 437 MiB (rays, incomplete) | 86.6 ms / 97 MiB (complete) |
| Depth 5, 64 x 64 grid | 29.2 ms / 652 MiB (rays) | 130 ms / 379 MiB (depths 1-4 tree, 5 rays; longest kernel 50 ms) |
| Depth 1, 128 x 128 grid | 13.6 ms | 5.9 ms |
| Depth 1, 128 x 128 grid, 41,472 triangles | 51 ms | 7.7 ms |

Image-tree sizes on that scene (unpruned in parentheses): depth 2 632 (1,056), depth 3 13,190
(33,792), depth 4 263,433 (1,081,344). Receiver-test survivors at 4,096 receivers: depth 2
105,499, depth 3 340,465, depth 4 1,056,110.

Completeness and equivalence (`tests/accuracy/test_consistency.py`): the tree with the receiver
test returns exactly the brute-force rows (unpruned sequences, every pair to the validator) at
depths 1-3 on the 4x4 city with a 16 x 16 grid and depth 4 on the 3x3 city with a 12 x 12 grid;
`auto` and `exhaustive` give identical rows on the C-1 scene; ray discovery is a subset, and with
4,194,304 rays per transmitter equals the complete set at `max_depth` 2 and 3; `max_depth=3` and
`4` give identical rows at depths 1-3 (the C-1 tests, strict xfail before).

C-1 scene, 16 x 16 grid, rows per depth: 0.5.0 `max_depth=3` 164 / 218 / 206 and `max_depth=4`
164 / 202 / 250; now 164 / 202 / 190 for both. The removed rows were the 16 coincident-edge
depth-2 rows, their 16 depth-3 extensions and the 60 `[g, g, g]` ground rows.

## Consequences

- The C-1 strict-xfail tests pass; bit-level baselines that held depth >= 2 reflection rows move
  (fewer degenerate rows; per-receiver summation order follows the new row order).
- Open, unoccluded scenes prune poorly (the tests are geometric, not occlusion-based); depth 4+
  can be slower than 0.5.0's incomplete ray discovery while being complete. `discovery="rays"`
  keeps the old speed for callers that accept incompleteness.
- Each tree depth and each receiver-test chunk adds one count copy; depth-1 solves of small scenes
  keep their previous counts.
