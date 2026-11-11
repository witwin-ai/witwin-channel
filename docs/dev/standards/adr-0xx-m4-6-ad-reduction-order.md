# ADR-0xx: AD reduction order for shared leaves and frequency

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `ad-reduction-order`.
- **Scope:** shared-leaf VJP of the MC transmission wall product
  (`mc_transmission.cu`), the frequency cotangent of the free-space, reflection
  sequence and rough-scale field VJPs (`fields.cu`) and of the coupled R-D / D-D
  VJPs (`coupled.cu`), and the MC reflection radiomap VJP (`reflection.cu`).
  Revises the reduction paragraph of ADR-027.

## Context

Finding P-12 measured three reductions:

1. The transmission VJP's shared layer leaves and frequency were owned by one GPU
   thread each, which walked every row serially and recomputed the dual layer stack
   of every wall: 598 ms on a 64^2 grid with one layer per wall, 9.1 s at 256^2.
2. The frequency cotangent of every field VJP was one `atomicAdd` per row into one
   address: 8.6x slower than the geometry-only VJP at 16 M rows, and the result
   changed from run to run.
3. The MC reflection VJP added per-face material cotangents with one atomic per
   bounce and lane; scenes with few faces serialise on a handful of addresses.

## Decision

1. **Transmission shared leaves.** A row-parallel kernel writes, per row and wall
   slot, the term `g * prod_{other walls} T * dT/dp` of every layer parameter of
   that wall's material and of frequency, with the material's first layer and layer
   count. A second kernel gives each output to one warp: the lanes load 32 entries
   at a time and lane 0 adds the matching terms in ascending row, then slot order.
   The translation unit keeps `--fmad=false`, and the term is formed with the same
   multiplication order, so the sums are **bit-identical to the serial owner**. Rows
   are processed in chunks that bound the term buffer at 64 MiB; the owner carries
   its partial sum across chunks through the output tensor, which keeps the order.
2. **Frequency cotangents.** Each of the five kernels writes its per-row frequency
   cotangent into a row buffer (zero for rows that contribute nothing), and
   `ordered_sum::add_sum` (new header `native/channel/kernels/ordered_sum.cuh`)
   adds the buffer to the scalar with a partition fixed by the row count: up to
   1024 blocks over contiguous ranges, each a strided serial sum per lane and a fixed
   shared-memory tree, then one block folds the partials the same way. The result
   is bitwise reproducible; it differs in the last bits from any previous atomic
   order (which was not reproducible to begin with).
3. **MC reflection VJP.** The per-face material cotangents are combined within the
   warp (lanes with the same face add their values through shuffles, one atomic per
   distinct face) and the per-ray frequency cotangent goes through the same ordered
   sum. Face cotangents stay atomic, so they remain order-dependent in the last bits.

## Evidence

- Transmission: the existing wall-product VJP tests
  (`tests/montecarlo/basic/test_transmission_wall_product.py`,
  `test_mc_transmission_wall_product_prefix.py`) pass unchanged; the change is
  bit-identical by construction (same terms, same serial order).
- Frequency: `tests/ad/test_frequency_gradient_reproducible.py` (four repeated
  free-space VJPs over 2 M rows are bitwise equal; the ordered sum equals the double
  precision sum of single-row VJPs to 1e-5), plus the existing AD suites for the
  touched kernels (finite-difference and JVP/VJP duality checks under `tests/ad`).
- Performance numbers are in the task report (contended GPU, same-process A/B).

## Consequences

One extra `O(rows)` float buffer and two small launches per frequency VJP; the
transmission VJP holds at most 64 MiB of terms and launches two kernels per chunk.
Frequency gradients no longer change between runs.
