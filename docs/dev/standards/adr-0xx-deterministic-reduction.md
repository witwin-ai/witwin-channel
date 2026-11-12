# ADR-0xx: Order-independent reduction of aggregated outputs

- **Status:** Accepted.
- **Date:** 2026-11-11.
- **Decision ID:** `deterministic-reduction`.
- **Scope:** the Coherent cell accumulator and its JVP (`deterministic.cu`), the MC
  diffraction tape map, its JVP and its frequency cotangent (`diffraction.cu`), the MC
  reflection map and its JVP (`reflection.cu`), the MC component power totals
  (`montecarlo.cu`), and the frequency cotangent of the MC LoS matrix (`los.cu`). Extends the
  frequency-cotangent part of the AD reduction-order ADR.

## Context

Every aggregated output above was a float `atomicAdd` of one term per row, lane or deposit
into its receiver cell. Float addition is not associative, so the result depended on the
order in which threads reached each address. With one or two terms per cell (LoS, most
specular reflection) the sum is still exact up to commutativity, but cells with many terms
(diffraction edges, coupled chains, Kirchhoff scattering patches, Monte Carlo deposits)
changed in the last bits from run to run: a `d1_pol` case gave 8 distinct
`component_power["diffraction"]` maps in 5 runs, the coupled slot moved by about 1e-6
relative, and the scalar `grad_frequency` by about 7e-8 relative. Per-path columns were
already reproducible.

Alternatives considered:

1. **Sort the contributions by cell, then a fixed-order segmented sum.** Reproducible for a
   fixed row order, but the row order itself changes with launch chunking and receiver
   windows unless the sort key also orders the values. It needs an `O(rows)` key/value
   sort (two 8-byte buffers plus CUB scratch) and several launches per call.
2. **Fixed-order tree per cell.** Needs the rows grouped by cell first, which is the sort
   again, and the per-row MC deposits have no row buffer at all.
3. **Fixed-point integer accumulation** with a per-cell scale. Integer addition is
   associative and commutative, so the result is independent of thread order, chunking,
   windows and row order by construction, and needs `O(cells)` scratch only.

## Decision

One shared utility, `native/channel/kernels/fixed_sum.cuh`, implements option 3:

- Per output element it keeps a 32-bit scale word and a 64-bit integer sum.
- **Observe pass:** every contribution raises the element's scale word to
  `ilogb(|v|) + 1` (an integer `atomicMax`). Non-finite values set flag bits instead
  (`atomicOr`).
- **Deposit pass:** every contribution is rounded to the nearest integer multiple of the
  element's quantum `2^(e - H)`, where `2^e` bounds the element's largest magnitude and
  `H = 62 - ceil(log2 N)` for at most `N` terms per element, and added with a 64-bit integer
  `atomicAdd`.
- **Resolve:** the integer is converted once to the output type and scaled back by `2^(H - e)`.
  A flagged element resolves to NaN, +inf or -inf exactly as an IEEE sum would.

Range: every rounded term is below `2^H` in magnitude and there are at most `N <= 2^(62-H)`
of them, so the sum stays below `2^62` and cannot overflow. `H` is computed on the host from
an upper bound of the terms per element (row count, lane count, or rays times depth).

Error: each term is off by at most half a quantum, so an element with `n` terms differs from
the exact real sum by at most `n * 2^(e - H - 1)` before the final rounding. With
`N = 2^24` rows that is `n * 2^-38` of the element's largest term, against roughly
`n * 2^-24` of the running partial sums for a float32 atomic sum. The result is therefore
the exact sum rounded once, up to an error far below float32 resolution, except that terms
smaller than `2^(e - H)` (about `2^-38` of the cell's largest term at `2^24` rows) are
rounded to the quantum. The float64 companions used for gradcheck share the code; at their
test sizes (`N <= 2^9`) the quantum is at most one float64 ulp of the largest term.

Where the contributions are computed by a kernel that is expensive to run twice (the MC
diffraction tape and its dual JVP), the kernel writes one value and one target per lane into
an `O(lanes)` buffer and `scatter_sum` runs the two passes over the buffer. The Coherent
accumulator reads its rows twice. Scalar frequency cotangents of the tape and the MC LoS
matrix write one term per lane or pair and reduce through the same `scatter_sum` with a
single target.

The MC reflection kernel has no row buffer (a ray deposits into a variable number of
receivers) and a second pass would cost one launch per receiver block, which the MC launch
budget does not allow. It uses the utility's single-pass **exact binned** variant instead:
every finite float32 is `M * 2^(p - 149)` with `M < 2^24` and `0 <= p <= 253`, so
`M * 2^(p mod 32)` is split into two 32-bit halves and added (signed 64-bit integer atomics)
to bins `p / 32` and `p / 32 + 1` of nine bins per receiver. The bins hold the exact real sum
for up to `2^31` terms per receiver. The last block of the deposit launch (a counter and
`__threadfence`, the CUDA threadfence-reduction pattern) adds the bins from the top down in
float64 and rounds once to float32, so no extra launch is needed. Scratch is 80 bytes per
receiver.

The MC component power totals (`montecarlo.cu` finalize) already reduced each block with a
fixed tree, but added the block totals to the scalars with float atomics. The blocks now
store their totals, and the last block to finish adds them in block order (one warp per
component, a fixed shuffle tree).

## Evidence

Filled from the task report (`reports/determ.md`): repeat spread per output before and
after, the numeric shift against the mean of the old runs, the full-wave gate, the AD suite,
and launch/peak-memory counts.

## Consequences

- The listed outputs are byte-identical across runs, launch chunking, receiver windows and
  row order. They shift against the old atomic results at float-rounding level.
- Costs: the Coherent accumulator gains one row pass (observe) and replaces three float
  memsets by two scratch memsets; scratch is 12 bytes per output element and stream. The MC
  diffraction tape and its JVP gain an 8-byte per-lane buffer and three small launches (two
  passes and the resolve). The MC reflection map and the MC finalize keep their launch
  counts; the reflection map's scratch is 80 bytes per receiver.
- Not covered (still float atomics, run-to-run noise in the last bits): leaf gradients of
  positions, powers, polarizations and material parameters (`los.cu`, `diffraction.cu`
  tape VJP material and source slots, `reflection.cu` face cotangents, `montecarlo.cu`
  scattering table cotangents, `kirchhoff.cu` table AD), and the experimental BDPT
  accumulators (`bdpt_connect.cu`, `bdpt_paths.cu`, `montecarlo.cu` BDPT finalize).
  They can adopt the same utility.
