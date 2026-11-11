# ADR-049: BDPT wavefront sampler, counter-based random numbers and sorted accumulation

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `bdpt-wavefront-sampler`.
- **Scope:** `witwin.channel.montecarlo.bdpt` event-selected shooting sampler
  (mixed reflection+transmission chains), its random-number source, the
  `staged` connection accumulation and the accumulate JVP. Supersedes the
  sampler orchestration described in ADR-020 (polarization physics unchanged)
  and the `staged` reduction recorded in ADR-019; ADR-022 AD scope unchanged.

## Context

The BDPT shooting sampler was composed in Python per transmitter and per
bounce: a RayD intersection, both the reflected and the transmitted subpath
kernels on every row, a Torch row-wise merge, Torch event-probability algebra,
a dense `N x R` endpoint-connection table, a dense `N x R` visibility query
and a filter, then a concatenation of every block before accumulation. Event
uniforms came from a `torch.Generator` seeded per `(seed, tx, depth)`. Measured
in the audit (finding P-13): about 123-410 kernel events and 5-17 host syncs
per bounce, about 1,253 kernel events per transmitter, wall time independent of
the sample count, and 1,076 MiB peak for 4 receivers, 1 M samples, depth 2,
of which 0.31 % of the 8.4 M connection rows were valid; the 1 GiB workspace
guard refused small configurations. The `staged` accumulator and the
accumulate JVP looped every row for every output cell (`O(cells x rows)`).

## Decision

1. **Wavefront.** One native driver (`bdpt_wavefront_transmission`) runs the
   light subpaths of all transmitters as one wavefront, in chunks of
   `2^18` samples. Per chunk and bounce it issues one RayD intersection
   (shading normal + geometry only), one *bounce step* kernel and, from the
   second bounce on (a mixed vertex needs two events), per receiver tile of at
   most `2^20` pairs one visibility-input kernel, one RayD visibility query and
   one fused connect-and-accumulate kernel.
   - The bounce step selects reflect/transmit with one uniform, evaluates
     **only the selected branch** (the same device functions the per-event
     kernels use), divides the Jones field by `sqrt(p_event)`, multiplies the
     proposal density by `p_event`, updates the subpath in place, and tallies
     transmit/reflect events and the live count on the device (block counts,
     one atomic per block).
   - Visibility inputs are active only for live mixed vertices that face the
     receiver; inactive pairs cost no traversal. The connect kernel evaluates
     the connection row (the same device function as the row kernel) and adds
     `contribution * mis_weight` straight into a double `[tx, rx]`
     accumulator. No connection row is stored.
   - The sampled estimate is added to the enumerated transmission matrix in
     one finalize kernel. Peak memory is one chunk of state, one hit record and
     one pair tile, independent of the sample count.
2. **Where it runs.** The wavefront owns the sampler whenever
   `ad_mode='none'`, including rough (Kirchhoff) faces and `export_paths` /
   `diagnostics`:
   - rough faces: the bounce step splits scatter / transmit / reflect with the
     three-way budgets, samples the Kirchhoff lobe and evaluates the BSDF from
     the stacked resident tables (RayD `scattering_table.cuh` device math),
     seeds the unpolarized post-scatter carrier and applies the scatter-order
     cap; each recorded scatter vertex makes its next-event row per receiver
     tile (offset segment, RayD visibility, fused accumulation), so scattering
     builds no N x R table;
   - rows: a counting pass, the one sanctioned row-count copy, and a second
     pass that writes exactly those rows (sorted by sample, bounce, receiver)
     into the connection-sample schema for the compact export and the variance.
   Under `jvp` / `vjp` the event-composed sampler still runs (AD companions);
   it draws the same random numbers and returns bitwise the same primal.
   `Result.metadata["sampler"]` names the one used.
3. **Random numbers.** Every sampler uniform is a pure function
   `u = h(seed, stream, g, depth)` of the global sample index
   `g = tx * samples + i` (splitmix64 mixing, 24 high bits, always in
   `[0, 1)`). Streams: event selection, and the two Kirchhoff direction lanes.
   Launch directions keep their existing native formula. A draw cannot depend
   on chunking, transmitter batching or which sampler runs.
   `torch.Generator` is no longer used.
4. **Counts.** Metadata counts (straight chains, event tallies, valid
   contribution rows, scatter rows) stay on the device and are read in one
   device-to-host copy per solve.
5. **Enumerated oracle once.** Reflection, diffraction, transmission and the
   coupled classes come from one `evaluate_enumerated_paths` call; each row's
   enumerated component id names its accumulation bucket (coupled 3/4/7 fold
   into diffraction). The coherent combine does the same with LoS included.
6. **Accumulation.** `auto` never selects `staged`. `staged` is a stable radix
   sort of rows by `(tx * R + rx) * 5 + component` followed by a CUB
   reduce-by-key: each bin is the same fixed-order double sum on every run
   (`O(rows log rows)`). The power and coherent accumulate JVPs use the same
   sorted reduction (row-parallel, deterministic) instead of the per-cell row
   loop. Metadata names the kernel `sorted_segment_reduce`.

## Consequences

- **Bit-level baselines move.** Event uniforms changed source, so every BDPT
  estimate with sampled transmission or scattering changes at the seed level;
  the estimator, its expectation and its variance do not. Seed-reproducibility
  tests still hold (same seed, same result). LoS, reflection, diffraction and
  straight transmission rows (enumerated) are unaffected except for the
  accumulation order.
- `staged` sums change order (now deterministic); JVP tangents of the
  accumulate change order (now deterministic).
- The mixed-chain sampler no longer scales its memory or launch count with
  `samples x receivers`; the workspace estimate counts one chunk and one tile.

## Estimator limitation (mixed chains)

The mixed reflection+transmission estimator connects the light vertex after a
specular event (a point on a wall, carrying the sampled Jones field) straight
to the receiver through one connection segment. A specular chain has measure
zero in that connection; the estimator is a point-to-receiver radiometric
connection, not an image-method path, so it is not the field the deterministic
solver would assign to a reflect-then-transmit path and there is no
deterministic reference for it (deterministic transmission is straight-segment
only). Its equivalence evidence is therefore statistical: same-seed agreement
between implementations and means over seeds against the previous sampler.
Straight transmission, reflection and diffraction stay enumerated and agree
with the deterministic solver by construction.

## Acceptance evidence

Recorded in `docs/dev/montecarlo/README.md` ("BDPT wavefront evidence") and the
tests `tests/montecarlo/bdpt/test_wavefront.py`:

- wavefront and event-composed samplers agree per `(tx, rx)` and in every event
  count for the same seed (same random numbers);
- chunk and receiver-tile invariance;
- statistical equivalence against the previous `torch.Generator` sampler
  (means over seeds within confidence intervals) and, on a vacuum wall
  geometry where mixed chains are absent, against the deterministic solver;
- `staged` equals `atomic` and is bitwise reproducible.
