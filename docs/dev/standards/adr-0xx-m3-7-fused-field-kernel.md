# ADR-0xx: One fused field launch over the canonical rows

- **Status:** Accepted.
- **Date:** 2026-11-11.
- **Decision ID:** `fused-canonical-field-stage`.
- **Scope:** the primal (`ad_mode="none"`) field stage of the enumerated engine
  (`propagation.fields.evaluate_path_fields`), the native operation
  `field_canonical_rows` (`native/channel/kernels/fields.cu`), and the removal of the
  diffraction vector-field sidecar. Fusion-boundary change under section 2 of the
  contributor rules; amends the field-stage paragraph of `docs/dev/propagation/README.md`.

## Context

Finding P-8 measured the field stage as a Python loop over row groups: for every
component (and every reflection depth) it gathered about a dozen per-row columns with
Torch advanced indexing, launched one native field kernel, and scattered seven output
columns back with `index_copy_`. Agent work for M2-4 removed the `torch.nonzero`
synchronizations (the groups now come from the compact owner's control record), but
the per-group gathers, launches and scatters stayed: a depth-3 LoS + reflection +
diffraction solve still paid one gather/launch/scatter round per group, plus the
`tx_positions[tx_id]`-style endpoint gathers of every row, plus the rough-surface
launch per reflection depth, plus the Torch arithmetic of the diffraction projection.

The discovery stage also still built a `[T, R, 3]` `diffraction_vector_field` sidecar
with a Torch `index_add_` and sanitized it with its own native kernel, although no
solver has read it since the layout-independent diffraction change.

## Decision

1. **One launch for the operators this translation unit owns.** The new native
   operation `field_canonical_rows` runs one thread per canonical row and switches on
   `(component_id, depth)`:
   - LoS rows run the free-space lane;
   - reflection rows of depth `1..W` run the reflection lane (finite slab, or the
     layer stack of a multilayer material) and, on a scene with rough or
     realization-replaced faces, the rough-surface coherent factor;
   - order-1 diffraction rows run the receiver projection of the exported UTD field
     (arrival direction, projection, unit-source columns);
   - every other row (transmission, coupled R-D/D-R/D-D, scattering, reflection depth
     0, LoS when the boundary taper owns it) keeps its input values.
   Endpoints come from the per-transmitter and per-receiver tables through `tx_id` /
   `rx_id`, slab parameters from the per-face tables through `primitive_sequence`, and
   interaction geometry from the `[N, W, 3]` tables at the row's own stride. The launch
   writes all seven field-stage columns into fresh tensors. When the compact owner
   supplies a row-group order, thread `i` evaluates row `order[i]`, so a warp sees one
   row kind at a time; the output stays in canonical row order.
2. **Bit-identical arithmetic, shared code.** Floating-point contraction (fma fusion) is
   decided by the compiler per inlined context, so a row evaluated by a copy of the same
   source inside another kernel does not round like the original: a first version that
   inlined a refactored reflection body into both kernels moved reflection fields by about
   1e-8 relative on every scene. The reflection lane is therefore one `__noinline__`
   device function, `reflection_row`, that computes every output of a row (field vector,
   projection, source-excited projection, path gain, length, delay, direction) and that
   both `reflection_sequence_kernel` and the fused kernel call; its machine code, hence its
   rounding, is the same for both callers, and the per-operator kernel is bitwise the
   pre-change kernel. Slab parameters are read either at `index * D + bounce` of gathered
   tables or at the bounce's face of the per-face tables. The free-space lane and the store
   are shared `__forceinline__` functions (bitwise unchanged on every tested input). The
   rough factor became `rough_bounce_factor` over a bounce accessor and keeps its explicit
   `__fmul_rn` / `__fadd_rn` products; the fused kernel applies it at the store with the
   rough-scale kernel's multiplications (a smooth row multiplies by 1, which is exact). The
   diffraction projection reproduces the Torch `clamp_min(1e-30).sqrt()` and the
   complex-by-real division with the same `c10::complex` operator.
3. **What stays outside.** Transmission (`rayd::torch::field_transmission_sequence`)
   and the diffraction wedge used by AD are RayD operations whose device code is not
   exported by the locked RayD; the coupled field kernels are owned by the coupled
   chain work in `coupled.cu`. They keep their per-group launch after the fused launch
   and write their rows in place into the fused output (no clone). Their per-row
   endpoint gathers are only formed when such rows exist. The LoS boundary taper keeps
   its per-group route when it is on.
4. **AD route unchanged.** `ad_mode="jvp"`/`"vjp"` keeps one differentiable Function
   per row group. A fused VJP would change two reductions that are part of the
   numerical contract: the frequency cotangent is an ordered per-launch sum (ADR on AD
   reduction order), and per-face material cotangents are accumulated by the gather
   backward per group and then summed across groups by autograd. Merging the groups
   reassociates both sums, so fused companions and bit-identical gradients are
   mutually exclusive; this change keeps the gradients bit-identical and leaves AD
   fusion to a separate numerical decision.
5. **Sidecar removal.** `interactions.diffraction` no longer builds the
   `diffraction_vector_field` (`index_add_`), `EvaluatedPathSidecars` loses the field,
   and the native `enumerated_capacity_failure_vector_sanitize` kernel and binding are
   deleted.
6. **Wideband replay.** The consumer's fixed-topology `[K, F]` replay keeps one launch
   per `(component, depth)` bucket: its inputs are per-bucket EPC re-solves and
   visibility re-tests, not canonical rows, so folding them into this launch would
   need a gather of every bucket into canonical tables first. Not natural; not done.

## Consequences

- Field-stage launches of a primal solve drop from `groups + rough depths + diffraction
  Torch ops + endpoint gathers` to one, plus one per transmission / coupled group.
- Rows whose `tx_id`, `rx_id` or face index is outside its table publish inert zeros
  instead of a device-side Torch indexing assert; negative indices count from the end as
  in a Torch gather.
- Unwritten primal output columns are no longer aliases of the discovery columns; every
  primal output is a fresh tensor.

## Evidence

See "Acceptance" below; numbers are from the M3-7 branch, contended GPU 1, counts by
`torch.profiler` and the budget harness.

## Acceptance
