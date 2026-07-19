# Plan 10a — Scattering v2 native interface specification (frozen)

- **Status:** DRAFT specification (P2 pre-implementation). Interface freeze for
  the ADR-021 chain ops (A/B) and the ADR-022 BDPT companion set. This document
  is normative for the Python facades and the C++ Torch bridges; no `.cu`/`.py`
  production code is authored here.
- **Worktree:** `.worktrees/scattering-v2`, branch `wt/scattering-v2`.
- **Authoritative design:** ADR-021 (multi-bounce coherent diffuse scattering),
  ADR-022 (BDPT full fixed-topology AD), plan 10.
- **Derived from as-built patterns:**
  - `scattering/kernels/functional.py` — op-1/op-2 facades
    (`scattering_ensemble_eval`, `scattering_patch_integral_eval`) and their
    `_backward`/`_jvp` companions (ADR-014).
  - `native/.../kernels/scattering_ensemble.cu`,
    `scattering_patch_integral.cu` — bridge arg lists, launch structure,
    fixed-order reductions, `--fmad=false` TU policy.
  - `native/.../kernels/field_transport_reflection.cu` +
    `field_transport_ad_common.cuh` — fused chain transport, `ReflectionChain`
    on-stack state, `kMaxAdDepth = 8`, per-bounce `[N, D, ...]` geometry/material
    passing, dual `slab_fresnel` companions, output-cotangent folding.
  - `native/.../kernels/bdpt_connect_accumulation.cu` +
    `montecarlo/bdpt/kernels/paths.py` — `bdpt_accumulate_connection_samples`
    (ADR-019 `combine_domain`/`coeff_real`/`coeff_imag`), the 12-field
    `_BDPT_CONNECTION_SCHEMA`, the 19-field `_BDPT_SUBPATH_SCHEMA`.
  - `materials/kernels/functional.py` — `em_layer_stack_eval` CSR layer passing
    and its `_backward`/`_jvp` companions.
  - `montecarlo/bdpt/kernels/maps.py` — `bdpt_finalize_point_components`,
    `bdpt_finalize_component_maps`.

Everything below obeys the CLAUDE.md compute policy: hot-path physics is native,
Python facades only validate/dispatch/assemble, all defaults are bitwise no-ops,
and every AD companion is a registered native symbol (never a Torch
reconstruction).

## 0. Established conventions this spec inherits (do not re-litigate)

These are fixed by the existing code; the new symbols must match them exactly.

1. **Facade shape.** Each Python facade: `validate_cuda_tensor(...)` every
   tensor arg (dtype, ndim, trailing shape), cross-check shapes/devices, call
   `_required_native_op("<name>")(...)`, assert the returned dict key-set with
   an exact `set(out) == {...}` check, return the dict (or named tuple). Scalar
   Python args are cast with `float(...)`/`int(...)`/`bool(...)` at the boundary.
2. **AD-live scalars are 0-dim tensors (ADR-014).** A differentiable scalar
   (`frequency`, `k0`, `coef`, `tx_power`-as-scalar) crosses the autograd graph
   as a 0-dim `float32`/`float64` tensor and its numerical value crosses the
   native ABI as a `double` positional (the forward takes `frequency_value:
   double`; the `Function` carries the 0-dim tensor for the tape). Non-AD scalar
   config (thresholds, `n_quad`, mode ids, strategy ids) crosses as a plain
   `double`/`int64`. This mirrors `field_free_space` (`frequency` 0-dim tensor +
   `frequency_value` double) and `scattering_*_eval` (`coef`, `k0` doubles;
   `tangent_coef`, `tangent_k0` doubles in the JVP).
3. **Need-flag groups.** Backward kernels gate work by boolean `need_grad_*`
   flags that own a *group* of outputs, not one tensor each. Missing group ⇒
   the kernel skips that math and the facade returns `None` for those keys.
   Precedent groups: ensemble backward uses `need_grad_rows`,
   `need_grad_samples`, `need_grad_tables`, `need_grad_coef`; patch backward uses
   `need_grad_heights`, `need_grad_jones`, `need_grad_geometry`, `need_grad_k0`.
4. **atomicAdd vs deterministic.** Per-row/per-node outputs are direct stores
   (deterministic run-to-run). Shared/reduced buffers (CSR layer-parameter
   grads, frequency grad, a `total` phasor sum) use either a fixed-order tree
   reduction (preferred, patch-integral precedent) or `atomicAdd` (transmission
   backward precedent for CSR layer grads). The convention: **primal/JVP outputs
   are always deterministic (fixed-order tree reductions, no float atomics);
   backward CSR/shared grads may use `atomicAdd`** (the documented atomic
   nondeterminism policy, ADR-022 Consequences).
5. **`--fmad=false` lockstep.** Any new TU that shares a numerical duplicate
   with an existing forward (Op A vs op 1, chain transport vs
   `reflection_sequence`) compiles `--fmad=false` and preserves the exact
   Torch/forward expression association order; it joins the duplication ledger
   and the lockstep test list.
6. **Dict outputs.** Native ops return `pybind11::dict`; the facade validates
   the key-set. `None`-able backward keys are `pybind11::none()` when their flag
   is off.

## 1. Chain geometry encoding decision (normative)

**Decision: padded `[R, Dmax, ...]` per-leg arrays plus a per-row depth, with
`Dmax = kMaxAdDepth = 8` for each chain leg independently.**

Rationale, following the reflection-sequence pattern:

- `field_transport_reflection.cu` carries per-bounce geometry as
  `interaction_positions[N, D, 3]` / `interaction_normals[N, D, 3]` with a
  **single batch depth `D`** and on-stack fixed arrays `frames[kMaxAdDepth]`,
  `value_in[kMaxAdDepth]`, etc. (`ReflectionChain`). `check_reflection_primal`
  enforces `0 < D <= kMaxAdDepth`.
- A chain row has two legs of *independent* depth `d1, d2 >= 0`
  (`1 <= d1 + d2 <= scattering_chain_max_depth`). Rather than a single ragged
  batch depth, each leg gets its own padded block and its own per-row depth
  vector. One kernel thread walks C1 with a `ReflectionChain` of capacity
  `kMaxAdDepth`, records the vertex state, then walks C2 with a second
  `ReflectionChain` of capacity `kMaxAdDepth`.
- **Bound:** `d1 <= Dmax` and `d2 <= Dmax` with `Dmax = kMaxAdDepth = 8`. This
  keeps the on-stack fixed-size arrays valid and reuses the identical
  `ReflectionChain`/dual machinery. `scattering_chain_max_depth` (the config
  cap on `d1 + d2`) is therefore additionally clamped to `2 * kMaxAdDepth` at
  the facade, and each leg is validated `<= kMaxAdDepth`. A single diffuse
  vertex sits between the legs (never inside a padded block).
- **Padding semantics:** rows shorter than `Dmax` leave trailing slots
  unread; the per-row `d1_rows[r]`/`d2_rows[r]` bound every loop
  (`for bounce < d1_rows[r]`), exactly as `depth` bounds the reflection loop.
  Padded slots are never dereferenced for physics and contribute zero gradient.
- **`d1 = d2 = 0` degenerate row** (single-bounce collapse, ADR-021 D2): both
  leg depths are 0, the chain transfer `A_1 = A_2 = I` projection, and the row
  reduces symbol-for-symbol to op 1 / op 2. This case is lockstep-pinned but
  **not dispatched in production** (the single-bounce class keeps op 1 / op 2);
  it exists only as the collapse fixture.

Per-leg block layout (both ops):

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `c1_positions` | `[R, Dmax, 3]` | f32 | C1 per-bounce hit points (image-unfolded winners) |
| `c1_normals` | `[R, Dmax, 3]` | f32 | C1 per-bounce surface normals |
| `c1_eps_r`,`c1_sigma_e`,`c1_mu_r`,`c1_gain`,`c1_thickness` | `[R, Dmax]` | f32 | C1 per-bounce single-slab Fresnel inputs (frozen material ids, live params) |
| `c1_depth` | `[R]` | int32 | per-row `d1` (`0 <= d1 <= Dmax`) |
| `c2_positions`,`c2_normals`,`c2_*` (same set) | as above | f32 | C2 leg, indexed from the vertex outward toward RX |
| `c2_depth` | `[R]` | int32 | per-row `d2` |

Contiguity: every array C-contiguous (`is_contiguous()`), device = the shared
CUDA device of the row block. The bridge validates `d1_rows.max() <= Dmax` and
`d2_rows.max() <= Dmax` on host-free device state is not possible, so the
**facade** validates the config cap and the bridge validates the static
`Dmax == kMaxAdDepth` shape; per-row depth values are trusted from the
discovery contract (they are structural winners, frozen).

## 2. Python typed contract: chain discovery output (C1/C2 join)

The enumerated scatter-chain path class (ADR-021 D1) is discovered by running
the existing RayD image-method reflection enumeration twice against the chain
sample set as virtual endpoints — `tx -> {samples}` for C1 and `rx -> {samples}`
for C2 — then **joining on the sample index**. The facade for Op A / Op B
consumes a single typed contract produced by that join. It is a dataclass in
`propagation/enumerated/` (owner: enumerated engine; solver-agnostic), not a
native structure — it is structural packing, allowed Python boundary work.

```
@dataclass(frozen=True)
class ScatterChainDiscovery:
    # Row identity (R = joined chain rows, budgeted by keep-strongest-per-pair
    # + scattering_chain_max_rows per (tx, rx)).
    tx_id:        Tensor  # [R]     int32
    rx_id:        Tensor  # [R]     int32
    sample_index: Tensor  # [R]     int64   vertex v_s in the chain-sample set
    d1:           Tensor  # [R]     int32   C1 reflection depth (0..Dmax)
    d2:           Tensor  # [R]     int32   C2 reflection depth (0..Dmax)

    # C1 leg (TX -> v_s), padded to Dmax = kMaxAdDepth = 8.
    c1_positions: Tensor  # [R, Dmax, 3] f32   winner hit points
    c1_normals:   Tensor  # [R, Dmax, 3] f32
    c1_primitive: Tensor  # [R, Dmax]   int32  frozen face ids (material lookup)
    c1_material:  Tensor  # [R, Dmax]   int32  frozen material ids
    L1:           Tensor  # [R]     f32     image-unfolded C1 length
    sp1:          Tensor  # [R]     f32     C1 spreading (1/L1 planar; general per model)
    d_i:          Tensor  # [R, 3]  f32     incident dir of the last C1 leg at v_s

    # C2 leg (v_s -> RX), padded to Dmax.
    c2_positions: Tensor  # [R, Dmax, 3] f32
    c2_normals:   Tensor  # [R, Dmax, 3] f32
    c2_primitive: Tensor  # [R, Dmax]   int32
    c2_material:  Tensor  # [R, Dmax]   int32
    L2:           Tensor  # [R]     f32
    sp2:          Tensor  # [R]     f32
    d_o:          Tensor  # [R, 3]  f32     outgoing dir of the first C2 leg at v_s

    # Vertex data (indexed by sample_index into the chain-sample set).
    v_pos:        Tensor  # [R, 3]  f32     scatter vertex position
    v_normal:     Tensor  # [R, 3]  f32     vertex surface normal
    v_material:   Tensor  # [R]     int32   vertex material id (table/layer lookup)
    cos_i:        Tensor  # [R]     f32     |d_i . n| at the vertex
    cos_o:        Tensor  # [R]     f32     |d_o . n| at the vertex
    patch_row:    Tensor  # [R]     int64   vertex patch index (Op B only; -1 for ensemble)
```

Contract rules:

- **Join key** is `sample_index`. C1 and C2 are enumerated independently; the
  join keeps only rows where the same `v_s` is reachable from both TX (C1) and
  RX (C2), matching on `(tx_id, sample_index)` × `(rx_id, sample_index)`. The
  Cartesian join is budgeted by the existing keep-strongest-per-pair policy
  extended with `scattering_chain_max_rows` per `(tx, rx)` (documented default).
- **Row order** is deterministic (stable sort on `(tx_id, rx_id, sample_index,
  d1, d2)`) so downstream reductions are reproducible.
- `d1 = d2 = 0` rows are excluded from production discovery (single-bounce keeps
  its own path). `d1 + d2 >= 1` always holds for discovered rows.
- All tensors share one CUDA device; every array is C-contiguous. `Dmax` is the
  compile-time `kMaxAdDepth = 8`; the discovery builder pads shorter legs with
  zeros and marks their depth in `d1`/`d2`.
- The contract is **read-only** to the scattering facade; the facade slices
  the leg blocks and the vertex data straight into the native op args with no
  copy (aliasing preserved, ADR-007).

Two derived facades consume it: the ensemble facade (Op A) ignores `patch_row`
and reads the resident tables; the realization facade (Op B) reads `patch_row`
+ phase-screen heights and the layer stack at the local specular angle.

## 3. ADR-021 Op A — `scattering_chain_ensemble_eval` (+ `_backward`, `_jvp`)

Power-domain generalization of op 1 (`scattering_ensemble_eval`). One launch per
`(tx, rx-chunk)` over the `R` joined chain rows. Per row: C1 coherent Jones
transport of the tx polarization to `v_s` yielding the incident coherency
diagonal `(P_te, P_tm)`; ensemble Kirchhoff table lookup at the vertex; outgoing
coherency `J_out = diag(f_te P_te, f_tm P_tm)`; C2 power-domain Jones sandwich
`J_rx = A_2 J_out A_2^H`; receiver projection; radiometric assembly. Zero-phase
power rows, identical accumulation contract as op-1 ensemble rows.

Device primitives reused: `field_transport.cuh`
(`ReflectFrame`/`reflect_frame`, `slab_fresnel`, `complex3_dot_real`,
`reflect_complex3`) for the C1/C2 Jones legs; `scattering_table.cuh`
(`eval_te_tm`) for the vertex BSDF; the `ReflectionChain` on-stack layout and
`load_sequence3f` from `field_transport_ad_common.cuh`.

### 3.1 Forward `scattering_chain_ensemble_eval`

**SUPERVISOR RULING (2026-07-18, supersedes the table below):** the as-built
bridge signature in `scattering_chain_ensemble.cu` is authoritative — it follows
the committed float64 oracle and the op-1 conventions where this sketch
diverged. Binding order:
`tx_pol, rx_pol, source, vertex, target, c1_positions, c1_normals, c1_eps_r,
c1_sigma_e, c1_mu_r, c1_gain, c1_thickness, c1_depth, c2_(same 8), n_o, t1r,
t2r, backup_axis, wi_local, cos_i, cos_o, d_i, d_o, l1, l2, weights,
material_id, fte_flat, ftm_flat, table_offset, table_dims, material_slot,
coef: double, threshold: double, frequency_hz: double`.
Deltas vs the sketch, all RULED accepted: (a) radiometric form is
`gain = coef · (p^H J p) · cos_i · cos_o · weights / (L1² L2²)` — per-row
`weights` (A_patch) replaces `sp1/sp2` (op-1/oracle convention); (b)
`source/vertex/target` `[R,3]` added (structurally required by the leg
transport, `field_reflection_sequence` parity); (c) vertex frame args use the
op-1 names `n_o/t1r/t2r`; (d) `wi_local` is a FROZEN table-axis input
(op-1 convention; the oracle's d_i-differentiated table chain is a documented
gradient-convention divergence); (e) the reserved `f_sp/f_ps` cross-pol slots
are dropped from the v1 ABI — they land with the 4-channel table builder
change; (f) per-bounce rough `C_r` is NOT applied in-kernel: v1 chain bounces
are evaluated smooth (native reflection-sequence parity); rough chain-bounce
attenuation composes via the existing `field_rough_reflection_scale` op at the
orchestration seam (two calls: source→C1 and reversed C2), deferred together
with (g) reverse-mode chain geometry: `need_grad_geometry` is REJECTED LOUDLY
by the backward in this wave (JVP covers geometry forward-mode); the reverse
geometry adjoint is a documented staged follow-up, not a silent zero.
`weights` carries no gradient/tangent (frozen, spec-consistent).

Original sketch (historical, superseded where it conflicts with the ruling):

| Arg | Shape | Dtype | Contig | Notes |
|---|---|---|---|---|
| `tx_pol` | `[R, 3]` | f32 | yes | tx polarization per row |
| `rx_pol` | `[R, 3]` | f32 | yes | rx polarization per row (frozen) |
| `c1_positions` | `[R, Dmax, 3]` | f32 | yes | C1 leg (§1) |
| `c1_normals` | `[R, Dmax, 3]` | f32 | yes | |
| `c1_eps_r`,`c1_sigma_e`,`c1_mu_r`,`c1_gain`,`c1_thickness` | `[R, Dmax]` | f32 | yes | C1 per-bounce single-slab Fresnel inputs |
| `c1_depth` | `[R]` | int32 | yes | `d1` |
| `c2_positions`,`c2_normals`,`c2_eps_r`,`c2_sigma_e`,`c2_mu_r`,`c2_gain`,`c2_thickness`,`c2_depth` | as C1 | | | C2 leg |
| `d_i` | `[R, 3]` | f32 | yes | last-C1-leg incident dir at vertex |
| `d_o` | `[R, 3]` | f32 | yes | first-C2-leg outgoing dir at vertex |
| `v_normal` | `[R, 3]` | f32 | yes | vertex normal |
| `v_tangent1`,`v_tangent2` | `[R, 3]` | f32 | yes | vertex s/p local frame axes (t1r/t2r analog) |
| `backup_axis` | `[R, 3]` | f32 | yes | grazing-incidence backup (op-1 parity) |
| `cos_i`,`cos_o` | `[R]` | f32 | yes | vertex cosines |
| `L1`,`L2`,`sp1`,`sp2` | `[R]` | f32 | yes | unfolded lengths + spreading |
| `wi_local` | `[R, 3]` | f32 | yes | vertex-local incident dir (table axis) |
| `material_id` | `[R]` | int32 | yes | vertex material (table slot) |
| `f_te_flat`,`f_tm_flat` | `[T]` | f32 | yes | resident stacked co-pol tables |
| `f_sp_flat`,`f_ps_flat` | `[0]` or `[T]` | f32 | yes | **reserved v2 cross-pol slots**, empty in v1 |
| `table_offset` | `[M]` | int64 | yes | per-slot table base |
| `table_dims` | `[M, 4]` | int32 | yes | `(nti, npi, nto, npo)` |
| `material_slot` | `[K]` | int32 | yes | material id → table slot |

Scalar keyword args (non-AD ⇒ plain `double`; AD-live ⇒ 0-dim tensor at the
`Function`, `double` value at the ABI): `coef: double` (AD-live radiometric
coefficient / frequency chain — carried as 0-dim tensor by the autograd
Function, value crosses as `double` exactly as op-1 `coef`), `threshold: double`
(frozen keep gate), `frequency_hz: double` (AD-live; the per-bounce
`slab_fresnel` and the vertex table share it, matching
`reflection_sequence`). Constant `Dmax` is a compile-time constant, not an arg.

Output dict (keys asserted exactly): `{"gain", "amplitude", "length", "keep"}`,
all `[R]` (`gain`/`amplitude`/`length` f32, `keep` bool) — identical schema to
op 1. `length = L1 + L2` per row; `amplitude = sqrt(max(gain, 0))`;
`keep = gain > threshold`.

Launch: 1D grid, `launch_blocks(R)` blocks × `kBlockSize = 256`, one thread per
row, grid-stride loop. Elementwise, **no atomics** — deterministic run-to-run
(op-1 parity). Compiled `--fmad=false`.

### 3.2 `scattering_chain_ensemble_eval_backward`

Same positional args as the forward, plus the incoming output cotangents and the
need-flag groups. Output cotangents are `None`-able: `grad_gain`,
`grad_amplitude`, `grad_length` (`[R]` f32, `require_contiguous=False` at the
facade → `.contiguous()` at the ABI, `em_layer_stack_backward` precedent).

Need-flag groups (gate math + which keys are non-`None`):

| Flag | Live outputs |
|---|---|
| `need_grad_chain1` | `grad_c1_eps_r`,`grad_c1_sigma_e`,`grad_c1_gain`,`grad_c1_thickness` (`[R, Dmax]`) |
| `need_grad_chain2` | `grad_c2_eps_r`,`grad_c2_sigma_e`,`grad_c2_gain`,`grad_c2_thickness` (`[R, Dmax]`) |
| `need_grad_tables` | `grad_f_te`,`grad_f_tm` (`[T]`, 16-corner scatter) |
| `need_grad_geometry` | `grad_c1_positions`,`grad_c1_normals`,`grad_c2_positions`,`grad_c2_normals` (`[R, Dmax, 3]`), `grad_d_i`,`grad_d_o`,`grad_v_normal` (`[R, 3]`), `grad_L1`,`grad_L2`,`grad_sp1`,`grad_sp2`,`grad_cos_i`,`grad_cos_o` (`[R]`) |
| `need_grad_coef` | `grad_coef` (`[1]`) |
| `need_grad_frequency` | `grad_frequency` (`[1]`) |

Accumulation policy per output:
- Per-row / per-bounce grads (`grad_c1_*`, `grad_c2_*`, `grad_d_i`,`grad_d_o`,
  `grad_L*`, positions/normals) are **direct stores** — deterministic. Each row
  owns its `[Dmax]`/`[3]` slice (`reflection_sequence_backward` precedent).
- `grad_f_te`/`grad_f_tm` are the table-shaped **16-corner `atomicAdd`
  scatter** (op-1 backward `need_grad_tables` precedent — many rows hit shared
  table corners).
- `grad_coef` and `grad_frequency` are scalar reductions via `atomicAdd`
  (`reflection_sequence_backward` frequency-grad precedent).

Fixed (reject loudly via `_ad_reject_fixed_inputs`): topology, `sample_index`,
`c1_depth`/`c2_depth`, `material_id`, `material_slot`, `table_offset`,
`table_dims`, `rx_pol`, `threshold`, `backup_axis`, `v_tangent*`.

**Supervisor ruling (a_te2/a_tm2 removed):** unlike op 1, the incident
polarization state at the vertex is NOT a caller-supplied projection pair — it
is the C1 Jones transport of `tx_pol`, computed inside the kernel
(`P_te = |E_s|^2`, `P_tm = |E_p|^2` in the vertex s/p basis). Passing op-1's
`a_te2`/`a_tm2` alongside would be redundant and could disagree with the
transported field; the `d1 = 0` degenerate collapse recovers op-1's projections
exactly because `A_1 = I` makes the in-kernel projection equal op-1's caller
projection. The backward
recomputes the C1/C2 chain intermediates in primal expression order via the
shared `ReflectionChain` eval, then applies per-bounce dual `slab_fresnel`
(`ad::slab_fresnel_dual`, one dual eval per requested basis direction, exactly
as `reflection_sequence_backward`). New TU joins the `--fmad=false` lockstep
list.

Backward output dict key-set (facade asserts): the union of every `grad_*` key
listed above; keys whose owning flag is off are `pybind11::none()`.

### 3.3 `scattering_chain_ensemble_eval_jvp`

Same positional args plus `None`-able tangents (one per differentiable input):
`tangent_tx_pol`? no — tx_pol is frozen structural; the differentiable-input
tangents are `tangent_c1_eps_r`,`tangent_c1_sigma_e`,`tangent_c1_gain`,
`tangent_c1_thickness`,`tangent_c2_*` (`[R, Dmax]`),
`tangent_f_te_flat`,`tangent_f_tm_flat` (`[T]`),
`tangent_c1_positions`,`tangent_c1_normals`,`tangent_c2_positions`,
`tangent_c2_normals` (`[R, Dmax, 3]`), `tangent_d_i`,`tangent_d_o`,
`tangent_v_normal` (`[R, 3]`), `tangent_L1`,`tangent_L2`,`tangent_sp1`,
`tangent_sp2`,`tangent_cos_i`,`tangent_cos_o` (`[R]`), `tangent_coef: double`,
`tangent_frequency: double`. A missing tangent is a zero tangent.

Output dict (keys asserted): `{"tangent_gain", "tangent_amplitude",
"tangent_length"}`, all `[R]` f32 — mirrors op-1 JVP `_ENSEMBLE_TANGENT_FIELDS`
(`keep` is non-differentiable, no tangent). Deterministic forward-mode dual
sweep (fixed-order, no atomics), mirroring `reflection_sequence_jvp_kernel`.

## 4. ADR-021 Op B — `scattering_chain_realization_eval` (+ `_backward`, `_jvp`)

Coherent generalization of op 2 (`scattering_patch_integral_eval`). Per row over
the phase-screen patch set of the vertex surface:
`E_rx = A_2 · S_patch(d_i, d_o; h) · A_1 · e_tx`, with the carrier
`exp(-j k0 (L1 + L2))`, spreading `sp1·sp2` (`1/(L1 L2)` planar), `r_te/r_tm`
from `em_layer_stack_eval` at the local specular angle, the same Duffy-mapped
16×16 GL quadrature (`n_quad = 16`), and the same two-stage fixed-order tree
reduction as op 2. Output: per-row complex `path_field` + `path_gain =
|path_field|^2`, plus the 0-dim `total` (coherent sum over rows).

Device primitives reused: the op-2 patch machinery (`sample_height`,
`sp_basis`, `stable_tangent`, the Duffy quadrature nodes from `_duffy_nodes`);
`field_transport.cuh` for the `A_1`/`A_2` complex 2×2 Jones legs;
`em_layer_stack.cuh` (via the resident CSR stack) for `r_te/r_tm` at the local
specular angle. Op B is fully polarimetric (complex 2×2 sandwich, no diagonal
approximation).

### 4.1 Forward `scattering_chain_realization_eval`

Facade signature (extends the op-2 facade; new leg blocks + layer stack replace
the single-surface `r_te`/`r_tm`/`r1_rows`/`r2_rows`):

| Arg | Shape | Dtype | Contig | Notes |
|---|---|---|---|---|
| `patch_tris` | `[P, 3, 3]` | f32 | yes | vertex-surface patch mesh (frozen) |
| `patch_uvs` | `[P, 3, 2]` | f32 | yes | frozen |
| `rows` | `[R]` | int64 | yes | patch index per row (`patch_row`) |
| `d_i`,`d_o` | `[R, 3]` | f32 | yes | local incident/outgoing dirs at vertex |
| `n_rows` | `[R, 3]` | f32 | yes | vertex normal per row |
| `c1_positions`,`c1_normals`,`c1_eps_r`,`c1_sigma_e`,`c1_mu_r`,`c1_gain`,`c1_thickness`,`c1_depth` | §1 | | | C1 Jones leg `A_1` |
| `c2_*` (same set) | §1 | | | C2 Jones leg `A_2` |
| `tx_pol`,`rx_pol` | `[R, 3]` | f32 | yes | endpoint polarizations (rx frozen) |
| `L1`,`L2`,`sp1`,`sp2` | `[R]` | f32 | yes | unfolded lengths + spreading |
| `centroids` | `[R, 3]` | f32 | yes | patch centroid (carrier `q·c` term, op-2 parity) |
| `heights` | `[Hs, Ws]` | f32 | yes | phase-screen height field (AD-live) |
| `cos_spec` | `[R]` | f32 | yes | local specular cosine feeding the layer stack |
| `material_id` | `[R]` | int32 | yes | vertex material (layer CSR) |
| `layer_offset`,`layer_count` | `[M]` | int32 | yes | CSR (materials facade parity) |
| `layer_thickness_m`,`layer_eps_r`,`layer_sigma_e`,`layer_mu_r` | `[L]` | f32 | yes | CSR layer params (AD-live) |

Quadrature nodes `quad_a`,`quad_b`,`quad_w` (`[256]` f32) are cached and appended
by the facade (`_duffy_nodes`), never passed by the caller — op-2 parity.

Scalar keyword args: `k0: double` (AD-live; 0-dim tensor at the Function,
`double` at the ABI — op-2 parity), `frequency_hz: double` (AD-live; feeds the
layer stack). The layer stack `r_te/r_tm` are computed **inside** the fused
kernel from the CSR at `cos_spec` (no separate `em_layer_stack_eval` launch —
ADR-009 fusion boundary is the complete row), so the layer CSR arrays are direct
inputs and the material gradient composes through the in-kernel stack dual
(§4.2), exactly as the transmission-sequence kernel embeds `stack_rt`.

Output dict (keys asserted): `{"total", "path_field", "path_gain", "integral",
"row_value"}`. `total` is 0-dim complex64 (coherent row sum);
`path_field`/`row_value` are `[R]` complex64; `path_gain` is `[R]` f32
(`|path_field|^2`); `integral` is `[R]` complex64 (per-row Duffy integral, tests
buffer). This extends op-2's `{"total", "integral", "row_value"}` with the
explicit per-row `path_field`/`path_gain` the coherent combine (D3) consumes.

Launch: stage 1 = one block per row (`R` blocks) × `kQuadPoints = 256` threads,
per-node phasor into shared memory, fixed-order shared tree reduction, thread 0
assembles the row coefficient `(j·pref) · [A_2 S A_1] · carrier · sp1·sp2 ·
integral`; stage 2 = a single block tree-reduces `row_value[R]` into `total`.
**No float atomics** — bitwise stable run-to-run (op-2 parity).

### 4.2 `scattering_chain_realization_eval_backward`

Same positional args plus `grad_total` (0-dim complex64, required — the coherent
scalar cotangent, op-2 backward parity) and optionally `grad_path_field` (`[R]`
complex64) / `grad_path_gain` (`[R]` f32) when the row fields are on the graph
(the deterministic coherent combine, D3, backprops through `path_field`).

Need-flag groups:

| Flag | Live outputs |
|---|---|
| `need_grad_heights` | `grad_heights` (`[Hs, Ws]`) |
| `need_grad_layers` | `grad_layer_thickness`,`grad_layer_eps_r`,`grad_layer_sigma_e` (`[L]`) |
| `need_grad_chain1` | `grad_c1_eps_r`,`grad_c1_sigma_e`,`grad_c1_gain`,`grad_c1_thickness` (`[R, Dmax]`) |
| `need_grad_chain2` | `grad_c2_*` (`[R, Dmax]`) |
| `need_grad_geometry` | `grad_d_i`,`grad_d_o` (`[R, 3]`), `grad_c1_positions`/`grad_c1_normals`/`grad_c2_positions`/`grad_c2_normals` (`[R, Dmax, 3]`), `grad_L1`,`grad_L2`,`grad_sp1`,`grad_sp2` (`[R]`), `grad_centroids` (`[R, 3]`) |
| `need_grad_k0` | `grad_k0` (`[1]`) |
| `need_grad_frequency` | `grad_frequency` (`[1]`) |

Accumulation policy:
- `grad_heights` — `atomicAdd` scatter (op-2 backward `need_grad_heights`
  precedent: many quadrature nodes across rows hit shared texels).
- `grad_layer_*` — CSR layer-parameter grads via `atomicAdd` (the
  `em_layer_stack_backward` / transmission-sequence CSR policy; many rows share a
  material's layers). The in-kernel `stack_rt_dual`
  (`field_transport_ad.cuh`) produces the `r_te/r_tm` partials, then
  `adj_dot` folds them onto the CSR layer grads — the same lockstep stack dual
  the transmission-sequence backward uses.
- `grad_c1_*`,`grad_c2_*` per-bounce — direct stores (deterministic).
- `grad_d_i`,`grad_d_o`,`grad_L*`,`grad_centroids`,geometry — direct stores.
- `grad_k0`,`grad_frequency` — scalar `atomicAdd` reductions.

Fixed (reject loudly): `patch_tris`, `patch_uvs`, `rows`, quadrature
nodes/weights, `c1_depth`/`c2_depth`, `material_id`, `layer_offset`,
`layer_count`, `rx_pol`, `cos_spec` is derived-frozen (a function of the frozen
winner geometry; its gradient rides `need_grad_geometry` through `d_i`/`d_o`, not
a standalone input). Backward recomputes the forward per-row intermediates in
primal expression order; `--fmad=false` lockstep TU.

Output dict key-set: union of every `grad_*` above; off-flag keys are
`pybind11::none()`.

### 4.3 `scattering_chain_realization_eval_jvp`

Same positional args plus `None`-able tangents: `tangent_heights` (`[Hs, Ws]`),
`tangent_layer_thickness`/`tangent_layer_eps_r`/`tangent_layer_sigma_e` (`[L]`),
`tangent_c1_*`/`tangent_c2_*` (`[R, Dmax]`), `tangent_d_i`/`tangent_d_o`
(`[R, 3]`), `tangent_c1_positions`/`...normals`/`c2` (`[R, Dmax, 3]`),
`tangent_L1`/`L2`/`sp1`/`sp2` (`[R]`), `tangent_centroids` (`[R, 3]`),
`tangent_k0: double`, `tangent_frequency: double`. Missing tangent ⇒ zero.

Output dict (keys asserted): `{"tangent_total", "tangent_path_field",
"tangent_path_gain"}` — `tangent_total` 0-dim complex64, `tangent_path_field`
`[R]` complex64, `tangent_path_gain` `[R]` f32. Extends op-2 JVP's
`{"tangent_total"}` with the per-row tangents D3 needs. Deterministic fixed-order
dual sweep, no atomics.

## 5. ADR-021 deterministic coherent combine (D3) — no new symbol

D3 reuses the existing `bdpt_accumulate_connection_samples` `combine_domain`
precedent applied to the **deterministic** accumulator
(`deterministic_accumulate_flat`): a new **defaulted** `scattering_combine_domain`
argument (`0 = power`, default; `1 = coherent`) added to the existing op, no
sibling op, no schema change, no new ABI symbol for the primal. `combine == 0`
never enters the new branch and the kernels stay byte-identical (bitwise default).
The coherent branch sums the row `path_field` (from Op B) into the scattering
slot and finalizes `|sum|^2`, and joins the op's existing `_backward`/`_jvp`
companions with the coherent-domain derivative (§6.4 mirror:
`grad_c_r = 2 · grad_P · S`). This ADR-021 item is called out here so the Op B
`path_field` output schema (§4.1) is understood as its input; the interface
freeze for the deterministic accumulator's new argument is owned by the
deterministic solver domain and specified in that owner's change, not duplicated
here (open issue §8).

## 6. ADR-022 — BDPT companion set (12 symbols)

Six forward ops each gain `_backward` + `_jvp` companions. Owner:
`montecarlo.bdpt.kernels`. Plan-07 `torch.autograd.Function` pattern
(`setup_context`, `once_differentiable`, `set_materialize_grads(False)`, dual
unpacking, `_ad_reject_fixed_*`). Fixed-topology / fixed-winner contract: every
sampled quantity, pdf, MIS weight, visibility mask, and the 12-field
`_BDPT_CONNECTION_SCHEMA` / 19-field `_BDPT_SUBPATH_SCHEMA` layouts are frozen;
gradients ride the SAME rows (ADR-019 separate-argument precedent, never a
widened schema). Primal values under `ad != none` are bitwise the `none` values.

Torch complex-pair convention (ADR-014): a complex output `F` with real
cotangent chain uses `d/dF = 2·conj(F)·rest` for `|F|^2` and pairwise real
adjoints elsewhere, exactly `fold_output_cotangents`.

### 6.1 `bdpt_reflected_light_subpath_state_{backward,jvp}`

Forward (`bdpt_reflected_light_subpath_state`, paths.py): advances a light
subpath dict through one specular reflection event; multiplies the carried
Complex3 Jones field by `O = ReflectFrame rotation × Fresnel diag` and scales
power terms. Differentiable inputs: `material_eps_r`, `material_sigma_e`,
`material_mu_r`, `material_thickness` (`[Nmat]` f32, per-face single-slab),
`frequency_hz` (AD-live 0-dim tensor / `double` value). Frozen: the `intersection`
dict (`t`,`p`,`n`,`global_prim_id` — hit-point geometry stays frozen in v1,
ADR-022 stochastic-sampler stance), `material_valid`, `material_gain`, the input
`light` dict's structural fields, event partition.

**Backward** args: the forward positional args + the output subpath cotangents.
The differentiable output fields of a subpath dict are `field_real`,`field_imag`
(`[N, 3]` f32) and `throughput_real`,`throughput_imag` (`[N]` f32); the incoming
cotangents are `grad_field_real`,`grad_field_imag`,`grad_throughput_real`,
`grad_throughput_imag` (`None`-able, `[N,3]`/`[N]`). Structural fields
(`origin`,`direction`,`depth`,`pdf_*`,`valid`,`event_type`,…) carry no gradient.
Derivative: `grad_field_in = O^H · grad_field_out`; material partials via
`field_transport_ad.cuh::stack_rt_dual` (the transmission-sequence lockstep
dual); per-hit CSR/material-parameter grads accumulate by **`atomicAdd`** (many
subpath rows share a face material). Need-flags: `need_grad_material`
(→ `grad_eps_r`,`grad_sigma_e`,`grad_gain`,`grad_thickness` `[Nmat]`),
`need_grad_field_in` (→ `grad_light_field_real`,`grad_light_field_imag` `[N,3]`,
`grad_light_throughput_real`,`grad_throughput_imag` `[N]` — the upstream subpath
cotangents, direct stores), `need_grad_frequency` (→ `grad_frequency` `[1]`,
`atomicAdd`). `_ad_reject_fixed_*`: intersection geometry, `material_valid`.
Output dict: union of the above `grad_*` keys, off-flag ⇒ `None`.

**JVP** args: forward + `None`-able tangents `tangent_eps_r`,`tangent_sigma_e`,
`tangent_mu_r`,`tangent_thickness` (`[Nmat]`), `tangent_frequency` (`double`),
`tangent_light_field_real`,`tangent_light_field_imag` (`[N,3]`),
`tangent_light_throughput_real`,`tangent_throughput_imag` (`[N]`). Output: the
tangent of the advanced subpath's differentiable fields —
`{"tangent_field_real","tangent_field_imag","tangent_throughput_real",
"tangent_throughput_imag"}` (`[N,3]`/`[N]` f32). Elementwise deterministic dual
(tangent-forward of `O`), no atomics.

### 6.2 `bdpt_transmitted_light_subpath_state_{backward,jvp}`

Forward (`bdpt_transmitted_light_subpath_state`): advances through one slab
transmission; the operator is the WallFrame slab operator built from the **CSR
layer stack** (`layer_offset`,`layer_count`,`layer_thickness_m`,`layer_eps_r`,
`layer_sigma_e`,`layer_mu_r`, `face_material_id`). Differentiable: layer
`thickness`/`eps_r`/`sigma_e` (CSR `[L]`), `frequency_hz`. Frozen: intersection
geometry, `face_material_id`, `layer_offset`/`layer_count` (structural CSR
index), `material_valid`.

**Backward**: forward args + the same subpath-field cotangents as §6.1. Layer
grads via `stack_rt_dual` folded onto the CSR by **`atomicAdd`** (identical to
`em_layer_stack_backward` / `field_transmission_sequence_backward`). Need-flags:
`need_grad_layers` (→ `grad_layer_thickness`,`grad_layer_eps_r`,
`grad_layer_sigma_e` `[L]`), `need_grad_field_in` (→ upstream subpath field
cotangents, direct stores), `need_grad_frequency` (→ `grad_frequency` `[1]`).
Reject: intersection geometry, `face_material_id`, CSR index arrays.

**JVP**: forward + `None`-able `tangent_layer_thickness`,`tangent_layer_eps_r`,
`tangent_layer_sigma_e` (`[L]`), `tangent_frequency` (`double`), and the upstream
subpath-field tangents. Output: `{"tangent_field_real","tangent_field_imag",
"tangent_throughput_real","tangent_throughput_imag"}`. Deterministic dual.

### 6.3 `bdpt_endpoint_connection_samples_{backward,jvp}`

Forward (`bdpt_endpoint_connection_samples`): LoS/NEE endpoint contribution
`contribution = P_src · |F|^2 · (lambda/(4·pi·L))^2 / N`, producing the 12-field
connection-sample dict. Differentiable: the endpoint field `F` (rides the
subpath `field_*` inputs), `frequency` (→ `lambda`), `P_src` (`tx_power`).
Frozen: `L` (visibility/geometry), `N` (`samples_per_tx`), visibility mask, the
connection topology, `component_id`, MIS mode.

**Backward** args: the forward `light`/`sensor` subpath dicts + `frequency_hz`
+ the connection scalar-args, plus the incoming `grad_contribution` (`[Ns]` f32,
the only differentiable output of the connection schema; `pdf`,`mis_weight`,
`topology`,`component_id`,`valid`,`path_length_m` are frozen structure).
Derivative: `d/dF = 2·conj(F)·rest` (pair convention) folded back onto the
light/sensor subpath `field_*` cotangents; `d/d lambda` chains into
`grad_frequency`; `d/d P_src` direct onto `grad_tx_power`. Need-flags:
`need_grad_field` (→ `grad_light_field_real/imag`, `grad_sensor_field_real/imag`
`[N,3]`, direct stores), `need_grad_frequency` (→ `grad_frequency` `[1]`,
`atomicAdd`), `need_grad_tx_power` (→ `grad_tx_power` `[Ntx]`, `atomicAdd` —
many rows share a tx). Reject: `L`, `N`, visibility, `mis`, `component_id`.

**JVP**: forward + `None`-able `tangent_light_field_*`,`tangent_sensor_field_*`
(`[N,3]`), `tangent_frequency` (`double`), `tangent_tx_power` (`[Ntx]`). Output:
`{"tangent_contribution"}` (`[Ns]` f32). Deterministic.

### 6.4 `bdpt_accumulate_connection_samples_{backward,jvp}` (power AND coherent)

Forward is `bdpt_accumulate_connection_samples` (§ bridge above). It bins
`contribution·mis_weight` into `[tx,rx]` component matrices — power domain
(`combine_domain=0`) — or sums the complex `coeff_real/coeff_imag` into phasor
bins and finalizes `|sum|^2` — coherent domain (`combine_domain=1`). Both
companions cover **both domains** (a `combine_domain` positional selects the
branch, matching the forward).

**Power domain.** `M[b] = SUM_r contribution_r · mis_r` is linear in
`contribution`. Backward: `grad_contribution_r = mis_r · grad_M[bin(r)]` — a
**gather** (each row reads its bin's cotangent), **no atomics**, deterministic.
`mis_r` frozen. This also implements the **concat backward as a split view**:
the connection-sample block boundaries index into the concatenated
`contribution`, so the split is a slice of `grad_contribution`, handled in this
op's backward (never a Torch physics reconstruction — ADR-022).

**Coherent domain.** `P[b] = |S_b|^2`, `S_b = SUM_r c_r` (`c_r =
coeff_real + j·coeff_imag`). Backward: `grad_c_r = 2 · grad_P[b] · S_b` (real
cotangent × complex bin sum, pair convention). Requires `S_b` from the forward —
already materialized in the accumulator's per-component
`{los,reflection,…}_re/_im` double buffers, so **no new tape**; the backward
reads the forward-retained bin-sum buffers, passed as explicit tensor args
(supervisor ruling: the coherent forward returns them as non-differentiable
outputs; no in-backward re-reduction — that would be a second numerical
duplicate of the atomic-double reduction).
`path_gain = SUM_c P_c` chains linearly. Result `grad_coeff_real/imag` are
`[Ns]` gathers (deterministic).

**Backward** args: the `samples` dict + `tx_count`,`rx_count`,
`accumulation_strategy`,`combine_domain`,`coeff_real`,`coeff_imag` (forward
signature) + the six output-matrix cotangents `grad_path_gain`,`grad_los`,
`grad_reflection`,`grad_diffraction`,`grad_transmission`,`grad_scattering`
(`[tx,rx]` f32, `None`-able). Need-flags: `need_grad_contribution` (power →
`grad_contribution` `[Ns]`), `need_grad_coeff` (coherent →
`grad_coeff_real`,`grad_coeff_imag` `[Ns]`). Both are gathers, deterministic.
Reject: `mis_weight`, `tx_id`,`rx_id`,`component_id`,`valid` (frozen structure).
Output dict: `{"grad_contribution","grad_coeff_real","grad_coeff_imag"}` with
`None` for the domain not selected.

**JVP** args: forward + `None`-able `tangent_contribution` (`[Ns]`, power) /
`tangent_coeff_real`,`tangent_coeff_imag` (`[Ns]`, coherent). Output:
`{"tangent_path_gain","tangent_los","tangent_reflection","tangent_diffraction",
"tangent_transmission","tangent_scattering"}` (`[tx,rx]` f32). Power JVP:
`t_M[b] = SUM_r mis_r · tangent_contribution_r` (fixed-order per-bin sum).
Coherent JVP: `t_P = 2·Re(conj(S_b)·t_S_b)`, `t_S_b = SUM_r t_c_r` — deterministic
fixed-order sum (ADR-022 §Accumulate coherent). No float atomics on the JVP
(the primal/JVP-must-be-deterministic rule); the coherent-domain forward uses
atomic-double bins, but the JVP recomputes the tangent bin sums in fixed order.

### 6.5 `bdpt_finalize_point_components_{backward,jvp}`

Forward (`bdpt_finalize_point_components`, maps.py): inputs
`los`,`reflection`,`diffraction`,`transmission`,`scattering` (`[tx,rx]` f32),
outputs `path_gain` (`[tx,rx]`, elementwise sum of the five) + per-component 0-dim
`*_power` scalar sums over the map. Linear map.

**Backward** args: the five component matrices + the output cotangents
`grad_path_gain` (`[tx,rx]`, `None`-able) and `grad_los_power`,
`grad_reflection_power`,`grad_diffraction_power`,`grad_transmission_power`,
`grad_scattering_power` (0-dim, `None`-able). Derivative (transpose of a linear
map): `grad_component[i] = grad_path_gain[i] + grad_<component>_power` (the
0-dim power cotangent broadcasts to every cell). Elementwise, **deterministic,
no atomics**. Single need-flag `need_grad_components` (all five share the linear
transpose). Output: `{"grad_los","grad_reflection","grad_diffraction",
"grad_transmission","grad_scattering"}` (`[tx,rx]`).

**JVP**: forward + `None`-able `tangent_los`…`tangent_scattering` (`[tx,rx]`).
Output: `{"tangent_path_gain","tangent_los_power","tangent_reflection_power",
"tangent_diffraction_power","tangent_transmission_power",
"tangent_scattering_power"}` — `tangent_path_gain` `[tx,rx]`, the `*_power`
tangents 0-dim (fixed-order sum). Deterministic.

### 6.6 `bdpt_finalize_component_maps_{backward,jvp}`

Identical algebra to §6.5 but the component tensors are `[tx, H, W]` (ndim 3)
radiomaps; `path_gain` is `[tx,H,W]`, the `*_power` outputs are 0-dim sums over
the whole map. Backward/JVP are the same linear transpose / forward map with the
3-D shapes. Deterministic, no atomics. Same need-flag and key structure as §6.5
with the 3-D shapes.

### 6.7 Structural ops need NO companions (ADR-022)

`bdpt_concat_connection_samples`, `bdpt_compact_connection_samples`,
`bdpt_filter_connection_samples`, `bdpt_count_valid_connection_samples`,
`bdpt_connection_variance`, `bdpt_mis_weights`, `bdpt_launch_state`,
`bdpt_endpoint_subpath_state`, `bdpt_subpath_intersection_inputs`,
`bdpt_*_visibility_inputs`, and the sampling ops are index/copy/diagnostic
operations on frozen structure. Cotangents route through the stored index maps
in the backward of the ops that consumed them (concat backward is the split view
inside §6.4's accumulate backward). No new symbol; asserting this here prevents
a reviewer adding one.

## 7. Python-side AD wiring (interface obligations, not new symbols)

- Each forward facade above gains a sibling `torch.autograd.Function` in the
  owning `kernels/` package (`scattering/kernels/autograd_chain.py`,
  `montecarlo/bdpt/kernels/autograd_*.py`) following the plan-07 template:
  `forward` dispatches the registered forward symbol (float64 variant when the
  input batch is f64, for `gradcheck`), `setup_context` unpacks duals + saves
  primals + `mark_non_differentiable` the structural outputs, `backward`
  (`@once_differentiable`) calls the `_backward` symbol with the need-flags
  derived from `ctx.needs_input_grad` and `_ad_reject_fixed_inputs` for every
  frozen slot, `jvp` calls the `_jvp` symbol.
- AD-live scalars (`coef`, `k0`, `frequency`, `tx_power`) enter each `Function`
  as a 0-dim tensor plus a plain-Python `*_value` (ADR-014 / `field_free_space`
  precedent); the tape carries the tensor, the ABI takes the `double`.
- `bdpt/config.py`: `NO_AD_MODES` lifts to `{"none","jvp","vjp"}` with
  per-feature readiness gates failing loudly for any combination whose
  companions are unregistered (never silent detach). The ADR-019 coherent+AD
  refusal is replaced by support. The enumerated oracle call threads `ad_mode`
  through the public `evaluate_enumerated_paths` config (read-only boundary
  preserved, no internal imports). The shooting sampler uses live
  `scene.frequency`/`tx_power` tensors under `ad != none` and registers launches
  in the existing `AdLaunchLedger`; sampling/masks/seeds stay bitwise identical.
- Metadata: `ad_status` reports the active mode; `ad_geometry:
  "enumerated_blocks_only"` for BDPT; the differentiable-parameter inventory is
  reported. `montecarlo/basic` metadata continues to report
  `scattering max_depth = 1` (D4 scope note).

## 8. Governance and symbol accounting

Native binding count today: **193** (`EXPECTED_NATIVE_BINDING_COUNT = 193`,
`ci/check_contract_coverage.py`).

New ABI symbols this specification freezes:

| Group | Symbols | Count |
|---|---|---|
| ADR-021 Op A | `scattering_chain_ensemble_eval`, `_backward`, `_jvp` | 3 |
| ADR-021 Op B | `scattering_chain_realization_eval`, `_backward`, `_jvp` | 3 |
| ADR-022 BDPT | 6 forwards × (`_backward`+`_jvp`) | 12 |

D3's deterministic coherent combine adds **no** new primal symbol (a defaulted
argument on the existing `deterministic_accumulate_flat`, mirroring ADR-019's
`combine_domain`), and rides that op's existing `_backward`/`_jvp` companions.

Each new symbol requires, in the same change (CLAUDE.md native-binding rule):
`ci/native-binding-manifest.json` entry, contract-coverage manifest entry, owner
inventory entry, a direct contract test, a negative no-fallback test, and ≥1
end-to-end caller; `EXPECTED_NATIVE_BINDING_COUNT` bumped with the rebuild;
duplication-ledger entries for every primal/JVP/VJP lockstep pair; launch-ledger
entries; `ci/public-api-snapshot.json` for the new config fields
(`scattering_chain_max_depth`, `scattering_chain_samples_per_m2`,
`scattering_chain_max_rows`, `scattering_coherent`, `max_scattering_order`, and
the BDPT `ad_mode` set); `ci/check_import_graph.py` clean.

**Supervisor rulings (2026-07-18, Fable):**

1. ADR-021's "+4 symbols" undercounted (companions only). RULED: ADR-021 adds
   **6** symbols (2 forwards + 4 companions), `193 -> 199`; ADR-022 adds 12 on
   top, `199 -> 211`. ADR-021 lands first; both ADR governance lines are
   corrected to these numbers.
2. Op A drops `a_te2`/`a_tm2` (see §3.2 ruling) — the incident coherency is
   computed in-kernel from the C1 transport of `tx_pol`.
3. §6.4 coherent backward reads the forward-retained bin-sum buffers passed as
   explicit args (see §6.4 ruling); no in-backward re-reduction.
4. D3's `scattering_combine_domain` defaulted-arg signature was frozen and
   implemented in the deterministic owner's change
   (`deterministic_accum.cu` + `deterministic/kernels/accumulation.py` +
   `binding/path.cpp`, wave 1) consistent with §5: trailing
   `int64_t scattering_combine_domain = 0` on all four symbols.
