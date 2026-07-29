# ADR-021: Multi-bounce coherent polarimetric diffuse scattering

- **Status:** Accepted (2026-07-18; implemented on `wt/scattering-v2`, final
  verification 1037 tests / 0 failures on the 211-binding build, user-accepted)
- **Date:** 2026-07-18
- **Kind:** New numerical capability (new native ops + AD companions + path
  class). Every existing primal path stays bitwise unchanged: all new behavior
  is behind new config that defaults OFF.
- **Related:** ADR-010 (native scattering kernels, single-bounce), ADR-014/015
  (scattering AD), ADR-019 (BDPT coherent combine, per-component phasor
  precedent), ADR-009 (fusion ownership), plan 07 (fixed-topology AD), plan 10
  (this work's plan document).

## Context

Diffuse scattering is single-bounce and terminal in every solver:

- Deterministic/Path append standalone `component_id=6` rows
  `TX -> patch -> RX` outside the enumerated engine
  (`propagation/enumerated/scattering.py:925-937`); ensemble rows are
  zero-phase POWER rows, and the only coherent evaluation (the phase-screen
  patch integral) is single-surface and REPLACES the specular+ensemble for
  that surface (contract 6.7.3).
- The native accumulator defaults scattering to an incoherent power slot
  (`deterministic.cu:1013-1031`); the accepted opt-in coherent branch is beside
  it at `deterministic.cu:1014-1025`.
- The BDPT shooting sampler allows specular PREFIXES before a scatter but
  kills the subpath at the scatter event
  (`montecarlo/bdpt/connections.py:454-456`, "v1 depth rule",
  `montecarlo/events/scattering.py:30-32`), and the scattered Jones carrier is
  zeroed (POWER only).
- No kernel chains a diffuse scatter with further coherent bounces; the fused
  `reflection_sequence_kernel` transports only specular Jones operators.

Reference target: Sionna RT's effective-roughness diffuse model is a scalar
lobe family with heuristic XPD polarization, per-sample random phase, and
power accumulation. This repo already exceeds it on the BSDF itself
(layer-stack-derived polarimetric Kirchhoff tables, Sinkhorn energy balance,
`(1 - C_r^2)|R|^2` diffuse budget). This ADR closes the remaining gaps: chains,
coherence, and end-to-end polarization.

## Decision

Four capabilities, all default-OFF, all native-pure, all shipping WITH their
JVP/VJP companions at birth (no ADR-010-style forward-only deferral).

### D1. Enumerated scatter-chain path class (Deterministic + Path)

New path class with exactly one diffuse vertex and specular reflection chains
on either side:

```
TX --C1 (reflections, depth d1 >= 0)--> v_s --C2 (reflections, depth d2 >= 0)--> RX
1 <= d1 + d2 <= scattering_chain_max_depth      (new config, default 0 = OFF)
```

- `d1 = d2 = 0` remains the EXISTING single-bounce code path, untouched and
  bitwise frozen.
- `v_s` is drawn from a dedicated chain-sample set (new config
  `scattering_chain_samples_per_m2`, default lower density than the
  single-bounce sampler; same R2 low-discrepancy scheme, same
  `_MAX_SAMPLES_PER_FACE` cap).
- Chain discovery reuses the existing RayD image-method reflection
  enumeration with the samples as virtual endpoints: C1 chains are enumerated
  as `tx -> {samples}` reflection paths, C2 chains as `rx -> {samples}`
  (reciprocal), through the existing `propagation.geometry` bridge. No
  geometry is recomputed in Python/Torch; no new RayD symbols are expected.
  Chains are joined on the sample index; the joined row set is budgeted by
  the existing keep-strongest-per-pair policy extended with a per-(tx,rx)
  row cap (`scattering_chain_max_rows`, documented default).
- Topology encoding: the interaction sequence of a chain row uses the
  existing per-slot `interaction_type` machinery — REFLECTION=1 slots for C1,
  one SCATTERING=8 slot at the vertex, REFLECTION=1 slots for C2;
  `component_id` stays 6; `depth = d1 + 1 + d2`. Slot value 8 inside a chain
  is already legal in the contract and merely unused today.
- Transmission events inside C1/C2, diffraction+scatter composition, and
  >= 2 diffuse vertices per enumerated path are explicitly OUT of v1 (the
  latter stays Monte Carlo territory; a deterministic double-vertex pass is a
  documented v2 opt-in with an O(S^2) budget gate).

### D2. Native chain evaluation ops (owner: `scattering/kernels/`)

Two new fused native op families evaluate complete scatter-chain rows in one
launch per (tx, rx-chunk), reusing the existing device primitives
(`field_transport.cuh` ReflectFrame/reflect_complex3/slab_fresnel,
`scattering_table.cuh` eval_te_tm/eval_te_tm_grad). Per ADR-009 the fusion
boundary is the complete row: chain-1 transport, scatter operator, chain-2
transport, receiver projection — no materialized intermediates, no extra
launches.

**Op A `scattering_chain_ensemble_eval`** (power domain, generalizes ADR-010
op 1). Per row with chain geometries (positions, normals, material CSR per
bounce, unfolded lengths `L1`, `L2`), sample data, and resident tables:

1. Chain-1 coherent Jones transport of the tx polarization to `v_s` in the
   incident s/p basis (identical per-bounce math and expression order as
   the `field_transport_reflection.cu` provenance section in
   `field_transport.cu`, including per-bounce rough `C_r`
   attenuation), yielding the incident coherency diagonal
   `P_te = |E_s|^2, P_tm = |E_p|^2` and the incident direction `d_i` of the
   last C1 leg.
2. Ensemble Kirchhoff BSDF at the vertex: quadrilinear table lookup
   `(f_te, f_tm) = T(wi_local, wo_local)` exactly as op 1, with
   `wo_local` from the first C2 leg direction `d_o`.
3. Outgoing coherency `J_out = diag(f_te * P_te, f_tm * P_tm)` (per
   steradian; co-pol diagonal contract, section "Polarization" below).
4. Chain-2 Jones sandwich in the power domain: `J_rx = A_2 J_out A_2^dagger`
   with `A_2` the chain-2 complex Jones transfer (2x2), then the receiver
   projection `p_rx^dagger J_rx p_rx`.
5. Radiometric assembly in the module-docstring units:
   `gain = P_t * (p^dagger J p) * cos_i * cos_o * A_patch * lambda^2
   / ((4*pi)^2 * L1^2 * L2^2)`, `length = L1 + L2`,
   `amplitude = sqrt(max(gain, 0))`, `keep` threshold. Zero-phase power rows,
   identical accumulation contract as existing ensemble rows.

The `d1 = d2 = 0` degenerate case of Op A reduces symbol-for-symbol to the op-1
expression set; a lockstep test pins this (it is NOT dispatched in production
for the single-bounce class, which keeps op 1).

**Op B `scattering_chain_realization_eval`** (coherent, generalizes ADR-010
op 2). Per row over the phase-screen patch set of the vertex surface:

```
E_rx = A_2 . S_patch(d_i, d_o; h) . A_1 . e_tx
S_patch = (j * pref(q)) * [s_o p_o] diag(r_te, r_tm) [s_i p_i]^T
          * exp(-j k0 (L1 + L2)) / (sp_1 * sp_2)^-1
          * SUM_t w_t exp(-j (q . x_t + q_n h_t))          (Duffy GL quadrature)
```

with `q = k0 (d_o - d_i)` built from the LOCAL directions at the vertex,
`r_te/r_tm` from `em_layer_stack_eval` at the local specular angle,
`sp_1 * sp_2 = 1/(L1 * L2)` for planar specular chains (image theory), and the
same fixed-order two-stage tree reduction, `--fmad=false` TU policy, and
`n_quad = 16` Duffy nodes as op 2. Output: per-row complex `path_field` plus
`path_gain = |path_field|^2`. The `d1 = d2 = 0` case collapses to the op-2
expression set (lockstep-pinned, not production-dispatched).

Both ops ship `_backward`/`_jvp` companions in the same change (D5).

### D3. Deterministic coherent scattering combine (opt-in)

New deterministic/path config `scattering_coherent: bool = False`.

- OFF (default): bit-identical to today — scattering slot 4 stays the
  incoherent `SUM |field|^2` power term.
- ON (requires realization mode / Op B rows, refuses ensemble-only solves
  loudly): the accumulator sums the complex `path_field` of scattering rows
  per (tx, rx) into the scattering slot and finalizes `|sum|^2`, exactly the
  ADR-019 per-component phasor precedent. Components still combine
  incoherently into `path_gain`; cross-component interference stays out of
  scope (same revisit condition as ADR-019).
- Implementation follows the ADR-019 mechanism: a new defaulted
  `scattering_combine_domain` argument on the EXISTING
  `deterministic_accumulate_flat` op (no sibling op, no schema change, no new
  ABI symbol for the primal); `combine == 0` never enters the new branch and
  the kernels stay byte-identical. The coherent branch gets its own
  fixed-order complex reduction and joins the op's existing
  `_backward`/`_jvp` companions.
- Contract 6.7.3 (realization REPLACES specular delta + ensemble lobe for
  that surface) continues to hold per surface; the coherent combine only
  changes how scattering rows combine with EACH OTHER. No double counting:
  a surface contributes through exactly one scattering model, and specular
  `C_r` attenuation plus the `(1 - C_r^2)|R|^2` table budget keep the
  specular/diffuse energy split conserved per bounce, including chain
  bounces (chain transport applies `C_r` per specular bounce identically to
  the reflection component).

### D4. Monte Carlo multi-bounce diffuse (BDPT; MC-basic scoped out)

BDPT lifts the v1 terminal rule behind new config
`max_scattering_order: int = 1` (default 1 = today's behavior, bitwise: the
default keeps the existing seed consumption, event partition, and NEE rows
untouched).

For `max_scattering_order > 1`:

- At a scatter-selected hit the subpath (a) emits NEE connection rows exactly
  as today, and (b) CONTINUES with a directional sample drawn from the
  resident reciprocal table CDF (`scattering_table_sample`, existing native
  symbol), consuming one new seeded uniform pair from a NEW salted seed
  stream (existing smooth-face and three-way streams untouched).
- The continued subpath multiplies `pdf_forward *= pdf(wo|wi)` and
  `pdf_reverse *= pdf(wi|wo)` (already recorded today for diagnostics) and
  divides the carried POWER by `p_scatter * pdf(wo)` per the standard
  unbiased estimator; reflection/transmission events may follow, and further
  scatter events are allowed up to `max_scattering_order`.
- The Jones carrier through a scatter event stays power-only in this ADR
  (`P_te/P_tm` incident powers weight the table channels). Post-scatter
  carrier contract: the Complex3 field is cleared at the scatter vertex and
  the scalar `throughput` is RE-SEEDED from the field-based incident power
  `sqrt(P_te + P_tm)` (excluding `source_power`, which the connection
  convention multiplies separately) times the unbiased continuation
  amplitude `sqrt(f_weighted cos_o / (pdf(wo) p_scatter))`. From that vertex
  on `|throughput|^2` is the authoritative unpolarized power weight — the
  per-bounce specular `sqrt(gain * R_eff)` scaling at the actual incidence
  angle is exact unpolarized transport — and a later scatter vertex reads
  its incident power from it, split evenly across the local TE/TM channels.
  The pre-scatter throughput remains the contract-section-5 sampling proxy
  and never enters a contribution. Mixed-transmission endpoint rows are not
  emitted for post-scatter subpaths (`S -> ... -> T` chains are a documented
  v1 coverage gap, not a biased zero).
  Scattering remains EXCLUDED from the ADR-019 coherent combine; a coherent
  MC diffuse carrier (phase-screen height at the sampled point + unfolded
  phase) is the documented follow-up that ADR-019's revisit condition
  anticipates, gated on D2/D3 acceptance first.
- MIS: NEE remains the only strategy reaching a sensor through a scatter
  vertex (directional continuation still has zero probability of hitting a
  point/cell sensor), so connection rows keep weight 1; the recorded pdfs
  make the future sensor-strategy MIS possible without schema change.

MC-basic keeps its single-scatter analytic deposit: it is the Sionna-parity
baseline solver, its estimator has no walking loop, and adding one is a
solver-scale change with no user requirement behind it. Documented, not
silent: `montecarlo/basic` metadata continues to report
`scattering max_depth = 1`.

### Polarization contract (v1 diagonal, v2 slots reserved)

The resident tables carry co-pol `f_te/f_tm` only, so the ensemble BSDF acts
diagonally in the LOCAL s/p basis of each leg; all basis rotations along the
chains are exact complex Jones (no scalar XPD anywhere). Cross-pol arises
from frame rotation between legs — exactly the physics Sionna's XPD scalar
approximates. The table builder's vector-Kirchhoff kernel can later emit the
full 4-channel `(f_ss, f_sp, f_ps, f_pp)`; the op signatures reserve the two
cross channels as optional table inputs (empty tensors in v1) so the runtime
contract does not change when the builder grows them. The realization path
(Op B) is already fully polarimetric (complex 2x2 sandwich, no diagonal
approximation).

### D5. AD companions (born differentiable)

New symbols (owner `scattering/kernels/`, plan-07 Function pattern,
ADR-014 conventions):

| Symbol | Kind |
|---|---|
| `scattering_chain_ensemble_eval_backward` / `_jvp` | VJP/JVP of Op A |
| `scattering_chain_realization_eval_backward` / `_jvp` | VJP/JVP of Op B |

- Live inputs, Op A: table values (16-corner scatter), per-bounce chain
  Fresnel inputs (eps_r/sigma_e/thickness CSR via the same lockstep stack
  dual as `field_transmission_sequence`), per-row chain geometry
  (positions/normals/lengths, fixed-winner), `C_r` inputs (sigma per bounce),
  incident/outgoing directions, radiometric `coef` (0-dim tensor, frequency
  chain), tx polarization projections. Fixed (reject loudly): topology,
  sample indices, visibility, material ids, `keep` threshold, rx pol.
- Live inputs, Op B: phase-screen `heights` (4-texel scatter per node),
  `r_te/r_tm` (composing with `em_layer_stack_ad` for material gradients),
  chain geometry rows, `centroids`, `k0` (0-dim tensor). Fixed: patch mesh
  (`patch_tris/uvs`), quadrature nodes/weights, rows.
- Backward kernels recompute forward intermediates in primal expression
  order; the new TUs join the `--fmad=false` lockstep list; per-row grads are
  direct stores, shared-buffer grads are `atomicAdd` (transmission-backward
  policy); JVPs are deterministic (fixed-order tree reductions, no atomics).
- The BDPT continuation estimator's differentiable factors reuse the already
  registered `scattering_table_eval_ad` companions; sampling CDFs/pdfs stay
  detached (fixed importance distribution, ADR-015 stance). BDPT-side AD
  wiring is ADR-022's scope.

## Acceptance protocol (frozen before merge)

1. **Bitwise defaults.** With `scattering_chain_max_depth=0`,
   `scattering_coherent=False`, `max_scattering_order=1`: every existing
   suite, exact-hash baseline, seed stream, and launch-ledger entry is
   bit-identical. New configs OFF never allocate, launch, or consume RNG.
2. **Degenerate collapse.** Op A(d1=d2=0) matches op-1 outputs and Op
   B(d1=d2=0) matches op-2 outputs within max-rel 1e-6 on randomized and
   canonical fixtures (expression-order parity documented per-expression in
   the PR).
3. **Specular-limit oracle.** For a smooth (h=0, sigma_h -> 0) large plate at
   the vertex, the R-S and S-R chain classes collapse to the corresponding
   two-bounce image-source reflection power (extension of the existing
   tested specular-delta collapse) within the existing oracle tolerances.
4. **Energy conservation.** Hemispherical integration gates: chain rows'
   total scattered power per vertex stays within the `(1 - C_r^2)|R|^2`
   budget times the chain throughput (no gate weakened); joint
   specular+diffuse energy conservation holds per bounce.
5. **Reciprocity.** Swapping TX/RX (and chains C1/C2) reproduces the same
   `path_gain` within f32 reassociation tolerance on canonical fixtures
   (table reciprocity + sandwich symmetry).
6. **Coherent combine parity.** Deterministic `scattering_coherent=True` on a
   phase-screen fixture: (a) equals the incoherent value when a single
   scattering row exists; (b) exhibits speckle interference for multi-row
   fixtures, reproducible bit-for-bit run-to-run; (c) BDPT-side estimator
   (unchanged, power) stays within its ADR-018-style [0.5x, 2x] gate against
   the deterministic incoherent reference.
7. **BDPT multi-order.** `max_scattering_order=2` on the wedge+rough fixture:
   unbiasedness cross-check against a brute-force reference estimator
   (tests-only oracle), default-order bitwise regression, seed-stream
   isolation test (existing streams unchanged).
8. **AD.** Lockstep native-vs-float64-Torch-oracle tests for both new ops
   (`tests/reference/` additions), FD cross-checks at solver level (heights,
   tables, layer params, frequency through a chain-scatter solve),
   JVP-vs-VJP consistency, loud rejection of every fixed input,
   `ad_mode="none"` builds no tape.
9. **Governance.** Binding manifest (+6 symbols: 2 chain-op forwards + 4
   companions, 193 -> 199), contract-coverage manifest,
   owner inventory, duplication ledger, launch ledger, public-api snapshot
   (new config fields), FEATURE_LIST, and migration notes move in the same
   change; `ci/check_import_graph.py` passes with no new debt.

## Consequences

- Deterministic/Path gain physically coherent, fully polarimetric diffuse
  scattering embedded in multi-bounce trajectories — beyond Sionna's ER
  model on lobe physics, energy balance, coherence, and polarization — and
  it is differentiable end-to-end at birth.
- BDPT gains unbiased multi-order diffuse walks in the power domain; its
  coherent diffuse carrier and MC-basic multi-bounce remain documented
  follow-ups with explicit revisit conditions.
- The scattering slot's coherent combine is opt-in and per-component,
  preserving the frozen accumulator contract by default.
