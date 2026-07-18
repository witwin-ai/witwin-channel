# Plan 10 — Scattering v2: multi-bounce coherent diffuse scattering + BDPT full AD

- **Status:** DRAFT (math framework frozen first; code-facing sections pending the
  parallel understanding scan)
- **Worktree:** `.worktrees/scattering-v2`, branch `wt/scattering-v2`, base `main@ead79eb`
- **Roles:** Fable — math design, supervision, audit, debug. Opus subagents —
  mechanical implementation, tests, experiments (via Workflow orchestration).
- **Related:** ADR-010 (native scattering kernels), ADR-014/015 (scattering AD),
  ADR-019 (BDPT coherent combine), ADR-020 (MC Jones transmission unification),
  plan 07 (two-layer fixed-topology AD).

## 1. Goals

1. **Multi-bounce diffuse scattering** in every solver: a diffuse-scatter event
   embedded at an arbitrary position inside a multi-interaction trajectory
   (specular reflections / transmissions before and after the scatter vertex),
   not only the current single-bounce `tx -> sample -> rx` topology.
2. **A scattering model more rigorous than Sionna's ER model, coherent for the
   deterministic solver.** Sionna's diffuse model is scalar effective-roughness
   lobes (Lambertian/directive/backscatter) with heuristic XPD polarization,
   random phase, and power-domain accumulation. We exceed it on four axes:
   physically derived polarimetric Kirchhoff BSDF from the actual layer stack
   (already resident), explicit energy balance (Sinkhorn-normalized diffuse
   budget `(1-C_r^2)|R|^2`, already resident), **true coherent evaluation from a
   surface-height realization** (phase screen -> deterministic complex speckle
   field, not random phase), and **full Jones/coherency-matrix polarization
   through the whole multi-bounce chain** (no scalar XPD heuristic).
3. **BDPT full AD** under the plan-07 fixed-topology / fixed-winner contract
   (geometric discontinuities explicitly out of scope). Lift the solver-wide
   `ad_mode` rejection; gradients flow to material/scattering/frequency/power
   parameters through native backward/JVP companions only.

## 2. Physics framework (authoritative math, ADR-021 core)

### 2.1 Path classes

Deterministic/path solvers gain the enumerated path class

```
TX --C1--> v_s --C2--> RX
```

where `v_s` is a diffuse-scatter vertex drawn from the surface sample set
(existing R2 low-discrepancy sampler / patch subdivision), and `C1`, `C2` are
specular interaction chains (reflection/transmission; diffraction composition
deferred, see 2.6) of depth `d1, d2 >= 0` solved by the existing enumerated
image-method engine with `v_s` as a virtual endpoint. v1 enumerates **exactly one
diffuse vertex per path**; `>=2` diffuse vertices remain Monte Carlo territory
(a deterministic double surface integral is O(S^2) and is a later opt-in).

MC-basic and BDPT support **any number** of diffuse events per trajectory
(stochastic sampling already walks multi-bounce chains; the estimator below is
per-event multiplicative, so depth composes).

### 2.2 Coherent (realization) model

Let `A_1(v_s)` be the 2x2 complex Jones transfer of chain C1 evaluated at the
scatter vertex (existing deterministic multi-bounce Jones accumulation: Fresnel
stack coefficients per bounce, unfolded length `L_1`, spreading `sp_1`), in the
local incident s/p basis at `v_s`; likewise `A_2` for C2 from the local outgoing
basis to the RX polarization basis, unfolded length `L_2`, spreading `sp_2`.

The scattered contribution of one surface patch `P` with height realization
`h(x)` (phase screen) is the Kirchhoff aperture integral already implemented in
`scattering_patch_integral_eval`, generalized in three ways:

1. **Carrier:** `exp(-j k0 (r1 + r2))` becomes `exp(-j k0 (L_1 + L_2))` with
   image-unfolded chain lengths; the local phase term
   `exp(-j (q . x + q_n h(x)))` is unchanged (`q = k0 (d_o - d_i)` with `d_i`,
   `d_o` the local incident/outgoing directions at the patch, i.e. the last leg
   of C1 and the first leg of C2).
2. **Spreading:** `1/(r1 r2)` becomes `sp_1 * sp_2` (for pure planar specular
   chains this is `1/(L_1 L_2)`, the image-theory result; the general form
   reuses whatever spreading the deterministic reflection model already
   applies per chain).
3. **Polarization:** the scalar projected coefficient
   `jones = r_te (a_te g_te) + r_tm (a_tm g_tm)` becomes the full sandwich

   ```
   E_rx = A_2 . S(d_i, d_o) . A_1 . e_tx
   S(d_i, d_o) = pref(q) * [ s_o p_o ] . diag(r_te, r_tm)_at_local_angle . [ s_i p_i ]^T
   ```

   i.e. `S` is the 2x2 local Kirchhoff scattering operator in the s/p bases of
   the incident and outgoing legs (tangent-plane Kirchhoff: the smooth-stack
   Jones `r_te/r_tm` from `em_layer_stack_eval` at the local specular angle of
   `(d_i, d_o)`, framed by the exact basis rotations). The single-bounce case
   `C1 = C2 = identity` reduces bit-recognizably to today's projected scalar,
   which becomes the degenerate sandwich with `A_1 e_tx = pol_t` projection and
   `A_2^T` row = `pol_r` projection.

The per-path complex field enters the standard `PathFields.path_field`
per-component coherent accumulator, so deterministic `coherent=True` interferes
scattering paths with each other and (per existing component semantics) the
component powers combine per the existing policy. Speckle is deterministic
given the phase-screen seed — reproducible, differentiable w.r.t. heights.

### 2.3 Incoherent (ensemble) model

The ensemble average of the diffuse field is zero; only the second moment
survives. The rigorous incoherent object is the 2x2 coherency matrix
`J = E[e e^dagger]`:

```
J_in  = A_1 J_tx A_1^dagger                       (chain C1, Jones sandwich)
J_out = M_bsdf(J_in; wi, wo)                      (ensemble Kirchhoff BSDF)
J_rx  = A_2 J_out A_2^dagger                      (chain C2)
gain  = coef * cos_i * cos_o * w / (r_eff^2 ...) * (p_rx^dagger J_rx p_rx)
```

v1 table scope: the resident tables carry co-pol `f_te`, `f_tm` only, so
`M_bsdf` is diagonal in the local s/p basis:
`J_out = diag(f_te, f_tm) .* diag(J_in,ss, J_in,pp)` (element-wise on the
diagonal, cross terms dropped). This already upgrades Sionna (basis-exact
projection of an arbitrarily polarized incident coherency, no XPD scalar), and
the chain sandwiches are exact. A 4-channel cross-pol table
(`f_ss, f_sp, f_ps, f_pp` from the vector Kirchhoff kernel) is a documented v2
extension slot of the table builder; the runtime contract below is written
against the diagonal case with the off-diagonal slots reserved.

Ensemble rows remain zero-phase power rows (component-power accumulation);
that is the *definition* of ensemble mode, not a shortcut.

### 2.4 Monte Carlo estimator (MC-basic, multi-bounce)

Trajectory sampling is unchanged (fixed, detached importance sampling). For a
sampled trajectory with diffuse events at vertices `k in D` and specular events
elsewhere, the unbiased power estimator per event is multiplicative:

```
w_path = prod_{k in D} [ f_unpol(wi_k, wo_k) cos_o_k / p(wo_k) ] * (chain Jones powers)
```

with `p(wo_k)` the lobe-sampling pdf from the resident CDF tables (the current
single-scatter area-sampling estimator is the `|D|=1`, area-measure special
case; multi-bounce uses solid-angle lobe sampling between events — the sampling
kernels already exist). Polarization: same coherency-diagonal contract as 2.3.

### 2.5 Coherent MC / BDPT carrier (optional, gated)

A coherent diffuse estimator for BDPT's coherent combine: carry the complex
amplitude `A_2 S A_1 e_tx * exp(-j k0 L_tot)` per sampled trajectory with the
phase-screen height at the sampled point, divided by the same fixed pdf. This
is an unbiased estimator of the realization field of 2.2 and slots into the
ADR-019 revisit condition ("coherent field carrier for those samplers"). Scoped
as a follow-up phase, only after the deterministic coherent model is locked.

### 2.6 Interaction-composition scope

v1 chains C1/C2 contain reflection and transmission events. Diffraction+scatter
composition (a UTD vertex in C1/C2) is excluded from v1 enumeration (the
existing coupled-path policy stays authoritative); revisit with measurement.
Scatter-scatter composition deterministic: excluded (2.1). Rough-reflection
`C_r` attenuation applies per specular bounce inside C1/C2 exactly as today, and
the diffuse budget at `v_s` keeps the Sinkhorn-balanced `(1-C_r^2)|R|^2`
normalization — energy is conserved jointly across the specular/diffuse split
at every bounce.

## 3. BDPT full AD framework (ADR-022 core)

Contract: plan-07 fixed-topology/fixed-winner. All sampling decisions, pdfs,
MIS weights, visibility masks, event branches, and connection topologies are
**frozen constants** of the solve (`E[df/p]` is the unbiased gradient of
`E[f/p]` under fixed `p`; MIS weights are functions of the frozen pdfs, hence
constants). Differentiable inputs: material parameters (layer eps_r/sigma_e/
thickness), roughness (sigma_h, corr lengths) through the ADR-015 table-build
chain, phase-screen heights, frequency, tx power. Geometry discontinuities out
of scope by user directive.

Consequences of the frozen-weight structure: the only differentiable factors
are the per-sample contribution weights (Jones/field/BSDF/radiometric terms)
and the accumulation reductions. Plan:

1. Every contribution-weight native op used by BDPT gains (or already has)
   backward/JVP companions — inventory from the understanding scan; known
   already-live: `em_layer_stack_eval` AD, `scattering_table_eval` AD
   (ADR-015), enumerated-engine field AD (plan 07/ADR-014) reached through the
   ADR-008 oracle boundary.
2. New companions for the BDPT-owned kernels: connection-sample evaluation
   (`bdpt_connect_samples.cu` families) and
   `bdpt_accumulate_connection_samples` (power domain: linear scatter of the
   bin cotangent back to per-sample `contribution`; coherent domain: complex
   phasor-sum backward through `|sum|^2` finalize).
3. `ad_mode` validation lifted with per-component readiness gates; anything
   not yet covered fails loudly (no silent detach — the ADR-014 lesson).

Exact per-kernel derivative specifications are written after the scan reports
the as-built kernel inventory.

## 4. Open questions for the understanding scan (to reconcile before freezing)

1. Exact current scattering topology in enumerated mode: confirmed single
   `tx -> sample -> rx` only? Where does the row builder live and what would
   chain-endpoint batching against the enumerated engine cost?
2. Does the deterministic multi-bounce Jones accumulation expose a reusable
   "chain transfer up to vertex k" (`A_1`) and "from vertex k" (`A_2`), or is
   it fused end-to-end in one kernel (fusion boundary question, ADR-009)?
3. MC-basic: does the event walk already support diffuse at depth > 1 with
   lobe sampling, or only the single-scatter map deposit?
4. BDPT kernel inventory: full list of ops on the contribution path with their
   current AD status.
5. Phase-screen resources: per-structure screens only? Seeding/reproducibility
   contract for multi-bounce reuse?
6. Where `C_r` is applied in chains today (op 3) vs the diffuse budget at the
   vertex — confirm no double counting when a chain bounce is itself rough.

## 5. Execution phases (Workflow-orchestrated, Opus implements, Fable audits)

- **P0 Understand** — 6-reader parallel scan (running, wf_a8051871-a4d).
- **P1 ADR-021 + ADR-022** — Fable writes both ADRs from this plan + scan
  results; freeze derivative specs and acceptance gates.
- **P2 Native kernels** — generalized patch-integral/sandwich kernels + AD
  companions; BDPT backward/JVP kernels. Opus agents per kernel family,
  lockstep tests against float64 torch oracles under `tests/reference/`.
- **P3 Topology/orchestration** — enumerated scatter-chain path class, solver
  wiring, MC event glue extension, BDPT ad_mode lift. Opus agents per domain.
- **P4 Validation** — energy conservation, reciprocity, specular-limit
  collapse, single-bounce regression (bit-exact where frozen), FD/JVP-vs-VJP,
  CI quick+cuda tiers, Fable physics audit.

Every phase lands as separate commits in `wt/scattering-v2`; merge to main only
after user acceptance.
