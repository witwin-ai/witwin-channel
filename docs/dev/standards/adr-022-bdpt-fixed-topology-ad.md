# ADR-022: BDPT full fixed-topology AD

- **Status:** Accepted (2026-07-18; implemented on `wt/scattering-v2`, 24
  companion-lockstep + 22 solver-level AD gates green, user-accepted)
- **Date:** 2026-07-18
- **Kind:** New native derivative companions + solver AD wiring. Every primal
  path bitwise unchanged; `ad_mode="none"` remains the default and builds no
  tape anywhere.
- **Related:** plan 07 (two-layer fixed-topology AD, companion-dispatch
  pattern), ADR-014/015 (scattering AD; ADR-015 explicitly deferred BDPT as a
  solver-wide plan), ADR-018/019/020 (BDPT estimator structure), ADR-008
  (enumerated oracle boundary), ADR-021 (multi-bounce diffuse; its BDPT
  estimator is covered here), ADR-004 (duplication ledger).

## Context

BDPT rejects all AD: `NO_AD_MODES = {"none"}` is its only valid set
(`bdpt/config.py:131-135`), metadata hardcodes `ad_status="none"`. Its
contribution paths split three ways (per the as-built audit):

1. **LoS** — native `bdpt_endpoint_connection_samples` per (tx, rx) with a
   RayD visibility gate.
2. **Enumerated discrete blocks** — reflection (cid 1), standalone UTD
   diffraction (cid 2), coupled (cid >= 3), pure transmission (cid 5) routed
   read-only through `evaluate_enumerated_paths` (ADR-008/018/020) as
   unit-mass rows (`pdf = mis_weight = 1`), `contribution = path_gain`.
3. **Stochastic shooting sampler** — mixed reflection+transmission chains and
   scattering NEE (`connections.py:128-466`), built from native per-event ops
   (`bdpt_reflected/transmitted_light_subpath_state`, `em_layer_stack_eval`,
   `scattering_table_eval`, `scattering_table_sample`) under the
   plan-sanctioned Torch event glue.

All blocks funnel through native `bdpt_concat_connection_samples` ->
`bdpt_accumulate_connection_samples` (power, or coherent per ADR-019) ->
`bdpt_finalize_point_components` / `bdpt_finalize_component_maps`.

The underlying field/material/scattering ops already carry native AD
companions (plan 07, ADR-014/015). What has NO companions is the BDPT-owned
layer: subpath state advance, endpoint connections, accumulate, finalize.

## Decision

Lift the BDPT `ad_mode` rejection to `{"none", "jvp", "vjp"}` under the
plan-07 **fixed-topology / fixed-winner** contract, with geometry
discontinuities explicitly out of scope (user directive; same stance as every
other solver).

### Frozen-by-construction set

All of the following are constants of the solve; requesting their gradients
fails loudly (`_ad_reject_fixed_*`), and their values are bitwise identical to
`ad_mode="none"`:

- every sampled quantity: launch directions, event-selection uniforms,
  multinomial face picks, scatter directional samples, seed streams;
- every pdf and every MIS weight (with fixed sampling, `E[df/p]` is the
  unbiased gradient of `E[f/p]`; MIS weights are functions of the frozen
  pdfs and multiply through as constants);
- visibility masks, valid masks, compaction/filter/concat index structure,
  connection topology, component ids, `keep` gates;
- the 12-field `_BDPT_CONNECTION_SCHEMA` layout (frozen ABI; gradients ride
  the SAME rows, never a widened schema — the ADR-019 separate-argument
  precedent).

### Differentiable parameter set (v1)

- **All blocks:** layer `eps_r` / `sigma_e` / `thickness`, roughness
  `sigma_h` / `corr_x` / `corr_y` (through the ADR-015 Part C table-build
  chain), resident BSDF table values, phase-screen heights (where ADR-021
  Op B rows appear via the oracle), carrier `frequency`, `tx_power`.
- **Enumerated discrete blocks only:** fixed-winner geometry (tx/rx
  endpoints, mesh vertices) — inherited for free from the enumerated
  engine's existing two-layer AD when `ad_mode` is threaded through the
  oracle call.
- **Stochastic sampler:** hit-point geometry stays frozen in v1 (ray-walk
  positions are functions of frozen launch samples; differentiating them is
  a RayD-adjoint-through-the-walk follow-up, not required for material/EM
  optimization). Documented loudly in metadata
  (`ad_geometry: "enumerated_blocks_only"`).

### New native AD companions (owner: `montecarlo.bdpt.kernels`)

| Symbol | Kind |
|---|---|
| `bdpt_reflected_light_subpath_state_backward` / `_jvp` | per-event specular Jones advance |
| `bdpt_transmitted_light_subpath_state_backward` / `_jvp` | per-event slab transmission advance |
| `bdpt_endpoint_connection_samples_backward` / `_jvp` | LoS/NEE endpoint contribution |
| `bdpt_accumulate_connection_samples_backward` / `_jvp` | power AND coherent combine domains |
| `bdpt_finalize_point_components_backward` / `_jvp` | linear map finalize |
| `bdpt_finalize_component_maps_backward` / `_jvp` | linear map finalize |

Twelve new ABI symbols (+12; after ADR-021's +6 the count moves 199 -> 211;
ADR-021 lands first). Structural ops (concat, compact, filter,
count, variance, zero, mis_weights, launch-input/sampling ops) need no
companions: they are index/copy/diagnostic operations on frozen structure;
the autograd Functions route cotangents through the stored index maps in the
backward of the ops that consumed them (concat backward is a split view —
implemented in the accumulate/companion kernels, not as Torch physics).

Derivative specifications (authoritative; Torch complex pair convention as in
ADR-014):

- **Subpath advance ops.** Each event multiplies the carried Complex3 Jones
  field by a per-hit operator `O` (ReflectFrame rotation x Fresnel diag, or
  WallFrame slab operator) and scales power terms. Backward:
  `grad_field_in = O^H grad_field_out`; material partials via the SAME
  lockstep stack dual (`field_transport_ad.cuh::stack_rt_dual`) the
  transmission-sequence backward uses; per-hit accumulation into CSR
  layer-parameter grads by `atomicAdd`. JVP is the tangent-forward dual,
  elementwise, deterministic. Forward intermediates are recomputed in primal
  expression order (`--fmad=false` TU policy where the forward TU uses it).
- **Endpoint connections.** `contribution = P_src * |F|^2 * (lambda/(4*pi*L))^2 / N`:
  `d/dF = 2 * conj(F) * rest` (pair convention), `d/d lambda` chains into the
  frequency scalar, `d/d P_src` direct; `L`, `N`, visibility frozen.
- **Accumulate, power domain.** `M[b] = SUM_r contribution_r * mis_r` is
  linear: `grad_contribution_r = mis_r * grad_M[bin(r)]` — a gather, no
  atomics, deterministic. `mis_r` frozen.
- **Accumulate, coherent domain.** `P[b] = |S_b|^2`, `S_b = SUM_r c_r`:
  `grad_c_r = 2 * grad_P[b] * S_b` (real cotangent times complex sum, pair
  convention); requires saving `S_b` from forward (already materialized in
  the accumulator buffer — no new tape). `path_gain = SUM_c P_c` chains
  linearly. JVP: `t_P = 2 Re(conj(S_b) t_S_b)`, deterministic fixed-order sum.
- **Finalize ops.** Elementwise linear scaling into maps/components; backward
  is the transpose scaling, JVP elementwise.

### Python wiring

- `bdpt/config.py`: `ad_mode` validation becomes `{"none","jvp","vjp"}`;
  per-feature readiness gates fail loudly for any combination whose
  companions are not registered (never silently detach — the ADR-014
  lesson). The ADR-019 coherent+AD refusal is REPLACED by support (the
  coherent accumulate now has companions).
- The enumerated oracle call threads `ad_mode` through the public
  `evaluate_enumerated_paths` config (read-only boundary preserved; no
  internal imports added).
- The shooting sampler follows the ADR-015 Part A template: under
  `ad != none` the event glue uses live `scene.frequency` / `tx_power`
  tensors, dispatches `em_layer_stack_ad`, `scattering_table_eval_ad`, and
  the new subpath `_ad` wrappers, and registers launches in the existing
  `AdLaunchLedger`. Sampling, masks, and seeds remain bitwise identical to
  the primal (asserted by test).
- `torch.autograd.Function` wrappers follow the plan-07 pattern
  (`setup_context`, `once_differentiable`, `set_materialize_grads(False)`,
  dual unpacking, `_ad_reject_fixed_*`).
- Metadata: `ad_status` reports the active mode, plus
  `ad_geometry: "enumerated_blocks_only"` and the differentiable parameter
  inventory.

## Acceptance protocol (frozen before merge)

1. **Primal bitwise.** `ad_mode="none"` solves are bit-identical (no tape, no
   Function, no extra launches/RNG); with `ad_mode != "none"` the PRIMAL
   values equal the `"none"` values bitwise (the AD wrappers call the same
   forward symbols).
2. **Lockstep.** Each new companion vs a float64 Torch-autograd oracle under
   `tests/reference/` on randomized + canonical inputs, within the existing
   `tests/ad` tolerance framework (no gate weakened).
3. **Estimator-level FD.** Central-difference cross-checks
   (`tests/ad/_fd.py`) at the solver level with FIXED seeds: gradients w.r.t.
   layer eps_r/sigma_e/thickness, table values, sigma_h, frequency, tx_power
   on (a) an enumerated-dominated fixture, (b) a shooting-sampler fixture
   (mixed chain), (c) a scattering NEE fixture, (d) a coherent-combine
   fixture. JVP-vs-VJP consistency everywhere.
4. **Unbiased-gradient sanity.** On a fixture with an analytic expectation,
   the mean of per-seed gradients converges to the FD gradient of the mean
   (validates the frozen-pdf estimator commutation).
5. **Negative tests.** Every frozen input rejects grads/tangents loudly;
   missing symbols fail loudly; geometry gradients through the stochastic
   sampler are refused with the documented message; higher-order transforms
   rejected (first-order contract).
6. **Governance.** Binding manifest (+12 symbols -> 205), contract-coverage
   manifest, owner inventory, duplication ledger, public-api snapshot
   (config contract change), FEATURE_LIST, migration note; import graph
   clean; `EXPECTED_NATIVE_BINDING_COUNT` updated with the rebuild (final
   211 after ADR-021's 199).

## Consequences

- BDPT joins the plan-07 AD system: material/EM/frequency/power gradients
  end-to-end for every estimator block, endpoint/mesh fixed-winner geometry
  gradients for the enumerated blocks, both AD modes, coherent combine
  included. ADR-015's "BDPT rejects all ad_mode" context and ADR-019's AD
  refusal are superseded by this ADR.
- The stochastic blocks' gradient accumulation inherits the documented
  atomic nondeterminism policy; the accumulate/finalize backward chain is
  deterministic.
- Follow-ups documented, not silently attempted: RayD adjoints through the
  sampled walk (stochastic-block geometry AD), differentiable sampling
  distributions (would break the frozen-pdf argument and is intentionally
  out), coherent diffuse carrier AD (lands automatically when ADR-021's
  follow-up adds the carrier, since the accumulate companions already cover
  the coherent domain).
