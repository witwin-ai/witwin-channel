# ADR-014: Native JVP/VJP companions for deterministic scattering (ops 1 and 2)

- **Status:** Accepted
- **Date:** 2026-07-18
- **Kind:** Numerical-kernel change (new native derivative companions; forward
  kernels unchanged bit-for-bit).
- **Related:** ADR-010 (native scattering kernels; ops 1/2 were specified
  forward-only, op 3 already carries AD), ADR-004 (numerical duplication),
  plan 07 (two-layer AD architecture and the `torch.autograd.Function`
  companion-dispatch pattern), CLAUDE.md compute policy (production AD must
  call registered native forward/JVP/VJP/backward companions).

## Context

The deterministic scattering component (`component_id=6`) is the last
enumerated field contributor with no derivative support. Its two production
ops are plain native forwards with no autograd registration, so scattering
rows silently detach from every differentiable parameter:

1. `scattering_ensemble_eval` (ADR-010 op 1) — Kirchhoff ensemble row
   physics. No gradient reaches the BSDF tables, the incident/outgoing
   geometry rows, or the radiometric scale.
2. `scattering_patch_integral_eval` (ADR-010 op 2) — realization-coherent
   phase-screen patch integral. No gradient reaches the phase-screen heights,
   the smooth-stack Jones coefficients `r_te`/`r_tm`, the per-row geometry,
   or `k0`.

This is not merely a missing feature: when a solve runs with
`ad_mode != "none"` and scattering enabled, upstream parameters (frequency,
endpoint geometry) receive gradients from the specular components while the
scattering contribution is dropped from the gradient, i.e. the total
derivative is wrong. ADR-010 explicitly scoped ops 1/2 forward-only, so
adding companions requires this ADR.

Out of scope (unchanged by this ADR):

- Discontinuity handling: visibility masks, `keep` thresholds, row
  selection, `_keep_strongest_per_pair`, and the sign/branch selections in
  frame construction remain frozen non-differentiable structure. The AD
  contract is fixed-topology / fixed-winner, exactly like plan 07.
- Monte Carlo scattering event glue and the CDF/sampling kernels
  (`scattering_table_sample/pdf`, `scattering_event_probabilities`) — the
  ADR-010 sanctioned islands keep their own future ADRs.
- Offline table construction (`scattering/tables.py`) stays a frozen
  compile-time input; this ADR makes the op differentiable **w.r.t. the
  resident table values**, which composes with any future differentiable
  table builder.
- Mesh-vertex gradients for the patch geometry (`patch_tris`, `patch_uvs`,
  patch normals) — frozen in v1.

## Decision

Add four native ABI symbols, owned by `scattering/kernels/` (same facade
owner as the forwards), following the exact plan-07 companion pattern:

| Symbol | Kind |
|---|---|
| `scattering_ensemble_eval_backward` | VJP of op 1 |
| `scattering_ensemble_eval_jvp` | JVP of op 1 |
| `scattering_patch_integral_eval_backward` | VJP of op 2 |
| `scattering_patch_integral_eval_jvp` | JVP of op 2 |

Python side:

- `scattering/kernels/functional.py` gains `_backward`/`_jvp` facades
  (validate contracts, dispatch required symbols, return named dicts).
- New `scattering/kernels/autograd.py` with
  `_ScatteringEnsembleEvalAdFunction` / `_ScatteringPatchIntegralAdFunction`
  (`torch.autograd.Function` with `forward` / `setup_context` /
  `once_differentiable backward` / `jvp`, `set_materialize_grads(False)`,
  dual-primal unpacking, `save_for_backward` + `save_for_forward`,
  `_ad_reject_fixed_inputs` / `_ad_reject_fixed_tangents` on fixed inputs)
  plus public `scattering_ensemble_eval_ad` / `scattering_patch_integral_eval_ad`
  wrappers, mirroring `propagation/fields/kernels/rough_scale.py`.
- `propagation/enumerated/scattering.py` dispatches the `_ad` wrappers when
  `getattr(config, "ad_mode", "none") != "none"`; with `ad_mode == "none"`
  the plain forward path runs, allocates no tape, and never routes through
  `torch.autograd.Function` (same gate as ADR-010 op 3).

### Differentiable input sets

Op 1 (`scattering_ensemble_eval`) — outputs `gain`, `amplitude`, `length`
are differentiable; `keep` is `mark_non_differentiable`.

- Live per-row: `wo_rows`, `r2_rows`, `cos_o_rows`.
- Live per-sample (VJP scatter-adds over rows sharing a sample):
  `n_o`, `t1r`, `t2r`, `wi_local`, `cos_i`, `r1`, `a_te2`, `a_tm2`,
  `weights`.
- Live tables: `f_te_flat`, `f_tm_flat` (VJP scatter into the 16
  interpolation corners).
- Live scalar: `coef` — the AD wrapper takes `coef` as a 0-dim float32 CUDA
  tensor (the plain facade keeps its float signature). The caller builds
  `coef = tx_power_value * (C0 / frequency)**2 / (4*pi)**2` as a Torch
  scalar expression so frequency gradients flow through the radiometric
  scale (ensemble rows are zero-phase power rows; `coef` is their only
  frequency dependence).
- Fixed (reject grads/tangents loudly): `rx_pol`, `backup_axis`,
  `material_id`, `rc_idx`, `sc_idx`, table metadata, `threshold`.

Op 2 (`scattering_patch_integral_eval`) — output `total` (complex64) is
differentiable; `integral` and `row_value` are test buffers,
`mark_non_differentiable`.

- Live: `heights` (VJP scatter into 4 bilinear texels per quadrature node),
  `r_te`, `r_tm` (complex, per-row), `d_i`, `d_o`, `r1_rows`, `r2_rows`,
  `centroids`, and the scalar `k0` (0-dim tensor at the AD wrapper; the
  caller builds `k0 = 2*pi*frequency/C0` as a Torch scalar expression).
- Fixed (reject loudly): `patch_tris`, `patch_uvs`, `rows`, `n_rows`,
  `pol_t`, `pol_r`, quadrature nodes/weights.

### Kernel structure and determinism

- Forward kernels are untouched; all existing forward baselines stay bitwise
  identical, including `ad_mode != "none"` solves (the AD wrapper's forward
  calls the same forward symbol).
- Backward kernels recompute the forward intermediates in lockstep with the
  primal expression order (ADR-004 duplication; ledger updated in the same
  change). Per-row gradients are direct stores; per-sample, table, texel,
  and scalar gradients accumulate with `atomicAdd` — the same accumulation
  policy as the existing transmission-layer backward (documented
  run-to-run-nondeterministic gradient accumulation; tolerances in
  `tests/ad` already absorb this).
- JVP kernels are tangent-forward duals: elementwise for op 1; for op 2 the
  tangent phasor flows through the same fixed-order shared-memory tree
  reductions as the forward, so JVP results are run-to-run deterministic.
- Op 1 AD kernels compile in the same `--fmad=false` translation-unit policy
  as the forward so recomputed primal values round identically.
- Need-flag groups gate work and allocation: op 1
  `need_grad_rows` / `need_grad_samples` / `need_grad_tables` /
  `need_grad_coef`; op 2 `need_grad_heights` / `need_grad_jones` /
  `need_grad_geometry` (d_i, d_o, r1, r2, centroids) / `need_grad_k0`.

### Derivative specification (authoritative math)

Conventions: real cotangents follow Torch (`grad_x = dL/dx`); complex
cotangents follow Torch's pair convention `g = dL/dRe(o) + j*dL/dIm(o)`,
so for a real input `x` of a complex output `o`,
`grad_x = Re(conj(g) * do/dx)`, and for a complex input `z` with `o = K*z`
(`K` the complex coefficient), `grad_z = g * conj(K)`.

**Op 1.** Per row `r` with sample `s = sc[r]`, receiver `c = rc[r]`
(fixed indices):

```
wo_local = (wo.t1r[s], wo.t2r[s], cos_o)
(f_te, f_tm) = T(wi_local[s], wo_local)            # quadrilinear table
s_o = normalize(n_o[s] x wo)   (backup branch: constant, zero partials)
p_o = s_o x wo
pol_perp = rx_pol[c] - (rx_pol[c].wo) wo
g_te = pol_perp.s_o ; g_tm = pol_perp.p_o
f_eff = f_te*a_te2[s]*g_te^2 + f_tm*a_tm2[s]*g_tm^2
base = coef * cos_i[s] * cos_o * weights[s] / (r1[s]^2 * r2^2)
gain = base * f_eff
amplitude = sqrt(max(gain, 0)) ; length = r1[s] + r2
```

Cotangent folding: `gbar = grad_gain + grad_amplitude * (gain > 0 ?
0.5/amplitude : 0)`; `lbar = grad_length`.

Partials (all products written division-free where a factor can be zero):

- `d gain/d f_eff = base`; `d f_eff/d f_te = a_te2*g_te^2`;
  `d f_eff/d a_te2 = f_te*g_te^2`; `d f_eff/d g_te = 2*f_te*a_te2*g_te`
  (TM analogous).
- Radiometric: `d gain/d cos_i = coef*f_eff*cos_o*w/(r1^2 r2^2)`,
  `d gain/d weights = coef*f_eff*cos_i*cos_o/(r1^2 r2^2)`,
  `d gain/d r1 = -2*gain/r1`, `d gain/d r2 = -2*gain/r2` (r1, r2 are
  clamped positive upstream), `d gain/d coef = f_eff*cos_i*cos_o*w/(r1^2
  r2^2)`; `d length/d r1 = d length/d r2 = 1`.
- `cos_o` enters twice: the radiometric factor
  (`coef*f_eff*cos_i*w/(r1^2 r2^2)`) and the table via `wo_local[2]`.
- Table `T` (shared header `scattering_table.cuh`; derivative helpers are
  added there so both AD kernels and any future consumer share one
  implementation):
  - Horizon gate `wi[2] <= 0 || wo[2] <= 0` → value and all partials 0.
  - Non-periodic axis (`theta_i` on `wi[2]`, `theta_o` on `wo[2]`, period
    1): `d w/d coord = n` when the pre-clamp coordinate `t = coord*n - 0.5`
    lies in `(0, n-1)`, else 0; axis with `n == 1` contributes 0.
  - Periodic axis (`phi`, period `2*pi`): `d w/d coord = n/(2*pi)`.
  - `d interp4/d w_axis` = the multilinear difference along that axis
    (interp of `table[hi] - table[lo]` over the remaining three axes).
  - `phi_i = atan2p(wi[1], wi[0])`: `d phi/d x = -y/(x^2+y^2)`,
    `d phi/d y = x/(x^2+y^2)`; same for `phi_o` from `wo_local[0/1]`.
  - `npi == 1` relative-azimuth mode: the `phi_i` axis weight is constant,
    but `phi_o' = wrap(phi_o - phi_i)` adds
    `d f/d phi_i -= d interp4/d qw * npo/(2*pi)`-style coupling
    (i.e. `d phi_o'/d phi_i = -1`).
  - Table-value VJP: scatter `gbar * base * (a_te2*g_te^2)` (TE) and
    `gbar * base * (a_tm2*g_tm^2)` (TM) times each corner weight
    `wa*wb*wc*wd` into the 16 flat-table entries (atomicAdd).
- Frames: with `u = n x wo`, `sn = |u|`, unclamped branch:
  `d s_o = (I - s_o s_o^T)/sn * (dn x wo + n x dwo)`;
  `d p_o = d s_o x wo + s_o x dwo`;
  `d pol_perp = -(rx_pol.dwo) wo - (rx_pol.wo) dwo`;
  `d g_te = d pol_perp . s_o + pol_perp . d s_o` (TM analogous).
  Degenerate branch (`sn < 1e-6`): `s_o = backup_axis[s]` is constant;
  only the `p_o = s_o x wo` and `pol_perp` chains survive.
- `wo` accumulates from: `wo_local[0/1]` table chain (`+= (df/dwo_local0)
  t1r + (df/dwo_local1) t2r`), the frame chain, and the `pol_perp` chain.
  `t1r`/`t2r` VJP: `(df/dwo_local0) * wo` and `(df/dwo_local1) * wo`.
- `wi_local` VJP: `(df/dphi_i * (-wi1/(wi0^2+wi1^2)),
  df/dphi_i * (wi0/(wi0^2+wi1^2)), df/dwi2)` with `df/dwi2` the theta_i
  axis difference times `nti` (0 when clamped or `nti == 1`).
- `n_o` VJP: frame chain only (`dn` term of `d s_o`).

JVP: the same chain evaluated tangent-forward per row, producing
`tangent_gain`, `tangent_amplitude = tangent_gain * (gain > 0 ?
0.5/amplitude : 0)`, `tangent_length = t_r1[s] + t_r2[r]`. Missing tangents
are zeros; no atomics.

**Op 2.** Per row (patch index `P = rows[row]` fixed): with fixed
`e1, e2, n_hat, A2 = |e1 x e2|`, `pos_t = p0 + a_t e1 + b_t e2`, fixed
bilinear texel weights `b_{t,k}` and `h_t = sum_k b_{t,k} H[k]`:

```
q = k0*(d_o - d_i) ; q_int = -q ; q_int_n = n_hat . q_int
phase_t = pos_t . q_int + q_int_n * h_t
I = A2 * sum_t w_t exp(-j phase_t)                  # 'integral'
pref = |q|^2 / (4*pi * max(q . n, 1e-9))            # n = n_rows[row]
jones = r_te*(a_te*g_te) + r_tm*(a_tm*g_tm)         # bases from (n, d_i|d_o)
carrier = exp(j * cphase),  cphase = -(k0*(r1+r2) + q . c)
value = (j*pref) * jones * carrier / (r1*r2)
row_value = value * I ;  total = sum_rows row_value
```

With `g = grad_total` (every row sees the same `g` since `total` is a
plain sum):

- `heights`: `d row_value/d h_t = value * A2 * w_t * (-j q_int_n)
  exp(-j phase_t)`; scatter `Re(conj(g) * that) * b_{t,k}` into the 4
  texels per node (atomicAdd).
- `r_te`: `grad_r_te[row] = g * conj((j*pref) * (a_te*g_te) * carrier * I
  / (r1*r2))`; `r_tm` analogous.
- `r1`: `d row_value/d r1 = I * value * (-j*k0 - 1/r1)`; `r2` analogous.
- `centroids`: `d row_value/d c = I * value * (-j) * q`.
- `d_i` (per-row 3-vector; `d q/d d_i = -k0 I3`,
  `d q_int/d d_i = +k0 I3`):
  - phase sum: `sum_t value * A2 * w_t * (-j) * k0*(pos_t + h_t*n_hat)
    * exp(-j phase_t)` (needs the per-node loop; reduce in the same
    fixed-order shared-memory tree as the forward),
  - prefactor: `d pref/d q = (2 q * qn_c - |q|^2 * [unclamped] * n)
    / (4*pi*qn_c^2)` with `qn_c = max(q . n, 1e-9)`, chained by `-k0`,
  - carrier: `d cphase/d d_i = +k0 * c`, contribution `I * value * j *
    k0*c`,
  - Jones: `a_te`, `a_tm` depend on `d_i` through `s_i = normalize(n x
    d_i)` (backup branch constant), `p_i = s_i x d_i`, `pt_perp = pol_t -
    (pol_t . d_i) d_i` — same frame partials as op 1.
  `d_o` symmetric (`d q/d d_o = +k0 I3`; `g_te`, `g_tm` chains).
- `k0` (scalar; `Delta = d_o - d_i`, `d q/d k0 = Delta`):
  `d phase_t/d k0 = pos_t . (-Delta) + (n_hat . (-Delta)) h_t`,
  `d pref/d k0 = (d pref/d q) . Delta`,
  `d cphase/d k0 = -(r1+r2) - Delta . c`;
  accumulate `Re(conj(g) * d row_value/d k0)` over rows into a 0-dim
  float32 (atomicAdd). The Python wrapper chains
  `grad_frequency = grad_k0 * 2*pi/C0` through the Torch scalar graph.

JVP: per-node tangent phasor `t_E = w_t * (-j * t_phase) * exp(-j phase_t)`
with `t_phase` assembled from the live tangents; `t_I` reduced in the
fixed-order tree alongside the primal recompute; thread 0 assembles
`t_row_value = t_value * I + value * t_I`; a second fixed-order stage
reduces `tangent_total`. Deterministic (no atomics).

### Facade / caller wiring

- The `_ensemble_rows` call site builds `coef` as a Torch scalar
  (`tx_power` host float stays fixed; the frequency factor becomes a graph
  node when `scene.frequency` is a live tensor) and selects
  `scattering_ensemble_eval_ad` when `ad_mode != "none"`.
- The `_realization_rows` call site builds `k0` and the outer
  `amplitude_scale = sqrt(tx_power) * lambda/(4*pi)` as Torch scalar
  expressions (both frequency-dependent) and selects
  `scattering_patch_integral_eval_ad` when `ad_mode != "none"`. `r_te`/
  `r_tm` keep their existing `em_layer_stack_eval` source; material-EM
  gradients through that op are out of scope here (boundary grads for
  `r_te`/`r_tm` are produced so the chain composes when that op gains AD).
- Phase-screen heights: `PhaseScreenRuntime.heights_m` already preserves
  the autograd graph of a `requires_grad` height tensor; a solver-level
  test locks this end-to-end.

## Acceptance protocol

1. Forward kernels byte-identical; all existing forward baselines and
   `ad_mode="none"` behavior unchanged (no tape, no autograd.Function).
2. Lockstep derivative references: extend `tests/reference/`
   (`kirchhoff_ensemble.py`, `phase_screen_realization.py`) with
   double-precision Torch-autograd oracles; native VJP/JVP must match on
   randomized and canonical inputs within the existing `tests/ad`
   tolerance framework (`tests/ad/_tolerances.py`; no gate weakened).
3. Finite-difference cross-checks via `tests/ad/_fd.py`
   (`central_difference_gradient`) at the solver level: phase-screen
   heights, BSDF table values, frequency, and endpoint geometry each get a
   non-zero, FD-consistent gradient through a deterministic solve with
   scattering enabled; `ad_mode="none"` asserts `requires_grad` is False.
4. JVP-vs-VJP consistency (forward-ad dual level vs backward) per the
   existing `tests/ad` pattern.
5. Negative tests: requesting grads/tangents for every fixed input fails
   loudly (`_ad_reject_fixed_*`); missing native symbols fail loudly (no
   fallback).
6. Manifests move together: `ci/native-binding-manifest.json`,
   contract-coverage manifest, owner inventory, duplication ledger,
   `FEATURE_LIST.md`.

## Consequences

- Deterministic/path solver scattering contributions join the plan-07 AD
  system: phase-screen heights, resident BSDF table values, frequency, and
  fixed-topology geometry become optimizable, and total gradients stop
  silently omitting the scattering component.
- Gradient accumulation for shared per-sample/table/texel/scalar buffers is
  atomic and therefore run-to-run nondeterministic at float32, matching the
  existing transmission-layer backward policy.
- Two follow-ups are explicitly deferred: differentiable table
  construction (`scattering/tables.py`) and `em_layer_stack_eval` AD for
  realization-mode material gradients.
