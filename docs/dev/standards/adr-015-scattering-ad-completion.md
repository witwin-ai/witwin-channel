# ADR-015: Scattering AD completion — MC-basic, layer-stack chain, differentiable tables

- **Status:** Accepted
- **Date:** 2026-07-18
- **Kind:** Numerical-kernel change (new native derivative companions; every
  primal path unchanged bit-for-bit).
- **Related:** ADR-014 (deterministic scattering AD; its two deferred
  follow-ups are delivered here), ADR-010, ADR-004, plan 07 (fixed-topology
  MC AD contract).

## Context

After ADR-014 the deterministic/path scattering component is differentiable,
with two deferred follow-ups (table construction, `em_layer_stack_eval`
chain) and one untouched island (Monte Carlo scattering). Findings that
shape this ADR:

- The MC-basic scattering estimator (`montecarlo/events/scattering.py::
  scattering_map_matrix`) is area-sampling: deposit =
  `(A/N) * f_unpol * cos_i * cos_o * (lambda/4pi)^2 / (r1^2 r2^2)`, with NO
  pdf division. Its only native op in the contribution weight is
  `scattering_table_eval` (forward-only). Layout/finalize/ledger AD
  infrastructure already exists (the MC transmission template:
  `mc_los_grid_maps_ad`, `_McFinalizeComponentMapsAdFunction`,
  `AdLaunchLedger`).
- `em_layer_stack_eval` ALREADY has native AD companions
  (`em_layer_stack_backward`/`em_layer_stack_jvp` in `em_debug.cu`, dual
  library `field_transport_ad.cuh::stack_rt_dual`) and an autograd wrapper
  `materials/kernels/autograd.py::em_layer_stack_ad`, production-used by MC
  transmission. The realization scattering path still calls the plain
  forward, so `r_te`/`r_tm` remain detached from material parameters.
- The Kirchhoff table build (`scattering/tables.py::build_kirchhoff_table`)
  is a float64 CPU numpy pipeline behind `float()` casts
  (`scene/scattering_resources.py:98-110`, `tables.py:354-358`); no
  gradient reaches the resident `f_te`/`f_tm` values from roughness/layer
  parameters.
- BDPT rejects ALL ad_mode (`bdpt/config.py`, plan-07 deferral): its
  scattering probabilities are simultaneously discrete selectors and 1/p
  contribution weights inside monolithic coherent kernels. BDPT AD is a
  solver-wide future plan, NOT a scattering-specific gap; it stays out of
  scope here and keeps its explicit ad_mode rejection.

Out of scope (unchanged): geometry/visibility discontinuities (visibility
masks, multinomial face selection, event branches stay frozen winners),
`scattering_event_probabilities` / `scattering_table_sample` / `_pdf`
(sampling-only; a fixed importance distribution makes `E[df/p]` the
unbiased gradient of `E[f/p]`, so sampling tables and pdfs stay detached),
BDPT, `principal_axis_rad` native work (its runtime chain is the already
live `t1r`/`t2r` inputs of ADR-014 op 1; keeping the compiled
`rough_axis_rad` on the Torch graph is pure orchestration).

## Decision

### Part A — MC-basic scattering AD

New ABI symbols (owner `scattering/kernels/`):
`scattering_table_eval_backward`, `scattering_table_eval_jvp`.

- Forward (unchanged): `(f_te_row, f_tm_row) = T(wi, wo)` per row, the
  shared quadrilinear lookup of `scattering_table.cuh`.
- Backward inputs: `wi [N,3]`, `wo [N,3]`, `f_te`, `f_tm` (4-D tables, same
  layout as the forward), optional cotangents `grad_out_f_te [N]`,
  `grad_out_f_tm [N]`, flags `need_grad_dirs`, `need_grad_tables`. Outputs:
  `grad_wi [N,3]`, `grad_wo [N,3]` (direct stores),
  `grad_f_te`/`grad_f_tm` (16-corner atomicAdd scatter). All derivative
  math is `eval_te_tm_grad` from `scattering_table.cuh` (ADR-014) — no new
  table math.
- JVP: optional tangents for `wi`, `wo`, `f_te`, `f_tm` →
  `tangent_f_te_out [N]`, `tangent_f_tm_out [N]`; elementwise, no atomics.
- Python: `_ScatteringTableEvalAdFunction` + `scattering_table_eval_ad`
  (ADR-014 Function pattern); `tables.eval_bsdf` gains an AD dispatch.
- MC glue (follow the transmission template exactly):
  `scattering_map_matrix` / `eval_bsdf_rows` accept `ad`/`ledger`; under
  `ad` use the live `scene.frequency` tensor for `lambda`/`amplitude_sq`,
  AD-live `tx_power`, and `scattering_table_eval_ad`; the multinomial
  sample set, both visibility masks and the active gates stay frozen
  (detached). `scattering_component_map` lays out with
  `mc_los_grid_maps_ad` when `ad`. Remove `"scattering"` from
  `montecarlo/basic/pipeline.py::_AD_PENDING_COMPONENTS` and its guard
  branch; ledger-register the companion launches.
- ad_mode="none" keeps the current path bitwise (no tape, no Function).

### Part B — realization material chain

`propagation/enumerated/scattering.py::_realization_rows` dispatches
`materials.kernels.autograd.em_layer_stack_ad` when `ad_enabled` (frequency
threaded as the Torch scalar, mirroring `montecarlo/events/transmission.py:
228`), completing grad chains `total -> r_te/r_tm -> layer eps_r / sigma_e
/ thickness / frequency / cos_theta`. Plain path untouched. The
frequency-AD-over-dispersive-materials guard already runs per solve; no new
guard.

### Part C — differentiable Kirchhoff table construction

Design principle: **the float64 numpy primal build is unchanged bit-for-bit
(it stays the sanctioned compile-time island); only the derivative is new,
and it is native.** A `torch.autograd.Function`
(`_KirchhoffTableBuildAdFunction`, owner `scattering/`) wraps the build per
rough material:

- Function inputs (leaf views of the compiled store, so gradients land on
  the same tensors the transmission/reflection AD already targets):
  `rough_sigma_h_m[index]`, `rough_corr_x_m[index]`,
  `rough_corr_y_m[index]`, the material's CSR `layer_thickness_m`,
  `layer_eps_r`, `layer_sigma_e` slices, and the frequency scalar tensor.
  Fixed (reject loudly): `layer_mu_r`, grids, `principal_axis_rad` (not a
  table input).
- Forward: exactly `build_kirchhoff_table` (host float reads as today).
  Returns `f_te`, `f_tm`; the sampling tables (`sample_density`,
  `marginal_cdf`, `conditional_cdf`) are built as today and are
  non-differentiable outputs.
- Saved for backward/forward (exported by the build, downcast f32 CUDA — no
  f32 recompute drift for structural values): pre-balance symmetrized lobes
  `S_te`/`S_tm [Nti,Npi,Nto,Npo]`, balance factors `a_te`/`a_tm [Nti,Npi]`
  (= `normalization_applied` channels), budgets `r_diff_te`/`r_diff_tm
  [Nti,Npi]`, axis vectors, scalars (`sigma_h, lx, ly, k0, frequency_hz`),
  the material CSR.

New ABI symbols (owner `scattering/kernels/`):
`kirchhoff_table_build_backward`, `kirchhoff_table_build_jvp`.

Derivative specification (per polarization channel c in {TE, TM}; final
table `F_ij = a_i S_ij a_j` over directional states i = (ti, pi),
j = (to, po); `w_j = cos_o(j) * dOmega`):

1. **Balanced-table adjoint.** Given `Gbar = grad_F`:
   `abar_i = sum_j Gbar_ij S_ij a_j + sum_j Gbar_ji a_j S_ji`;
   direct term `Sbar_ij += Gbar_ij a_i a_j`.
2. **Implicit Sinkhorn adjoint.** At convergence
   `phi_i(a) = a_i * (S (w ⊙ a))_i - rhs_i = 0` with Jacobian
   `J_ik = delta_ik (S (w ⊙ a))_i + a_i S_ik w_k`. Solve `J^T lambda =
   abar` as a **float64 minimum-norm least-squares solve (dense SVD
   pseudo-inverse with a relative singular-value cutoff ~1e-10, via
   cuSOLVER in the native bridge)**; iso: 32x32 per channel — the
   isotropic reverse state collapses to cos_theta only, with
   `J_ik = delta_ik * sum_{to,po} S[i,to,po] w[to,po] a_to + a_i *
   sum_po S[i,k,po] w[k,po]`; aniso: (Nti*Npi)^2. Then
   `rhsbar_i = lambda_i` and
   `Sbar_ij += -lambda_i a_i w_j a_j`. Inactive rows (`rhs_i <= 0`, factor
   0) drop out of the system (identity rows, zero adjoint).

   *Why pseudo-inverse, and why it is exact (not a regularization):* the
   anisotropic kernel commutes with the azimuth half-turn permutation
   (`phi -> phi + pi` on both arguments leaves `q_n`, `rho^2` and `cos_h`
   invariant) while `rhs` is azimuth-independent, so `J` at the
   symmetric fixed point the uniform-init Sinkhorn iteration selects has a
   symmetry-induced null space spanned by half-turn-ANTISYMMETRIC azimuth
   modes. Every differentiable parameter (`sigma_h`, `lx`, `ly`, layer
   params, frequency) preserves that symmetry, so the perturbations
   `dS`/`drhs` and the cotangent load `abar` all lie in the SYMMETRIC
   subspace, orthogonal to the null space; the minimum-norm solution is
   therefore exactly the derivative of the symmetric selection branch —
   no information is discarded. The solve and the `a`-product
   intermediates run in float64 (balance factors can reach ~1e21 near
   grazing on coarse grids, whose pairwise products overflow float32);
   elementwise lobe/stack partials stay float32.
3. **Budget chain.** `rhs = R_bar_c(cos_i) * (1 - c_r^2)`,
   `c_r = exp(-2 (k0 cos_i sigma_h)^2)`:
   `d rhs/d sigma_h = R_bar * c_r^2 * 8 (k0 cos_i)^2 sigma_h`;
   `d rhs/d (layer params, frequency)` via the stack dual (`stack_rt_dual`
   seeds at `cos_i`, using `d|r|^2 = 2 Re(conj(r) dr)`) plus the explicit
   `k0` term of `c_r` for frequency.
4. **Raw-lobe adjoint.** `S = 0.5 (Raw + Raw_swap)` where `Raw_swap` is the
   same analytic kernel evaluated on the swapped node set: propagate
   `0.5 * Sbar` to each node set and accumulate parameter partials from
   both. Per node, `Raw = P(q) * I(q; sigma_h, lx, ly) * R_c(cos_h)` with
   `q = k0 (wo + wi)`, `P = |q|^4 / (16 pi^2 q_n^2 cos_i cos_o)`,
   `cos_h = clamp((1 + wi.wo) k0 / |q|, 1e-6, 1)` (k0-invariant), and the
   Beckmann series `I = pi lx ly sum_m exp(T_m)`,
   `T_m = m ln g - ln m! - ln m - rho^2/(4m) - g`, `g = (q_n sigma_h)^2`,
   `rho^2 = (qx lx)^2 + (qy ly)^2` (recompute the series in f32 with the
   same `n_terms` as the build):
   - `dI/d sigma_h = pi lx ly sum_m e^{T_m} (m/g - 1) * 2 q_n^2 sigma_h`
     (horizon/g=0 guarded: zero when g = 0);
   - `dI/d lx = I/lx - pi lx ly sum_m e^{T_m} qx^2 lx / (2m)` (ly
     analogous with qy);
   - `dR/d(layer params)` via `stack_rt_dual` at `cos_h` (clamp boundary →
     zero);
   - frequency: `q ∝ k0` ⇒ `dP/dk0 = 2P/k0`, `dg/dk0 = 2g/k0`,
     `d rho^2/dk0 = 2 rho^2/k0`, plus the direct stack frequency dual;
     chain `dk0/df = 2 pi / C0`.
   Scalar/CSR accumulation via atomicAdd (transmission-backward policy).
5. **JVP** mirrors 1–4 forward: parameter tangents → `dS`, `drhs`; solve
   `J da = drhs_eff - (dPhi/dS : dS)` (cuSOLVER, same J); `tangent_F =
   da_i S_ij a_j + a_i S_ij da_j + a_i dS_ij a_j`. Deterministic.

Wiring: `build_kirchhoff_resources` applies the Function when any input
requires grad (else today's path, bitwise); the graph is created at
resource-build time and cached with the resource (documented: repeated
backward through one compile follows normal retain-graph semantics). The
compiled stack `f_te_flat`/`f_tm_flat` concatenation keeps the graph, so
ADR-014 op-1 table cotangents flow into the build adjoint; MC-basic table
cotangents (Part A) flow into the same leaves.

Known follow-ups documented, not silently fixed: (1) implementers must
verify whether `ScatteringResourceKey` rebuilds tables on frequency change
and REPORT (do not change primal caching semantics in this change); (2)
frequency gradients for rough scenes are complete only through this Part C
chain — if Part C is disabled the ADR-014 coef/k0 chain remains the
documented partial derivative.

## Acceptance protocol

1. All primal paths bitwise unchanged (`ad_mode="none"`, table values,
   MC maps, BDPT).
2. Lockstep: native companions vs float64 autograd oracles under
   `tests/reference/` (table build: a float64 torch reimplementation of
   the build as test-only oracle, differentiated by torch autograd,
   including an unrolled-Sinkhorn comparison validating the implicit
   adjoint). FD cross-checks via `tests/ad/_fd.py` at op and solver level:
   MC-basic scattering map grads (table values, frequency, tx_power),
   realization material grads (eps_r/sigma_e/thickness), table-parameter
   grads (sigma_h, lx, ly, layer params) through a deterministic ensemble
   solve. JVP-vs-VJP consistency everywhere.
3. Negative tests: fixed inputs reject loudly; BDPT keeps rejecting all
   ad_mode; `ad_mode="none"` builds no graph anywhere new.
4. Manifests/coverage/ledger/FEATURE_LIST move together (binding count
   183 -> 187).
5. No tolerance, manifest, or guard weakened; the flipped MC guard test is
   replaced by enabled-path coverage.
