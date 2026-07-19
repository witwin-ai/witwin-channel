# ADR-010: Native CUDA ownership of scattering and rough-reflection physics

- **Status:** Accepted; numerical contracts retained. Cross-repository
  implementation ownership is superseded by ADR-026 after each Phase 10
  activation.
- **Date:** 2026-07-16
- **Kind:** Numerical-kernel change (Plan 08 section 6 / G2 exception path). This is
  NOT an architecture move; it deliberately replaces Torch-computed production
  physics with native CUDA kernels under pre-frozen tolerances.
- **Related:** ADR-004 (numerical duplication), ADR-009 (native fusion ownership),
  Plan 05 section 6 (scattering contract), Plan 08 section 2 (native extension is
  the only production compute backend).

## Context

Three production compute paths remained Torch-implemented after Plan 08:

1. **Ensemble Kirchhoff scattering** (`propagation/enumerated/scattering.py::_ensemble_rows`):
   per-sample frame construction, s/p polarization projection, BSDF table lookup
   dispatch, radiometric gain assembly - Torch GPU math with Python loops over
   transmitters, receiver chunks, and materials.
2. **Realization-coherent phase-screen scattering** (`_realization_rows` +
   `scattering/phase_screen.py::patch_phase_integral`): a host Python loop over
   patches (`rows.tolist()`) invoking a Torch Gauss-Legendre quadrature per patch.
3. **Rough-reflection coherent attenuation C_r**
   (`propagation/fields/evaluation.py::_rough_reflection_factor`): per-bounce
   `exp(-2 (k0 cos_theta sigma_h)^2)` product computed in Torch, differentiated by
   Torch autograd (no native companion).

These were documented Plan-05 deviations that Plan 08 (move-only, bitwise-exact)
was forbidden to touch. This ADR authorizes their migration as an independent
numerical-kernel change.

## Decision

ADR-010 authorizes and freezes the native CUDA numerical behavior described
below. It originally placed the implementations in Channel because that was the
active native boundary at the time. ADR-026 later accepts RayD as the final
source owner for the generic scattering runtime families without changing this
ADR's numerical, fusion, launch, reduction, AD, or acceptance contracts.
Channel remains the production numerical owner until the corresponding Phase
10 pin/switch/delete commit activates the complete RayD family.

Introduce three native op families, each with a single Python kernel facade owner:

### Op 1: `scattering_ensemble_eval` (owner: `scattering/kernels/`)

Replaces the per-row Torch physics of `_ensemble_rows` between/after the RayD
visibility calls. One launch per (tx, rx-chunk).

- Inputs: sample arrays `[S]` (`points`, `n_o`, `t1r`, `t2r`, `wi_local`, `cos_i`,
  `r1`, `a_te2`, `a_tm2`, `weights`, `material_id`, `backup_axis`), rx-block arrays
  `[Rc]` (`rx_positions`, `rx_pol`), visibility-surviving row index pairs `[R]`
  (`rc`, `sc`), stacked per-material BSDF tables `f_te`/`f_tm` `[M, ti, pi, to, po]`
  plus a `material -> table slot` map, scalars (`tx_power`, `power_scale`,
  `threshold`, `min_cos`).
- Per-row math (identical expression order to the Torch source, documented
  per-expression in the migration PR): `to_rx`, `r2`, `wo_w`, `cos_o`, `wo_local`,
  bilinear table lookup (MUST reuse the existing device interpolation primitive
  from `kernels/scattering.cu` - shared device header, not a copy), outgoing s/p
  basis, receiver co-pol projections, `f_eff`, radiometric `gain`, `keep`
  threshold, `amplitude = sqrt(max(gain, 0))`, `length = r1 + r2`.
- Outputs `[R]`: `gain` (f32), `amplitude` (f32), `length` (f32), `wo_w` (f32x3),
  `keep` (bool). Row compaction and `_RowCollector` assembly stay Torch
  (structural, not physics).
- Elementwise, no atomics: bitwise run-to-run deterministic.
- R2 low-discrepancy sampling, `_keep_strongest_per_pair` (float64 sort keys), and
  RayD visibility remain unchanged outside the op.

### Op 2: `scattering_patch_integral_eval` (owner: `scattering/kernels/`)

Replaces the host per-patch loop and the per-patch Torch quadrature of
realization mode. One launch per (tx, rx, structure).

- Inputs: `patch_tris [P,3,3]`, `patch_uvs [P,3,2]`, selected patch rows `[R]`,
  per-row `k_i_vec`/`k_s_vec` (already swapped at the call site per the module
  docstring phase convention), phase-screen height grid `[H,W]` with the
  half-texel edge-clamp bilinear convention of `PhaseScreenRuntime.sample_height`,
  Duffy-mapped Gauss-Legendre nodes/weights (host-precomputed, `n_quad = 16`),
  per-row Jones inputs (`r_te`, `r_tm` from the existing native
  `em_layer_stack_eval` launch, polarization vectors, `n_o`, `d_i`, `d_o`),
  per-row `r1`, `r2`, `centroids`, scalar `k0`.
- Per-row math: prefactor `j k0 |q|^2 / (k0 q_n clamp) / (4 pi)`, Jones assembly,
  carrier `exp(-j (k0 (r1+r2) + q . c))`, triangle quadrature
  `I_p = 2A sum_q w_q exp(-j (q . x + q_n h(u,v)))`, and the weighted total
  `sum_p coef_p I_p / (r1_p r2_p)`.
- Reduction: fixed-order two-stage tree reduction (no float atomics) so the total
  is bitwise stable run-to-run on the same binary.
- Output: 0-dim complex64 `total` (+ per-row integral buffer for tests).
- Patch subdivision, area-weighted mean geometry/exports stay Torch (setup and
  metadata, not the hot loop).

### Op 3: `field_rough_reflection_scale` + `_backward` + `_jvp` (owner: `propagation/fields/kernels/`)

Replaces `_rough_reflection_factor` and its Torch-autograd differentiation.

- Forward inputs: `positions [R,D,3]`, `normals [R,D,3]`, `source [R,3]`,
  per-bounce `sigma_b [R,D]`, `rough_b [R,D]` (bool), `replaced [R]` (bool,
  realization delta replacement at depth 1), `frequency` (0-dim tensor or
  scalar). Output: `factor [R]` (f32).
- Math: `seg = pos_b - pos_{b-1}` (with `pos_0 = source`), `cos_b =
  |seg_dir . n_b|`, `att_b = exp(-2 (k0 cos_b sigma_b)^2)` where rough, else 1;
  `factor = prod_b att_b`, then `factor = 0` where `replaced`.
- AD contract: the backward/jvp companions must produce gradients for exactly the
  input set that Torch autograd currently reaches (enumerate from `tests/ad`
  before implementation; at minimum `frequency`, and `positions` when the
  fixed-winner geometry AD path is live). `ad_mode=none` must not allocate tape
  and must not route through `torch.autograd.Function`.
- The application of the factor onto `field_vector` / `coefficient` /
  `path_field` / `path_gain` moves into the same facade so the reflection
  hot path has no residual Torch physics; the facade registers its launches in
  the existing AD launch ledger exactly like the other field kernels.

## Acceptance protocol (frozen BEFORE merge)

1. Baseline: the extended runtime exact-hash profile (Plan 08 section 9 backfill,
   frozen at the pre-change commit) is the comparison object. Cells that do not
   exercise these ops (LoS, smooth reflection, transmission, diffraction, coupled,
   all MC-basic and BDPT cells) MUST remain bitwise identical.
2. Tolerances for changed cells (numerical-kernel PR gates; may not be relaxed
   after merge):
   - ensemble per-row `gain` / `path_field`: max-rel <= 1e-6 (f32 reassociation
     budget), max-abs <= 1e-9 * max|baseline|.
   - realization `total` per (tx, rx, structure): max-rel <= 1e-5 (quadrature
     reduction-order change).
   - `C_r` factor: max-rel <= 1e-6; gradients: all existing `tests/ad` tolerances
     unchanged.
3. Lockstep references: the previous Torch implementations move to
   `tests/reference/` (test-only; MUST NOT remain importable from production
   packages) and dedicated lockstep tests compare native vs reference on
   randomized and canonical inputs.
4. All existing oracle/analytic/parity gates (specular-limit collapse,
   `physics.oracle` quadrature cross-checks, Munich, energy conservation) run
   unchanged and must pass without tolerance edits.
5. Launch-count changes are expected (fewer launches) and are documented
   cell-by-cell in the migration PR; the post-change launch ledger is frozen as a
   new baseline artifact after acceptance.
6. Binding manifest, contract-coverage manifest, native owner inventory, and
   no-fallback tests are updated in the same PR per their update protocols; every
   new symbol gets a shape/dtype/device contract test and an end-to-end caller.

## Consequences

- The enumerated scattering stage and rough-reflection attenuation become
  native-pure; the remaining Torch in those paths is structural (row selection,
  concatenation, metadata).
- Two intentional Torch physics islands remain out of scope: Monte Carlo
  scattering event glue (`montecarlo/events/scattering.py`, plan-sanctioned
  Python event semantics over native sampling/CDF/BSDF kernels) and CPU/numpy
  compile-time table construction (`scattering/tables.py`, cached per compiled
  scene). Revisit only with their own ADRs.
- ADR-026 moves only the final native source owner for the generic runtime
  operations. It does not move table/phase-screen lifecycle, MC/BDPT event
  policy, topology, accumulation, or the test-only references accepted here.
