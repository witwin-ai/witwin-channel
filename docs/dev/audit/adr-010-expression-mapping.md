# ADR-010 expression mapping: Torch source to native kernel

Per-expression mapping for the three ADR-010 ops. "Torch source" cites the
pre-migration production code (now preserved verbatim under `tests/reference/`);
"kernel" cites the native CUDA implementation. The op TUs
`scattering_ensemble.cu` and `scattering_patch_integral.cu` compile with
`--fmad=false` so implicit fma contraction cannot change the rounding of the
mul/add chains relative to Torch's per-op kernels; explicit `fmaf` calls (the
shared table interpolation) are unaffected.

## Op 1: `scattering_ensemble_eval`

Torch source: `_ensemble_rows` inner rx-chunk loop
(reference: `tests/reference/kirchhoff_ensemble.py`).
Kernel: `native/channel_native/kernels/scattering_ensemble.cu::ensemble_eval_kernel`.

The candidate grid (`to_rx`, `r2`, `wo_w`, `cos_o` over `[Rc, S]`) STAYS Torch
per the ADR's explicit allowance; the facade gathers the surviving rows
(`wo_w[rc, sc]`, `r2[rc, sc]`, `cos_o[rc, sc]`) so the kernel consumes bitwise
the values the previous Torch physics consumed (the old code performed the same
gathers). The `torch.nonzero` compaction and the `_RowCollector` concat stay
Torch (structural).

| Torch expression | Kernel expression |
|---|---|
| `wo_row = wo_w[rc, sc]` | `wo = wo_rows[row]` (facade gather) |
| `r2_row = r2[rc, sc]` | `r2 = r2_rows[row]` (facade gather) |
| `cos_o_row = cos_o[rc, sc]` | `cos_o = cos_o_rows[row]` (facade gather) |
| `(wo_row * t1r[sc]).sum(-1)` | `dot3(wo, t1)` = `(p0 + p2) + p1` (Torch's 2-accumulator `sum(-1)` order) |
| `(wo_row * t2r[sc]).sum(-1)` | `dot3(wo, t2)` (same order) |
| `wo_local = stack((.., .., cos_o_row))` | `wo_local[3] = {dot3(wo,t1), dot3(wo,t2), cos_o}` |
| `eval_bsdf(table, wi_local, wo_local)` per material mask | `st::eval_te_tm(fte+off, ftm+off, dims...)` on the stacked `[M]` tables via `material_slot[material_id[s]]`; device interpolation shared verbatim from `scattering.cu` through `scattering_table.cuh` |
| `s = cross(n_o[sc], wo_row)` | `s_raw = cross3(n, wo)` |
| `norm(s) < 1e-6 -> backup_axis[sc]` | `sn < 1e-6f -> load3(backup_axis, s)` |
| `normalize_vec3(s)` = `s / norm.clamp_min(1e-12)` | `s_raw / fmaxf(sn, 1e-12f)` |
| `p_o = cross(s_o, wo_row)` | `p_o = cross3(s_o, wo)` |
| `pol_r_perp = pol_r - (pol_r.wo) wo` | `pol_r_perp = pol_r - dot3(pol_r, wo) * wo` |
| `g_te2 = (pol_r_perp . s_o)^2` | `g_te * g_te` |
| `g_tm2 = (pol_r_perp . p_o)^2` | `g_tm * g_tm` |
| `f_eff = f_te*a_te2[sc]*g_te2 + f_tm*a_tm2[sc]*g_tm2` | `(f_te * a_te2[s]) * g_te2 + (f_tm * a_tm2[s]) * g_tm2` |
| `gain = coef * f_eff * cos_i[sc] * cos_o_row * weights[sc] / (r1[sc].square() * r2_row.square())` with `coef = float(tx_power) * power_scale` (host double product, rounded to f32 at the first tensor op) | `num = coef*f_eff; num *= cos_i[s]; num *= cos_o; num *= weights[s]; gain = num / ((r1s*r1s) * (r2*r2))` with `coef` passed as double and cast to f32 |
| `keep = gain > max(threshold, 0.0)` | `gain > threshold` (facade passes `max(threshold, 0.0)`) |
| `amplitude = gain.clamp_min(0.0).sqrt()` | `sqrtf(fmaxf(gain, 0.0f))` |
| `length = r1[sc] + r2_row` | `r1s + r2` |
| `direction = wo_row` | facade reuses the gathered `wo_row` (kernel emits no direction) |

## Op 2: `scattering_patch_integral_eval`

Torch source: `_realization_rows` per-row assembly plus the
`rows.tolist()` loop over `patch_phase_integral`
(reference: `tests/reference/phase_screen_realization.py`;
`patch_phase_integral` itself remains a public utility in
`scattering/phase_screen.py`, now test/reference-only in production terms).
Kernel: `native/channel_native/kernels/scattering_patch_integral.cu`
(`patch_integral_rows_kernel` + `patch_integral_total_kernel`).

| Torch expression | Kernel expression |
|---|---|
| `backup_axis = _stable_tangent(n_rows)` | `stable_tangent(n)` (first-min one-hot axis, Gram-Schmidt, normalize) |
| `s_i, p_i = _sp_basis(n_rows, d_i, backup)` | `sp_basis(n, di, backup, s_i, p_i)` |
| `s_o, p_o = _sp_basis(n_rows, d_o, backup)` | `sp_basis(n, dov, backup, s_o, p_o)` |
| `pol_t_perp = pol_t - (pol_t.d_i) d_i` | `pt_perp = pt - dot3(pt, di) * di` |
| `pol_r_perp = pol_r - (pol_r.d_o) d_o` | `pr_perp = pr - dot3(pr, dov) * dov` |
| `jones = r_te*(a_te*g_te) + r_tm*(a_tm*g_tm)` | complex `jones = te*(a_te*g_te) + tm*(a_tm*g_tm)` |
| `k_i_vec = d_i * k0`, `k_s_vec = d_o * k0` | `kiv = di * k0`, `ksv = dov * k0` (per-component rounding preserved) |
| `q = k_s_vec - k_i_vec` | `q = sub3(ksv, kiv)` |
| swapped-argument call `patch_phase_integral(.., k_s_vec, k_i_vec, ..)` giving `q' = k_i_vec - k_s_vec = -q` | `q_int = sub3(kiv, ksv)` |
| `prefactor = 1j*k0*(q_norm^2/(k0*q_n.clamp_min(1e-9)))/(4*pi)` | purely imaginary `pref_im = k0 * (dot3(q,q) / (k0 * fmaxf(dot3(q,n), 1e-9f))) / (4*pi)` |
| `carrier = polar(1, -(k0*(r1+r2) + q.centroids))` | `sincosf(-(k0*(r1v+r2v) + dot3(q, c_row)))` |
| Duffy nodes: `leggauss(16)` float64, `xi = 0.5*(nodes+1)`, `a = xi`, `b = eta*(1-xi)`, `w2d = w1 w1 (1-xi)` | identical host-side construction in the facade `_duffy_nodes` (float64 -> float32), passed as `[256]` buffers |
| `pos = tri0 + a*e1 + b*e2` | same, one node per thread |
| `uv = uv0 + a*(uv1-uv0) + b*(uv2-uv0)` | same |
| `runtime.sample_height(uv)` (half-texel edge-clamp bilinear) | `sample_height(heights, H, W, u, v)` replicating the clamp-before-floor convention exactly |
| `phase = pos @ q' + q_n' * h` with `q_n' = n_hat @ q'`, `n_hat` from the winding normal | `phase = dot3(pos, q_int) + q_int_n * h` with `q_int_n = dot3(n_hat, q_int)` |
| `phasor = polar(1, -phase)` | `sincosf(-phase)` |
| `contrib = (phasor * w2d).sum() * double_area` | fixed-order shared-memory tree reduction over the 256 node terms, then `* double_area` |
| `total += (prefactor*jones*carrier/(r1*r2)) * integral` per patch (host loop) | `row_value[row] = ((j*pref_im)*jones)*carrier/(r1v*r2v)*integral`; `patch_integral_total_kernel` reduces `row_value` in a fixed strided + tree order (no atomics) |
| `info["realization_patch_integrals"] += 1` per patch | facade adds `int(rows.numel())` host-side (same value) |

Deviations (within the 1e-5 gate): the quadrature sum is a tree reduction
rather than Torch's reduction order; `sincosf` differs from Torch's
`sin`/`cos` at the ulp level; the per-row `|k| == k0` re-validation of the
public utility is skipped (the per-row wave vectors are constructed as
`d * k0` at the call site by construction). Measured: canonical realization
cell total rel 2.8e-6; coherent randomized cases <= 4.3e-6.

## Op 3: `field_rough_reflection_scale` (+ `_backward`, `_jvp`)

Torch source: `_rough_reflection_factor` plus the Python-side application in
`_evaluate_reflection_fields` (reference: `tests/reference/rough_reflection.py`).
Kernel: `native/channel_native/kernels/field_rough_scale.cu`.

| Torch expression | Kernel expression |
|---|---|
| `prev = cat((source.unsqueeze(1), positions[:, :d-1]))`; `seg = positions - prev` | per-bounce `seg = positions[b] - (b == 0 ? source : positions[b-1])` |
| `seg_dir = seg / norm(seg).clamp_min(1e-9)` | `seg / fmaxf(len, 1e-9f)` (division, not reciprocal multiply) |
| `cos_b = (seg_dir * normals).sum(-1).abs()` | `fabsf(dot)` with the products kept in registers via `__fmul_rn`/`__fadd_rn` (no fma contraction) |
| `att = exp(-2*(k0*cos_b*sigma_b).square())` | `expf(-2.0f * (u*u))`, `u = k0*cos*sigma` (square-then-scale association preserved) |
| `c_r = where(rough_b, att, 1)`; `factor = c_r.prod(dim=1)` | in-thread product over the depth loop, skipping non-rough bounces |
| `factor = where(replaced, 0, factor)` | `if (replaced[row]) factor = 0` |
| `field_vector * factor[:, None]` | `cscale(field_vector, factor)` |
| `coefficient * factor`, `path_field * factor` | `cscale(.., factor)` |
| `path_gain * factor.square()` | `path_gain * factor * factor` |

AD contract (enumerated from `tests/ad` + the frozen rough-reflection-cr
jvp/vjp cells before implementation): gradients/tangents flow to frequency
(`dC_r/df`, always) and to the four field inputs; positions/normals/source
receive gradients only under the fixed-winner geometry AD path (`sigma_b`,
`rough_b`, `replaced` are fixed and fail loudly). `ad_mode="none"` calls the
plain facade: no `torch.autograd.Function`, no tape.
