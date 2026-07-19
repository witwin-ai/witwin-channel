// ADR-021 Op A AD companions: scattering_chain_ensemble_eval_backward / _jvp.
//
// Lockstep VJP/JVP of scattering_chain_ensemble.cu. Both companions recompute
// the forward chain intermediates in primal expression order (shared
// ReflectionChain-style on-stack state, per-bounce dual slab_fresnel exactly as
// field_transport_reflection.cu), so the derivatives differentiate the identical
// forward chain. Per-row / per-bounce grads are direct stores (deterministic);
// the table 16-corner scatter and the scalar coef/frequency grads use atomicAdd
// (transmission-backward policy, plan 10a section 3.2). Compiled --fmad=false.
//
// Wave scope (see the change report): the material/table/coef/frequency VJP
// groups (need_grad_chain1, need_grad_chain2, need_grad_tables, need_grad_coef,
// need_grad_frequency) are complete. The reverse-mode continuous chain-geometry
// group (need_grad_geometry: positions/normals/d_i/d_o/v_normal/L/cos) is NOT
// yet derived in this wave and is rejected loudly (no silent zeros); the JVP
// covers geometry in forward mode, and a follow-up wave adds the reverse.

#include "field_transport_ad_common.cuh"
#include "scattering_table.cuh"

namespace {

namespace st = channel_native::scattering_tables;

__device__ __forceinline__ field::float3a load_chain3f(
    const float* p, int64_t row, int bounce) {
    const int64_t b = (row * kMaxAdDepth + bounce) * 3;
    return field::make_f3(p[b], p[b + 1], p[b + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual_chain3f(
    const float* values, const float* tangents, int64_t row, int bounce) {
    return {
        load_chain3f(values, row, bounce),
        tangents != nullptr ? load_chain3f(tangents, row, bounce)
                            : field::f3_zero()};
}

// Per-bounce forward state saved for the reverse pass (materials path).
struct BounceSave {
    transport::ReflectFrame frame;
    field::Complex3 value_in;
    field::Complex e_s;
    field::Complex e_p;
    field::Complex r_te;
    field::Complex r_tm;
};

// Vertex s/p basis (mirrors scattering_chain_ensemble.cu::sp_basis).
__device__ __forceinline__ void sp_basis(
    field::float3a n, field::float3a d, field::float3a backup,
    field::float3a& s, field::float3a& p) {
    const field::float3a s_raw = field::f3_cross(n, d);
    const float sn = field::safe_length(s_raw);
    if (sn < 1.0e-6f) {
        s = backup;
    } else {
        s = field::f3_mul(s_raw, 1.0f / fmaxf(sn, 1.0e-12f));
    }
    p = field::f3_cross(s, d);
}

// Forward specular Jones transport of one leg, saving per-bounce state and
// returning the final field plus the final-leg direction.
__device__ __forceinline__ field::Complex3 walk_leg_save(
    field::Complex3 value,
    field::float3a start,
    field::float3a end,
    const float* positions,
    const float* normals,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    int64_t row,
    int depth,
    float frequency_hz,
    BounceSave* saves,
    field::float3a& last_dir) {
    const field::float3a e_z = field::make_f3(0.0f, 0.0f, 1.0f);
    field::float3a previous = start;
    const field::float3a first_target =
        depth > 0 ? load_chain3f(positions, row, 0) : end;
    field::float3a outgoing = field::safe_normalize(
        field::f3_sub(first_target, start), e_z);
    for (int bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_chain3f(positions, row, bounce);
        const field::float3a incident = field::safe_normalize(
            field::f3_sub(hit, previous), outgoing);
        const transport::ReflectFrame frame = transport::reflect_frame(
            incident, load_chain3f(normals, row, bounce));
        const int64_t s = row * kMaxAdDepth + bounce;
        field::Complex r_te;
        field::Complex r_tm;
        transport::slab_fresnel(
            frame.cos_theta, eps_r[s], sigma_e[s], mu_r[s], gain[s],
            thickness[s], frequency_hz, r_te, r_tm);
        const field::Complex e_s = transport::complex3_dot_real(
            value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(
            value, frame.p_in);
        saves[bounce].frame = frame;
        saves[bounce].value_in = value;
        saves[bounce].e_s = e_s;
        saves[bounce].e_p = e_p;
        saves[bounce].r_te = r_te;
        saves[bounce].r_tm = r_tm;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    last_dir = field::safe_normalize(field::f3_sub(end, previous), outgoing);
    return value;
}

// Reverse the saved leg for the material (eps/sigma/gain/thickness) and
// frequency gradients only. g_field is the cotangent on the leg's output field.
// Material grads are ADDED into the per-bounce slices (row-owned, no atomics);
// g_freq accumulates the frequency cotangent across bounces.
__device__ __forceinline__ void reverse_leg_materials(
    const BounceSave* saves,
    int depth,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    int64_t row,
    float frequency_hz,
    field::Complex3 g_field,
    bool need_material,
    float* grad_eps_r,
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    bool need_frequency,
    float& g_freq) {
    field::Complex3 g_chain = g_field;
    for (int bounce = depth - 1; bounce >= 0; --bounce) {
        const BounceSave& sv = saves[bounce];
        field::float3a g_dump = field::f3_zero();
        field::Complex gs = field::cplx_zero();
        field::Complex gp = field::cplx_zero();
        field::adj_cplx_scale_real(
            sv.frame.s_axis, field::cplx_mul(sv.r_te, sv.e_s), g_chain,
            g_dump, gs);
        field::adj_cplx_scale_real(
            sv.frame.p_out, field::cplx_mul(sv.r_tm, sv.e_p), g_chain,
            g_dump, gp);
        field::Complex g_r_te = field::cplx_zero();
        field::Complex g_e_s = field::cplx_zero();
        field::Complex g_r_tm = field::cplx_zero();
        field::Complex g_e_p = field::cplx_zero();
        field::adj_cplx_mul(sv.r_te, sv.e_s, gs, g_r_te, g_e_s);
        field::adj_cplx_mul(sv.r_tm, sv.e_p, gp, g_r_tm, g_e_p);
        field::Complex3 g_value_in = field::c3_zero();
        field::adj_cplx_dot_real(
            sv.value_in, sv.frame.s_axis, g_e_s, g_value_in, g_dump);
        field::adj_cplx_dot_real(
            sv.value_in, sv.frame.p_in, g_e_p, g_value_in, g_dump);
        g_chain = g_value_in;

        const int64_t s = row * kMaxAdDepth + bounce;
        const float b_eps = eps_r[s];
        const float b_sigma = sigma_e[s];
        const float b_mu = mu_r[s];
        const float b_gain = gain[s];
        const float b_thick = thickness[s];
        DualC rte_d;
        DualC rtm_d;
        if (need_material) {
            ad::slab_fresnel_dual(
                sv.frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thick,
                frequency_hz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, rte_d, rtm_d);
            grad_eps_r[s] += adj_dot(g_r_te, rte_d.d) + adj_dot(g_r_tm, rtm_d.d);
            ad::slab_fresnel_dual(
                sv.frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thick,
                frequency_hz, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, rte_d, rtm_d);
            grad_sigma_e[s] += adj_dot(g_r_te, rte_d.d) + adj_dot(g_r_tm, rtm_d.d);
            ad::slab_fresnel_dual(
                sv.frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thick,
                frequency_hz, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, rte_d, rtm_d);
            grad_gain[s] += adj_dot(g_r_te, rte_d.d) + adj_dot(g_r_tm, rtm_d.d);
            ad::slab_fresnel_dual(
                sv.frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thick,
                frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, rte_d, rtm_d);
            grad_thickness[s] += adj_dot(g_r_te, rte_d.d) + adj_dot(g_r_tm, rtm_d.d);
        }
        if (need_frequency) {
            ad::slab_fresnel_dual(
                sv.frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thick,
                frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, rte_d, rtm_d);
            g_freq += adj_dot(g_r_te, rte_d.d) + adj_dot(g_r_tm, rtm_d.d);
        }
    }
}

__global__ void chain_ensemble_backward_kernel(
    int64_t count,
    const float* __restrict__ tx_pol,
    const float* __restrict__ rx_pol,
    const float* __restrict__ source,
    const float* __restrict__ vertex,
    const float* __restrict__ target,
    const float* __restrict__ c1_positions,
    const float* __restrict__ c1_normals,
    const float* __restrict__ c1_eps_r,
    const float* __restrict__ c1_sigma_e,
    const float* __restrict__ c1_mu_r,
    const float* __restrict__ c1_gain,
    const float* __restrict__ c1_thickness,
    const int* __restrict__ c1_depth,
    const float* __restrict__ c2_positions,
    const float* __restrict__ c2_normals,
    const float* __restrict__ c2_eps_r,
    const float* __restrict__ c2_sigma_e,
    const float* __restrict__ c2_mu_r,
    const float* __restrict__ c2_gain,
    const float* __restrict__ c2_thickness,
    const int* __restrict__ c2_depth,
    const float* __restrict__ n_o,
    const float* __restrict__ t1r,
    const float* __restrict__ t2r,
    const float* __restrict__ backup_axis,
    const float* __restrict__ wi_local,
    const float* __restrict__ cos_i,
    const float* __restrict__ cos_o,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ l1,
    const float* __restrict__ l2,
    const float* __restrict__ weights,
    const int* __restrict__ material_id,
    const float* __restrict__ fte_flat,
    const float* __restrict__ ftm_flat,
    const int64_t* __restrict__ table_offset,
    const int* __restrict__ table_dims,
    const int* __restrict__ material_slot,
    float coef,
    float frequency_hz,
    const float* __restrict__ grad_gain,
    const float* __restrict__ grad_amplitude,
    float* __restrict__ grad_c1_eps_r,
    float* __restrict__ grad_c1_sigma_e,
    float* __restrict__ grad_c1_gain,
    float* __restrict__ grad_c1_thickness,
    float* __restrict__ grad_c2_eps_r,
    float* __restrict__ grad_c2_sigma_e,
    float* __restrict__ grad_c2_gain,
    float* __restrict__ grad_c2_thickness,
    float* __restrict__ grad_f_te,
    float* __restrict__ grad_f_tm,
    float* __restrict__ grad_coef,
    float* __restrict__ grad_frequency,
    bool need_chain1,
    bool need_chain2,
    bool need_tables,
    bool need_coef,
    bool need_frequency) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d1 = c1_depth[row];
        const int d2 = c2_depth[row];
        const field::float3a n = load3f(n_o, row);
        const field::float3a backup = load3f(backup_axis, row);
        const field::float3a di = load3f(d_i, row);
        const field::float3a dobj = load3f(d_o, row);
        const field::float3a src = load3f(source, row);
        const field::float3a vtx = load3f(vertex, row);
        const field::float3a tgt = load3f(target, row);

        // ---- Forward recompute (primal expression order). ----
        BounceSave c1_saves[kMaxAdDepth];
        BounceSave c2s_saves[kMaxAdDepth];
        BounceSave c2p_saves[kMaxAdDepth];
        const field::float3a first_target =
            d1 > 0 ? load_chain3f(c1_positions, row, 0) : vtx;
        const field::float3a first_leg = field::safe_normalize(
            field::f3_sub(first_target, src), field::make_f3(0.0f, 0.0f, 1.0f));
        const field::float3a tx_axis = field::project_to_wedge_plane(
            load3f(tx_pol, row), first_leg);
        field::float3a c1_last;
        const field::Complex3 e_in = walk_leg_save(
            field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f)), src, vtx,
            c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
            c1_thickness, row, d1, frequency_hz, c1_saves, c1_last);
        field::float3a s_i;
        field::float3a p_i;
        sp_basis(n, di, backup, s_i, p_i);
        const field::Complex c_s = transport::complex3_dot_real(e_in, s_i);
        const field::Complex c_p = transport::complex3_dot_real(e_in, p_i);
        const float p_te = field::cplx_abs_sqr(c_s);
        const float p_tm = field::cplx_abs_sqr(c_p);

        const field::float3a t1 = load3f(t1r, row);
        const field::float3a t2 = load3f(t2r, row);
        const float co = cos_o[row];
        const float wo_local[3] = {
            field::f3_dot(dobj, t1), field::f3_dot(dobj, t2), co};
        st::TableEvalGrad tg;
        tg.active = false;
        tg.te = 0.0f;
        tg.tm = 0.0f;
        const int slot = material_slot[material_id[row]];
        int64_t table_base = 0;
        if (slot >= 0) {
            table_base = table_offset[slot];
            const int nti = table_dims[slot * 4 + 0];
            const int npi = table_dims[slot * 4 + 1];
            const int nto = table_dims[slot * 4 + 2];
            const int npo = table_dims[slot * 4 + 3];
            st::eval_te_tm_grad(
                fte_flat + table_base, ftm_flat + table_base, nti, npi, nto, npo,
                wi_local + row * 3, wo_local, tg);
        }
        const float f_te = tg.te;
        const float f_tm = tg.tm;

        field::float3a s_o;
        field::float3a p_o;
        sp_basis(n, dobj, backup, s_o, p_o);
        field::float3a c2_last;
        const field::Complex3 field_s = walk_leg_save(
            field::cplx_scale_real(s_o, field::cplx(1.0f, 0.0f)), vtx, tgt,
            c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, row, d2, frequency_hz, c2s_saves, c2_last);
        field::float3a c2_last_p;
        const field::Complex3 field_p = walk_leg_save(
            field::cplx_scale_real(p_o, field::cplx(1.0f, 0.0f)), vtx, tgt,
            c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, row, d2, frequency_hz, c2p_saves, c2_last_p);
        const field::float3a rx_axis = field::project_to_wedge_plane(
            load3f(rx_pol, row), c2_last);
        const field::Complex cs = transport::complex3_dot_real(field_s, rx_axis);
        const field::Complex cp = transport::complex3_dot_real(field_p, rx_axis);
        const float g_te2 = field::cplx_abs_sqr(cs);
        const float g_tm2 = field::cplx_abs_sqr(cp);

        const float f_eff = (f_te * p_te) * g_te2 + (f_tm * p_tm) * g_tm2;
        const float len1 = l1[row];
        const float len2 = l2[row];
        const float wt = weights[row];
        const float ci = cos_i[row];
        const float den = (len1 * len1) * (len2 * len2);
        const float gain = coef * f_eff * ci * co * wt / den;

        // ---- Fold output cotangents onto gain. ----
        float g_gain = 0.0f;
        if (grad_gain != nullptr)
            g_gain += grad_gain[row];
        if (grad_amplitude != nullptr)
            g_gain += grad_amplitude[row] *
                      (gain > 0.0f ? 0.5f / sqrtf(gain) : 0.0f);
        // grad_length rides need_grad_geometry (L1/L2), rejected this wave.

        const float radiometric = coef * ci * co * wt / den;  // d gain / d f_eff
        const float g_f_eff = g_gain * radiometric;

        if (need_coef && grad_coef != nullptr) {
            const float g_coef = g_gain * (f_eff * ci * co * wt / den);
            atomicAdd(grad_coef, g_coef);
        }

        // f_eff = (f_te*p_te)*g_te2 + (f_tm*p_tm)*g_tm2.
        const float g_f_te = g_f_eff * p_te * g_te2;
        const float g_f_tm = g_f_eff * p_tm * g_tm2;
        const float g_p_te = g_f_eff * f_te * g_te2;
        const float g_p_tm = g_f_eff * f_tm * g_tm2;
        const float g_g_te2 = g_f_eff * f_te * p_te;
        const float g_g_tm2 = g_f_eff * f_tm * p_tm;

        // Table 16-corner scatter (value path only; wi_local frozen, wo_local
        // geometry rides need_grad_geometry).
        if (need_tables && tg.active && slot >= 0) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                atomicAdd(grad_f_te + table_base + tg.idx[k], g_f_te * tg.cw[k]);
                atomicAdd(grad_f_tm + table_base + tg.idx[k], g_f_tm * tg.cw[k]);
            }
        }

        float g_freq = 0.0f;

        // C1 reverse: g_p_te / g_p_tm -> E_in -> chain1 materials + frequency.
        if (need_chain1 || need_frequency) {
            const field::Complex g_c_s = field::cplx(
                2.0f * g_p_te * c_s.re, 2.0f * g_p_te * c_s.im);
            const field::Complex g_c_p = field::cplx(
                2.0f * g_p_tm * c_p.re, 2.0f * g_p_tm * c_p.im);
            field::Complex3 g_e_in = field::c3_zero();
            field::float3a g_dump = field::f3_zero();
            field::adj_cplx_dot_real(e_in, s_i, g_c_s, g_e_in, g_dump);
            field::adj_cplx_dot_real(e_in, p_i, g_c_p, g_e_in, g_dump);
            reverse_leg_materials(
                c1_saves, d1, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
                c1_thickness, row, frequency_hz, g_e_in, need_chain1,
                grad_c1_eps_r, grad_c1_sigma_e, grad_c1_gain, grad_c1_thickness,
                need_frequency, g_freq);
        }

        // C2 reverse: g_g_te2 (field_s) and g_g_tm2 (field_p) -> chain2
        // materials + frequency (two passes accumulate).
        if (need_chain2 || need_frequency) {
            const field::Complex g_cs = field::cplx(
                2.0f * g_g_te2 * cs.re, 2.0f * g_g_te2 * cs.im);
            field::Complex3 g_field_s = field::c3_zero();
            field::float3a g_dump = field::f3_zero();
            field::adj_cplx_dot_real(field_s, rx_axis, g_cs, g_field_s, g_dump);
            reverse_leg_materials(
                c2s_saves, d2, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
                c2_thickness, row, frequency_hz, g_field_s, need_chain2,
                grad_c2_eps_r, grad_c2_sigma_e, grad_c2_gain, grad_c2_thickness,
                need_frequency, g_freq);
            const field::Complex g_cp = field::cplx(
                2.0f * g_g_tm2 * cp.re, 2.0f * g_g_tm2 * cp.im);
            field::Complex3 g_field_p = field::c3_zero();
            field::adj_cplx_dot_real(field_p, rx_axis, g_cp, g_field_p, g_dump);
            reverse_leg_materials(
                c2p_saves, d2, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
                c2_thickness, row, frequency_hz, g_field_p, need_chain2,
                grad_c2_eps_r, grad_c2_sigma_e, grad_c2_gain, grad_c2_thickness,
                need_frequency, g_freq);
        }

        if (need_frequency && grad_frequency != nullptr)
            atomicAdd(grad_frequency, g_freq);
    }
}

// ---------------------------------------------------------------------------
// JVP: full forward-mode dual sweep over every differentiable input.
// ---------------------------------------------------------------------------

// Dual complex3-dot-real of a dual field (value + tangent) with a dual axis.
__device__ __forceinline__ DualC dual_dot_real(
    field::Complex3 v, field::Complex3 dv, ad::DualF3 axis) {
    DualC out;
    out.v = transport::complex3_dot_real(v, axis.v);
    out.d = field::cplx_add(
        transport::complex3_dot_real(dv, axis.v),
        transport::complex3_dot_real(v, axis.d));
    return out;
}

// Dual vertex s/p basis (mirror of sp_basis with tangents; backup frozen).
__device__ __forceinline__ void dual_sp_basis(
    ad::DualF3 n, ad::DualF3 d, field::float3a backup,
    ad::DualF3& s, ad::DualF3& p) {
    const ad::DualF3 s_raw = ad::df3_cross(n, d);
    const float sn = field::safe_length(s_raw.v);
    if (sn < 1.0e-6f) {
        s = ad::df3_const(backup);
    } else {
        const float inv = 1.0f / fmaxf(sn, 1.0e-12f);
        s.v = field::f3_mul(s_raw.v, inv);
        const float dn = field::f3_dot(s_raw.v, s_raw.d) / sn;
        s.d = field::f3_sub(
            field::f3_mul(s_raw.d, inv), field::f3_mul(s_raw.v, dn * inv * inv));
    }
    p = ad::df3_cross(s, d);
}

// Dual specular Jones transport (mirror of walk_leg_save with duals). Returns
// the dual final field via value/d_value refs and the final-leg dual direction.
__device__ __forceinline__ void dual_walk_leg(
    field::Complex3& value,
    field::Complex3& d_value,
    ad::DualF3 start,
    ad::DualF3 end,
    const float* positions,
    const float* tangent_positions,
    const float* normals,
    const float* tangent_normals,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    const float* t_eps_r,
    const float* t_sigma_e,
    const float* t_gain,
    const float* t_thickness,
    float frequency_hz,
    float tangent_frequency,
    int64_t row,
    int depth,
    ad::DualF3& last_dir) {
    const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
    ad::DualF3 previous = start;
    const ad::DualF3 first_target =
        depth > 0 ? load_dual_chain3f(positions, tangent_positions, row, 0)
                  : end;
    ad::DualF3 outgoing = ad::dual_safe_normalize(
        ad::df3_sub(first_target, start), e_z);
    for (int bounce = 0; bounce < depth; ++bounce) {
        const ad::DualF3 hit = load_dual_chain3f(
            positions, tangent_positions, row, bounce);
        const ad::DualF3 incident = ad::dual_safe_normalize(
            ad::df3_sub(hit, previous), outgoing);
        const ad::DualF3 raw_normal = load_dual_chain3f(
            normals, tangent_normals, row, bounce);
        const ad::DualReflectFrame frame = ad::dual_reflect_frame(
            incident, raw_normal);
        const int64_t s = row * kMaxAdDepth + bounce;
        DualC r_te;
        DualC r_tm;
        ad::slab_fresnel_dual(
            frame.cos_theta.v, eps_r[s], sigma_e[s], mu_r[s], gain[s],
            thickness[s], frequency_hz, frame.cos_theta.d,
            t_eps_r != nullptr ? t_eps_r[s] : 0.0f,
            t_sigma_e != nullptr ? t_sigma_e[s] : 0.0f,
            t_gain != nullptr ? t_gain[s] : 0.0f,
            t_thickness != nullptr ? t_thickness[s] : 0.0f,
            tangent_frequency, r_te, r_tm);
        const field::Complex e_s = transport::complex3_dot_real(
            value, frame.s_axis.v);
        const field::Complex e_p = transport::complex3_dot_real(
            value, frame.p_in.v);
        const field::Complex d_e_s = field::cplx_add(
            transport::complex3_dot_real(d_value, frame.s_axis.v),
            transport::complex3_dot_real(value, frame.s_axis.d));
        const field::Complex d_e_p = field::cplx_add(
            transport::complex3_dot_real(d_value, frame.p_in.v),
            transport::complex3_dot_real(value, frame.p_in.d));
        const field::Complex w_te = field::cplx_mul(r_te.v, e_s);
        const field::Complex w_tm = field::cplx_mul(r_tm.v, e_p);
        const field::Complex d_w_te = field::cplx_add(
            field::cplx_mul(r_te.d, e_s), field::cplx_mul(r_te.v, d_e_s));
        const field::Complex d_w_tm = field::cplx_add(
            field::cplx_mul(r_tm.d, e_p), field::cplx_mul(r_tm.v, d_e_p));
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis.v, w_te),
            field::cplx_scale_real(frame.p_out.v, w_tm));
        d_value = field::c3_add(
            field::c3_add(
                field::cplx_scale_real(frame.s_axis.d, w_te),
                field::cplx_scale_real(frame.s_axis.v, d_w_te)),
            field::c3_add(
                field::cplx_scale_real(frame.p_out.d, w_tm),
                field::cplx_scale_real(frame.p_out.v, d_w_tm)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    last_dir = ad::dual_safe_normalize(ad::df3_sub(end, previous), outgoing);
}

__global__ void chain_ensemble_jvp_kernel(
    int64_t count,
    const float* __restrict__ tx_pol,
    const float* __restrict__ rx_pol,
    const float* __restrict__ source,
    const float* __restrict__ vertex,
    const float* __restrict__ target,
    const float* __restrict__ c1_positions,
    const float* __restrict__ c1_normals,
    const float* __restrict__ c1_eps_r,
    const float* __restrict__ c1_sigma_e,
    const float* __restrict__ c1_mu_r,
    const float* __restrict__ c1_gain,
    const float* __restrict__ c1_thickness,
    const int* __restrict__ c1_depth,
    const float* __restrict__ c2_positions,
    const float* __restrict__ c2_normals,
    const float* __restrict__ c2_eps_r,
    const float* __restrict__ c2_sigma_e,
    const float* __restrict__ c2_mu_r,
    const float* __restrict__ c2_gain,
    const float* __restrict__ c2_thickness,
    const int* __restrict__ c2_depth,
    const float* __restrict__ n_o,
    const float* __restrict__ t1r,
    const float* __restrict__ t2r,
    const float* __restrict__ backup_axis,
    const float* __restrict__ wi_local,
    const float* __restrict__ cos_i,
    const float* __restrict__ cos_o,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ l1,
    const float* __restrict__ l2,
    const float* __restrict__ weights,
    const int* __restrict__ material_id,
    const float* __restrict__ fte_flat,
    const float* __restrict__ ftm_flat,
    const int64_t* __restrict__ table_offset,
    const int* __restrict__ table_dims,
    const int* __restrict__ material_slot,
    float coef,
    float frequency_hz,
    const float* __restrict__ t_c1_eps_r,
    const float* __restrict__ t_c1_sigma_e,
    const float* __restrict__ t_c1_gain,
    const float* __restrict__ t_c1_thickness,
    const float* __restrict__ t_c2_eps_r,
    const float* __restrict__ t_c2_sigma_e,
    const float* __restrict__ t_c2_gain,
    const float* __restrict__ t_c2_thickness,
    const float* __restrict__ t_f_te,
    const float* __restrict__ t_f_tm,
    const float* __restrict__ t_c1_positions,
    const float* __restrict__ t_c1_normals,
    const float* __restrict__ t_c2_positions,
    const float* __restrict__ t_c2_normals,
    const float* __restrict__ t_d_i,
    const float* __restrict__ t_d_o,
    const float* __restrict__ t_v_normal,
    const float* __restrict__ t_l1,
    const float* __restrict__ t_l2,
    const float* __restrict__ t_cos_i,
    const float* __restrict__ t_cos_o,
    float tangent_coef,
    float tangent_frequency,
    float* __restrict__ tangent_gain,
    float* __restrict__ tangent_amplitude,
    float* __restrict__ tangent_length) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d1 = c1_depth[row];
        const int d2 = c2_depth[row];
        const ad::DualF3 n = ad::df3_make(
            load3f(n_o, row),
            t_v_normal != nullptr ? load3f(t_v_normal, row) : field::f3_zero());
        const field::float3a backup = load3f(backup_axis, row);
        const ad::DualF3 di = ad::df3_make(
            load3f(d_i, row),
            t_d_i != nullptr ? load3f(t_d_i, row) : field::f3_zero());
        const ad::DualF3 dobj = ad::df3_make(
            load3f(d_o, row),
            t_d_o != nullptr ? load3f(t_d_o, row) : field::f3_zero());
        const ad::DualF3 src = ad::df3_const(load3f(source, row));
        const ad::DualF3 vtx = ad::df3_const(load3f(vertex, row));
        const ad::DualF3 tgt = ad::df3_const(load3f(target, row));

        // C1 transport of the tx field.
        const ad::DualF3 first_target =
            d1 > 0 ? load_dual_chain3f(c1_positions, t_c1_positions, row, 0)
                   : vtx;
        const ad::DualF3 first_leg = ad::dual_safe_normalize(
            ad::df3_sub(first_target, src),
            ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f)));
        const ad::DualF3 tx_axis = ad::dual_transverse_project(
            first_leg, ad::df3_const(load3f(tx_pol, row)));
        field::Complex3 value = field::cplx_scale_real(
            tx_axis.v, field::cplx(1.0f, 0.0f));
        field::Complex3 d_value = field::cplx_scale_real(
            tx_axis.d, field::cplx(1.0f, 0.0f));
        ad::DualF3 c1_last;
        dual_walk_leg(
            value, d_value, src, vtx, c1_positions, t_c1_positions, c1_normals,
            t_c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain, c1_thickness,
            t_c1_eps_r, t_c1_sigma_e, t_c1_gain, t_c1_thickness, frequency_hz,
            tangent_frequency, row, d1, c1_last);

        ad::DualF3 s_i;
        ad::DualF3 p_i;
        dual_sp_basis(n, di, backup, s_i, p_i);
        const DualC c_s = dual_dot_real(value, d_value, s_i);
        const DualC c_p = dual_dot_real(value, d_value, p_i);
        const ad::DualF p_te = {
            field::cplx_abs_sqr(c_s.v),
            2.0f * (c_s.v.re * c_s.d.re + c_s.v.im * c_s.d.im)};
        const ad::DualF p_tm = {
            field::cplx_abs_sqr(c_p.v),
            2.0f * (c_p.v.re * c_p.d.re + c_p.v.im * c_p.d.im)};

        // Table lookup and its wo_local tangent.
        const field::float3a t1 = load3f(t1r, row);
        const field::float3a t2 = load3f(t2r, row);
        const ad::DualF co = {
            cos_o[row], t_cos_o != nullptr ? t_cos_o[row] : 0.0f};
        const float wo_local[3] = {
            field::f3_dot(dobj.v, t1), field::f3_dot(dobj.v, t2), co.v};
        const float d_wo_local[3] = {
            field::f3_dot(dobj.d, t1), field::f3_dot(dobj.d, t2), co.d};
        ad::DualF f_te = {0.0f, 0.0f};
        ad::DualF f_tm = {0.0f, 0.0f};
        const int slot = material_slot[material_id[row]];
        if (slot >= 0) {
            const int64_t base = table_offset[slot];
            const int nti = table_dims[slot * 4 + 0];
            const int npi = table_dims[slot * 4 + 1];
            const int nto = table_dims[slot * 4 + 2];
            const int npo = table_dims[slot * 4 + 3];
            st::TableEvalGrad tg;
            st::eval_te_tm_grad(
                fte_flat + base, ftm_flat + base, nti, npi, nto, npo,
                wi_local + row * 3, wo_local, tg);
            f_te.v = tg.te;
            f_tm.v = tg.tm;
            if (tg.active) {
                float dte = 0.0f;
                float dtm = 0.0f;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    dte += tg.dte_dwo[k] * d_wo_local[k];
                    dtm += tg.dtm_dwo[k] * d_wo_local[k];
                }
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    if (t_f_te != nullptr)
                        dte += tg.cw[k] * t_f_te[base + tg.idx[k]];
                    if (t_f_tm != nullptr)
                        dtm += tg.cw[k] * t_f_tm[base + tg.idx[k]];
                }
                f_te.d = dte;
                f_tm.d = dtm;
            }
        }

        // C2 receiver responses.
        ad::DualF3 s_o;
        ad::DualF3 p_o;
        dual_sp_basis(n, dobj, backup, s_o, p_o);
        field::Complex3 value_s = field::cplx_scale_real(
            s_o.v, field::cplx(1.0f, 0.0f));
        field::Complex3 d_value_s = field::cplx_scale_real(
            s_o.d, field::cplx(1.0f, 0.0f));
        ad::DualF3 c2_last;
        dual_walk_leg(
            value_s, d_value_s, vtx, tgt, c2_positions, t_c2_positions,
            c2_normals, t_c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, t_c2_eps_r, t_c2_sigma_e, t_c2_gain, t_c2_thickness,
            frequency_hz, tangent_frequency, row, d2, c2_last);
        field::Complex3 value_p = field::cplx_scale_real(
            p_o.v, field::cplx(1.0f, 0.0f));
        field::Complex3 d_value_p = field::cplx_scale_real(
            p_o.d, field::cplx(1.0f, 0.0f));
        ad::DualF3 c2_last_p;
        dual_walk_leg(
            value_p, d_value_p, vtx, tgt, c2_positions, t_c2_positions,
            c2_normals, t_c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, t_c2_eps_r, t_c2_sigma_e, t_c2_gain, t_c2_thickness,
            frequency_hz, tangent_frequency, row, d2, c2_last_p);
        const ad::DualF3 rx_axis = ad::dual_transverse_project(
            c2_last, ad::df3_const(load3f(rx_pol, row)));
        const DualC cs = dual_dot_real(value_s, d_value_s, rx_axis);
        const DualC cp = dual_dot_real(value_p, d_value_p, rx_axis);
        const ad::DualF g_te2 = {
            field::cplx_abs_sqr(cs.v),
            2.0f * (cs.v.re * cs.d.re + cs.v.im * cs.d.im)};
        const ad::DualF g_tm2 = {
            field::cplx_abs_sqr(cp.v),
            2.0f * (cp.v.re * cp.d.re + cp.v.im * cp.d.im)};

        // f_eff = (f_te*p_te)*g_te2 + (f_tm*p_tm)*g_tm2 (dual products).
        const float fte_pte = f_te.v * p_te.v;
        const float d_fte_pte = f_te.d * p_te.v + f_te.v * p_te.d;
        const float ftm_ptm = f_tm.v * p_tm.v;
        const float d_ftm_ptm = f_tm.d * p_tm.v + f_tm.v * p_tm.d;
        const float term_te = fte_pte * g_te2.v;
        const float d_term_te = d_fte_pte * g_te2.v + fte_pte * g_te2.d;
        const float term_tm = ftm_ptm * g_tm2.v;
        const float d_term_tm = d_ftm_ptm * g_tm2.v + ftm_ptm * g_tm2.d;
        const float f_eff = term_te + term_tm;
        const float d_f_eff = d_term_te + d_term_tm;

        const ad::DualF len1 = {
            l1[row], t_l1 != nullptr ? t_l1[row] : 0.0f};
        const ad::DualF len2 = {
            l2[row], t_l2 != nullptr ? t_l2[row] : 0.0f};
        const ad::DualF ci = {
            cos_i[row], t_cos_i != nullptr ? t_cos_i[row] : 0.0f};
        const float wt = weights[row];  // frozen (no tangent)

        // den = (l1^2)(l2^2); num = coef * f_eff * cos_i * cos_o * weights.
        const float l1sq = len1.v * len1.v;
        const float d_l1sq = 2.0f * len1.v * len1.d;
        const float l2sq = len2.v * len2.v;
        const float d_l2sq = 2.0f * len2.v * len2.d;
        const float den = l1sq * l2sq;
        const float d_den = d_l1sq * l2sq + l1sq * d_l2sq;

        // num = coef * f_eff * ci * co * wt.
        const float num = coef * f_eff * ci.v * co.v * wt;
        const float d_num =
            tangent_coef * f_eff * ci.v * co.v * wt +
            coef * d_f_eff * ci.v * co.v * wt +
            coef * f_eff * ci.d * co.v * wt +
            coef * f_eff * ci.v * co.d * wt;
        const float gain = num / den;
        const float d_gain = (d_num * den - num * d_den) / (den * den);

        tangent_gain[row] = d_gain;
        tangent_amplitude[row] =
            gain > 0.0f ? 0.5f * d_gain / sqrtf(gain) : 0.0f;
        tangent_length[row] = len1.d + len2.d;
    }
}

// ---------------------------------------------------------------------------
// Host bridges.
// ---------------------------------------------------------------------------

// Validate the shared primal tensors and return the row count.
int64_t check_chain_primal(const at::Tensor& tx_pol) {
    using channel_native::check_vec3_table;
    check_vec3_table(tx_pol, "tx_pol");
    return tx_pol.size(0);
}

}  // namespace

pybind11::dict cn_scattering_chain_ensemble_eval_backward(
    at::Tensor tx_pol,
    at::Tensor rx_pol,
    at::Tensor source,
    at::Tensor vertex,
    at::Tensor target,
    at::Tensor c1_positions,
    at::Tensor c1_normals,
    at::Tensor c1_eps_r,
    at::Tensor c1_sigma_e,
    at::Tensor c1_mu_r,
    at::Tensor c1_gain,
    at::Tensor c1_thickness,
    at::Tensor c1_depth,
    at::Tensor c2_positions,
    at::Tensor c2_normals,
    at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e,
    at::Tensor c2_mu_r,
    at::Tensor c2_gain,
    at::Tensor c2_thickness,
    at::Tensor c2_depth,
    at::Tensor n_o,
    at::Tensor t1r,
    at::Tensor t2r,
    at::Tensor backup_axis,
    at::Tensor wi_local,
    at::Tensor cos_i,
    at::Tensor cos_o,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor l1,
    at::Tensor l2,
    at::Tensor weights,
    at::Tensor material_id,
    at::Tensor fte_flat,
    at::Tensor ftm_flat,
    at::Tensor table_offset,
    at::Tensor table_dims,
    at::Tensor material_slot,
    double coef,
    double threshold,
    double frequency_hz,
    pybind11::object grad_gain,
    pybind11::object grad_amplitude,
    pybind11::object grad_length,
    bool need_grad_chain1,
    bool need_grad_chain2,
    bool need_grad_tables,
    bool need_grad_geometry,
    bool need_grad_coef,
    bool need_grad_frequency) {
    const int64_t count = check_chain_primal(tx_pol);
    TORCH_CHECK(
        !need_grad_geometry,
        "scattering_chain_ensemble_eval_backward: reverse-mode chain geometry "
        "(need_grad_geometry) is not implemented in this wave; use the _jvp "
        "companion for forward-mode geometry gradients.");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t table_size = fte_flat.size(0);

    at::Tensor gg_storage;
    at::Tensor ga_storage;
    const at::Tensor* gg = optional_grad(
        std::move(grad_gain), gg_storage, "grad_gain", at::kFloat, {count},
        tx_pol);
    const at::Tensor* ga = optional_grad(
        std::move(grad_amplitude), ga_storage, "grad_amplitude", at::kFloat,
        {count}, tx_pol);
    // grad_length only feeds L1/L2 (geometry, rejected this wave); accepted for
    // signature parity and ignored.
    (void)grad_length;

    auto leg_grad = [&](bool needed) {
        return needed ? zero_filled({count, kMaxAdDepth}, tx_pol.options())
                      : at::Tensor();
    };
    at::Tensor g_c1_eps = leg_grad(need_grad_chain1);
    at::Tensor g_c1_sigma = leg_grad(need_grad_chain1);
    at::Tensor g_c1_gain = leg_grad(need_grad_chain1);
    at::Tensor g_c1_thick = leg_grad(need_grad_chain1);
    at::Tensor g_c2_eps = leg_grad(need_grad_chain2);
    at::Tensor g_c2_sigma = leg_grad(need_grad_chain2);
    at::Tensor g_c2_gain = leg_grad(need_grad_chain2);
    at::Tensor g_c2_thick = leg_grad(need_grad_chain2);
    at::Tensor g_f_te = need_grad_tables
                            ? zero_filled({table_size}, tx_pol.options())
                            : at::Tensor();
    at::Tensor g_f_tm = need_grad_tables
                            ? zero_filled({table_size}, tx_pol.options())
                            : at::Tensor();
    at::Tensor g_coef = need_grad_coef
                            ? zero_filled({1}, tx_pol.options())
                            : at::Tensor();
    at::Tensor g_freq = need_grad_frequency
                            ? zero_filled({1}, tx_pol.options())
                            : at::Tensor();

    const bool any_out = need_grad_chain1 || need_grad_chain2 ||
                         need_grad_tables || need_grad_coef ||
                         need_grad_frequency;
    const bool any_in = gg != nullptr || ga != nullptr;
    if (count > 0 && any_out && any_in) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tx_pol.get_device()).stream();
        chain_ensemble_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            tx_pol.data_ptr<float>(), rx_pol.data_ptr<float>(),
            source.data_ptr<float>(), vertex.data_ptr<float>(),
            target.data_ptr<float>(),
            c1_positions.data_ptr<float>(), c1_normals.data_ptr<float>(),
            c1_eps_r.data_ptr<float>(), c1_sigma_e.data_ptr<float>(),
            c1_mu_r.data_ptr<float>(), c1_gain.data_ptr<float>(),
            c1_thickness.data_ptr<float>(), c1_depth.data_ptr<int>(),
            c2_positions.data_ptr<float>(), c2_normals.data_ptr<float>(),
            c2_eps_r.data_ptr<float>(), c2_sigma_e.data_ptr<float>(),
            c2_mu_r.data_ptr<float>(), c2_gain.data_ptr<float>(),
            c2_thickness.data_ptr<float>(), c2_depth.data_ptr<int>(),
            n_o.data_ptr<float>(), t1r.data_ptr<float>(),
            t2r.data_ptr<float>(), backup_axis.data_ptr<float>(),
            wi_local.data_ptr<float>(), cos_i.data_ptr<float>(),
            cos_o.data_ptr<float>(), d_i.data_ptr<float>(),
            d_o.data_ptr<float>(), l1.data_ptr<float>(), l2.data_ptr<float>(),
            weights.data_ptr<float>(), material_id.data_ptr<int>(),
            fte_flat.data_ptr<float>(), ftm_flat.data_ptr<float>(),
            table_offset.data_ptr<int64_t>(), table_dims.data_ptr<int>(),
            material_slot.data_ptr<int>(),
            static_cast<float>(coef), static_cast<float>(frequency_hz),
            grad_ptr<float>(gg), grad_ptr<float>(ga),
            need_grad_chain1 ? g_c1_eps.data_ptr<float>() : nullptr,
            need_grad_chain1 ? g_c1_sigma.data_ptr<float>() : nullptr,
            need_grad_chain1 ? g_c1_gain.data_ptr<float>() : nullptr,
            need_grad_chain1 ? g_c1_thick.data_ptr<float>() : nullptr,
            need_grad_chain2 ? g_c2_eps.data_ptr<float>() : nullptr,
            need_grad_chain2 ? g_c2_sigma.data_ptr<float>() : nullptr,
            need_grad_chain2 ? g_c2_gain.data_ptr<float>() : nullptr,
            need_grad_chain2 ? g_c2_thick.data_ptr<float>() : nullptr,
            need_grad_tables ? g_f_te.data_ptr<float>() : nullptr,
            need_grad_tables ? g_f_tm.data_ptr<float>() : nullptr,
            need_grad_coef ? g_coef.data_ptr<float>() : nullptr,
            need_grad_frequency ? g_freq.data_ptr<float>() : nullptr,
            need_grad_chain1, need_grad_chain2, need_grad_tables,
            need_grad_coef, need_grad_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto as_obj = [](const at::Tensor& t) {
        return t.defined() ? pybind11::cast(t)
                           : pybind11::object(pybind11::none());
    };
    pybind11::dict out;
    out["grad_c1_eps_r"] = as_obj(g_c1_eps);
    out["grad_c1_sigma_e"] = as_obj(g_c1_sigma);
    out["grad_c1_gain"] = as_obj(g_c1_gain);
    out["grad_c1_thickness"] = as_obj(g_c1_thick);
    out["grad_c2_eps_r"] = as_obj(g_c2_eps);
    out["grad_c2_sigma_e"] = as_obj(g_c2_sigma);
    out["grad_c2_gain"] = as_obj(g_c2_gain);
    out["grad_c2_thickness"] = as_obj(g_c2_thick);
    out["grad_f_te"] = as_obj(g_f_te);
    out["grad_f_tm"] = as_obj(g_f_tm);
    out["grad_coef"] = as_obj(g_coef);
    out["grad_frequency"] = as_obj(g_freq);
    return out;
}

pybind11::dict cn_scattering_chain_ensemble_eval_jvp(
    at::Tensor tx_pol,
    at::Tensor rx_pol,
    at::Tensor source,
    at::Tensor vertex,
    at::Tensor target,
    at::Tensor c1_positions,
    at::Tensor c1_normals,
    at::Tensor c1_eps_r,
    at::Tensor c1_sigma_e,
    at::Tensor c1_mu_r,
    at::Tensor c1_gain,
    at::Tensor c1_thickness,
    at::Tensor c1_depth,
    at::Tensor c2_positions,
    at::Tensor c2_normals,
    at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e,
    at::Tensor c2_mu_r,
    at::Tensor c2_gain,
    at::Tensor c2_thickness,
    at::Tensor c2_depth,
    at::Tensor n_o,
    at::Tensor t1r,
    at::Tensor t2r,
    at::Tensor backup_axis,
    at::Tensor wi_local,
    at::Tensor cos_i,
    at::Tensor cos_o,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor l1,
    at::Tensor l2,
    at::Tensor weights,
    at::Tensor material_id,
    at::Tensor fte_flat,
    at::Tensor ftm_flat,
    at::Tensor table_offset,
    at::Tensor table_dims,
    at::Tensor material_slot,
    double coef,
    double threshold,
    double frequency_hz,
    pybind11::object tangent_c1_eps_r,
    pybind11::object tangent_c1_sigma_e,
    pybind11::object tangent_c1_gain,
    pybind11::object tangent_c1_thickness,
    pybind11::object tangent_c2_eps_r,
    pybind11::object tangent_c2_sigma_e,
    pybind11::object tangent_c2_gain,
    pybind11::object tangent_c2_thickness,
    pybind11::object tangent_f_te_flat,
    pybind11::object tangent_f_tm_flat,
    pybind11::object tangent_c1_positions,
    pybind11::object tangent_c1_normals,
    pybind11::object tangent_c2_positions,
    pybind11::object tangent_c2_normals,
    pybind11::object tangent_d_i,
    pybind11::object tangent_d_o,
    pybind11::object tangent_v_normal,
    pybind11::object tangent_l1,
    pybind11::object tangent_l2,
    pybind11::object tangent_cos_i,
    pybind11::object tangent_cos_o,
    double tangent_coef,
    double tangent_frequency) {
    const int64_t count = check_chain_primal(tx_pol);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t table_size = fte_flat.size(0);

    at::Tensor s_c1_eps, s_c1_sigma, s_c1_gain, s_c1_thick;
    at::Tensor s_c2_eps, s_c2_sigma, s_c2_gain, s_c2_thick;
    at::Tensor s_fte, s_ftm;
    at::Tensor s_c1_pos, s_c1_nrm, s_c2_pos, s_c2_nrm;
    at::Tensor s_di, s_do, s_vn, s_l1, s_l2, s_ci, s_co;
    const at::Tensor* leg = nullptr;
    auto leg_t = [&](pybind11::object o, at::Tensor& st, const char* name) {
        return optional_grad(std::move(o), st, name, at::kFloat,
                             {count, kMaxAdDepth}, tx_pol);
    };
    auto vec_t = [&](pybind11::object o, at::Tensor& st, const char* name) {
        return optional_grad(std::move(o), st, name, at::kFloat, {count, 3},
                             tx_pol);
    };
    auto legvec_t = [&](pybind11::object o, at::Tensor& st, const char* name) {
        return optional_grad(std::move(o), st, name, at::kFloat,
                             {count, kMaxAdDepth, 3}, tx_pol);
    };
    auto scal_t = [&](pybind11::object o, at::Tensor& st, const char* name) {
        return optional_grad(std::move(o), st, name, at::kFloat, {count},
                             tx_pol);
    };
    const at::Tensor* p_c1_eps = leg_t(std::move(tangent_c1_eps_r), s_c1_eps, "tangent_c1_eps_r");
    const at::Tensor* p_c1_sigma = leg_t(std::move(tangent_c1_sigma_e), s_c1_sigma, "tangent_c1_sigma_e");
    const at::Tensor* p_c1_gain = leg_t(std::move(tangent_c1_gain), s_c1_gain, "tangent_c1_gain");
    const at::Tensor* p_c1_thick = leg_t(std::move(tangent_c1_thickness), s_c1_thick, "tangent_c1_thickness");
    const at::Tensor* p_c2_eps = leg_t(std::move(tangent_c2_eps_r), s_c2_eps, "tangent_c2_eps_r");
    const at::Tensor* p_c2_sigma = leg_t(std::move(tangent_c2_sigma_e), s_c2_sigma, "tangent_c2_sigma_e");
    const at::Tensor* p_c2_gain = leg_t(std::move(tangent_c2_gain), s_c2_gain, "tangent_c2_gain");
    const at::Tensor* p_c2_thick = leg_t(std::move(tangent_c2_thickness), s_c2_thick, "tangent_c2_thickness");
    const at::Tensor* p_fte = optional_grad(
        std::move(tangent_f_te_flat), s_fte, "tangent_f_te_flat", at::kFloat,
        {table_size}, tx_pol);
    const at::Tensor* p_ftm = optional_grad(
        std::move(tangent_f_tm_flat), s_ftm, "tangent_f_tm_flat", at::kFloat,
        {table_size}, tx_pol);
    const at::Tensor* p_c1_pos = legvec_t(std::move(tangent_c1_positions), s_c1_pos, "tangent_c1_positions");
    const at::Tensor* p_c1_nrm = legvec_t(std::move(tangent_c1_normals), s_c1_nrm, "tangent_c1_normals");
    const at::Tensor* p_c2_pos = legvec_t(std::move(tangent_c2_positions), s_c2_pos, "tangent_c2_positions");
    const at::Tensor* p_c2_nrm = legvec_t(std::move(tangent_c2_normals), s_c2_nrm, "tangent_c2_normals");
    const at::Tensor* p_di = vec_t(std::move(tangent_d_i), s_di, "tangent_d_i");
    const at::Tensor* p_do = vec_t(std::move(tangent_d_o), s_do, "tangent_d_o");
    const at::Tensor* p_vn = vec_t(std::move(tangent_v_normal), s_vn, "tangent_v_normal");
    const at::Tensor* p_l1 = scal_t(std::move(tangent_l1), s_l1, "tangent_l1");
    const at::Tensor* p_l2 = scal_t(std::move(tangent_l2), s_l2, "tangent_l2");
    const at::Tensor* p_ci = scal_t(std::move(tangent_cos_i), s_ci, "tangent_cos_i");
    const at::Tensor* p_co = scal_t(std::move(tangent_cos_o), s_co, "tangent_cos_o");
    (void)leg;

    auto t_gain = at::empty({count}, tx_pol.options());
    auto t_amp = at::empty({count}, tx_pol.options());
    auto t_len = at::empty({count}, tx_pol.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tx_pol.get_device()).stream();
        chain_ensemble_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            tx_pol.data_ptr<float>(), rx_pol.data_ptr<float>(),
            source.data_ptr<float>(), vertex.data_ptr<float>(),
            target.data_ptr<float>(),
            c1_positions.data_ptr<float>(), c1_normals.data_ptr<float>(),
            c1_eps_r.data_ptr<float>(), c1_sigma_e.data_ptr<float>(),
            c1_mu_r.data_ptr<float>(), c1_gain.data_ptr<float>(),
            c1_thickness.data_ptr<float>(), c1_depth.data_ptr<int>(),
            c2_positions.data_ptr<float>(), c2_normals.data_ptr<float>(),
            c2_eps_r.data_ptr<float>(), c2_sigma_e.data_ptr<float>(),
            c2_mu_r.data_ptr<float>(), c2_gain.data_ptr<float>(),
            c2_thickness.data_ptr<float>(), c2_depth.data_ptr<int>(),
            n_o.data_ptr<float>(), t1r.data_ptr<float>(),
            t2r.data_ptr<float>(), backup_axis.data_ptr<float>(),
            wi_local.data_ptr<float>(), cos_i.data_ptr<float>(),
            cos_o.data_ptr<float>(), d_i.data_ptr<float>(),
            d_o.data_ptr<float>(), l1.data_ptr<float>(), l2.data_ptr<float>(),
            weights.data_ptr<float>(), material_id.data_ptr<int>(),
            fte_flat.data_ptr<float>(), ftm_flat.data_ptr<float>(),
            table_offset.data_ptr<int64_t>(), table_dims.data_ptr<int>(),
            material_slot.data_ptr<int>(),
            static_cast<float>(coef), static_cast<float>(frequency_hz),
            grad_ptr<float>(p_c1_eps), grad_ptr<float>(p_c1_sigma),
            grad_ptr<float>(p_c1_gain), grad_ptr<float>(p_c1_thick),
            grad_ptr<float>(p_c2_eps), grad_ptr<float>(p_c2_sigma),
            grad_ptr<float>(p_c2_gain), grad_ptr<float>(p_c2_thick),
            grad_ptr<float>(p_fte), grad_ptr<float>(p_ftm),
            grad_ptr<float>(p_c1_pos), grad_ptr<float>(p_c1_nrm),
            grad_ptr<float>(p_c2_pos), grad_ptr<float>(p_c2_nrm),
            grad_ptr<float>(p_di), grad_ptr<float>(p_do),
            grad_ptr<float>(p_vn), grad_ptr<float>(p_l1),
            grad_ptr<float>(p_l2), grad_ptr<float>(p_ci),
            grad_ptr<float>(p_co),
            static_cast<float>(tangent_coef),
            static_cast<float>(tangent_frequency),
            t_gain.data_ptr<float>(), t_amp.data_ptr<float>(),
            t_len.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_gain"] = t_gain;
    out["tangent_amplitude"] = t_amp;
    out["tangent_length"] = t_len;
    return out;
}
