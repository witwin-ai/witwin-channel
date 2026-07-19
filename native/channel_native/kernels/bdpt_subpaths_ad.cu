#include "field_transport_ad_common.cuh"

#include "../em/layer_stack.cuh"

// ADR-022 6.1 / 6.2: backward + jvp companions for the BDPT light-subpath
// advance ops.
//
// A subpath advance multiplies the carried Complex3 Jones field by a per-hit
// operator O and scales the (real-amplitude proxy) throughput. Under the
// fixed-topology / fixed-winner contract the hit geometry (positions, normals,
// incident direction, frame, event partition, validity) is frozen; the
// differentiable inputs are the EM response parameters (single-slab eps_r /
// sigma_e / gain / thickness for reflection, CSR layer thickness / eps_r /
// sigma_e for transmission), the carrier frequency, and the upstream subpath
// field / throughput. grad_field_in = O^H grad_field_out; material / layer
// partials use the SAME lockstep duals as the field-transport companions
// (slab_fresnel_dual for reflection, stack_rt_dual for transmission). Per-hit
// shared parameter grads accumulate with atomicAdd; the upstream subpath field
// / throughput grads are per-row direct stores.

namespace {

namespace ad = channel_native::field_transport_ad;

constexpr float kSubpathEps = 1.0e-9f;
constexpr float kSubpathEpsilon0 = 8.8541878128e-12f;

// ---------------------------------------------------------------------------
// Reflection amplitude proxy (effective_power_reflectance) and its dual. This
// is a distinct arithmetic from the slab Jones response: it is the
// single-interface power reflectance driving the real throughput proxy. Only
// eps_r / sigma_e / frequency are live (geometry and mu are frozen); the dual
// mirrors the primal SubpathComplex operations for lockstep.
// ---------------------------------------------------------------------------

struct SubC {
    float r;
    float i;
};

__device__ __forceinline__ SubC subc(float r, float i) { return {r, i}; }
__device__ __forceinline__ SubC subc_add(SubC a, SubC b) { return {a.r + b.r, a.i + b.i}; }
__device__ __forceinline__ SubC subc_sub(SubC a, SubC b) { return {a.r - b.r, a.i - b.i}; }
__device__ __forceinline__ SubC subc_mul(SubC a, SubC b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
__device__ __forceinline__ SubC subc_scale(SubC a, float s) { return {a.r * s, a.i * s}; }
__device__ __forceinline__ SubC subc_div(SubC a, SubC b) {
    const float denom = fmaxf(b.r * b.r + b.i * b.i, kSubpathEps);
    return {(a.r * b.r + a.i * b.i) / denom, (a.i * b.r - a.r * b.i) / denom};
}
__device__ __forceinline__ SubC subc_sqrt(SubC z) {
    const float magnitude = hypotf(z.r, z.i);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + z.r)));
    const float imag_sign = z.i < 0.0f ? -1.0f : 1.0f;
    const float imag = imag_sign * sqrtf(fmaxf(0.0f, 0.5f * (magnitude - z.r)));
    return {real, imag};
}

struct DualSC {
    SubC v;
    SubC d;
};

__device__ __forceinline__ DualSC dsc_make(float re, float im, float dre, float dim) {
    return {{re, im}, {dre, dim}};
}
__device__ __forceinline__ DualSC dsc_const(SubC value) { return {value, {0.0f, 0.0f}}; }
__device__ __forceinline__ DualSC dsc_add(DualSC a, DualSC b) {
    return {subc_add(a.v, b.v), subc_add(a.d, b.d)};
}
__device__ __forceinline__ DualSC dsc_sub(DualSC a, DualSC b) {
    return {subc_sub(a.v, b.v), subc_sub(a.d, b.d)};
}
__device__ __forceinline__ DualSC dsc_mul(DualSC a, DualSC b) {
    return {subc_mul(a.v, b.v), subc_add(subc_mul(a.d, b.v), subc_mul(a.v, b.d))};
}
__device__ __forceinline__ DualSC dsc_scale(DualSC a, float s) {
    return {subc_scale(a.v, s), subc_scale(a.d, s)};
}
// Dual of subc_div (regularized denom; clamped branch keeps constant denom).
__device__ __forceinline__ DualSC dsc_div(DualSC a, DualSC b) {
    const float mag2 = b.v.r * b.v.r + b.v.i * b.v.i;
    const float denom = fmaxf(mag2, kSubpathEps);
    DualSC out;
    out.v = subc_div(a.v, b.v);
    const float d_denom = mag2 > kSubpathEps ? 2.0f * (b.v.r * b.d.r + b.v.i * b.d.i) : 0.0f;
    // d(a*conj(b)) = a.d*conj(b) + a.v*conj(b.d)
    const SubC conj_b = {b.v.r, -b.v.i};
    const SubC conj_bd = {b.d.r, -b.d.i};
    const SubC d_num = subc_add(subc_mul(a.d, conj_b), subc_mul(a.v, conj_bd));
    out.d = {
        (d_num.r - out.v.r * d_denom) / denom,
        (d_num.i - out.v.i * d_denom) / denom};
    return out;
}
// Dual of subc_sqrt: dw = dz/(2w) (withheld at the branch point).
__device__ __forceinline__ DualSC dsc_sqrt(DualSC a) {
    DualSC out;
    out.v = subc_sqrt(a.v);
    const float w2 = out.v.r * out.v.r + out.v.i * out.v.i;
    if (w2 <= kSubpathEps) {
        out.d = {0.0f, 0.0f};
        return out;
    }
    const SubC two_w = {2.0f * out.v.r, 2.0f * out.v.i};
    const float denom = two_w.r * two_w.r + two_w.i * two_w.i;
    // dz / (2w) = dz * conj(2w) / |2w|^2.
    out.d = {
        (a.d.r * two_w.r + a.d.i * two_w.i) / denom,
        (a.d.i * two_w.r - a.d.r * two_w.i) / denom};
    return out;
}

// Frozen geometry of the reflection amplitude proxy: cos_theta and the
// polarization power weights e_s^2 / e_p^2.
struct ReflectanceGeometry {
    float cos_theta;
    float sin2;
    float e_s2;
    float e_p2;
};

__device__ ReflectanceGeometry reflectance_geometry(
    const float* incident_dir, const float* normal_in) {
    ReflectanceGeometry geo;
    float ix = incident_dir[0], iy = incident_dir[1], iz = incident_dir[2];
    const float inv_ilen = rsqrtf(fmaxf(ix * ix + iy * iy + iz * iz, 1.0e-20f));
    ix *= inv_ilen; iy *= inv_ilen; iz *= inv_ilen;
    float nx = normal_in[0], ny = normal_in[1], nz = normal_in[2];
    const float inv_nlen = rsqrtf(fmaxf(nx * nx + ny * ny + nz * nz, 1.0e-20f));
    nx *= inv_nlen; ny *= inv_nlen; nz *= inv_nlen;
    float dot_in = ix * nx + iy * ny + iz * nz;
    if (dot_in > 0.0f) { nx = -nx; ny = -ny; nz = -nz; dot_in = -dot_in; }
    geo.cos_theta = fminf(fmaxf(-dot_in, kSubpathEps), 1.0f);
    geo.sin2 = fmaxf(0.0f, 1.0f - geo.cos_theta * geo.cos_theta);
    float sx = ny * iz - nz * iy;
    float sy = nz * ix - nx * iz;
    float sz = nx * iy - ny * ix;
    const float s_len = sqrtf(fmaxf(sx * sx + sy * sy + sz * sz, 0.0f));
    if (s_len <= kSubpathEps) { geo.e_s2 = 1.0f; geo.e_p2 = 0.0f; return geo; }
    sx /= s_len; sy /= s_len; sz /= s_len;
    const float px = sy * iz - sz * iy;
    const float py = sz * ix - sx * iz;
    const float pz = sx * iy - sy * ix;
    float tx = 1.0f - ix * ix, ty = -ix * iy, tz = -ix * iz;
    const float t_len = sqrtf(fmaxf(tx * tx + ty * ty + tz * tz, 0.0f));
    float e_s, e_p;
    if (t_len <= kSubpathEps) { e_s = 1.0f; e_p = 0.0f; } else {
        e_s = (tx * sx + ty * sy + tz * sz) / t_len;
        e_p = (tx * px + ty * py + tz * pz) / t_len;
    }
    geo.e_s2 = e_s * e_s;
    geo.e_p2 = e_p * e_p;
    return geo;
}

// R_eff value + directional derivative for one (d_eps, d_sigma, d_freq) seed.
__device__ float effective_reflectance_dual(
    const ReflectanceGeometry& geo,
    float eps_r,
    float sigma_e,
    float mu_r,
    float frequency_hz,
    float d_eps,
    float d_sigma,
    float d_freq,
    float& d_reflectance) {
    const float omega_raw = 2.0f * field::UTD_PI * frequency_hz;
    const float omega = fmaxf(omega_raw, kSubpathEps);
    const float d_omega = omega_raw > kSubpathEps ? 2.0f * field::UTD_PI * d_freq : 0.0f;
    const float eps_clamped = fmaxf(eps_r, kSubpathEps);
    const float d_eps_clamped = eps_r > kSubpathEps ? d_eps : 0.0f;
    const float sigma_clamped = fmaxf(sigma_e, 0.0f);
    const float d_sigma_clamped = sigma_e >= 0.0f ? d_sigma : 0.0f;
    const float eta_im = -sigma_clamped / (omega * kSubpathEpsilon0);
    const float d_eta_im =
        (-d_sigma_clamped / omega + sigma_clamped * d_omega / (omega * omega)) /
        kSubpathEpsilon0;
    const DualSC eta = dsc_make(eps_clamped, eta_im, d_eps_clamped, d_eta_im);
    const float mu_value = fmaxf(mu_r, kSubpathEps);
    const DualSC root = dsc_sqrt(dsc_sub(dsc_scale(eta, mu_value), dsc_const({geo.sin2, 0.0f})));
    const DualSC mu_cos = dsc_const({mu_value * geo.cos_theta, 0.0f});
    const DualSC eta_cos = dsc_scale(eta, geo.cos_theta);
    const DualSC r_te = dsc_div(dsc_sub(mu_cos, root), dsc_add(mu_cos, root));
    const DualSC r_tm = dsc_div(dsc_sub(eta_cos, root), dsc_add(eta_cos, root));
    const float te2 = r_te.v.r * r_te.v.r + r_te.v.i * r_te.v.i;
    const float tm2 = r_tm.v.r * r_tm.v.r + r_tm.v.i * r_tm.v.i;
    const float d_te2 = 2.0f * (r_te.v.r * r_te.d.r + r_te.v.i * r_te.d.i);
    const float d_tm2 = 2.0f * (r_tm.v.r * r_tm.d.r + r_tm.v.i * r_tm.d.i);
    d_reflectance = d_te2 * geo.e_s2 + d_tm2 * geo.e_p2;
    return te2 * geo.e_s2 + tm2 * geo.e_p2;
}

// ---------------------------------------------------------------------------
// Reflected light subpath advance companions.
// ---------------------------------------------------------------------------

__device__ __forceinline__ field::Complex3 load_field3(
    const float* re, const float* im, int64_t index) {
    const int64_t base = index * 3;
    return {
        field::cplx(re[base], im[base]),
        field::cplx(re[base + 1], im[base + 1]),
        field::cplx(re[base + 2], im[base + 2])};
}

__device__ __forceinline__ field::Complex opt_c(
    const float* re, const float* im, int64_t index) {
    return field::cplx(re != nullptr ? re[index] : 0.0f, im != nullptr ? im[index] : 0.0f);
}

__global__ void reflected_subpath_backward_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const bool* light_valid,
    const float* light_field_real,
    const float* light_field_imag,
    const float* hit_t,
    const float* hit_n,
    const int* hit_global_prim_id,
    const float* material_gain,
    const bool* material_valid,
    const float* material_eps_r,
    const float* material_sigma_e,
    const float* material_mu_r,
    const float* material_thickness,
    float frequency_hz,
    int64_t material_count,
    const float* grad_field_real,
    const float* grad_field_imag,
    const float* grad_throughput_real,
    const float* grad_throughput_imag,
    float* grad_eps_r,
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    float* grad_light_field_real,
    float* grad_light_field_imag,
    float* grad_light_throughput_real,
    float* grad_light_throughput_imag,
    float* grad_frequency,
    bool need_grad_material,
    bool need_grad_field_in,
    bool need_grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int prim = hit_global_prim_id[index];
        const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < material_count;
        const bool material_ok = prim_in_range && material_valid[prim];
        const bool is_valid =
            light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
        if (!is_valid) {
            if (need_grad_field_in) {
                const int64_t base = index * 3;
                grad_light_field_real[base] = 0.0f;
                grad_light_field_real[base + 1] = 0.0f;
                grad_light_field_real[base + 2] = 0.0f;
                grad_light_field_imag[base] = 0.0f;
                grad_light_field_imag[base + 1] = 0.0f;
                grad_light_field_imag[base + 2] = 0.0f;
                grad_light_throughput_real[index] = 0.0f;
                grad_light_throughput_imag[index] = 0.0f;
            }
            continue;
        }
        const float eps_r = material_eps_r[prim];
        const float sigma_e = material_sigma_e[prim];
        const float mu_r = material_mu_r[prim];
        const float gain = material_gain[prim];
        const float thickness = material_thickness[prim];

        const field::float3a incident = load3f(light_direction, index);
        const field::float3a normal = load3f(hit_n, index);
        const transport::ReflectFrame frame = transport::reflect_frame(incident, normal);
        field::Complex r_te, r_tm;
        transport::slab_fresnel(
            frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, r_te, r_tm);
        const field::Complex3 incoming = load_field3(light_field_real, light_field_imag, index);
        const field::Complex e_s = transport::complex3_dot_real(incoming, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(incoming, frame.p_in);

        // Field cotangent -> upstream field, r_te/r_tm cotangents.
        const field::Complex3 g_updated = load_field3(grad_field_real, grad_field_imag, index);
        field::float3a g_axis_dump = field::f3_zero();
        field::Complex g_w_te = field::cplx_zero();
        field::Complex g_w_tm = field::cplx_zero();
        field::adj_cplx_scale_real(
            frame.s_axis, field::cplx_mul(r_te, e_s), g_updated, g_axis_dump, g_w_te);
        field::adj_cplx_scale_real(
            frame.p_out, field::cplx_mul(r_tm, e_p), g_updated, g_axis_dump, g_w_tm);
        field::Complex g_r_te = field::cplx_zero();
        field::Complex g_r_tm = field::cplx_zero();
        field::Complex g_e_s = field::cplx_zero();
        field::Complex g_e_p = field::cplx_zero();
        field::adj_cplx_mul(r_te, e_s, g_w_te, g_r_te, g_e_s);
        field::adj_cplx_mul(r_tm, e_p, g_w_tm, g_r_tm, g_e_p);
        field::Complex3 g_incoming = field::c3_zero();
        field::adj_cplx_dot_real(incoming, frame.s_axis, g_e_s, g_incoming, g_axis_dump);
        field::adj_cplx_dot_real(incoming, frame.p_in, g_e_p, g_incoming, g_axis_dump);

        // Throughput amplitude proxy.
        const ReflectanceGeometry geo = reflectance_geometry(
            light_direction + index * 3, hit_n + index * 3);
        float d_reflectance_eps = 0.0f, d_reflectance_sigma = 0.0f, d_reflectance_freq = 0.0f;
        const float reflectance = effective_reflectance_dual(
            geo, eps_r, sigma_e, mu_r, frequency_hz, 1.0f, 0.0f, 0.0f, d_reflectance_eps);
        effective_reflectance_dual(
            geo, eps_r, sigma_e, mu_r, frequency_hz, 0.0f, 1.0f, 0.0f, d_reflectance_sigma);
        effective_reflectance_dual(
            geo, eps_r, sigma_e, mu_r, frequency_hz, 0.0f, 0.0f, 1.0f, d_reflectance_freq);
        const float gain_clamped = fmaxf(gain, 0.0f);
        const float prod = gain_clamped * reflectance;
        const float amplitude = sqrtf(fmaxf(prod, 0.0f));
        const float tp_in_real = light_throughput_real[index];
        const float tp_in_imag = light_throughput_imag[index];
        const float g_tp_out_real =
            grad_throughput_real != nullptr ? grad_throughput_real[index] : 0.0f;
        const float g_tp_out_imag =
            grad_throughput_imag != nullptr ? grad_throughput_imag[index] : 0.0f;
        const float g_amplitude = tp_in_real * g_tp_out_real + tp_in_imag * g_tp_out_imag;
        const float g_prod = (prod > 0.0f) ? g_amplitude * 0.5f / amplitude : 0.0f;
        const float g_gain_throughput = (gain >= 0.0f) ? g_prod * reflectance : 0.0f;
        const float g_reflectance = g_prod * gain_clamped;

        if (need_grad_material) {
            field::Complex dte, dtm;
            // eps
            {
                ad::DualC te_d, tm_d;
                ad::slab_fresnel_dual(
                    frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, te_d, tm_d);
                const float g = adj_dot(g_r_te, te_d.d) + adj_dot(g_r_tm, tm_d.d) +
                    g_reflectance * d_reflectance_eps;
                atomicAdd(grad_eps_r + prim, g);
            }
            // sigma
            {
                ad::DualC te_d, tm_d;
                ad::slab_fresnel_dual(
                    frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, te_d, tm_d);
                const float g = adj_dot(g_r_te, te_d.d) + adj_dot(g_r_tm, tm_d.d) +
                    g_reflectance * d_reflectance_sigma;
                atomicAdd(grad_sigma_e + prim, g);
            }
            // gain
            {
                ad::DualC te_d, tm_d;
                ad::slab_fresnel_dual(
                    frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
                    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, te_d, tm_d);
                const float g = adj_dot(g_r_te, te_d.d) + adj_dot(g_r_tm, tm_d.d) +
                    g_gain_throughput;
                atomicAdd(grad_gain + prim, g);
            }
            // thickness (field only)
            {
                ad::DualC te_d, tm_d;
                ad::slab_fresnel_dual(
                    frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, te_d, tm_d);
                const float g = adj_dot(g_r_te, te_d.d) + adj_dot(g_r_tm, tm_d.d);
                atomicAdd(grad_thickness + prim, g);
            }
        }
        if (need_grad_frequency) {
            ad::DualC te_d, tm_d;
            ad::slab_fresnel_dual(
                frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, te_d, tm_d);
            const float g = adj_dot(g_r_te, te_d.d) + adj_dot(g_r_tm, tm_d.d) +
                g_reflectance * d_reflectance_freq;
            atomicAdd(grad_frequency, g);
        }
        if (need_grad_field_in) {
            const int64_t base = index * 3;
            grad_light_field_real[base] = g_incoming.x.re;
            grad_light_field_real[base + 1] = g_incoming.y.re;
            grad_light_field_real[base + 2] = g_incoming.z.re;
            grad_light_field_imag[base] = g_incoming.x.im;
            grad_light_field_imag[base + 1] = g_incoming.y.im;
            grad_light_field_imag[base + 2] = g_incoming.z.im;
            grad_light_throughput_real[index] = amplitude * g_tp_out_real;
            grad_light_throughput_imag[index] = amplitude * g_tp_out_imag;
        }
    }
}

__global__ void reflected_subpath_jvp_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const bool* light_valid,
    const float* light_field_real,
    const float* light_field_imag,
    const float* hit_t,
    const float* hit_n,
    const int* hit_global_prim_id,
    const float* material_gain,
    const bool* material_valid,
    const float* material_eps_r,
    const float* material_sigma_e,
    const float* material_mu_r,
    const float* material_thickness,
    float frequency_hz,
    int64_t material_count,
    const float* tangent_eps_r,
    const float* tangent_sigma_e,
    const float* tangent_gain,
    const float* tangent_thickness,
    float tangent_frequency,
    const float* tangent_light_field_real,
    const float* tangent_light_field_imag,
    const float* tangent_light_throughput_real,
    const float* tangent_light_throughput_imag,
    float* tangent_field_real,
    float* tangent_field_imag,
    float* tangent_throughput_real,
    float* tangent_throughput_imag) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const int prim = hit_global_prim_id[index];
        const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < material_count;
        const bool material_ok = prim_in_range && material_valid[prim];
        const bool is_valid =
            light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
        if (!is_valid) {
            const c10::complex<float> zero(0.0f, 0.0f);
            tangent_field_real[base] = 0.0f; tangent_field_real[base + 1] = 0.0f;
            tangent_field_real[base + 2] = 0.0f;
            tangent_field_imag[base] = 0.0f; tangent_field_imag[base + 1] = 0.0f;
            tangent_field_imag[base + 2] = 0.0f;
            tangent_throughput_real[index] = 0.0f;
            tangent_throughput_imag[index] = 0.0f;
            continue;
        }
        const float eps_r = material_eps_r[prim];
        const float sigma_e = material_sigma_e[prim];
        const float mu_r = material_mu_r[prim];
        const float gain = material_gain[prim];
        const float thickness = material_thickness[prim];
        const float t_eps = tangent_eps_r != nullptr ? tangent_eps_r[prim] : 0.0f;
        const float t_sigma = tangent_sigma_e != nullptr ? tangent_sigma_e[prim] : 0.0f;
        const float t_gain = tangent_gain != nullptr ? tangent_gain[prim] : 0.0f;
        const float t_thick = tangent_thickness != nullptr ? tangent_thickness[prim] : 0.0f;

        const field::float3a incident = load3f(light_direction, index);
        const field::float3a normal = load3f(hit_n, index);
        const transport::ReflectFrame frame = transport::reflect_frame(incident, normal);
        ad::DualC r_te, r_tm;
        ad::slab_fresnel_dual(
            frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
            0.0f, t_eps, t_sigma, t_gain, t_thick, tangent_frequency, r_te, r_tm);
        const field::Complex3 incoming = load_field3(light_field_real, light_field_imag, index);
        const field::Complex3 t_incoming = {
            opt_c(tangent_light_field_real, tangent_light_field_imag, base),
            opt_c(tangent_light_field_real, tangent_light_field_imag, base + 1),
            opt_c(tangent_light_field_real, tangent_light_field_imag, base + 2)};
        const field::Complex e_s = transport::complex3_dot_real(incoming, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(incoming, frame.p_in);
        const field::Complex t_e_s = transport::complex3_dot_real(t_incoming, frame.s_axis);
        const field::Complex t_e_p = transport::complex3_dot_real(t_incoming, frame.p_in);
        const field::Complex w_te = field::cplx_mul(r_te.v, e_s);
        const field::Complex w_tm = field::cplx_mul(r_tm.v, e_p);
        const field::Complex t_w_te = field::cplx_add(
            field::cplx_mul(r_te.d, e_s), field::cplx_mul(r_te.v, t_e_s));
        const field::Complex t_w_tm = field::cplx_add(
            field::cplx_mul(r_tm.d, e_p), field::cplx_mul(r_tm.v, t_e_p));
        const field::Complex3 t_updated = field::c3_add(
            field::cplx_scale_real(frame.s_axis, t_w_te),
            field::cplx_scale_real(frame.p_out, t_w_tm));
        tangent_field_real[base] = t_updated.x.re;
        tangent_field_real[base + 1] = t_updated.y.re;
        tangent_field_real[base + 2] = t_updated.z.re;
        tangent_field_imag[base] = t_updated.x.im;
        tangent_field_imag[base + 1] = t_updated.y.im;
        tangent_field_imag[base + 2] = t_updated.z.im;

        // Throughput proxy tangent.
        const ReflectanceGeometry geo = reflectance_geometry(
            light_direction + index * 3, hit_n + index * 3);
        float d_reflectance = 0.0f;
        const float reflectance = effective_reflectance_dual(
            geo, eps_r, sigma_e, mu_r, frequency_hz, t_eps, t_sigma, tangent_frequency,
            d_reflectance);
        const float gain_clamped = fmaxf(gain, 0.0f);
        const float t_gain_clamped = (gain >= 0.0f) ? t_gain : 0.0f;
        const float prod = gain_clamped * reflectance;
        const float t_prod = t_gain_clamped * reflectance + gain_clamped * d_reflectance;
        const float amplitude = sqrtf(fmaxf(prod, 0.0f));
        const float t_amplitude = (prod > 0.0f) ? 0.5f / amplitude * t_prod : 0.0f;
        const float tp_in_real = light_throughput_real[index];
        const float tp_in_imag = light_throughput_imag[index];
        const float t_tp_in_real =
            tangent_light_throughput_real != nullptr ? tangent_light_throughput_real[index] : 0.0f;
        const float t_tp_in_imag =
            tangent_light_throughput_imag != nullptr ? tangent_light_throughput_imag[index] : 0.0f;
        tangent_throughput_real[index] = t_tp_in_real * amplitude + tp_in_real * t_amplitude;
        tangent_throughput_imag[index] = t_tp_in_imag * amplitude + tp_in_imag * t_amplitude;
    }
}

// ---------------------------------------------------------------------------
// Transmitted light subpath advance companions. The wall operator is the CSR
// slab Jones response plus the exact lateral-shift compensation phase
//   phi = k_par * lateral - k0 * jump.
// ---------------------------------------------------------------------------

struct DFloat {
    float v;
    float d;
};

__device__ __forceinline__ DFloat dfl(float v) { return {v, 0.0f}; }
__device__ __forceinline__ DFloat dfl_add(DFloat a, DFloat b) { return {a.v + b.v, a.d + b.d}; }
__device__ __forceinline__ DFloat dfl_sub(DFloat a, DFloat b) { return {a.v - b.v, a.d - b.d}; }
__device__ __forceinline__ DFloat dfl_mul(DFloat a, DFloat b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
__device__ __forceinline__ DFloat dfl_scale(DFloat a, float s) { return {a.v * s, a.d * s}; }
__device__ __forceinline__ DFloat dfl_div(DFloat a, DFloat b) {
    const float inv = 1.0f / b.v;
    return {a.v * inv, (a.d * b.v - a.v * b.d) * inv * inv};
}
__device__ __forceinline__ DFloat dfl_sqrt_floor(DFloat a, float floor_value) {
    if (a.v > floor_value) {
        const float s = sqrtf(a.v);
        return {s, 0.5f * a.d / s};
    }
    return {sqrtf(fmaxf(a.v, floor_value)), 0.0f};
}

struct TransmitFrozen {
    field::float3a incident;
    field::float3a normal_in;
    field::float3a u_par;
    field::float3a s_axis;
    field::float3a p_axis;
    float cos_theta;
    float sin_theta;
};

__device__ TransmitFrozen transmit_frozen(const float* light_direction, const float* hit_n, int64_t index) {
    TransmitFrozen out;
    out.incident = field::safe_normalize(load3f(light_direction, index), field::make_f3(0.0f, 0.0f, 1.0f));
    field::float3a normal_in = field::safe_normalize(load3f(hit_n, index), field::make_f3(0.0f, 0.0f, 1.0f));
    if (field::f3_dot(out.incident, normal_in) > 0.0f)
        normal_in = field::f3_neg(normal_in);
    out.normal_in = normal_in;
    out.cos_theta = fminf(fmaxf(-field::f3_dot(out.incident, normal_in), kSubpathEps), 1.0f);
    out.sin_theta = sqrtf(fmaxf(1.0f - out.cos_theta * out.cos_theta, 0.0f));
    out.u_par = field::safe_normalize(
        field::f3_add(out.incident, field::f3_mul(normal_in, out.cos_theta)),
        field::stable_perp_basis(normal_in, out.incident));
    field::float3a s_axis = field::f3_cross(normal_in, out.incident);
    out.s_axis = field::safe_normalize(s_axis, field::stable_perp_basis(out.incident, normal_in));
    out.p_axis = field::safe_normalize(
        field::f3_cross(out.s_axis, out.incident),
        field::stable_perp_basis(out.incident, out.s_axis));
    return out;
}

// Dual of phi = k_par*lateral - k0*jump for one layer/frequency seed.
template <typename SeedFn>
__device__ DFloat transmit_phi_dual(
    const TransmitFrozen& geo,
    const em::LayerView& layers,
    int material,
    float frequency_hz,
    float d_freq,
    SeedFn&& seed) {
    const DFloat omega = {2.0f * field::UTD_PI * frequency_hz, 2.0f * field::UTD_PI * d_freq};
    const DFloat k0 = dfl_scale(omega, 1.0f / transport::kSpeedOfLight);
    const DFloat k_par = dfl_scale(k0, geo.sin_theta);
    DFloat total_thickness = dfl(0.0f);
    DFloat lateral = dfl(0.0f);
    const int first = layers.layer_offset[material];
    const int layers_in_wall = layers.layer_count[material];
    for (int layer = 0; layer < layers_in_wall; ++layer) {
        const int slot = first + layer;
        const ad::LayerSeed s = seed(slot);
        const float thickness_raw = layers.layer_thickness_m[slot];
        const DFloat thickness = {
            fmaxf(thickness_raw, 0.0f), thickness_raw >= 0.0f ? s.d_thickness : 0.0f};
        const ad::DualMedium medium = ad::make_medium_dual(
            layers.layer_eps_r[slot], layers.layer_sigma_e[slot], layers.layer_mu_r[slot],
            {omega.v, omega.d}, s.d_eps, s.d_sigma);
        const DFloat k0_clamped = {
            fmaxf(k0.v, kSubpathEps), k0.v > kSubpathEps ? k0.d : 0.0f};
        DFloat ratio = dfl_div({medium.k.v.re, medium.k.d.re}, k0_clamped);
        if (!(ratio.v > field::UTD_SMALL_EPS))
            ratio = {field::UTD_SMALL_EPS, 0.0f};
        const DFloat sin_layer = dfl_div(dfl(geo.sin_theta), ratio);
        const DFloat cos_layer = dfl_sqrt_floor(
            dfl_sub(dfl(1.0f), dfl_mul(sin_layer, sin_layer)), 1.0e-6f);
        total_thickness = dfl_add(total_thickness, thickness);
        lateral = dfl_add(lateral, dfl_mul(thickness, dfl_div(sin_layer, cos_layer)));
    }
    // A = u_par*lateral - normal_in*total_thickness (frozen unit vectors).
    const DFloat ax = dfl_sub(dfl_scale(lateral, geo.u_par.x), dfl_scale(total_thickness, geo.normal_in.x));
    const DFloat ay = dfl_sub(dfl_scale(lateral, geo.u_par.y), dfl_scale(total_thickness, geo.normal_in.y));
    const DFloat az = dfl_sub(dfl_scale(lateral, geo.u_par.z), dfl_scale(total_thickness, geo.normal_in.z));
    const DFloat sq = dfl_add(dfl_add(dfl_mul(ax, ax), dfl_mul(ay, ay)), dfl_mul(az, az));
    const DFloat jump = dfl_sqrt_floor(sq, 0.0f);
    return dfl_sub(dfl_mul(k_par, lateral), dfl_mul(k0, jump));
}

struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct BasisSeed {
    int slot;
    int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed s{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0) s.d_thickness = 1.0f;
            else if (param == 1) s.d_eps = 1.0f;
            else s.d_sigma = 1.0f;
        }
        return s;
    }
};

struct TangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {
            t_thickness != nullptr ? t_thickness[query] : 0.0f,
            t_eps != nullptr ? t_eps[query] : 0.0f,
            t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

// Frozen w_s/w_p transverse-projected polarization power weights (matches
// sp_proxy_weights in the forward).
__device__ void transmit_proxy_weights(
    field::float3a incident, field::float3a normal_in, float& w_s, float& w_p) {
    field::float3a s_axis = field::f3_cross(normal_in, incident);
    const float s_len = field::safe_length(s_axis);
    if (s_len <= kSubpathEps) { w_s = 1.0f; w_p = 0.0f; return; }
    s_axis = field::f3_div(s_axis, s_len);
    const field::float3a p_axis = field::f3_cross(s_axis, incident);
    const field::float3a x_hat = field::make_f3(1.0f, 0.0f, 0.0f);
    const field::float3a transverse = field::f3_sub(
        x_hat, field::f3_mul(incident, field::f3_dot(x_hat, incident)));
    const float t_len = field::safe_length(transverse);
    if (t_len <= kSubpathEps) { w_s = 1.0f; w_p = 0.0f; return; }
    const float e_s = field::f3_dot(transverse, s_axis) / t_len;
    const float e_p = field::f3_dot(transverse, p_axis) / t_len;
    w_s = e_s * e_s;
    w_p = e_p * e_p;
}

__global__ void transmitted_subpath_backward_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const bool* light_valid,
    const float* light_field_real,
    const float* light_field_imag,
    const float* hit_t,
    const float* hit_n,
    const int* hit_global_prim_id,
    const int* face_material_id,
    int64_t face_count,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const float* grad_field_real,
    const float* grad_field_imag,
    const float* grad_throughput_real,
    const float* grad_throughput_imag,
    float* grad_layer_thickness,
    float* grad_layer_eps_r,
    float* grad_layer_sigma_e,
    float* grad_light_field_real,
    float* grad_light_field_imag,
    float* grad_light_throughput_real,
    float* grad_light_throughput_imag,
    float* grad_frequency,
    bool need_grad_layers,
    bool need_grad_field_in,
    bool need_grad_frequency) {
    const em::LayerView layers_base{
        layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, 0};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int prim = hit_global_prim_id[index];
        const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < face_count;
        const int material = prim_in_range ? face_material_id[prim] : -1;
        const bool material_ok = material >= 0 && static_cast<int64_t>(material) < material_count;
        const bool is_valid =
            light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
        if (!is_valid) {
            if (need_grad_field_in) {
                const int64_t base = index * 3;
                grad_light_field_real[base] = 0.0f;
                grad_light_field_real[base + 1] = 0.0f;
                grad_light_field_real[base + 2] = 0.0f;
                grad_light_field_imag[base] = 0.0f;
                grad_light_field_imag[base + 1] = 0.0f;
                grad_light_field_imag[base + 2] = 0.0f;
                grad_light_throughput_real[index] = 0.0f;
                grad_light_throughput_imag[index] = 0.0f;
            }
            continue;
        }
        em::LayerView layers = layers_base;
        layers.material = material;
        const TransmitFrozen geo = transmit_frozen(light_direction, hit_n, index);
        const em::StackRT te = em::stack_rt(geo.cos_theta, layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(geo.cos_theta, layers, frequency_hz, em::kPolTM);
        const field::Complex3 incoming = load_field3(light_field_real, light_field_imag, index);
        const field::Complex e_s = transport::complex3_dot_real(incoming, geo.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(incoming, geo.p_axis);
        const field::Complex3 updated_pre = field::c3_add(
            field::cplx_scale_real(geo.s_axis, field::cplx_mul(te.t, e_s)),
            field::cplx_scale_real(geo.p_axis, field::cplx_mul(tm.t, e_p)));
        const DFloat phi = transmit_phi_dual(
            geo, layers, material, frequency_hz, 0.0f, ZeroSeed{});
        const field::Complex compensation = em::c_exp_neg_j(static_cast<double>(phi.v));

        // updated = updated_pre * compensation.
        const field::Complex3 g_updated = load_field3(grad_field_real, grad_field_imag, index);
        field::Complex3 g_updated_pre = field::c3_zero();
        field::Complex g_compensation = field::cplx_zero();
        field::adj_cplx_mul(updated_pre.x, compensation, g_updated.x, g_updated_pre.x, g_compensation);
        field::adj_cplx_mul(updated_pre.y, compensation, g_updated.y, g_updated_pre.y, g_compensation);
        field::adj_cplx_mul(updated_pre.z, compensation, g_updated.z, g_updated_pre.z, g_compensation);
        // g_phi from compensation = (cos phi, -sin phi).
        const float g_phi = g_compensation.re * compensation.im -
            g_compensation.im * compensation.re;

        // Jones adjoint of updated_pre.
        field::float3a g_axis_dump = field::f3_zero();
        field::Complex g_w_te = field::cplx_zero();
        field::Complex g_w_tm = field::cplx_zero();
        field::adj_cplx_scale_real(
            geo.s_axis, field::cplx_mul(te.t, e_s), g_updated_pre, g_axis_dump, g_w_te);
        field::adj_cplx_scale_real(
            geo.p_axis, field::cplx_mul(tm.t, e_p), g_updated_pre, g_axis_dump, g_w_tm);
        field::Complex g_t_te = field::cplx_zero();
        field::Complex g_t_tm = field::cplx_zero();
        field::Complex g_e_s = field::cplx_zero();
        field::Complex g_e_p = field::cplx_zero();
        field::adj_cplx_mul(te.t, e_s, g_w_te, g_t_te, g_e_s);
        field::adj_cplx_mul(tm.t, e_p, g_w_tm, g_t_tm, g_e_p);
        field::Complex3 g_incoming = field::c3_zero();
        field::adj_cplx_dot_real(incoming, geo.s_axis, g_e_s, g_incoming, g_axis_dump);
        field::adj_cplx_dot_real(incoming, geo.p_axis, g_e_p, g_incoming, g_axis_dump);

        // Throughput.
        float w_s, w_p;
        transmit_proxy_weights(geo.incident, geo.normal_in, w_s, w_p);
        const float effective_transmittance = fmaxf(te.cap_t * w_s + tm.cap_t * w_p, 0.0f);
        const float amplitude = sqrtf(effective_transmittance);
        const float tp_in_real = light_throughput_real[index];
        const float tp_in_imag = light_throughput_imag[index];
        const float g_tp_out_real =
            grad_throughput_real != nullptr ? grad_throughput_real[index] : 0.0f;
        const float g_tp_out_imag =
            grad_throughput_imag != nullptr ? grad_throughput_imag[index] : 0.0f;
        const float g_amplitude = tp_in_real * g_tp_out_real + tp_in_imag * g_tp_out_imag;
        const float raw_transmittance = te.cap_t * w_s + tm.cap_t * w_p;
        const float g_E = (raw_transmittance > 0.0f) ? g_amplitude * 0.5f / amplitude : 0.0f;

        const int first = layer_offset[material];
        const int layers_in_wall = layer_count[material];
        if (need_grad_layers) {
            for (int layer = 0; layer < layers_in_wall; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT te_d = ad::stack_rt_dual(
                        geo.cos_theta, layers, frequency_hz, 0.0f, 0.0f, em::kPolTE, seed);
                    const ad::DualStackRT tm_d = ad::stack_rt_dual(
                        geo.cos_theta, layers, frequency_hz, 0.0f, 0.0f, em::kPolTM, seed);
                    float g = adj_dot(g_t_te, te_d.t.d) + adj_dot(g_t_tm, tm_d.t.d);
                    g += g_E * (te_d.cap_t.d * w_s + tm_d.cap_t.d * w_p);
                    const DFloat phi_d = transmit_phi_dual(
                        geo, layers, material, frequency_hz, 0.0f, seed);
                    g += g_phi * phi_d.d;
                    float* destination = param == 0 ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    atomicAdd(destination + slot, g);
                }
            }
        }
        if (need_grad_frequency) {
            const ZeroSeed zero_seed;
            const ad::DualStackRT te_d = ad::stack_rt_dual(
                geo.cos_theta, layers, frequency_hz, 0.0f, 1.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm_d = ad::stack_rt_dual(
                geo.cos_theta, layers, frequency_hz, 0.0f, 1.0f, em::kPolTM, zero_seed);
            float g = adj_dot(g_t_te, te_d.t.d) + adj_dot(g_t_tm, tm_d.t.d);
            g += g_E * (te_d.cap_t.d * w_s + tm_d.cap_t.d * w_p);
            const DFloat phi_d = transmit_phi_dual(
                geo, layers, material, frequency_hz, 1.0f, zero_seed);
            g += g_phi * phi_d.d;
            atomicAdd(grad_frequency, g);
        }
        if (need_grad_field_in) {
            const int64_t base = index * 3;
            grad_light_field_real[base] = g_incoming.x.re;
            grad_light_field_real[base + 1] = g_incoming.y.re;
            grad_light_field_real[base + 2] = g_incoming.z.re;
            grad_light_field_imag[base] = g_incoming.x.im;
            grad_light_field_imag[base + 1] = g_incoming.y.im;
            grad_light_field_imag[base + 2] = g_incoming.z.im;
            grad_light_throughput_real[index] = amplitude * g_tp_out_real;
            grad_light_throughput_imag[index] = amplitude * g_tp_out_imag;
        }
    }
}

__global__ void transmitted_subpath_jvp_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const bool* light_valid,
    const float* light_field_real,
    const float* light_field_imag,
    const float* hit_t,
    const float* hit_n,
    const int* hit_global_prim_id,
    const int* face_material_id,
    int64_t face_count,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    const float* tangent_layer_thickness,
    const float* tangent_layer_eps_r,
    const float* tangent_layer_sigma_e,
    float tangent_frequency,
    const float* tangent_light_field_real,
    const float* tangent_light_field_imag,
    const float* tangent_light_throughput_real,
    const float* tangent_light_throughput_imag,
    float* tangent_field_real,
    float* tangent_field_imag,
    float* tangent_throughput_real,
    float* tangent_throughput_imag) {
    const em::LayerView layers_base{
        layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, 0};
    const TangentSeed tangent_seed{
        tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const int prim = hit_global_prim_id[index];
        const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < face_count;
        const int material = prim_in_range ? face_material_id[prim] : -1;
        const bool material_ok = material >= 0 && static_cast<int64_t>(material) < material_count;
        const bool is_valid =
            light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
        if (!is_valid) {
            tangent_field_real[base] = 0.0f; tangent_field_real[base + 1] = 0.0f;
            tangent_field_real[base + 2] = 0.0f;
            tangent_field_imag[base] = 0.0f; tangent_field_imag[base + 1] = 0.0f;
            tangent_field_imag[base + 2] = 0.0f;
            tangent_throughput_real[index] = 0.0f;
            tangent_throughput_imag[index] = 0.0f;
            continue;
        }
        em::LayerView layers = layers_base;
        layers.material = material;
        const TransmitFrozen geo = transmit_frozen(light_direction, hit_n, index);
        const ad::DualStackRT te = ad::stack_rt_dual(
            geo.cos_theta, layers, frequency_hz, 0.0f, tangent_frequency, em::kPolTE, tangent_seed);
        const ad::DualStackRT tm = ad::stack_rt_dual(
            geo.cos_theta, layers, frequency_hz, 0.0f, tangent_frequency, em::kPolTM, tangent_seed);
        const field::Complex3 incoming = load_field3(light_field_real, light_field_imag, index);
        const field::Complex3 t_incoming = {
            opt_c(tangent_light_field_real, tangent_light_field_imag, base),
            opt_c(tangent_light_field_real, tangent_light_field_imag, base + 1),
            opt_c(tangent_light_field_real, tangent_light_field_imag, base + 2)};
        const field::Complex e_s = transport::complex3_dot_real(incoming, geo.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(incoming, geo.p_axis);
        const field::Complex t_e_s = transport::complex3_dot_real(t_incoming, geo.s_axis);
        const field::Complex t_e_p = transport::complex3_dot_real(t_incoming, geo.p_axis);
        const field::Complex w_te = field::cplx_mul(te.t.v, e_s);
        const field::Complex w_tm = field::cplx_mul(tm.t.v, e_p);
        const field::Complex t_w_te = field::cplx_add(
            field::cplx_mul(te.t.d, e_s), field::cplx_mul(te.t.v, t_e_s));
        const field::Complex t_w_tm = field::cplx_add(
            field::cplx_mul(tm.t.d, e_p), field::cplx_mul(tm.t.v, t_e_p));
        const field::Complex3 updated_pre = field::c3_add(
            field::cplx_scale_real(geo.s_axis, w_te),
            field::cplx_scale_real(geo.p_axis, w_tm));
        const field::Complex3 t_updated_pre = field::c3_add(
            field::cplx_scale_real(geo.s_axis, t_w_te),
            field::cplx_scale_real(geo.p_axis, t_w_tm));
        const DFloat phi = transmit_phi_dual(
            geo, layers, material, frequency_hz, tangent_frequency, tangent_seed);
        const field::Complex compensation = em::c_exp_neg_j(static_cast<double>(phi.v));
        // t_compensation = (d cos/dphi, d(-sin)/dphi) * t_phi = (comp.im, -comp.re)*t_phi.
        const field::Complex t_compensation = field::cplx(
            compensation.im * phi.d, -compensation.re * phi.d);
        const field::Complex3 t_updated = field::c3_add(
            field::c3_scale(t_updated_pre, compensation),
            field::c3_scale(updated_pre, t_compensation));
        tangent_field_real[base] = t_updated.x.re;
        tangent_field_real[base + 1] = t_updated.y.re;
        tangent_field_real[base + 2] = t_updated.z.re;
        tangent_field_imag[base] = t_updated.x.im;
        tangent_field_imag[base + 1] = t_updated.y.im;
        tangent_field_imag[base + 2] = t_updated.z.im;

        // Throughput.
        float w_s, w_p;
        transmit_proxy_weights(geo.incident, geo.normal_in, w_s, w_p);
        const float raw_transmittance = te.cap_t.v * w_s + tm.cap_t.v * w_p;
        const float t_raw = te.cap_t.d * w_s + tm.cap_t.d * w_p;
        const float effective = fmaxf(raw_transmittance, 0.0f);
        const float amplitude = sqrtf(effective);
        const float t_amplitude = (raw_transmittance > 0.0f) ? 0.5f / amplitude * t_raw : 0.0f;
        const float tp_in_real = light_throughput_real[index];
        const float tp_in_imag = light_throughput_imag[index];
        const float t_tp_in_real =
            tangent_light_throughput_real != nullptr ? tangent_light_throughput_real[index] : 0.0f;
        const float t_tp_in_imag =
            tangent_light_throughput_imag != nullptr ? tangent_light_throughput_imag[index] : 0.0f;
        tangent_throughput_real[index] = t_tp_in_real * amplitude + tp_in_real * t_amplitude;
        tangent_throughput_imag[index] = t_tp_in_imag * amplitude + tp_in_imag * t_amplitude;
    }
}

}  // namespace

// ===========================================================================
// Host bridges.
// ===========================================================================

pybind11::dict cn_bdpt_reflected_light_subpath_state_backward(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_material,
    bool need_grad_field_in,
    bool need_grad_frequency) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    check_flat_tensor(material_eps_r, "material_eps_r", at::kFloat);
    check_flat_tensor(material_sigma_e, "material_sigma_e", at::kFloat);
    check_flat_tensor(material_mu_r, "material_mu_r", at::kFloat);
    check_flat_tensor(material_thickness, "material_thickness", at::kFloat);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = light_direction.size(0);
    const int64_t material_count = material_gain.size(0);

    at::Tensor gfr_s, gfi_s, gtr_s, gti_s;
    const at::Tensor* gfr = optional_grad(
        std::move(grad_field_real), gfr_s, "grad_field_real", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* gfi = optional_grad(
        std::move(grad_field_imag), gfi_s, "grad_field_imag", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* gtr = optional_grad(
        std::move(grad_throughput_real), gtr_s, "grad_throughput_real", at::kFloat, {count}, light_direction);
    const at::Tensor* gti = optional_grad(
        std::move(grad_throughput_imag), gti_s, "grad_throughput_imag", at::kFloat, {count}, light_direction);
    // The Jones field cotangent must be present as a real/imag pair.
    const bool have_field_grad = gfr != nullptr && gfi != nullptr;

    at::Tensor grad_eps, grad_sigma, grad_gain, grad_thick;
    at::Tensor grad_lfr, grad_lfi, grad_ltr, grad_lti, grad_frequency;
    if (need_grad_material) {
        grad_eps = zero_filled({material_count}, light_direction.options());
        grad_sigma = zero_filled({material_count}, light_direction.options());
        grad_gain = zero_filled({material_count}, light_direction.options());
        grad_thick = zero_filled({material_count}, light_direction.options());
    }
    if (need_grad_field_in) {
        grad_lfr = zero_filled({count, 3}, light_direction.options());
        grad_lfi = zero_filled({count, 3}, light_direction.options());
        grad_ltr = zero_filled({count}, light_direction.options());
        grad_lti = zero_filled({count}, light_direction.options());
    }
    if (need_grad_frequency) {
        grad_frequency = zero_filled({1}, light_direction.options());
    }
    if (count > 0 && (need_grad_material || need_grad_field_in || need_grad_frequency)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_direction.get_device()).stream();
        reflected_subpath_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_valid.data_ptr<bool>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            material_eps_r.data_ptr<float>(),
            material_sigma_e.data_ptr<float>(),
            material_mu_r.data_ptr<float>(),
            material_thickness.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            material_count,
            have_field_grad ? gfr->data_ptr<float>() : nullptr,
            have_field_grad ? gfi->data_ptr<float>() : nullptr,
            gtr != nullptr ? gtr->data_ptr<float>() : nullptr,
            gti != nullptr ? gti->data_ptr<float>() : nullptr,
            need_grad_material ? grad_eps.data_ptr<float>() : nullptr,
            need_grad_material ? grad_sigma.data_ptr<float>() : nullptr,
            need_grad_material ? grad_gain.data_ptr<float>() : nullptr,
            need_grad_material ? grad_thick.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lfr.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lfi.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_ltr.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lti.data_ptr<float>() : nullptr,
            need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
            need_grad_material,
            need_grad_field_in,
            need_grad_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    auto emit = [](const at::Tensor& t, bool on) {
        return on ? pybind11::cast(t) : pybind11::object(pybind11::none());
    };
    out["grad_eps_r"] = emit(grad_eps, need_grad_material);
    out["grad_sigma_e"] = emit(grad_sigma, need_grad_material);
    out["grad_gain"] = emit(grad_gain, need_grad_material);
    out["grad_thickness"] = emit(grad_thick, need_grad_material);
    out["grad_light_field_real"] = emit(grad_lfr, need_grad_field_in);
    out["grad_light_field_imag"] = emit(grad_lfi, need_grad_field_in);
    out["grad_light_throughput_real"] = emit(grad_ltr, need_grad_field_in);
    out["grad_light_throughput_imag"] = emit(grad_lti, need_grad_field_in);
    out["grad_frequency"] = emit(grad_frequency, need_grad_frequency);
    return out;
}

pybind11::dict cn_bdpt_reflected_light_subpath_state_jvp(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
    double frequency_hz,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_light_throughput_real,
    pybind11::object tangent_light_throughput_imag) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    check_flat_tensor(material_eps_r, "material_eps_r", at::kFloat);
    check_flat_tensor(material_sigma_e, "material_sigma_e", at::kFloat);
    check_flat_tensor(material_mu_r, "material_mu_r", at::kFloat);
    check_flat_tensor(material_thickness, "material_thickness", at::kFloat);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = light_direction.size(0);
    const int64_t material_count = material_gain.size(0);

    at::Tensor te_s, ts_s, tg_s, tt_s, tlfr_s, tlfi_s, tltr_s, tlti_s;
    const at::Tensor* t_eps = optional_grad(
        std::move(tangent_eps_r), te_s, "tangent_eps_r", at::kFloat, {material_count}, light_direction);
    const at::Tensor* t_sigma = optional_grad(
        std::move(tangent_sigma_e), ts_s, "tangent_sigma_e", at::kFloat, {material_count}, light_direction);
    const at::Tensor* t_gain = optional_grad(
        std::move(tangent_gain), tg_s, "tangent_gain", at::kFloat, {material_count}, light_direction);
    const at::Tensor* t_thick = optional_grad(
        std::move(tangent_thickness), tt_s, "tangent_thickness", at::kFloat, {material_count}, light_direction);
    const at::Tensor* t_lfr = optional_grad(
        std::move(tangent_light_field_real), tlfr_s, "tangent_light_field_real", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* t_lfi = optional_grad(
        std::move(tangent_light_field_imag), tlfi_s, "tangent_light_field_imag", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* t_ltr = optional_grad(
        std::move(tangent_light_throughput_real), tltr_s, "tangent_light_throughput_real", at::kFloat, {count}, light_direction);
    const at::Tensor* t_lti = optional_grad(
        std::move(tangent_light_throughput_imag), tlti_s, "tangent_light_throughput_imag", at::kFloat, {count}, light_direction);

    auto tangent_field_real = at::empty({count, 3}, light_direction.options());
    auto tangent_field_imag = at::empty({count, 3}, light_direction.options());
    auto tangent_throughput_real = at::empty({count}, light_direction.options());
    auto tangent_throughput_imag = at::empty({count}, light_direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_direction.get_device()).stream();
        reflected_subpath_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_valid.data_ptr<bool>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            material_eps_r.data_ptr<float>(),
            material_sigma_e.data_ptr<float>(),
            material_mu_r.data_ptr<float>(),
            material_thickness.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            material_count,
            grad_ptr<float>(t_eps),
            grad_ptr<float>(t_sigma),
            grad_ptr<float>(t_gain),
            grad_ptr<float>(t_thick),
            static_cast<float>(tangent_frequency),
            grad_ptr<float>(t_lfr),
            grad_ptr<float>(t_lfi),
            grad_ptr<float>(t_ltr),
            grad_ptr<float>(t_lti),
            tangent_field_real.data_ptr<float>(),
            tangent_field_imag.data_ptr<float>(),
            tangent_throughput_real.data_ptr<float>(),
            tangent_throughput_imag.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_real"] = tangent_field_real;
    out["tangent_field_imag"] = tangent_field_imag;
    out["tangent_throughput_real"] = tangent_throughput_real;
    out["tangent_throughput_imag"] = tangent_throughput_imag;
    return out;
}

pybind11::dict cn_bdpt_transmitted_light_subpath_state_backward(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_layers,
    bool need_grad_field_in,
    bool need_grad_frequency) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(face_material_id, "face_material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = light_direction.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t face_count = face_material_id.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    at::Tensor gfr_s, gfi_s, gtr_s, gti_s;
    const at::Tensor* gfr = optional_grad(
        std::move(grad_field_real), gfr_s, "grad_field_real", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* gfi = optional_grad(
        std::move(grad_field_imag), gfi_s, "grad_field_imag", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* gtr = optional_grad(
        std::move(grad_throughput_real), gtr_s, "grad_throughput_real", at::kFloat, {count}, light_direction);
    const at::Tensor* gti = optional_grad(
        std::move(grad_throughput_imag), gti_s, "grad_throughput_imag", at::kFloat, {count}, light_direction);
    const bool have_field_grad = gfr != nullptr && gfi != nullptr;

    at::Tensor grad_thickness, grad_eps, grad_sigma;
    at::Tensor grad_lfr, grad_lfi, grad_ltr, grad_lti, grad_frequency;
    if (need_grad_layers) {
        grad_thickness = zero_filled({layer_total}, light_direction.options());
        grad_eps = zero_filled({layer_total}, light_direction.options());
        grad_sigma = zero_filled({layer_total}, light_direction.options());
    }
    if (need_grad_field_in) {
        grad_lfr = zero_filled({count, 3}, light_direction.options());
        grad_lfi = zero_filled({count, 3}, light_direction.options());
        grad_ltr = zero_filled({count}, light_direction.options());
        grad_lti = zero_filled({count}, light_direction.options());
    }
    if (need_grad_frequency) {
        grad_frequency = zero_filled({1}, light_direction.options());
    }
    if (count > 0 && (need_grad_layers || need_grad_field_in || need_grad_frequency)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_direction.get_device()).stream();
        transmitted_subpath_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_valid.data_ptr<bool>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            face_material_id.data_ptr<int>(),
            face_count,
            layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(),
            layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(),
            material_count,
            static_cast<float>(frequency_hz),
            have_field_grad ? gfr->data_ptr<float>() : nullptr,
            have_field_grad ? gfi->data_ptr<float>() : nullptr,
            gtr != nullptr ? gtr->data_ptr<float>() : nullptr,
            gti != nullptr ? gti->data_ptr<float>() : nullptr,
            need_grad_layers ? grad_thickness.data_ptr<float>() : nullptr,
            need_grad_layers ? grad_eps.data_ptr<float>() : nullptr,
            need_grad_layers ? grad_sigma.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lfr.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lfi.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_ltr.data_ptr<float>() : nullptr,
            need_grad_field_in ? grad_lti.data_ptr<float>() : nullptr,
            need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
            need_grad_layers,
            need_grad_field_in,
            need_grad_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    auto emit = [](const at::Tensor& t, bool on) {
        return on ? pybind11::cast(t) : pybind11::object(pybind11::none());
    };
    out["grad_layer_thickness"] = emit(grad_thickness, need_grad_layers);
    out["grad_layer_eps_r"] = emit(grad_eps, need_grad_layers);
    out["grad_layer_sigma_e"] = emit(grad_sigma, need_grad_layers);
    out["grad_light_field_real"] = emit(grad_lfr, need_grad_field_in);
    out["grad_light_field_imag"] = emit(grad_lfi, need_grad_field_in);
    out["grad_light_throughput_real"] = emit(grad_ltr, need_grad_field_in);
    out["grad_light_throughput_imag"] = emit(grad_lti, need_grad_field_in);
    out["grad_frequency"] = emit(grad_frequency, need_grad_frequency);
    return out;
}

pybind11::dict cn_bdpt_transmitted_light_subpath_state_jvp(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_light_throughput_real,
    pybind11::object tangent_light_throughput_imag) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(face_material_id, "face_material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = light_direction.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t face_count = face_material_id.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    at::Tensor tt_s, te_s, ts_s, tlfr_s, tlfi_s, tltr_s, tlti_s;
    const at::Tensor* t_thick = optional_grad(
        std::move(tangent_layer_thickness), tt_s, "tangent_layer_thickness", at::kFloat, {layer_total}, light_direction);
    const at::Tensor* t_eps = optional_grad(
        std::move(tangent_layer_eps_r), te_s, "tangent_layer_eps_r", at::kFloat, {layer_total}, light_direction);
    const at::Tensor* t_sigma = optional_grad(
        std::move(tangent_layer_sigma_e), ts_s, "tangent_layer_sigma_e", at::kFloat, {layer_total}, light_direction);
    const at::Tensor* t_lfr = optional_grad(
        std::move(tangent_light_field_real), tlfr_s, "tangent_light_field_real", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* t_lfi = optional_grad(
        std::move(tangent_light_field_imag), tlfi_s, "tangent_light_field_imag", at::kFloat, {count, 3}, light_direction);
    const at::Tensor* t_ltr = optional_grad(
        std::move(tangent_light_throughput_real), tltr_s, "tangent_light_throughput_real", at::kFloat, {count}, light_direction);
    const at::Tensor* t_lti = optional_grad(
        std::move(tangent_light_throughput_imag), tlti_s, "tangent_light_throughput_imag", at::kFloat, {count}, light_direction);

    auto tangent_field_real = at::empty({count, 3}, light_direction.options());
    auto tangent_field_imag = at::empty({count, 3}, light_direction.options());
    auto tangent_throughput_real = at::empty({count}, light_direction.options());
    auto tangent_throughput_imag = at::empty({count}, light_direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_direction.get_device()).stream();
        transmitted_subpath_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_valid.data_ptr<bool>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            face_material_id.data_ptr<int>(),
            face_count,
            layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(),
            layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(),
            material_count,
            static_cast<float>(frequency_hz),
            grad_ptr<float>(t_thick),
            grad_ptr<float>(t_eps),
            grad_ptr<float>(t_sigma),
            static_cast<float>(tangent_frequency),
            grad_ptr<float>(t_lfr),
            grad_ptr<float>(t_lfi),
            grad_ptr<float>(t_ltr),
            grad_ptr<float>(t_lti),
            tangent_field_real.data_ptr<float>(),
            tangent_field_imag.data_ptr<float>(),
            tangent_throughput_real.data_ptr<float>(),
            tangent_throughput_imag.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_real"] = tangent_field_real;
    out["tangent_field_imag"] = tangent_field_imag;
    out["tangent_throughput_real"] = tangent_throughput_real;
    out["tangent_throughput_imag"] = tangent_throughput_imag;
    return out;
}
