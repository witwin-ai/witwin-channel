// ADR-014 op 1: native JVP/VJP companions of the Kirchhoff ensemble scattering
// row physics (kernels/scattering_ensemble.cu). The forward is untouched; these
// kernels recompute every forward intermediate in the primal expression order
// (this TU compiles with --fmad=false so the recomputed values round exactly
// like the forward) and differentiate the live inputs.
//
// Backward: one thread per surviving row (same grid-stride loop as the forward).
// Per-row gradients (wo_rows, r2_rows, cos_o_rows) are direct stores; per-sample
// gradients (indexed by sc), the 16 table corners and the scalar coef gradient
// accumulate with atomicAdd into zero-initialised buffers (same run-to-run
// nondeterministic accumulation policy as the transmission-layer backward).
//
// JVP: elementwise, tangent-forward, no atomics; a missing tangent is a zero
// tangent. Both companions share the quadrilinear table derivative helper in
// scattering_table.cuh with the forward.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "scattering_table.cuh"

namespace {

constexpr int kBlockSize = 256;
namespace st = channel_native::scattering_tables;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

const at::Tensor* optional_arg(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

struct V3 { float x, y, z; };

__device__ __forceinline__ V3 load3(const float* __restrict__ p, int64_t i) {
    return {p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]};
}
// Match the forward's parallel-accumulator reduction (p0 + p2) + p1 exactly.
__device__ __forceinline__ float dot3(V3 a, V3 b) {
    const float p0 = a.x * b.x;
    const float p1 = a.y * b.y;
    const float p2 = a.z * b.z;
    return (p0 + p2) + p1;
}
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ V3 vadd(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ V3 vsub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ V3 vscale(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

// Shared forward recompute: fills the frame/table intermediates a row needs for
// both the VJP and JVP. Returns via out-params. ``tg.active`` reports whether a
// differentiable table lookup happened (slot present and above the horizon).
struct RowPrimal {
    int64_t s, c;
    V3 n, wo, t1, t2, s_o, p_o, pol_r, pol_r_perp;
    float r2, cos_o, cos_is, ws, r1s, a_te2s, a_tm2s;
    float f_te, f_tm, g_te, g_tm, g_te2, g_tm2, f_eff;
    float sn, prw, gain, amplitude, den;
    bool degen;
    int64_t table_base;
    st::TableEvalGrad tg;
};

__device__ __forceinline__ void recompute_row(
    int64_t row, float coef,
    const float* __restrict__ wo_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ cos_o_rows,
    const float* __restrict__ n_o,
    const float* __restrict__ t1r,
    const float* __restrict__ t2r,
    const float* __restrict__ wi_local,
    const float* __restrict__ cos_i,
    const float* __restrict__ r1,
    const float* __restrict__ a_te2,
    const float* __restrict__ a_tm2,
    const float* __restrict__ weights,
    const int* __restrict__ material_id,
    const float* __restrict__ backup_axis,
    const float* __restrict__ rx_pol,
    const int64_t* __restrict__ rc_idx,
    const int64_t* __restrict__ sc_idx,
    const float* __restrict__ fte_flat,
    const float* __restrict__ ftm_flat,
    const int64_t* __restrict__ table_offset,
    const int* __restrict__ table_dims,
    const int* __restrict__ material_slot,
    RowPrimal& p) {
    p.s = sc_idx[row];
    p.c = rc_idx[row];
    p.n = load3(n_o, p.s);
    p.wo = load3(wo_rows, row);
    p.r2 = r2_rows[row];
    p.cos_o = cos_o_rows[row];
    p.t1 = load3(t1r, p.s);
    p.t2 = load3(t2r, p.s);
    const float wo_local[3] = {dot3(p.wo, p.t1), dot3(p.wo, p.t2), p.cos_o};

    // Kirchhoff table lookup with derivative companions.
    p.f_te = 0.0f; p.f_tm = 0.0f;
    p.tg.active = false;
    p.table_base = 0;
    const int slot = material_slot[material_id[p.s]];
    if (slot >= 0) {
        p.table_base = table_offset[slot];
        const int nti = table_dims[slot * 4 + 0];
        const int npi = table_dims[slot * 4 + 1];
        const int nto = table_dims[slot * 4 + 2];
        const int npo = table_dims[slot * 4 + 3];
        st::eval_te_tm_grad(
            fte_flat + p.table_base, ftm_flat + p.table_base, nti, npi, nto, npo,
            wi_local + p.s * 3, wo_local, p.tg);
        p.f_te = p.tg.te;
        p.f_tm = p.tg.tm;
    }

    // Outgoing s/p basis: s_o = normalize(n x wo) with backup at grazing.
    const V3 s_raw = cross3(p.n, p.wo);
    p.sn = sqrtf(dot3(s_raw, s_raw));
    p.degen = p.sn < 1.0e-6f;
    if (p.degen) {
        p.s_o = load3(backup_axis, p.s);
    } else {
        const float d = fmaxf(p.sn, 1.0e-12f);
        p.s_o = {s_raw.x / d, s_raw.y / d, s_raw.z / d};
    }
    p.p_o = cross3(p.s_o, p.wo);

    // Receiver co-pol projections.
    p.pol_r = load3(rx_pol, p.c);
    p.prw = dot3(p.pol_r, p.wo);
    p.pol_r_perp = {p.pol_r.x - p.prw * p.wo.x,
                    p.pol_r.y - p.prw * p.wo.y,
                    p.pol_r.z - p.prw * p.wo.z};
    p.g_te = dot3(p.pol_r_perp, p.s_o);
    p.g_tm = dot3(p.pol_r_perp, p.p_o);
    p.g_te2 = p.g_te * p.g_te;
    p.g_tm2 = p.g_tm * p.g_tm;
    p.a_te2s = a_te2[p.s];
    p.a_tm2s = a_tm2[p.s];
    p.f_eff = (p.f_te * p.a_te2s) * p.g_te2 + (p.f_tm * p.a_tm2s) * p.g_tm2;

    // Radiometric gain (Torch association preserved).
    float num = coef * p.f_eff;
    num = num * cos_i[p.s];
    num = num * p.cos_o;
    num = num * weights[p.s];
    p.cos_is = cos_i[p.s];
    p.ws = weights[p.s];
    p.r1s = r1[p.s];
    p.den = (p.r1s * p.r1s) * (p.r2 * p.r2);
    p.gain = num / p.den;
    p.amplitude = sqrtf(fmaxf(p.gain, 0.0f));
}

__global__ void ensemble_eval_backward_kernel(
    int64_t count, float coef,
    const float* __restrict__ wo_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ cos_o_rows,
    const float* __restrict__ n_o,
    const float* __restrict__ t1r,
    const float* __restrict__ t2r,
    const float* __restrict__ wi_local,
    const float* __restrict__ cos_i,
    const float* __restrict__ r1,
    const float* __restrict__ a_te2,
    const float* __restrict__ a_tm2,
    const float* __restrict__ weights,
    const int* __restrict__ material_id,
    const float* __restrict__ backup_axis,
    const float* __restrict__ rx_pol,
    const int64_t* __restrict__ rc_idx,
    const int64_t* __restrict__ sc_idx,
    const float* __restrict__ fte_flat,
    const float* __restrict__ ftm_flat,
    const int64_t* __restrict__ table_offset,
    const int* __restrict__ table_dims,
    const int* __restrict__ material_slot,
    const float* __restrict__ grad_gain,
    const float* __restrict__ grad_amplitude,
    const float* __restrict__ grad_length,
    float* __restrict__ out_grad_wo_rows,
    float* __restrict__ out_grad_r2_rows,
    float* __restrict__ out_grad_cos_o_rows,
    float* __restrict__ out_grad_n_o,
    float* __restrict__ out_grad_t1r,
    float* __restrict__ out_grad_t2r,
    float* __restrict__ out_grad_wi_local,
    float* __restrict__ out_grad_cos_i,
    float* __restrict__ out_grad_r1,
    float* __restrict__ out_grad_a_te2,
    float* __restrict__ out_grad_a_tm2,
    float* __restrict__ out_grad_weights,
    float* __restrict__ out_grad_fte,
    float* __restrict__ out_grad_ftm,
    float* __restrict__ out_grad_coef,
    bool need_rows, bool need_samples, bool need_tables, bool need_coef) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        RowPrimal p;
        recompute_row(row, coef, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r,
                      wi_local, cos_i, r1, a_te2, a_tm2, weights, material_id,
                      backup_axis, rx_pol, rc_idx, sc_idx, fte_flat, ftm_flat,
                      table_offset, table_dims, material_slot, p);

        // Cotangent folding.
        const float gg = grad_gain != nullptr ? grad_gain[row] : 0.0f;
        const float ga = grad_amplitude != nullptr ? grad_amplitude[row] : 0.0f;
        const float gl = grad_length != nullptr ? grad_length[row] : 0.0f;
        const float gbar = gg + ga * (p.gain > 0.0f ? 0.5f / p.amplitude : 0.0f);
        const float lbar = gl;

        // Radiometric partials (each divides only by the strictly-positive den).
        float base = coef * p.cos_is; base *= p.cos_o; base *= p.ws; base /= p.den;
        float dg_dcos_i = coef * p.f_eff; dg_dcos_i *= p.cos_o; dg_dcos_i *= p.ws; dg_dcos_i /= p.den;
        float dg_dw = coef * p.f_eff; dg_dw *= p.cos_is; dg_dw *= p.cos_o; dg_dw /= p.den;
        float dg_dcos_o = coef * p.f_eff; dg_dcos_o *= p.cos_is; dg_dcos_o *= p.ws; dg_dcos_o /= p.den;
        float dg_dcoef = p.f_eff * p.cos_is; dg_dcoef *= p.cos_o; dg_dcoef *= p.ws; dg_dcoef /= p.den;
        const float dg_dr1 = -2.0f * p.gain / p.r1s;
        const float dg_dr2 = -2.0f * p.gain / p.r2;

        const float Sfeff = gbar * base;                       // gbar * d gain/d f_eff
        const float coeff_te = Sfeff * p.a_te2s * p.g_te2;     // d L/d f_te (table)
        const float coeff_tm = Sfeff * p.a_tm2s * p.g_tm2;     // d L/d f_tm (table)
        const float A_gte = Sfeff * 2.0f * p.f_te * p.a_te2s * p.g_te;  // d L/d g_te
        const float A_gtm = Sfeff * 2.0f * p.f_tm * p.a_tm2s * p.g_tm;  // d L/d g_tm

        // Reverse frame/projection chain.
        // g_te = pol_r_perp . s_o ; g_tm = pol_r_perp . p_o.
        V3 vp = vadd(vscale(p.s_o, A_gte), vscale(p.p_o, A_gtm));   // d L/d pol_r_perp
        V3 vs = vscale(p.pol_r_perp, A_gte);                       // d L/d s_o
        const V3 vpo = vscale(p.pol_r_perp, A_gtm);                // d L/d p_o
        // p_o = cross3(s_o, wo).
        vs = vadd(vs, cross3(p.wo, vpo));
        V3 g_wo = cross3(vpo, p.s_o);
        // pol_r_perp = pol_r - (pol_r.wo) wo.
        const float vp_dot_wo = dot3(vp, p.wo);
        g_wo = vadd(g_wo, vsub(vscale(vp, -p.prw), vscale(p.pol_r, vp_dot_wo)));
        // s_o = normalize(n x wo) (non-degenerate branch only).
        V3 grad_n = {0.0f, 0.0f, 0.0f};
        if (!p.degen) {
            const float s_dot = dot3(p.s_o, vs);
            const V3 proj = vsub(vs, vscale(p.s_o, s_dot));
            const V3 grad_s_raw = vscale(proj, 1.0f / p.sn);
            grad_n = cross3(p.wo, grad_s_raw);
            g_wo = vadd(g_wo, cross3(grad_s_raw, p.n));
        }

        // Table-coordinate contributions (combined te/tm coeffs).
        V3 Twi = {0.0f, 0.0f, 0.0f};
        V3 Two = {0.0f, 0.0f, 0.0f};
        if (p.tg.active) {
            Twi.x = coeff_te * p.tg.dte_dwi[0] + coeff_tm * p.tg.dtm_dwi[0];
            Twi.y = coeff_te * p.tg.dte_dwi[1] + coeff_tm * p.tg.dtm_dwi[1];
            Twi.z = coeff_te * p.tg.dte_dwi[2] + coeff_tm * p.tg.dtm_dwi[2];
            Two.x = coeff_te * p.tg.dte_dwo[0] + coeff_tm * p.tg.dtm_dwo[0];
            Two.y = coeff_te * p.tg.dte_dwo[1] + coeff_tm * p.tg.dtm_dwo[1];
            Two.z = coeff_te * p.tg.dte_dwo[2] + coeff_tm * p.tg.dtm_dwo[2];
        }
        // wo also receives the wo_local[0/1] table chain via t1/t2.
        g_wo = vadd(g_wo, vadd(vscale(p.t1, Two.x), vscale(p.t2, Two.y)));

        // cos_o: radiometric plus table via wo_local[2].
        const float R_cos_o = gbar * dg_dcos_o;
        const float grad_cos_o = R_cos_o + Two.z;

        if (need_rows) {
            out_grad_wo_rows[row * 3 + 0] = g_wo.x;
            out_grad_wo_rows[row * 3 + 1] = g_wo.y;
            out_grad_wo_rows[row * 3 + 2] = g_wo.z;
            out_grad_r2_rows[row] = gbar * dg_dr2 + lbar;
            out_grad_cos_o_rows[row] = grad_cos_o;
        }
        if (need_samples) {
            atomicAdd(&out_grad_n_o[p.s * 3 + 0], grad_n.x);
            atomicAdd(&out_grad_n_o[p.s * 3 + 1], grad_n.y);
            atomicAdd(&out_grad_n_o[p.s * 3 + 2], grad_n.z);
            atomicAdd(&out_grad_t1r[p.s * 3 + 0], Two.x * p.wo.x);
            atomicAdd(&out_grad_t1r[p.s * 3 + 1], Two.x * p.wo.y);
            atomicAdd(&out_grad_t1r[p.s * 3 + 2], Two.x * p.wo.z);
            atomicAdd(&out_grad_t2r[p.s * 3 + 0], Two.y * p.wo.x);
            atomicAdd(&out_grad_t2r[p.s * 3 + 1], Two.y * p.wo.y);
            atomicAdd(&out_grad_t2r[p.s * 3 + 2], Two.y * p.wo.z);
            atomicAdd(&out_grad_wi_local[p.s * 3 + 0], Twi.x);
            atomicAdd(&out_grad_wi_local[p.s * 3 + 1], Twi.y);
            atomicAdd(&out_grad_wi_local[p.s * 3 + 2], Twi.z);
            atomicAdd(&out_grad_cos_i[p.s], gbar * dg_dcos_i);
            atomicAdd(&out_grad_weights[p.s], gbar * dg_dw);
            atomicAdd(&out_grad_r1[p.s], gbar * dg_dr1 + lbar);
            atomicAdd(&out_grad_a_te2[p.s], Sfeff * p.f_te * p.g_te2);
            atomicAdd(&out_grad_a_tm2[p.s], Sfeff * p.f_tm * p.g_tm2);
        }
        if (need_tables && p.tg.active) {
            for (int k = 0; k < 16; ++k) {
                const int64_t off = p.table_base + p.tg.idx[k];
                atomicAdd(&out_grad_fte[off], coeff_te * p.tg.cw[k]);
                atomicAdd(&out_grad_ftm[off], coeff_tm * p.tg.cw[k]);
            }
        }
        if (need_coef) {
            atomicAdd(&out_grad_coef[0], gbar * dg_dcoef);
        }
    }
}

__global__ void ensemble_eval_jvp_kernel(
    int64_t count, float coef, float tangent_coef,
    const float* __restrict__ wo_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ cos_o_rows,
    const float* __restrict__ n_o,
    const float* __restrict__ t1r,
    const float* __restrict__ t2r,
    const float* __restrict__ wi_local,
    const float* __restrict__ cos_i,
    const float* __restrict__ r1,
    const float* __restrict__ a_te2,
    const float* __restrict__ a_tm2,
    const float* __restrict__ weights,
    const int* __restrict__ material_id,
    const float* __restrict__ backup_axis,
    const float* __restrict__ rx_pol,
    const int64_t* __restrict__ rc_idx,
    const int64_t* __restrict__ sc_idx,
    const float* __restrict__ fte_flat,
    const float* __restrict__ ftm_flat,
    const int64_t* __restrict__ table_offset,
    const int* __restrict__ table_dims,
    const int* __restrict__ material_slot,
    const float* __restrict__ t_wo_rows,
    const float* __restrict__ t_r2_rows,
    const float* __restrict__ t_cos_o_rows,
    const float* __restrict__ t_n_o,
    const float* __restrict__ t_t1r,
    const float* __restrict__ t_t2r,
    const float* __restrict__ t_wi_local,
    const float* __restrict__ t_cos_i,
    const float* __restrict__ t_r1,
    const float* __restrict__ t_a_te2,
    const float* __restrict__ t_a_tm2,
    const float* __restrict__ t_weights,
    const float* __restrict__ t_fte_flat,
    const float* __restrict__ t_ftm_flat,
    float* __restrict__ out_tangent_gain,
    float* __restrict__ out_tangent_amplitude,
    float* __restrict__ out_tangent_length) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        RowPrimal p;
        recompute_row(row, coef, wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r,
                      wi_local, cos_i, r1, a_te2, a_tm2, weights, material_id,
                      backup_axis, rx_pol, rc_idx, sc_idx, fte_flat, ftm_flat,
                      table_offset, table_dims, material_slot, p);

        // Live tangents (missing = zero).
        const V3 t_wo = t_wo_rows != nullptr ? load3(t_wo_rows, row) : V3{0.0f, 0.0f, 0.0f};
        const V3 t_n = t_n_o != nullptr ? load3(t_n_o, p.s) : V3{0.0f, 0.0f, 0.0f};
        const V3 t_t1 = t_t1r != nullptr ? load3(t_t1r, p.s) : V3{0.0f, 0.0f, 0.0f};
        const V3 t_t2 = t_t2r != nullptr ? load3(t_t2r, p.s) : V3{0.0f, 0.0f, 0.0f};
        const V3 t_wil = t_wi_local != nullptr ? load3(t_wi_local, p.s) : V3{0.0f, 0.0f, 0.0f};
        const float t_cos_o = t_cos_o_rows != nullptr ? t_cos_o_rows[row] : 0.0f;
        const float t_r2 = t_r2_rows != nullptr ? t_r2_rows[row] : 0.0f;
        const float t_r1v = t_r1 != nullptr ? t_r1[p.s] : 0.0f;
        const float t_ci = t_cos_i != nullptr ? t_cos_i[p.s] : 0.0f;
        const float t_wv = t_weights != nullptr ? t_weights[p.s] : 0.0f;
        const float t_ate = t_a_te2 != nullptr ? t_a_te2[p.s] : 0.0f;
        const float t_atm = t_a_tm2 != nullptr ? t_a_tm2[p.s] : 0.0f;

        // Tangent of wo_local = (wo.t1, wo.t2, cos_o).
        const V3 t_wol = {dot3(t_wo, p.t1) + dot3(p.wo, t_t1),
                          dot3(t_wo, p.t2) + dot3(p.wo, t_t2),
                          t_cos_o};

        // Tangent of the table values: coordinate chain plus table-value tangent.
        float t_fte = 0.0f, t_ftm = 0.0f;
        if (p.tg.active) {
            t_fte = dot3(V3{p.tg.dte_dwi[0], p.tg.dte_dwi[1], p.tg.dte_dwi[2]}, t_wil) +
                    dot3(V3{p.tg.dte_dwo[0], p.tg.dte_dwo[1], p.tg.dte_dwo[2]}, t_wol);
            t_ftm = dot3(V3{p.tg.dtm_dwi[0], p.tg.dtm_dwi[1], p.tg.dtm_dwi[2]}, t_wil) +
                    dot3(V3{p.tg.dtm_dwo[0], p.tg.dtm_dwo[1], p.tg.dtm_dwo[2]}, t_wol);
            if (t_fte_flat != nullptr) {
                for (int k = 0; k < 16; ++k)
                    t_fte += p.tg.cw[k] * t_fte_flat[p.table_base + p.tg.idx[k]];
            }
            if (t_ftm_flat != nullptr) {
                for (int k = 0; k < 16; ++k)
                    t_ftm += p.tg.cw[k] * t_ftm_flat[p.table_base + p.tg.idx[k]];
            }
        }

        // Tangent of the outgoing s/p basis.
        const V3 t_s_raw = vadd(cross3(t_n, p.wo), cross3(p.n, t_wo));
        V3 t_s_o = {0.0f, 0.0f, 0.0f};
        if (!p.degen) {
            const float s_dot = dot3(p.s_o, t_s_raw);
            t_s_o = vscale(vsub(t_s_raw, vscale(p.s_o, s_dot)), 1.0f / p.sn);
        }
        const V3 t_p_o = vadd(cross3(t_s_o, p.wo), cross3(p.s_o, t_wo));

        // Tangent of the receiver co-pol projection.
        const float t_prw = dot3(p.pol_r, t_wo);
        const V3 t_pol_perp = vsub(vscale(p.wo, -t_prw), vscale(t_wo, p.prw));
        const float t_g_te = dot3(t_pol_perp, p.s_o) + dot3(p.pol_r_perp, t_s_o);
        const float t_g_tm = dot3(t_pol_perp, p.p_o) + dot3(p.pol_r_perp, t_p_o);

        // Tangent of f_eff = f_te*a_te2*g_te^2 + f_tm*a_tm2*g_tm^2.
        const float t_feff =
            t_fte * p.a_te2s * p.g_te2 + p.f_te * t_ate * p.g_te2 +
            p.f_te * p.a_te2s * (2.0f * p.g_te * t_g_te) +
            t_ftm * p.a_tm2s * p.g_tm2 + p.f_tm * t_atm * p.g_tm2 +
            p.f_tm * p.a_tm2s * (2.0f * p.g_tm * t_g_tm);

        // Radiometric partials (division-free apart from the positive den).
        float base = coef * p.cos_is; base *= p.cos_o; base *= p.ws; base /= p.den;
        float dg_dcos_i = coef * p.f_eff; dg_dcos_i *= p.cos_o; dg_dcos_i *= p.ws; dg_dcos_i /= p.den;
        float dg_dw = coef * p.f_eff; dg_dw *= p.cos_is; dg_dw *= p.cos_o; dg_dw /= p.den;
        float dg_dcos_o = coef * p.f_eff; dg_dcos_o *= p.cos_is; dg_dcos_o *= p.ws; dg_dcos_o /= p.den;
        float dg_dcoef = p.f_eff * p.cos_is; dg_dcoef *= p.cos_o; dg_dcoef *= p.ws; dg_dcoef /= p.den;
        const float dg_dr1 = -2.0f * p.gain / p.r1s;
        const float dg_dr2 = -2.0f * p.gain / p.r2;

        float t_gain = base * t_feff;
        t_gain += dg_dcoef * tangent_coef;
        t_gain += dg_dcos_i * t_ci;
        t_gain += dg_dcos_o * t_cos_o;   // radiometric cos_o path (table path in t_feff)
        t_gain += dg_dw * t_wv;
        t_gain += dg_dr1 * t_r1v;
        t_gain += dg_dr2 * t_r2;

        out_tangent_gain[row] = t_gain;
        out_tangent_amplitude[row] =
            t_gain * (p.gain > 0.0f ? 0.5f / p.amplitude : 0.0f);
        out_tangent_length[row] = t_r1v + t_r2;
    }
}

// Mirror the forward entry's validation of the 22 primal tensors, returning the
// row/sample counts.
void check_ensemble_inputs(
    const at::Tensor& wo_rows, const at::Tensor& r2_rows, const at::Tensor& cos_o_rows,
    const at::Tensor& n_o, const at::Tensor& t1r, const at::Tensor& t2r,
    const at::Tensor& wi_local, const at::Tensor& cos_i, const at::Tensor& r1,
    const at::Tensor& a_te2, const at::Tensor& a_tm2, const at::Tensor& weights,
    const at::Tensor& material_id, const at::Tensor& backup_axis,
    const at::Tensor& rx_pol, const at::Tensor& rc_idx, const at::Tensor& sc_idx,
    const at::Tensor& fte_flat, const at::Tensor& ftm_flat,
    const at::Tensor& table_offset, const at::Tensor& table_dims,
    const at::Tensor& material_slot,
    int64_t& count, int64_t& samples) {
    using channel_native::check_tensor;
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(wo_rows, "wo_rows");
    count = wo_rows.size(0);
    check_flat_tensor(r2_rows, "r2_rows", at::kFloat);
    check_flat_tensor(cos_o_rows, "cos_o_rows", at::kFloat);
    check_vec3_table(n_o, "n_o");
    samples = n_o.size(0);
    check_vec3_table(t1r, "t1r");
    check_vec3_table(t2r, "t2r");
    check_vec3_table(wi_local, "wi_local");
    check_flat_tensor(cos_i, "cos_i", at::kFloat);
    check_flat_tensor(r1, "r1", at::kFloat);
    check_flat_tensor(a_te2, "a_te2", at::kFloat);
    check_flat_tensor(a_tm2, "a_tm2", at::kFloat);
    check_flat_tensor(weights, "weights", at::kFloat);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_vec3_table(backup_axis, "backup_axis");
    check_vec3_table(rx_pol, "rx_pol");
    check_flat_tensor(rc_idx, "rc_idx", at::kLong);
    check_flat_tensor(sc_idx, "sc_idx", at::kLong);
    check_flat_tensor(fte_flat, "fte_flat", at::kFloat);
    check_flat_tensor(ftm_flat, "ftm_flat", at::kFloat);
    check_flat_tensor(table_offset, "table_offset", at::kLong);
    check_tensor(table_dims, "table_dims", at::kInt, 2);
    check_flat_tensor(material_slot, "material_slot", at::kInt);
    TORCH_CHECK(
        r2_rows.size(0) == count && cos_o_rows.size(0) == count &&
            rc_idx.size(0) == count && sc_idx.size(0) == count,
        "per-row arrays must match wo_rows rows");
    TORCH_CHECK(
        t1r.size(0) == samples && t2r.size(0) == samples &&
            wi_local.size(0) == samples && cos_i.size(0) == samples &&
            r1.size(0) == samples && a_te2.size(0) == samples &&
            a_tm2.size(0) == samples && weights.size(0) == samples &&
            material_id.size(0) == samples && backup_axis.size(0) == samples,
        "per-sample arrays must match n_o rows");
    TORCH_CHECK(table_dims.size(1) == 4, "table_dims must have shape (M, 4)");
    for (const auto& t : {r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local, cos_i, r1,
                          a_te2, a_tm2, weights, material_id, backup_axis, rx_pol,
                          rc_idx, sc_idx, fte_flat, ftm_flat, table_offset,
                          table_dims, material_slot}) {
        TORCH_CHECK(t.get_device() == wo_rows.get_device(),
                    "ensemble tensors must share device");
    }
}

}  // namespace

pybind11::dict cn_scattering_ensemble_eval_backward(
    at::Tensor wo_rows,
    at::Tensor r2_rows,
    at::Tensor cos_o_rows,
    at::Tensor n_o,
    at::Tensor t1r,
    at::Tensor t2r,
    at::Tensor wi_local,
    at::Tensor cos_i,
    at::Tensor r1,
    at::Tensor a_te2,
    at::Tensor a_tm2,
    at::Tensor weights,
    at::Tensor material_id,
    at::Tensor backup_axis,
    at::Tensor rx_pol,
    at::Tensor rc_idx,
    at::Tensor sc_idx,
    at::Tensor fte_flat,
    at::Tensor ftm_flat,
    at::Tensor table_offset,
    at::Tensor table_dims,
    at::Tensor material_slot,
    double coef,
    double threshold,
    pybind11::object grad_gain,
    pybind11::object grad_amplitude,
    pybind11::object grad_length,
    bool need_grad_rows,
    bool need_grad_samples,
    bool need_grad_tables,
    bool need_grad_coef) {
    (void)threshold;  // topology (keep) is frozen non-differentiable.
    int64_t count = 0, samples = 0;
    check_ensemble_inputs(wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local,
                          cos_i, r1, a_te2, a_tm2, weights, material_id,
                          backup_axis, rx_pol, rc_idx, sc_idx, fte_flat, ftm_flat,
                          table_offset, table_dims, material_slot, count, samples);
    at::Tensor storage[3];
    const at::Tensor* g_gain = optional_arg(
        std::move(grad_gain), storage[0], "grad_gain", at::kFloat, {count}, wo_rows);
    const at::Tensor* g_amp = optional_arg(
        std::move(grad_amplitude), storage[1], "grad_amplitude", at::kFloat, {count}, wo_rows);
    const at::Tensor* g_len = optional_arg(
        std::move(grad_length), storage[2], "grad_length", at::kFloat, {count}, wo_rows);

    at::Tensor grad_wo_rows, grad_r2_rows, grad_cos_o_rows, grad_n_o, grad_t1r,
        grad_t2r, grad_wi_local, grad_cos_i, grad_r1, grad_a_te2, grad_a_tm2,
        grad_weights, grad_fte, grad_ftm, grad_coef;
    if (need_grad_rows) {
        grad_wo_rows = at::empty({count, 3}, wo_rows.options());
        grad_r2_rows = at::empty({count}, r2_rows.options());
        grad_cos_o_rows = at::empty({count}, cos_o_rows.options());
    }
    if (need_grad_samples) {
        grad_n_o = zero_filled({samples, 3}, n_o.options());
        grad_t1r = zero_filled({samples, 3}, t1r.options());
        grad_t2r = zero_filled({samples, 3}, t2r.options());
        grad_wi_local = zero_filled({samples, 3}, wi_local.options());
        grad_cos_i = zero_filled({samples}, cos_i.options());
        grad_r1 = zero_filled({samples}, r1.options());
        grad_a_te2 = zero_filled({samples}, a_te2.options());
        grad_a_tm2 = zero_filled({samples}, a_tm2.options());
        grad_weights = zero_filled({samples}, weights.options());
    }
    if (need_grad_tables) {
        grad_fte = zero_filled(fte_flat.sizes(), fte_flat.options());
        grad_ftm = zero_filled(ftm_flat.sizes(), ftm_flat.options());
    }
    if (need_grad_coef) {
        grad_coef = zero_filled({1}, wo_rows.options());
    }
    const bool any_grad = g_gain != nullptr || g_amp != nullptr || g_len != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(wo_rows.get_device()).stream();
        ensemble_eval_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, static_cast<float>(coef),
            wo_rows.data_ptr<float>(), r2_rows.data_ptr<float>(),
            cos_o_rows.data_ptr<float>(), n_o.data_ptr<float>(),
            t1r.data_ptr<float>(), t2r.data_ptr<float>(),
            wi_local.data_ptr<float>(), cos_i.data_ptr<float>(),
            r1.data_ptr<float>(), a_te2.data_ptr<float>(), a_tm2.data_ptr<float>(),
            weights.data_ptr<float>(), material_id.data_ptr<int>(),
            backup_axis.data_ptr<float>(), rx_pol.data_ptr<float>(),
            rc_idx.data_ptr<int64_t>(), sc_idx.data_ptr<int64_t>(),
            fte_flat.data_ptr<float>(), ftm_flat.data_ptr<float>(),
            table_offset.data_ptr<int64_t>(), table_dims.data_ptr<int>(),
            material_slot.data_ptr<int>(),
            opt_ptr<float>(g_gain), opt_ptr<float>(g_amp), opt_ptr<float>(g_len),
            need_grad_rows ? grad_wo_rows.data_ptr<float>() : nullptr,
            need_grad_rows ? grad_r2_rows.data_ptr<float>() : nullptr,
            need_grad_rows ? grad_cos_o_rows.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_n_o.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_t1r.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_t2r.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_wi_local.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_cos_i.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_r1.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_a_te2.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_a_tm2.data_ptr<float>() : nullptr,
            need_grad_samples ? grad_weights.data_ptr<float>() : nullptr,
            need_grad_tables ? grad_fte.data_ptr<float>() : nullptr,
            need_grad_tables ? grad_ftm.data_ptr<float>() : nullptr,
            need_grad_coef ? grad_coef.data_ptr<float>() : nullptr,
            need_grad_rows, need_grad_samples, need_grad_tables, need_grad_coef);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_wo_rows"] =
        need_grad_rows ? pybind11::cast(grad_wo_rows) : pybind11::object(pybind11::none());
    out["grad_r2_rows"] =
        need_grad_rows ? pybind11::cast(grad_r2_rows) : pybind11::object(pybind11::none());
    out["grad_cos_o_rows"] =
        need_grad_rows ? pybind11::cast(grad_cos_o_rows) : pybind11::object(pybind11::none());
    out["grad_n_o"] =
        need_grad_samples ? pybind11::cast(grad_n_o) : pybind11::object(pybind11::none());
    out["grad_t1r"] =
        need_grad_samples ? pybind11::cast(grad_t1r) : pybind11::object(pybind11::none());
    out["grad_t2r"] =
        need_grad_samples ? pybind11::cast(grad_t2r) : pybind11::object(pybind11::none());
    out["grad_wi_local"] =
        need_grad_samples ? pybind11::cast(grad_wi_local) : pybind11::object(pybind11::none());
    out["grad_cos_i"] =
        need_grad_samples ? pybind11::cast(grad_cos_i) : pybind11::object(pybind11::none());
    out["grad_r1"] =
        need_grad_samples ? pybind11::cast(grad_r1) : pybind11::object(pybind11::none());
    out["grad_a_te2"] =
        need_grad_samples ? pybind11::cast(grad_a_te2) : pybind11::object(pybind11::none());
    out["grad_a_tm2"] =
        need_grad_samples ? pybind11::cast(grad_a_tm2) : pybind11::object(pybind11::none());
    out["grad_weights"] =
        need_grad_samples ? pybind11::cast(grad_weights) : pybind11::object(pybind11::none());
    out["grad_f_te"] =
        need_grad_tables ? pybind11::cast(grad_fte) : pybind11::object(pybind11::none());
    out["grad_f_tm"] =
        need_grad_tables ? pybind11::cast(grad_ftm) : pybind11::object(pybind11::none());
    out["grad_coef"] =
        need_grad_coef ? pybind11::cast(grad_coef) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_scattering_ensemble_eval_jvp(
    at::Tensor wo_rows,
    at::Tensor r2_rows,
    at::Tensor cos_o_rows,
    at::Tensor n_o,
    at::Tensor t1r,
    at::Tensor t2r,
    at::Tensor wi_local,
    at::Tensor cos_i,
    at::Tensor r1,
    at::Tensor a_te2,
    at::Tensor a_tm2,
    at::Tensor weights,
    at::Tensor material_id,
    at::Tensor backup_axis,
    at::Tensor rx_pol,
    at::Tensor rc_idx,
    at::Tensor sc_idx,
    at::Tensor fte_flat,
    at::Tensor ftm_flat,
    at::Tensor table_offset,
    at::Tensor table_dims,
    at::Tensor material_slot,
    double coef,
    double threshold,
    pybind11::object t_wo_rows,
    pybind11::object t_r2_rows,
    pybind11::object t_cos_o_rows,
    pybind11::object t_n_o,
    pybind11::object t_t1r,
    pybind11::object t_t2r,
    pybind11::object t_wi_local,
    pybind11::object t_cos_i,
    pybind11::object t_r1,
    pybind11::object t_a_te2,
    pybind11::object t_a_tm2,
    pybind11::object t_weights,
    pybind11::object t_fte_flat,
    pybind11::object t_ftm_flat,
    double tangent_coef) {
    (void)threshold;  // topology (keep) is frozen non-differentiable.
    int64_t count = 0, samples = 0;
    check_ensemble_inputs(wo_rows, r2_rows, cos_o_rows, n_o, t1r, t2r, wi_local,
                          cos_i, r1, a_te2, a_tm2, weights, material_id,
                          backup_axis, rx_pol, rc_idx, sc_idx, fte_flat, ftm_flat,
                          table_offset, table_dims, material_slot, count, samples);
    at::Tensor storage[14];
    const at::Tensor* tw_wo = optional_arg(
        std::move(t_wo_rows), storage[0], "t_wo_rows", at::kFloat, {count, 3}, wo_rows);
    const at::Tensor* tw_r2 = optional_arg(
        std::move(t_r2_rows), storage[1], "t_r2_rows", at::kFloat, {count}, wo_rows);
    const at::Tensor* tw_cos_o = optional_arg(
        std::move(t_cos_o_rows), storage[2], "t_cos_o_rows", at::kFloat, {count}, wo_rows);
    const at::Tensor* tw_n = optional_arg(
        std::move(t_n_o), storage[3], "t_n_o", at::kFloat, {samples, 3}, wo_rows);
    const at::Tensor* tw_t1 = optional_arg(
        std::move(t_t1r), storage[4], "t_t1r", at::kFloat, {samples, 3}, wo_rows);
    const at::Tensor* tw_t2 = optional_arg(
        std::move(t_t2r), storage[5], "t_t2r", at::kFloat, {samples, 3}, wo_rows);
    const at::Tensor* tw_wil = optional_arg(
        std::move(t_wi_local), storage[6], "t_wi_local", at::kFloat, {samples, 3}, wo_rows);
    const at::Tensor* tw_ci = optional_arg(
        std::move(t_cos_i), storage[7], "t_cos_i", at::kFloat, {samples}, wo_rows);
    const at::Tensor* tw_r1 = optional_arg(
        std::move(t_r1), storage[8], "t_r1", at::kFloat, {samples}, wo_rows);
    const at::Tensor* tw_ate = optional_arg(
        std::move(t_a_te2), storage[9], "t_a_te2", at::kFloat, {samples}, wo_rows);
    const at::Tensor* tw_atm = optional_arg(
        std::move(t_a_tm2), storage[10], "t_a_tm2", at::kFloat, {samples}, wo_rows);
    const at::Tensor* tw_w = optional_arg(
        std::move(t_weights), storage[11], "t_weights", at::kFloat, {samples}, wo_rows);
    const at::Tensor* tw_fte = optional_arg(
        std::move(t_fte_flat), storage[12], "t_fte_flat", at::kFloat, fte_flat.sizes(), wo_rows);
    const at::Tensor* tw_ftm = optional_arg(
        std::move(t_ftm_flat), storage[13], "t_ftm_flat", at::kFloat, ftm_flat.sizes(), wo_rows);

    auto tangent_gain = at::empty({count}, r2_rows.options());
    auto tangent_amplitude = at::empty({count}, r2_rows.options());
    auto tangent_length = at::empty({count}, r2_rows.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(wo_rows.get_device()).stream();
        ensemble_eval_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, static_cast<float>(coef), static_cast<float>(tangent_coef),
            wo_rows.data_ptr<float>(), r2_rows.data_ptr<float>(),
            cos_o_rows.data_ptr<float>(), n_o.data_ptr<float>(),
            t1r.data_ptr<float>(), t2r.data_ptr<float>(),
            wi_local.data_ptr<float>(), cos_i.data_ptr<float>(),
            r1.data_ptr<float>(), a_te2.data_ptr<float>(), a_tm2.data_ptr<float>(),
            weights.data_ptr<float>(), material_id.data_ptr<int>(),
            backup_axis.data_ptr<float>(), rx_pol.data_ptr<float>(),
            rc_idx.data_ptr<int64_t>(), sc_idx.data_ptr<int64_t>(),
            fte_flat.data_ptr<float>(), ftm_flat.data_ptr<float>(),
            table_offset.data_ptr<int64_t>(), table_dims.data_ptr<int>(),
            material_slot.data_ptr<int>(),
            opt_ptr<float>(tw_wo), opt_ptr<float>(tw_r2), opt_ptr<float>(tw_cos_o),
            opt_ptr<float>(tw_n), opt_ptr<float>(tw_t1), opt_ptr<float>(tw_t2),
            opt_ptr<float>(tw_wil), opt_ptr<float>(tw_ci), opt_ptr<float>(tw_r1),
            opt_ptr<float>(tw_ate), opt_ptr<float>(tw_atm), opt_ptr<float>(tw_w),
            opt_ptr<float>(tw_fte), opt_ptr<float>(tw_ftm),
            tangent_gain.data_ptr<float>(), tangent_amplitude.data_ptr<float>(),
            tangent_length.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_gain"] = tangent_gain;
    out["tangent_amplitude"] = tangent_amplitude;
    out["tangent_length"] = tangent_length;
    return out;
}
