// ADR-010 op 1: native Kirchhoff ensemble scattering row physics.
//
// One launch per (tx, rx-chunk) replacing the Torch per-row physics of
// propagation/enumerated/scattering.py::_ensemble_rows between the RayD
// visibility calls. The candidate grid (to_rx/r2/wo/cos_o over [Rc, S]) stays
// Torch per the ADR; the surviving rows' wo/r2/cos_o are gathered from that
// grid and passed in (bitwise the values the previous Torch physics used, so
// the steep-lobe table interpolation sees identical weights). Per row the
// kernel builds wo_local, looks up the per-material Kirchhoff table (shared
// device interpolation from scattering_table.cuh), builds the outgoing s/p
// basis and receiver projections, and assembles the radiometric gain.
// Elementwise, no atomics: bitwise run-to-run deterministic. Compiled with
// --fmad=false so mul/add chains round exactly like Torch's per-op kernels.
// Expression order mirrors the Torch source (see
// docs/dev/audit/adr-010-expression-mapping.md).

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

struct V3 { float x, y, z; };

__device__ __forceinline__ V3 load3(const float* __restrict__ p, int64_t i) {
    return {p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]};
}
// Torch's batched (a*b).sum(-1) reduction over a 3-wide inner dim accumulates
// with two parallel accumulators: (p0 + p2) + p1. Replicated exactly (this TU
// compiles with --fmad=false, so each product/add rounds like the Torch
// per-op kernels).
__device__ __forceinline__ float dot3(V3 a, V3 b) {
    const float p0 = a.x * b.x;
    const float p1 = a.y * b.y;
    const float p2 = a.z * b.z;
    return (p0 + p2) + p1;
}
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__global__ void ensemble_eval_kernel(
    int64_t count,
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
    float coef, float threshold,
    float* __restrict__ out_gain,
    float* __restrict__ out_amplitude,
    float* __restrict__ out_length,
    bool* __restrict__ out_keep) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t s = sc_idx[row];
        const int64_t c = rc_idx[row];
        const V3 n = load3(n_o, s);
        const V3 wo = load3(wo_rows, row);
        const float r2 = r2_rows[row];
        const float cos_o = cos_o_rows[row];

        // wo_local = (wo.t1r, wo.t2r, cos_o).
        const V3 t1 = load3(t1r, s);
        const V3 t2 = load3(t2r, s);
        const float wo_local[3] = {dot3(wo, t1), dot3(wo, t2), cos_o};

        // Kirchhoff table lookup (shared device interpolation).
        float f_te = 0.0f, f_tm = 0.0f;
        const int slot = material_slot[material_id[s]];
        if (slot >= 0) {
            const int64_t base = table_offset[slot];
            const int nti = table_dims[slot * 4 + 0];
            const int npi = table_dims[slot * 4 + 1];
            const int nto = table_dims[slot * 4 + 2];
            const int npo = table_dims[slot * 4 + 3];
            st::eval_te_tm(
                fte_flat + base, ftm_flat + base, nti, npi, nto, npo,
                wi_local + s * 3, wo_local, f_te, f_tm);
        }

        // Outgoing s/p basis: s_o = normalize(n x wo) with backup at grazing.
        const V3 s_raw = cross3(n, wo);
        const float sn = sqrtf(dot3(s_raw, s_raw));
        V3 s_o;
        if (sn < 1.0e-6f) {
            s_o = load3(backup_axis, s);
        } else {
            const float d = fmaxf(sn, 1.0e-12f);
            s_o = {s_raw.x / d, s_raw.y / d, s_raw.z / d};
        }
        const V3 p_o = cross3(s_o, wo);

        // Receiver co-pol projections.
        const V3 pol_r = load3(rx_pol, c);
        const float prw = dot3(pol_r, wo);
        const V3 pol_r_perp = {
            pol_r.x - prw * wo.x, pol_r.y - prw * wo.y, pol_r.z - prw * wo.z};
        const float g_te = dot3(pol_r_perp, s_o);
        const float g_tm = dot3(pol_r_perp, p_o);
        const float g_te2 = g_te * g_te;
        const float g_tm2 = g_tm * g_tm;
        const float f_eff = (f_te * a_te2[s]) * g_te2 + (f_tm * a_tm2[s]) * g_tm2;

        // Radiometric gain (Torch association preserved).
        float num = coef * f_eff;
        num = num * cos_i[s];
        num = num * cos_o;
        num = num * weights[s];
        const float r1s = r1[s];
        const float den = (r1s * r1s) * (r2 * r2);
        const float gain = num / den;

        out_gain[row] = gain;
        out_keep[row] = gain > threshold;
        out_amplitude[row] = sqrtf(fmaxf(gain, 0.0f));
        out_length[row] = r1s + r2;
    }
}

}  // namespace

pybind11::dict cn_scattering_ensemble_eval(
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
    double threshold) {
    using channel_native::check_tensor;
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(wo_rows, "wo_rows");
    const int64_t count = wo_rows.size(0);
    check_flat_tensor(r2_rows, "r2_rows", at::kFloat);
    check_flat_tensor(cos_o_rows, "cos_o_rows", at::kFloat);
    check_vec3_table(n_o, "n_o");
    const int64_t samples = n_o.size(0);
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
    auto gain = at::empty({count}, r2_rows.options());
    auto amplitude = at::empty({count}, r2_rows.options());
    auto length = at::empty({count}, r2_rows.options());
    auto keep = at::empty({count}, r2_rows.options().dtype(at::kBool));
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(wo_rows.get_device()).stream();
        ensemble_eval_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
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
            static_cast<float>(coef), static_cast<float>(threshold),
            gain.data_ptr<float>(), amplitude.data_ptr<float>(),
            length.data_ptr<float>(), keep.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["gain"] = gain;
    out["amplitude"] = amplitude;
    out["length"] = length;
    out["keep"] = keep;
    return out;
}
