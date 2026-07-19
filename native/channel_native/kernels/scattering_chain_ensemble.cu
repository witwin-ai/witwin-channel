// ADR-021 Op A: native multi-bounce Kirchhoff ensemble scattering row physics.
//
// Power-domain generalization of ADR-010 op 1 (scattering_ensemble.cu). Per
// joined chain row TX --C1(d1 reflections)--> v_s --C2(d2 reflections)--> RX:
//
//   1. C1 coherent Jones transport of the tx polarization to the vertex v_s,
//      yielding the incident coherency diagonal (P_te, P_tm) in the vertex s/p
//      basis of the last C1 leg (supervisor ruling: computed in-kernel from the
//      C1 transport of tx_pol, never a caller-supplied a_te2/a_tm2 pair).
//   2. Ensemble Kirchhoff table lookup (f_te, f_tm) at the vertex (op-1 shared
//      device interpolation, scattering_table.cuh::eval_te_tm).
//   3. Outgoing coherency J_out = diag(f_te*P_te, f_tm*P_tm).
//   4. C2 receiver responses (g_te2, g_tm2): |p_rx . A_2 s_o|^2 and
//      |p_rx . A_2 p_o|^2, the diagonal of A_2^H p_rx p_rx^H A_2 (op-1's
//      g_te2/g_tm2 when C2 is empty). Because J_out is diagonal the receiver
//      projection p^H J p reduces to f_te*P_te*g_te2 + f_tm*P_tm*g_tm2.
//   5. Radiometric assembly, op-1 association preserved.
//
// The per-bounce C1/C2 transport reuses the shared device primitives
// transport::reflect_frame / transport::slab_fresnel / transport::
// complex3_dot_real and the field:: complex helpers (field_transport.cuh),
// exactly as field_transport_reflection.cu::reflection_chain_eval; no device
// function is copied. Padded [R, Dmax, ...] leg blocks with per-row depths
// bound every loop (plan 10a section 1, Dmax = kMaxAdDepth = 8). Elementwise,
// no atomics: bitwise run-to-run deterministic (op-1 parity). Compiled with
// --fmad=false so the radiometric mul/add chain rounds like the Torch oracle.
//
// NOTE (interface reconciliation, see the change report): the committed float64
// oracle tests/reference/chain_ensemble.py and the existing native op-1
// convention drive the radiometric assembly (per-row `weights` = A_patch and
// 1/(L1^2 L2^2) spreading), which differs from the frozen plan-10a section 3.1
// argument sketch (sp1/sp2, no weights). Following the task rule "existing
// native op-1/op-2 conventions win", this kernel uses the op-1 weights + length
// convention. The endpoint positions source/vertex/target (required by the C1/C2
// transport, omitted from the section 3.1 sketch) are explicit arguments here,
// mirroring field_reflection_sequence.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../field_transport.cuh"
#include "../tensor_checks.h"
#include "scattering_table.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = witwin::channel::native_ext;
namespace transport = channel_native::field_transport;
namespace st = channel_native::scattering_tables;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

__device__ __forceinline__ field::float3a load3f(const float* p, int64_t i) {
    const int64_t b = i * 3;
    return field::make_f3(p[b], p[b + 1], p[b + 2]);
}

__device__ __forceinline__ field::float3a load_chain3f(
    const float* p, int64_t row, int bounce) {
    const int64_t b = (row * kMaxAdDepth + bounce) * 3;
    return field::make_f3(p[b], p[b + 1], p[b + 2]);
}

// Outgoing s/p basis at the vertex: s = normalize(n x d) with a grazing backup,
// p = s x d. Mirrors kirchhoff_ensemble._sp_basis and the op-1 native s_o/p_o
// construction (scattering_ensemble.cu lines 110-120).
__device__ __forceinline__ void sp_basis(
    field::float3a n, field::float3a d, field::float3a backup,
    field::float3a& s, field::float3a& p) {
    const field::float3a s_raw = field::f3_cross(n, d);
    const float sn = field::safe_length(s_raw);
    if (sn < 1.0e-6f) {
        s = backup;
    } else {
        const float inv = 1.0f / fmaxf(sn, 1.0e-12f);
        s = field::f3_mul(s_raw, inv);
    }
    p = field::f3_cross(s, d);
}

// C1/C2 specular Jones transport of a Complex3 field from `start` to `end`
// through `depth` reflections (mirrors reflection_chain_eval's bounce loop, no
// C_r attenuation and no propagation carrier: those live in the radiometric
// factors, op-1 parity). Returns the field arriving at `end` and, through
// `last_dir`, the propagation direction of the final leg (end - previous).
__device__ __forceinline__ field::Complex3 transport_leg(
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
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    last_dir = field::safe_normalize(field::f3_sub(end, previous), outgoing);
    return value;
}

__global__ void chain_ensemble_eval_kernel(
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
    float threshold,
    float frequency_hz,
    float* __restrict__ out_gain,
    float* __restrict__ out_amplitude,
    float* __restrict__ out_length,
    bool* __restrict__ out_keep) {
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

        // C1: transport the transmit field to the vertex. The tx polarization
        // is projected transverse to the first propagation direction (F1
        // unnormalized projection, op parity).
        const field::float3a first_target =
            d1 > 0 ? load_chain3f(c1_positions, row, 0) : vtx;
        const field::float3a first_leg = field::safe_normalize(
            field::f3_sub(first_target, src),
            field::make_f3(0.0f, 0.0f, 1.0f));
        const field::float3a tx_axis = field::project_to_wedge_plane(
            load3f(tx_pol, row), first_leg);
        field::Complex3 e_tx = field::cplx_scale_real(
            tx_axis, field::cplx(1.0f, 0.0f));
        field::float3a c1_last;
        const field::Complex3 e_in = transport_leg(
            e_tx, src, vtx, c1_positions, c1_normals, c1_eps_r, c1_sigma_e,
            c1_mu_r, c1_gain, c1_thickness, row, d1, frequency_hz, c1_last);

        // Incident coherency diagonal in the vertex s/p basis (last C1 leg d_i).
        field::float3a s_i;
        field::float3a p_i;
        sp_basis(n, di, backup, s_i, p_i);
        const float p_te = field::cplx_abs_sqr(
            transport::complex3_dot_real(e_in, s_i));
        const float p_tm = field::cplx_abs_sqr(
            transport::complex3_dot_real(e_in, p_i));

        // Vertex Kirchhoff table lookup. wi_local is the frozen incident table
        // axis (op-1 convention); wo_local is built in-kernel from the first C2
        // leg direction d_o (op-1's wo_local).
        const field::float3a t1 = load3f(t1r, row);
        const field::float3a t2 = load3f(t2r, row);
        const float co = cos_o[row];
        const float wo_local[3] = {
            field::f3_dot(dobj, t1), field::f3_dot(dobj, t2), co};
        float f_te = 0.0f;
        float f_tm = 0.0f;
        const int slot = material_slot[material_id[row]];
        if (slot >= 0) {
            const int64_t base = table_offset[slot];
            const int nti = table_dims[slot * 4 + 0];
            const int npi = table_dims[slot * 4 + 1];
            const int nto = table_dims[slot * 4 + 2];
            const int npo = table_dims[slot * 4 + 3];
            st::eval_te_tm(
                fte_flat + base, ftm_flat + base, nti, npi, nto, npo,
                wi_local + row * 3, wo_local, f_te, f_tm);
        }

        // C2 receiver responses: transport the two outgoing basis fields to the
        // receiver and project onto the transverse of p_rx at the final leg.
        field::float3a s_o;
        field::float3a p_o;
        sp_basis(n, dobj, backup, s_o, p_o);
        field::float3a c2_last;
        const field::Complex3 field_s = transport_leg(
            field::cplx_scale_real(s_o, field::cplx(1.0f, 0.0f)), vtx, tgt,
            c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, row, d2, frequency_hz, c2_last);
        field::float3a c2_last_p;
        const field::Complex3 field_p = transport_leg(
            field::cplx_scale_real(p_o, field::cplx(1.0f, 0.0f)), vtx, tgt,
            c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
            c2_thickness, row, d2, frequency_hz, c2_last_p);
        const field::float3a rx_axis = field::project_to_wedge_plane(
            load3f(rx_pol, row), c2_last);
        const float g_te2 = field::cplx_abs_sqr(
            transport::complex3_dot_real(field_s, rx_axis));
        const float g_tm2 = field::cplx_abs_sqr(
            transport::complex3_dot_real(field_p, rx_axis));

        // Receiver projection p^H J p (diagonal J_out) and radiometric gain
        // (op-1 association: scattering_ensemble.cu lines 131-140).
        const float f_eff = (f_te * p_te) * g_te2 + (f_tm * p_tm) * g_tm2;
        float num = coef * f_eff;
        num = num * cos_i[row];
        num = num * co;
        num = num * weights[row];
        const float len1 = l1[row];
        const float len2 = l2[row];
        const float den = (len1 * len1) * (len2 * len2);
        const float gain = num / den;

        out_gain[row] = gain;
        out_keep[row] = gain > threshold;
        out_amplitude[row] = sqrtf(fmaxf(gain, 0.0f));
        out_length[row] = len1 + len2;
    }
}

// Shared argument validation (forward + AD companions call this).
void check_chain_ensemble_primal(
    const at::Tensor& tx_pol,
    const at::Tensor& source,
    const at::Tensor& c1_positions,
    const at::Tensor& c1_depth,
    int64_t count) {
    using channel_native::check_vec3_table;
    using channel_native::check_flat_tensor;
    check_vec3_table(tx_pol, "tx_pol");
    check_vec3_table(source, "source");
    TORCH_CHECK(
        c1_positions.scalar_type() == at::kFloat &&
            c1_positions.is_contiguous() && c1_positions.dim() == 3 &&
            c1_positions.size(0) == count &&
            c1_positions.size(1) == kMaxAdDepth && c1_positions.size(2) == 3,
        "chain leg positions must be contiguous f32 (R, ", kMaxAdDepth, ", 3)");
    check_flat_tensor(c1_depth, "c1_depth", at::kInt);
    TORCH_CHECK(c1_depth.size(0) == count, "c1_depth must match R rows");
}

}  // namespace

pybind11::dict cn_scattering_chain_ensemble_eval(
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
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    const int64_t count = tx_pol.size(0);
    check_chain_ensemble_primal(tx_pol, source, c1_positions, c1_depth, count);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_flat_tensor(material_slot, "material_slot", at::kInt);
    check_flat_tensor(table_offset, "table_offset", at::kLong);
    check_tensor(table_dims, "table_dims", at::kInt, 2);
    check_flat_tensor(fte_flat, "fte_flat", at::kFloat);
    check_flat_tensor(ftm_flat, "ftm_flat", at::kFloat);
    TORCH_CHECK(table_dims.size(1) == 4, "table_dims must have shape (M, 4)");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    for (const auto& t : {rx_pol, vertex, target, n_o, t1r, t2r, backup_axis,
                          wi_local, d_i, d_o}) {
        check_vec3_table(t, "chain vec3");
        TORCH_CHECK(t.size(0) == count, "chain vec3 rows must match R");
    }
    for (const auto& t : {c1_normals, c2_positions, c2_normals}) {
        TORCH_CHECK(
            t.scalar_type() == at::kFloat && t.is_contiguous() &&
                t.dim() == 3 && t.size(0) == count &&
                t.size(1) == 8 && t.size(2) == 3,
            "chain leg block must be contiguous f32 (R, 8, 3)");
    }
    for (const auto& t : {cos_i, cos_o, l1, l2, weights}) {
        check_flat_tensor(t, "chain scalar", at::kFloat);
        TORCH_CHECK(t.size(0) == count, "chain scalar rows must match R");
    }
    for (const auto& t : {tx_pol, rx_pol, source, vertex, target, c1_positions,
                          c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
                          c1_thickness, c1_depth, c2_positions, c2_normals,
                          c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain, c2_thickness,
                          c2_depth, n_o, t1r, t2r, backup_axis, wi_local, cos_i,
                          cos_o, d_i, d_o, l1, l2, weights, material_id,
                          fte_flat, ftm_flat, table_offset, table_dims,
                          material_slot}) {
        TORCH_CHECK(t.get_device() == tx_pol.get_device(),
                    "chain ensemble tensors must share device");
    }
    auto gain = at::empty({count}, tx_pol.options());
    auto amplitude = at::empty({count}, tx_pol.options());
    auto length = at::empty({count}, tx_pol.options());
    auto keep = at::empty({count}, tx_pol.options().dtype(at::kBool));
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tx_pol.get_device()).stream();
        chain_ensemble_eval_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
            static_cast<float>(coef), static_cast<float>(threshold),
            static_cast<float>(frequency_hz),
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
