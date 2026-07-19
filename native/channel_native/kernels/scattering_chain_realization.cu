// ADR-021 Op B: native multi-bounce coherent phase-screen chain realization.
//
// Coherent (fully polarimetric) generalization of ADR-010 op 2
// (scattering_patch_integral.cu). Per joined chain row the vertex sits on a
// phase-screen patch; a specular reflection chain C1 transports the transmit
// field to the vertex, the vertex applies the layer-stack reflection Jones
// operator diag(r_te, r_tm) in the local s/p basis, and a specular reflection
// chain C2 transports the outgoing field to the receiver:
//
//     E_rx = A_2 . S_patch(d_i, d_o; h) . A_1 . e_tx
//
// with the carrier exp(-j k0 (L1 + L2)) over the image-unfolded lengths, the
// planar-chain spreading sp1*sp2 (= 1/(L1 L2) for planar image chains), the
// r_te/r_tm computed IN-KERNEL from the resident CSR layer stack at the local
// specular cosine (no separate em_layer_stack launch, ADR-009 fusion boundary
// is the complete row), and the same Duffy-mapped 16x16 Gauss-Legendre patch
// quadrature and two-stage fixed-order tree reduction as op 2.
//
// The scalar Jones response E_rx replaces op 2's caller-supplied
// jones = r_te*(a_te*g_te) + r_tm*(a_tm*g_tm) scalar. The degenerate
// d1 = d2 = 0 row (empty chains, A_1 = A_2 = I) collapses symbol-for-symbol to
// op 2 (lockstep-pinned, NOT dispatched in production). Every per-bounce
// specular event reuses field_transport.cuh reflect_frame / slab_fresnel /
// complex3_dot_real exactly like field_transport_reflection.cu; the vertex
// stack reuses em::stack_rt.
//
// Phase convention (module docstring of propagation/enumerated/scattering.py):
// physical q = k0*(d_o - d_i); the aperture integral evaluates the swapped
// integrand exp(-j*(q_int . x + q_int_n * h)) with q_int = -q against each
// patch triangle winding normal; the leftover absolute-position phase is
// removed by the carrier's q . centroid term.
//
// No float atomics: stage 1 is one block per row (256-node shared tree
// reduction) and stage 2 tree-reduces the row values into the 0-dim total in a
// fixed order, so total / path_field / path_gain are bitwise stable
// run-to-run. Compiled --fmad=false (lockstep with op 2).
//
// NOTE (see the plan 10a section 4 gap reported by the owner): the chain
// transport requires the absolute endpoint positions source (tx), vertex
// (v_s) and target (rx) to reconstruct the per-bounce incident directions,
// exactly as field_transport_reflection.cu takes source/target. The frozen
// section 4.1 facade table omits them; this bridge adds source/vertex/target
// right after n_rows, matching the reflection-kernel precedent.

#include "field_transport_ad_common.cuh"

namespace {

constexpr int kQuadPoints = 256;  // 16 x 16 Duffy-mapped Gauss-Legendre nodes
constexpr int kReduceBlock = 256;
constexpr float kPi = 3.14159265358979323846f;

using cfloat = c10::complex<float>;

// Per-bounce (index, bounce) load with the fixed kMaxAdDepth padding stride.
__device__ __forceinline__ field::float3a load_leg3(
    const float* __restrict__ values, int64_t row, int bounce) {
    return load_sequence3f(values, row, bounce, kMaxAdDepth);
}

// PhaseScreenRuntime.sample_height: bilinear with half-texel edge clamp
// (verbatim op-2 forward).
__device__ __forceinline__ float sample_height(
    const float* __restrict__ heights, int h_rows, int w_cols, float u, float v) {
    const float tx = fminf(fmaxf(u * w_cols - 0.5f, 0.0f), static_cast<float>(w_cols - 1));
    const float ty = fminf(fmaxf(v * h_rows - 0.5f, 0.0f), static_cast<float>(h_rows - 1));
    const float x0 = floorf(tx);
    const float y0 = floorf(ty);
    const float wx = tx - x0;
    const float wy = ty - y0;
    const int ix0 = static_cast<int>(x0);
    const int iy0 = static_cast<int>(y0);
    const int ix1 = min(ix0 + 1, w_cols - 1);
    const int iy1 = min(iy0 + 1, h_rows - 1);
    const float t00 = heights[iy0 * w_cols + ix0];
    const float t01 = heights[iy0 * w_cols + ix1];
    const float t10 = heights[iy1 * w_cols + ix0];
    const float t11 = heights[iy1 * w_cols + ix1];
    const float top = t00 * (1.0f - wx) + t01 * wx;
    const float bot = t10 * (1.0f - wx) + t11 * wx;
    return top * (1.0f - wy) + bot * wy;
}

// _stable_tangent: one-hot at the FIRST smallest |component| of n,
// Gram-Schmidt against n, normalized (verbatim op-2 forward).
__device__ __forceinline__ field::float3a stable_tangent(field::float3a n) {
    const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    field::float3a axis = field::make_f3(0.0f, 0.0f, 0.0f);
    if (ax <= ay && ax <= az) axis.x = 1.0f;
    else if (ay <= az) axis.y = 1.0f;
    else axis.z = 1.0f;
    const float proj = field::f3_dot(axis, n);
    field::float3a t = field::make_f3(
        axis.x - proj * n.x, axis.y - proj * n.y, axis.z - proj * n.z);
    const float norm = fmaxf(sqrtf(field::f3_dot(t, t)), 1.0e-12f);
    return field::f3_mul(t, 1.0f / norm);
}

// _sp_basis: s = normalize(n x d) with the deterministic backup axis at normal
// incidence; p = s x d (verbatim op-2 forward).
__device__ __forceinline__ void sp_basis(
    field::float3a n, field::float3a d, field::float3a backup,
    field::float3a& s, field::float3a& p) {
    const field::float3a raw = field::f3_cross(n, d);
    const float norm = sqrtf(field::f3_dot(raw, raw));
    if (norm < 1.0e-6f)
        s = backup;
    else
        s = field::f3_mul(raw, 1.0f / fmaxf(norm, 1.0e-12f));
    p = field::f3_cross(s, d);
}

// One specular leg (C1: tx -> vertex, or C2: vertex -> rx). Transports the
// Complex3 field from start to end through the padded per-bounce blocks,
// mirroring reflection_chain_eval's bounce loop (frame -> slab Fresnel -> s/p
// decomposition -> field update). Returns the field arriving at end and the
// final propagation direction (end - previous), i.e. the incident/outgoing
// direction the receiver projection needs. No C_r rough attenuation (native
// reflection-kernel convention; see the ensemble oracle's C_r discrepancy).
__device__ __forceinline__ field::Complex3 transport_leg(
    field::Complex3 value,
    field::float3a start,
    field::float3a end,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ eps_r,
    const float* __restrict__ sigma_e,
    const float* __restrict__ mu_r,
    const float* __restrict__ gain,
    const float* __restrict__ thickness,
    int64_t row,
    int depth,
    float frequency_hz,
    field::float3a& last_dir) {
    const field::float3a ez = field::make_f3(0.0f, 0.0f, 1.0f);
    field::float3a previous = start;
    const field::float3a first_hit = depth > 0 ? load_leg3(positions, row, 0) : end;
    field::float3a outgoing = field::safe_normalize(
        field::f3_sub(first_hit, start), ez);
    for (int bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_leg3(positions, row, bounce);
        const field::float3a incident = field::safe_normalize(
            field::f3_sub(hit, previous), outgoing);
        const int64_t slot = row * kMaxAdDepth + bounce;
        const transport::ReflectFrame frame = transport::reflect_frame(
            incident, load_leg3(normals, row, bounce));
        field::Complex r_te;
        field::Complex r_tm;
        transport::slab_fresnel(
            frame.cos_theta, eps_r[slot], sigma_e[slot], mu_r[slot], gain[slot],
            thickness[slot], frequency_hz, r_te, r_tm);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_in);
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    last_dir = field::safe_normalize(field::f3_sub(end, previous), outgoing);
    return value;
}

// Full vertex Jones scalar E_rx = <A_2 S A_1 e_tx, rx_axis>. Recomputed
// identically by the backward/jvp companions.
__device__ __forceinline__ field::Complex chain_jones(
    int64_t row,
    int d1,
    int d2,
    field::float3a n,
    field::float3a di,
    field::float3a dov,
    field::float3a source,
    field::float3a vertex,
    field::float3a target,
    field::float3a tx_pol,
    field::float3a rx_pol,
    const float* c1_positions, const float* c1_normals, const float* c1_eps_r,
    const float* c1_sigma_e, const float* c1_mu_r, const float* c1_gain,
    const float* c1_thickness,
    const float* c2_positions, const float* c2_normals, const float* c2_eps_r,
    const float* c2_sigma_e, const float* c2_mu_r, const float* c2_gain,
    const float* c2_thickness,
    const em::LayerView& layers,
    float cos_spec,
    float frequency_hz) {
    const field::float3a ez = field::make_f3(0.0f, 0.0f, 1.0f);
    const field::float3a backup = stable_tangent(n);
    field::float3a s_i, p_i, s_o, p_o;
    sp_basis(n, di, backup, s_i, p_i);
    sp_basis(n, dov, backup, s_o, p_o);

    // A_1 e_tx: transverse projection of the transmit polarization onto the
    // first C1 leg direction, then C1 transport to the vertex.
    const field::float3a first_hit1 = d1 > 0 ? load_leg3(c1_positions, row, 0) : vertex;
    const field::float3a incident_pre = field::safe_normalize(
        field::f3_sub(first_hit1, source), ez);
    const field::float3a tx_axis = field::project_to_wedge_plane(tx_pol, incident_pre);
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    field::float3a dump;
    const field::Complex3 e_in = transport_leg(
        value, source, vertex, c1_positions, c1_normals, c1_eps_r, c1_sigma_e,
        c1_mu_r, c1_gain, c1_thickness, row, d1, frequency_hz, dump);
    const field::Complex e_s_in = transport::complex3_dot_real(e_in, s_i);
    const field::Complex e_p_in = transport::complex3_dot_real(e_in, p_i);

    // Vertex layer-stack reflection Jones diag(r_te, r_tm) at cos_spec.
    const em::StackRT te = em::stack_rt(cos_spec, layers, frequency_hz, em::kPolTE);
    const em::StackRT tm = em::stack_rt(cos_spec, layers, frequency_hz, em::kPolTM);
    const field::Complex3 e_out = field::c3_add(
        field::cplx_scale_real(s_o, field::cplx_mul(te.r, e_s_in)),
        field::cplx_scale_real(p_o, field::cplx_mul(tm.r, e_p_in)));

    // A_2: C2 transport to the receiver, then projection onto the rx axis.
    field::float3a last_dir;
    const field::Complex3 e_rx_field = transport_leg(
        e_out, vertex, target, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
        c2_mu_r, c2_gain, c2_thickness, row, d2, frequency_hz, last_dir);
    const field::float3a rx_axis = field::project_to_wedge_plane(rx_pol, last_dir);
    return transport::complex3_dot_real(e_rx_field, rx_axis);
}

__global__ void chain_realization_rows_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris,
    const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
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
    const float* __restrict__ tx_pol,
    const float* __restrict__ rx_pol,
    const float* __restrict__ l1_rows,
    const float* __restrict__ l2_rows,
    const float* __restrict__ sp1_rows,
    const float* __restrict__ sp2_rows,
    const float* __restrict__ centroids,
    const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ cos_spec,
    const int* __restrict__ material_id,
    const int* __restrict__ layer_offset,
    const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m,
    const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e,
    const float* __restrict__ layer_mu_r,
    const float* __restrict__ quad_a,
    const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0,
    float frequency_hz,
    cfloat* __restrict__ out_integral,
    cfloat* __restrict__ out_row_value,
    cfloat* __restrict__ out_path_field,
    float* __restrict__ out_path_gain) {
    __shared__ float sh_re[kQuadPoints];
    __shared__ float sh_im[kQuadPoints];
    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    // Triangle frame (patch_phase_integral): edges, winding normal, area.
    const field::float3a p0 = load3f(patch_tris, patch * 3 + 0);
    const field::float3a p1 = load3f(patch_tris, patch * 3 + 1);
    const field::float3a p2 = load3f(patch_tris, patch * 3 + 2);
    const field::float3a e1 = field::f3_sub(p1, p0);
    const field::float3a e2 = field::f3_sub(p2, p0);
    const field::float3a winding = field::f3_cross(e1, e2);
    const float double_area = sqrtf(field::f3_dot(winding, winding));
    const field::float3a n_hat = field::f3_mul(winding, 1.0f / fmaxf(double_area, 1.0e-30f));

    const field::float3a di = load3f(d_i, row);
    const field::float3a dov = load3f(d_o, row);
    const field::float3a kiv = field::f3_mul(di, k0);
    const field::float3a ksv = field::f3_mul(dov, k0);
    const field::float3a q = field::f3_sub(ksv, kiv);
    const field::float3a q_int = field::f3_sub(kiv, ksv);
    const float q_int_n = field::f3_dot(n_hat, q_int);

    // Quadrature node phasor (one node per thread).
    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const field::float3a pos = field::make_f3(
        p0.x + a * e1.x + b * e2.x,
        p0.y + a * e1.y + b * e2.y,
        p0.z + a * e1.z + b * e2.z);
    const float u0 = patch_uvs[(patch * 3 + 0) * 2 + 0];
    const float v0 = patch_uvs[(patch * 3 + 0) * 2 + 1];
    const float u1 = patch_uvs[(patch * 3 + 1) * 2 + 0];
    const float v1 = patch_uvs[(patch * 3 + 1) * 2 + 1];
    const float u2 = patch_uvs[(patch * 3 + 2) * 2 + 0];
    const float v2 = patch_uvs[(patch * 3 + 2) * 2 + 1];
    const float uu = u0 + a * (u1 - u0) + b * (u2 - u0);
    const float vv = v0 + a * (v1 - v0) + b * (v2 - v0);
    const float h = sample_height(heights, h_rows_dim, w_cols_dim, uu, vv);
    const float phase = field::f3_dot(pos, q_int) + q_int_n * h;
    float c, s;
    sincosf(-phase, &s, &c);
    sh_re[t] = c * w;
    sh_im[t] = s * w;
    __syncthreads();

#pragma unroll
    for (int stride = kQuadPoints / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_re[t] += sh_re[t + stride];
            sh_im[t] += sh_im[t + stride];
        }
        __syncthreads();
    }
    if (t != 0) return;

    const cfloat integral = cfloat(sh_re[0] * double_area, sh_im[0] * double_area);
    out_integral[row] = integral;

    // Row coefficient: (j*pref) * E_rx * carrier * (sp1*sp2), then * integral.
    const field::float3a n = load3f(n_rows, row);
    const float q_norm2 = field::f3_dot(q, q);
    const float q_n = fmaxf(field::f3_dot(q, n), 1.0e-9f);
    const float pref_im = k0 * (q_norm2 / (k0 * q_n)) / (4.0f * kPi);

    em::LayerView layers{
        layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, material_id[row]};
    const field::Complex e_rx = chain_jones(
        row, c1_depth[row], c2_depth[row], n, di, dov,
        load3f(source, row), load3f(vertex, row), load3f(target, row),
        load3f(tx_pol, row), load3f(rx_pol, row),
        c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness,
        c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness,
        layers, cos_spec[row], frequency_hz);

    const float l1v = l1_rows[row];
    const float l2v = l2_rows[row];
    const float sp = sp1_rows[row] * sp2_rows[row];
    const field::float3a c_row = load3f(centroids, row);
    const float carrier_phase = -(k0 * (l1v + l2v) + field::f3_dot(q, c_row));
    float cc, cs;
    sincosf(carrier_phase, &cs, &cc);

    // value = (j*pref) * E_rx * carrier * sp.
    field::Complex value = field::cplx(0.0f, pref_im);           // j*pref
    value = field::cplx_mul(value, e_rx);
    value = field::cplx_mul(value, field::cplx(cc, cs));         // * carrier
    value = field::cplx_mul_real(value, sp);                     // * sp1*sp2

    const field::Complex integral_f = field::cplx(integral.real(), integral.imag());
    const field::Complex row_value = field::cplx_mul(value, integral_f);
    const cfloat rv = cfloat(row_value.re, row_value.im);
    out_row_value[row] = rv;
    out_path_field[row] = rv;
    out_path_gain[row] = row_value.re * row_value.re + row_value.im * row_value.im;
}

__global__ void chain_realization_total_kernel(
    int64_t row_count,
    const cfloat* __restrict__ row_values,
    cfloat* __restrict__ out_total) {
    __shared__ float sh_re[kReduceBlock];
    __shared__ float sh_im[kReduceBlock];
    const int t = threadIdx.x;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int64_t index = t; index < row_count; index += kReduceBlock) {
        const cfloat value = row_values[index];
        acc_re += value.real();
        acc_im += value.imag();
    }
    sh_re[t] = acc_re;
    sh_im[t] = acc_im;
    __syncthreads();
#pragma unroll
    for (int stride = kReduceBlock / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_re[t] += sh_re[t + stride];
            sh_im[t] += sh_im[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) out_total[0] = cfloat(sh_re[0], sh_im[0]);
}

// Shared validation of the forward positional inputs (also reused by the AD
// companions in scattering_chain_realization_ad.cu via the exported helper).
void check_leg_block(
    const at::Tensor& positions, const at::Tensor& normals,
    const at::Tensor& eps_r, const at::Tensor& sigma_e, const at::Tensor& mu_r,
    const at::Tensor& gain, const at::Tensor& thickness, const at::Tensor& depth,
    int64_t row_count, const char* tag) {
    using channel_native::check_tensor;
    using channel_native::check_flat_tensor;
    TORCH_CHECK(positions.scalar_type() == at::kFloat && positions.is_contiguous() &&
                    positions.dim() == 3 && positions.size(0) == row_count &&
                    positions.size(1) == kMaxAdDepth && positions.size(2) == 3,
                tag, " positions must be contiguous f32 (R, kMaxAdDepth, 3)");
    TORCH_CHECK(normals.sizes() == positions.sizes() && normals.is_contiguous() &&
                    normals.scalar_type() == at::kFloat,
                tag, " normals must match positions");
    for (const auto& m : {eps_r, sigma_e, mu_r, gain, thickness})
        TORCH_CHECK(m.scalar_type() == at::kFloat && m.is_contiguous() &&
                        m.dim() == 2 && m.size(0) == row_count &&
                        m.size(1) == kMaxAdDepth,
                    tag, " material tensors must be contiguous f32 (R, kMaxAdDepth)");
    TORCH_CHECK(depth.scalar_type() == at::kInt && depth.is_contiguous() &&
                    depth.dim() == 1 && depth.size(0) == row_count,
                tag, " depth must be contiguous int32 (R,)");
}

}  // namespace

// Exposed for the AD companions (same-family validation reuse).
int64_t cn_scattering_chain_realization_check(
    const at::Tensor& patch_tris, const at::Tensor& patch_uvs,
    const at::Tensor& rows, const at::Tensor& d_i, const at::Tensor& d_o,
    const at::Tensor& n_rows, const at::Tensor& source, const at::Tensor& vertex,
    const at::Tensor& target,
    const at::Tensor& c1_positions, const at::Tensor& c1_normals,
    const at::Tensor& c1_eps_r, const at::Tensor& c1_sigma_e,
    const at::Tensor& c1_mu_r, const at::Tensor& c1_gain,
    const at::Tensor& c1_thickness, const at::Tensor& c1_depth,
    const at::Tensor& c2_positions, const at::Tensor& c2_normals,
    const at::Tensor& c2_eps_r, const at::Tensor& c2_sigma_e,
    const at::Tensor& c2_mu_r, const at::Tensor& c2_gain,
    const at::Tensor& c2_thickness, const at::Tensor& c2_depth,
    const at::Tensor& tx_pol, const at::Tensor& rx_pol, const at::Tensor& l1_rows,
    const at::Tensor& l2_rows, const at::Tensor& sp1_rows,
    const at::Tensor& sp2_rows, const at::Tensor& centroids,
    const at::Tensor& heights, const at::Tensor& cos_spec,
    const at::Tensor& material_id, const at::Tensor& layer_offset,
    const at::Tensor& layer_count, const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r, const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r, const at::Tensor& quad_a,
    const at::Tensor& quad_b, const at::Tensor& quad_w) {
    using channel_native::check_tensor;
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_tensor(patch_tris, "patch_tris", at::kFloat, 3);
    TORCH_CHECK(patch_tris.size(1) == 3 && patch_tris.size(2) == 3,
                "patch_tris must have shape (P, 3, 3)");
    check_tensor(patch_uvs, "patch_uvs", at::kFloat, 3);
    TORCH_CHECK(patch_uvs.size(0) == patch_tris.size(0) && patch_uvs.size(1) == 3 &&
                    patch_uvs.size(2) == 2,
                "patch_uvs must have shape (P, 3, 2)");
    check_flat_tensor(rows, "rows", at::kLong);
    const int64_t row_count = rows.size(0);
    check_vec3_table(d_i, "d_i");
    check_vec3_table(d_o, "d_o");
    check_vec3_table(n_rows, "n_rows");
    check_vec3_table(source, "source");
    check_vec3_table(vertex, "vertex");
    check_vec3_table(target, "target");
    check_leg_block(c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r,
                    c1_gain, c1_thickness, c1_depth, row_count, "c1");
    check_leg_block(c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r,
                    c2_gain, c2_thickness, c2_depth, row_count, "c2");
    check_vec3_table(tx_pol, "tx_pol");
    check_vec3_table(rx_pol, "rx_pol");
    check_flat_tensor(l1_rows, "l1_rows", at::kFloat);
    check_flat_tensor(l2_rows, "l2_rows", at::kFloat);
    check_flat_tensor(sp1_rows, "sp1_rows", at::kFloat);
    check_flat_tensor(sp2_rows, "sp2_rows", at::kFloat);
    check_vec3_table(centroids, "centroids");
    check_tensor(heights, "heights", at::kFloat, 2);
    check_flat_tensor(cos_spec, "cos_spec", at::kFloat);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    check_flat_tensor(quad_a, "quad_a", at::kFloat);
    check_flat_tensor(quad_b, "quad_b", at::kFloat);
    check_flat_tensor(quad_w, "quad_w", at::kFloat);
    TORCH_CHECK(quad_a.size(0) == kQuadPoints && quad_b.size(0) == kQuadPoints &&
                    quad_w.size(0) == kQuadPoints,
                "quadrature arrays must hold 16x16 Duffy-mapped nodes");
    TORCH_CHECK(d_i.size(0) == row_count && d_o.size(0) == row_count &&
                    n_rows.size(0) == row_count && source.size(0) == row_count &&
                    vertex.size(0) == row_count && target.size(0) == row_count &&
                    tx_pol.size(0) == row_count && rx_pol.size(0) == row_count &&
                    l1_rows.size(0) == row_count && l2_rows.size(0) == row_count &&
                    sp1_rows.size(0) == row_count && sp2_rows.size(0) == row_count &&
                    centroids.size(0) == row_count && cos_spec.size(0) == row_count &&
                    material_id.size(0) == row_count,
                "per-row arrays must match rows");
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(layer_count.size(0) == material_count,
                "layer_count must match layer_offset rows");
    for (const auto& tref : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tref.size(0) == layer_total,
                    "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tref : {patch_uvs, rows, d_i, d_o, n_rows, source, vertex,
                             target, c1_positions, c2_positions, tx_pol, rx_pol,
                             l1_rows, l2_rows, sp1_rows, sp2_rows, centroids,
                             heights, cos_spec, material_id, layer_offset, quad_a})
        TORCH_CHECK(tref.get_device() == patch_tris.get_device(),
                    "chain-realization tensors must share device");
    return row_count;
}

pybind11::dict cn_scattering_chain_realization_eval(
    at::Tensor patch_tris,
    at::Tensor patch_uvs,
    at::Tensor rows,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor n_rows,
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
    at::Tensor tx_pol,
    at::Tensor rx_pol,
    at::Tensor l1_rows,
    at::Tensor l2_rows,
    at::Tensor sp1_rows,
    at::Tensor sp2_rows,
    at::Tensor centroids,
    at::Tensor heights,
    at::Tensor cos_spec,
    at::Tensor material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    at::Tensor quad_a,
    at::Tensor quad_b,
    at::Tensor quad_w,
    double k0,
    double frequency_hz) {
    const int64_t row_count = cn_scattering_chain_realization_check(
        patch_tris, patch_uvs, rows, d_i, d_o, n_rows, source, vertex, target,
        c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
        c2_mu_r, c2_gain, c2_thickness, c2_depth, tx_pol, rx_pol, l1_rows,
        l2_rows, sp1_rows, sp2_rows, centroids, heights, cos_spec, material_id,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, quad_a, quad_b, quad_w);
    const int64_t material_count = layer_offset.size(0);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = patch_tris.options().dtype(at::kComplexFloat);
    auto integral = at::empty({row_count}, complex_options);
    auto row_value = at::empty({row_count}, complex_options);
    auto path_field = at::empty({row_count}, complex_options);
    auto path_gain = at::empty({row_count}, patch_tris.options());
    auto total = at::empty({}, complex_options);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
    if (row_count > 0) {
        chain_realization_rows_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), source.data_ptr<float>(),
            vertex.data_ptr<float>(), target.data_ptr<float>(),
            c1_positions.data_ptr<float>(), c1_normals.data_ptr<float>(),
            c1_eps_r.data_ptr<float>(), c1_sigma_e.data_ptr<float>(),
            c1_mu_r.data_ptr<float>(), c1_gain.data_ptr<float>(),
            c1_thickness.data_ptr<float>(), c1_depth.data_ptr<int>(),
            c2_positions.data_ptr<float>(), c2_normals.data_ptr<float>(),
            c2_eps_r.data_ptr<float>(), c2_sigma_e.data_ptr<float>(),
            c2_mu_r.data_ptr<float>(), c2_gain.data_ptr<float>(),
            c2_thickness.data_ptr<float>(), c2_depth.data_ptr<int>(),
            tx_pol.data_ptr<float>(), rx_pol.data_ptr<float>(),
            l1_rows.data_ptr<float>(), l2_rows.data_ptr<float>(),
            sp1_rows.data_ptr<float>(), sp2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            cos_spec.data_ptr<float>(), material_id.data_ptr<int>(),
            layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(),
            quad_w.data_ptr<float>(),
            static_cast<float>(k0), static_cast<float>(frequency_hz),
            integral.data_ptr<cfloat>(), row_value.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        chain_realization_total_kernel<<<1, kReduceBlock, 0, stream>>>(
            row_count, row_value.data_ptr<cfloat>(), total.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        C10_CUDA_CHECK(cudaMemsetAsync(
            total.data_ptr(), 0, total.element_size(), stream));
    }
    (void)material_count;
    pybind11::dict out;
    out["total"] = total;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["integral"] = integral;
    out["row_value"] = row_value;
    return out;
}
