// ADR-010 op 2: native realization-coherent phase-screen patch integral.
//
// One fused launch per (tx, rx, structure) replacing the host per-patch
// Python loop (rows.tolist()) of _realization_rows, the per-patch Torch
// Gauss-Legendre quadrature (patch_phase_integral), and the per-row
// jones/prefactor/carrier assembly. Stage 1: one block per selected patch
// row, one thread per Duffy-mapped quadrature node, fixed-order shared-memory
// tree reduction of the phasor sum, then thread 0 assembles the row
// coefficient (prefactor * jones * carrier / (r1 * r2)) times the patch
// integral. Stage 2: a single block tree-reduces the row values into the
// 0-dim total in a fixed order. No float atomics: the total is bitwise
// stable run-to-run on the same binary.
//
// Height sampling replicates PhaseScreenRuntime.sample_height exactly:
// texel centers at (i + 0.5) / N, the continuous texel coordinate clamped to
// the span of texel centers before flooring (edge clamp, no wrap).
//
// Phase convention (module docstring of propagation/enumerated/scattering.py):
// the physical q = k0 * (d_o - d_i); the integral evaluates the SWAPPED
// integrand exp(-j * (q_int . x + q_int_n * h)) with q_int = -q, i.e. the
// physical +j integrand, with q_int_n taken against the WINDING normal of
// each patch triangle (exactly patch_phase_integral's convention). The
// leftover absolute-position phase is removed by the carrier's q . c term
// (q against the flipped illuminated-side normal path).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

namespace {

constexpr int kQuadPoints = 256;  // 16 x 16 Duffy-mapped Gauss-Legendre nodes
constexpr int kReduceBlock = 256;
constexpr float kPi = 3.14159265358979323846f;

using cfloat = c10::complex<float>;

struct V3 { float x, y, z; };
struct C2 { float re, im; };

__device__ __forceinline__ V3 load3(const float* __restrict__ p, int64_t i) {
    return {p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]};
}
__device__ __forceinline__ float dot3(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ V3 sub3(V3 a, V3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __forceinline__ V3 scale3(V3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

// PhaseScreenRuntime.sample_height: bilinear with half-texel edge clamp.
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

// Deterministic unit tangent (_stable_tangent): one-hot at the FIRST
// smallest |component| of n, Gram-Schmidt against n, normalized.
__device__ __forceinline__ V3 stable_tangent(V3 n) {
    const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    V3 axis = {0.0f, 0.0f, 0.0f};
    if (ax <= ay && ax <= az) axis.x = 1.0f;
    else if (ay <= az) axis.y = 1.0f;
    else axis.z = 1.0f;
    const float proj = dot3(axis, n);
    V3 t = {axis.x - proj * n.x, axis.y - proj * n.y, axis.z - proj * n.z};
    const float norm = fmaxf(sqrtf(dot3(t, t)), 1.0e-12f);
    return scale3(t, 1.0f / norm);
}

// _sp_basis: s = normalize(n x d) with the deterministic backup axis at
// normal incidence; p = s x d.
__device__ __forceinline__ void sp_basis(
    V3 n, V3 d, V3 backup, V3& s, V3& p) {
    const V3 raw = cross3(n, d);
    const float norm = sqrtf(dot3(raw, raw));
    if (norm < 1.0e-6f) {
        s = backup;
    } else {
        s = scale3(raw, 1.0f / fmaxf(norm, 1.0e-12f));
    }
    p = cross3(s, d);
}

__global__ void patch_integral_rows_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris,
    const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
    const cfloat* __restrict__ r_te,
    const cfloat* __restrict__ r_tm,
    const float* __restrict__ pol_t,
    const float* __restrict__ pol_r,
    const float* __restrict__ r1_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ centroids,
    const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ quad_a,
    const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0,
    cfloat* __restrict__ out_integral,
    cfloat* __restrict__ out_row_value) {
    __shared__ float sh_re[kQuadPoints];
    __shared__ float sh_im[kQuadPoints];
    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    // Triangle frame (patch_phase_integral): edges, winding normal, area.
    const V3 p0 = load3(patch_tris, patch * 3 + 0);
    const V3 p1 = load3(patch_tris, patch * 3 + 1);
    const V3 p2 = load3(patch_tris, patch * 3 + 2);
    const V3 e1 = sub3(p1, p0);
    const V3 e2 = sub3(p2, p0);
    const V3 winding = cross3(e1, e2);
    const float double_area = sqrtf(dot3(winding, winding));
    const V3 n_hat = scale3(winding, 1.0f / fmaxf(double_area, 1.0e-30f));

    // Torch rounding order preserved: k_i_vec = d_i * k0 and k_s_vec =
    // d_o * k0 round per component BEFORE the subtraction. Physical
    // q = k_s_vec - k_i_vec; the integrand uses q_int = -q (the documented
    // swapped-argument call of patch_phase_integral).
    const V3 di = load3(d_i, row);
    const V3 dov = load3(d_o, row);
    const V3 kiv = {di.x * k0, di.y * k0, di.z * k0};
    const V3 ksv = {dov.x * k0, dov.y * k0, dov.z * k0};
    const V3 q = sub3(ksv, kiv);
    const V3 q_int = sub3(kiv, ksv);
    const float q_int_n = dot3(n_hat, q_int);

    // Quadrature node phasor (one node per thread).
    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const V3 pos = {
        p0.x + a * e1.x + b * e2.x,
        p0.y + a * e1.y + b * e2.y,
        p0.z + a * e1.z + b * e2.z,
    };
    const float u0 = patch_uvs[(patch * 3 + 0) * 2 + 0];
    const float v0 = patch_uvs[(patch * 3 + 0) * 2 + 1];
    const float u1 = patch_uvs[(patch * 3 + 1) * 2 + 0];
    const float v1 = patch_uvs[(patch * 3 + 1) * 2 + 1];
    const float u2 = patch_uvs[(patch * 3 + 2) * 2 + 0];
    const float v2 = patch_uvs[(patch * 3 + 2) * 2 + 1];
    const float uu = u0 + a * (u1 - u0) + b * (u2 - u0);
    const float vv = v0 + a * (v1 - v0) + b * (v2 - v0);
    const float h = sample_height(heights, h_rows_dim, w_cols_dim, uu, vv);
    const float phase = dot3(pos, q_int) + q_int_n * h;
    float c, s;
    sincosf(-phase, &s, &c);
    sh_re[t] = c * w;
    sh_im[t] = s * w;
    __syncthreads();

    // Fixed-order shared-memory tree reduction over the 256 nodes.
#pragma unroll
    for (int stride = kQuadPoints / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_re[t] += sh_re[t + stride];
            sh_im[t] += sh_im[t + stride];
        }
        __syncthreads();
    }
    if (t != 0) return;

    const C2 integral = {sh_re[0] * double_area, sh_im[0] * double_area};
    out_integral[row] = cfloat(integral.re, integral.im);

    // Row coefficient: prefactor * jones * carrier / (r1 * r2).
    const V3 n = load3(n_rows, row);
    const float q_norm2 = dot3(q, q);
    const float q_n = fmaxf(dot3(q, n), 1.0e-9f);
    // prefactor = 1j * k0 * (|q|^2 / (k0 * q_n)) / (4 pi): purely imaginary.
    const float pref_im = k0 * (q_norm2 / (k0 * q_n)) / (4.0f * kPi);

    const V3 backup = stable_tangent(n);
    V3 s_i, p_i, s_o, p_o;
    sp_basis(n, di, backup, s_i, p_i);
    sp_basis(n, dov, backup, s_o, p_o);
    const V3 pt = {pol_t[0], pol_t[1], pol_t[2]};
    const V3 pr = {pol_r[0], pol_r[1], pol_r[2]};
    const float pt_di = dot3(pt, di);
    const V3 pt_perp = {pt.x - pt_di * di.x, pt.y - pt_di * di.y, pt.z - pt_di * di.z};
    const float pr_do = dot3(pr, dov);
    const V3 pr_perp = {pr.x - pr_do * dov.x, pr.y - pr_do * dov.y, pr.z - pr_do * dov.z};
    const float a_te = dot3(pt_perp, s_i);
    const float a_tm = dot3(pt_perp, p_i);
    const float g_te = dot3(pr_perp, s_o);
    const float g_tm = dot3(pr_perp, p_o);
    const cfloat te = r_te[row];
    const cfloat tm = r_tm[row];
    const C2 jones = {
        te.real() * (a_te * g_te) + tm.real() * (a_tm * g_tm),
        te.imag() * (a_te * g_te) + tm.imag() * (a_tm * g_tm),
    };

    const V3 c_row = load3(centroids, row);
    const float r1v = r1_rows[row];
    const float r2v = r2_rows[row];
    const float carrier_phase = -(k0 * (r1v + r2v) + dot3(q, c_row));
    float cc, cs;
    sincosf(carrier_phase, &cs, &cc);

    // (j * pref_im) * jones
    C2 value = {-pref_im * jones.im, pref_im * jones.re};
    // * carrier
    value = {value.re * cc - value.im * cs, value.re * cs + value.im * cc};
    // / (r1 * r2)
    const float inv_rr = 1.0f / (r1v * r2v);
    value = {value.re * inv_rr, value.im * inv_rr};
    // * integral
    out_row_value[row] = cfloat(
        value.re * integral.re - value.im * integral.im,
        value.re * integral.im + value.im * integral.re);
}

__global__ void patch_integral_total_kernel(
    int64_t row_count,
    const cfloat* __restrict__ row_values,
    cfloat* __restrict__ out_total) {
    __shared__ float sh_re[kReduceBlock];
    __shared__ float sh_im[kReduceBlock];
    const int t = threadIdx.x;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    // Fixed strided accumulation order (deterministic for a given row count).
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

}  // namespace

pybind11::dict cn_scattering_patch_integral_eval(
    at::Tensor patch_tris,
    at::Tensor patch_uvs,
    at::Tensor rows,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor n_rows,
    at::Tensor r_te,
    at::Tensor r_tm,
    at::Tensor pol_t,
    at::Tensor pol_r,
    at::Tensor r1_rows,
    at::Tensor r2_rows,
    at::Tensor centroids,
    at::Tensor heights,
    at::Tensor quad_a,
    at::Tensor quad_b,
    at::Tensor quad_w,
    double k0) {
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
    check_tensor(r_te, "r_te", at::kComplexFloat, 1);
    check_tensor(r_tm, "r_tm", at::kComplexFloat, 1);
    check_flat_tensor(pol_t, "pol_t", at::kFloat);
    check_flat_tensor(pol_r, "pol_r", at::kFloat);
    TORCH_CHECK(pol_t.size(0) == 3 && pol_r.size(0) == 3,
                "pol_t and pol_r must have shape (3,)");
    check_flat_tensor(r1_rows, "r1_rows", at::kFloat);
    check_flat_tensor(r2_rows, "r2_rows", at::kFloat);
    check_vec3_table(centroids, "centroids");
    check_tensor(heights, "heights", at::kFloat, 2);
    check_flat_tensor(quad_a, "quad_a", at::kFloat);
    check_flat_tensor(quad_b, "quad_b", at::kFloat);
    check_flat_tensor(quad_w, "quad_w", at::kFloat);
    TORCH_CHECK(quad_a.size(0) == kQuadPoints && quad_b.size(0) == kQuadPoints &&
                    quad_w.size(0) == kQuadPoints,
                "quadrature arrays must hold 16x16 Duffy-mapped nodes");
    TORCH_CHECK(d_i.size(0) == row_count && d_o.size(0) == row_count &&
                    n_rows.size(0) == row_count && r_te.size(0) == row_count &&
                    r_tm.size(0) == row_count && r1_rows.size(0) == row_count &&
                    r2_rows.size(0) == row_count && centroids.size(0) == row_count,
                "per-row arrays must match rows");
    for (const auto& t : {patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t,
                          pol_r, r1_rows, r2_rows, centroids, heights, quad_a,
                          quad_b, quad_w}) {
        TORCH_CHECK(t.get_device() == patch_tris.get_device(),
                    "patch-integral tensors must share device");
    }
    auto integral = at::empty(
        {row_count}, patch_tris.options().dtype(at::kComplexFloat));
    auto row_value = at::empty_like(integral);
    auto total = at::empty({}, patch_tris.options().dtype(at::kComplexFloat));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
    if (row_count > 0) {
        patch_integral_rows_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), r_te.data_ptr<cfloat>(), r_tm.data_ptr<cfloat>(),
            pol_t.data_ptr<float>(), pol_r.data_ptr<float>(),
            r1_rows.data_ptr<float>(), r2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(), quad_w.data_ptr<float>(),
            static_cast<float>(k0),
            integral.data_ptr<cfloat>(), row_value.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        patch_integral_total_kernel<<<1, kReduceBlock, 0, stream>>>(
            row_count, row_value.data_ptr<cfloat>(), total.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        C10_CUDA_CHECK(cudaMemsetAsync(
            total.data_ptr(), 0, total.element_size(), stream));
    }
    pybind11::dict out;
    out["total"] = total;
    out["integral"] = integral;
    out["row_value"] = row_value;
    return out;
}
