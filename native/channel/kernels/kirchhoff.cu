// Copyright Xingyu Chen.
// Implements Kirchhoff table derivative CUDA operations.

// Native JVP/VJP companions for the offline float64 Kirchhoff table builder.
// They differentiate resident values F = a S a with respect to roughness,
// CSR layer parameters, and carrier frequency. Inputs include the builder's
// float32 structural intermediates: S, balance factors, and diffuse budgets.
//
// Derivative structure for each polarization
// channel c and final table F_ij = a_i S_ij a_j over directional states
// i = (ti, pi), j = (to, po), w_j = cos_o(j) * dOmega:
//
// 1. Balanced-table adjoint: Gbar = grad_F ->
// abar_i = sum_j (Gbar_ij + Gbar_ji) S_ij a_j, Sbar_ij += Gbar_ij a_i a_j.
// 2. Implicit Sinkhorn adjoint at the converged factor a with
// phi_i(a) = a_i (S (w o a))_i - rhs_i = 0,
// J_ik = delta_ik (S (w o a))_i + a_i S_ik w_k.
// Solve J^T lambda = abar (dense LU via cuSOLVER; iso 32x32 per channel on
// the cos-collapsed system, aniso nti*npi square). Then rhsbar_i = lambda_i
// and Sbar_ij += -lambda_i a_i w_j a_j. Inactive rows (rhs_i <= 0, a_i = 0)
// are identity rows with zero adjoint.
// 3. Budget chain: rhs = R_bar_c(cos_i) (1 - c_r^2), c_r = exp(-2 (k0 cos_i
// sigma_h)^2). d rhs feeds sigma_h / layer / frequency via stack_rt_dual
// seeded at cos_i (d|r|^2 = 2 Re(conj(r) dr)) plus the explicit k0 term of
// c_r for frequency.
// 4. Raw-lobe adjoint over BOTH node sets (S = 0.5 (Raw + Raw_swap)):
// Raw = P(q) I(q; sigma_h, lx, ly) R_c(cos_h), q = k0 (wo + wi),
// P = |q|^4 / (16 pi^2 q_n^2 cos_i cos_o),
// cos_h = clamp((1 + wi.wo) k0 / |q|, 1e-6, 1) (k0-invariant),
// Beckmann series I = pi lx ly sum_m exp(T_m), recomputed in f32 with
// the build's n_terms. Scalar/CSR partials accumulate with atomicAdd.
// 5. JVP mirrors 1-4 forward: parameter tangents -> dS, drhs; solve J da =
// drhs - a_i sum_k dS_ik w_k a_k; tangent_F = da_i S_ij a_j + a_i S_ij da_j
// + a_i dS_ij a_j.
//
// The Beckmann series is recomputed in float32 (numerical parity requirement) and the stack
// reflectance derivatives reuse the shared field_transport_ad::stack_rt_dual
// dual library (mirrors em::stack_rt clamp for clamp). Requires linking
// cusolver (flag to the CMake owner).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusolverDn.h>
#include "torch_cuda.h"

#include <src/transmission_device.cuh>
#include <src/field_transport_ad.cuh>
#include "../tensor_checks.h"

namespace {

constexpr int kBlockSize = 256;
namespace em = rayd::shared::transmission;
namespace ad = rayd::torch::field_transport_ad;
namespace utd = rayd::shared::diffraction;

constexpr float kTwoPi = 6.283185307179586f;
constexpr float kDkDf = kTwoPi / em::kSpeedOfLight;  // dk0/df

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

// --------------------------------------------------------------------------
// Layer-stack seed functors (mirror the RayD typed layer-stack seeds; a private
// copy keeps this TU self-contained).
// --------------------------------------------------------------------------
struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct BasisSeed {
    int slot;
    int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
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

// --------------------------------------------------------------------------
// Beckmann diffuse-lobe series and its parameter partials, recomputed in f32
// with the build's n_terms. Returns I and dI/{sigma_h, lx, ly, k0}. Mirrors
// scattering/tables.py::_kirchhoff_diffuse_lobe_series term by term:
// T_m = m ln g - ln m! - ln m - rho^2/(4m) - g, g = (q_n sigma_h)^2,
// rho^2 = (qx lx)^2 + (qy ly)^2, I = pi lx ly sum_m exp(T_m).
// The g = 0 horizon is guarded (all partials vanish, matching exp(T_m) -> 0).
// --------------------------------------------------------------------------
struct LobePartials {
    float value;      // I
    float d_sigma_h;  // dI/dsigma_h
    float d_lx;       // dI/dlx
    float d_ly;       // dI/dly
    float d_k0;       // dI/dk0 (explicit; frequency chain multiplies dk0/df)
};

__device__ LobePartials kirchhoff_lobe_partials(
    float qx, float qy, float qn, float sigma_h, float lx, float ly, float k0,
    int n_terms) {
    LobePartials out{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float g = (qn * sigma_h) * (qn * sigma_h);
    const float rho2 = (qx * lx) * (qx * lx) + (qy * ly) * (qy * ly);
    const float pi_lxly = utd::UTD_PI * lx * ly;
    if (!(g > 0.0f)) {
        // Smooth / horizon limit: exp(m ln g) -> 0 for every m >= 1.
        return out;
    }
    const float log_g = logf(g);
    float S0 = 0.0f;    // sum e
    float Sg = 0.0f;    // sum e (m/g - 1)
    float Sx = 0.0f;    // sum e qx^2/(2m)
    float Sy = 0.0f;    // sum e qy^2/(2m)
    float Sm = 0.0f;    // sum e m
    float Srho = 0.0f;  // sum e rho^2/(2m)
    const float qx2 = qx * qx;
    const float qy2 = qy * qy;
    for (int mi = 1; mi <= n_terms; ++mi) {
        const float m = static_cast<float>(mi);
        const float log_term =
            m * log_g - lgammaf(m + 1.0f) - logf(m) - rho2 / (4.0f * m) - g;
        const float e = expf(log_term);
        S0 += e;
        Sg += e * (m / g - 1.0f);
        Sx += e * (qx2 / (2.0f * m));
        Sy += e * (qy2 / (2.0f * m));
        Sm += e * m;
        Srho += e * (rho2 / (2.0f * m));
    }
    out.value = pi_lxly * S0;
    out.d_sigma_h = pi_lxly * Sg * 2.0f * (qn * qn) * sigma_h;
    // dI/dlx = I/lx - pi lx ly * lx * Sx (Sx = sum e qx^2/(2m)).
    out.d_lx = out.value / lx - pi_lxly * lx * Sx;
    out.d_ly = out.value / ly - pi_lxly * ly * Sy;
    // dT_m/dk0 = (2m - rho^2/(2m) - 2g) / k0.
    out.d_k0 = (pi_lxly / k0) * (2.0f * Sm - Srho - 2.0f * g * S0);
    return out;
}

// Geometry of one raw-lobe node: q, prefactor P and its dP/dk0, cos_h.
struct NodeGeom {
    float qx, qy, qn, q_sq;
    float cos_h;
    float prefactor;  // P
    float dP_dk0;     // 2 P / k0
};

__device__ __forceinline__ NodeGeom node_geometry(
    float cos_inc, float phi_inc, float cos_out, float phi_out, float k0) {
    const float sin_inc = sqrtf(fmaxf(0.0f, 1.0f - cos_inc * cos_inc));
    const float sin_out = sqrtf(fmaxf(0.0f, 1.0f - cos_out * cos_out));
    float ci, si;
    sincosf(phi_inc, &si, &ci);
    float co, so;
    sincosf(phi_out, &so, &co);
    const float wi_x = sin_inc * ci, wi_y = sin_inc * si, wi_z = cos_inc;
    const float wo_x = sin_out * co, wo_y = sin_out * so, wo_z = cos_out;
    NodeGeom n;
    n.qx = k0 * (wo_x + wi_x);
    n.qy = k0 * (wo_y + wi_y);
    n.qn = k0 * (wo_z + wi_z);
    n.q_sq = n.qx * n.qx + n.qy * n.qy + n.qn * n.qn;
    const float wi_dot_wo = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z;
    n.cos_h = fminf(
        fmaxf((1.0f + wi_dot_wo) * k0 / sqrtf(n.q_sq), 1.0e-6f), 1.0f);
    n.prefactor = (n.q_sq * n.q_sq) /
                  (16.0f * utd::UTD_PI * utd::UTD_PI * n.qn * n.qn * wi_z * wo_z);
    n.dP_dk0 = 2.0f * n.prefactor / k0;
    return n;
}

// --------------------------------------------------------------------------
// Balance-state helpers. The balance system acts on states s = (cos, phi). For
// isotropic tables (npi == 1) the reverse state collapses to cos only, so the
// state count is nti and a state's phi index is always 0; for anisotropic
// tables the state is the full (cos, phi) pair. af reads the balance factor
// with the isotropic collapse; the outgoing weight is cos_o * dOmega.
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// Kernel: Sbar direct term Sbar_ij = grad_f_ij a_i a_j (both iso and aniso;
// a_j uses the isotropic collapse when npi == 1).
// --------------------------------------------------------------------------
__global__ void sbar_direct_kernel(
    int64_t count, int nti, int npi, int nto, int npo,
    const float* __restrict__ grad_f,
    const float* __restrict__ a,  // [nti, npi]
    double* __restrict__ sbar) {
    // a_i * a_j (balance factors) reach ~1e21 near grazing; the pairwise product
    // overflows float32, so the balanced-table adjoint accumulates in double.
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int po = static_cast<int>(idx % npo);
        const int to = static_cast<int>((idx / npo) % nto);
        const int pi = static_cast<int>((idx / (static_cast<int64_t>(npo) * nto)) % npi);
        const int ti = static_cast<int>(idx / (static_cast<int64_t>(npo) * nto * npi));
        const double a_i = a[ti * npi + pi];
        const int a_out_phi = (npi == 1) ? 0 : po;
        const double a_j = a[to * npi + a_out_phi];
        sbar[idx] = static_cast<double>(grad_f[idx]) * a_i * a_j;
    }
}

// --------------------------------------------------------------------------
// Kernel: abar (balanced-table adjoint), iso and aniso paths.
// aniso: abar_i = sum_{to,po} (grad_f[ti,pi,to,po] + grad_f[to,po,ti,pi])
// S[ti,pi,to,po] a[to,po].
// iso: abar_i = sum_{to,po} grad_f[i,0,to,po] S[i,0,to,po] a[to]
// + sum_{i',po} grad_f[i',0,i,po] a[i'] S[i',0,i,po].
// One thread per balance state.
// --------------------------------------------------------------------------
__global__ void abar_kernel(
    int n_states, int nti, int npi, int nto, int npo, bool iso,
    const float* __restrict__ grad_f,
    const float* __restrict__ s,
    const float* __restrict__ a,        // [nti, npi]
    const float* __restrict__ r_diff,   // [nti, npi]
    double* __restrict__ abar) {
    for (int state = blockIdx.x * blockDim.x + threadIdx.x; state < n_states;
         state += blockDim.x * gridDim.x) {
        const int ti = state / npi;
        const int pi = state % npi;
        if (!(r_diff[ti * npi + pi] > 0.0f)) {
            abar[state] = 0.0;  // inactive row: identity, zero adjoint.
            continue;
        }
        double acc = 0.0;
        if (iso) {
            // term A: i as incidence.
            for (int to = 0; to < nto; ++to) {
                const double a_to = a[to * npi + 0];
                for (int po = 0; po < npo; ++po) {
                    const int64_t f = ((static_cast<int64_t>(ti) * npi + 0) * nto + to) * npo + po;
                    acc += static_cast<double>(grad_f[f]) * static_cast<double>(s[f]) * a_to;
                }
            }
            // term B: i (cos) as outgoing state.
            for (int ip = 0; ip < nti; ++ip) {
                const double a_ip = a[ip * npi + 0];
                for (int po = 0; po < npo; ++po) {
                    const int64_t f = ((static_cast<int64_t>(ip) * npi + 0) * nto + ti) * npo + po;
                    acc += static_cast<double>(grad_f[f]) * a_ip * static_cast<double>(s[f]);
                }
            }
        } else {
            for (int to = 0; to < nto; ++to) {
                for (int po = 0; po < npo; ++po) {
                    const int64_t fwd = ((static_cast<int64_t>(ti) * npi + pi) * nto + to) * npo + po;
                    const int64_t rev = ((static_cast<int64_t>(to) * npi + po) * nto + ti) * npo + pi;
                    const double a_j = a[to * npi + po];
                    acc += (static_cast<double>(grad_f[fwd]) + static_cast<double>(grad_f[rev])) *
                           static_cast<double>(s[fwd]) * a_j;
                }
            }
        }
        abar[state] = acc;
    }
}

// Collapsed state matrix Smat[i*n+k] (row-major), in double. iso: sum over po of
// s[i,0,k,po]; aniso: s.flat[i*n+k] (state matrix is s reshaped to n x n). The
// input lobe s may be the exported f32 lobe (backward) or the double dS tangent
// (JVP), so the input element type is a template parameter.
template <typename Tin>
__global__ void smat_kernel(
    int n, int nti, int npi, int nto, int npo, bool iso,
    const Tin* __restrict__ s, double* __restrict__ smat) {
    const int64_t total = static_cast<int64_t>(n) * n;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / n);
        const int k = static_cast<int>(idx % n);
        if (iso) {
            double acc = 0.0;
            for (int po = 0; po < npo; ++po)
                acc += static_cast<double>(
                    s[((static_cast<int64_t>(i) * npi + 0) * nto + k) * npo + po]);
            smat[idx] = acc;
        } else {
            smat[idx] = static_cast<double>(s[idx]);
        }
    }
}

// Per-state outgoing weight w and factor a as flat [n] arrays.
__global__ void state_weight_kernel(
    int n, int nti, int npi, int nto, int npo, bool iso, double d_omega,
    const float* __restrict__ cos_o,
    const float* __restrict__ a,       // [nti, npi]
    const float* __restrict__ r_diff,  // [nti, npi]
    double* __restrict__ w_state,
    double* __restrict__ a_state,
    int* __restrict__ active) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += blockDim.x * gridDim.x) {
        if (iso) {
            w_state[k] = static_cast<double>(cos_o[k]) * d_omega;
            a_state[k] = a[k * npi + 0];
            active[k] = (r_diff[k * npi + 0] > 0.0f) ? 1 : 0;
        } else {
            const int to = k / npo;
            const int po = k % npo;
            w_state[k] = static_cast<double>(cos_o[to]) * d_omega;
            a_state[k] = a[to * npi + po];
            active[k] = (r_diff[to * npi + po] > 0.0f) ? 1 : 0;
        }
    }
}

// D_i = sum_k Smat[i,k] w_k a_k.
__global__ void diag_kernel(
    int n, const double* __restrict__ smat, const double* __restrict__ w_state,
    const double* __restrict__ a_state, double* __restrict__ diag) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k)
            acc += smat[static_cast<int64_t>(i) * n + k] * w_state[k] * a_state[k];
        diag[i] = acc;
    }
}

// Assemble the dense system matrix in column-major order.
// transpose=true -> A = J^T: A[i,k] = delta_ik D_i + a_k Smat[k,i] w_i.
// transpose=false -> J: J[i,k] = delta_ik D_i + a_i Smat[i,k] w_k.
// Inactive rows i become identity rows.
__global__ void assemble_matrix_kernel(
    int n, bool transpose,
    const double* __restrict__ smat, const double* __restrict__ diag,
    const double* __restrict__ w_state, const double* __restrict__ a_state,
    const int* __restrict__ active, double* __restrict__ mat_colmajor) {
    const int64_t total = static_cast<int64_t>(n) * n;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx % n);  // row
        const int k = static_cast<int>(idx / n);  // col (column-major)
        double value;
        if (!active[i]) {
            value = (i == k) ? 1.0 : 0.0;
        } else {
            const double d = (i == k) ? diag[i] : 0.0;
            double off;
            if (transpose)
                off = a_state[k] * smat[static_cast<int64_t>(k) * n + i] * w_state[i];
            else
                off = a_state[i] * smat[static_cast<int64_t>(i) * n + k] * w_state[k];
            value = d + off;
        }
        mat_colmajor[idx] = value;
    }
}

// Add the implicit-adjoint contribution Sbar_ij += -lambda_i a_i w_j a_j.
__global__ void sbar_implicit_kernel(
    int64_t count, int nti, int npi, int nto, int npo, double d_omega,
    const float* __restrict__ cos_o,
    const float* __restrict__ a,       // [nti, npi]
    const double* __restrict__ lambda,  // [n_states]
    double* __restrict__ sbar) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int po = static_cast<int>(idx % npo);
        const int to = static_cast<int>((idx / npo) % nto);
        const int pi = static_cast<int>((idx / (static_cast<int64_t>(npo) * nto)) % npi);
        const int ti = static_cast<int>(idx / (static_cast<int64_t>(npo) * nto * npi));
        const int state_i = ti * npi + pi;
        const double a_i = a[ti * npi + pi];
        const int a_out_phi = (npi == 1) ? 0 : po;
        const double a_j = a[to * npi + a_out_phi];
        const double w_j = static_cast<double>(cos_o[to]) * d_omega;
        sbar[idx] += -lambda[state_i] * a_i * w_j * a_j;
    }
}

// Aggregate lambda over the incidence-azimuth axis into budget_adjoint[ti].
__global__ void budget_adjoint_kernel(
    int nti, int npi, const double* __restrict__ lambda,
    double* __restrict__ budget_adjoint) {
    for (int ti = blockIdx.x * blockDim.x + threadIdx.x; ti < nti;
         ti += blockDim.x * gridDim.x) {
        double acc = 0.0;
        for (int pi = 0; pi < npi; ++pi)
            acc += lambda[ti * npi + pi];
        budget_adjoint[ti] = acc;
    }
}

// --------------------------------------------------------------------------
// Step 3 budget chain (backward): per incidence cos, propagate budget_adjoint
// to sigma_h / layers / frequency. R_bar and its derivatives come from the
// native stack (stack_rt_dual seeded at cos_i).
// --------------------------------------------------------------------------
__global__ void budget_chain_backward_kernel(
    int nti, int pol, float sigma_h, float k0, float frequency_hz,
    const float* __restrict__ cos_i,
    const double* __restrict__ budget_adjoint,
    const int* __restrict__ layer_offset,
    const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m,
    const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e,
    const float* __restrict__ layer_mu_r,
    int layer_total,
    float* __restrict__ grad_sigma_h,
    float* __restrict__ grad_layer_thickness,
    float* __restrict__ grad_layer_eps_r,
    float* __restrict__ grad_layer_sigma_e,
    float* __restrict__ grad_frequency) {
    for (int ti = blockIdx.x * blockDim.x + threadIdx.x; ti < nti;
         ti += blockDim.x * gridDim.x) {
        // budget_adjoint (lambda aggregated) is the double implicit-adjoint
        // output; the elementwise stack/c_r partials stay float32.
        const double lam = budget_adjoint[ti];
        if (lam == 0.0)
            continue;
        const float ct = cos_i[ti];
        const em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, 0};
        const ZeroSeed zero;
        const ad::DualStackRT base =
            ad::stack_rt_dual(ct, layers, frequency_hz, 0.0f, 0.0f, pol, zero);
        const float R_bar = base.cap_r.v;
        const float ks = k0 * ct * sigma_h;
        const float c_r = expf(-2.0f * ks * ks);
        const float c_r2 = c_r * c_r;
        const float one_minus = 1.0f - c_r2;
        if (grad_sigma_h != nullptr) {
            const float d = R_bar * c_r2 * 8.0f * (k0 * ct) * (k0 * ct) * sigma_h;
            atomicAdd(grad_sigma_h, static_cast<float>(lam * d));
        }
        if (grad_layer_thickness != nullptr || grad_layer_eps_r != nullptr ||
            grad_layer_sigma_e != nullptr) {
            const int count = layer_count[0];
            for (int slot = 0; slot < count; ++slot) {
                for (int param = 0; param < 3; ++param) {
                    float* dst = param == 0 ? grad_layer_thickness
                                 : param == 1 ? grad_layer_eps_r
                                              : grad_layer_sigma_e;
                    if (dst == nullptr)
                        continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT ld = ad::stack_rt_dual(
                        ct, layers, frequency_hz, 0.0f, 0.0f, pol, seed);
                    atomicAdd(dst + slot,
                              static_cast<float>(lam * one_minus * ld.cap_r.d));
                }
            }
        }
        if (grad_frequency != nullptr) {
            const ad::DualStackRT fd = ad::stack_rt_dual(
                ct, layers, frequency_hz, 0.0f, 1.0f, pol, zero);
            const float dR_df = fd.cap_r.d;
            // Explicit c_r frequency term: d(c_r^2)/df = c_r^2 (-8 k0 (ct
            // sigma_h)^2) dk0/df; d rhs = -R_bar d(c_r^2).
            const float c_r_freq =
                R_bar * 8.0f * k0 * (ct * sigma_h) * (ct * sigma_h) * c_r2 * kDkDf;
            atomicAdd(grad_frequency,
                      static_cast<float>(lam * (one_minus * dR_df + c_r_freq)));
        }
    }
}

// --------------------------------------------------------------------------
// Step 4 raw-lobe adjoint (backward), one launch per (channel, node-set). The
// node inc/out grids swap for the swapped node set; the weight is always
// 0.5 * Sbar[flat]. Accumulates sigma_h / corr_x / corr_y / layers / frequency.
// --------------------------------------------------------------------------
__global__ void raw_lobe_backward_kernel(
    int64_t count, int nti, int npi, int nto, int npo, bool swapped, int pol,
    float sigma_h, float lx, float ly, float k0, float frequency_hz, int n_terms,
    const float* __restrict__ cos_i, const float* __restrict__ phi_i,
    const float* __restrict__ cos_o, const float* __restrict__ phi_o,
    const double* __restrict__ sbar,
    const int* __restrict__ layer_offset,
    const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m,
    const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e,
    const float* __restrict__ layer_mu_r,
    float* __restrict__ grad_sigma_h,
    float* __restrict__ grad_corr_x,
    float* __restrict__ grad_corr_y,
    float* __restrict__ grad_layer_thickness,
    float* __restrict__ grad_layer_eps_r,
    float* __restrict__ grad_layer_sigma_e,
    float* __restrict__ grad_frequency) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        // Sbar (balanced-table adjoint) is the double a_i*a_j product; the
        // Beckmann/stack partials below stay float32 (scattering AD).
        const double w = 0.5 * sbar[idx];
        if (w == 0.0)
            continue;
        const int po = static_cast<int>(idx % npo);
        const int to = static_cast<int>((idx / npo) % nto);
        const int pi = static_cast<int>((idx / (static_cast<int64_t>(npo) * nto)) % npi);
        const int ti = static_cast<int>(idx / (static_cast<int64_t>(npo) * nto * npi));
        // Forward node: inc = (cos_i[ti], phi_i[pi]), out = (cos_o[to], phi_o[po]).
        // Swapped node: inc = (cos_o[to], phi_o[po]), out = (cos_i[ti], phi_i[pi]).
        float cos_inc, phi_inc, cos_out, phi_out;
        if (!swapped) {
            cos_inc = cos_i[ti];
            phi_inc = phi_i[pi];
            cos_out = cos_o[to];
            phi_out = phi_o[po];
        } else {
            cos_inc = cos_o[to];
            phi_inc = phi_o[po];
            cos_out = cos_i[ti];
            phi_out = phi_i[pi];
        }
        const NodeGeom node =
            node_geometry(cos_inc, phi_inc, cos_out, phi_out, k0);
        const LobePartials lobe = kirchhoff_lobe_partials(
            node.qx, node.qy, node.qn, sigma_h, lx, ly, k0, n_terms);
        const float P = node.prefactor;
        const float I = lobe.value;
        const float shape = P * I;
        const em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, 0};
        const ZeroSeed zero;
        const ad::DualStackRT base = ad::stack_rt_dual(
            node.cos_h, layers, frequency_hz, 0.0f, 0.0f, pol, zero);
        const float R = base.cap_r.v;

        if (grad_sigma_h != nullptr)
            atomicAdd(grad_sigma_h, static_cast<float>(w * P * lobe.d_sigma_h * R));
        if (grad_corr_x != nullptr)
            atomicAdd(grad_corr_x, static_cast<float>(w * P * lobe.d_lx * R));
        if (grad_corr_y != nullptr)
            atomicAdd(grad_corr_y, static_cast<float>(w * P * lobe.d_ly * R));
        if (grad_layer_thickness != nullptr || grad_layer_eps_r != nullptr ||
            grad_layer_sigma_e != nullptr) {
            const int count = layer_count[0];
            for (int slot = 0; slot < count; ++slot) {
                for (int param = 0; param < 3; ++param) {
                    float* dst = param == 0 ? grad_layer_thickness
                                 : param == 1 ? grad_layer_eps_r
                                              : grad_layer_sigma_e;
                    if (dst == nullptr)
                        continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT ld = ad::stack_rt_dual(
                        node.cos_h, layers, frequency_hz, 0.0f, 0.0f, pol, seed);
                    atomicAdd(dst + slot, static_cast<float>(w * shape * ld.cap_r.d));
                }
            }
        }
        if (grad_frequency != nullptr) {
            const ad::DualStackRT fd = ad::stack_rt_dual(
                node.cos_h, layers, frequency_hz, 0.0f, 1.0f, pol, zero);
            const float dR_df = fd.cap_r.d;
            const float dP_df = node.dP_dk0 * kDkDf;
            const float dI_df = lobe.d_k0 * kDkDf;
            const float draw_df = R * (dP_df * I + P * dI_df) + shape * dR_df;
            atomicAdd(grad_frequency, static_cast<float>(w * draw_df));
        }
    }
}

// --------------------------------------------------------------------------
// JVP kernels.
// --------------------------------------------------------------------------

// dS forward over both node sets: dS[idx] = 0.5 (dRaw_fwd + dRaw_swap), one
// launch per node-set accumulating into dsbar (per channel).
__global__ void raw_lobe_jvp_kernel(
    int64_t count, int nti, int npi, int nto, int npo, bool swapped, int pol,
    float sigma_h, float lx, float ly, float k0, float frequency_hz, int n_terms,
    float t_sigma_h, float t_lx, float t_ly, float t_frequency,
    const float* __restrict__ cos_i, const float* __restrict__ phi_i,
    const float* __restrict__ cos_o, const float* __restrict__ phi_o,
    const int* __restrict__ layer_offset,
    const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m,
    const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e,
    const float* __restrict__ layer_mu_r,
    const float* __restrict__ t_thickness,
    const float* __restrict__ t_eps,
    const float* __restrict__ t_sigma,
    double* __restrict__ dsbar) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int po = static_cast<int>(idx % npo);
        const int to = static_cast<int>((idx / npo) % nto);
        const int pi = static_cast<int>((idx / (static_cast<int64_t>(npo) * nto)) % npi);
        const int ti = static_cast<int>(idx / (static_cast<int64_t>(npo) * nto * npi));
        float cos_inc, phi_inc, cos_out, phi_out;
        if (!swapped) {
            cos_inc = cos_i[ti]; phi_inc = phi_i[pi];
            cos_out = cos_o[to]; phi_out = phi_o[po];
        } else {
            cos_inc = cos_o[to]; phi_inc = phi_o[po];
            cos_out = cos_i[ti]; phi_out = phi_i[pi];
        }
        const NodeGeom node =
            node_geometry(cos_inc, phi_inc, cos_out, phi_out, k0);
        const LobePartials lobe = kirchhoff_lobe_partials(
            node.qx, node.qy, node.qn, sigma_h, lx, ly, k0, n_terms);
        const float P = node.prefactor;
        const float I = lobe.value;
        const float shape = P * I;
        const em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, 0};
        const TangentSeed seed{t_thickness, t_eps, t_sigma};
        const ad::DualStackRT rd = ad::stack_rt_dual(
            node.cos_h, layers, frequency_hz, 0.0f, t_frequency, pol, seed);
        const float R = rd.cap_r.v;
        const float dR = rd.cap_r.d;  // stack tangent (layers + frequency)
        const float dP_df = node.dP_dk0 * kDkDf;
        const float dI = lobe.d_sigma_h * t_sigma_h + lobe.d_lx * t_lx +
                         lobe.d_ly * t_ly + lobe.d_k0 * kDkDf * t_frequency;
        const float dP = dP_df * t_frequency;
        // Elementwise lobe/stack tangent stays float32; dS accumulates in double
        // so the downstream a_i*a_j-scaled forcing does not overflow.
        const float draw = dP * I * R + P * dI * R + shape * dR;
        dsbar[idx] += 0.5 * static_cast<double>(draw);
    }
}

// drhs forward per incidence cos: drhs[ti] = (1 - c_r^2) dR_bar - R_bar d(c_r^2).
__global__ void budget_chain_jvp_kernel(
    int nti, int pol, float sigma_h, float k0, float frequency_hz,
    float t_sigma_h, float t_frequency,
    const float* __restrict__ cos_i,
    const int* __restrict__ layer_offset,
    const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m,
    const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e,
    const float* __restrict__ layer_mu_r,
    const float* __restrict__ t_thickness,
    const float* __restrict__ t_eps,
    const float* __restrict__ t_sigma,
    double* __restrict__ drhs) {
    for (int ti = blockIdx.x * blockDim.x + threadIdx.x; ti < nti;
         ti += blockDim.x * gridDim.x) {
        const float ct = cos_i[ti];
        const em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, 0};
        const TangentSeed seed{t_thickness, t_eps, t_sigma};
        const ad::DualStackRT rd = ad::stack_rt_dual(
            ct, layers, frequency_hz, 0.0f, t_frequency, pol, seed);
        const float R_bar = rd.cap_r.v;
        const float dR_bar = rd.cap_r.d;
        const float ks = k0 * ct * sigma_h;
        const float c_r = expf(-2.0f * ks * ks);
        const float c_r2 = c_r * c_r;
        // d(c_r^2) = c_r^2 (-8 (k0 ct)^2 sigma_h) dsigma_h
        // + c_r^2 (-8 k0 (ct sigma_h)^2) dk0; dk0 = kDkDf t_frequency.
        const float dc_r2 =
            c_r2 * (-8.0f * (k0 * ct) * (k0 * ct) * sigma_h) * t_sigma_h +
            c_r2 * (-8.0f * k0 * (ct * sigma_h) * (ct * sigma_h)) *
                (kDkDf * t_frequency);
        drhs[ti] = static_cast<double>((1.0f - c_r2) * dR_bar - R_bar * dc_r2);
    }
}

// JVP forcing e_i = drhs[state_i] - a_i sum_k dSmat[i,k] w_k a_k. Uses the
// collapsed dSmat (iso) or full dS state matrix (aniso).
__global__ void jvp_forcing_kernel(
    int n_states, int nti, int npi, int nto, int npo, bool iso,
    const double* __restrict__ dsmat,   // [n, n]
    const double* __restrict__ w_state,
    const double* __restrict__ a_state,
    const float* __restrict__ a,        // [nti, npi]
    const double* __restrict__ drhs,     // [nti]
    const int* __restrict__ active,
    double* __restrict__ e) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_states;
         i += blockDim.x * gridDim.x) {
        if (!active[i]) {
            e[i] = 0.0;
            continue;
        }
        const int ti = i / npi;
        const int pi = i % npi;
        const double a_i = a[ti * npi + pi];
        double acc = 0.0;
        for (int k = 0; k < n_states; ++k)
            acc += dsmat[static_cast<int64_t>(i) * n_states + k] * w_state[k] * a_state[k];
        e[i] = drhs[ti] - a_i * acc;
    }
}

// tangent_F[i,j] = da_i S_ij a_j + a_i S_ij da_j + a_i dS_ij a_j.
__global__ void tangent_f_kernel(
    int64_t count, int nti, int npi, int nto, int npo,
    const float* __restrict__ s,
    const double* __restrict__ dsbar,
    const float* __restrict__ a,        // [nti, npi]
    const double* __restrict__ da_state, // [n_states]
    float* __restrict__ tangent_f) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int po = static_cast<int>(idx % npo);
        const int to = static_cast<int>((idx / npo) % nto);
        const int pi = static_cast<int>((idx / (static_cast<int64_t>(npo) * nto)) % npi);
        const int ti = static_cast<int>(idx / (static_cast<int64_t>(npo) * nto * npi));
        // a_i*a_j products in double (grazing overflow); output tangent_F is the
        // order-unity table tangent, downcast to the f32 output.
        const double a_i = a[ti * npi + pi];
        const int a_out_phi = (npi == 1) ? 0 : po;
        const double a_j = a[to * npi + a_out_phi];
        const double da_i = da_state[ti * npi + pi];
        const int da_out = (npi == 1) ? to : (to * npo + po);
        const double da_j = da_state[da_out];
        const double S = static_cast<double>(s[idx]);
        tangent_f[idx] = static_cast<float>(
            da_i * S * a_j + a_i * S * da_j + a_i * dsbar[idx] * a_j);
    }
}

// --------------------------------------------------------------------------
// float64 SVD pseudo-inverse solve of A x = b (column-major A [n*n], b [n]) in
// place. The reverse/tangent balance Jacobian J (and J^T) is structurally
// singular for anisotropic tables: the azimuth half-turn permutation commutes
// with the kernel while rhs is azimuth-independent, so J at the symmetric fixed
// point has a null space of half-turn-antisymmetric azimuth modes. Every
// differentiable perturbation and the cotangent load lie in the symmetric
// subspace (orthogonal to that null space), so the minimum-norm least-squares
// solution x = V diag(1/s_i | s_i > eps_rel s_max) U^T b is EXACTLY the
// derivative of the symmetric selection branch. eps_rel = 1e-10. Everything is
// float64 because the balance factors reach ~1e21 near grazing.
// --------------------------------------------------------------------------
constexpr double kPinvRelCutoff = 1.0e-10;

// c_i = (U^T b)_i / s_i, thresholded at eps_rel * s_max (S sorted descending).
__global__ void pinv_project_kernel(
    int n, const double* __restrict__ U, const double* __restrict__ b,
    const double* __restrict__ S, double eps_rel, double* __restrict__ c) {
    const double s_max = S[0];
    const double cutoff = eps_rel * s_max;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        double dot = 0.0;
        for (int r = 0; r < n; ++r)
            dot += U[r + static_cast<int64_t>(i) * n] * b[r];  // column-major U
        const double s_i = S[i];
        c[i] = (s_i > cutoff) ? dot / s_i : 0.0;
    }
}

// x_r = sum_i V[r,i] c_i = sum_i VT[i,r] c_i (VT column-major = V^T).
__global__ void pinv_reconstruct_kernel(
    int n, const double* __restrict__ VT, const double* __restrict__ c,
    double* __restrict__ x) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n;
         r += blockDim.x * gridDim.x) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
            acc += VT[i + static_cast<int64_t>(r) * n] * c[i];
        x[r] = acc;
    }
}

void solve_dense(at::Tensor& A_colmajor, at::Tensor& b, int n) {
    // A_colmajor and b are float64. A is overwritten by gesvd; b is overwritten
    // with the minimum-norm solution.
    auto opts = A_colmajor.options();  // float64 CUDA
    cusolverDnHandle_t handle = nullptr;
    TORCH_CHECK(cusolverDnCreate(&handle) == CUSOLVER_STATUS_SUCCESS,
                "cusolverDnCreate failed");
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(A_colmajor.get_device()).stream();
    cusolverDnSetStream(handle, stream);

    auto U = at::empty({static_cast<int64_t>(n) * n}, opts);
    auto VT = at::empty({static_cast<int64_t>(n) * n}, opts);
    auto S = at::empty({n}, opts);
    int lwork = 0;
    TORCH_CHECK(
        cusolverDnDgesvd_bufferSize(handle, n, n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDnDgesvd_bufferSize failed");
    auto work = at::empty({lwork}, opts);
    auto rwork = at::empty({n > 1 ? n - 1 : 1}, opts);
    auto info = at::empty({1}, opts.dtype(at::kInt));
    // jobu = jobvt = 'A': full U (n x n) and V^T (n x n), both column-major.
    const cusolverStatus_t status = cusolverDnDgesvd(
        handle, 'A', 'A', n, n, A_colmajor.data_ptr<double>(), n,
        S.data_ptr<double>(), U.data_ptr<double>(), n, VT.data_ptr<double>(), n,
        work.data_ptr<double>(), lwork, rwork.data_ptr<double>(),
        info.data_ptr<int>());
    cusolverDnDestroy(handle);
    TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, "cusolverDnDgesvd failed");

    auto c = at::empty({n}, opts);
    pinv_project_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, U.data_ptr<double>(), b.data_ptr<double>(), S.data_ptr<double>(),
        kPinvRelCutoff, c.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pinv_reconstruct_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, VT.data_ptr<double>(), c.data_ptr<double>(), b.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------------------------------------------------
// Shared input validation for the two entry points.
// --------------------------------------------------------------------------
struct TableDims {
    int nti, npi, nto, npo, n_states, layer_total;
    bool iso;
    float d_omega, k0;
    int n_terms;
};

TableDims check_inputs(
    const at::Tensor& s_te, const at::Tensor& s_tm, const at::Tensor& a_te,
    const at::Tensor& a_tm, const at::Tensor& r_diff_te, const at::Tensor& r_diff_tm,
    const at::Tensor& cos_i, const at::Tensor& phi_i, const at::Tensor& cos_o,
    const at::Tensor& phi_o, const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r, const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r, double sigma_h, double frequency_hz) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    check_tensor(s_te, "s_te", at::kFloat, 4);
    check_tensor(s_tm, "s_tm", at::kFloat, 4);
    TableDims d;
    d.nti = static_cast<int>(s_te.size(0));
    d.npi = static_cast<int>(s_te.size(1));
    d.nto = static_cast<int>(s_te.size(2));
    d.npo = static_cast<int>(s_te.size(3));
    TORCH_CHECK(s_tm.sizes() == s_te.sizes(), "s_tm must match s_te shape");
    TORCH_CHECK(d.nti == d.nto, "reciprocal balance requires nti == nto");
    check_tensor(a_te, "a_te", at::kFloat, 2);
    check_tensor(a_tm, "a_tm", at::kFloat, 2);
    check_tensor(r_diff_te, "r_diff_te", at::kFloat, 2);
    check_tensor(r_diff_tm, "r_diff_tm", at::kFloat, 2);
    TORCH_CHECK(a_te.size(0) == d.nti && a_te.size(1) == d.npi,
                "a_te must have shape (nti, npi)");
    TORCH_CHECK(a_tm.sizes() == a_te.sizes(), "a_tm must match a_te shape");
    TORCH_CHECK(r_diff_te.sizes() == a_te.sizes(), "r_diff_te must be (nti, npi)");
    TORCH_CHECK(r_diff_tm.sizes() == a_te.sizes(), "r_diff_tm must be (nti, npi)");
    check_flat_tensor(cos_i, "cos_i", at::kFloat);
    check_flat_tensor(phi_i, "phi_i", at::kFloat);
    check_flat_tensor(cos_o, "cos_o", at::kFloat);
    check_flat_tensor(phi_o, "phi_o", at::kFloat);
    TORCH_CHECK(cos_i.size(0) == d.nti && phi_i.size(0) == d.npi &&
                    cos_o.size(0) == d.nto && phi_o.size(0) == d.npo,
                "axis tensors must match table dims");
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    d.layer_total = static_cast<int>(layer_thickness_m.size(0));
    TORCH_CHECK(layer_eps_r.size(0) == d.layer_total &&
                    layer_sigma_e.size(0) == d.layer_total &&
                    layer_mu_r.size(0) == d.layer_total,
                "layer parameter tensors must share length");
    for (const auto& t : {s_tm, a_te, a_tm, r_diff_te, r_diff_tm, cos_i, phi_i,
                          cos_o, phi_o, layer_thickness_m, layer_eps_r,
                          layer_sigma_e, layer_mu_r}) {
        TORCH_CHECK(t.get_device() == s_te.get_device(),
                    "kirchhoff table build tensors must share device");
    }
    d.iso = d.npi == 1;
    d.n_states = d.nti * d.npi;
    d.d_omega = (1.0f / d.nto) * (kTwoPi / d.npo);
    d.k0 = static_cast<float>(
        2.0 * 3.14159265358979323846 * frequency_hz / em::kSpeedOfLight);
    const float g_max =
        (2.0f * d.k0 * static_cast<float>(sigma_h)) *
        (2.0f * d.k0 * static_cast<float>(sigma_h));
    d.n_terms = static_cast<int>(fmaxf(
        64.0f, g_max + 12.0f * sqrtf(g_max) + 16.0f));
    return d;
}

// Single-material CSR layer_offset=[0], layer_count=[L] device tensors.
std::pair<at::Tensor, at::Tensor> single_material_csr(int layer_total,
                                                      const at::Tensor& reference) {
    auto int_opts = reference.options().dtype(at::kInt);
    auto offset = zero_filled({1}, int_opts);
    auto count = at::full({1}, layer_total, int_opts);
    return {offset, count};
}

// Compute Sbar (steps 1 + 2) and budget_adjoint for one channel. Returns Sbar
// [nti,npi,nto,npo] and budget_adjoint [nti], both on device.
struct ChannelAdjoint {
    at::Tensor sbar;             // [nti,npi,nto,npo]
    at::Tensor budget_adjoint;   // [nti]
};

ChannelAdjoint channel_backward_adjoint(
    const TableDims& d, const at::Tensor& s, const at::Tensor& a,
    const at::Tensor& r_diff, const at::Tensor& cos_o, const at::Tensor& grad_f) {
    // The balance linear algebra (abar/b, J, the solve) and the a_i*a_j-scaled
    // Sbar run in float64; the exported lobe s and factors a are the f32 inputs.
    auto d_opts = s.options().dtype(at::kDouble);
    const int n = d.n_states;
    const int64_t elems = s.numel();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto sbar = at::empty(s.sizes(), d_opts);
    sbar_direct_kernel<<<launch_blocks(elems), kBlockSize, 0, stream>>>(
        elems, d.nti, d.npi, d.nto, d.npo, grad_f.data_ptr<float>(),
        a.data_ptr<float>(), sbar.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto abar = at::empty({n}, d_opts);
    abar_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, grad_f.data_ptr<float>(),
        s.data_ptr<float>(), a.data_ptr<float>(), r_diff.data_ptr<float>(),
        abar.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto smat = at::empty({static_cast<int64_t>(n) * n}, d_opts);
    smat_kernel<float><<<launch_blocks(static_cast<int64_t>(n) * n), kBlockSize, 0,
                         stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, s.data_ptr<float>(),
        smat.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto w_state = at::empty({n}, d_opts);
    auto a_state = at::empty({n}, d_opts);
    auto active = at::empty({n}, s.options().dtype(at::kInt));
    state_weight_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, static_cast<double>(d.d_omega),
        cos_o.data_ptr<float>(), a.data_ptr<float>(), r_diff.data_ptr<float>(),
        w_state.data_ptr<double>(), a_state.data_ptr<double>(),
        active.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto diag = at::empty({n}, d_opts);
    diag_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, smat.data_ptr<double>(), w_state.data_ptr<double>(),
        a_state.data_ptr<double>(), diag.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // A = J^T (column-major); solve A lambda = abar (min-norm least squares).
    auto A = at::empty({static_cast<int64_t>(n) * n}, d_opts);
    assemble_matrix_kernel<<<launch_blocks(static_cast<int64_t>(n) * n), kBlockSize,
                             0, stream>>>(
        n, /*transpose=*/true, smat.data_ptr<double>(), diag.data_ptr<double>(),
        w_state.data_ptr<double>(), a_state.data_ptr<double>(),
        active.data_ptr<int>(), A.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto lambda = abar.clone();  // solve overwrites in place.
    solve_dense(A, lambda, n);

    sbar_implicit_kernel<<<launch_blocks(elems), kBlockSize, 0, stream>>>(
        elems, d.nti, d.npi, d.nto, d.npo, static_cast<double>(d.d_omega),
        cos_o.data_ptr<float>(), a.data_ptr<float>(), lambda.data_ptr<double>(),
        sbar.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto budget_adjoint = at::empty({d.nti}, d_opts);
    budget_adjoint_kernel<<<launch_blocks(d.nti), kBlockSize, 0, stream>>>(
        d.nti, d.npi, lambda.data_ptr<double>(), budget_adjoint.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {sbar, budget_adjoint};
}

}  // namespace

// ==========================================================================
// Symbol 3: channel_kirchhoff_table_build_backward.
// ==========================================================================
pybind11::dict channel_kirchhoff_table_build_backward(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, at::Tensor grad_f_te, at::Tensor grad_f_tm,
    bool need_grad_rough, bool need_grad_layers, bool need_grad_frequency) {
    const c10::cuda::CUDAGuard device_guard(s_te.device());
    const TableDims d = check_inputs(
        s_te, s_tm, a_te, a_tm, r_diff_te, r_diff_tm, cos_i, phi_i, cos_o, phi_o,
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, sigma_h,
        frequency_hz);
    using channel::check_tensor;
    check_tensor(grad_f_te, "grad_f_te", at::kFloat, 4);
    check_tensor(grad_f_tm, "grad_f_tm", at::kFloat, 4);
    TORCH_CHECK(grad_f_te.sizes() == s_te.sizes() &&
                    grad_f_tm.sizes() == s_te.sizes(),
                "grad_f_* must match the table shape");
    grad_f_te = grad_f_te.contiguous();
    grad_f_tm = grad_f_tm.contiguous();

    auto options = s_te.options();
    auto [layer_offset, layer_count] = single_material_csr(d.layer_total, s_te);

    at::Tensor grad_sigma_h, grad_corr_x, grad_corr_y;
    at::Tensor grad_layer_thickness, grad_layer_eps_r, grad_layer_sigma_e;
    at::Tensor grad_frequency;
    if (need_grad_rough) {
        grad_sigma_h = zero_filled({1}, options);
        grad_corr_x = zero_filled({1}, options);
        grad_corr_y = zero_filled({1}, options);
    }
    if (need_grad_layers) {
        grad_layer_thickness = zero_filled({d.layer_total}, options);
        grad_layer_eps_r = zero_filled({d.layer_total}, options);
        grad_layer_sigma_e = zero_filled({d.layer_total}, options);
    }
    if (need_grad_frequency)
        grad_frequency = zero_filled({1}, options);

    const int64_t elems = s_te.numel();
    const int channels[2] = {em::kPolTE, em::kPolTM};
    const at::Tensor* s_ch[2] = {&s_te, &s_tm};
    const at::Tensor* a_ch[2] = {&a_te, &a_tm};
    const at::Tensor* r_ch[2] = {&r_diff_te, &r_diff_tm};
    const at::Tensor* gf_ch[2] = {&grad_f_te, &grad_f_tm};

    for (int c = 0; c < 2; ++c) {
        const ChannelAdjoint adj = channel_backward_adjoint(
            d, *s_ch[c], *a_ch[c], *r_ch[c], cos_o, *gf_ch[c]);
        // Step 3: budget chain.
        if (need_grad_rough || need_grad_layers || need_grad_frequency) {
            budget_chain_backward_kernel<<<launch_blocks(d.nti), kBlockSize, 0,
                                           at::cuda::getCurrentCUDAStream().stream()>>>(
                d.nti, channels[c], static_cast<float>(sigma_h), d.k0,
                static_cast<float>(frequency_hz), cos_i.data_ptr<float>(),
                adj.budget_adjoint.data_ptr<double>(), layer_offset.data_ptr<int>(),
                layer_count.data_ptr<int>(), layer_thickness_m.data_ptr<float>(),
                layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
                layer_mu_r.data_ptr<float>(), d.layer_total,
                need_grad_rough ? grad_sigma_h.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_thickness.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_eps_r.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_sigma_e.data_ptr<float>() : nullptr,
                need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        // Step 4: raw-lobe adjoint over both node sets.
        for (int swapped = 0; swapped < 2; ++swapped) {
            raw_lobe_backward_kernel<<<launch_blocks(elems), kBlockSize, 0,
                                       at::cuda::getCurrentCUDAStream().stream()>>>(
                elems, d.nti, d.npi, d.nto, d.npo, swapped == 1, channels[c],
                static_cast<float>(sigma_h), static_cast<float>(corr_x),
                static_cast<float>(corr_y), d.k0, static_cast<float>(frequency_hz),
                d.n_terms, cos_i.data_ptr<float>(), phi_i.data_ptr<float>(),
                cos_o.data_ptr<float>(), phi_o.data_ptr<float>(),
                adj.sbar.data_ptr<double>(), layer_offset.data_ptr<int>(),
                layer_count.data_ptr<int>(), layer_thickness_m.data_ptr<float>(),
                layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
                layer_mu_r.data_ptr<float>(),
                need_grad_rough ? grad_sigma_h.data_ptr<float>() : nullptr,
                need_grad_rough ? grad_corr_x.data_ptr<float>() : nullptr,
                need_grad_rough ? grad_corr_y.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_thickness.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_eps_r.data_ptr<float>() : nullptr,
                need_grad_layers ? grad_layer_sigma_e.data_ptr<float>() : nullptr,
                need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    pybind11::dict out;
    out["grad_sigma_h"] =
        need_grad_rough ? pybind11::cast(grad_sigma_h) : pybind11::object(pybind11::none());
    out["grad_corr_x"] =
        need_grad_rough ? pybind11::cast(grad_corr_x) : pybind11::object(pybind11::none());
    out["grad_corr_y"] =
        need_grad_rough ? pybind11::cast(grad_corr_y) : pybind11::object(pybind11::none());
    out["grad_layer_thickness_m"] =
        need_grad_layers ? pybind11::cast(grad_layer_thickness) : pybind11::object(pybind11::none());
    out["grad_layer_eps_r"] =
        need_grad_layers ? pybind11::cast(grad_layer_eps_r) : pybind11::object(pybind11::none());
    out["grad_layer_sigma_e"] =
        need_grad_layers ? pybind11::cast(grad_layer_sigma_e) : pybind11::object(pybind11::none());
    out["grad_frequency"] =
        need_grad_frequency ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    return out;
}

// ==========================================================================
// Symbol 4: channel_kirchhoff_table_build_jvp.
// ==========================================================================
namespace {

// Forward pass for one channel: assemble dS (both node sets), drhs, solve
// J da = e, and write tangent_F.
void channel_jvp(
    const TableDims& d, int pol, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, double t_sigma_h, double t_corr_x, double t_corr_y,
    double t_frequency, const at::Tensor& s, const at::Tensor& a,
    const at::Tensor& r_diff, const at::Tensor& cos_i, const at::Tensor& phi_i,
    const at::Tensor& cos_o, const at::Tensor& phi_o,
    const at::Tensor& layer_offset, const at::Tensor& layer_count,
    const at::Tensor& layer_thickness_m, const at::Tensor& layer_eps_r,
    const at::Tensor& layer_sigma_e, const at::Tensor& layer_mu_r,
    const float* t_thickness, const float* t_eps, const float* t_sigma,
    at::Tensor& tangent_f) {
    auto options = s.options();
    auto d_opts = options.dtype(at::kDouble);
    const int n = d.n_states;
    const int64_t elems = s.numel();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // dS over both node sets (double: dS feeds the a_i*a_j-scaled forcing).
    auto dsbar = zero_filled(s.sizes(), d_opts);
    for (int swapped = 0; swapped < 2; ++swapped) {
        raw_lobe_jvp_kernel<<<launch_blocks(elems), kBlockSize, 0, stream>>>(
            elems, d.nti, d.npi, d.nto, d.npo, swapped == 1, pol,
            static_cast<float>(sigma_h), static_cast<float>(corr_x),
            static_cast<float>(corr_y), d.k0, static_cast<float>(frequency_hz),
            d.n_terms, static_cast<float>(t_sigma_h), static_cast<float>(t_corr_x),
            static_cast<float>(t_corr_y), static_cast<float>(t_frequency),
            cos_i.data_ptr<float>(), phi_i.data_ptr<float>(),
            cos_o.data_ptr<float>(), phi_o.data_ptr<float>(),
            layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(),
            t_thickness, t_eps, t_sigma, dsbar.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // drhs per incidence cos.
    auto drhs = at::empty({d.nti}, d_opts);
    budget_chain_jvp_kernel<<<launch_blocks(d.nti), kBlockSize, 0, stream>>>(
        d.nti, pol, static_cast<float>(sigma_h), d.k0,
        static_cast<float>(frequency_hz), static_cast<float>(t_sigma_h),
        static_cast<float>(t_frequency), cos_i.data_ptr<float>(),
        layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
        layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
        layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(),
        t_thickness, t_eps, t_sigma, drhs.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // State-space Smat / dSmat / weights (all float64).
    auto smat = at::empty({static_cast<int64_t>(n) * n}, d_opts);
    smat_kernel<float><<<launch_blocks(static_cast<int64_t>(n) * n), kBlockSize, 0,
                         stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, s.data_ptr<float>(),
        smat.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto dsmat = at::empty({static_cast<int64_t>(n) * n}, d_opts);
    smat_kernel<double><<<launch_blocks(static_cast<int64_t>(n) * n), kBlockSize, 0,
                          stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, dsbar.data_ptr<double>(),
        dsmat.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto w_state = at::empty({n}, d_opts);
    auto a_state = at::empty({n}, d_opts);
    auto active = at::empty({n}, options.dtype(at::kInt));
    state_weight_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, static_cast<double>(d.d_omega),
        cos_o.data_ptr<float>(), a.data_ptr<float>(), r_diff.data_ptr<float>(),
        w_state.data_ptr<double>(), a_state.data_ptr<double>(),
        active.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto diag = at::empty({n}, d_opts);
    diag_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, smat.data_ptr<double>(), w_state.data_ptr<double>(),
        a_state.data_ptr<double>(), diag.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // J (column-major, not transposed) and forcing e; solve J da = e.
    auto J = at::empty({static_cast<int64_t>(n) * n}, d_opts);
    assemble_matrix_kernel<<<launch_blocks(static_cast<int64_t>(n) * n), kBlockSize,
                             0, stream>>>(
        n, /*transpose=*/false, smat.data_ptr<double>(), diag.data_ptr<double>(),
        w_state.data_ptr<double>(), a_state.data_ptr<double>(),
        active.data_ptr<int>(), J.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto e = at::empty({n}, d_opts);
    jvp_forcing_kernel<<<launch_blocks(n), kBlockSize, 0, stream>>>(
        n, d.nti, d.npi, d.nto, d.npo, d.iso, dsmat.data_ptr<double>(),
        w_state.data_ptr<double>(), a_state.data_ptr<double>(), a.data_ptr<float>(),
        drhs.data_ptr<double>(), active.data_ptr<int>(), e.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto da = e.clone();
    solve_dense(J, da, n);

    tangent_f_kernel<<<launch_blocks(elems), kBlockSize, 0, stream>>>(
        elems, d.nti, d.npi, d.nto, d.npo, s.data_ptr<float>(),
        dsbar.data_ptr<double>(), a.data_ptr<float>(), da.data_ptr<double>(),
        tangent_f.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

pybind11::dict channel_kirchhoff_table_build_jvp(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, pybind11::object t_layer_thickness_m,
    pybind11::object t_layer_eps_r, pybind11::object t_layer_sigma_e,
    double t_sigma_h, double t_corr_x, double t_corr_y, double t_frequency) {
    const c10::cuda::CUDAGuard device_guard(s_te.device());
    const TableDims d = check_inputs(
        s_te, s_tm, a_te, a_tm, r_diff_te, r_diff_tm, cos_i, phi_i, cos_o, phi_o,
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, sigma_h,
        frequency_hz);
    auto options = s_te.options();
    at::Tensor storage[3];
    const at::Tensor* t_th = optional_arg(
        std::move(t_layer_thickness_m), storage[0], "t_layer_thickness_m",
        at::kFloat, {d.layer_total}, s_te);
    const at::Tensor* t_ep = optional_arg(
        std::move(t_layer_eps_r), storage[1], "t_layer_eps_r", at::kFloat,
        {d.layer_total}, s_te);
    const at::Tensor* t_si = optional_arg(
        std::move(t_layer_sigma_e), storage[2], "t_layer_sigma_e", at::kFloat,
        {d.layer_total}, s_te);

    auto [layer_offset, layer_count] = single_material_csr(d.layer_total, s_te);
    auto tangent_f_te = at::empty(s_te.sizes(), options);
    auto tangent_f_tm = at::empty(s_te.sizes(), options);

    channel_jvp(d, em::kPolTE, sigma_h, corr_x, corr_y, frequency_hz, t_sigma_h,
                t_corr_x, t_corr_y, t_frequency, s_te, a_te, r_diff_te, cos_i,
                phi_i, cos_o, phi_o, layer_offset, layer_count, layer_thickness_m,
                layer_eps_r, layer_sigma_e, layer_mu_r, opt_ptr<float>(t_th),
                opt_ptr<float>(t_ep), opt_ptr<float>(t_si), tangent_f_te);
    channel_jvp(d, em::kPolTM, sigma_h, corr_x, corr_y, frequency_hz, t_sigma_h,
                t_corr_x, t_corr_y, t_frequency, s_tm, a_tm, r_diff_tm, cos_i,
                phi_i, cos_o, phi_o, layer_offset, layer_count, layer_thickness_m,
                layer_eps_r, layer_sigma_e, layer_mu_r, opt_ptr<float>(t_th),
                opt_ptr<float>(t_ep), opt_ptr<float>(t_si), tangent_f_tm);

    pybind11::dict out;
    out["tangent_f_te"] = tangent_f_te;
    out["tangent_f_tm"] = tangent_f_tm;
    return out;
}
