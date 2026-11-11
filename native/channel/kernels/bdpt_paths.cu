// Copyright Xingyu Chen.
// Implements BDPT path CUDA operations.

// ==== Section: BDPT subpaths ====
#include "torch_cuda.h"
#include "math.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <rayd/field_transport.cuh>
#include <src/transmission_device.cuh>

#include "bdpt_wavefront.cuh"

#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
namespace em = rayd::shared::transmission;
namespace utd = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have rank 2");
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_flat_tensor(const at::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ float uniform01_from_u64(unsigned long long value) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<float>(((value >> 11) & 0x1fffffffffffffULL) * scale);
}

__device__ void direction_from_seed(unsigned long long seed, float* dir) {
    const float u0 = uniform01_from_u64(splitmix64(seed ^ 0x57c5d1f5f8c0a9b3ULL));
    const float u1 = uniform01_from_u64(splitmix64(seed ^ 0xa24baed4963ee407ULL));
    const float z = 1.0f - 2.0f * u0;
    const float phi = static_cast<float>(2.0 * kPi) * u1;
    const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    dir[0] = radial * cosf(phi);
    dir[1] = radial * sinf(phi);
    dir[2] = z;
}

// SubpathState tensor schema (see _BDPT_SUBPATH_SCHEMA in ops.py).
//
// throughput_real/imag semantics split at the first diffuse-scatter event
// (component_mask MASK_SCATTERING bit, coherent scattering):
//
// Before scattering it is a real-valued diagnostic amplitude proxy. At
// specular events it is scaled by the amplitude
// sqrt(material_gain * R_eff) (reflection) or sqrt(T_eff) (transmission),
// never by the power itself. It may only be used for event/Russian-roulette
// probabilities and MUST NOT enter connection contributions - the Complex3
// Jones field (field_real/field_imag) is the single authoritative amplitude
// carrier (verified: the connection kernels read light_source_power and the
// field tensors only; pre-scatter throughput never reaches a contribution).
//
// POST-scatter (multi-order continuation only; at max_scattering_order == 1
// scattered subpaths terminate) the Complex3 carrier is cleared and
// |throughput|^2 becomes the authoritative UNPOLARIZED power weight of the
// subpath: it is re-seeded at the scatter vertex from the field-based
// incident power (excluding source_power, which connection kernels multiply
// separately) times the unbiased continuation weight, and the specular
// sqrt(gain * R_eff) scaling at the actual incidence angle is thereafter the
// exact unpolarized power transport, not a proxy. Only the torch-side
// scattering NEE glue consumes it (scatter_carried_incident_power); native
// connection kernels still never read it.
std::vector<at::Tensor> allocate_subpath_state(const at::Tensor& reference, int64_t count) {
    auto float_options = reference.options().dtype(at::kFloat);
    auto int_options = reference.options().dtype(at::kInt);
    auto bool_options = reference.options().dtype(at::kBool);
    return {
        at::empty({count, 3}, float_options),
        at::empty({count, 3}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, bool_options),
        at::empty({count}, float_options),
        at::empty({count, 3}, float_options),
        at::empty({count, 3}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, int_options),
    };
}

constexpr float kSubpathEps = 1.0e-9f;
constexpr float kSubpathEpsilon0 = 8.8541878128e-12f;

using SubpathComplex = channel::math::Complex;
namespace cmath = channel::math;

/// Effective power reflectance for the fixed x-hat transmit polarization:
/// |r_te * e_s|^2 + |r_tm * e_p|^2 with e_s/e_p from the transverse-projected
/// polarization (matches the deterministic reflection field convention).
__device__ float effective_power_reflectance(
    const float* incident_dir,
    const float* normal_in,
    float eps_r,
    float sigma_e,
    float mu_r,
    float frequency_hz) {
    float ix = incident_dir[0];
    float iy = incident_dir[1];
    float iz = incident_dir[2];
    const float inv_ilen = rsqrtf(fmaxf(ix * ix + iy * iy + iz * iz, 1.0e-20f));
    ix *= inv_ilen;
    iy *= inv_ilen;
    iz *= inv_ilen;
    float nx = normal_in[0];
    float ny = normal_in[1];
    float nz = normal_in[2];
    const float inv_nlen = rsqrtf(fmaxf(nx * nx + ny * ny + nz * nz, 1.0e-20f));
    nx *= inv_nlen;
    ny *= inv_nlen;
    nz *= inv_nlen;
    float dot_in = ix * nx + iy * ny + iz * nz;
    if (dot_in > 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
        dot_in = -dot_in;
    }
    const float cos_theta = fminf(fmaxf(-dot_in, kSubpathEps), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - cos_theta * cos_theta);
    const float omega = fmaxf(static_cast<float>(2.0 * kPi) * frequency_hz, kSubpathEps);
    const SubpathComplex eta = cmath::complex(fmaxf(eps_r, kSubpathEps), -fmaxf(sigma_e, 0.0f) / (omega * kSubpathEpsilon0));
    const float mu_value = fmaxf(mu_r, kSubpathEps);
    const SubpathComplex root = cmath::complex_sqrt_passive(cmath::complex_sub(cmath::complex_scale(eta, mu_value), cmath::complex(sin2, 0.0f)));
    const SubpathComplex mu_cos = cmath::complex(mu_value * cos_theta, 0.0f);
    const SubpathComplex eta_cos = cmath::complex_scale(eta, cos_theta);
    const SubpathComplex r_te = cmath::complex_div_floor(cmath::complex_sub(mu_cos, root), cmath::complex_add(mu_cos, root), kSubpathEps);
    const SubpathComplex r_tm = cmath::complex_div_floor(cmath::complex_sub(eta_cos, root), cmath::complex_add(eta_cos, root), kSubpathEps);

    // s basis = n x incident; p basis = s x incident.
    float sx = ny * iz - nz * iy;
    float sy = nz * ix - nx * iz;
    float sz = nx * iy - ny * ix;
    const float s_len = sqrtf(fmaxf(sx * sx + sy * sy + sz * sz, 0.0f));
    if (s_len <= kSubpathEps) {
        // Normal incidence: r_te == r_tm.
        return cmath::complex_abs2(r_te);
    }
    sx /= s_len;
    sy /= s_len;
    sz /= s_len;
    const float px = sy * iz - sz * iy;
    const float py = sz * ix - sx * iz;
    const float pz = sx * iy - sy * ix;
    // Transverse projection of the global x-hat transmit polarization.
    float tx_ = 1.0f - ix * ix;
    float ty = -ix * iy;
    float tz = -ix * iz;
    const float t_len = sqrtf(fmaxf(tx_ * tx_ + ty * ty + tz * tz, 0.0f));
    float e_s;
    float e_p;
    if (t_len <= kSubpathEps) {
        e_s = 1.0f;
        e_p = 0.0f;
    } else {
        e_s = (tx_ * sx + ty * sy + tz * sz) / t_len;
        e_p = (tx_ * px + ty * py + tz * pz) / t_len;
    }
    return cmath::complex_abs2(r_te) * e_s * e_s + cmath::complex_abs2(r_tm) * e_p * e_p;
}

// Specular reflection of the incident Jones field at a surface hit. The vertex
// moves to the hit point and the direction mirrors about the shading normal;
// the inputs are read before any output is written, so the outputs may alias
// the incident direction (in-place wavefront update).
__device__ utd::Complex3 bdpt_reflect_vertex(
    const float* incident_direction,
    const float* hit_point,
    const float* hit_normal,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    const utd::Complex3& incoming,
    float* out_origin,
    float* out_direction) {
    const float d0 = incident_direction[0];
    const float d1 = incident_direction[1];
    const float d2 = incident_direction[2];
    const float n0 = hit_normal[0];
    const float n1 = hit_normal[1];
    const float n2 = hit_normal[2];
    const float dot = d0 * n0 + d1 * n1 + d2 * n2;
    float rx = d0 - 2.0f * dot * n0;
    float ry = d1 - 2.0f * dot * n1;
    float rz = d2 - 2.0f * dot * n2;
    const float inv_len = rsqrtf(fmaxf(rx * rx + ry * ry + rz * rz, 1.0e-20f));
    utd::float3a reflected;
    const utd::Complex3 updated = transport::reflect_complex3(
        incoming,
        utd::make_f3(d0, d1, d2),
        utd::make_f3(n0, n1, n2),
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency_hz,
        reflected);
    out_origin[0] = hit_point[0];
    out_origin[1] = hit_point[1];
    out_origin[2] = hit_point[2];
    out_direction[0] = rx * inv_len;
    out_direction[1] = ry * inv_len;
    out_direction[2] = rz * inv_len;
    return updated;
}

__global__ void bdpt_light_endpoint_subpaths_kernel(
    int64_t count,
    const float* tx_positions,
    const float* tx_power,
    const float* tx_polarization,
    const int* launch_tx_id,
    const int64_t* light_seed,
    int64_t tx_count,
    float* origin,
    float* direction,
    float* throughput_real,
    float* throughput_imag,
    float* pdf_forward,
    float* pdf_reverse,
    int* depth,
    int* component_mask,
    int* primitive_id,
    int* edge_id,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    bool* valid,
    float* path_length,
    float* field_real,
    float* field_imag,
    float* source_power,
    int* event_type) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    int tx = launch_tx_id[index];
    const bool is_valid = tx >= 0 && tx < tx_count;
    const float* src = tx_positions + static_cast<int64_t>(is_valid ? tx : 0) * 3;
    float* dst = origin + index * 3;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    float* dir = direction + index * 3;
    if (is_valid) {
        direction_from_seed(static_cast<unsigned long long>(light_seed[index]), dir);
    } else {
        dir[0] = 0.0f;
        dir[1] = 0.0f;
        dir[2] = 0.0f;
    }
    throughput_real[index] = is_valid ? sqrtf(fmaxf(tx_power[tx], 0.0f)) : 0.0f;
    throughput_imag[index] = 0.0f;
    pdf_forward[index] = is_valid ? static_cast<float>(1.0 / (4.0 * kPi)) : 0.0f;
    // Store cumulative non-delta proposal densities. Endpoint masses and
    // specular Dirac masses are classified separately by event_type.
    pdf_reverse[index] = is_valid ? static_cast<float>(1.0 / (4.0 * kPi)) : 0.0f;
    depth[index] = 0;
    component_mask[index] = 1;
    primitive_id[index] = -1;
    edge_id[index] = -1;
    tx_id[index] = tx;
    rx_id[index] = -1;
    grid_linear_id[index] = -1;
    valid[index] = is_valid;
    path_length[index] = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        field_real[index * 3 + axis] = is_valid ? tx_polarization[tx * 3 + axis] : 0.0f;
        field_imag[index * 3 + axis] = 0.0f;
    }
    source_power[index] = is_valid ? tx_power[tx] : 0.0f;
    event_type[index] = is_valid ? 0 : -1;
}

__global__ void bdpt_sensor_endpoint_subpaths_kernel(
    int64_t count,
    const float* rx_positions,
    const float* rx_polarization,
    float* origin,
    float* direction,
    float* throughput_real,
    float* throughput_imag,
    float* pdf_forward,
    float* pdf_reverse,
    int* depth,
    int* component_mask,
    int* primitive_id,
    int* edge_id,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    bool* valid,
    float* path_length,
    float* field_real,
    float* field_imag,
    float* source_power,
    int* event_type) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float* src = rx_positions + index * 3;
    float* dst = origin + index * 3;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    float* dir = direction + index * 3;
    dir[0] = 0.0f;
    dir[1] = 0.0f;
    dir[2] = -1.0f;
    throughput_real[index] = 1.0f;
    throughput_imag[index] = 0.0f;
    pdf_forward[index] = 1.0f;
    pdf_reverse[index] = 1.0f;
    depth[index] = 0;
    component_mask[index] = 1;
    primitive_id[index] = -1;
    edge_id[index] = -1;
    tx_id[index] = -1;
    rx_id[index] = static_cast<int>(index);
    grid_linear_id[index] = static_cast<int>(index);
    valid[index] = true;
    path_length[index] = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        field_real[index * 3 + axis] = rx_polarization[index * 3 + axis];
        field_imag[index * 3 + axis] = 0.0f;
    }
    source_power[index] = 1.0f;
    event_type[index] = 0;
}

// Transverse-projected power weights of the fixed x-hat transmit polarization
// onto the wall s/p basis (the same diagnostic proxy convention as
// effective_power_reflectance). Returns w_s + w_p = 1; degenerate geometry
// (normal incidence or x-hat parallel to the ray) collapses to w_s = 1, which
// is exact there because the TE and TM coefficients coincide.
__device__ void sp_proxy_weights(
    utd::float3a incident,
    utd::float3a normal_in,
    float& w_s,
    float& w_p) {
    utd::float3a s_axis = utd::f3_cross(normal_in, incident);
    const float s_len = utd::safe_length(s_axis);
    if (s_len <= kSubpathEps) {
        w_s = 1.0f;
        w_p = 0.0f;
        return;
    }
    s_axis = utd::f3_div(s_axis, s_len);
    const utd::float3a p_axis = utd::f3_cross(s_axis, incident);
    const utd::float3a x_hat = utd::make_f3(1.0f, 0.0f, 0.0f);
    const utd::float3a transverse = utd::f3_sub(
        x_hat, utd::f3_mul(incident, utd::f3_dot(x_hat, incident)));
    const float t_len = utd::safe_length(transverse);
    if (t_len <= kSubpathEps) {
        w_s = 1.0f;
        w_p = 0.0f;
        return;
    }
    const float e_s = utd::f3_dot(transverse, s_axis) / t_len;
    const float e_p = utd::f3_dot(transverse, p_axis) / t_len;
    w_s = e_s * e_s;
    w_p = e_p * e_p;
}

struct BdptTransmitVertex {
    utd::float3a exit_point;
    utd::float3a direction;
    utd::Complex3 field;
    float amplitude;
    float jump;
};

// Specular transmission of the incident Jones field through the hit wall's
// layer stack. Pure function of its inputs.
// The outgoing direction equals the incident direction; the ray
// restarts from the exact lateral exit point
// x_e = x_i - d_total*n_in + (sum_l d_l*tan(theta_l))*u_par
// with n_in the surface normal flipped toward the incident side, u_par the
// normalized tangential component of the incident direction, and theta_l the
// per-layer Snell angle from the phase index Re(k_l)/k0.
//
// The Jones field is multiplied by diag(t_TE, t_TM) in the wall s/p basis
// (t_stack already carries all interior k_z*d phase/absorption) and
// ADDITIONALLY by exp(+j*k0*||x_e - x_i||) * exp(-j*k_par*|dx_par|):
// the first factor pre-compensates the free-space carrier phase the
// connection kernel later applies over the interior jump (path_length
// includes ||x_e - x_i||), the second is the transverse (lateral chord)
// phase with k_par = k0*sin(theta_i).
//
// Vacuum-layer identity: theta_l = theta_i so x_e = x_i + (d/cos)*d_hat lies
// ON the original ray with jump = d/cos(theta); the combined factor is
// t * exp(+j*k0*d/cos) * exp(-j*k0*sin*d*tan)
// = exp(-j*k0*d*(cos - 1/cos + sin^2/cos)) = exp(0) = 1
// exactly, so a vacuum wall leaves field, ray, and throughput unchanged
// while path_length grows by the interior chord.
__device__ BdptTransmitVertex bdpt_transmit_vertex(
    const float* light_direction,
    const float* hit_p,
    const float* hit_n,
    const utd::Complex3& incoming,
    const em::LayerView& layers,
    float frequency_hz) {
    const utd::float3a incident = utd::safe_normalize(
        utd::make_f3(
            light_direction[0],
            light_direction[1],
            light_direction[2]),
        utd::make_f3(0.0f, 0.0f, 1.0f));
    utd::float3a normal_in = utd::safe_normalize(
        utd::make_f3(hit_n[0], hit_n[1], hit_n[2]),
        utd::make_f3(0.0f, 0.0f, 1.0f));
    // Flip the mean-plane normal toward the incident side.
    if (utd::f3_dot(incident, normal_in) > 0.0f)
        normal_in = utd::f3_neg(normal_in);
    const float cos_theta = fminf(
        fmaxf(-utd::f3_dot(incident, normal_in), static_cast<float>(kSubpathEps)),
        1.0f);
    const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
    const float omega = 2.0f * static_cast<float>(kPi) * frequency_hz;
    const float k0 = omega / transport::kSpeedOfLight;
    const float k_par = k0 * sin_theta;

    const em::StackRT te = em::stack_rt(
        cos_theta, layers, frequency_hz, em::kPolTE);
    const em::StackRT tm = em::stack_rt(
        cos_theta, layers, frequency_hz, em::kPolTM);

    // Exact lateral exit point: per-layer Snell angles from the phase index.
    float total_thickness = 0.0f;
    float lateral = 0.0f;
    const int first = layers.layer_offset[layers.material];
    const int layers_in_wall = layers.layer_count[layers.material];
    for (int layer = 0; layer < layers_in_wall; ++layer) {
        const int slot = first + layer;
        const float thickness = fmaxf(layers.layer_thickness_m[slot], 0.0f);
        const em::Medium medium = em::make_medium(
            layers.layer_eps_r[slot], layers.layer_sigma_e[slot], layers.layer_mu_r[slot], omega);
        const float phase_index = fmaxf(
            medium.k.re / fmaxf(k0, static_cast<float>(kSubpathEps)),
            static_cast<float>(utd::UTD_SMALL_EPS));
        const float sin_layer = sin_theta / phase_index;
        const float cos_layer = sqrtf(
            fmaxf(1.0f - sin_layer * sin_layer, 1.0e-6f));
        total_thickness += thickness;
        lateral += thickness * (sin_layer / cos_layer);
    }
    const utd::float3a u_par = utd::safe_normalize(
        utd::f3_add(incident, utd::f3_mul(normal_in, cos_theta)),
        utd::stable_perp_basis(normal_in, incident));
    const utd::float3a hit_point = utd::make_f3(
        hit_p[0], hit_p[1], hit_p[2]);
    const utd::float3a exit_point = utd::f3_add(
        utd::f3_sub(hit_point, utd::f3_mul(normal_in, total_thickness)),
        utd::f3_mul(u_par, lateral));
    const float jump = utd::safe_length(utd::f3_sub(exit_point, hit_point));

    // Jones diag(t_TE, t_TM) in the wall s/p basis; incident and exit bases
    // coincide because the outgoing direction equals the incident direction.
    utd::float3a s_axis = utd::f3_cross(normal_in, incident);
    s_axis = utd::safe_normalize(
        s_axis, utd::stable_perp_basis(incident, normal_in));
    const utd::float3a p_axis = utd::safe_normalize(
        utd::f3_cross(s_axis, incident),
        utd::stable_perp_basis(incident, s_axis));
    const utd::Complex e_s = transport::complex3_dot_real(incoming, s_axis);
    const utd::Complex e_p = transport::complex3_dot_real(incoming, p_axis);
    utd::Complex3 updated = utd::c3_add(
        utd::cplx_scale_real(s_axis, utd::cplx_mul(te.t, e_s)),
        utd::cplx_scale_real(p_axis, utd::cplx_mul(tm.t, e_p)));
    // exp(+j*k0*jump) * exp(-j*k_par*lateral) == exp(-j*(k_par*lateral - k0*jump)).
    const utd::Complex compensation = em::c_exp_neg_j(
        static_cast<double>(k_par) * static_cast<double>(lateral) -
        static_cast<double>(k0) * static_cast<double>(jump));
    updated = utd::c3_scale(updated, compensation);

    float w_s;
    float w_p;
    sp_proxy_weights(incident, normal_in, w_s, w_p);
    // The amplitude proxy is the square root of polarization-weighted power.
    // T_eff mirrors effective_power_reflectance.
    const float effective_transmittance =
        fmaxf(te.cap_t * w_s + tm.cap_t * w_p, 0.0f);
    return {exit_point, incident, updated, sqrtf(effective_transmittance), jump};
}

// ---- Counter-based sampling streams ----
//
// A BDPT random number is a pure function of (seed, stream, global sample,
// depth): sample g of transmitter t is g = t * samples + i. Chunking the
// wavefront or batching transmitters never changes a draw, and the replay
// of the reverse pass draws the identical numbers.
constexpr unsigned long long kStreamEventSelection = 0x243f6a8885a308d3ULL;
constexpr unsigned long long kStreamScatterU0 = 0x13198a2e03707344ULL;
constexpr unsigned long long kStreamScatterU1 = 0xa4093822299f31d0ULL;

__device__ __forceinline__ float bdpt_counter_uniform(
    unsigned long long seed,
    unsigned long long stream,
    unsigned long long sample,
    unsigned long long depth) {
    unsigned long long x = splitmix64(seed ^ stream);
    x = splitmix64(x ^ ((sample + 1ULL) * 0xd1b54a32d192ed03ULL));
    x = splitmix64(x ^ ((depth + 1ULL) * 0x8cb92ba72f3d8dd7ULL));
    // 24 high bits: exactly representable in float32, always in [0, 1).
    return static_cast<float>(x >> 40) * (1.0f / 16777216.0f);
}

struct BdptWavefrontMaterials {
    int64_t face_count;
    const int* face_material_id;
    const bool* face_valid;
    const float* face_gain;
    const float* face_eps_r;
    const float* face_sigma_e;
    const float* face_mu_r;
    const float* face_thickness;
    int64_t material_count;
    int64_t layer_total;
    const int* layer_offset;
    const int* layer_count;
    const float* layer_thickness_m;
    const float* layer_eps_r;
    const float* layer_sigma_e;
    const float* layer_mu_r;
};

namespace wf = channel::bdpt_wavefront;

}  // namespace

std::vector<at::Tensor> channel_bdpt_light_endpoint_subpath_state_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor launch_tx_id,
    at::Tensor light_seed) {
    check_vec3_table(tx_positions, "tx_positions");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_flat_tensor(launch_tx_id, "launch_tx_id", at::kInt);
    check_flat_tensor(light_seed, "light_seed", at::kLong);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(tx_polarization.size(0) == tx_positions.size(0), "tx_polarization must match tx_positions");
    TORCH_CHECK(light_seed.size(0) == launch_tx_id.size(0), "light_seed must match launch_tx_id");
    TORCH_CHECK(tx_power.get_device() == tx_positions.get_device(), "tx_power must share tx_positions device");
    TORCH_CHECK(launch_tx_id.get_device() == tx_positions.get_device(), "launch_tx_id must share tx_positions device");
    TORCH_CHECK(light_seed.get_device() == tx_positions.get_device(), "light_seed must share tx_positions device");
    auto state = allocate_subpath_state(tx_positions, launch_tx_id.size(0));
    const int64_t count = launch_tx_id.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        bdpt_light_endpoint_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            launch_tx_id.data_ptr<int>(),
            light_seed.data_ptr<int64_t>(),
            tx_positions.size(0),
            state[0].data_ptr<float>(),
            state[1].data_ptr<float>(),
            state[2].data_ptr<float>(),
            state[3].data_ptr<float>(),
            state[4].data_ptr<float>(),
            state[5].data_ptr<float>(),
            state[6].data_ptr<int>(),
            state[7].data_ptr<int>(),
            state[8].data_ptr<int>(),
            state[9].data_ptr<int>(),
            state[10].data_ptr<int>(),
            state[11].data_ptr<int>(),
            state[12].data_ptr<int>(),
            state[13].data_ptr<bool>(),
            state[14].data_ptr<float>(),
            state[15].data_ptr<float>(),
            state[16].data_ptr<float>(),
            state[17].data_ptr<float>(),
            state[18].data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state;
}

std::vector<at::Tensor> channel_bdpt_sensor_endpoint_subpath_state_cuda(
    at::Tensor rx_positions, at::Tensor rx_polarization) {
    check_vec3_table(rx_positions, "rx_positions");
    check_vec3_table(rx_polarization, "rx_polarization");
    TORCH_CHECK(rx_polarization.size(0) == rx_positions.size(0), "rx_polarization must match rx_positions");
    auto state = allocate_subpath_state(rx_positions, rx_positions.size(0));
    const int64_t count = rx_positions.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
        bdpt_sensor_endpoint_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            rx_positions.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            state[0].data_ptr<float>(),
            state[1].data_ptr<float>(),
            state[2].data_ptr<float>(),
            state[3].data_ptr<float>(),
            state[4].data_ptr<float>(),
            state[5].data_ptr<float>(),
            state[6].data_ptr<int>(),
            state[7].data_ptr<int>(),
            state[8].data_ptr<int>(),
            state[9].data_ptr<int>(),
            state[10].data_ptr<int>(),
            state[11].data_ptr<int>(),
            state[12].data_ptr<int>(),
            state[13].data_ptr<bool>(),
            state[14].data_ptr<float>(),
            state[15].data_ptr<float>(),
            state[16].data_ptr<float>(),
            state[17].data_ptr<float>(),
            state[18].data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state;
}

// ==== Section: BDPT wavefront event duals ====
#include "field_ad.cuh"

#include <src/transmission_device.cuh>

// Event duals of the wavefront derivative passes. A reflection or transmission
// event multiplies the carried Complex3 Jones field by a per-hit operator and
// scales the post-scatter carrier by a real amplitude. Under the fixed-topology
// contract the hit geometry, frame, event choice and validity are frozen; the
// differentiable inputs are the EM response parameters (single-slab eps_r /
// sigma_e / gain / thickness for reflection, CSR layer thickness / eps_r /
// sigma_e for transmission), the carrier frequency and the incoming field and
// carrier, with the lockstep duals slab_fresnel_dual and stack_rt_dual.

#define kSubpathEps kSubpathAdEps
#define kSubpathEpsilon0 kSubpathAdEpsilon0

namespace {

namespace ad = rayd::torch::field_transport_ad;

constexpr float kSubpathEps = 1.0e-9f;
constexpr float kSubpathEpsilon0 = 8.8541878128e-12f;

// ---------------------------------------------------------------------------
// Reflection amplitude proxy (effective_power_reflectance) and its dual. This
// is a distinct arithmetic from the slab Jones response: it is the
// single-interface power reflectance driving the real throughput proxy. Only
// eps_r / sigma_e / frequency are live (geometry and mu are frozen); the dual
// mirrors the primal SubpathComplex operations for lockstep.
// ---------------------------------------------------------------------------

using SubC = channel::math::Complex;
namespace cmath = channel::math;

struct DualSC {
    SubC v;
    SubC d;
};

__device__ __forceinline__ DualSC dsc_make(float re, float im, float dre, float dim) {
    return {{re, im}, {dre, dim}};
}
__device__ __forceinline__ DualSC dsc_const(SubC value) { return {value, {0.0f, 0.0f}}; }
__device__ __forceinline__ DualSC dsc_add(DualSC a, DualSC b) {
    return {cmath::complex_add(a.v, b.v), cmath::complex_add(a.d, b.d)};
}
__device__ __forceinline__ DualSC dsc_sub(DualSC a, DualSC b) {
    return {cmath::complex_sub(a.v, b.v), cmath::complex_sub(a.d, b.d)};
}
__device__ __forceinline__ DualSC dsc_scale(DualSC a, float s) {
    return {cmath::complex_scale(a.v, s), cmath::complex_scale(a.d, s)};
}
// Dual of subc_div (regularized denom; clamped branch keeps constant denom).
__device__ __forceinline__ DualSC dsc_div(DualSC a, DualSC b) {
    const float mag2 = b.v.r * b.v.r + b.v.i * b.v.i;
    const float denom = fmaxf(mag2, kSubpathEps);
    DualSC out;
    out.v = cmath::complex_div_floor(a.v, b.v, kSubpathEps);
    const float d_denom = mag2 > kSubpathEps ? 2.0f * (b.v.r * b.d.r + b.v.i * b.d.i) : 0.0f;
    // d(a*conj(b)) = a.d*conj(b) + a.v*conj(b.d)
    const SubC conj_b = {b.v.r, -b.v.i};
    const SubC conj_bd = {b.d.r, -b.d.i};
    const SubC d_num = cmath::complex_add(cmath::complex_mul(a.d, conj_b), cmath::complex_mul(a.v, conj_bd));
    out.d = {
        (d_num.r - out.v.r * d_denom) / denom,
        (d_num.i - out.v.i * d_denom) / denom};
    return out;
}
// Dual of subc_sqrt: dw = dz/(2w) (withheld at the branch point).
__device__ __forceinline__ DualSC dsc_sqrt(DualSC a) {
    DualSC out;
    out.v = cmath::complex_sqrt_passive(a.v);
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

// ---------------------------------------------------------------------------
// Transmitted light subpath advance companions. The wall operator is the CSR
// slab Jones response plus the exact lateral-shift compensation phase
// phi = k_par * lateral - k0 * jump.
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

// ---------------------------------------------------------------------------
// Wavefront event tangents. One reflection or transmission event maps the
// incoming Jones field linearly (F_out = M F_in) and scales the post-scatter
// carrier by a real amplitude. These return the tangent of M F_in and of the
// amplitude for one parameter seed and one incoming field tangent, with the
// same slab Fresnel and layer-stack duals as above. A zero seed with a unit incoming
// tangent e_k gives column k of M (the reverse pass uses M^H).
// ---------------------------------------------------------------------------

struct EventTangent {
    field::Complex3 field;
    float amplitude;
    float d_amplitude;
};

struct ReflectSeed {
    float eps_r;
    float sigma_e;
    float gain;
    float thickness;
    float frequency;
};

__device__ EventTangent wavefront_reflect_tangent(
    const float* incident_direction,
    const float* hit_n,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    const ReflectSeed& seed,
    const field::Complex3& incoming,
    const field::Complex3& t_incoming) {
    const field::float3a incident = load3f(incident_direction, 0);
    const field::float3a normal = load3f(hit_n, 0);
    const transport::ReflectFrame frame = transport::reflect_frame(incident, normal);
    ad::DualC r_te, r_tm;
    ad::slab_fresnel_dual(
        frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz,
        0.0f, seed.eps_r, seed.sigma_e, seed.gain, seed.thickness, seed.frequency, r_te, r_tm);
    const field::Complex e_s = transport::complex3_dot_real(incoming, frame.s_axis);
    const field::Complex e_p = transport::complex3_dot_real(incoming, frame.p_in);
    const field::Complex t_e_s = transport::complex3_dot_real(t_incoming, frame.s_axis);
    const field::Complex t_e_p = transport::complex3_dot_real(t_incoming, frame.p_in);
    const field::Complex t_w_te = field::cplx_add(
        field::cplx_mul(r_te.d, e_s), field::cplx_mul(r_te.v, t_e_s));
    const field::Complex t_w_tm = field::cplx_add(
        field::cplx_mul(r_tm.d, e_p), field::cplx_mul(r_tm.v, t_e_p));
    EventTangent out;
    out.field = field::c3_add(
        field::cplx_scale_real(frame.s_axis, t_w_te),
        field::cplx_scale_real(frame.p_out, t_w_tm));
    const ReflectanceGeometry geo = reflectance_geometry(incident_direction, hit_n);
    float d_reflectance = 0.0f;
    const float reflectance = effective_reflectance_dual(
        geo, eps_r, sigma_e, mu_r, frequency_hz, seed.eps_r, seed.sigma_e, seed.frequency, d_reflectance);
    const float gain_clamped = fmaxf(gain, 0.0f);
    const float t_gain_clamped = (gain >= 0.0f) ? seed.gain : 0.0f;
    const float prod = gain_clamped * reflectance;
    const float t_prod = t_gain_clamped * reflectance + gain_clamped * d_reflectance;
    out.amplitude = sqrtf(fmaxf(prod, 0.0f));
    out.d_amplitude = (prod > 0.0f) ? 0.5f / out.amplitude * t_prod : 0.0f;
    return out;
}

template <typename SeedFn>
__device__ EventTangent wavefront_transmit_tangent(
    const float* incident_direction,
    const float* hit_n,
    const em::LayerView& layers,
    float frequency_hz,
    float t_frequency,
    const SeedFn& seed,
    const field::Complex3& incoming,
    const field::Complex3& t_incoming) {
    const TransmitFrozen geo = transmit_frozen(incident_direction, hit_n, 0);
    const ad::DualStackRT te = ad::stack_rt_dual(
        geo.cos_theta, layers, frequency_hz, 0.0f, t_frequency, em::kPolTE, seed);
    const ad::DualStackRT tm = ad::stack_rt_dual(
        geo.cos_theta, layers, frequency_hz, 0.0f, t_frequency, em::kPolTM, seed);
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
    const DFloat phi = transmit_phi_dual(geo, layers, layers.material, frequency_hz, t_frequency, seed);
    const field::Complex compensation = em::c_exp_neg_j(static_cast<double>(phi.v));
    const field::Complex t_compensation = field::cplx(compensation.im * phi.d, -compensation.re * phi.d);
    EventTangent out;
    out.field = field::c3_add(
        field::c3_scale(t_updated_pre, compensation),
        field::c3_scale(updated_pre, t_compensation));
    float w_s, w_p;
    transmit_proxy_weights(geo.incident, geo.normal_in, w_s, w_p);
    const float raw_transmittance = te.cap_t.v * w_s + tm.cap_t.v * w_p;
    const float t_raw = te.cap_t.d * w_s + tm.cap_t.d * w_p;
    out.amplitude = sqrtf(fmaxf(raw_transmittance, 0.0f));
    out.d_amplitude = (raw_transmittance > 0.0f) ? 0.5f / out.amplitude * t_raw : 0.0f;
    return out;
}

// Per-face tangent of the reflection leaves (null arrays are zero tangents).
__device__ __forceinline__ ReflectSeed reflect_tangent_seed(const channel::bdpt_wavefront::Tangent& tangent, int prim) {
    return {
        tangent.face_eps_r != nullptr ? tangent.face_eps_r[prim] : 0.0f,
        tangent.face_sigma_e != nullptr ? tangent.face_sigma_e[prim] : 0.0f,
        tangent.face_gain != nullptr ? tangent.face_gain[prim] : 0.0f,
        tangent.face_thickness != nullptr ? tangent.face_thickness[prim] : 0.0f,
        tangent.frequency};
}

}  // namespace

// ===========================================================================
// Host bridges.
// ===========================================================================

#undef kSubpathEps
#undef kSubpathEpsilon0

// ==== Section: BDPT wavefront sampler ====

namespace {

constexpr int kStepPrimal = 0;
constexpr int kStepTangent = 1;
constexpr int kStepTape = 2;

// One wavefront bounce of every live light subpath. One counter-based uniform
// partitions scatter (rough faces only) / transmit / reflect; only the selected
// branch is evaluated. Reflect and transmit divide the Jones field by
// sqrt(p_event) and multiply the proposal density by p_event (the rough
// reflect branch also by the coherent attenuation C_r). Scatter samples a
// Kirchhoff lobe direction, clears the Jones field, seeds the unpolarized
// carrier and records the vertex for next-event estimation; it terminates the
// subpath at max_order scatter events. Dead subpaths are untouched.
// counters: [0] transmit, [1] reflect, [2] scatter events; alive counts survivors.
// kStepTangent also carries the forward-mode tangent state; kStepTape records
// each event's incoming state on the replay tape (primal arithmetic unchanged).
template <int Mode>
__global__ void bdpt_wavefront_step_kernel(
    int64_t count,
    int64_t first,
    int depth,
    unsigned long long seed,
    const float* hit_t,
    const float* hit_p,
    const float* hit_n,
    const int* hit_prim,
    BdptWavefrontMaterials materials,
    wf::ScatterTables tables,
    float frequency_hz,
    wf::State state,
    wf::Tangent tangent,
    wf::Tape tape,
    unsigned long long* counters,
    unsigned long long* alive) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool transmitted = false;
    bool reflected = false;
    bool scattered = false;
    if (index < count) {
        // A scatter vertex is connected at the bounce that recorded it only.
        state.component_mask[index] = static_cast<uint8_t>(state.component_mask[index] & ~wf::kMaskScatterVertex);
        if constexpr (Mode == kStepTape) {
            const int64_t slot = static_cast<int64_t>(depth) * tape.capacity + index;
            tape.code[slot] = wf::kTapeNone;
            for (int axis = 0; axis < 3; ++axis) {
                tape.g_field_real[slot * 3 + axis] = 0.0f;
                tape.g_field_imag[slot * 3 + axis] = 0.0f;
            }
            tape.g_p[slot * 2] = 0.0f;
            tape.g_p[slot * 2 + 1] = 0.0f;
        }
    }
    if (index < count && state.valid[index]) {
        const int64_t base = index * 3;
        if (tables.enabled && depth == 0) {
            // The first bounce of a chunk starts every subpath at zero scatter events.
            state.scatter_count[index] = 0;
        }
        float* direction = state.direction + base;
        const int prim = hit_prim[index];
        const float t = hit_t[index];
        const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < materials.face_count;
        const int material = prim_in_range ? materials.face_material_id[prim] : -1;
        const bool hit_ok = prim_in_range && t >= 0.0f && material >= 0;
        const float cos_theta = fminf(
            fmaxf(fabsf(direction[0] * hit_n[base] + direction[1] * hit_n[base + 1] +
                        direction[2] * hit_n[base + 2]),
                  1.0e-6f),
            1.0f);
        const em::LayerView layers{
            materials.layer_offset,
            materials.layer_count,
            materials.layer_thickness_m,
            materials.layer_eps_r,
            materials.layer_sigma_e,
            materials.layer_mu_r,
            material,
        };
        const bool stack_ok = hit_ok && static_cast<int64_t>(material) < materials.material_count &&
            em::layer_material_usable(layers, materials.layer_total);
        float r_eff = 0.0f;
        float t_eff = 0.0f;
        if (stack_ok) {
            const em::StackRT te = em::stack_rt(cos_theta, layers, frequency_hz, em::kPolTE);
            const em::StackRT tm = em::stack_rt(cos_theta, layers, frequency_hz, em::kPolTM);
            r_eff = 0.5f * (te.cap_r + tm.cap_r);
            t_eff = 0.5f * (te.cap_t + tm.cap_t);
        }
        float p_transmit = 0.0f;
        float p_scatter = 0.0f;
        float coherent = 1.0f;
        const bool rough = tables.enabled && hit_ok && static_cast<int64_t>(material) < materials.material_count &&
            tables.scatter_model[material] == 1 && tables.material_slot[material] >= 0;
        if (rough) {
            // Three-way budgets (R_coh, R_diff, T_bar), each positive budget
            // floored at 0.05 and the triple renormalized.
            const float k0 = 6.2831853071795864769f * frequency_hz / 299792458.0f;
            const float roughness = k0 * cos_theta * tables.rough_sigma_h[material];
            coherent = expf(-2.0f * roughness * roughness);
            const float r_coh = r_eff * coherent * coherent;
            const float r_diff = fmaxf(0.0f, r_eff - r_coh);
            const float total = fmaxf(r_coh + r_diff + t_eff, 1.0e-12f);
            float pr = r_coh / total;
            float ps = r_diff / total;
            float pt = t_eff / total;
            if (r_coh > 0.0f) pr = fmaxf(pr, 0.05f);
            if (r_diff > 0.0f) ps = fmaxf(ps, 0.05f);
            if (t_eff > 0.0f) pt = fmaxf(pt, 0.05f);
            const float norm = fmaxf(pr + ps + pt, 1.0e-12f);
            p_scatter = ps / norm;
            p_transmit = pt / norm;
        } else if (hit_ok) {
            const float total = fmaxf(r_eff + t_eff, 1.0e-12f);
            p_transmit = fminf(fmaxf(t_eff / total, 0.05f), 0.95f);
            if (t_eff <= 0.0f) p_transmit = 0.0f;
            if (r_eff <= 0.0f && t_eff > 0.0f) p_transmit = 1.0f;
        }
        const unsigned long long global = static_cast<unsigned long long>(first + index);
        const float u = bdpt_counter_uniform(seed, kStreamEventSelection, global, static_cast<unsigned long long>(depth));
        const bool choose_scatter = hit_ok && u < p_scatter;
        const bool choose_transmit = hit_ok && !choose_scatter && u < p_scatter + p_transmit;
        const bool already_scattered = (state.component_mask[index] & wf::kMaskScattering) != 0;
        const utd::Complex3 incoming = {
            utd::cplx(state.field_real[base], state.field_imag[base]),
            utd::cplx(state.field_real[base + 1], state.field_imag[base + 1]),
            utd::cplx(state.field_real[base + 2], state.field_imag[base + 2]),
        };
        // Incoming tangent state (the launch field has no tangent).
        utd::Complex3 t_incoming = {};
        float t_carrier_in = 0.0f;
        const float carrier_in = already_scattered ? state.carrier[index] : 0.0f;
        const float incident_direction[3] = {direction[0], direction[1], direction[2]};
        const int64_t tape_slot = static_cast<int64_t>(depth) * tape.capacity + index;
        if constexpr (Mode == kStepTangent) {
            if (depth > 0) {
                t_incoming = {
                    utd::cplx(tangent.field_real[base], tangent.field_imag[base]),
                    utd::cplx(tangent.field_real[base + 1], tangent.field_imag[base + 1]),
                    utd::cplx(tangent.field_real[base + 2], tangent.field_imag[base + 2]),
                };
                t_carrier_in = already_scattered ? tangent.carrier[index] : 0.0f;
            }
        }
        if constexpr (Mode == kStepTape) {
            tape.prim[tape_slot] = prim;
            for (int axis = 0; axis < 3; ++axis) {
                tape.direction[tape_slot * 3 + axis] = incident_direction[axis];
                tape.normal[tape_slot * 3 + axis] = hit_n[base + axis];
                tape.field_real[tape_slot * 3 + axis] = state.field_real[base + axis];
                tape.field_imag[tape_slot * 3 + axis] = state.field_imag[base + axis];
            }
            tape.carrier[tape_slot] = carrier_in;
        }
        if (choose_scatter) {
            const int slot = tables.material_slot[material];
            const wf::TableView table = wf::table_view(tables, slot);
            float normal[3] = {hit_n[base], hit_n[base + 1], hit_n[base + 2]};
            const float in[3] = {direction[0], direction[1], direction[2]};
            if (wf::dot_values(in, normal) > 0.0f) {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
            }
            float t1[3];
            float t2[3];
            wf::local_frame(normal, tables.rough_axis_rad[material], t1, t2);
            const float wi_world[3] = {-in[0], -in[1], -in[2]};
            const float wi_local[3] = {wf::dot_values(wi_world, t1), wf::dot_values(wi_world, t2), wf::dot_values(wi_world, normal)};
            float p_te;
            float p_tm;
            float s[3] = {0.0f, 0.0f, 0.0f};
            float p[3] = {0.0f, 0.0f, 0.0f};
            if (already_scattered) {
                // The carried unpolarized power splits evenly over TE/TM.
                const float carried = state.carrier[index] * state.carrier[index];
                p_te = 0.5f * carried;
                p_tm = 0.5f * carried;
            } else {
                wf::cross_values(in, normal, s);
                if (wf::dot_values(s, s) > 1.0e-12f) {
                    wf::unit_values(s);
                } else {
                    float substitute_t2[3];
                    wf::local_frame(in, 0.0f, s, substitute_t2);
                }
                wf::cross_values(s, in, p);
                const float re[3] = {incoming.x.re, incoming.y.re, incoming.z.re};
                const float im[3] = {incoming.x.im, incoming.y.im, incoming.z.im};
                const float sr = wf::dot_values(re, s);
                const float si = wf::dot_values(im, s);
                const float pr = wf::dot_values(re, p);
                const float pi = wf::dot_values(im, p);
                p_te = sr * sr + si * si;
                p_tm = pr * pr + pi * pi;
            }
            const float u0 = bdpt_counter_uniform(seed, kStreamScatterU0, global, static_cast<unsigned long long>(depth));
            const float u1 = bdpt_counter_uniform(seed, kStreamScatterU1, global, static_cast<unsigned long long>(depth));
            float wo_local[3];
            float pdf_lobe = 0.0f;
            float pdf_lobe_reverse = 0.0f;
            rayd::shared::scattering::sample_direction(
                table.marginal, table.conditional, table.density, table.nti, table.npi, table.nto, table.npo, wi_local,
                u0, u1, wo_local, pdf_lobe, pdf_lobe_reverse);
            float f_te = 0.0f;
            float f_tm = 0.0f;
            rayd::shared::scattering::eval_te_tm(
                table.f_te, table.f_tm, table.nti, table.npi, table.nto, table.npo, wi_local, wo_local, f_te, f_tm);
            const float incident_power = fmaxf(p_te + p_tm, 1.0e-20f);
            const float f_weighted = (p_te * f_te + p_tm * f_tm) / incident_power;
            const float cos_o = fmaxf(wo_local[2], 0.0f);
            const bool lobe_ok = pdf_lobe > 0.0f && cos_o > 0.0f;
            if (lobe_ok) {
                const float power_scale =
                    f_weighted * cos_o / (fmaxf(pdf_lobe, 1.0e-12f) * fmaxf(p_scatter, 1.0e-4f));
                const float hit_point[3] = {hit_p[base], hit_p[base + 1], hit_p[base + 2]};
                const float epsilon = wf::scale_epsilon(hit_point, tables.epsilon_floor);
                for (int axis = 0; axis < 3; ++axis) {
                    const float wo_world = wo_local[0] * t1[axis] + wo_local[1] * t2[axis] + wo_local[2] * normal[axis];
                    state.nee_position[base + axis] = hit_point[axis];
                    state.nee_normal[base + axis] = normal[axis];
                    state.nee_wi_local[base + axis] = wi_local[axis];
                    direction[axis] = wo_world;
                    state.origin[base + axis] = hit_point[axis] + wo_world * epsilon;
                    state.field_real[base + axis] = 0.0f;
                    state.field_imag[base + axis] = 0.0f;
                }
                state.nee_p_te[index] = p_te;
                state.nee_p_tm[index] = p_tm;
                state.nee_p_scatter[index] = p_scatter;
                state.nee_material[index] = material;
                state.path_length[index] = state.path_length[index] + fmaxf(t, 0.0f);
                state.nee_path_length[index] = state.path_length[index];
                state.carrier[index] = sqrtf(incident_power) * sqrtf(power_scale);
                const float lobe_scale = cos_o / (fmaxf(pdf_lobe, 1.0e-12f) * fmaxf(p_scatter, 1.0e-4f));
                if constexpr (Mode == kStepTangent) {
                    // The carrier is sqrt((P_te f_te + P_tm f_tm) * lobe_scale); the
                    // lobe density and event probability are frozen.
                    float dp_te;
                    float dp_tm;
                    if (already_scattered) {
                        dp_te = carrier_in * t_carrier_in;
                        dp_tm = dp_te;
                    } else {
                        const float t_re[3] = {t_incoming.x.re, t_incoming.y.re, t_incoming.z.re};
                        const float t_im[3] = {t_incoming.x.im, t_incoming.y.im, t_incoming.z.im};
                        const float re[3] = {incoming.x.re, incoming.y.re, incoming.z.re};
                        const float im[3] = {incoming.x.im, incoming.y.im, incoming.z.im};
                        dp_te = 2.0f * (wf::dot_values(re, s) * wf::dot_values(t_re, s) +
                                        wf::dot_values(im, s) * wf::dot_values(t_im, s));
                        dp_tm = 2.0f * (wf::dot_values(re, p) * wf::dot_values(t_re, p) +
                                        wf::dot_values(im, p) * wf::dot_values(t_im, p));
                    }
                    float df_te = 0.0f;
                    float df_tm = 0.0f;
                    if (tangent.f_te != nullptr) {
                        const int64_t offset = tables.table_offset[slot];
                        rayd::shared::scattering::eval_te_tm(
                            tangent.f_te + offset, tangent.f_tm + offset, table.nti, table.npi, table.nto,
                            table.npo, wi_local, wo_local, df_te, df_tm);
                    }
                    const float kernel = p_te * f_te + p_tm * f_tm;
                    const float d_kernel = dp_te * f_te + p_te * df_te + dp_tm * f_tm + p_tm * df_tm;
                    const float carrier = sqrtf(fmaxf(kernel * lobe_scale, 0.0f));
                    tangent.carrier[index] = carrier > 0.0f ? 0.5f * lobe_scale * d_kernel / carrier : 0.0f;
                    tangent.nee_p_te[index] = dp_te;
                    tangent.nee_p_tm[index] = dp_tm;
                    for (int axis = 0; axis < 3; ++axis) {
                        tangent.field_real[base + axis] = 0.0f;
                        tangent.field_imag[base + axis] = 0.0f;
                    }
                }
                if constexpr (Mode == kStepTape) {
                    float* lanes = tape.scatter + tape_slot * wf::kTapeScatterLanes;
                    for (int axis = 0; axis < 3; ++axis) {
                        lanes[axis] = wi_local[axis];
                        lanes[3 + axis] = wo_local[axis];
                        lanes[6 + axis] = s[axis];
                        lanes[9 + axis] = p[axis];
                    }
                    lanes[12] = p_te;
                    lanes[13] = p_tm;
                    tape.scale[tape_slot] = lobe_scale;
                    tape.code[tape_slot] = static_cast<uint8_t>(
                        wf::kTapeScatter | (already_scattered ? wf::kTapeCarried : 0));
                }
                state.pdf_forward[index] = state.pdf_forward[index] * p_scatter * pdf_lobe;
                state.component_mask[index] =
                    static_cast<uint8_t>(state.component_mask[index] | wf::kMaskScattering | wf::kMaskScatterVertex);
                const int scatter_events = static_cast<int>(state.scatter_count[index]) + 1;
                state.scatter_count[index] = static_cast<uint8_t>(min(scatter_events, 255));
                scattered = true;
                // Order 1 connects and terminates; higher orders continue
                // until the scatter cap.
                state.valid[index] = scatter_events < tables.max_order;
            } else {
                state.valid[index] = false;
            }
        } else {
            utd::Complex3 updated = {};
            EventTangent event_tangent = {};
            float advance = 0.0f;
            float event_amplitude = 1.0f;
            bool ok = false;
            if (choose_transmit) {
                ok = static_cast<int64_t>(material) < materials.material_count;
                if (ok) {
                    if constexpr (Mode == kStepTangent) {
                        const TangentSeed seed{tangent.layer_thickness_m, tangent.layer_eps_r, tangent.layer_sigma_e};
                        event_tangent = wavefront_transmit_tangent(
                            incident_direction, hit_n + base, layers, frequency_hz, tangent.frequency, seed, incoming,
                            t_incoming);
                    }
                    const BdptTransmitVertex vertex = bdpt_transmit_vertex(
                        direction, hit_p + base, hit_n + base, incoming, layers, frequency_hz);
                    state.origin[base] = vertex.exit_point.x;
                    state.origin[base + 1] = vertex.exit_point.y;
                    state.origin[base + 2] = vertex.exit_point.z;
                    direction[0] = vertex.direction.x;
                    direction[1] = vertex.direction.y;
                    direction[2] = vertex.direction.z;
                    updated = vertex.field;
                    advance = fmaxf(t, 0.0f) + vertex.jump;
                    event_amplitude = vertex.amplitude;
                }
            } else {
                ok = prim_in_range && materials.face_valid[prim] && t >= 0.0f;
                if (ok) {
                    if constexpr (Mode == kStepTangent) {
                        event_tangent = wavefront_reflect_tangent(
                            incident_direction, hit_n + base, materials.face_eps_r[prim], materials.face_sigma_e[prim],
                            materials.face_mu_r[prim], materials.face_gain[prim], materials.face_thickness[prim],
                            frequency_hz, reflect_tangent_seed(tangent, prim), incoming, t_incoming);
                    }
                    if (already_scattered) {
                        event_amplitude = sqrtf(fmaxf(materials.face_gain[prim], 0.0f) *
                                                effective_power_reflectance(
                                                    direction, hit_n + base, materials.face_eps_r[prim],
                                                    materials.face_sigma_e[prim], materials.face_mu_r[prim],
                                                    frequency_hz));
                    }
                    updated = bdpt_reflect_vertex(
                        direction,
                        hit_p + base,
                        hit_n + base,
                        materials.face_eps_r[prim],
                        materials.face_sigma_e[prim],
                        materials.face_mu_r[prim],
                        materials.face_gain[prim],
                        materials.face_thickness[prim],
                        frequency_hz,
                        incoming,
                        state.origin + base,
                        direction);
                    advance = fmaxf(t, 0.0f);
                }
            }
            if (!ok) {
                state.valid[index] = false;
            } else {
                const float p_event = choose_transmit ? p_transmit : 1.0f - p_scatter - p_transmit;
                float scale = rsqrtf(fmaxf(p_event, 1.0e-4f));
                if (rough && !choose_transmit) {
                    scale = scale * coherent;
                }
                state.field_real[base] = updated.x.re * scale;
                state.field_real[base + 1] = updated.y.re * scale;
                state.field_real[base + 2] = updated.z.re * scale;
                state.field_imag[base] = updated.x.im * scale;
                state.field_imag[base + 1] = updated.y.im * scale;
                state.field_imag[base + 2] = updated.z.im * scale;
                if (already_scattered) {
                    state.carrier[index] = state.carrier[index] * event_amplitude * scale;
                }
                if constexpr (Mode == kStepTangent) {
                    const utd::Complex3& t_out = event_tangent.field;
                    tangent.field_real[base] = t_out.x.re * scale;
                    tangent.field_real[base + 1] = t_out.y.re * scale;
                    tangent.field_real[base + 2] = t_out.z.re * scale;
                    tangent.field_imag[base] = t_out.x.im * scale;
                    tangent.field_imag[base + 1] = t_out.y.im * scale;
                    tangent.field_imag[base + 2] = t_out.z.im * scale;
                    if (already_scattered) {
                        tangent.carrier[index] =
                            (t_carrier_in * event_amplitude + carrier_in * event_tangent.d_amplitude) * scale;
                    }
                }
                if constexpr (Mode == kStepTape) {
                    tape.scale[tape_slot] = scale;
                    tape.code[tape_slot] = static_cast<uint8_t>(
                        (choose_transmit ? wf::kTapeTransmit : wf::kTapeReflect) |
                        (already_scattered ? wf::kTapeCarried : 0));
                }
                state.pdf_forward[index] = state.pdf_forward[index] * p_event;
                state.path_length[index] = state.path_length[index] + advance;
                state.component_mask[index] = static_cast<uint8_t>(
                    state.component_mask[index] | (choose_transmit ? wf::kMaskTransmission : wf::kMaskReflection));
                transmitted = choose_transmit;
                reflected = !choose_transmit;
            }
        }
    }
    const int block_transmitted = __syncthreads_count(transmitted);
    const int block_reflected = __syncthreads_count(reflected);
    const int block_scattered = __syncthreads_count(scattered);
    const int block_alive = __syncthreads_count(index < count && state.valid[index]);
    if (threadIdx.x == 0) {
        if (block_transmitted > 0) atomicAdd(counters, static_cast<unsigned long long>(block_transmitted));
        if (block_reflected > 0) atomicAdd(counters + 1, static_cast<unsigned long long>(block_reflected));
        if (block_scattered > 0) atomicAdd(counters + 2, static_cast<unsigned long long>(block_scattered));
        if (block_alive + block_scattered > 0) {
            atomicAdd(alive, static_cast<unsigned long long>(block_alive + block_scattered));
        }
    }
}

__device__ __forceinline__ void add_gradient(double* buffer, int64_t index, float value) {
    if (buffer != nullptr && value != 0.0f) {
        atomicAdd(buffer + index, static_cast<double>(value));
    }
}

__device__ __forceinline__ float real_inner(const utd::Complex3& a, const utd::Complex3& b) {
    return a.x.re * b.x.re + a.x.im * b.x.im + a.y.re * b.y.re + a.y.im * b.y.im + a.z.re * b.z.re +
        a.z.im * b.z.im;
}

__device__ __forceinline__ utd::Complex conj_mul(utd::Complex a, utd::Complex b) {
    // conj(a) * b
    return utd::cplx(a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re);
}

// out = scale * M^H cot, with the columns of M from unit incoming tangents.
template <typename ColumnFn>
__device__ utd::Complex3 adjoint_field(const utd::Complex3& cot, float scale, ColumnFn&& column) {
    utd::Complex out[3];
    for (int k = 0; k < 3; ++k) {
        utd::Complex3 unit = {};
        if (k == 0) unit.x = utd::cplx(1.0f, 0.0f);
        if (k == 1) unit.y = utd::cplx(1.0f, 0.0f);
        if (k == 2) unit.z = utd::cplx(1.0f, 0.0f);
        const utd::Complex3 col = column(unit);
        const utd::Complex sum = utd::cplx_add(
            utd::cplx_add(conj_mul(col.x, cot.x), conj_mul(col.y, cot.y)), conj_mul(col.z, cot.z));
        out[k] = utd::cplx(sum.re * scale, sum.im * scale);
    }
    return {out[0], out[1], out[2]};
}

// Reverse pass of one chunk: each sample walks its taped events from the last
// bounce to the first. The field cotangent (dL/dRe + i dL/dIm) and carrier
// cotangent of the state after event b collect the connection cotangents of
// that state, then pull back through the event: reflection and transmission by
// the adjoint of their linear Jones map plus one dual per parameter, scatter
// through P_te/P_tm and the sampled table values. Parameter gradients
// accumulate in double.
__global__ void bdpt_wavefront_reverse_kernel(
    int64_t count,
    int bounces,
    BdptWavefrontMaterials materials,
    wf::ScatterTables tables,
    float frequency_hz,
    wf::Tape tape,
    wf::Gradients grads) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    utd::Complex3 cot = {};
    float cot_carrier = 0.0f;
    float g_frequency = 0.0f;
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        const int64_t slot = static_cast<int64_t>(bounce) * tape.capacity + index;
        const uint8_t code = tape.code[slot];
        cot.x = utd::cplx_add(cot.x, utd::cplx(tape.g_field_real[slot * 3], tape.g_field_imag[slot * 3]));
        cot.y = utd::cplx_add(cot.y, utd::cplx(tape.g_field_real[slot * 3 + 1], tape.g_field_imag[slot * 3 + 1]));
        cot.z = utd::cplx_add(cot.z, utd::cplx(tape.g_field_real[slot * 3 + 2], tape.g_field_imag[slot * 3 + 2]));
        const int event = code & 7;
        const bool carried = (code & wf::kTapeCarried) != 0;
        if (event == wf::kTapeNone) {
            cot = {};
            cot_carrier = 0.0f;
            continue;
        }
        const int prim = tape.prim[slot];
        const int material = materials.face_material_id[prim];
        const float* direction = tape.direction + slot * 3;
        const float* normal = tape.normal + slot * 3;
        const utd::Complex3 incoming = {
            utd::cplx(tape.field_real[slot * 3], tape.field_imag[slot * 3]),
            utd::cplx(tape.field_real[slot * 3 + 1], tape.field_imag[slot * 3 + 1]),
            utd::cplx(tape.field_real[slot * 3 + 2], tape.field_imag[slot * 3 + 2]),
        };
        const float carrier_in = tape.carrier[slot];
        const float scale = tape.scale[slot];
        if (event == wf::kTapeScatter) {
            const float* lanes = tape.scatter + slot * wf::kTapeScatterLanes;
            const float p_te = lanes[12];
            const float p_tm = lanes[13];
            float cot_p_te = tape.g_p[slot * 2];
            float cot_p_tm = tape.g_p[slot * 2 + 1];
            if (cot_carrier != 0.0f) {
                const int table_slot = tables.material_slot[material];
                const wf::TableView table = wf::table_view(tables, table_slot);
                rayd::shared::scattering::TableEvalGrad eval;
                rayd::shared::scattering::eval_te_tm_grad(
                    table.f_te, table.f_tm, table.nti, table.npi, table.nto, table.npo, lanes, lanes + 3, eval);
                const float kernel = p_te * eval.te + p_tm * eval.tm;
                const float carrier = sqrtf(fmaxf(kernel * scale, 0.0f));
                if (carrier > 0.0f) {
                    const float cot_kernel = cot_carrier * 0.5f * scale / carrier;
                    cot_p_te += cot_kernel * eval.te;
                    cot_p_tm += cot_kernel * eval.tm;
                    if (eval.active) {
                        const int64_t offset = tables.table_offset[table_slot];
                        for (int k = 0; k < 16; ++k) {
                            add_gradient(grads.f_te, offset + eval.idx[k], cot_kernel * p_te * eval.cw[k]);
                            add_gradient(grads.f_tm, offset + eval.idx[k], cot_kernel * p_tm * eval.cw[k]);
                        }
                    }
                }
            }
            if (carried) {
                // P_te = P_tm = carrier^2 / 2.
                cot_carrier = carrier_in * (cot_p_te + cot_p_tm);
                cot = {};
            } else {
                const float* s = lanes + 6;
                const float* p = lanes + 9;
                const float re[3] = {incoming.x.re, incoming.y.re, incoming.z.re};
                const float im[3] = {incoming.x.im, incoming.y.im, incoming.z.im};
                const float sr = 2.0f * cot_p_te * wf::dot_values(re, s);
                const float si = 2.0f * cot_p_te * wf::dot_values(im, s);
                const float pr = 2.0f * cot_p_tm * wf::dot_values(re, p);
                const float pi = 2.0f * cot_p_tm * wf::dot_values(im, p);
                cot = {
                    utd::cplx(sr * s[0] + pr * p[0], si * s[0] + pi * p[0]),
                    utd::cplx(sr * s[1] + pr * p[1], si * s[1] + pi * p[1]),
                    utd::cplx(sr * s[2] + pr * p[2], si * s[2] + pi * p[2]),
                };
                cot_carrier = 0.0f;
            }
            continue;
        }
        const bool field_live = real_inner(cot, cot) > 0.0f;
        const bool carrier_live = carried && cot_carrier != 0.0f;
        if (!field_live && !carrier_live) {
            cot = {};
            cot_carrier = 0.0f;
            continue;
        }
        const utd::Complex3 zero_field = {};
        // Gradient of one parameter seed: <cot, scale dF> + cot_carrier * carrier_in * scale * d_amplitude.
        auto seed_gradient = [&](const EventTangent& t) {
            float g = scale * real_inner(cot, t.field);
            if (carrier_live) {
                g += cot_carrier * carrier_in * scale * t.d_amplitude;
            }
            return g;
        };
        float amplitude = 0.0f;
        utd::Complex3 cot_in;
        if (event == wf::kTapeReflect) {
            const float eps_r = materials.face_eps_r[prim];
            const float sigma_e = materials.face_sigma_e[prim];
            const float mu_r = materials.face_mu_r[prim];
            const float gain = materials.face_gain[prim];
            const float thickness = materials.face_thickness[prim];
            auto reflect = [&](const ReflectSeed& seed, const utd::Complex3& t_in) {
                return wavefront_reflect_tangent(
                    direction, normal, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, seed, incoming, t_in);
            };
            const ReflectSeed none{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            cot_in = field_live
                ? adjoint_field(cot, scale, [&](const utd::Complex3& unit) { return reflect(none, unit).field; })
                : zero_field;
            const EventTangent t_eps = reflect({1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, zero_field);
            amplitude = t_eps.amplitude;
            add_gradient(grads.face_eps_r, prim, seed_gradient(t_eps));
            add_gradient(grads.face_sigma_e, prim, seed_gradient(reflect({0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, zero_field)));
            add_gradient(grads.face_gain, prim, seed_gradient(reflect({0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, zero_field)));
            add_gradient(
                grads.face_thickness, prim, seed_gradient(reflect({0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, zero_field)));
            if (grads.frequency != nullptr) {
                g_frequency += seed_gradient(reflect({0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, zero_field));
            }
        } else {
            const em::LayerView layers{
                materials.layer_offset,
                materials.layer_count,
                materials.layer_thickness_m,
                materials.layer_eps_r,
                materials.layer_sigma_e,
                materials.layer_mu_r,
                material,
            };
            cot_in = field_live
                ? adjoint_field(cot, scale, [&](const utd::Complex3& unit) {
                      return wavefront_transmit_tangent(
                          direction, normal, layers, frequency_hz, 0.0f, ZeroSeed{}, incoming, unit).field;
                  })
                : zero_field;
            const int first_layer = materials.layer_offset[material];
            const int layer_total = materials.layer_count[material];
            double* const layer_grads[3] = {grads.layer_thickness_m, grads.layer_eps_r, grads.layer_sigma_e};
            for (int layer = 0; layer < layer_total; ++layer) {
                for (int param = 0; param < 3; ++param) {
                    if (layer_grads[param] == nullptr) {
                        continue;
                    }
                    const EventTangent t = wavefront_transmit_tangent(
                        direction, normal, layers, frequency_hz, 0.0f, BasisSeed{first_layer + layer, param},
                        incoming, zero_field);
                    add_gradient(layer_grads[param], first_layer + layer, seed_gradient(t));
                }
            }
            const EventTangent t_frequency = wavefront_transmit_tangent(
                direction, normal, layers, frequency_hz, 1.0f, ZeroSeed{}, incoming, zero_field);
            amplitude = t_frequency.amplitude;
            if (grads.frequency != nullptr) {
                g_frequency += seed_gradient(t_frequency);
            }
        }
        cot = cot_in;
        cot_carrier = carrier_live ? cot_carrier * amplitude * scale : 0.0f;
    }
    if (grads.frequency != nullptr && g_frequency != 0.0f) {
        atomicAdd(grads.frequency, static_cast<double>(g_frequency));
    }
}

}  // namespace

namespace {

// materials: face material id, valid, gain, eps_r, sigma_e, mu_r, thickness,
// layer offset, layer count, layer thickness, eps_r, sigma_e, mu_r.
BdptWavefrontMaterials wavefront_material_view(const std::vector<at::Tensor>& materials) {
    return {
        materials[0].size(0),
        materials[0].data_ptr<int>(),
        materials[1].data_ptr<bool>(),
        materials[2].data_ptr<float>(),
        materials[3].data_ptr<float>(),
        materials[4].data_ptr<float>(),
        materials[5].data_ptr<float>(),
        materials[6].data_ptr<float>(),
        materials[7].size(0),
        materials[10].size(0),
        materials[7].data_ptr<int>(),
        materials[8].data_ptr<int>(),
        materials[9].data_ptr<float>(),
        materials[10].data_ptr<float>(),
        materials[11].data_ptr<float>(),
        materials[12].data_ptr<float>(),
    };
}
}  // namespace

void channel_bdpt_wavefront_step_cuda(
    int64_t count,
    int64_t first,
    int64_t depth,
    int64_t seed,
    const at::Tensor& hit_t,
    const at::Tensor& hit_p,
    const at::Tensor& hit_n,
    const at::Tensor& hit_prim,
    const std::vector<at::Tensor>& materials,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double frequency_hz,
    const channel::bdpt_wavefront::State& state,
    at::Tensor& counters,
    at::Tensor& alive,
    const channel::bdpt_wavefront::Tangent* tangent,
    const channel::bdpt_wavefront::Tape* tape) {
    if (count == 0) {
        return;
    }
    const BdptWavefrontMaterials view = wavefront_material_view(materials);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(hit_t.get_device()).stream();
    const channel::bdpt_wavefront::Tangent no_tangent{};
    const channel::bdpt_wavefront::Tape no_tape{};
    auto kernel = tangent != nullptr ? bdpt_wavefront_step_kernel<kStepTangent>
        : tape != nullptr            ? bdpt_wavefront_step_kernel<kStepTape>
                                     : bdpt_wavefront_step_kernel<kStepPrimal>;
    kernel<<<blocks, threads, 0, stream>>>(
        count,
        first,
        static_cast<int>(depth),
        static_cast<unsigned long long>(seed),
        hit_t.data_ptr<float>(),
        hit_p.data_ptr<float>(),
        hit_n.data_ptr<float>(),
        hit_prim.data_ptr<int>(),
        view,
        tables,
        static_cast<float>(frequency_hz),
        state,
        tangent != nullptr ? *tangent : no_tangent,
        tape != nullptr ? *tape : no_tape,
        reinterpret_cast<unsigned long long*>(counters.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(alive.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void channel_bdpt_wavefront_reverse_cuda(
    int64_t count,
    int64_t bounces,
    const std::vector<at::Tensor>& materials,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double frequency_hz,
    const channel::bdpt_wavefront::Tape& tape,
    const channel::bdpt_wavefront::Gradients& grads,
    const at::Tensor& reference) {
    if (count == 0 || bounces == 0) {
        return;
    }
    const BdptWavefrontMaterials view = wavefront_material_view(materials);
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    bdpt_wavefront_reverse_kernel<<<blocks, threads, 0, stream>>>(
        count, static_cast<int>(bounces), view, tables, static_cast<float>(frequency_hz), tape, grads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
