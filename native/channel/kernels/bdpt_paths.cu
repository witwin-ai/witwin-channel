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

#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
namespace em = rayd::shared::transmission;
namespace utd = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;

// Subpath event codes (event_type). Endpoint and specular events are delta
// events: they never multiply the stored non-delta proposal densities.
constexpr int kEventInvalid = -1;
constexpr int kEventEndpoint = 0;
constexpr int kEventReflectSpecular = 1;
constexpr int kEventTransmitSpecular = 2;

void check_reference(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(tensor.dim() == 2, "reference must have rank 2");
    TORCH_CHECK(tensor.is_contiguous(), "reference must be contiguous");
}

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

__global__ void bdpt_reflected_light_subpaths_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const float* light_pdf_forward,
    const float* light_pdf_reverse,
    const int* light_depth,
    const int* light_component_mask,
    const int* light_tx_id,
    const int* light_rx_id,
    const int* light_grid_linear_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const float* hit_t,
    const float* hit_p,
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
    const int prim = hit_global_prim_id[index];
    const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < material_count;
    const bool material_ok = prim_in_range && material_valid[prim];
    const bool is_valid = light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
    float amplitude = 0.0f;
    if (is_valid) {
        const float reflectance = effective_power_reflectance(
            light_direction + index * 3,
            hit_n + index * 3,
            material_eps_r[prim],
            material_sigma_e[prim],
            material_mu_r[prim],
            frequency_hz);
        // Throughput is a real amplitude proxy: scale it by the AMPLITUDE
        // sqrt(gain * R_eff), not the power reflectance itself.

        amplitude = sqrtf(fmaxf(material_gain[prim], 0.0f) * reflectance);
    }
    float* dst_origin = origin + index * 3;
    float* dst_direction = direction + index * 3;
    if (is_valid) {
        const float* src_direction = light_direction + index * 3;
        const float* hit_point = hit_p + index * 3;
        const float* normal = hit_n + index * 3;
        dst_origin[0] = hit_point[0];
        dst_origin[1] = hit_point[1];
        dst_origin[2] = hit_point[2];
        const float dot =
            src_direction[0] * normal[0] +
            src_direction[1] * normal[1] +
            src_direction[2] * normal[2];
        float rx = src_direction[0] - 2.0f * dot * normal[0];
        float ry = src_direction[1] - 2.0f * dot * normal[1];
        float rz = src_direction[2] - 2.0f * dot * normal[2];
        const float inv_len = rsqrtf(fmaxf(rx * rx + ry * ry + rz * rz, 1.0e-20f));
        dst_direction[0] = rx * inv_len;
        dst_direction[1] = ry * inv_len;
        dst_direction[2] = rz * inv_len;
    } else {
        dst_origin[0] = 0.0f;
        dst_origin[1] = 0.0f;
        dst_origin[2] = 0.0f;
        dst_direction[0] = 0.0f;
        dst_direction[1] = 0.0f;
        dst_direction[2] = 0.0f;
    }
    throughput_real[index] = is_valid ? light_throughput_real[index] * amplitude : 0.0f;
    throughput_imag[index] = is_valid ? light_throughput_imag[index] * amplitude : 0.0f;
    pdf_forward[index] = is_valid ? light_pdf_forward[index] : 0.0f;
    // Ideal specular reflection has unit discrete mass; it does not multiply
    // the stored non-delta proposal density in either orientation.
    pdf_reverse[index] = is_valid ? light_pdf_forward[index] : 0.0f;
    depth[index] = is_valid ? light_depth[index] + 1 : 0;
    component_mask[index] = is_valid ? (light_component_mask[index] | 2) : 0;
    primitive_id[index] = is_valid ? prim : -1;
    edge_id[index] = -1;
    tx_id[index] = is_valid ? light_tx_id[index] : -1;
    rx_id[index] = is_valid ? light_rx_id[index] : -1;
    grid_linear_id[index] = is_valid ? light_grid_linear_id[index] : -1;
    valid[index] = is_valid;
    path_length[index] = is_valid ? light_path_length[index] + fmaxf(hit_t[index], 0.0f) : 0.0f;
    if (is_valid) {
        const int64_t base = index * 3;
        const utd::Complex3 incoming = {
            utd::cplx(light_field_real[base], light_field_imag[base]),
            utd::cplx(light_field_real[base + 1], light_field_imag[base + 1]),
            utd::cplx(light_field_real[base + 2], light_field_imag[base + 2]),
        };
        const utd::float3a incident = utd::make_f3(
            light_direction[base], light_direction[base + 1], light_direction[base + 2]);
        const utd::float3a normal = utd::make_f3(
            hit_n[base], hit_n[base + 1], hit_n[base + 2]);
        utd::float3a reflected;
        const utd::Complex3 updated = transport::reflect_complex3(
            incoming,
            incident,
            normal,
            material_eps_r[prim],
            material_sigma_e[prim],
            material_mu_r[prim],
            material_gain[prim],
            material_thickness[prim],
            frequency_hz,
            reflected);
        field_real[base] = updated.x.re;
        field_real[base + 1] = updated.y.re;
        field_real[base + 2] = updated.z.re;
        field_imag[base] = updated.x.im;
        field_imag[base + 1] = updated.y.im;
        field_imag[base + 2] = updated.z.im;
        source_power[index] = light_source_power[index];
    } else {
        for (int axis = 0; axis < 3; ++axis) {
            field_real[index * 3 + axis] = 0.0f;
            field_imag[index * 3 + axis] = 0.0f;
        }
        source_power[index] = 0.0f;
    }
    event_type[index] = is_valid ? kEventReflectSpecular : kEventInvalid;
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

// Shooting-context specular transmission through a thin-sheet wall.
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
// while path_length grows by the interior chord (unit tested).
__global__ void bdpt_transmitted_light_subpaths_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const float* light_pdf_forward,
    const int* light_depth,
    const int* light_component_mask,
    const int* light_tx_id,
    const int* light_rx_id,
    const int* light_grid_linear_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const float* hit_t,
    const float* hit_p,
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
    const int prim = hit_global_prim_id[index];
    const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < face_count;
    const int material = prim_in_range ? face_material_id[prim] : -1;
    const bool material_ok =
        material >= 0 && static_cast<int64_t>(material) < material_count;
    const bool is_valid =
        light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;

    if (!is_valid) {
        const int64_t base = index * 3;
        for (int axis = 0; axis < 3; ++axis) {
            origin[base + axis] = 0.0f;
            direction[base + axis] = 0.0f;
            field_real[base + axis] = 0.0f;
            field_imag[base + axis] = 0.0f;
        }
        throughput_real[index] = 0.0f;
        throughput_imag[index] = 0.0f;
        pdf_forward[index] = 0.0f;
        pdf_reverse[index] = 0.0f;
        depth[index] = 0;
        component_mask[index] = 0;
        primitive_id[index] = -1;
        edge_id[index] = -1;
        tx_id[index] = -1;
        rx_id[index] = -1;
        grid_linear_id[index] = -1;
        valid[index] = false;
        path_length[index] = 0.0f;
        source_power[index] = 0.0f;
        event_type[index] = kEventInvalid;
        return;
    }

    const int64_t base = index * 3;
    const utd::float3a incident = utd::safe_normalize(
        utd::make_f3(
            light_direction[base],
            light_direction[base + 1],
            light_direction[base + 2]),
        utd::make_f3(0.0f, 0.0f, 1.0f));
    utd::float3a normal_in = utd::safe_normalize(
        utd::make_f3(hit_n[base], hit_n[base + 1], hit_n[base + 2]),
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

    em::LayerView layers{
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        material,
    };
    const em::StackRT te = em::stack_rt(
        cos_theta, layers, frequency_hz, em::kPolTE);
    const em::StackRT tm = em::stack_rt(
        cos_theta, layers, frequency_hz, em::kPolTM);

    // Exact lateral exit point: per-layer Snell angles from the phase index.
    float total_thickness = 0.0f;
    float lateral = 0.0f;
    const int first = layer_offset[material];
    const int layers_in_wall = layer_count[material];
    for (int layer = 0; layer < layers_in_wall; ++layer) {
        const int slot = first + layer;
        const float thickness = fmaxf(layer_thickness_m[slot], 0.0f);
        const em::Medium medium = em::make_medium(
            layer_eps_r[slot], layer_sigma_e[slot], layer_mu_r[slot], omega);
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
        hit_p[base], hit_p[base + 1], hit_p[base + 2]);
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
    const utd::Complex3 incoming = {
        utd::cplx(light_field_real[base], light_field_imag[base]),
        utd::cplx(light_field_real[base + 1], light_field_imag[base + 1]),
        utd::cplx(light_field_real[base + 2], light_field_imag[base + 2]),
    };
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

    origin[base] = exit_point.x;
    origin[base + 1] = exit_point.y;
    origin[base + 2] = exit_point.z;
    direction[base] = incident.x;
    direction[base + 1] = incident.y;
    direction[base + 2] = incident.z;
    field_real[base] = updated.x.re;
    field_real[base + 1] = updated.y.re;
    field_real[base + 2] = updated.z.re;
    field_imag[base] = updated.x.im;
    field_imag[base + 1] = updated.y.im;
    field_imag[base + 2] = updated.z.im;

    float w_s;
    float w_p;
    sp_proxy_weights(incident, normal_in, w_s, w_p);
    // The amplitude proxy is the square root of polarization-weighted power.
    // T_eff mirrors effective_power_reflectance.
    const float effective_transmittance =
        fmaxf(te.cap_t * w_s + tm.cap_t * w_p, 0.0f);
    const float amplitude = sqrtf(effective_transmittance);
    throughput_real[index] = light_throughput_real[index] * amplitude;
    throughput_imag[index] = light_throughput_imag[index] * amplitude;
    // Ideal specular transmission is a delta event with unit discrete mass;
    // it does not multiply the stored non-delta proposal density in either
    // orientation (identical handling to specular reflection above).
    pdf_forward[index] = light_pdf_forward[index];
    pdf_reverse[index] = light_pdf_forward[index];
    depth[index] = light_depth[index] + 1;
    component_mask[index] = light_component_mask[index] | 8;
    primitive_id[index] = prim;
    edge_id[index] = -1;
    tx_id[index] = light_tx_id[index];
    rx_id[index] = light_rx_id[index];
    grid_linear_id[index] = light_grid_linear_id[index];
    valid[index] = true;
    path_length[index] = light_path_length[index] + fmaxf(hit_t[index], 0.0f) + jump;
    source_power[index] = light_source_power[index];
    event_type[index] = kEventTransmitSpecular;
}

}  // namespace

std::vector<at::Tensor> channel_bdpt_empty_subpath_state_cuda(at::Tensor reference) {
    check_reference(reference);
    return allocate_subpath_state(reference, 0);
}

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

std::vector<at::Tensor> channel_bdpt_transmitted_light_subpath_state_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_pdf_forward,
    at::Tensor light_pdf_reverse,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_rx_id,
    at::Tensor light_grid_linear_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor hit_t,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz) {
    check_vec3_table(light_origin, "light.origin");
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_pdf_forward, "light.pdf_forward", at::kFloat);
    check_flat_tensor(light_pdf_reverse, "light.pdf_reverse", at::kFloat);
    check_flat_tensor(light_depth, "light.depth", at::kInt);
    check_flat_tensor(light_component_mask, "light.component_mask", at::kInt);
    check_flat_tensor(light_tx_id, "light.tx_id", at::kInt);
    check_flat_tensor(light_rx_id, "light.rx_id", at::kInt);
    check_flat_tensor(light_grid_linear_id, "light.grid_linear_id", at::kInt);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_flat_tensor(light_path_length, "light.path_length", at::kFloat);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(light_source_power, "light.source_power", at::kFloat);
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_p, "intersection.p");
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(face_material_id, "face_material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(layer_count.size(0) == material_count,
                "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total,
                    "layer parameter tensors must match layer_thickness_m rows");

    const int64_t count = light_origin.size(0);
    for (const auto& tensor : {
             light_direction,
             light_throughput_real,
             light_throughput_imag,
             light_pdf_forward,
             light_pdf_reverse,
             light_depth,
             light_component_mask,
             light_tx_id,
             light_rx_id,
             light_grid_linear_id,
             light_valid,
             light_path_length,
             light_field_real,
             light_field_imag,
             light_source_power,
             hit_t,
             hit_p,
             hit_n,
             hit_global_prim_id,
         }) {
        TORCH_CHECK(tensor.size(0) == count, "transmitted light subpath tensors must share batch size");
        TORCH_CHECK(tensor.get_device() == light_origin.get_device(),
                    "transmitted light subpath tensors must share device");
    }
    for (const auto& tensor : {face_material_id, layer_offset, layer_count,
                               layer_thickness_m, layer_eps_r, layer_sigma_e,
                               layer_mu_r})
        TORCH_CHECK(tensor.get_device() == light_origin.get_device(),
                    "material tensors must share light device");
    auto state = allocate_subpath_state(light_origin, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_transmitted_light_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_pdf_forward.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_component_mask.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_rx_id.data_ptr<int>(),
            light_grid_linear_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_p.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            face_material_id.data_ptr<int>(),
            face_material_id.size(0),
            layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(),
            layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(),
            material_count,
            static_cast<float>(frequency_hz),
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

std::vector<at::Tensor> channel_bdpt_reflected_light_subpath_state_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_pdf_forward,
    at::Tensor light_pdf_reverse,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_rx_id,
    at::Tensor light_grid_linear_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor hit_t,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
    double frequency_hz) {
    check_vec3_table(light_origin, "light.origin");
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_pdf_forward, "light.pdf_forward", at::kFloat);
    check_flat_tensor(light_pdf_reverse, "light.pdf_reverse", at::kFloat);
    check_flat_tensor(light_depth, "light.depth", at::kInt);
    check_flat_tensor(light_component_mask, "light.component_mask", at::kInt);
    check_flat_tensor(light_tx_id, "light.tx_id", at::kInt);
    check_flat_tensor(light_rx_id, "light.rx_id", at::kInt);
    check_flat_tensor(light_grid_linear_id, "light.grid_linear_id", at::kInt);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_flat_tensor(light_path_length, "light.path_length", at::kFloat);
    check_vec3_table(light_field_real, "light.field_real");
    check_vec3_table(light_field_imag, "light.field_imag");
    check_flat_tensor(light_source_power, "light.source_power", at::kFloat);
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_p, "intersection.p");
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    check_flat_tensor(material_eps_r, "material_eps_r", at::kFloat);
    check_flat_tensor(material_sigma_e, "material_sigma_e", at::kFloat);
    check_flat_tensor(material_mu_r, "material_mu_r", at::kFloat);
    check_flat_tensor(material_thickness, "material_thickness", at::kFloat);
    TORCH_CHECK(material_gain.size(0) == material_valid.size(0), "material_gain and material_valid must match");
    TORCH_CHECK(material_eps_r.size(0) == material_gain.size(0), "material_eps_r must match material_gain");
    TORCH_CHECK(material_sigma_e.size(0) == material_gain.size(0), "material_sigma_e must match material_gain");
    TORCH_CHECK(material_mu_r.size(0) == material_gain.size(0), "material_mu_r must match material_gain");
    TORCH_CHECK(material_thickness.size(0) == material_gain.size(0), "material_thickness must match material_gain");

    const int64_t count = light_origin.size(0);
    for (const auto& tensor : {
             light_direction,
             light_throughput_real,
             light_throughput_imag,
              light_pdf_forward,
              light_pdf_reverse,
              light_depth,
              light_component_mask,
              light_tx_id,
             light_rx_id,
             light_grid_linear_id,
             light_valid,
             light_path_length,
             light_field_real,
             light_field_imag,
             light_source_power,
             hit_t,
              hit_p,
              hit_n,
              hit_global_prim_id,
          }) {
        TORCH_CHECK(tensor.size(0) == count, "reflected light subpath tensors must share batch size");
        TORCH_CHECK(tensor.get_device() == light_origin.get_device(), "reflected light subpath tensors must share device");
    }
    TORCH_CHECK(material_gain.get_device() == light_origin.get_device(), "material_gain must share light device");
    TORCH_CHECK(material_valid.get_device() == light_origin.get_device(), "material_valid must share light device");
    auto state = allocate_subpath_state(light_origin, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_reflected_light_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_pdf_forward.data_ptr<float>(),
            light_pdf_reverse.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_component_mask.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_rx_id.data_ptr<int>(),
            light_grid_linear_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            hit_t.data_ptr<float>(),
            hit_p.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            material_eps_r.data_ptr<float>(),
            material_sigma_e.data_ptr<float>(),
            material_mu_r.data_ptr<float>(),
            material_thickness.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            material_gain.size(0),
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

// ==== Section: BDPT subpath AD ====
#include "field_ad.cuh"

#include <src/transmission_device.cuh>

// BDPT AD: backward + jvp companions for the BDPT light-subpath
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
__device__ __forceinline__ DualSC dsc_mul(DualSC a, DualSC b) {
    return {cmath::complex_mul(a.v, b.v), cmath::complex_add(cmath::complex_mul(a.d, b.v), cmath::complex_mul(a.v, b.d))};
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

pybind11::dict channel_bdpt_reflected_light_subpath_state_backward(
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
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
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

pybind11::dict channel_bdpt_reflected_light_subpath_state_jvp(
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
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
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

pybind11::dict channel_bdpt_transmitted_light_subpath_state_backward(
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
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
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

pybind11::dict channel_bdpt_transmitted_light_subpath_state_jvp(
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
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
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

#undef kSubpathEps
#undef kSubpathEpsilon0
