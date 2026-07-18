#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "../em/layer_stack.cuh"
#include "../field_transport.cuh"

#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
namespace em = channel_native::em;
namespace utd = witwin::channel::native_ext;
namespace transport = channel_native::field_transport;

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
// (component_mask MASK_SCATTERING bit, ADR-021 D4):
//
// PRE-scatter it is a REAL-VALUED diagnostic amplitude proxy (contract
// section 5): at specular events it is scaled by the amplitude
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

struct SubpathComplex {
    float r;
    float i;
};

__device__ SubpathComplex sp_c_make(float r, float i) { return {r, i}; }

__device__ SubpathComplex sp_c_add(SubpathComplex a, SubpathComplex b) { return {a.r + b.r, a.i + b.i}; }

__device__ SubpathComplex sp_c_sub(SubpathComplex a, SubpathComplex b) { return {a.r - b.r, a.i - b.i}; }

__device__ SubpathComplex sp_c_mul(SubpathComplex a, SubpathComplex b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

__device__ SubpathComplex sp_c_scale(SubpathComplex a, float s) { return {a.r * s, a.i * s}; }

__device__ SubpathComplex sp_c_div(SubpathComplex a, SubpathComplex b) {
    const float denom = fmaxf(b.r * b.r + b.i * b.i, kSubpathEps);
    return {(a.r * b.r + a.i * b.i) / denom, (a.i * b.r - a.r * b.i) / denom};
}

__device__ SubpathComplex sp_c_sqrt(SubpathComplex z) {
    const float magnitude = hypotf(z.r, z.i);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + z.r)));
    const float imag_sign = z.i < 0.0f ? -1.0f : 1.0f;
    const float imag = imag_sign * sqrtf(fmaxf(0.0f, 0.5f * (magnitude - z.r)));
    return {real, imag};
}

__device__ float sp_c_abs2(SubpathComplex a) { return a.r * a.r + a.i * a.i; }

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
    const SubpathComplex eta = sp_c_make(fmaxf(eps_r, kSubpathEps), -fmaxf(sigma_e, 0.0f) / (omega * kSubpathEpsilon0));
    const float mu_value = fmaxf(mu_r, kSubpathEps);
    const SubpathComplex root = sp_c_sqrt(sp_c_sub(sp_c_scale(eta, mu_value), sp_c_make(sin2, 0.0f)));
    const SubpathComplex mu_cos = sp_c_make(mu_value * cos_theta, 0.0f);
    const SubpathComplex eta_cos = sp_c_scale(eta, cos_theta);
    const SubpathComplex r_te = sp_c_div(sp_c_sub(mu_cos, root), sp_c_add(mu_cos, root));
    const SubpathComplex r_tm = sp_c_div(sp_c_sub(eta_cos, root), sp_c_add(eta_cos, root));

    // s basis = n x incident; p basis = s x incident.
    float sx = ny * iz - nz * iy;
    float sy = nz * ix - nx * iz;
    float sz = nx * iy - ny * ix;
    const float s_len = sqrtf(fmaxf(sx * sx + sy * sy + sz * sz, 0.0f));
    if (s_len <= kSubpathEps) {
        // Normal incidence: r_te == r_tm.
        return sp_c_abs2(r_te);
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
    return sp_c_abs2(r_te) * e_s * e_s + sp_c_abs2(r_tm) * e_p * e_p;
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
        // sqrt(gain * R_eff), not the power reflectance itself (contract
        // section 5).
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

// Shooting-context specular transmission through a thin_sheet wall (contract
// section 4). The outgoing direction equals the incident direction; the ray
// restarts from the exact lateral exit point
//   x_e = x_i - d_total*n_in + (sum_l d_l*tan(theta_l))*u_par
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
//   t * exp(+j*k0*d/cos) * exp(-j*k0*sin*d*tan)
//     = exp(-j*k0*d*(cos - 1/cos + sin^2/cos)) = exp(0) = 1
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
    // Amplitude proxy: sqrt of the pol-weighted power transmittance (contract
    // section 5); T_eff mirrors the effective_power_reflectance construction.
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

std::vector<at::Tensor> cn_bdpt_empty_subpath_state_cuda(at::Tensor reference) {
    check_reference(reference);
    return allocate_subpath_state(reference, 0);
}

std::vector<at::Tensor> cn_bdpt_light_endpoint_subpath_state_cuda(
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

std::vector<at::Tensor> cn_bdpt_sensor_endpoint_subpath_state_cuda(
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

std::vector<at::Tensor> cn_bdpt_transmitted_light_subpath_state_cuda(
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

std::vector<at::Tensor> cn_bdpt_reflected_light_subpath_state_cuda(
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
