// Copyright Xingyu Chen.
// Implements deterministic CUDA operations.

// ==== Section: Deterministic field ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"
#include "math.cuh"
#include <rayd/field_transport.cuh>

#include <cmath>

namespace {

constexpr int kBlockSize = 256;
constexpr float kLightSpeed = 299792458.0f;
constexpr float kEpsilon0 = 8.854187817e-12f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1.0e-6f;

using channel::check_tensor;
namespace transport = rayd::shared::field_transport;
namespace utd = rayd::shared::diffraction;

using channel::math::Complex;
using channel::math::Complex3;
using Float3 = channel::math::Vec3;
namespace cmath = channel::math;

/// -k*d reduced mod 2*pi in double precision: the f32 product loses
/// ~k*d*2^-24 of phase, which shifts coherent multipath nulls at mmWave
/// ranges (matches the reference cplx_exp_neg_kd convention).
__device__ float neg_kd_phase(float k, float d) {
    return transport::precise_neg_kd(k, d);
}

__device__ Float3 orthogonal_transverse(Float3 direction) {
    Float3 axis = fabsf(direction.z) < 0.9f
        ? cmath::vec3(0.0f, 0.0f, 1.0f)
        : cmath::vec3(0.0f, 1.0f, 0.0f);
    return cmath::normalize_min_length(
        cmath::sub(axis, cmath::scale(direction, cmath::dot(axis, direction))),
        kEps);
}

__device__ void fresnel_coefficients(
    float cos_theta_in,
    float eps_r,
    float sigma_e,
    float mu_r,
    float frequency_hz,
    Complex &r_te,
    Complex &r_tm) {
    const float cos_theta = fminf(fmaxf(cos_theta_in, kEps), 1.0f);
    const float omega = fmaxf(2.0f * kPi * frequency_hz, kEps);
    const float eta_r = fmaxf(eps_r, kEps);
    const float sigma = fmaxf(sigma_e, 0.0f);
    const float mu_value = fmaxf(mu_r, kEps);
    utd::Complex shared_te;
    utd::Complex shared_tm;
    transport::legacy_interface_fresnel(
        cos_theta,
        eta_r,
        sigma,
        mu_value,
        omega,
        kEpsilon0,
        kEps,
        shared_te,
        shared_tm);
    r_te = cmath::complex(shared_te.re, shared_te.im);
    r_tm = cmath::complex(shared_tm.re, shared_tm.im);
}

/// Initial transverse polarization: the default transmit polarization
/// projected perpendicular to the launch direction and renormalized.
__device__ Float3 initial_transverse_polarization(Float3 incident) {
    // Default isotropic-transmitter convention: the transmit polarization is
    // world vertical, +z. An isotropic radiator has no preferred frame of its
    // own, so the source vector must be a fixed world direction rather than a
    // per-ray quantity, and every map that shares this transmitter (LoS,
    // reflection, diffraction) must read the same vector to stay
    // polarization-consistent. Vertical is that shared convention; changing it
    // rotates the pattern of every component at once.
    const Float3 tx_pol = cmath::vec3(0.0f, 0.0f, 1.0f);
    Float3 transverse = cmath::sub(tx_pol, cmath::scale(incident, cmath::dot(tx_pol, incident)));
    const float transverse_norm = cmath::length(transverse);
    if (transverse_norm > kEps) {
        return cmath::scale(transverse, 1.0f / transverse_norm);
    }
    return orthogonal_transverse(incident);
}

/// Reflect a complex field 3-vector at a planar interface: decompose into
/// s/p components, apply the Fresnel coefficients, and recompose on the
/// rotated outgoing basis (mirrors RayD's epc_field.cu reflect_field_vector).
__device__ Complex3 reflect_field_vector(
    Complex3 field,
    Float3 incident_direction,
    Float3 normal,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float frequency_hz,
    Float3 &reflected_direction) {
    const Float3 incident = cmath::normalize_min_length(incident_direction, kEps);
    Float3 n = cmath::normalize_min_length(normal, kEps);
    if (cmath::dot(incident, n) > 0.0f) {
        n = cmath::scale(n, -1.0f);
    }
    const float dot_dn = cmath::dot(incident, n);
    reflected_direction = cmath::normalize_min_length(cmath::sub(incident, cmath::scale(n, 2.0f * dot_dn)), kEps);

    Float3 s_hat = cmath::cross(n, incident);
    const float s_norm = cmath::length(s_hat);
    const Float3 transverse_basis = orthogonal_transverse(incident);
    s_hat = s_norm > kEps ? cmath::scale(s_hat, 1.0f / s_norm) : transverse_basis;
    const Float3 p_in = cmath::normalize_min_length(cmath::cross(s_hat, incident), kEps);
    const Float3 p_out = cmath::normalize_min_length(cmath::cross(s_hat, reflected_direction), kEps);

    Complex r_te;
    Complex r_tm;
    fresnel_coefficients(fabsf(dot_dn), eps_r, sigma_e, mu_r, frequency_hz, r_te, r_tm);
    const Complex e_s = cmath::complex3_dot_real(field, s_hat);
    const Complex e_p = cmath::complex3_dot_real(field, p_in);
    Complex3 reflected = cmath::complex3_add(
        cmath::complex3_axis(s_hat, cmath::complex_mul(r_te, e_s)),
        cmath::complex3_axis(p_out, cmath::complex_mul(r_tm, e_p)));
    reflected.x = cmath::complex_scale(reflected.x, gain);
    reflected.y = cmath::complex_scale(reflected.y, gain);
    reflected.z = cmath::complex_scale(reflected.z, gain);
    return reflected;
}

/// Collapse a complex field 3-vector to the exported scalar: total power in
/// path_gain, and a scalar field carrying the dominant component's phase at
/// the total amplitude (same convention as the diffraction vector field).
__device__ void collapse_field_vector(
    Complex3 field,
    float &out_real,
    float &out_imag,
    float &out_gain) {
    const float px = cmath::complex_abs2(field.x);
    const float py = cmath::complex_abs2(field.y);
    const float pz = cmath::complex_abs2(field.z);
    const float total = px + py + pz;
    out_gain = total;

    Complex dominant = field.x;
    float dominant_power = px;
    if (py > dominant_power) {
        dominant_power = py;
        dominant = field.y;
    }
    if (pz > dominant_power) {
        dominant_power = pz;
        dominant = field.z;
    }
    const float dominant_amplitude = sqrtf(fmaxf(dominant_power, 0.0f));
    const float total_amplitude = sqrtf(fmaxf(total, 0.0f));
    if (dominant_amplitude > kEps) {
        const float scale = total_amplitude / dominant_amplitude;
        out_real = dominant.r * scale;
        out_imag = dominant.i * scale;
    } else {
        out_real = 0.0f;
        out_imag = 0.0f;
    }
}

__global__ void deterministic_reflection_field_kernel(
    const float *__restrict__ tx_position,
    const float *__restrict__ rx_position,
    const float *__restrict__ hit_position,
    const float *__restrict__ normal,
    const float *__restrict__ tx_power,
    const float *__restrict__ eps_r,
    const float *__restrict__ sigma_e,
    const float *__restrict__ mu_r,
    const float *__restrict__ gain,
    float frequency_hz,
    int64_t count,
    float *__restrict__ path_gain,
    float *__restrict__ field_real,
    float *__restrict__ field_imag,
    float *__restrict__ path_length_m,
    float *__restrict__ delay_s) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const Float3 tx = cmath::load_vec3(tx_position, index);
        const Float3 rx = cmath::load_vec3(rx_position, index);
        const Float3 hit = cmath::load_vec3(hit_position, index);
        const Float3 incident = cmath::normalize_min_length(cmath::sub(hit, tx), kEps);
        Float3 reflected_direction;
        Complex3 field_vector = reflect_field_vector(
            cmath::complex3_from_real(initial_transverse_polarization(incident)),
            incident,
            cmath::load_vec3(normal, index),
            eps_r[index],
            sigma_e[index],
            mu_r[index],
            gain[index],
            frequency_hz,
            reflected_direction);

        const float segment0 = cmath::length(cmath::sub(hit, tx));
        const float segment1 = cmath::length(cmath::sub(rx, hit));
        const float path_length = fmaxf(segment0 + segment1, kEps);
        const float wavelength = kLightSpeed / frequency_hz;
        const float amplitude = sqrtf(fmaxf(tx_power[index], 0.0f)) * (wavelength / (4.0f * kPi)) / path_length;
        const float phase = neg_kd_phase(2.0f * kPi / wavelength, path_length);
        const Complex scale = cmath::complex_scale(cmath::complex(cosf(phase), sinf(phase)), amplitude);
        field_vector.x = cmath::complex_mul(field_vector.x, scale);
        field_vector.y = cmath::complex_mul(field_vector.y, scale);
        field_vector.z = cmath::complex_mul(field_vector.z, scale);
        collapse_field_vector(field_vector, field_real[index], field_imag[index], path_gain[index]);
        path_length_m[index] = path_length;
        delay_s[index] = path_length / kLightSpeed;
    }
}

__global__ void deterministic_los_field_kernel(
    const float *__restrict__ input_path_gain,
    const float *__restrict__ path_length_m,
    float frequency_hz,
    int64_t count,
    float *__restrict__ path_gain,
    float *__restrict__ field_real,
    float *__restrict__ field_imag) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float gain = fmaxf(input_path_gain[index], 0.0f);
        const float wavelength = kLightSpeed / frequency_hz;
        const float phase = neg_kd_phase(2.0f * kPi / wavelength, path_length_m[index]);
        const float amplitude = sqrtf(gain);
        path_gain[index] = input_path_gain[index];
        field_real[index] = amplitude * cosf(phase);
        field_imag[index] = amplitude * sinf(phase);
    }
}

__global__ void deterministic_diffraction_vector_field_kernel(
    const float *__restrict__ x_re,
    const float *__restrict__ x_im,
    const float *__restrict__ y_re,
    const float *__restrict__ y_im,
    const float *__restrict__ z_re,
    const float *__restrict__ z_im,
    int64_t count,
    float *__restrict__ path_gain,
    float *__restrict__ field_real,
    float *__restrict__ field_imag) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float px = x_re[index] * x_re[index] + x_im[index] * x_im[index];
        const float py = y_re[index] * y_re[index] + y_im[index] * y_im[index];
        const float pz = z_re[index] * z_re[index] + z_im[index] * z_im[index];
        const float total = px + py + pz;
        path_gain[index] = total;

        float real = x_re[index];
        float imag = x_im[index];
        float dominant = px;
        if (py > dominant) {
            dominant = py;
            real = y_re[index];
            imag = y_im[index];
        }
        if (pz > dominant) {
            dominant = pz;
            real = z_re[index];
            imag = z_im[index];
        }

        const float dominant_amplitude = sqrtf(fmaxf(dominant, 0.0f));
        const float total_amplitude = sqrtf(fmaxf(total, 0.0f));
        if (dominant_amplitude > kEps) {
            const float scale = total_amplitude / dominant_amplitude;
            field_real[index] = real * scale;
            field_imag[index] = imag * scale;
        } else {
            field_real[index] = 0.0f;
            field_imag[index] = 0.0f;
        }
    }
}

__global__ void deterministic_reflection_sequence_field_kernel(
    const float *__restrict__ tx_position,
    const float *__restrict__ rx_position,
    const float *__restrict__ hit_positions,
    const float *__restrict__ normals,
    const float *__restrict__ tx_power,
    const float *__restrict__ eps_r,
    const float *__restrict__ sigma_e,
    const float *__restrict__ mu_r,
    const float *__restrict__ gain,
    float frequency_hz,
    int64_t count,
    int64_t depth,
    float *__restrict__ path_gain,
    float *__restrict__ field_real,
    float *__restrict__ field_imag,
    float *__restrict__ path_length_m,
    float *__restrict__ delay_s) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const Float3 tx = cmath::load_vec3(tx_position, index);
        const Float3 rx = cmath::load_vec3(rx_position, index);
        Float3 previous = tx;
        Complex3 field_vector = cmath::complex3_zero();
        bool field_initialized = false;
        float path_length = 0.0f;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const Float3 hit = cmath::load_sequence_vec3(hit_positions, index, bounce, depth);
            const Float3 segment = cmath::sub(hit, previous);
            path_length += cmath::length(segment);
            const Float3 incident = cmath::normalize_min_length(segment, kEps);
            if (!field_initialized) {
                field_vector = cmath::complex3_from_real(initial_transverse_polarization(incident));
                field_initialized = true;
            }
            const int64_t scalar_index = index * depth + bounce;
            Float3 reflected_direction;
            field_vector = reflect_field_vector(
                field_vector,
                incident,
                cmath::load_sequence_vec3(normals, index, bounce, depth),
                eps_r[scalar_index],
                sigma_e[scalar_index],
                mu_r[scalar_index],
                gain[scalar_index],
                frequency_hz,
                reflected_direction);
            previous = hit;
        }

        path_length = fmaxf(path_length + cmath::length(cmath::sub(rx, previous)), kEps);
        const float wavelength = kLightSpeed / frequency_hz;
        const float amplitude = sqrtf(fmaxf(tx_power[index], 0.0f)) * (wavelength / (4.0f * kPi)) / path_length;
        const float phase = neg_kd_phase(2.0f * kPi / wavelength, path_length);
        const Complex scale = cmath::complex_scale(cmath::complex(cosf(phase), sinf(phase)), amplitude);
        field_vector.x = cmath::complex_mul(field_vector.x, scale);
        field_vector.y = cmath::complex_mul(field_vector.y, scale);
        field_vector.z = cmath::complex_mul(field_vector.z, scale);
        collapse_field_vector(field_vector, field_real[index], field_imag[index], path_gain[index]);
        path_length_m[index] = path_length;
        delay_s[index] = path_length / kLightSpeed;
    }
}

__global__ void deterministic_delay_to_path_length_kernel(
    const float *__restrict__ delay_s,
    int64_t count,
    float *__restrict__ path_length_m) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        path_length_m[index] = delay_s[index] * kLightSpeed;
    }
}

__global__ void deterministic_pack_complex_kernel(
    const float *__restrict__ field_real,
    const float *__restrict__ field_imag,
    int64_t count,
    c10::complex<float> *__restrict__ field) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        field[index] = c10::complex<float>(field_real[index], field_imag[index]);
    }
}

__global__ void deterministic_phase_from_field_kernel(
    const float *__restrict__ field_real,
    const float *__restrict__ field_imag,
    int64_t count,
    float *__restrict__ phase_rad) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        float phase = -atan2f(field_imag[index], field_real[index]);
        phase = fmodf(phase, 2.0f * kPi);
        if (phase < 0.0f) {
            phase += 2.0f * kPi;
        }
        phase_rad[index] = phase;
    }
}

__global__ void deterministic_phase_from_length_kernel(
    const float *__restrict__ path_length_m,
    float frequency_hz,
    int64_t count,
    float *__restrict__ phase_rad) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float scale = 2.0f * kPi * frequency_hz / kLightSpeed;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        float phase = fmodf(scale * path_length_m[index], 2.0f * kPi);
        if (phase < 0.0f) {
            phase += 2.0f * kPi;
        }
        phase_rad[index] = phase;
    }
}

__global__ void deterministic_field_from_power_phase_kernel(
    const float *__restrict__ path_gain,
    const float *__restrict__ phase_rad,
    int64_t count,
    float *__restrict__ field_real,
    float *__restrict__ field_imag) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float amplitude = sqrtf(fmaxf(path_gain[index], 0.0f));
        const float phase = -phase_rad[index];
        field_real[index] = amplitude * cosf(phase);
        field_imag[index] = amplitude * sinf(phase);
    }
}

}  // namespace

pybind11::dict channel_deterministic_los_field(
    at::Tensor path_gain_input,
    at::Tensor path_length_m,
    double frequency_hz) {
    check_tensor(path_gain_input, "path_gain", at::kFloat, 1);
    check_tensor(path_length_m, "path_length_m", at::kFloat, 1);
    TORCH_CHECK(path_length_m.sizes() == path_gain_input.sizes(), "path_length_m must match path_gain");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    const int64_t count = path_gain_input.size(0);
    auto path_gain = at::empty({count}, path_gain_input.options());
    auto field_real = at::empty({count}, path_gain_input.options());
    auto field_imag = at::empty({count}, path_gain_input.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain_input.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_los_field_kernel<<<block_count, kBlockSize, 0, stream>>>(
            path_gain_input.data_ptr<float>(),
            path_length_m.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            count,
            path_gain.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["field_real"] = field_real;
    out["field_imag"] = field_imag;
    return out;
}

pybind11::dict channel_deterministic_diffraction_vector_field(
    at::Tensor x_re,
    at::Tensor x_im,
    at::Tensor y_re,
    at::Tensor y_im,
    at::Tensor z_re,
    at::Tensor z_im) {
    check_tensor(x_re, "x_re", at::kFloat, 1);
    check_tensor(x_im, "x_im", at::kFloat, 1);
    check_tensor(y_re, "y_re", at::kFloat, 1);
    check_tensor(y_im, "y_im", at::kFloat, 1);
    check_tensor(z_re, "z_re", at::kFloat, 1);
    check_tensor(z_im, "z_im", at::kFloat, 1);
    TORCH_CHECK(x_im.sizes() == x_re.sizes(), "x_im must match x_re");
    TORCH_CHECK(y_re.sizes() == x_re.sizes(), "y_re must match x_re");
    TORCH_CHECK(y_im.sizes() == x_re.sizes(), "y_im must match x_re");
    TORCH_CHECK(z_re.sizes() == x_re.sizes(), "z_re must match x_re");
    TORCH_CHECK(z_im.sizes() == x_re.sizes(), "z_im must match x_re");

    const int64_t count = x_re.size(0);
    auto path_gain = at::empty({count}, x_re.options());
    auto field_real = at::empty({count}, x_re.options());
    auto field_imag = at::empty({count}, x_re.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_re.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_diffraction_vector_field_kernel<<<block_count, kBlockSize, 0, stream>>>(
            x_re.data_ptr<float>(),
            x_im.data_ptr<float>(),
            y_re.data_ptr<float>(),
            y_im.data_ptr<float>(),
            z_re.data_ptr<float>(),
            z_im.data_ptr<float>(),
            count,
            path_gain.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["field_real"] = field_real;
    out["field_imag"] = field_imag;
    return out;
}

pybind11::dict channel_deterministic_reflection_field(
    at::Tensor tx_position,
    at::Tensor rx_position,
    at::Tensor hit_position,
    at::Tensor normal,
    at::Tensor tx_power,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    double frequency_hz) {
    check_tensor(tx_position, "tx_position", at::kFloat, 2);
    check_tensor(rx_position, "rx_position", at::kFloat, 2);
    check_tensor(hit_position, "hit_position", at::kFloat, 2);
    check_tensor(normal, "normal", at::kFloat, 2);
    check_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_tensor(eps_r, "eps_r", at::kFloat, 1);
    check_tensor(sigma_e, "sigma_e", at::kFloat, 1);
    check_tensor(mu_r, "mu_r", at::kFloat, 1);
    check_tensor(gain, "gain", at::kFloat, 1);
    TORCH_CHECK(tx_position.size(1) == 3, "tx_position must have shape (N, 3)");
    TORCH_CHECK(rx_position.sizes() == tx_position.sizes(), "rx_position must match tx_position");
    TORCH_CHECK(hit_position.sizes() == tx_position.sizes(), "hit_position must match tx_position");
    TORCH_CHECK(normal.sizes() == tx_position.sizes(), "normal must match tx_position");
    const int64_t count = tx_position.size(0);
    TORCH_CHECK(tx_power.size(0) == count, "tx_power must match path count");
    TORCH_CHECK(eps_r.size(0) == count, "eps_r must match path count");
    TORCH_CHECK(sigma_e.size(0) == count, "sigma_e must match path count");
    TORCH_CHECK(mu_r.size(0) == count, "mu_r must match path count");
    TORCH_CHECK(gain.size(0) == count, "gain must match path count");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto path_gain = at::empty({count}, tx_position.options());
    auto field_real = at::empty({count}, tx_position.options());
    auto field_imag = at::empty({count}, tx_position.options());
    auto path_length = at::empty({count}, tx_position.options());
    auto delay = at::empty({count}, tx_position.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_position.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_reflection_field_kernel<<<block_count, kBlockSize, 0, stream>>>(
            tx_position.data_ptr<float>(),
            rx_position.data_ptr<float>(),
            hit_position.data_ptr<float>(),
            normal.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            eps_r.data_ptr<float>(),
            sigma_e.data_ptr<float>(),
            mu_r.data_ptr<float>(),
            gain.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            count,
            path_gain.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["field_real"] = field_real;
    out["field_imag"] = field_imag;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    return out;
}

pybind11::dict channel_deterministic_reflection_sequence_field(
    at::Tensor tx_position,
    at::Tensor rx_position,
    at::Tensor hit_positions,
    at::Tensor normals,
    at::Tensor tx_power,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    double frequency_hz) {
    check_tensor(tx_position, "tx_position", at::kFloat, 2);
    check_tensor(rx_position, "rx_position", at::kFloat, 2);
    check_tensor(hit_positions, "hit_positions", at::kFloat, 3);
    check_tensor(normals, "normals", at::kFloat, 3);
    check_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_tensor(eps_r, "eps_r", at::kFloat, 2);
    check_tensor(sigma_e, "sigma_e", at::kFloat, 2);
    check_tensor(mu_r, "mu_r", at::kFloat, 2);
    check_tensor(gain, "gain", at::kFloat, 2);
    TORCH_CHECK(tx_position.size(1) == 3, "tx_position must have shape (N, 3)");
    TORCH_CHECK(rx_position.sizes() == tx_position.sizes(), "rx_position must match tx_position");
    TORCH_CHECK(hit_positions.size(0) == tx_position.size(0), "hit_positions must match path count");
    TORCH_CHECK(hit_positions.size(2) == 3, "hit_positions must have shape (N, D, 3)");
    TORCH_CHECK(normals.sizes() == hit_positions.sizes(), "normals must match hit_positions");
    const int64_t count = tx_position.size(0);
    const int64_t depth = hit_positions.size(1);
    TORCH_CHECK(depth > 0, "hit_positions depth must be positive");
    TORCH_CHECK(tx_power.size(0) == count, "tx_power must match path count");
    TORCH_CHECK(eps_r.size(0) == count && eps_r.size(1) == depth, "eps_r must have shape (N, D)");
    TORCH_CHECK(sigma_e.sizes() == eps_r.sizes(), "sigma_e must match eps_r");
    TORCH_CHECK(mu_r.sizes() == eps_r.sizes(), "mu_r must match eps_r");
    TORCH_CHECK(gain.sizes() == eps_r.sizes(), "gain must match eps_r");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto path_gain = at::empty({count}, tx_position.options());
    auto field_real = at::empty({count}, tx_position.options());
    auto field_imag = at::empty({count}, tx_position.options());
    auto path_length = at::empty({count}, tx_position.options());
    auto delay = at::empty({count}, tx_position.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_position.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_reflection_sequence_field_kernel<<<block_count, kBlockSize, 0, stream>>>(
            tx_position.data_ptr<float>(),
            rx_position.data_ptr<float>(),
            hit_positions.data_ptr<float>(),
            normals.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            eps_r.data_ptr<float>(),
            sigma_e.data_ptr<float>(),
            mu_r.data_ptr<float>(),
            gain.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            count,
            depth,
            path_gain.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["field_real"] = field_real;
    out["field_imag"] = field_imag;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    return out;
}

at::Tensor channel_deterministic_delay_to_path_length(at::Tensor delay_s) {
    check_tensor(delay_s, "delay_s", at::kFloat, 1);
    auto path_length = at::empty_like(delay_s);
    const int64_t count = delay_s.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(delay_s.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_delay_to_path_length_kernel<<<block_count, kBlockSize, 0, stream>>>(
            delay_s.data_ptr<float>(),
            count,
            path_length.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return path_length;
}

at::Tensor channel_deterministic_pack_complex(at::Tensor field_real, at::Tensor field_imag) {
    check_tensor(field_real, "field_real", at::kFloat, 1);
    check_tensor(field_imag, "field_imag", at::kFloat, 1);
    TORCH_CHECK(field_imag.sizes() == field_real.sizes(), "field_imag must match field_real");
    auto complex_options = field_real.options().dtype(at::kComplexFloat);
    auto field = at::empty({field_real.size(0)}, complex_options);
    const int64_t count = field_real.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(field_real.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_pack_complex_kernel<<<block_count, kBlockSize, 0, stream>>>(
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            count,
            field.data_ptr<c10::complex<float>>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return field;
}

at::Tensor channel_deterministic_phase_from_field(at::Tensor field_real, at::Tensor field_imag) {
    check_tensor(field_real, "field_real", at::kFloat, 1);
    check_tensor(field_imag, "field_imag", at::kFloat, 1);
    TORCH_CHECK(field_imag.sizes() == field_real.sizes(), "field_imag must match field_real");
    auto phase = at::empty_like(field_real);
    const int64_t count = field_real.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(field_real.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_phase_from_field_kernel<<<block_count, kBlockSize, 0, stream>>>(
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            count,
            phase.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return phase;
}

pybind11::dict channel_deterministic_zero_field_phase(at::Tensor reference) {
    check_tensor(reference, "reference", at::kFloat, 1);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    auto path_field = at::empty({reference.size(0)}, complex_options);
    auto phase = at::empty_like(reference);
    const int64_t count = reference.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            path_field.data_ptr<c10::complex<float>>(),
            0,
            static_cast<size_t>(count) * sizeof(c10::complex<float>),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            phase.data_ptr<float>(),
            0,
            static_cast<size_t>(count) * sizeof(float),
            stream));
    }
    pybind11::dict out;
    out["path_field"] = path_field;
    out["phase_rad"] = phase;
    return out;
}

at::Tensor channel_deterministic_phase_from_length(at::Tensor path_length_m, double frequency_hz) {
    check_tensor(path_length_m, "path_length_m", at::kFloat, 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    auto phase = at::empty_like(path_length_m);
    const int64_t count = path_length_m.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_length_m.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_phase_from_length_kernel<<<block_count, kBlockSize, 0, stream>>>(
            path_length_m.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            count,
            phase.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return phase;
}

pybind11::dict channel_deterministic_field_from_power_phase(at::Tensor path_gain, at::Tensor phase_rad) {
    check_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_tensor(phase_rad, "phase_rad", at::kFloat, 1);
    TORCH_CHECK(phase_rad.sizes() == path_gain.sizes(), "phase_rad must match path_gain");
    auto field_real = at::empty_like(path_gain);
    auto field_imag = at::empty_like(path_gain);
    const int64_t count = path_gain.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain.get_device()).stream();
        const int block_count = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        deterministic_field_from_power_phase_kernel<<<block_count, kBlockSize, 0, stream>>>(
            path_gain.data_ptr<float>(),
            phase_rad.data_ptr<float>(),
            count,
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_real"] = field_real;
    out["field_imag"] = field_imag;
    return out;
}

// ==== Section: Deterministic accumulation ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "torch_cuda.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <array>
#include <utility>

#define kBlockSize kAccumBlockSize

namespace {

constexpr int kBlockSize = 256;
// Named component counts exported by channel_deterministic_component_counts
// (los / reflection / diffraction only).
constexpr int kComponentCount = 3;
// Slots materialized by the flat accumulator: los, reflection, diffraction,
// transmission, scattering and coupled. Scattering is an incoherent POWER slot
// (the solver component layout): its rows fold into the totals in the power
// domain and never enter the coherent field sum; its complex cell field is
// kept as a diagnostic only. Coupled is an ordinary coherent field slot
// (coupled reflection and diffraction): reflection-diffraction and its reciprocal both land there and sum
// coherently in-cell, joining field_total / power_total like the first three
// slots.
constexpr int kAccumSlotCount = 6;
constexpr int kScatteringSlot = 4;
constexpr int kCoupledSlot = 5;

// Path component ids: 0=los, 1=reflection, 2=diffraction, 3/4=coupled
// reflection-diffraction and its reciprocal (coupled reflection and diffraction), 7=coupled double
// diffraction (coupled double diffraction). Ids 3/4/7 all map to the single coherent coupled slot
// 5 and sum in-cell. 5=transmission, 6=scattering. Ids without a slot return -1
// and are dropped by the scatter/gather gates.
__device__ __forceinline__ int accum_slot(int component_id) {
    if (component_id >= 0 && component_id < kComponentCount) {
        return component_id;
    }
    if (component_id == 3 || component_id == 4 || component_id == 7) {
        return kCoupledSlot;
    }
    if (component_id == 5) {
        return 3;
    }
    if (component_id == 6) {
        return kScatteringSlot;
    }
    return -1;
}

void check_flat_tensor(const at::Tensor &tensor, const char *name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape (path_count,)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Exact per-dtype intrinsics so the float32 instantiation keeps the primal
// kernel's code byte-identical while the float64 gradcheck companion shares
// the same source.
__device__ __forceinline__ float accum_sqrt(float value) { return sqrtf(value); }
__device__ __forceinline__ double accum_sqrt(double value) { return sqrt(value); }
__device__ __forceinline__ float accum_max_zero(float value) { return fmaxf(value, 0.0f); }
__device__ __forceinline__ double accum_max_zero(double value) { return fmax(value, 0.0); }

template <typename T>
__global__ void deterministic_accumulate_paths_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ path_gain,
    const T *__restrict__ field_real,
    const T *__restrict__ field_imag,
    T *__restrict__ component_power,
    T *__restrict__ component_field_real,
    T *__restrict__ component_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        if (!valid[idx]) {
            continue;
        }
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        atomicAdd(component_power + out, path_gain[idx]);
        atomicAdd(component_field_real + out, field_real[idx]);
        atomicAdd(component_field_imag + out, field_imag[idx]);
    }
}

template <typename T>
__global__ void deterministic_finalize_accumulation_kernel(
    T *__restrict__ component_power,
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    T *__restrict__ power_total,
    T *__restrict__ field_total_real,
    T *__restrict__ field_total_imag,
    int64_t cell_count,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        T real_sum = T(0);
        T imag_sum = T(0);
        T power_sum = T(0);
        T scattering_power = T(0);
        for (int slot = 0; slot < kAccumSlotCount; ++slot) {
            const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
            const T real = component_field_real[out];
            const T imag = component_field_imag[out];
            if (coherent) {
                if (slot == kScatteringSlot) {
                    if (scattering_coherent) {
                        // coherent scattering (opt-in): scattering rows combine
                        // coherently. The slot already holds the summed
                        // complex path field (scattered by the paths
                        // kernel); its |sum|^2 replaces the incoherent gain
                        // sum as the scattering component power and still
                        // folds into power_total as a power term (components
                        // stay mutually incoherent, exactly the coherent combination
                        // per-component phasor precedent).
                        const T coherent_power = real * real + imag * imag;
                        component_power[out] = coherent_power;
                        scattering_power += coherent_power;
                    } else {
                        // Power-domain slot: keep the scattered gains as the
                        // component power and fold them after the field
                        // square.
                        scattering_power += component_power[out];
                    }
                    continue;
                }
                const T coherent_power = real * real + imag * imag;
                component_power[out] = coherent_power;
            } else {
                if (slot == kScatteringSlot && scattering_coherent) {
                    // coherent scattering in an incoherent solve: scattering rows
                    // still interfere with each other, but the combined
                    // power adds incoherently to the other components.
                    const T coherent_power = real * real + imag * imag;
                    component_power[out] = coherent_power;
                    power_sum += coherent_power;
                } else {
                    power_sum += component_power[out];
                }
            }
            real_sum += real;
            imag_sum += imag;
        }
        if (coherent) {
            field_total_real[cell] = real_sum;
            field_total_imag[cell] = imag_sum;
            power_total[cell] =
                real_sum * real_sum + imag_sum * imag_sum + scattering_power;
        } else {
            power_total[cell] = power_sum;
            field_total_real[cell] = accum_sqrt(accum_max_zero(power_sum));
            field_total_imag[cell] = T(0);
        }
    }
}

// VJP of the flat accumulation (the AD contract). Every output is either a linear
// scatter of the per-path field/power (adjoint: gather through the same
// frozen slot/tx/rx gates) or a per-cell |.|^2 / sqrt nonlinearity
// linearized at the saved forward cell values. One gather per path, no
// atomics: dropped rows write exact zeros.
template <typename T>
__global__ void deterministic_accumulate_backward_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    const T *__restrict__ field_total_real,
    const T *__restrict__ field_total_imag,
    const T *__restrict__ power_total,
    const T *__restrict__ grad_power_total,
    const T *__restrict__ grad_field_total_real,
    const T *__restrict__ grad_field_total_imag,
    const T *__restrict__ grad_component_power,
    const T *__restrict__ grad_component_field_real,
    const T *__restrict__ grad_component_field_imag,
    T *__restrict__ grad_path_gain,
    T *__restrict__ grad_field_real,
    T *__restrict__ grad_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        if (!valid[idx]) {
            grad_path_gain[idx] = T(0);
            grad_field_real[idx] = T(0);
            grad_field_imag[idx] = T(0);
            continue;
        }
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            grad_path_gain[idx] = T(0);
            grad_field_real[idx] = T(0);
            grad_field_imag[idx] = T(0);
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        T g_real = T(0);
        T g_imag = T(0);
        T g_gain = T(0);
        // component_field = scatter(field): plain gather in both modes.
        if (grad_component_field_real != nullptr) {
            g_real += grad_component_field_real[out];
        }
        if (grad_component_field_imag != nullptr) {
            g_imag += grad_component_field_imag[out];
        }
        if (scattering_coherent && slot == kScatteringSlot) {
            // coherent scattering / BDPT AD accumulate spec: the scattering slot's
            // component_power and its contribution to power_total (and, in an
            // incoherent solve, to field_total via sqrt) are all |S|^2 with
            // S the summed complex field. d|S|^2 = 2 Re(S) dRe + 2 Im(S) dIm,
            // so every cotangent routes through the field with the pair
            // convention (grad_c = 2 grad_P S); the gain reaches no total.
            const T sr = component_field_real[out];
            const T si = component_field_imag[out];
            if (grad_component_power != nullptr) {
                const T g = grad_component_power[out];
                g_real += T(2) * sr * g;
                g_imag += T(2) * si * g;
            }
            if (grad_power_total != nullptr) {
                const T g = grad_power_total[cell];
                g_real += T(2) * sr * g;
                g_imag += T(2) * si * g;
            }
            if (!coherent && grad_field_total_real != nullptr) {
                const T total = power_total[cell];
                if (total > T(0)) {
                    const T factor =
                        grad_field_total_real[cell] / (T(2) * accum_sqrt(total));
                    g_real += T(2) * sr * factor;
                    g_imag += T(2) * si * factor;
                }
            }
        } else if (coherent && slot == kScatteringSlot) {
            // Power-domain slot inside the coherent totals:
            // component_power = scatter(gain) and power_total adds it
            // linearly after the field square; the cell field is a
            // diagnostic scatter that reaches no total.
            if (grad_component_power != nullptr) {
                g_gain += grad_component_power[out];
            }
            if (grad_power_total != nullptr) {
                g_gain += grad_power_total[cell];
            }
        } else if (coherent) {
            // component_power = |F|^2 and power_total = |sum_s F|^2 +
            // P_scatter with field_total = sum_s F over the coherent slots:
            // d|z|^2 = 2 Re(z) dRe + 2 Im(z) dIm.
            if (grad_component_power != nullptr) {
                const T g = grad_component_power[out];
                g_real += T(2) * component_field_real[out] * g;
                g_imag += T(2) * component_field_imag[out] * g;
            }
            if (grad_field_total_real != nullptr) {
                g_real += grad_field_total_real[cell];
            }
            if (grad_field_total_imag != nullptr) {
                g_imag += grad_field_total_imag[cell];
            }
            if (grad_power_total != nullptr) {
                const T g = grad_power_total[cell];
                g_real += T(2) * field_total_real[cell] * g;
                g_imag += T(2) * field_total_imag[cell] * g;
            }
        } else {
            // component_power = scatter(gain), power_total = sum_s of it and
            // field_total = sqrt(max(power_total, 0)) + 0j: the pseudo-field
            // chain is 1 / (2 sqrt(P)) with a zero subgradient at P <= 0
            // (the primal clamp gates negative sums to a constant zero).
            if (grad_component_power != nullptr) {
                g_gain += grad_component_power[out];
            }
            if (grad_power_total != nullptr) {
                g_gain += grad_power_total[cell];
            }
            if (grad_field_total_real != nullptr) {
                const T total = power_total[cell];
                if (total > T(0)) {
                    g_gain += grad_field_total_real[cell] / (T(2) * accum_sqrt(total));
                }
            }
            // field_total_imag is identically zero in incoherent mode; its
            // cotangent reaches no input.
        }
        grad_path_gain[idx] = g_gain;
        grad_field_real[idx] = g_real;
        grad_field_imag[idx] = g_imag;
    }
}

// JVP scatter: the same frozen-gate atomic scatter as the primal, with each
// tangent stream optional so absent tangents stay exact zeros. Coherent
// cells overwrite the tangent power of every field slot in the finalize, so
// their gain tangents never scatter; the power-domain scattering slot keeps
// its gain tangents in both modes.
template <typename T>
__global__ void deterministic_accumulate_tangent_scatter_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const T *__restrict__ tangent_path_gain,
    const T *__restrict__ tangent_field_real,
    const T *__restrict__ tangent_field_imag,
    T *__restrict__ t_component_power,
    T *__restrict__ t_component_field_real,
    T *__restrict__ t_component_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        if (!valid[idx]) {
            continue;
        }
        const int slot = accum_slot(component_id[idx]);
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (slot < 0 || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
        // Gain tangents scatter into the power buffer only where the finalize
        // derives that slot's power from the gain sum. The scattering slot
        // does so unless the coherent scattering coherent combine is active, in which
        // case its power comes from the summed field tangents instead.
        const bool scatter_gain = (slot == kScatteringSlot)
                                      ? !scattering_coherent
                                      : !coherent;
        if (tangent_path_gain != nullptr && scatter_gain) {
            atomicAdd(t_component_power + out, tangent_path_gain[idx]);
        }
        if (tangent_field_real != nullptr) {
            atomicAdd(t_component_field_real + out, tangent_field_real[idx]);
        }
        if (tangent_field_imag != nullptr) {
            atomicAdd(t_component_field_imag + out, tangent_field_imag[idx]);
        }
    }
}

// JVP finalize: push the scattered tangents through the cell nonlinearities
// linearized at the saved forward cell values. The coherent field total is
// re-summed from the component fields in the same cid order as the primal
// finalize, so the linearization point matches the forward bit for bit.
template <typename T>
__global__ void deterministic_accumulate_jvp_finalize_kernel(
    const T *__restrict__ component_field_real,
    const T *__restrict__ component_field_imag,
    const T *__restrict__ power_total,
    T *__restrict__ t_component_power,
    const T *__restrict__ t_component_field_real,
    const T *__restrict__ t_component_field_imag,
    T *__restrict__ t_power_total,
    T *__restrict__ t_field_total_real,
    T *__restrict__ t_field_total_imag,
    int64_t cell_count,
    int coherent,
    int scattering_coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        T real_sum = T(0);
        T imag_sum = T(0);
        T t_real_sum = T(0);
        T t_imag_sum = T(0);
        T t_power_sum = T(0);
        T t_scattering_power = T(0);
        for (int slot = 0; slot < kAccumSlotCount; ++slot) {
            const int64_t out = static_cast<int64_t>(slot) * cell_count + cell;
            const T t_real = t_component_field_real[out];
            const T t_imag = t_component_field_imag[out];
            if (coherent) {
                if (slot == kScatteringSlot) {
                    if (scattering_coherent) {
                        // coherent scattering: the scattering power is |S|^2 of the
                        // summed field, so its tangent is the linearized
                        // square 2 Re(conj(S) t_S), folded into the total as
                        // a power term (excluded from the coherent field sum).
                        const T real = component_field_real[out];
                        const T imag = component_field_imag[out];
                        const T t_power =
                            T(2) * (real * t_real + imag * t_imag);
                        t_component_power[out] = t_power;
                        t_scattering_power += t_power;
                    } else {
                        // Power-domain slot: its tangent power is the
                        // scattered gain tangents and its field tangent
                        // reaches no total.
                        t_scattering_power += t_component_power[out];
                    }
                    continue;
                }
                const T real = component_field_real[out];
                const T imag = component_field_imag[out];
                t_component_power[out] = T(2) * (real * t_real + imag * t_imag);
                real_sum += real;
                imag_sum += imag;
            } else {
                if (slot == kScatteringSlot && scattering_coherent) {
                    const T real = component_field_real[out];
                    const T imag = component_field_imag[out];
                    const T t_power = T(2) * (real * t_real + imag * t_imag);
                    t_component_power[out] = t_power;
                    t_power_sum += t_power;
                } else {
                    t_power_sum += t_component_power[out];
                }
            }
            t_real_sum += t_real;
            t_imag_sum += t_imag;
        }
        if (coherent) {
            t_field_total_real[cell] = t_real_sum;
            t_field_total_imag[cell] = t_imag_sum;
            t_power_total[cell] =
                T(2) * (real_sum * t_real_sum + imag_sum * t_imag_sum) +
                t_scattering_power;
        } else {
            t_power_total[cell] = t_power_sum;
            const T total = power_total[cell];
            t_field_total_real[cell] =
                total > T(0) ? t_power_sum / (T(2) * accum_sqrt(total)) : T(0);
            t_field_total_imag[cell] = T(0);
        }
    }
}

__global__ void deterministic_component_counts_kernel(
    const int *__restrict__ component_id,
    int64_t path_count,
    unsigned long long *__restrict__ counts) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int cid = component_id[idx];
        if (cid >= 0 && cid < kComponentCount) {
            atomicAdd(counts + cid, 1ULL);
        }
    }
}

__global__ void deterministic_edge_flags_kernel(
    const int *__restrict__ edge_id,
    int64_t path_count,
    int *__restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        flags[idx] = edge_id[idx] >= 0 ? 1 : 0;
    }
}

__global__ void deterministic_compact_edges_kernel(
    const int *__restrict__ edge_id,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    int64_t path_count,
    int *__restrict__ compacted) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        if (flags[idx] == 0) {
            continue;
        }
        compacted[offsets[idx]] = edge_id[idx];
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Tangent accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as field_transport_ad.cu) instead of ATen
// zero-fill.
at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions &options) {
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

const at::Tensor *optional_grad(
    pybind11::object value,
    at::Tensor &storage,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor &reference) {
    if (value.is_none()) {
        return nullptr;
    }
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
const T *grad_ptr(const at::Tensor *tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

void check_cell_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor &reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name, " must share the index device");
}

void check_accumulate_indices(
    const at::Tensor &valid,
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    int64_t num_tx,
    int64_t num_rx) {
    check_flat_tensor(valid, "valid", at::kBool);
    check_flat_tensor(tx_id, "tx_id", at::kInt);
    check_flat_tensor(rx_id, "rx_id", at::kInt);
    check_flat_tensor(component_id, "component_id", at::kInt);
    TORCH_CHECK(valid.sizes() == tx_id.sizes(), "valid must match tx_id");
    TORCH_CHECK(
        valid.get_device() == tx_id.get_device(),
        "valid must share the index device");
    TORCH_CHECK(rx_id.sizes() == tx_id.sizes(), "rx_id must match tx_id");
    TORCH_CHECK(component_id.sizes() == tx_id.sizes(), "component_id must match tx_id");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");
}

template <typename T>
pybind11::dict accumulate_flat_launch(
    const at::Tensor &valid,
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    const at::Tensor &path_gain,
    const at::Tensor &field_real,
    const at::Tensor &field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int scattering_coherent) {
    auto fopts = path_gain.options();
    at::Tensor component_power = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_real = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_imag = at::empty({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor power_total = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_real = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_imag = at::empty({num_tx, num_rx}, fopts);

    const int64_t path_count = tx_id.numel();
    const int64_t cell_count = num_tx * num_rx;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain.get_device()).stream();
    const int64_t component_element_count = static_cast<int64_t>(kAccumSlotCount) * cell_count;
    if (component_element_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_power.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_field_real.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            component_field_imag.data_ptr<T>(),
            0,
            component_element_count * sizeof(T),
            stream));
    }
    if (path_count > 0) {
        deterministic_accumulate_paths_kernel<T>
            <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                valid.data_ptr<bool>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                path_gain.data_ptr<T>(),
                field_real.data_ptr<T>(),
                field_imag.data_ptr<T>(),
                component_power.data_ptr<T>(),
                component_field_real.data_ptr<T>(),
                component_field_imag.data_ptr<T>(),
                path_count,
                num_tx,
                num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (cell_count > 0) {
        deterministic_finalize_accumulation_kernel<T>
            <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                component_power.data_ptr<T>(),
                component_field_real.data_ptr<T>(),
                component_field_imag.data_ptr<T>(),
                power_total.data_ptr<T>(),
                field_total_real.data_ptr<T>(),
                field_total_imag.data_ptr<T>(),
                cell_count,
                coherent ? 1 : 0,
                scattering_coherent);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["power_total"] = power_total;
    out["field_total_real"] = field_total_real;
    out["field_total_imag"] = field_total_imag;
    out["component_power"] = component_power;
    out["component_field_real"] = component_field_real;
    out["component_field_imag"] = component_field_imag;
    return out;
}

pybind11::dict accumulate_flat_checked(
    const at::Tensor &valid,
    const at::Tensor &tx_id,
    const at::Tensor &rx_id,
    const at::Tensor &component_id,
    const at::Tensor &path_gain,
    const at::Tensor &field_real,
    const at::Tensor &field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain,
    c10::ScalarType real_dtype) {
    check_accumulate_indices(valid, tx_id, rx_id, component_id, num_tx, num_rx);
    check_flat_tensor(path_gain, "path_gain", real_dtype);
    check_flat_tensor(field_real, "field_real", real_dtype);
    check_flat_tensor(field_imag, "field_imag", real_dtype);
    TORCH_CHECK(path_gain.sizes() == tx_id.sizes(), "path_gain must match tx_id");
    TORCH_CHECK(field_real.sizes() == tx_id.sizes(), "field_real must match tx_id");
    TORCH_CHECK(field_imag.sizes() == tx_id.sizes(), "field_imag must match tx_id");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    if (real_dtype == at::kFloat) {
        return accumulate_flat_launch<float>(
            valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag,
            num_tx, num_rx, coherent, scattering_coherent);
    }
    return accumulate_flat_launch<double>(
        valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_coherent);
}

}  // namespace

pybind11::dict channel_deterministic_accumulate_flat(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor path_gain,
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    return accumulate_flat_checked(
        valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_combine_domain, at::kFloat);
}

pybind11::dict channel_deterministic_accumulate_flat_fwd64(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor path_gain,
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    return accumulate_flat_checked(
        valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag,
        num_tx, num_rx, coherent, scattering_combine_domain, at::kDouble);
}

pybind11::dict channel_deterministic_accumulate_flat_backward(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor component_field_real,
    at::Tensor component_field_imag,
    at::Tensor field_total_real,
    at::Tensor field_total_imag,
    at::Tensor power_total,
    pybind11::object grad_power_total,
    pybind11::object grad_field_total_real,
    pybind11::object grad_field_total_imag,
    pybind11::object grad_component_power,
    pybind11::object grad_component_field_real,
    pybind11::object grad_component_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    check_accumulate_indices(valid, tx_id, rx_id, component_id, num_tx, num_rx);
    const c10::ScalarType real_dtype = component_field_real.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "deterministic_accumulate_flat_backward requires float32 or float64 cells");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    const std::array<int64_t, 3> component_sizes{kAccumSlotCount, num_tx, num_rx};
    const std::array<int64_t, 2> total_sizes{num_tx, num_rx};
    check_cell_tensor(
        component_field_real, "component_field_real", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        component_field_imag, "component_field_imag", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        field_total_real, "field_total_real", real_dtype, total_sizes, tx_id);
    check_cell_tensor(
        field_total_imag, "field_total_imag", real_dtype, total_sizes, tx_id);
    check_cell_tensor(power_total, "power_total", real_dtype, total_sizes, tx_id);
    at::Tensor gpt_storage;
    at::Tensor gftr_storage;
    at::Tensor gfti_storage;
    at::Tensor gcp_storage;
    at::Tensor gcfr_storage;
    at::Tensor gcfi_storage;
    const at::Tensor *gpt = optional_grad(
        std::move(grad_power_total), gpt_storage, "grad_power_total",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gftr = optional_grad(
        std::move(grad_field_total_real), gftr_storage, "grad_field_total_real",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gfti = optional_grad(
        std::move(grad_field_total_imag), gfti_storage, "grad_field_total_imag",
        real_dtype, total_sizes, tx_id);
    const at::Tensor *gcp = optional_grad(
        std::move(grad_component_power), gcp_storage, "grad_component_power",
        real_dtype, component_sizes, tx_id);
    const at::Tensor *gcfr = optional_grad(
        std::move(grad_component_field_real), gcfr_storage,
        "grad_component_field_real", real_dtype, component_sizes, tx_id);
    const at::Tensor *gcfi = optional_grad(
        std::move(grad_component_field_imag), gcfi_storage,
        "grad_component_field_imag", real_dtype, component_sizes, tx_id);

    const int64_t path_count = tx_id.numel();
    auto fopts = component_field_real.options();
    at::Tensor grad_path_gain = at::empty({path_count}, fopts);
    at::Tensor grad_field_real = at::empty({path_count}, fopts);
    at::Tensor grad_field_imag = at::empty({path_count}, fopts);
    if (path_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_backward_kernel<float>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    valid.data_ptr<bool>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    component_field_real.data_ptr<float>(),
                    component_field_imag.data_ptr<float>(),
                    field_total_real.data_ptr<float>(),
                    field_total_imag.data_ptr<float>(),
                    power_total.data_ptr<float>(),
                    grad_ptr<float>(gpt),
                    grad_ptr<float>(gftr),
                    grad_ptr<float>(gfti),
                    grad_ptr<float>(gcp),
                    grad_ptr<float>(gcfr),
                    grad_ptr<float>(gcfi),
                    grad_path_gain.data_ptr<float>(),
                    grad_field_real.data_ptr<float>(),
                    grad_field_imag.data_ptr<float>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_backward_kernel<double>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    valid.data_ptr<bool>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    component_field_real.data_ptr<double>(),
                    component_field_imag.data_ptr<double>(),
                    field_total_real.data_ptr<double>(),
                    field_total_imag.data_ptr<double>(),
                    power_total.data_ptr<double>(),
                    grad_ptr<double>(gpt),
                    grad_ptr<double>(gftr),
                    grad_ptr<double>(gfti),
                    grad_ptr<double>(gcp),
                    grad_ptr<double>(gcfr),
                    grad_ptr<double>(gcfi),
                    grad_path_gain.data_ptr<double>(),
                    grad_field_real.data_ptr<double>(),
                    grad_field_imag.data_ptr<double>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_path_gain"] = grad_path_gain;
    out["grad_field_real"] = grad_field_real;
    out["grad_field_imag"] = grad_field_imag;
    return out;
}

pybind11::dict channel_deterministic_accumulate_flat_jvp(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor component_field_real,
    at::Tensor component_field_imag,
    at::Tensor power_total,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_field_real,
    pybind11::object tangent_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain) {
    check_accumulate_indices(valid, tx_id, rx_id, component_id, num_tx, num_rx);
    const c10::ScalarType real_dtype = component_field_real.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "deterministic_accumulate_flat_jvp requires float32 or float64 cells");
    TORCH_CHECK(
        scattering_combine_domain == 0 || scattering_combine_domain == 1,
        "scattering_combine_domain must be 0 (power) or 1 (coherent)");
    const int scattering_coherent = static_cast<int>(scattering_combine_domain);
    const std::array<int64_t, 3> component_sizes{kAccumSlotCount, num_tx, num_rx};
    const std::array<int64_t, 2> total_sizes{num_tx, num_rx};
    const at::IntArrayRef path_sizes = tx_id.sizes();
    check_cell_tensor(
        component_field_real, "component_field_real", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(
        component_field_imag, "component_field_imag", real_dtype,
        component_sizes, tx_id);
    check_cell_tensor(power_total, "power_total", real_dtype, total_sizes, tx_id);
    at::Tensor tpg_storage;
    at::Tensor tfr_storage;
    at::Tensor tfi_storage;
    const at::Tensor *tpg = optional_grad(
        std::move(tangent_path_gain), tpg_storage, "tangent_path_gain",
        real_dtype, path_sizes, tx_id);
    const at::Tensor *tfr = optional_grad(
        std::move(tangent_field_real), tfr_storage, "tangent_field_real",
        real_dtype, path_sizes, tx_id);
    const at::Tensor *tfi = optional_grad(
        std::move(tangent_field_imag), tfi_storage, "tangent_field_imag",
        real_dtype, path_sizes, tx_id);

    auto fopts = component_field_real.options();
    at::Tensor t_component_power = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_component_field_real = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_component_field_imag = zero_filled({kAccumSlotCount, num_tx, num_rx}, fopts);
    at::Tensor t_power_total = at::empty({num_tx, num_rx}, fopts);
    at::Tensor t_field_total_real = at::empty({num_tx, num_rx}, fopts);
    at::Tensor t_field_total_imag = at::empty({num_tx, num_rx}, fopts);

    const int64_t path_count = tx_id.numel();
    const int64_t cell_count = num_tx * num_rx;
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
    const bool any_tangent = tpg != nullptr || tfr != nullptr || tfi != nullptr;
    if (path_count > 0 && any_tangent) {
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_tangent_scatter_kernel<float>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    valid.data_ptr<bool>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    grad_ptr<float>(tpg),
                    grad_ptr<float>(tfr),
                    grad_ptr<float>(tfi),
                    t_component_power.data_ptr<float>(),
                    t_component_field_real.data_ptr<float>(),
                    t_component_field_imag.data_ptr<float>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_tangent_scatter_kernel<double>
                <<<launch_blocks(path_count), kBlockSize, 0, stream>>>(
                    valid.data_ptr<bool>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    grad_ptr<double>(tpg),
                    grad_ptr<double>(tfr),
                    grad_ptr<double>(tfi),
                    t_component_power.data_ptr<double>(),
                    t_component_field_real.data_ptr<double>(),
                    t_component_field_imag.data_ptr<double>(),
                    path_count,
                    num_tx,
                    num_rx,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (cell_count > 0) {
        if (real_dtype == at::kFloat) {
            deterministic_accumulate_jvp_finalize_kernel<float>
                <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                    component_field_real.data_ptr<float>(),
                    component_field_imag.data_ptr<float>(),
                    power_total.data_ptr<float>(),
                    t_component_power.data_ptr<float>(),
                    t_component_field_real.data_ptr<float>(),
                    t_component_field_imag.data_ptr<float>(),
                    t_power_total.data_ptr<float>(),
                    t_field_total_real.data_ptr<float>(),
                    t_field_total_imag.data_ptr<float>(),
                    cell_count,
                    coherent ? 1 : 0,
                    scattering_coherent);
        } else {
            deterministic_accumulate_jvp_finalize_kernel<double>
                <<<launch_blocks(cell_count), kBlockSize, 0, stream>>>(
                    component_field_real.data_ptr<double>(),
                    component_field_imag.data_ptr<double>(),
                    power_total.data_ptr<double>(),
                    t_component_power.data_ptr<double>(),
                    t_component_field_real.data_ptr<double>(),
                    t_component_field_imag.data_ptr<double>(),
                    t_power_total.data_ptr<double>(),
                    t_field_total_real.data_ptr<double>(),
                    t_field_total_imag.data_ptr<double>(),
                    cell_count,
                    coherent ? 1 : 0,
                    scattering_coherent);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["power_total"] = t_power_total;
    out["field_total_real"] = t_field_total_real;
    out["field_total_imag"] = t_field_total_imag;
    out["component_power"] = t_component_power;
    out["component_field_real"] = t_component_field_real;
    out["component_field_imag"] = t_component_field_imag;
    return out;
}

pybind11::dict channel_deterministic_component_counts(at::Tensor component_id) {
    check_flat_tensor(component_id, "component_id", at::kInt);

    at::Tensor counts = at::empty({kComponentCount}, component_id.options().dtype(at::kLong));
    const int64_t path_count = component_id.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(component_id.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int64_t>(), 0, kComponentCount * sizeof(int64_t), stream));
    if (path_count > 0) {
        const int block_count = static_cast<int>((path_count + kBlockSize - 1) / kBlockSize);
        deterministic_component_counts_kernel<<<block_count, kBlockSize, 0, stream>>>(
            component_id.data_ptr<int>(),
            path_count,
            reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int64_t data[kComponentCount] = {0, 0, 0};
    C10_CUDA_CHECK(cudaMemcpyAsync(
        data,
        counts.data_ptr<int64_t>(),
        kComponentCount * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    pybind11::dict out;
    out["los"] = data[0];
    out["reflection"] = data[1];
    out["diffraction"] = data[2];
    return out;
}

int64_t channel_deterministic_selected_edge_count(at::Tensor edge_id) {
    check_flat_tensor(edge_id, "edge_id", at::kInt);

    const int64_t path_count = edge_id.numel();
    if (path_count == 0) {
        return 0;
    }
    auto int_options = edge_id.options().dtype(at::kInt);
    auto flags = at::empty({path_count}, int_options);
    auto offsets = at::empty({path_count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_id.get_device()).stream();
    const int block_count = static_cast<int>((path_count + kBlockSize - 1) / kBlockSize);
    deterministic_edge_flags_kernel<<<block_count, kBlockSize, 0, stream>>>(
        edge_id.data_ptr<int>(),
        path_count,
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + path_count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + path_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + path_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t selected_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (selected_count == 0) {
        return 0;
    }

    auto compacted = at::empty({selected_count}, int_options);
    deterministic_compact_edges_kernel<<<block_count, kBlockSize, 0, stream>>>(
        edge_id.data_ptr<int>(),
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        path_count,
        compacted.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto begin = thrust::device_pointer_cast(compacted.data_ptr<int>());
    auto end = begin + selected_count;
    thrust::sort(thrust::cuda::par.on(stream), begin, end);
    auto unique_end = thrust::unique(thrust::cuda::par.on(stream), begin, end);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int64_t>(unique_end - begin);
}

#undef kBlockSize

