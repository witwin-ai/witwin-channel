#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kBlockSize = 256;
constexpr float kLightSpeed = 299792458.0f;
constexpr float kEpsilon0 = 8.854187817e-12f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1.0e-6f;

void check_tensor(const at::Tensor &tensor, const char *name, c10::ScalarType dtype, int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

struct Float3 {
    float x;
    float y;
    float z;
};

struct Complex {
    float r;
    float i;
};

__device__ Float3 make_f3(float x, float y, float z) {
    return {x, y, z};
}

__device__ Float3 load_f3(const float *ptr, int64_t index) {
    const int64_t base = index * 3;
    return {ptr[base + 0], ptr[base + 1], ptr[base + 2]};
}

__device__ Float3 load_sequence_f3(const float *ptr, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return {ptr[base + 0], ptr[base + 1], ptr[base + 2]};
}

__device__ Float3 sub_f3(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Float3 scale_f3(Float3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ float dot_f3(Float3 a, Float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Float3 cross_f3(Float3 a, Float3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

__device__ float norm_f3(Float3 a) {
    return sqrtf(fmaxf(dot_f3(a, a), 0.0f));
}

__device__ Float3 normalize_f3(Float3 a) {
    const float inv = 1.0f / fmaxf(norm_f3(a), kEps);
    return scale_f3(a, inv);
}

__device__ Complex c_make(float r, float i) {
    return {r, i};
}

/// -k*d reduced mod 2*pi in double precision: the f32 product loses
/// ~k*d*2^-24 of phase, which shifts coherent multipath nulls at mmWave
/// ranges (matches the reference cplx_exp_neg_kd convention).
__device__ float neg_kd_phase(float k, float d) {
    const double kd = fmod(static_cast<double>(k) * static_cast<double>(d), 6.283185307179586476925287);
    return -static_cast<float>(kd);
}

__device__ Complex c_add(Complex a, Complex b) {
    return {a.r + b.r, a.i + b.i};
}

__device__ Complex c_sub(Complex a, Complex b) {
    return {a.r - b.r, a.i - b.i};
}

__device__ Complex c_mul(Complex a, Complex b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

__device__ Complex c_scale(Complex a, float s) {
    return {a.r * s, a.i * s};
}

__device__ Complex c_div(Complex a, Complex b) {
    const float denom = fmaxf(b.r * b.r + b.i * b.i, kEps);
    return {(a.r * b.r + a.i * b.i) / denom, (a.i * b.r - a.r * b.i) / denom};
}

__device__ Complex c_sqrt(Complex z) {
    const float magnitude = hypotf(z.r, z.i);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + z.r)));
    const float imag_sign = z.i < 0.0f ? -1.0f : 1.0f;
    const float imag = imag_sign * sqrtf(fmaxf(0.0f, 0.5f * (magnitude - z.r)));
    return {real, imag};
}

__device__ Float3 orthogonal_transverse(Float3 direction) {
    Float3 axis = fabsf(direction.z) < 0.9f ? make_f3(0.0f, 0.0f, 1.0f) : make_f3(0.0f, 1.0f, 0.0f);
    return normalize_f3(sub_f3(axis, scale_f3(direction, dot_f3(axis, direction))));
}

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

__device__ Complex3 c3_zero() {
    return {c_make(0.0f, 0.0f), c_make(0.0f, 0.0f), c_make(0.0f, 0.0f)};
}

__device__ Complex3 c3_from_real(Float3 v) {
    return {c_make(v.x, 0.0f), c_make(v.y, 0.0f), c_make(v.z, 0.0f)};
}

__device__ Complex3 c3_add(Complex3 a, Complex3 b) {
    return {c_add(a.x, b.x), c_add(a.y, b.y), c_add(a.z, b.z)};
}

__device__ Complex3 c3_scale_complex(Float3 v, Complex c) {
    return {c_scale(c, v.x), c_scale(c, v.y), c_scale(c, v.z)};
}

__device__ Complex c3_dot_real(Complex3 f, Float3 v) {
    return c_make(
        f.x.r * v.x + f.y.r * v.y + f.z.r * v.z,
        f.x.i * v.x + f.y.i * v.y + f.z.i * v.z);
}

__device__ float c_abs2(Complex a) {
    return a.r * a.r + a.i * a.i;
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
    const float sin2 = fmaxf(0.0f, 1.0f - cos_theta * cos_theta);
    const float omega = fmaxf(2.0f * kPi * frequency_hz, kEps);
    const float eta_r = fmaxf(eps_r, kEps);
    const float sigma = fmaxf(sigma_e, 0.0f);
    const float mu_value = fmaxf(mu_r, kEps);
    const Complex eta = c_make(eta_r, -sigma / (omega * kEpsilon0));
    const Complex mu = c_make(mu_value, 0.0f);
    const Complex root = c_sqrt(c_sub(c_mul(mu, eta), c_make(sin2, 0.0f)));
    const Complex mu_cos = c_make(mu_value * cos_theta, 0.0f);
    const Complex eta_cos = c_scale(eta, cos_theta);
    r_te = c_div(c_sub(mu_cos, root), c_add(mu_cos, root));
    r_tm = c_div(c_sub(eta_cos, root), c_add(eta_cos, root));
}

/// Initial transverse polarization: the global x-hat transmit polarization
/// projected perpendicular to the launch direction.
__device__ Float3 initial_transverse_polarization(Float3 incident) {
    const Float3 tx_pol = make_f3(1.0f, 0.0f, 0.0f);
    Float3 transverse = sub_f3(tx_pol, scale_f3(incident, dot_f3(tx_pol, incident)));
    const float transverse_norm = norm_f3(transverse);
    if (transverse_norm > kEps) {
        return scale_f3(transverse, 1.0f / transverse_norm);
    }
    return orthogonal_transverse(incident);
}

/// Reflect a complex field 3-vector at a planar interface: decompose into
/// s/p components, apply the Fresnel coefficients, and recompose on the
/// rotated outgoing basis (mirrors RayDN's epc_field.cu reflect_field_vector).
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
    const Float3 incident = normalize_f3(incident_direction);
    Float3 n = normalize_f3(normal);
    if (dot_f3(incident, n) > 0.0f) {
        n = scale_f3(n, -1.0f);
    }
    const float dot_dn = dot_f3(incident, n);
    reflected_direction = normalize_f3(sub_f3(incident, scale_f3(n, 2.0f * dot_dn)));

    Float3 s_hat = cross_f3(n, incident);
    const float s_norm = norm_f3(s_hat);
    const Float3 transverse_basis = orthogonal_transverse(incident);
    s_hat = s_norm > kEps ? scale_f3(s_hat, 1.0f / s_norm) : transverse_basis;
    const Float3 p_in = normalize_f3(cross_f3(s_hat, incident));
    const Float3 p_out = normalize_f3(cross_f3(s_hat, reflected_direction));

    Complex r_te;
    Complex r_tm;
    fresnel_coefficients(fabsf(dot_dn), eps_r, sigma_e, mu_r, frequency_hz, r_te, r_tm);
    const Complex e_s = c3_dot_real(field, s_hat);
    const Complex e_p = c3_dot_real(field, p_in);
    Complex3 reflected = c3_add(
        c3_scale_complex(s_hat, c_mul(r_te, e_s)),
        c3_scale_complex(p_out, c_mul(r_tm, e_p)));
    reflected.x = c_scale(reflected.x, gain);
    reflected.y = c_scale(reflected.y, gain);
    reflected.z = c_scale(reflected.z, gain);
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
    const float px = c_abs2(field.x);
    const float py = c_abs2(field.y);
    const float pz = c_abs2(field.z);
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
        const Float3 tx = load_f3(tx_position, index);
        const Float3 rx = load_f3(rx_position, index);
        const Float3 hit = load_f3(hit_position, index);
        const Float3 incident = normalize_f3(sub_f3(hit, tx));
        Float3 reflected_direction;
        Complex3 field_vector = reflect_field_vector(
            c3_from_real(initial_transverse_polarization(incident)),
            incident,
            load_f3(normal, index),
            eps_r[index],
            sigma_e[index],
            mu_r[index],
            gain[index],
            frequency_hz,
            reflected_direction);

        const float segment0 = norm_f3(sub_f3(hit, tx));
        const float segment1 = norm_f3(sub_f3(rx, hit));
        const float path_length = fmaxf(segment0 + segment1, kEps);
        const float wavelength = kLightSpeed / frequency_hz;
        const float amplitude = sqrtf(fmaxf(tx_power[index], 0.0f)) * (wavelength / (4.0f * kPi)) / path_length;
        const float phase = neg_kd_phase(2.0f * kPi / wavelength, path_length);
        const Complex scale = c_scale(c_make(cosf(phase), sinf(phase)), amplitude);
        field_vector.x = c_mul(field_vector.x, scale);
        field_vector.y = c_mul(field_vector.y, scale);
        field_vector.z = c_mul(field_vector.z, scale);
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
        const Float3 tx = load_f3(tx_position, index);
        const Float3 rx = load_f3(rx_position, index);
        Float3 previous = tx;
        Complex3 field_vector = c3_zero();
        bool field_initialized = false;
        float path_length = 0.0f;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const Float3 hit = load_sequence_f3(hit_positions, index, bounce, depth);
            const Float3 segment = sub_f3(hit, previous);
            path_length += norm_f3(segment);
            const Float3 incident = normalize_f3(segment);
            if (!field_initialized) {
                field_vector = c3_from_real(initial_transverse_polarization(incident));
                field_initialized = true;
            }
            const int64_t scalar_index = index * depth + bounce;
            Float3 reflected_direction;
            field_vector = reflect_field_vector(
                field_vector,
                incident,
                load_sequence_f3(normals, index, bounce, depth),
                eps_r[scalar_index],
                sigma_e[scalar_index],
                mu_r[scalar_index],
                gain[scalar_index],
                frequency_hz,
                reflected_direction);
            previous = hit;
        }

        path_length = fmaxf(path_length + norm_f3(sub_f3(rx, previous)), kEps);
        const float wavelength = kLightSpeed / frequency_hz;
        const float amplitude = sqrtf(fmaxf(tx_power[index], 0.0f)) * (wavelength / (4.0f * kPi)) / path_length;
        const float phase = neg_kd_phase(2.0f * kPi / wavelength, path_length);
        const Complex scale = c_scale(c_make(cosf(phase), sinf(phase)), amplitude);
        field_vector.x = c_mul(field_vector.x, scale);
        field_vector.y = c_mul(field_vector.y, scale);
        field_vector.z = c_mul(field_vector.z, scale);
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

pybind11::dict cn_deterministic_los_field(
    torch::Tensor path_gain_input,
    torch::Tensor path_length_m,
    double frequency_hz) {
    check_tensor(path_gain_input, "path_gain", torch::kFloat32, 1);
    check_tensor(path_length_m, "path_length_m", torch::kFloat32, 1);
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

pybind11::dict cn_deterministic_diffraction_vector_field(
    torch::Tensor x_re,
    torch::Tensor x_im,
    torch::Tensor y_re,
    torch::Tensor y_im,
    torch::Tensor z_re,
    torch::Tensor z_im) {
    check_tensor(x_re, "x_re", torch::kFloat32, 1);
    check_tensor(x_im, "x_im", torch::kFloat32, 1);
    check_tensor(y_re, "y_re", torch::kFloat32, 1);
    check_tensor(y_im, "y_im", torch::kFloat32, 1);
    check_tensor(z_re, "z_re", torch::kFloat32, 1);
    check_tensor(z_im, "z_im", torch::kFloat32, 1);
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

pybind11::dict cn_deterministic_reflection_field(
    torch::Tensor tx_position,
    torch::Tensor rx_position,
    torch::Tensor hit_position,
    torch::Tensor normal,
    torch::Tensor tx_power,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    double frequency_hz) {
    check_tensor(tx_position, "tx_position", torch::kFloat32, 2);
    check_tensor(rx_position, "rx_position", torch::kFloat32, 2);
    check_tensor(hit_position, "hit_position", torch::kFloat32, 2);
    check_tensor(normal, "normal", torch::kFloat32, 2);
    check_tensor(tx_power, "tx_power", torch::kFloat32, 1);
    check_tensor(eps_r, "eps_r", torch::kFloat32, 1);
    check_tensor(sigma_e, "sigma_e", torch::kFloat32, 1);
    check_tensor(mu_r, "mu_r", torch::kFloat32, 1);
    check_tensor(gain, "gain", torch::kFloat32, 1);
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

pybind11::dict cn_deterministic_reflection_sequence_field(
    torch::Tensor tx_position,
    torch::Tensor rx_position,
    torch::Tensor hit_positions,
    torch::Tensor normals,
    torch::Tensor tx_power,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    double frequency_hz) {
    check_tensor(tx_position, "tx_position", torch::kFloat32, 2);
    check_tensor(rx_position, "rx_position", torch::kFloat32, 2);
    check_tensor(hit_positions, "hit_positions", torch::kFloat32, 3);
    check_tensor(normals, "normals", torch::kFloat32, 3);
    check_tensor(tx_power, "tx_power", torch::kFloat32, 1);
    check_tensor(eps_r, "eps_r", torch::kFloat32, 2);
    check_tensor(sigma_e, "sigma_e", torch::kFloat32, 2);
    check_tensor(mu_r, "mu_r", torch::kFloat32, 2);
    check_tensor(gain, "gain", torch::kFloat32, 2);
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

torch::Tensor cn_deterministic_delay_to_path_length(torch::Tensor delay_s) {
    check_tensor(delay_s, "delay_s", torch::kFloat32, 1);
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

torch::Tensor cn_deterministic_pack_complex(torch::Tensor field_real, torch::Tensor field_imag) {
    check_tensor(field_real, "field_real", torch::kFloat32, 1);
    check_tensor(field_imag, "field_imag", torch::kFloat32, 1);
    TORCH_CHECK(field_imag.sizes() == field_real.sizes(), "field_imag must match field_real");
    auto complex_options = field_real.options().dtype(torch::kComplexFloat);
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

torch::Tensor cn_deterministic_phase_from_field(torch::Tensor field_real, torch::Tensor field_imag) {
    check_tensor(field_real, "field_real", torch::kFloat32, 1);
    check_tensor(field_imag, "field_imag", torch::kFloat32, 1);
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

pybind11::dict cn_deterministic_zero_field_phase(torch::Tensor reference) {
    check_tensor(reference, "reference", torch::kFloat32, 1);
    auto complex_options = reference.options().dtype(torch::kComplexFloat);
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

torch::Tensor cn_deterministic_phase_from_length(torch::Tensor path_length_m, double frequency_hz) {
    check_tensor(path_length_m, "path_length_m", torch::kFloat32, 1);
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

pybind11::dict cn_deterministic_field_from_power_phase(torch::Tensor path_gain, torch::Tensor phase_rad) {
    check_tensor(path_gain, "path_gain", torch::kFloat32, 1);
    check_tensor(phase_rad, "phase_rad", torch::kFloat32, 1);
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
