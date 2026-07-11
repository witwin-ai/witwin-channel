#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <tuple>

namespace {

constexpr int kReflectionBlockSize = 256;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon0 = 8.854187817e-12f;
constexpr float kSpeedOfLight = 299792458.0f;
constexpr float kReflectionEpsilon = 1.0e-6f;

struct Complex {
    float r;
    float i;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

__device__ __forceinline__ Complex c_make(float r, float i) { return {r, i}; }
__device__ __forceinline__ Complex c_add(Complex a, Complex b) { return {a.r + b.r, a.i + b.i}; }
__device__ __forceinline__ Complex c_sub(Complex a, Complex b) { return {a.r - b.r, a.i - b.i}; }
__device__ __forceinline__ Complex c_mul(Complex a, Complex b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
__device__ __forceinline__ Complex c_scale(Complex a, float s) { return {a.r * s, a.i * s}; }
__device__ __forceinline__ Complex c_div(Complex a, Complex b) {
    const float d = fmaxf(b.r * b.r + b.i * b.i, 1.0e-30f);
    return {(a.r * b.r + a.i * b.i) / d, (a.i * b.r - a.r * b.i) / d};
}
__device__ __forceinline__ Complex c_sqrt(Complex z) {
    const float m = hypotf(z.r, z.i);
    float r = sqrtf(fmaxf(0.0f, 0.5f * (m + z.r)));
    float i = copysignf(sqrtf(fmaxf(0.0f, 0.5f * (m - z.r))), z.i);
    return {r, i};
}
__device__ __forceinline__ Complex c_exp_neg_2i(Complex q) {
    const float amplitude = expf(fminf(2.0f * q.i, 80.0f));
    float s, c;
    sincosf(2.0f * q.r, &s, &c);
    return {amplitude * c, -amplitude * s};
}
__device__ __forceinline__ float c_abs2(Complex a) { return a.r * a.r + a.i * a.i; }

__device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return f3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float3 normalize3(float3 v) {
    const float inv = rsqrtf(fmaxf(dot3(v, v), 1.0e-30f));
    return f3(v.x*inv, v.y*inv, v.z*inv);
}
__device__ __forceinline__ float3 add3(float3 a, float3 b) { return f3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 scale3(float3 a, float s) { return f3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ Complex c3_dot(Complex3 a, float3 b) {
    return {a.x.r*b.x+a.y.r*b.y+a.z.r*b.z, a.x.i*b.x+a.y.i*b.y+a.z.i*b.z};
}
__device__ __forceinline__ Complex3 c3_axis(float3 axis, Complex value) {
    return {{axis.x*value.r, axis.x*value.i},
            {axis.y*value.r, axis.y*value.i},
            {axis.z*value.r, axis.z*value.i}};
}
__device__ __forceinline__ Complex3 c3_add(Complex3 a, Complex3 b) {
    return {c_add(a.x,b.x), c_add(a.y,b.y), c_add(a.z,b.z)};
}
__device__ __forceinline__ float c3_power(Complex3 a) {
    return c_abs2(a.x) + c_abs2(a.y) + c_abs2(a.z);
}

__device__ __forceinline__ float component3(float3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__device__ __forceinline__ void plane_coords(float3 v, int axis, float &a, float &b) {
    if (axis == 0) { a = v.y; b = v.z; }
    else if (axis == 1) { a = v.x; b = v.z; }
    else { a = v.x; b = v.y; }
}

__device__ __forceinline__ void slab_coefficients(
    float cos_theta,
    float eta_r,
    float sigma,
    float gain,
    float thickness,
    float wavelength,
    Complex &r_te,
    Complex &r_tm) {
    const float omega = 2.0f * kPi * kSpeedOfLight / fmaxf(wavelength, kReflectionEpsilon);
    const Complex eta = c_make(fmaxf(eta_r, kReflectionEpsilon),
                               -fmaxf(sigma, 0.0f) / (omega * kEpsilon0));
    const float ct = fminf(fmaxf(fabsf(cos_theta), kReflectionEpsilon), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);
    const Complex root = c_sqrt(c_sub(eta, c_make(sin2, 0.0f)));
    const Complex ct_c = c_make(ct, 0.0f);
    const Complex eta_ct = c_scale(eta, ct);
    const Complex rp_te = c_div(c_sub(ct_c, root), c_add(ct_c, root));
    const Complex rp_tm = c_div(c_sub(eta_ct, root), c_add(eta_ct, root));
    const Complex q = c_scale(root, 2.0f * kPi * fmaxf(thickness, 0.0f) /
                                      fmaxf(wavelength, kReflectionEpsilon));
    const Complex phase = c_exp_neg_2i(q);
    const Complex one = c_make(1.0f, 0.0f);
    const Complex phase_term = c_sub(one, phase);
    r_te = c_scale(c_div(c_mul(rp_te, phase_term),
                         c_sub(one, c_mul(c_mul(rp_te, rp_te), phase))), gain);
    r_tm = c_scale(c_div(c_mul(rp_tm, phase_term),
                         c_sub(one, c_mul(c_mul(rp_tm, rp_tm), phase))), gain);
}

__global__ void sionna_reflection_accumulate_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ trace_valid,
    const float *__restrict__ trace_t,
    const int *__restrict__ trace_prim,
    const float *__restrict__ face_normals,
    const float *__restrict__ eta_r,
    const float *__restrict__ sigma,
    const float *__restrict__ gain,
    const bool *__restrict__ material_valid,
    const float *__restrict__ thickness,
    float *__restrict__ output,
    int64_t ray_count,
    int trace_depth,
    int contribution_depth,
    int axis,
    float plane_position,
    float coord0_min,
    float coord0_max,
    float coord1_min,
    float coord1_max,
    int resolution0,
    int resolution1,
    float wavelength,
    float solid_angle_per_ray,
    float cell_area) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        float3 origin = f3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 direction = normalize3(f3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        float3 vertical = f3(0.0f, 0.0f, 1.0f);
        float3 initial = add3(vertical, scale3(direction, -dot3(vertical, direction)));
        if (dot3(initial, initial) < 1.0e-12f)
            initial = f3(1.0f, 0.0f, 0.0f);
        initial = normalize3(initial);
        Complex3 field = {c_make(initial.x, 0.0f), c_make(initial.y, 0.0f), c_make(initial.z, 0.0f)};

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = add3(origin, scale3(direction, t_hit));
            float3 normal = normalize3(f3(face_normals[3*prim], face_normals[3*prim+1], face_normals[3*prim+2]));
            if (dot3(direction, normal) > 0.0f) normal = scale3(normal, -1.0f);

            float3 s_hat = cross3(normal, direction);
            if (dot3(s_hat, s_hat) < 1.0e-12f)
                s_hat = normalize3(cross3(fabsf(direction.z) < 0.9f ? vertical : f3(0.0f,1.0f,0.0f), direction));
            else
                s_hat = normalize3(s_hat);
            const float3 p_in = normalize3(cross3(s_hat, direction));
            const float3 reflected = normalize3(add3(direction, scale3(normal, -2.0f*dot3(direction, normal))));
            const float3 p_out = normalize3(cross3(s_hat, reflected));
            Complex r_te, r_tm;
            slab_coefficients(fabsf(dot3(direction, normal)), eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, r_te, r_tm);
            const Complex e_s = c3_dot(field, s_hat);
            const Complex e_p = c3_dot(field, p_in);
            field = c3_add(c3_axis(s_hat, c_mul(r_te, e_s)),
                           c3_axis(p_out, c_mul(r_tm, e_p)));
            direction = reflected;
            origin = hit;

            const float axis_direction = component3(direction, axis);
            if (fabsf(axis_direction) <= kReflectionEpsilon) continue;
            const float t_plane = (plane_position - component3(origin, axis)) / axis_direction;
            const int next_depth = depth + 1;
            float blocker_t = 1.0e30f;
            if (next_depth < trace_depth) {
                const int64_t next_slot = ray * static_cast<int64_t>(trace_depth) + next_depth;
                if (trace_valid[next_slot]) blocker_t = trace_t[next_slot];
            }
            if (!(t_plane > 1.0e-4f && t_plane < blocker_t)) continue;
            const float3 target = add3(origin, scale3(direction, t_plane));
            float coord0, coord1;
            plane_coords(target, axis, coord0, coord1);
            if (coord0 < coord0_min || coord0 >= coord0_max ||
                coord1 < coord1_min || coord1 >= coord1_max) continue;
            const int i0 = min(max(static_cast<int>((coord0-coord0_min)/(coord0_max-coord0_min)*resolution0),0),resolution0-1);
            const int i1 = min(max(static_cast<int>((coord1-coord1_min)/(coord1_max-coord1_min)*resolution1),0),resolution1-1);
            const float norm = (wavelength/(4.0f*kPi))*(wavelength/(4.0f*kPi)) /
                               fmaxf(cell_area, kReflectionEpsilon);
            const float power = c3_power(field) * solid_angle_per_ray * norm /
                                fmaxf(fabsf(axis_direction), kReflectionEpsilon);
            if (power > 0.0f && isfinite(power)) atomicAdd(output + i1*resolution0+i0, power);
        }
    }
}

void check_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void reflection_launch_inputs_kernel(
    const float *__restrict__ tx_positions,
    float *__restrict__ ray_o,
    bool *__restrict__ active,
    float *__restrict__ tx_pol,
    int *__restrict__ tx_id,
    int64_t *__restrict__ light_seed,
    int64_t tx_index,
    int64_t sample_count) {
    const float tx_x = tx_positions[tx_index * 3 + 0];
    const float tx_y = tx_positions[tx_index * 3 + 1];
    const float tx_z = tx_positions[tx_index * 3 + 2];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         sample < sample_count;
         sample += stride) {
        float *origin = ray_o + sample * 3;
        origin[0] = tx_x;
        origin[1] = tx_y;
        origin[2] = tx_z;

        float *pol = tx_pol + sample * 3;
        pol[0] = 1.0f;
        pol[1] = 0.0f;
        pol[2] = 0.0f;

        active[sample] = true;
        if (tx_id != nullptr) {
            tx_id[sample] = static_cast<int>(tx_index);
        }
        if (light_seed != nullptr) {
            light_seed[sample] = static_cast<int64_t>(
                (static_cast<unsigned long long>(tx_index) << 32) ^
                static_cast<unsigned long long>(sample));
        }
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    check_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");

    auto ray_o = at::empty({sample_count, 3}, tx_positions.options());
    auto ray_tmax = at::empty({0}, tx_positions.options());
    auto active = at::empty({sample_count}, tx_positions.options().dtype(at::kBool));
    auto tx_pol = at::empty({sample_count, 3}, tx_positions.options());
    if (sample_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((sample_count + kReflectionBlockSize - 1) / kReflectionBlockSize);
        reflection_launch_inputs_kernel<<<block_count, kReflectionBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            ray_o.data_ptr<float>(),
            active.data_ptr<bool>(),
            tx_pol.data_ptr<float>(),
            nullptr,
            nullptr,
            tx_index,
            sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ray_o, ray_tmax, active, tx_pol};
}

at::Tensor cn_mc_sionna_reflection_accumulate_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area) {
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    auto output = at::empty({resolution1, resolution0}, ray_o.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr<float>(), 0, static_cast<size_t>(output.numel()) * sizeof(float), stream));
    if (ray_count == 0 || contribution_depth <= 0) return output;
    const int blocks = static_cast<int>(std::min<int64_t>((ray_count + kReflectionBlockSize - 1) / kReflectionBlockSize, 65535));
    sionna_reflection_accumulate_kernel<<<blocks, kReflectionBlockSize, 0, stream>>>(
        ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
        trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
        eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
        material_valid.data_ptr<bool>(), thickness.data_ptr<float>(), output.data_ptr<float>(),
        ray_count, trace_depth, static_cast<int>(contribution_depth), static_cast<int>(axis),
        static_cast<float>(plane_position), static_cast<float>(coord0_min), static_cast<float>(coord0_max),
        static_cast<float>(coord1_min), static_cast<float>(coord1_max), static_cast<int>(resolution0),
        static_cast<int>(resolution1), static_cast<float>(wavelength),
        static_cast<float>(solid_angle_per_ray), static_cast<float>(cell_area));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    check_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");

    auto ray_o = at::empty({sample_count, 3}, tx_positions.options());
    auto ray_tmax = at::empty({0}, tx_positions.options());
    auto active = at::empty({sample_count}, tx_positions.options().dtype(at::kBool));
    auto tx_pol = at::empty({sample_count, 3}, tx_positions.options());
    auto tx_id = at::empty({sample_count}, tx_positions.options().dtype(at::kInt));
    auto light_seed = at::empty({sample_count}, tx_positions.options().dtype(at::kLong));
    if (sample_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((sample_count + kReflectionBlockSize - 1) / kReflectionBlockSize);
        reflection_launch_inputs_kernel<<<block_count, kReflectionBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            ray_o.data_ptr<float>(),
            active.data_ptr<bool>(),
            tx_pol.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            light_seed.data_ptr<int64_t>(),
            tx_index,
            sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ray_o, ray_tmax, active, tx_pol, tx_id, light_seed};
}
