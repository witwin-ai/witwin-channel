#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"
#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/torch/rf/field_transport_ad.cuh>

#include <algorithm>
#include <tuple>

#define CN_REFLECTION_PREPARE_LAUNCH_INPUTS()                                         \
    check_tensor(tx_positions, "tx_positions", at::kFloat, 2);                       \
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");    \
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");\
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");               \
    auto ray_o = at::empty({sample_count, 3}, tx_positions.options());                 \
    auto ray_tmax = at::empty({0}, tx_positions.options());                           \
    auto active = at::empty({sample_count}, tx_positions.options().dtype(at::kBool)); \
    auto tx_pol = at::empty({sample_count, 3}, tx_positions.options())

#define CN_REFLECTION_LAUNCH_INPUT_PREFIX()                                            \
    tx_positions.data_ptr<float>(),                                                   \
    ray_o.data_ptr<float>(),                                                          \
    active.data_ptr<bool>(),                                                          \
    tx_pol.data_ptr<float>()

namespace {

constexpr int kReflectionBlockSize = 256;
constexpr int kReflectionAdMaxDepth = 8;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kReflectionEpsilon = 1.0e-6f;
namespace transport = rayd::shared::rf::field_transport;
namespace ad = rayd::torch::rf::field_transport_ad;

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
__device__ __forceinline__ Complex c_mul(Complex a, Complex b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
__device__ __forceinline__ Complex c_scale(Complex a, float s) { return {a.r * s, a.i * s}; }
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
    rayd::shared::utd::Complex shared_te;
    rayd::shared::utd::Complex shared_tm;
    transport::legacy_sionna_slab_fresnel(
        cos_theta,
        eta_r,
        sigma,
        gain,
        thickness,
        wavelength,
        shared_te,
        shared_tm);
    r_te = c_make(shared_te.re, shared_te.im);
    r_tm = c_make(shared_tm.re, shared_tm.im);
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
    float cell_area,
    const float *__restrict__ tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = f3(tx_pol[0], tx_pol[1], tx_pol[2]);
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        float3 origin = f3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 direction = normalize3(f3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        float3 vertical = f3(0.0f, 0.0f, 1.0f);
        // R5 polarization consistency: seed the transported field with the
        // UNNORMALIZED transverse projection of the true TX polarization onto
        // the launch direction (short-dipole sin(theta) pattern). No axial-null
        // special case: a zero here is the correct physical null, and |field|^2
        // carries the sin^2(theta) weight to match LoS/diffraction.
        float3 initial = add3(tx_polarization, scale3(direction, -dot3(tx_polarization, direction)));
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

// ---------------------------------------------------------------------------
// Backward / JVP companions of sionna_reflection_accumulate_kernel
// (plan 07 AD-3). Fixed-winner contract: the RayD trace tape (valid / t /
// prim), the sampled directions and the ray origins are frozen constants of
// the differentiation, so the deposit binning, the incidence cosines and the
// polarization frames are all constant; every derivative flows through the
// legacy slab Fresnel coefficients (eta_r / sigma / gain / thickness /
// wavelength) and through the (lambda/4pi)^2 aperture factor of the deposit
// weight. The per-ray deposit weight does not depend on the ray origin at
// all, so the ray-origin gradient of this map is exactly zero (the Python
// dispatch layer returns it without a launch). The walk below REPLAYS
// sionna_reflection_accumulate_kernel operation by operation; edit the primal
// kernel and these companions TOGETHER.
// ---------------------------------------------------------------------------

struct ReflectionBounceFrame {
    float3 normal;
    float3 s_hat;
    float3 p_in;
    float3 reflected;
    float3 p_out;
    float cos_theta;
};

__device__ __forceinline__ ReflectionBounceFrame reflection_bounce_frame(
    const float *__restrict__ face_normals,
    int prim,
    float3 direction) {
    ReflectionBounceFrame frame;
    const float3 vertical = f3(0.0f, 0.0f, 1.0f);
    frame.normal = normalize3(
        f3(face_normals[3 * prim], face_normals[3 * prim + 1], face_normals[3 * prim + 2]));
    if (dot3(direction, frame.normal) > 0.0f)
        frame.normal = scale3(frame.normal, -1.0f);
    float3 s_hat = cross3(frame.normal, direction);
    if (dot3(s_hat, s_hat) < 1.0e-12f)
        s_hat = normalize3(cross3(
            fabsf(direction.z) < 0.9f ? vertical : f3(0.0f, 1.0f, 0.0f), direction));
    else
        s_hat = normalize3(s_hat);
    frame.s_hat = s_hat;
    frame.p_in = normalize3(cross3(s_hat, direction));
    frame.reflected = normalize3(
        add3(direction, scale3(frame.normal, -2.0f * dot3(direction, frame.normal))));
    frame.p_out = normalize3(cross3(s_hat, frame.reflected));
    frame.cos_theta = fabsf(dot3(direction, frame.normal));
    return frame;
}

// Real-pair contribution of a dual coefficient against a complex cotangent.
__device__ __forceinline__ float adj_dot_local(
    Complex g, rayd::shared::utd::Complex d) {
    return g.r * d.re + g.i * d.im;
}

__global__ void sionna_reflection_accumulate_backward_kernel(
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
    const float *__restrict__ grad_output,
    float *__restrict__ grad_eta_r,
    float *__restrict__ grad_sigma,
    float *__restrict__ grad_gain,
    float *__restrict__ grad_thickness,
    float *__restrict__ grad_frequency,
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
    float cell_area,
    float wavelength_dfreq,
    int64_t grad_stride0,
    int64_t grad_stride1,
    const float *__restrict__ tx_pol) {
    const bool need_materials = grad_eta_r != nullptr;
    const bool need_frequency = grad_frequency != nullptr;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = f3(tx_pol[0], tx_pol[1], tx_pol[2]);
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        // Forward replay of sionna_reflection_accumulate_kernel, recording the
        // per-bounce state the reverse sweep needs.
        float3 origin = f3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 direction = normalize3(f3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        // R5: unnormalized transverse projection of the true TX polarization
        // (see the forward kernel). The seed field is a frozen winner of the
        // material/frequency differentiation, but its sin^2(theta) magnitude
        // scales every deposit and hence every gradient, so it must match the
        // forward exactly.
        float3 initial = add3(tx_polarization, scale3(direction, -dot3(tx_polarization, direction)));
        const Complex3 initial_field = {
            c_make(initial.x, 0.0f), c_make(initial.y, 0.0f), c_make(initial.z, 0.0f)};
        Complex3 field = initial_field;

        Complex3 field_after[kReflectionAdMaxDepth];
        float3 dir_in[kReflectionAdMaxDepth];
        int prim_at[kReflectionAdMaxDepth];
        int deposit_cell[kReflectionAdMaxDepth];
        float deposit_coeff[kReflectionAdMaxDepth];
        int bounce_count = 0;

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = add3(origin, scale3(direction, t_hit));
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, direction);
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, r_te, r_tm);
            const Complex e_s = c3_dot(field, frame.s_hat);
            const Complex e_p = c3_dot(field, frame.p_in);
            field = c3_add(c3_axis(frame.s_hat, c_mul(r_te, e_s)),
                           c3_axis(frame.p_out, c_mul(r_tm, e_p)));
            dir_in[depth] = direction;
            prim_at[depth] = prim;
            field_after[depth] = field;
            deposit_cell[depth] = -1;
            deposit_coeff[depth] = 0.0f;
            bounce_count = depth + 1;
            direction = frame.reflected;
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
            const float coeff = solid_angle_per_ray * norm /
                                fmaxf(fabsf(axis_direction), kReflectionEpsilon);
            const float power = c3_power(field) * coeff;
            if (power > 0.0f && isfinite(power)) {
                deposit_cell[depth] = i1 * resolution0 + i0;
                deposit_coeff[depth] = coeff;
            }
        }

        // Reverse sweep: fold deposit cotangents into the field chain, pull
        // them through each bounce, and dot the Fresnel cotangents against
        // per-parameter forward duals of the frozen legacy slab response.
        Complex3 g_field = {c_make(0.0f, 0.0f), c_make(0.0f, 0.0f), c_make(0.0f, 0.0f)};
        float g_lambda = 0.0f;
        for (int depth = bounce_count - 1; depth >= 0; --depth) {
            if (deposit_cell[depth] >= 0) {
                const int i1 = deposit_cell[depth] / resolution0;
                const int i0 = deposit_cell[depth] - i1 * resolution0;
                const float g_dep = grad_output[i1 * grad_stride0 + i0 * grad_stride1];
                const float scale = 2.0f * g_dep * deposit_coeff[depth];
                const Complex3 f = field_after[depth];
                g_field.x = c_add(g_field.x, c_scale(f.x, scale));
                g_field.y = c_add(g_field.y, c_scale(f.y, scale));
                g_field.z = c_add(g_field.z, c_scale(f.z, scale));
                // The deposit weight carries (lambda / 4 pi)^2: its direct
                // wavelength derivative is 2 * power / lambda.
                g_lambda += g_dep * c3_power(f) * deposit_coeff[depth] * 2.0f / wavelength;
            }
            const int prim = prim_at[depth];
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, dir_in[depth]);
            const Complex3 field_before =
                depth == 0 ? initial_field : field_after[depth - 1];
            const Complex e_s = c3_dot(field_before, frame.s_hat);
            const Complex e_p = c3_dot(field_before, frame.p_in);
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, r_te, r_tm);
            // field_after = s_hat * (r_te * e_s) + p_out * (r_tm * e_p); the
            // axis expansions are real-linear, so their adjoints are the same
            // real-axis dots.
            const Complex g_w_te = c3_dot(g_field, frame.s_hat);
            const Complex g_w_tm = c3_dot(g_field, frame.p_out);
            // Real-pair adjoint of the complex products w = r * e.
            const Complex g_r_te = c_make(
                g_w_te.r * e_s.r + g_w_te.i * e_s.i,
                -g_w_te.r * e_s.i + g_w_te.i * e_s.r);
            const Complex g_r_tm = c_make(
                g_w_tm.r * e_p.r + g_w_tm.i * e_p.i,
                -g_w_tm.r * e_p.i + g_w_tm.i * e_p.r);
            const Complex g_e_s = c_make(
                g_w_te.r * r_te.r + g_w_te.i * r_te.i,
                -g_w_te.r * r_te.i + g_w_te.i * r_te.r);
            const Complex g_e_p = c_make(
                g_w_tm.r * r_tm.r + g_w_tm.i * r_tm.i,
                -g_w_tm.r * r_tm.i + g_w_tm.i * r_tm.r);
            if (need_materials || need_frequency) {
                ad::DualC dual_te, dual_tm;
                if (need_materials) {
                    ad::legacy_sionna_slab_fresnel_dual(
                        frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                        thickness[prim], wavelength,
                        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, dual_te, dual_tm);
                    atomicAdd(grad_eta_r + prim,
                              adj_dot_local(g_r_te, dual_te.d) +
                              adj_dot_local(g_r_tm, dual_tm.d));
                    ad::legacy_sionna_slab_fresnel_dual(
                        frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                        thickness[prim], wavelength,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, dual_te, dual_tm);
                    atomicAdd(grad_sigma + prim,
                              adj_dot_local(g_r_te, dual_te.d) +
                              adj_dot_local(g_r_tm, dual_tm.d));
                    ad::legacy_sionna_slab_fresnel_dual(
                        frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                        thickness[prim], wavelength,
                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, dual_te, dual_tm);
                    atomicAdd(grad_gain + prim,
                              adj_dot_local(g_r_te, dual_te.d) +
                              adj_dot_local(g_r_tm, dual_tm.d));
                    ad::legacy_sionna_slab_fresnel_dual(
                        frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                        thickness[prim], wavelength,
                        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, dual_te, dual_tm);
                    atomicAdd(grad_thickness + prim,
                              adj_dot_local(g_r_te, dual_te.d) +
                              adj_dot_local(g_r_tm, dual_tm.d));
                }
                if (need_frequency) {
                    ad::legacy_sionna_slab_fresnel_dual(
                        frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                        thickness[prim], wavelength,
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, dual_te, dual_tm);
                    g_lambda += adj_dot_local(g_r_te, dual_te.d) +
                                adj_dot_local(g_r_tm, dual_tm.d);
                }
            }
            g_field = c3_add(c3_axis(frame.s_hat, g_e_s), c3_axis(frame.p_in, g_e_p));
        }
        if (need_frequency && g_lambda != 0.0f) {
            atomicAdd(grad_frequency, g_lambda * wavelength_dfreq);
        }
    }
}

__global__ void sionna_reflection_accumulate_jvp_kernel(
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
    const float *__restrict__ tangent_eta_r,
    const float *__restrict__ tangent_sigma,
    const float *__restrict__ tangent_gain,
    const float *__restrict__ tangent_thickness,
    float *__restrict__ output_tangent,
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
    float cell_area,
    float wavelength_tangent,
    const float *__restrict__ tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = f3(tx_pol[0], tx_pol[1], tx_pol[2]);
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        float3 origin = f3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 direction = normalize3(f3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        // R5: unnormalized transverse projection of the true TX polarization
        // (see the forward kernel). The seed field carries no material/frequency
        // tangent, so d_field starts at zero, but its sin^2(theta) magnitude
        // must match the forward.
        float3 initial = add3(tx_polarization, scale3(direction, -dot3(tx_polarization, direction)));
        Complex3 field = {c_make(initial.x, 0.0f), c_make(initial.y, 0.0f), c_make(initial.z, 0.0f)};
        Complex3 d_field = {c_make(0.0f, 0.0f), c_make(0.0f, 0.0f), c_make(0.0f, 0.0f)};

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = add3(origin, scale3(direction, t_hit));
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, direction);
            // Primal coefficients from the frozen legacy helper (bit-exact
            // with the forward); tangents from its dual companion.
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, r_te, r_tm);
            ad::DualC dual_te, dual_tm;
            ad::legacy_sionna_slab_fresnel_dual(
                frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                thickness[prim], wavelength,
                tangent_eta_r != nullptr ? tangent_eta_r[prim] : 0.0f,
                tangent_sigma != nullptr ? tangent_sigma[prim] : 0.0f,
                tangent_gain != nullptr ? tangent_gain[prim] : 0.0f,
                tangent_thickness != nullptr ? tangent_thickness[prim] : 0.0f,
                wavelength_tangent, dual_te, dual_tm);
            const Complex d_r_te = c_make(dual_te.d.re, dual_te.d.im);
            const Complex d_r_tm = c_make(dual_tm.d.re, dual_tm.d.im);
            const Complex e_s = c3_dot(field, frame.s_hat);
            const Complex e_p = c3_dot(field, frame.p_in);
            const Complex d_e_s = c3_dot(d_field, frame.s_hat);
            const Complex d_e_p = c3_dot(d_field, frame.p_in);
            const Complex d_w_te = c_add(c_mul(d_r_te, e_s), c_mul(r_te, d_e_s));
            const Complex d_w_tm = c_add(c_mul(d_r_tm, e_p), c_mul(r_tm, d_e_p));
            field = c3_add(c3_axis(frame.s_hat, c_mul(r_te, e_s)),
                           c3_axis(frame.p_out, c_mul(r_tm, e_p)));
            d_field = c3_add(c3_axis(frame.s_hat, d_w_te),
                             c3_axis(frame.p_out, d_w_tm));
            direction = frame.reflected;
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
            const float coeff = solid_angle_per_ray * norm /
                                fmaxf(fabsf(axis_direction), kReflectionEpsilon);
            const float power = c3_power(field) * coeff;
            if (power > 0.0f && isfinite(power)) {
                float d_power = 2.0f * (field.x.r * d_field.x.r + field.x.i * d_field.x.i +
                                        field.y.r * d_field.y.r + field.y.i * d_field.y.i +
                                        field.z.r * d_field.z.r + field.z.i * d_field.z.i) * coeff;
                // Aperture factor tangent: coeff carries (lambda / 4 pi)^2.
                d_power += power * 2.0f / wavelength * wavelength_tangent;
                atomicAdd(output_tangent + i1 * resolution0 + i0, d_power);
            }
        }
    }
}

using channel_native::check_tensor;

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
    CN_REFLECTION_PREPARE_LAUNCH_INPUTS();
    if (sample_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((sample_count + kReflectionBlockSize - 1) / kReflectionBlockSize);
        reflection_launch_inputs_kernel<<<block_count, kReflectionBlockSize, 0, stream>>>(
            CN_REFLECTION_LAUNCH_INPUT_PREFIX(),
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
    double solid_angle_per_ray, double cell_area, at::Tensor tx_pol) {
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
        static_cast<float>(solid_angle_per_ray), static_cast<float>(cell_area),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

namespace {

// Zero-initialized gradient accumulator (memset on the current stream; same
// pattern as los.cu / field_transport_ad.cu).
at::Tensor reflection_zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
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

}  // namespace

// Single source of truth for the reflection AD depth cap: the AD companions
// below stage per-bounce state in kReflectionAdMaxDepth-sized register
// arrays, so the solver reads this cap to reject an over-deep AD
// configuration before any forward launch.
int64_t cn_mc_reflection_ad_max_depth_cuda() {
    return kReflectionAdMaxDepth;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_mc_sionna_reflection_accumulate_backward_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, double wavelength_dfreq,
    at::Tensor tx_pol) {
    TORCH_CHECK(
        contribution_depth <= kReflectionAdMaxDepth,
        "reflection AD companions support contribution_depth <= ",
        kReflectionAdMaxDepth);
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat, "grad_output must be float32");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must have 2 dimensions");
    TORCH_CHECK(
        grad_output.size(0) == resolution1 && grad_output.size(1) == resolution0,
        "grad_output must match the (resolution1, resolution0) map");
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    auto face_options = eta_r.options();
    auto grad_eta_r = reflection_zero_filled({eta_r.size(0)}, face_options);
    auto grad_sigma = reflection_zero_filled({eta_r.size(0)}, face_options);
    auto grad_gain = reflection_zero_filled({eta_r.size(0)}, face_options);
    auto grad_thickness = reflection_zero_filled({eta_r.size(0)}, face_options);
    auto grad_frequency = reflection_zero_filled({1}, face_options);
    if (ray_count == 0 || contribution_depth <= 0 ||
        !(need_materials || need_frequency)) {
        return {grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_frequency};
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (ray_count + kReflectionBlockSize - 1) / kReflectionBlockSize, 65535));
    sionna_reflection_accumulate_backward_kernel<<<blocks, kReflectionBlockSize, 0, stream>>>(
        ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
        trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
        eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
        material_valid.data_ptr<bool>(), thickness.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        need_materials ? grad_eta_r.data_ptr<float>() : nullptr,
        need_materials ? grad_sigma.data_ptr<float>() : nullptr,
        need_materials ? grad_gain.data_ptr<float>() : nullptr,
        need_materials ? grad_thickness.data_ptr<float>() : nullptr,
        need_frequency ? grad_frequency.data_ptr<float>() : nullptr,
        ray_count, trace_depth, static_cast<int>(contribution_depth), static_cast<int>(axis),
        static_cast<float>(plane_position), static_cast<float>(coord0_min),
        static_cast<float>(coord0_max), static_cast<float>(coord1_min),
        static_cast<float>(coord1_max), static_cast<int>(resolution0),
        static_cast<int>(resolution1), static_cast<float>(wavelength),
        static_cast<float>(solid_angle_per_ray), static_cast<float>(cell_area),
        static_cast<float>(wavelength_dfreq),
        grad_output.stride(0), grad_output.stride(1),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_frequency};
}

at::Tensor cn_mc_sionna_reflection_accumulate_jvp_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, double wavelength_tangent,
    at::Tensor tx_pol) {
    TORCH_CHECK(
        contribution_depth <= kReflectionAdMaxDepth,
        "reflection AD companions support contribution_depth <= ",
        kReflectionAdMaxDepth);
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    auto output_tangent = reflection_zero_filled(
        {resolution1, resolution0}, ray_o.options());
    if (ray_count == 0 || contribution_depth <= 0) return output_tangent;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (ray_count + kReflectionBlockSize - 1) / kReflectionBlockSize, 65535));
    sionna_reflection_accumulate_jvp_kernel<<<blocks, kReflectionBlockSize, 0, stream>>>(
        ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
        trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
        eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
        material_valid.data_ptr<bool>(), thickness.data_ptr<float>(),
        has_tangent_eta_r ? tangent_eta_r.data_ptr<float>() : nullptr,
        has_tangent_sigma ? tangent_sigma.data_ptr<float>() : nullptr,
        has_tangent_gain ? tangent_gain.data_ptr<float>() : nullptr,
        has_tangent_thickness ? tangent_thickness.data_ptr<float>() : nullptr,
        output_tangent.data_ptr<float>(),
        ray_count, trace_depth, static_cast<int>(contribution_depth), static_cast<int>(axis),
        static_cast<float>(plane_position), static_cast<float>(coord0_min),
        static_cast<float>(coord0_max), static_cast<float>(coord1_min),
        static_cast<float>(coord1_max), static_cast<int>(resolution0),
        static_cast<int>(resolution1), static_cast<float>(wavelength),
        static_cast<float>(solid_angle_per_ray), static_cast<float>(cell_area),
        static_cast<float>(wavelength_tangent),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_tangent;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    CN_REFLECTION_PREPARE_LAUNCH_INPUTS();
    auto tx_id = at::empty({sample_count}, tx_positions.options().dtype(at::kInt));
    auto light_seed = at::empty({sample_count}, tx_positions.options().dtype(at::kLong));
    if (sample_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((sample_count + kReflectionBlockSize - 1) / kReflectionBlockSize);
        reflection_launch_inputs_kernel<<<block_count, kReflectionBlockSize, 0, stream>>>(
            CN_REFLECTION_LAUNCH_INPUT_PREFIX(),
            tx_id.data_ptr<int>(),
            light_seed.data_ptr<int64_t>(),
            tx_index,
            sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ray_o, ray_tmax, active, tx_pol, tx_id, light_seed};
}

#undef CN_REFLECTION_LAUNCH_INPUT_PREFIX
#undef CN_REFLECTION_PREPARE_LAUNCH_INPUTS
