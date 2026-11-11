// Copyright Xingyu Chen.
// Implements reflection CUDA operations.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"
#include "constants.cuh"
#include "math.cuh"
#include "ordered_sum.cuh"
#include "reflection_receivers.h"
#include <rayd/field_transport.cuh>
#include <src/field_transport_ad.cuh>
#include "layered_reflection.cuh"

#include <algorithm>
#include <tuple>
#include <vector>

#define CHANNEL_REFLECTION_PREPARE_LAUNCH_INPUTS()                                         \
    check_tensor(tx_positions, "tx_positions", at::kFloat, 2);                       \
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");    \
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");\
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");               \
    auto ray_o = at::empty({sample_count, 3}, tx_positions.options());                 \
    auto ray_tmax = at::empty({0}, tx_positions.options());                           \
    auto active = at::empty({sample_count}, tx_positions.options().dtype(at::kBool)); \
    auto tx_pol = at::empty({sample_count, 3}, tx_positions.options())

#define CHANNEL_REFLECTION_LAUNCH_INPUT_PREFIX()                                            \
    tx_positions.data_ptr<float>(),                                                   \
    ray_o.data_ptr<float>(),                                                          \
    active.data_ptr<bool>(),                                                          \
    tx_pol.data_ptr<float>()

namespace {

constexpr int kReflectionBlockSize = 256;
constexpr int kReflectionAdMaxDepth = 8;
using channel::kPiF;
constexpr float kReflectionEpsilon = 1.0e-6f;
namespace transport = rayd::shared::field_transport;
namespace ad = rayd::torch::field_transport_ad;

using channel::math::Complex;
using channel::math::Complex3;
namespace cmath = channel::math;

__device__ __forceinline__ float component3(float3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__device__ __forceinline__ void plane_coords(float3 v, int axis, float &a, float &b) {
    if (axis == 0) { a = v.y; b = v.z; }
    else if (axis == 1) { a = v.x; b = v.z; }
    else { a = v.x; b = v.y; }
}

namespace layered = channel::layered_reflection;

__device__ __forceinline__ void slab_coefficients(
    float cos_theta,
    float eta_r,
    float sigma,
    float gain,
    float thickness,
    float wavelength,
    int material,
    const layered::LayerTables &layers,
    Complex &r_te,
    Complex &r_tm) {
    rayd::shared::diffraction::Complex shared_te;
    rayd::shared::diffraction::Complex shared_tm;
    layered::legacy_face_fresnel(
        cos_theta,
        eta_r,
        sigma,
        gain,
        thickness,
        wavelength,
        material,
        layers,
        shared_te,
        shared_tm);
    r_te = cmath::complex(shared_te.re, shared_te.im);
    r_tm = cmath::complex(shared_tm.re, shared_tm.im);
}

// ---------------------------------------------------------------------------
// Backward / JVP companions of slab_reflection_accumulate_kernel
// (solver derivatives). Fixed-winner contract: the RayD trace tape (valid / t /
// prim), the sampled directions and the ray origins are frozen constants of
// the differentiation, so the deposit binning, the incidence cosines and the
// polarization frames are all constant; every derivative flows through the
// legacy slab Fresnel coefficients (eta_r / sigma / gain / thickness /
// wavelength) and through the (lambda/4pi)^2 aperture factor of the deposit
// weight. The per-ray deposit weight does not depend on the ray origin at
// all, so the ray-origin gradient of this map is exactly zero (the Python
// dispatch layer returns it without a launch). The walk below REPLAYS
// slab_reflection_accumulate_kernel operation by operation; edit the primal
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
    const float3 vertical = cmath::vec3(0.0f, 0.0f, 1.0f);
    frame.normal = cmath::normalize_rsqrt_safe(
        cmath::vec3(face_normals[3 * prim], face_normals[3 * prim + 1], face_normals[3 * prim + 2]));
    if (cmath::dot(direction, frame.normal) > 0.0f)
        frame.normal = cmath::scale(frame.normal, -1.0f);
    float3 s_hat = cmath::cross(frame.normal, direction);
    if (cmath::dot(s_hat, s_hat) < 1.0e-12f)
        s_hat = cmath::normalize_rsqrt_safe(cmath::cross(
            fabsf(direction.z) < 0.9f ? vertical : cmath::vec3(0.0f, 1.0f, 0.0f), direction));
    else
        s_hat = cmath::normalize_rsqrt_safe(s_hat);
    frame.s_hat = s_hat;
    frame.p_in = cmath::normalize_rsqrt_safe(cmath::cross(s_hat, direction));
    frame.reflected = cmath::normalize_rsqrt_safe(
        cmath::add(direction, cmath::scale(frame.normal, -2.0f * cmath::dot(direction, frame.normal))));
    frame.p_out = cmath::normalize_rsqrt_safe(cmath::cross(s_hat, frame.reflected));
    frame.cos_theta = fabsf(cmath::dot(direction, frame.normal));
    return frame;
}

// Real-pair contribution of a dual coefficient against a complex cotangent.
__device__ __forceinline__ float adj_dot_local(
    Complex g, rayd::shared::diffraction::Complex d) {
    return g.r * d.re + g.i * d.im;
}

// Receivers one reflection accumulate launch deposits into: one axis-aligned grid (kind 0), whose
// cells take the flux through their area, or a range of point receivers (kind 1), each a sphere
// that takes the flux through its cross-section pi * radius^2. A point's radius is `radius`
// when positive, else `relative_radius` times its straight distance to the transmitter (at least
// kMinPointRadius).
struct ReceiverTargets {
    int kind;
    int axis;
    float plane_position;
    float coord0_min;
    float coord0_max;
    float coord1_min;
    float coord1_max;
    int resolution0;
    int resolution1;
    float cell_area;
    const float* points;
    int64_t point_begin;
    int64_t point_end;
    float radius;
    float relative_radius;
};

constexpr float kMinPointRadius = 1.0e-3f;
constexpr float kMinSegmentT = 1.0e-4f;

__device__ __forceinline__ float point_radius(const ReceiverTargets& rx, float3 point, float3 tx) {
    if (rx.radius > 0.0f) return rx.radius;
    return fmaxf(kMinPointRadius, rx.relative_radius * cmath::length(cmath::sub(point, tx)));
}

// Calls visit(target, t, coeff) for every receiver the segment origin + t * direction,
// kMinSegmentT < t < blocker_t, deposits into; the deposited power is |field|^2 * coeff.
template <class Visit>
__device__ __forceinline__ void visit_deposits(
    const ReceiverTargets& rx, float3 tx, float3 origin, float3 direction, float blocker_t,
    float solid_angle, float wavelength, Visit&& visit) {
    const float aperture = (wavelength / (4.0f * kPiF)) * (wavelength / (4.0f * kPiF));
    if (rx.kind == 0) {
        const float axis_direction = component3(direction, rx.axis);
        if (fabsf(axis_direction) <= kReflectionEpsilon) return;
        const float t_plane = (rx.plane_position - component3(origin, rx.axis)) / axis_direction;
        if (!(t_plane > kMinSegmentT && t_plane < blocker_t)) return;
        const float3 target = cmath::add(origin, cmath::scale(direction, t_plane));
        float coord0, coord1;
        plane_coords(target, rx.axis, coord0, coord1);
        if (coord0 < rx.coord0_min || coord0 >= rx.coord0_max ||
            coord1 < rx.coord1_min || coord1 >= rx.coord1_max) return;
        const int i0 = min(max(static_cast<int>((coord0 - rx.coord0_min) / (rx.coord0_max - rx.coord0_min) *
                                                rx.resolution0), 0), rx.resolution0 - 1);
        const int i1 = min(max(static_cast<int>((coord1 - rx.coord1_min) / (rx.coord1_max - rx.coord1_min) *
                                                rx.resolution1), 0), rx.resolution1 - 1);
        const float coeff = solid_angle * (aperture / fmaxf(rx.cell_area, kReflectionEpsilon)) /
                            fmaxf(fabsf(axis_direction), kReflectionEpsilon);
        // Row-major receiver cell: i0 runs along the grid rows (coord0), i1 along its columns.
        visit(static_cast<int64_t>(i0) * rx.resolution1 + i1, t_plane, coeff);
        return;
    }
    for (int64_t index = rx.point_begin; index < rx.point_end; ++index) {
        const float3 point = cmath::vec3(rx.points[3 * index], rx.points[3 * index + 1], rx.points[3 * index + 2]);
        const float3 offset = cmath::sub(point, origin);
        const float t = cmath::dot(offset, direction);
        if (!(t > kMinSegmentT && t < blocker_t)) continue;
        const float distance_sq = fmaxf(cmath::dot(offset, offset) - t * t, 0.0f);
        const float radius = point_radius(rx, point, tx);
        if (!(distance_sq < radius * radius)) continue;
        visit(index - rx.point_begin, t, solid_angle * aperture / (kPiF * radius * radius));
    }
}

__device__ __forceinline__ float trace_blocker(
    const bool* trace_valid, const float* trace_t, int64_t ray, int trace_depth, int next_depth) {
    if (next_depth >= trace_depth) return 1.0e30f;
    const int64_t slot = ray * static_cast<int64_t>(trace_depth) + next_depth;
    return trace_valid[slot] ? trace_t[slot] : 1.0e30f;
}

__device__ __forceinline__ Complex3 initial_reflection_field(float3 tx_polarization, float3 direction) {
    // Unnormalized transverse projection of the true TX polarization onto the launch direction
    // (short-dipole sin(theta) pattern); a zero is the physical null.
    const float3 initial = cmath::add(
        tx_polarization, cmath::scale(direction, -cmath::dot(tx_polarization, direction)));
    return {cmath::complex(initial.x, 0.0f), cmath::complex(initial.y, 0.0f), cmath::complex(initial.z, 0.0f)};
}

__global__ void slab_reflection_accumulate_kernel(
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
    const int *__restrict__ face_material,
    layered::LayerTables layers,
    float *__restrict__ output,
    int64_t ray_count,
    int trace_depth,
    int contribution_depth,
    ReceiverTargets rx,
    float wavelength,
    const float *__restrict__ solid_angle_per_ray,
    const float *__restrict__ tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = cmath::vec3(tx_pol[0], tx_pol[1], tx_pol[2]);
    const float solid_angle = solid_angle_per_ray[0];
    const float3 vertical = cmath::vec3(0.0f, 0.0f, 1.0f);
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        const float3 tx = cmath::vec3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 origin = tx;
        float3 direction = cmath::normalize_rsqrt_safe(cmath::vec3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        Complex3 field = initial_reflection_field(tx_polarization, direction);

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = cmath::add(origin, cmath::scale(direction, t_hit));
            float3 normal = cmath::normalize_rsqrt_safe(cmath::vec3(face_normals[3*prim], face_normals[3*prim+1], face_normals[3*prim+2]));
            if (cmath::dot(direction, normal) > 0.0f) normal = cmath::scale(normal, -1.0f);

            float3 s_hat = cmath::cross(normal, direction);
            if (cmath::dot(s_hat, s_hat) < 1.0e-12f)
                s_hat = cmath::normalize_rsqrt_safe(cmath::cross(fabsf(direction.z) < 0.9f ? vertical : cmath::vec3(0.0f,1.0f,0.0f), direction));
            else
                s_hat = cmath::normalize_rsqrt_safe(s_hat);
            const float3 p_in = cmath::normalize_rsqrt_safe(cmath::cross(s_hat, direction));
            const float3 reflected = cmath::normalize_rsqrt_safe(cmath::add(direction, cmath::scale(normal, -2.0f*cmath::dot(direction, normal))));
            const float3 p_out = cmath::normalize_rsqrt_safe(cmath::cross(s_hat, reflected));
            Complex r_te, r_tm;
            slab_coefficients(fabsf(cmath::dot(direction, normal)), eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, face_material[prim], layers, r_te, r_tm);
            const Complex e_s = cmath::complex3_dot_real(field, s_hat);
            const Complex e_p = cmath::complex3_dot_real(field, p_in);
            field = cmath::complex3_add(cmath::complex3_axis(s_hat, cmath::complex_mul(r_te, e_s)),
                           cmath::complex3_axis(p_out, cmath::complex_mul(r_tm, e_p)));
            direction = reflected;
            origin = hit;

            const float blocker_t = trace_blocker(trace_valid, trace_t, ray, trace_depth, depth + 1);
            const float field_power = cmath::complex3_power(field);
            visit_deposits(rx, tx, origin, direction, blocker_t, solid_angle, wavelength,
                           [&](int64_t target, float, float coeff) {
                               const float power = field_power * coeff;
                               if (power > 0.0f && isfinite(power)) atomicAdd(output + target, power);
                           });
        }
    }
}

// Final reflected segment of every ray that completed all trace_depth bounces: its spawn point
// (the last hit moved kSpawnScale * max(1, |hit|_inf) along the segment), its direction, and the
// farthest deposit distance any receiver block needs (max over launches; first_block resets it).
// The bounded trace of that segment replaces an unbounded extra closest-hit bounce.
constexpr float kSpawnScale = 1.0e-5f;

__global__ void reflection_final_segment_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ trace_valid,
    const float *__restrict__ trace_t,
    const int *__restrict__ trace_prim,
    const float *__restrict__ face_normals,
    const bool *__restrict__ material_valid,
    int64_t ray_count,
    int trace_depth,
    ReceiverTargets rx,
    float wavelength,
    bool first_block,
    float *__restrict__ final_o,
    float *__restrict__ final_d,
    float *__restrict__ final_tmax,
    float *__restrict__ final_offset,
    bool *__restrict__ final_active) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        float reach = first_block ? 0.0f : final_tmax[ray] + final_offset[ray];
        const float3 tx = cmath::vec3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 origin = tx;
        float3 direction = cmath::normalize_rsqrt_safe(cmath::vec3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        bool complete = trace_depth > 0;
        for (int depth = 0; depth < trace_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            const int prim = trace_valid[slot] ? trace_prim[slot] : -1;
            if (prim < 0 || !material_valid[prim]) {
                complete = false;
                break;
            }
            origin = cmath::add(origin, cmath::scale(direction, trace_t[slot]));
            direction = reflection_bounce_frame(face_normals, prim, direction).reflected;
        }
        if (complete) {
            visit_deposits(rx, tx, origin, direction, 1.0e30f, 1.0f, wavelength,
                           [&](int64_t, float t, float) { reach = fmaxf(reach, t); });
        }
        const float offset = kSpawnScale * fmaxf(1.0f, fmaxf(fabsf(origin.x), fmaxf(fabsf(origin.y), fabsf(origin.z))));
        const float3 spawn = cmath::add(origin, cmath::scale(direction, offset));
        final_o[3*ray] = spawn.x;
        final_o[3*ray+1] = spawn.y;
        final_o[3*ray+2] = spawn.z;
        final_d[3*ray] = direction.x;
        final_d[3*ray+1] = direction.y;
        final_d[3*ray+2] = direction.z;
        final_offset[ray] = offset;
        final_tmax[ray] = reach - offset;
        final_active[ray] = complete && reach - offset > 0.0f;
    }
}

// Appends the bounded final-segment hit as trace column trace_depth (t measured from the last
// hit), so the accumulate kernels read every segment blocker from one trace table.
__global__ void reflection_append_final_kernel(
    const bool *__restrict__ trace_valid,
    const float *__restrict__ trace_t,
    const int *__restrict__ trace_prim,
    const bool *__restrict__ final_valid,
    const float *__restrict__ final_t,
    const int *__restrict__ final_prim,
    const float *__restrict__ final_offset,
    int64_t ray_count,
    int trace_depth,
    bool *__restrict__ out_valid,
    float *__restrict__ out_t,
    int *__restrict__ out_prim) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int out_depth = trace_depth + 1;
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        for (int depth = 0; depth < trace_depth; ++depth) {
            const int64_t in_slot = ray * trace_depth + depth;
            const int64_t out_slot = ray * out_depth + depth;
            out_valid[out_slot] = trace_valid[in_slot];
            out_t[out_slot] = trace_t[in_slot];
            out_prim[out_slot] = trace_prim[in_slot];
        }
        const int64_t out_slot = ray * out_depth + trace_depth;
        out_valid[out_slot] = final_valid[ray];
        out_t[out_slot] = final_t[ray] + final_offset[ray];
        out_prim[out_slot] = final_prim[ray];
    }
}

// Launch rays for every transmitter: the Fibonacci lattice of `count` directions, rotated by a
// uniformly random rotation drawn from (seed, transmitter), then folded into the half-spaces
// that can reach the scene bounds (a transmitter above the bounds keeps only downward
// directions, and so on per axis). A direction outside them cannot hit geometry, so the fold
// keeps the estimator unbiased; each kept ray carries 4 pi / (2^folds * count).
__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

__device__ __forceinline__ float unit_from_bits(uint64_t bits) {
    return static_cast<float>(bits & 0xFFFFFFull) * (1.0f / 16777216.0f);
}

__global__ void reflection_rays_kernel(
    const float *__restrict__ tx_positions,
    const float *__restrict__ bounds,
    int64_t tx_count,
    int64_t count,
    int64_t seed,
    float *__restrict__ ray_o,
    float *__restrict__ ray_d,
    bool *__restrict__ active,
    float *__restrict__ solid_angle) {
    const int64_t total = tx_count * count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const int64_t tx_index = index / count;
        const int64_t sample = index - tx_index * count;
        const float3 tx = cmath::vec3(
            tx_positions[3 * tx_index], tx_positions[3 * tx_index + 1], tx_positions[3 * tx_index + 2]);
        // Lattice point: z uniform in cell centers, azimuth by the golden ratio in 32-bit fixed point.
        const float z = 1.0f - (2.0f * static_cast<float>(sample) + 1.0f) / static_cast<float>(count);
        const uint32_t phase = static_cast<uint32_t>(static_cast<uint64_t>(sample) * 2654435769ull);
        const float phi = 2.0f * kPiF * (static_cast<float>(phase) * (1.0f / 4294967296.0f));
        const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        float sin_phi, cos_phi;
        sincosf(phi, &sin_phi, &cos_phi);
        const float3 lattice = cmath::vec3(radial * cos_phi, radial * sin_phi, z);
        // Uniform random rotation (Shoemake quaternion) from (seed, transmitter).
        const uint64_t key = splitmix64(static_cast<uint64_t>(seed) ^ splitmix64(static_cast<uint64_t>(tx_index)));
        const float u1 = unit_from_bits(key);
        const float u2 = unit_from_bits(key >> 24);
        const float u3 = unit_from_bits(splitmix64(key));
        const float a = sqrtf(1.0f - u1);
        const float b = sqrtf(u1);
        float s2, c2, s3, c3;
        sincosf(2.0f * kPiF * u2, &s2, &c2);
        sincosf(2.0f * kPiF * u3, &s3, &c3);
        const float qx = a * s2, qy = a * c2, qz = b * s3, qw = b * c3;
        const float3 qv = cmath::vec3(qx, qy, qz);
        const float3 uv = cmath::cross(qv, lattice);
        const float3 uuv = cmath::cross(qv, uv);
        float3 direction = cmath::add(lattice, cmath::scale(cmath::add(cmath::scale(uv, qw), uuv), 2.0f));
        int folds = 0;
        if (bounds != nullptr) {
            float* components[3] = {&direction.x, &direction.y, &direction.z};
            const float position[3] = {tx.x, tx.y, tx.z};
            for (int axis = 0; axis < 3; ++axis) {
                if (position[axis] > bounds[3 + axis]) {
                    *components[axis] = -fabsf(*components[axis]);
                    ++folds;
                } else if (position[axis] < bounds[axis]) {
                    *components[axis] = fabsf(*components[axis]);
                    ++folds;
                }
            }
        }
        direction = cmath::normalize_rsqrt_safe(direction);
        ray_o[3 * index] = tx.x;
        ray_o[3 * index + 1] = tx.y;
        ray_o[3 * index + 2] = tx.z;
        ray_d[3 * index] = direction.x;
        ray_d[3 * index + 1] = direction.y;
        ray_d[3 * index + 2] = direction.z;
        active[index] = true;
        if (sample == 0)
            solid_angle[tx_index] = 4.0f * kPiF / (static_cast<float>(1 << folds) * static_cast<float>(count));
    }
}

__global__ void scene_bounds_kernel(const float *__restrict__ vertices, int64_t count, float *__restrict__ bounds) {
    __shared__ float lower[3][kReflectionBlockSize];
    __shared__ float upper[3][kReflectionBlockSize];
    float lo[3] = {INFINITY, INFINITY, INFINITY};
    float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = vertices[3 * index + axis];
            lo[axis] = fminf(lo[axis], value);
            hi[axis] = fmaxf(hi[axis], value);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        lower[axis][threadIdx.x] = lo[axis];
        upper[axis][threadIdx.x] = hi[axis];
    }
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (static_cast<int>(threadIdx.x) < width) {
            for (int axis = 0; axis < 3; ++axis) {
                lower[axis][threadIdx.x] = fminf(lower[axis][threadIdx.x], lower[axis][threadIdx.x + width]);
                upper[axis][threadIdx.x] = fmaxf(upper[axis][threadIdx.x], upper[axis][threadIdx.x + width]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < 3) {
        bounds[threadIdx.x] = lower[threadIdx.x][0];
        bounds[3 + threadIdx.x] = upper[threadIdx.x][0];
    }
}

// Adds value to base[index] with one atomic per distinct index among the active lanes of the
// warp: lanes that share an index combine their values first. Face tables are small in typical
// scenes, so this removes most of the same-address contention.
__device__ __forceinline__ void warp_aggregated_add(float* base, int index, float value) {
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, index);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int leader = __ffs(peers) - 1;
    float sum = 0.0f;
    for (unsigned remaining = peers; remaining != 0u; remaining &= remaining - 1u) {
        const int source = __ffs(remaining) - 1;
        sum += __shfl_sync(peers, value, source);
    }
    if (lane == leader) atomicAdd(base + index, sum);
}

__global__ void slab_reflection_accumulate_backward_kernel(
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
    const int *__restrict__ face_material,
    layered::LayerTables layers,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_eta_r,
    float *__restrict__ grad_sigma,
    float *__restrict__ grad_gain,
    float *__restrict__ grad_thickness,
    float *__restrict__ ray_grad_frequency,
    int64_t ray_count,
    int trace_depth,
    int contribution_depth,
    ReceiverTargets rx,
    float wavelength,
    const float *__restrict__ solid_angle_per_ray,
    float wavelength_dfreq,
    int64_t grad_stride0,
    int64_t grad_stride1,
    const float *__restrict__ tx_pol) {
    const bool need_materials = grad_eta_r != nullptr;
    const bool need_frequency = ray_grad_frequency != nullptr;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = cmath::vec3(tx_pol[0], tx_pol[1], tx_pol[2]);
    const float solid_angle = solid_angle_per_ray[0];
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        // Forward replay of slab_reflection_accumulate_kernel, recording the per-bounce state the
        // reverse sweep needs: deposit_weight is sum over the segment's deposits of
        // grad_output[target] * coeff.
        const float3 tx = cmath::vec3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 origin = tx;
        float3 direction = cmath::normalize_rsqrt_safe(cmath::vec3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        const Complex3 initial_field = initial_reflection_field(tx_polarization, direction);
        Complex3 field = initial_field;

        Complex3 field_after[kReflectionAdMaxDepth];
        float3 dir_in[kReflectionAdMaxDepth];
        int prim_at[kReflectionAdMaxDepth];
        float deposit_weight[kReflectionAdMaxDepth];
        int bounce_count = 0;

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = cmath::add(origin, cmath::scale(direction, t_hit));
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, direction);
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, face_material[prim], layers, r_te, r_tm);
            const Complex e_s = cmath::complex3_dot_real(field, frame.s_hat);
            const Complex e_p = cmath::complex3_dot_real(field, frame.p_in);
            field = cmath::complex3_add(cmath::complex3_axis(frame.s_hat, cmath::complex_mul(r_te, e_s)),
                           cmath::complex3_axis(frame.p_out, cmath::complex_mul(r_tm, e_p)));
            dir_in[depth] = direction;
            prim_at[depth] = prim;
            field_after[depth] = field;
            bounce_count = depth + 1;
            direction = frame.reflected;
            origin = hit;

            const float blocker_t = trace_blocker(trace_valid, trace_t, ray, trace_depth, depth + 1);
            const float field_power = cmath::complex3_power(field);
            float weight = 0.0f;
            visit_deposits(rx, tx, origin, direction, blocker_t, solid_angle, wavelength,
                           [&](int64_t target, float, float coeff) {
                               const float power = field_power * coeff;
                               if (!(power > 0.0f && isfinite(power))) return;
                               const int64_t i0 = rx.kind == 0 ? target / rx.resolution1 : 0;
                               const int64_t i1 = target - i0 * rx.resolution1;
                               const float g_dep = rx.kind == 0
                                   ? grad_output[i0 * grad_stride0 + i1 * grad_stride1]
                                   : grad_output[target * grad_stride0];
                               weight += g_dep * coeff;
                           });
            deposit_weight[depth] = weight;
        }

        // Reverse sweep: fold deposit cotangents into the field chain, pull them through each
        // bounce, and dot the Fresnel cotangents against per-parameter forward duals of the
        // frozen legacy slab response.
        Complex3 g_field = {cmath::complex(0.0f, 0.0f), cmath::complex(0.0f, 0.0f), cmath::complex(0.0f, 0.0f)};
        float g_lambda = 0.0f;
        for (int depth = bounce_count - 1; depth >= 0; --depth) {
            if (deposit_weight[depth] != 0.0f) {
                const float scale = 2.0f * deposit_weight[depth];
                const Complex3 f = field_after[depth];
                g_field.x = cmath::complex_add(g_field.x, cmath::complex_scale(f.x, scale));
                g_field.y = cmath::complex_add(g_field.y, cmath::complex_scale(f.y, scale));
                g_field.z = cmath::complex_add(g_field.z, cmath::complex_scale(f.z, scale));
                // Every deposit weight carries (lambda / 4 pi)^2: its direct wavelength derivative
                // is 2 * power / lambda.
                g_lambda += deposit_weight[depth] * cmath::complex3_power(f) * 2.0f / wavelength;
            }
            const int prim = prim_at[depth];
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, dir_in[depth]);
            const Complex3 field_before =
                depth == 0 ? initial_field : field_after[depth - 1];
            const Complex e_s = cmath::complex3_dot_real(field_before, frame.s_hat);
            const Complex e_p = cmath::complex3_dot_real(field_before, frame.p_in);
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, face_material[prim], layers, r_te, r_tm);
            // field_after = s_hat * (r_te * e_s) + p_out * (r_tm * e_p); the axis expansions are
            // real-linear, so their adjoints are the same real-axis dots.
            const Complex g_w_te = cmath::complex3_dot_real(g_field, frame.s_hat);
            const Complex g_w_tm = cmath::complex3_dot_real(g_field, frame.p_out);
            // Real-pair adjoint of the complex products w = r * e.
            const Complex g_r_te = cmath::complex(
                g_w_te.r * e_s.r + g_w_te.i * e_s.i,
                -g_w_te.r * e_s.i + g_w_te.i * e_s.r);
            const Complex g_r_tm = cmath::complex(
                g_w_tm.r * e_p.r + g_w_tm.i * e_p.i,
                -g_w_tm.r * e_p.i + g_w_tm.i * e_p.r);
            const Complex g_e_s = cmath::complex(
                g_w_te.r * r_te.r + g_w_te.i * r_te.i,
                -g_w_te.r * r_te.i + g_w_te.i * r_te.r);
            const Complex g_e_p = cmath::complex(
                g_w_tm.r * r_tm.r + g_w_tm.i * r_tm.i,
                -g_w_tm.r * r_tm.i + g_w_tm.i * r_tm.r);
            if (need_materials || need_frequency) {
                // One dual evaluation carries the four material tangents and the wavelength.
                constexpr float kSeedEps[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                constexpr float kSeedSigma[5] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
                constexpr float kSeedGain[5] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
                constexpr float kSeedThickness[5] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
                constexpr float kSeedWavelength[5] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
                rayd::shared::diffraction::Complex dual_r_te, dual_r_tm;
                rayd::shared::diffraction::Complex d_te[5], d_tm[5];
                layered::legacy_face_fresnel_dual_multi<5>(
                    frame.cos_theta, eta_r[prim], sigma[prim], gain[prim], thickness[prim], wavelength,
                    face_material[prim], layers, kSeedEps, kSeedSigma, kSeedGain, kSeedThickness, kSeedWavelength, dual_r_te,
                    dual_r_tm, d_te, d_tm);
                if (need_materials) {
                    warp_aggregated_add(grad_eta_r, prim, adj_dot_local(g_r_te, d_te[0]) + adj_dot_local(g_r_tm, d_tm[0]));
                    warp_aggregated_add(grad_sigma, prim, adj_dot_local(g_r_te, d_te[1]) + adj_dot_local(g_r_tm, d_tm[1]));
                    warp_aggregated_add(grad_gain, prim, adj_dot_local(g_r_te, d_te[2]) + adj_dot_local(g_r_tm, d_tm[2]));
                    warp_aggregated_add(grad_thickness, prim,
                                        adj_dot_local(g_r_te, d_te[3]) + adj_dot_local(g_r_tm, d_tm[3]));
                }
                if (need_frequency)
                    g_lambda += adj_dot_local(g_r_te, d_te[4]) + adj_dot_local(g_r_tm, d_tm[4]);
            }
            g_field = cmath::complex3_add(cmath::complex3_axis(frame.s_hat, g_e_s), cmath::complex3_axis(frame.p_in, g_e_p));
        }
        if (need_frequency) ray_grad_frequency[ray] = g_lambda * wavelength_dfreq;
    }
}

__global__ void slab_reflection_accumulate_jvp_kernel(
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
    const int *__restrict__ face_material,
    layered::LayerTables layers,
    const float *__restrict__ tangent_eta_r,
    const float *__restrict__ tangent_sigma,
    const float *__restrict__ tangent_gain,
    const float *__restrict__ tangent_thickness,
    float *__restrict__ output_tangent,
    int64_t ray_count,
    int trace_depth,
    int contribution_depth,
    ReceiverTargets rx,
    float wavelength,
    const float *__restrict__ solid_angle_per_ray,
    float wavelength_tangent,
    const float *__restrict__ tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float3 tx_polarization = cmath::vec3(tx_pol[0], tx_pol[1], tx_pol[2]);
    const float solid_angle = solid_angle_per_ray[0];
    for (int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ray < ray_count; ray += stride) {
        const float3 tx = cmath::vec3(ray_o[3*ray], ray_o[3*ray+1], ray_o[3*ray+2]);
        float3 origin = tx;
        float3 direction = cmath::normalize_rsqrt_safe(cmath::vec3(ray_d[3*ray], ray_d[3*ray+1], ray_d[3*ray+2]));
        // The seed field carries no material/frequency tangent, so d_field starts at zero.
        Complex3 field = initial_reflection_field(tx_polarization, direction);
        Complex3 d_field = {cmath::complex(0.0f, 0.0f), cmath::complex(0.0f, 0.0f), cmath::complex(0.0f, 0.0f)};

        for (int depth = 0; depth < contribution_depth; ++depth) {
            const int64_t slot = ray * static_cast<int64_t>(trace_depth) + depth;
            if (!trace_valid[slot]) break;
            const int prim = trace_prim[slot];
            if (prim < 0 || !material_valid[prim]) break;
            const float t_hit = trace_t[slot];
            float3 hit = cmath::add(origin, cmath::scale(direction, t_hit));
            const ReflectionBounceFrame frame =
                reflection_bounce_frame(face_normals, prim, direction);
            // Primal coefficients from the frozen legacy helper (bit-exact with the forward);
            // tangents from its dual companion.
            Complex r_te, r_tm;
            slab_coefficients(frame.cos_theta, eta_r[prim], sigma[prim],
                              gain[prim], thickness[prim], wavelength, face_material[prim], layers, r_te, r_tm);
            ad::DualC dual_te, dual_tm;
            layered::legacy_face_fresnel_dual(
                frame.cos_theta, eta_r[prim], sigma[prim], gain[prim],
                thickness[prim], wavelength, face_material[prim], layers,
                tangent_eta_r != nullptr ? tangent_eta_r[prim] : 0.0f,
                tangent_sigma != nullptr ? tangent_sigma[prim] : 0.0f,
                tangent_gain != nullptr ? tangent_gain[prim] : 0.0f,
                tangent_thickness != nullptr ? tangent_thickness[prim] : 0.0f,
                wavelength_tangent, dual_te, dual_tm);
            const Complex d_r_te = cmath::complex(dual_te.d.re, dual_te.d.im);
            const Complex d_r_tm = cmath::complex(dual_tm.d.re, dual_tm.d.im);
            const Complex e_s = cmath::complex3_dot_real(field, frame.s_hat);
            const Complex e_p = cmath::complex3_dot_real(field, frame.p_in);
            const Complex d_e_s = cmath::complex3_dot_real(d_field, frame.s_hat);
            const Complex d_e_p = cmath::complex3_dot_real(d_field, frame.p_in);
            const Complex d_w_te = cmath::complex_add(cmath::complex_mul(d_r_te, e_s), cmath::complex_mul(r_te, d_e_s));
            const Complex d_w_tm = cmath::complex_add(cmath::complex_mul(d_r_tm, e_p), cmath::complex_mul(r_tm, d_e_p));
            field = cmath::complex3_add(cmath::complex3_axis(frame.s_hat, cmath::complex_mul(r_te, e_s)),
                           cmath::complex3_axis(frame.p_out, cmath::complex_mul(r_tm, e_p)));
            d_field = cmath::complex3_add(cmath::complex3_axis(frame.s_hat, d_w_te),
                             cmath::complex3_axis(frame.p_out, d_w_tm));
            direction = frame.reflected;
            origin = hit;

            const float blocker_t = trace_blocker(trace_valid, trace_t, ray, trace_depth, depth + 1);
            const float field_power = cmath::complex3_power(field);
            const float d_field_power = 2.0f * (field.x.r * d_field.x.r + field.x.i * d_field.x.i +
                                                field.y.r * d_field.y.r + field.y.i * d_field.y.i +
                                                field.z.r * d_field.z.r + field.z.i * d_field.z.i);
            visit_deposits(rx, tx, origin, direction, blocker_t, solid_angle, wavelength,
                           [&](int64_t target, float, float coeff) {
                               const float power = field_power * coeff;
                               if (!(power > 0.0f && isfinite(power))) return;
                               // Aperture factor tangent: coeff carries (lambda / 4 pi)^2.
                               const float d_power = d_field_power * coeff +
                                                     power * 2.0f / wavelength * wavelength_tangent;
                               atomicAdd(output_tangent + target, d_power);
                           });
        }
    }
}

using channel::check_tensor;

}  // namespace

namespace {

// Zero-initialized accumulator (memset on the current stream; same pattern as los.cu).
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

int reflection_blocks(int64_t count) {
    return static_cast<int>(std::min<int64_t>(
        (count + kReflectionBlockSize - 1) / kReflectionBlockSize, 65535));
}

// Point receivers per launch, so one launch visits at most kPointVisitsPerLaunch
// (ray, bounce, receiver) triples and stays short.
constexpr int64_t kPointVisitsPerLaunch = int64_t{1} << 27;

int64_t point_chunk(int64_t ray_count, int64_t depth, int64_t points) {
    const int64_t per_point = std::max<int64_t>(1, ray_count * std::max<int64_t>(depth, 1));
    return std::max<int64_t>(1, std::min<int64_t>(points, kPointVisitsPerLaunch / per_point));
}

ReceiverTargets grid_targets(const ReflectionReceiverArgs& args) {
    ReceiverTargets rx{};
    rx.kind = 0;
    rx.axis = static_cast<int>(args.axis);
    rx.plane_position = static_cast<float>(args.plane_position);
    rx.coord0_min = static_cast<float>(args.coord0_min);
    rx.coord0_max = static_cast<float>(args.coord0_max);
    rx.coord1_min = static_cast<float>(args.coord1_min);
    rx.coord1_max = static_cast<float>(args.coord1_max);
    rx.resolution0 = static_cast<int>(args.resolution0);
    rx.resolution1 = static_cast<int>(args.resolution1);
    rx.cell_area = static_cast<float>(args.cell_area);
    return rx;
}

ReceiverTargets point_targets(const ReflectionReceiverArgs& args, int64_t begin, int64_t end) {
    ReceiverTargets rx{};
    rx.kind = 1;
    rx.resolution0 = 1;
    rx.resolution1 = 1;
    rx.points = args.points.data_ptr<float>();
    rx.point_begin = begin;
    rx.point_end = end;
    rx.radius = args.point_radius > 0.0 ? static_cast<float>(args.point_radius) : 0.0f;
    rx.relative_radius = args.point_radius > 0.0 ? 0.0f : static_cast<float>(-args.point_radius);
    return rx;
}

bool point_mode(const ReflectionReceiverArgs& args) {
    return args.points.defined() && args.points.numel() > 0;
}

// Calls launch(targets, output_offset) once per receiver block chunk.
template <class Launch>
void for_each_target_chunk(const ReflectionReceiverArgs& args, int64_t ray_count, int64_t depth,
                           Launch&& launch) {
    if (!point_mode(args)) {
        launch(grid_targets(args), int64_t{0});
        return;
    }
    const int64_t points = args.points.size(0);
    const int64_t chunk = point_chunk(ray_count, depth, points);
    for (int64_t begin = 0; begin < points; begin += chunk) {
        const int64_t end = std::min(points, begin + chunk);
        launch(point_targets(args, begin, end), begin);
    }
}

std::vector<int64_t> receiver_output_shape(const ReflectionReceiverArgs& args) {
    if (point_mode(args)) return {1, args.points.size(0)};
    return {args.resolution0, args.resolution1};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_reflection_rays_cuda(
    at::Tensor tx_positions, at::Tensor bounds, bool use_bounds, int64_t sample_count, int64_t seed) {
    const int64_t tx_count = tx_positions.size(0);
    auto options = tx_positions.options();
    auto ray_o = at::empty({tx_count * sample_count, 3}, options);
    auto ray_d = at::empty({tx_count * sample_count, 3}, options);
    auto active = at::empty({tx_count * sample_count}, options.dtype(at::kBool));
    auto solid_angle = at::empty({tx_count}, options);
    const int64_t total = tx_count * sample_count;
    if (total > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        reflection_rays_kernel<<<reflection_blocks(total), kReflectionBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(), use_bounds ? bounds.data_ptr<float>() : nullptr,
            tx_count, sample_count, seed, ray_o.data_ptr<float>(), ray_d.data_ptr<float>(),
            active.data_ptr<bool>(), solid_angle.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ray_o, ray_d, active, solid_angle};
}

at::Tensor channel_mc_scene_bounds_cuda(at::Tensor vertices) {
    auto bounds = at::empty({6}, vertices.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    scene_bounds_kernel<<<1, kReflectionBlockSize, 0, stream>>>(
        vertices.data_ptr<float>(), vertices.size(0), bounds.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bounds;
}

void channel_mc_reflection_final_segment_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor material_valid,
    const ReflectionReceiverArgs& receivers, double wavelength, bool first_block,
    at::Tensor final_o, at::Tensor final_d, at::Tensor final_tmax, at::Tensor final_offset,
    at::Tensor final_active) {
    const int64_t ray_count = ray_o.size(0);
    if (ray_count == 0) return;
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    bool first = first_block;
    for_each_target_chunk(receivers, ray_count, 1, [&](ReceiverTargets rx, int64_t) {
        reflection_final_segment_kernel<<<reflection_blocks(ray_count), kReflectionBlockSize, 0, stream>>>(
            ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
            trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
            material_valid.data_ptr<bool>(), ray_count, trace_depth, rx,
            static_cast<float>(wavelength), first, final_o.data_ptr<float>(),
            final_d.data_ptr<float>(), final_tmax.data_ptr<float>(),
            final_offset.data_ptr<float>(), final_active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        first = false;
    });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_mc_reflection_append_final_cuda(
    at::Tensor trace_valid, at::Tensor trace_t, at::Tensor trace_prim, at::Tensor final_valid,
    at::Tensor final_t, at::Tensor final_prim, at::Tensor final_offset) {
    const int64_t ray_count = trace_valid.size(0);
    const int64_t depth = trace_valid.size(1);
    auto out_valid = at::empty({ray_count, depth + 1}, trace_valid.options());
    auto out_t = at::empty({ray_count, depth + 1}, trace_t.options());
    auto out_prim = at::empty({ray_count, depth + 1}, trace_prim.options());
    if (ray_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(trace_valid.get_device()).stream();
        reflection_append_final_kernel<<<reflection_blocks(ray_count), kReflectionBlockSize, 0, stream>>>(
            trace_valid.data_ptr<bool>(), trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(),
            final_valid.data_ptr<bool>(), final_t.data_ptr<float>(), final_prim.data_ptr<int>(),
            final_offset.data_ptr<float>(), ray_count, static_cast<int>(depth),
            out_valid.data_ptr<bool>(), out_t.data_ptr<float>(), out_prim.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_valid, out_t, out_prim};
}

// Layer view of an MC reflection request: [face_material (F), layer_offset, layer_count,
// layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r].
layered::LayerTables mc_layer_tables(const std::vector<at::Tensor>& layers, int64_t face_count) {
    TORCH_CHECK(layers.size() == 7, "reflection layers must hold face_material and six CSR layer tables");
    for (size_t i = 0; i < layers.size(); ++i) {
        TORCH_CHECK(layers[i].is_cuda() && layers[i].is_contiguous() && layers[i].dim() == 1,
                    "reflection layer tensors must be contiguous one-dimensional CUDA tensors");
        TORCH_CHECK(layers[i].scalar_type() == (i < 3 ? at::kInt : at::kFloat),
                    "reflection layer tensor has the wrong dtype");
    }
    TORCH_CHECK(layers[0].size(0) == face_count, "face_material must cover every face");
    TORCH_CHECK(layers[2].size(0) == layers[1].size(0), "layer_count must match layer_offset");
    for (size_t i = 4; i < 7; ++i)
        TORCH_CHECK(layers[i].size(0) == layers[3].size(0), "layer tables must share one length");
    layered::LayerTables tables;
    tables.offset = layers[1].data_ptr<int>();
    tables.count = layers[2].data_ptr<int>();
    tables.thickness_m = layers[3].data_ptr<float>();
    tables.eps_r = layers[4].data_ptr<float>();
    tables.sigma_e = layers[5].data_ptr<float>();
    tables.mu_r = layers[6].data_ptr<float>();
    tables.material_count = layers[2].size(0);
    return tables;
}

at::Tensor channel_mc_slab_reflection_accumulate_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness, std::vector<at::Tensor> layers,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, at::Tensor tx_pol) {
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    const layered::LayerTables tables = mc_layer_tables(layers, eta_r.numel());
    auto output = reflection_zero_filled(receiver_output_shape(receivers), ray_o.options());
    if (ray_count == 0 || contribution_depth <= 0) return output;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    for_each_target_chunk(receivers, ray_count, contribution_depth, [&](ReceiverTargets rx, int64_t offset) {
        slab_reflection_accumulate_kernel<<<reflection_blocks(ray_count), kReflectionBlockSize, 0, stream>>>(
            ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
            trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
            eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(), thickness.data_ptr<float>(),
            layers[0].data_ptr<int>(), tables,
            output.data_ptr<float>() + offset, ray_count, trace_depth,
            static_cast<int>(contribution_depth), rx, static_cast<float>(wavelength),
            solid_angle.data_ptr<float>(), tx_pol.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}

// Single source of truth for the reflection AD depth cap: the AD companions
// below stage per-bounce state in kReflectionAdMaxDepth-sized register
// arrays, so the solver reads this cap to reject an over-deep AD
// configuration before any forward launch.
int64_t channel_mc_reflection_ad_max_depth_cuda() {
    return kReflectionAdMaxDepth;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_slab_reflection_accumulate_backward_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness, std::vector<at::Tensor> layers,
    at::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, double wavelength_dfreq, at::Tensor tx_pol) {
    TORCH_CHECK(
        contribution_depth <= kReflectionAdMaxDepth,
        "reflection AD companions support contribution_depth <= ",
        kReflectionAdMaxDepth);
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat, "grad_output must be float32");
    const std::vector<int64_t> shape = receiver_output_shape(receivers);
    TORCH_CHECK(grad_output.sizes() == at::IntArrayRef(shape),
                "grad_output must match the reflection receiver output");
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    const layered::LayerTables tables = mc_layer_tables(layers, eta_r.numel());
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
    // Per-ray frequency cotangents, summed below in a fixed order.
    at::Tensor ray_frequency = need_frequency ? at::empty({ray_count}, face_options) : at::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    // Point receivers read grad_output[0, target]; grids read grad_output[i1, i0].
    const bool points = point_mode(receivers);
    const int64_t stride0 = points ? grad_output.stride(1) : grad_output.stride(0);
    const int64_t stride1 = points ? 0 : grad_output.stride(1);
    for_each_target_chunk(receivers, ray_count, contribution_depth, [&](ReceiverTargets rx, int64_t offset) {
        // Each point chunk adds its frequency share to grad_frequency after its launch.
        slab_reflection_accumulate_backward_kernel<<<reflection_blocks(ray_count), kReflectionBlockSize, 0, stream>>>(
            ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
            trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
            eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(), thickness.data_ptr<float>(),
            layers[0].data_ptr<int>(), tables,
            grad_output.data_ptr<float>() + offset * stride0,
            need_materials ? grad_eta_r.data_ptr<float>() : nullptr,
            need_materials ? grad_sigma.data_ptr<float>() : nullptr,
            need_materials ? grad_gain.data_ptr<float>() : nullptr,
            need_materials ? grad_thickness.data_ptr<float>() : nullptr,
            need_frequency ? ray_frequency.data_ptr<float>() : nullptr,
            ray_count, trace_depth, static_cast<int>(contribution_depth), rx,
            static_cast<float>(wavelength), solid_angle.data_ptr<float>(),
            static_cast<float>(wavelength_dfreq), stride0, stride1, tx_pol.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_frequency)
            channel::ordered_sum::add_sum<float>(
                ray_frequency.data_ptr<float>(), ray_count, grad_frequency.data_ptr<float>(),
                face_options, stream);
    });
    return {grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_frequency};
}

at::Tensor channel_mc_slab_reflection_accumulate_jvp_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness, std::vector<at::Tensor> layers,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, double wavelength_tangent, at::Tensor tx_pol) {
    TORCH_CHECK(
        contribution_depth <= kReflectionAdMaxDepth,
        "reflection AD companions support contribution_depth <= ",
        kReflectionAdMaxDepth);
    const int64_t ray_count = ray_o.size(0);
    const int trace_depth = static_cast<int>(trace_valid.size(1));
    const layered::LayerTables tables = mc_layer_tables(layers, eta_r.numel());
    auto output_tangent = reflection_zero_filled(receiver_output_shape(receivers), ray_o.options());
    if (ray_count == 0 || contribution_depth <= 0) return output_tangent;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(ray_o.get_device()).stream();
    for_each_target_chunk(receivers, ray_count, contribution_depth, [&](ReceiverTargets rx, int64_t offset) {
        slab_reflection_accumulate_jvp_kernel<<<reflection_blocks(ray_count), kReflectionBlockSize, 0, stream>>>(
            ray_o.data_ptr<float>(), ray_d.data_ptr<float>(), trace_valid.data_ptr<bool>(),
            trace_t.data_ptr<float>(), trace_prim.data_ptr<int>(), face_normals.data_ptr<float>(),
            eta_r.data_ptr<float>(), sigma.data_ptr<float>(), gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(), thickness.data_ptr<float>(),
            layers[0].data_ptr<int>(), tables,
            has_tangent_eta_r ? tangent_eta_r.data_ptr<float>() : nullptr,
            has_tangent_sigma ? tangent_sigma.data_ptr<float>() : nullptr,
            has_tangent_gain ? tangent_gain.data_ptr<float>() : nullptr,
            has_tangent_thickness ? tangent_thickness.data_ptr<float>() : nullptr,
            output_tangent.data_ptr<float>() + offset,
            ray_count, trace_depth, static_cast<int>(contribution_depth), rx,
            static_cast<float>(wavelength), solid_angle.data_ptr<float>(),
            static_cast<float>(wavelength_tangent), tx_pol.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output_tangent;
}

#undef CHANNEL_REFLECTION_LAUNCH_INPUT_PREFIX
#undef CHANNEL_REFLECTION_PREPARE_LAUNCH_INPUTS
