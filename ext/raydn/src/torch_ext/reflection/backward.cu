#include <raydn/scene/geometry_kernels.h>
#include <raydn/reflection/kernels.h>
#include <raydn/common/math.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <vector>

namespace raydn {

namespace {

const bool *optional_bool_ptr(const at::Tensor &active) {
    if (!active.defined() || active.numel() == 0)
        return nullptr;
    return active.data_ptr<bool>();
}

void zero_float_tensor_async(const at::Tensor &tensor, cudaStream_t stream) {
    if (tensor.defined() && tensor.numel() > 0) {
        cudaMemsetAsync(tensor.data_ptr<float>(), 0, static_cast<size_t>(tensor.numel()) * sizeof(float), stream);
    }
}

int64_t optional_stride(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 || tensor->dim() <= dim)
        return 0;
    return tensor->stride(dim);
}

__device__ float read_scalar_or_zero(const float *base, int64_t index, int64_t stride0) {
    return base == nullptr ? 0.f : base[index * stride0];
}

__device__ float3 read_vec3_or_zero(const float *base, int64_t index, int64_t stride0, int64_t stride1) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[index * stride0 + 0 * stride1],
                                         base[index * stride0 + 1 * stride1],
                                         base[index * stride0 + 2 * stride1]);
}

__device__ void write_vec3_or_skip(float *base, int64_t index, float3 value) {
    if (base == nullptr)
        return;
    base[index * 3 + 0] = value.x;
    base[index * 3 + 1] = value.y;
    base[index * 3 + 2] = value.z;
}

__device__ float det3(float3 c0, float3 c1, float3 c2) {
    return dot3(c0, cross3(c1, c2));
}

__device__ float3 solve_columns(float3 c0, float3 c1, float3 c2, float3 rhs) {
    float determinant = det3(c0, c1, c2);
    if (fabsf(determinant) < 1e-12f)
        determinant = copysignf(1e-12f, determinant == 0.f ? 1.f : determinant);
    const float inv_det = 1.f / determinant;
    return make_float3(
        det3(rhs, c1, c2) * inv_det,
        det3(c0, rhs, c2) * inv_det,
        det3(c0, c1, rhs) * inv_det);
}

__device__ float3 solve_transpose_columns(float3 c0, float3 c1, float3 c2, float3 rhs) {
    const float3 r0 = make_float3(c0.x, c1.x, c2.x);
    const float3 r1 = make_float3(c0.y, c1.y, c2.y);
    const float3 r2 = make_float3(c0.z, c1.z, c2.z);
    return solve_columns(r0, r1, r2, rhs);
}

__device__ float3 bary3_from_tape(const float *tape_bary, int tape_bary_width, int64_t ray_idx) {
    if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        return make_float3(1.f - u - v, u, v);
    }
    return make_f3(tape_bary + ray_idx * 3);
}

__device__ int64_t ray_bounce_index(int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return ray_idx * max_bounces + bounce;
}

__device__ int64_t ray_bounce_vec3_index(int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return (ray_idx * max_bounces + bounce) * 3;
}

__device__ int64_t state_vec3_index(int64_t bounce, int64_t ray_idx, int64_t ray_count) {
    return (bounce * ray_count + ray_idx) * 3;
}

__device__ float3 read_ray_vec3(const float *base, int64_t ray_idx) {
    return make_f3(base + ray_idx * 3);
}

__device__ float3 read_ray_bounce_vec3(const float *base, int64_t ray_idx, int64_t bounce, int64_t max_bounces) {
    return make_f3(base + ray_bounce_vec3_index(ray_idx, bounce, max_bounces));
}

__device__ void write_ray_vec3(float *base, int64_t ray_idx, float3 value) {
    base[ray_idx * 3 + 0] = value.x;
    base[ray_idx * 3 + 1] = value.y;
    base[ray_idx * 3 + 2] = value.z;
}

__device__ void write_ray_bounce_vec3(float *base, int64_t ray_idx, int64_t bounce, int64_t max_bounces, float3 value) {
    const int64_t idx = ray_bounce_vec3_index(ray_idx, bounce, max_bounces);
    base[idx + 0] = value.x;
    base[idx + 1] = value.y;
    base[idx + 2] = value.z;
}

__device__ float3 read_state_vec3(const float *base, int64_t bounce, int64_t ray_idx, int64_t ray_count) {
    return make_f3(base + state_vec3_index(bounce, ray_idx, ray_count));
}

__device__ void write_state_vec3(float *base, int64_t bounce, int64_t ray_idx, int64_t ray_count, float3 value) {
    const int64_t idx = state_vec3_index(bounce, ray_idx, ray_count);
    base[idx + 0] = value.x;
    base[idx + 1] = value.y;
    base[idx + 2] = value.z;
}

__device__ float read_grad_t_or_zero(
    const float *base,
    int grad_dim,
    int64_t stride0,
    int64_t stride1,
    int64_t ray_idx,
    int64_t bounce) {
    if (base == nullptr)
        return 0.f;
    if (grad_dim <= 1)
        return base[ray_idx * stride0];
    return base[ray_idx * stride0 + bounce * stride1];
}

__device__ float3 read_grad_image_or_zero(
    const float *base,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t ray_idx,
    int64_t bounce) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[ray_idx * stride0 + bounce * stride1 + 0 * stride2],
                                         base[ray_idx * stride0 + bounce * stride1 + 1 * stride2],
                                         base[ray_idx * stride0 + bounce * stride1 + 2 * stride2]);
}

__device__ float3 normal_from_edges(float3 e1, float3 e2, float *length_out) {
    const float3 q = cross3(e1, e2);
    const float length = sqrtf(fmaxf(dot3(q, q), 1e-20f));
    if (length_out != nullptr)
        *length_out = length;
    return mul3(1.f / length, q);
}

__device__ float3 normal_jvp(float3 e1, float3 e2, float3 de1, float3 de2) {
    float length = 0.f;
    const float3 n = normal_from_edges(e1, e2, &length);
    const float3 dq = add3(cross3(de1, e2), cross3(e1, de2));
    return mul3(1.f / length, sub3(dq, mul3(dot3(n, dq), n)));
}

__global__ void reflection_chain_state_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ image_sources,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ origins,
    float *__restrict__ directions,
    float *__restrict__ image_states) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    float3 origin = read_ray_vec3(ray_o, ray_idx);
    float3 direction = read_ray_vec3(ray_d, ray_idx);
    float3 image_state = origin;
    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        write_state_vec3(origins, bounce, ray_idx, ray_count, origin);
        write_state_vec3(directions, bounce, ray_idx, ray_count, direction);
        write_state_vec3(image_states, bounce, ray_idx, ray_count, image_state);
        if (bounce + 1 >= max_bounces)
            continue;
        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);
        const float dir_dot_n = dot3(direction, normal);
        const float3 next_direction = sub3(direction, mul3(2.f * dir_dot_n, normal));
        origin = add3(hit, mul3(static_cast<float>(kRayBias), next_direction));
        direction = next_direction;
        image_state = read_ray_bounce_vec3(image_sources, ray_idx, bounce, max_bounces);
    }
}

__global__ void reflection_chain_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ origins,
    const float *__restrict__ directions,
    const float *__restrict__ image_states,
    const float *__restrict__ grad_t,
    int grad_t_dim,
    int64_t grad_t_stride0,
    int64_t grad_t_stride1,
    const float *__restrict__ grad_image_sources,
    int64_t grad_image_stride0,
    int64_t grad_image_stride1,
    int64_t grad_image_stride2,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    float *__restrict__ grad_ray_tmax) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    write_ray_vec3(grad_ray_o, ray_idx, make_float3(0.f, 0.f, 0.f));
    write_ray_vec3(grad_ray_d, ray_idx, make_float3(0.f, 0.f, 0.f));
    grad_ray_tmax[ray_idx] = 0.f;

    float3 grad_origin_next = make_float3(0.f, 0.f, 0.f);
    float3 grad_direction_next = make_float3(0.f, 0.f, 0.f);
    float3 grad_image_next = make_float3(0.f, 0.f, 0.f);

    for (int64_t bounce = max_bounces - 1; bounce >= 0; --bounce) {
        const int64_t rb = ray_bounce_index(ray_idx, bounce, max_bounces);
        const int prim_id = tape_prim_id[rb];
        const bool active_b = prim_id >= 0 && (active == nullptr || active[ray_idx]);
        if (!active_b) {
            grad_origin_next = make_float3(0.f, 0.f, 0.f);
            grad_direction_next = make_float3(0.f, 0.f, 0.f);
            grad_image_next = make_float3(0.f, 0.f, 0.f);
            continue;
        }

        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);
        const float3 direction = read_state_vec3(directions, bounce, ray_idx, ray_count);
        const float3 origin = read_state_vec3(origins, bounce, ray_idx, ray_count);
        const float3 image_before = read_state_vec3(image_states, bounce, ray_idx, ray_count);

        const float3 grad_image_out = add3(
            read_grad_image_or_zero(
                grad_image_sources,
                grad_image_stride0,
                grad_image_stride1,
                grad_image_stride2,
                ray_idx,
                bounce),
            grad_image_next);
        const float3 image_delta = sub3(image_before, hit);
        const float image_dist = dot3(image_delta, normal);
        const float image_gdotn = dot3(grad_image_out, normal);
        const float3 grad_image_prev = sub3(grad_image_out, mul3(2.f * image_gdotn, normal));

        float3 grad_p = mul3(2.f * image_gdotn, normal);
        float3 grad_signed_n =
            mul3(-2.f, add3(mul3(image_gdotn, image_delta), mul3(image_dist, grad_image_out)));

        grad_p = add3(grad_p, grad_origin_next);
        const float3 grad_reflected =
            add3(grad_direction_next, mul3(static_cast<float>(kRayBias), grad_origin_next));
        const float dir_dot_n = dot3(direction, normal);
        const float refl_gdotn = dot3(grad_reflected, normal);
        const float3 grad_direction_current =
            sub3(grad_reflected, mul3(2.f * refl_gdotn, normal));
        grad_signed_n = sub3(
            grad_signed_n,
            mul3(2.f, add3(mul3(refl_gdotn, direction), mul3(dir_dot_n, grad_reflected))));

        const int i0 = faces[prim_id * 3 + 0];
        const int i1 = faces[prim_id * 3 + 1];
        const int i2 = faces[prim_id * 3 + 2];
        const float3 v0 = make_f3(vertices + i0 * 3);
        const float3 v1 = make_f3(vertices + i1 * 3);
        const float3 v2 = make_f3(vertices + i2 * 3);
        const float3 e1 = sub3(v1, v0);
        const float3 e2 = sub3(v2, v0);
        const float3 raw_normal = normal_from_edges(e1, e2, nullptr);
        const float sign = dot3(raw_normal, normal) >= 0.f ? 1.f : -1.f;
        const float3 grad_raw_n = mul3(sign, grad_signed_n);

        float3 g_vertices0 = make_float3(0.f, 0.f, 0.f);
        float3 g_vertices1 = make_float3(0.f, 0.f, 0.f);
        float3 g_vertices2 = make_float3(0.f, 0.f, 0.f);
        const float normal_length = sqrtf(fmaxf(dot3(cross3(e1, e2), cross3(e1, e2)), 1e-20f));
        const float3 gq = mul3(
            1.f / normal_length,
            sub3(grad_raw_n, mul3(dot3(raw_normal, grad_raw_n), raw_normal)));
        const float3 ge1_normal = cross3(e2, gq);
        const float3 ge2_normal = cross3(gq, e1);
        g_vertices0 = sub3(g_vertices0, add3(ge1_normal, ge2_normal));
        g_vertices1 = add3(g_vertices1, ge1_normal);
        g_vertices2 = add3(g_vertices2, ge2_normal);

        const float3 d = direction;
        const float3 c0 = mul3(-1.f, d);
        const float3 bary = bary3_from_tape(tape_bary + rb * tape_bary_width, tape_bary_width, 0);
        const float gt = read_grad_t_or_zero(
            grad_t,
            grad_t_dim,
            grad_t_stride0,
            grad_t_stride1,
            ray_idx,
            bounce);
        const float t_bar_from_p = dot3(grad_p, d);
        float3 grad_ray_o_hit = grad_p;
        const float3 gy = make_float3(gt + t_bar_from_p, 0.f, 0.f);
        const float3 lambda = solve_transpose_columns(c0, e1, e2, gy);
        grad_ray_o_hit = add3(grad_ray_o_hit, lambda);
        const float solved_t = solve_columns(c0, e1, e2, sub3(origin, v0)).x;
        const float3 grad_ray_d_hit = mul3(solved_t, add3(lambda, grad_p));

        g_vertices0 = sub3(g_vertices0, mul3(bary.x, lambda));
        g_vertices1 = sub3(g_vertices1, mul3(bary.y, lambda));
        g_vertices2 = sub3(g_vertices2, mul3(bary.z, lambda));
        atomic_add3(grad_vertices, i0, g_vertices0);
        atomic_add3(grad_vertices, i1, g_vertices1);
        atomic_add3(grad_vertices, i2, g_vertices2);

        grad_origin_next = grad_ray_o_hit;
        grad_direction_next = add3(grad_ray_d_hit, grad_direction_current);
        grad_image_next = grad_image_prev;
    }

    write_ray_vec3(grad_ray_o, ray_idx, add3(grad_origin_next, grad_image_next));
    write_ray_vec3(grad_ray_d, ray_idx, grad_direction_next);
}

__global__ void reflection_chain_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_hit_points,
    const float *__restrict__ tape_normals,
    const float *__restrict__ tangent_vertices,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    const float *__restrict__ tangent_ray_o,
    int64_t tangent_ray_o_stride0,
    int64_t tangent_ray_o_stride1,
    const float *__restrict__ tangent_ray_d,
    int64_t tangent_ray_d_stride0,
    int64_t tangent_ray_d_stride1,
    const float *__restrict__ image_sources,
    int64_t ray_count,
    int64_t max_bounces,
    float *__restrict__ tangent_t,
    float *__restrict__ tangent_image_sources) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    float3 origin = read_ray_vec3(ray_o, ray_idx);
    float3 direction = read_ray_vec3(ray_d, ray_idx);
    float3 tangent_origin =
        read_vec3_or_zero(tangent_ray_o, ray_idx, tangent_ray_o_stride0, tangent_ray_o_stride1);
    float3 tangent_direction =
        read_vec3_or_zero(tangent_ray_d, ray_idx, tangent_ray_d_stride0, tangent_ray_d_stride1);
    float3 image_state = origin;
    float3 tangent_image_state = tangent_origin;

    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        const int64_t rb = ray_bounce_index(ray_idx, bounce, max_bounces);
        const int prim_id = tape_prim_id[rb];
        const bool active_b = prim_id >= 0 && (active == nullptr || active[ray_idx]);
        const float3 normal = read_ray_bounce_vec3(tape_normals, ray_idx, bounce, max_bounces);
        const float3 hit = read_ray_bounce_vec3(tape_hit_points, ray_idx, bounce, max_bounces);

        float tangent_hit_t = 0.f;
        float3 tangent_hit = make_float3(0.f, 0.f, 0.f);
        float3 tangent_normal = make_float3(0.f, 0.f, 0.f);
        if (active_b) {
            const int i0 = faces[prim_id * 3 + 0];
            const int i1 = faces[prim_id * 3 + 1];
            const int i2 = faces[prim_id * 3 + 2];
            const float3 v0 = make_f3(vertices + i0 * 3);
            const float3 v1 = make_f3(vertices + i1 * 3);
            const float3 v2 = make_f3(vertices + i2 * 3);
            const float3 dv0 =
                read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 dv1 =
                read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 dv2 =
                read_vec3_or_zero(tangent_vertices, i2, tangent_vertices_stride0, tangent_vertices_stride1);
            const float3 e1 = sub3(v1, v0);
            const float3 e2 = sub3(v2, v0);
            const float3 de1 = sub3(dv1, dv0);
            const float3 de2 = sub3(dv2, dv0);
            const float3 bary = bary3_from_tape(tape_bary + rb * tape_bary_width, tape_bary_width, 0);
            const float solved_t = solve_columns(
                                      mul3(-1.f, direction),
                                      e1,
                                      e2,
                                      sub3(origin, v0))
                                      .x;
            const float3 vertex_tangent =
                add3(add3(mul3(bary.x, dv0), mul3(bary.y, dv1)), mul3(bary.z, dv2));
            const float3 rhs = sub3(
                add3(tangent_origin, mul3(solved_t, tangent_direction)),
                vertex_tangent);
            const float3 dy = solve_columns(mul3(-1.f, direction), e1, e2, rhs);
            tangent_hit_t = dy.x;
            tangent_hit = add3(tangent_origin, add3(mul3(dy.x, direction), mul3(solved_t, tangent_direction)));
            const float3 raw_normal = normal_from_edges(e1, e2, nullptr);
            const float sign = dot3(raw_normal, normal) >= 0.f ? 1.f : -1.f;
            tangent_normal = mul3(sign, normal_jvp(e1, e2, de1, de2));
        }
        tangent_t[ray_bounce_index(ray_idx, bounce, max_bounces)] = active_b ? tangent_hit_t : 0.f;

        const float3 image_delta = sub3(image_state, hit);
        const float3 tangent_image_delta = sub3(tangent_image_state, tangent_hit);
        const float image_dist = dot3(image_delta, normal);
        const float tangent_image_dist =
            dot3(tangent_image_delta, normal) + dot3(image_delta, tangent_normal);
        const float3 next_image_state = sub3(image_state, mul3(2.f * image_dist, normal));
        float3 next_tangent_image_state =
            sub3(tangent_image_state,
                 mul3(2.f, add3(mul3(tangent_image_dist, normal), mul3(image_dist, tangent_normal))));
        if (!active_b) {
            next_tangent_image_state = make_float3(0.f, 0.f, 0.f);
        }
        write_ray_bounce_vec3(
            tangent_image_sources,
            ray_idx,
            bounce,
            max_bounces,
            next_tangent_image_state);

        const float dir_dot_n = dot3(direction, normal);
        const float tangent_dir_dot_n =
            dot3(tangent_direction, normal) + dot3(direction, tangent_normal);
        const float3 next_direction = sub3(direction, mul3(2.f * dir_dot_n, normal));
        float3 next_tangent_direction =
            sub3(tangent_direction,
                 mul3(2.f, add3(mul3(tangent_dir_dot_n, normal), mul3(dir_dot_n, tangent_normal))));
        const float3 next_origin = add3(hit, mul3(static_cast<float>(kRayBias), next_direction));
        float3 next_tangent_origin =
            add3(tangent_hit, mul3(static_cast<float>(kRayBias), next_tangent_direction));
        if (!active_b) {
            next_tangent_origin = make_float3(0.f, 0.f, 0.f);
            next_tangent_direction = make_float3(0.f, 0.f, 0.f);
        }

        origin = next_origin;
        direction = next_direction;
        tangent_origin = next_tangent_origin;
        tangent_direction = next_tangent_direction;
        image_state = next_image_state;
        tangent_image_state = next_tangent_image_state;
        if (bounce + 1 < max_bounces) {
            image_state = read_ray_bounce_vec3(image_sources, ray_idx, bounce, max_bounces);
        }
    }
}

__global__ void refl_epc_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_t,
    const float *__restrict__ grad_field_real,
    const float *__restrict__ grad_field_imag,
    const float *__restrict__ grad_path_length,
    int64_t grad_field_real_stride0,
    int64_t grad_field_imag_stride0,
    int64_t grad_path_length_stride0,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_source,
    float *__restrict__ grad_receiver) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    write_vec3_or_skip(grad_source, ray_idx, make_float3(0.f, 0.f, 0.f));
    write_vec3_or_skip(grad_receiver, ray_idx, make_float3(0.f, 0.f, 0.f));
    if (active != nullptr && !active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;

    const float t = tape_t[ray_idx];
    const float inv_denom = 1.f / (1.f + t);
    const float s = sinf(t);
    const float c = cosf(t);
    const float real_dt = -s * inv_denom - c * inv_denom * inv_denom;
    const float imag_dt = c * inv_denom - s * inv_denom * inv_denom;
    const float gt =
        read_scalar_or_zero(grad_path_length, ray_idx, grad_path_length_stride0) +
        read_scalar_or_zero(grad_field_real, ray_idx, grad_field_real_stride0) * real_dt +
        read_scalar_or_zero(grad_field_imag, ray_idx, grad_field_imag_stride0) * imag_dt;
    if (gt == 0.f)
        return;

    const int i0 = faces[prim_id * 3 + 0];
    const int i1 = faces[prim_id * 3 + 1];
    const int i2 = faces[prim_id * 3 + 2];
    const float3 v0 = make_f3(vertices + i0 * 3);
    const float3 v1 = make_f3(vertices + i1 * 3);
    const float3 v2 = make_f3(vertices + i2 * 3);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 o = make_f3(source + ray_idx * 3);
    const float3 r = make_f3(receiver + ray_idx * 3);
    const float3 d = sub3(r, o);
    const float3 c0 = mul3(-1.f, d);
    const float3 lambda = solve_transpose_columns(c0, e1, e2, make_float3(gt, 0.f, 0.f));

    if (grad_source != nullptr || grad_receiver != nullptr) {
        const float solved_t = solve_columns(c0, e1, e2, sub3(o, v0)).x;
        const float3 grad_ray_d = mul3(solved_t, lambda);
        write_vec3_or_skip(grad_source, ray_idx, sub3(lambda, grad_ray_d));
        write_vec3_or_skip(grad_receiver, ray_idx, grad_ray_d);
    }

    if (grad_vertices == nullptr)
        return;
    const float3 bary = bary3_from_tape(tape_bary, tape_bary_width, ray_idx);
    atomic_add3(grad_vertices, i0, mul3(-bary.x, lambda));
    atomic_add3(grad_vertices, i1, mul3(-bary.y, lambda));
    atomic_add3(grad_vertices, i2, mul3(-bary.z, lambda));
}

__global__ void refl_epc_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tape_t,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_source,
    const float *__restrict__ tangent_receiver,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_source_stride0,
    int64_t tangent_source_stride1,
    int64_t tangent_receiver_stride0,
    int64_t tangent_receiver_stride1,
    int64_t ray_count,
    float *__restrict__ tangent_field_real,
    float *__restrict__ tangent_field_imag,
    float *__restrict__ tangent_path_length) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    tangent_field_real[ray_idx] = 0.f;
    tangent_field_imag[ray_idx] = 0.f;
    tangent_path_length[ray_idx] = 0.f;
    if (active != nullptr && !active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;

    const int i0 = faces[prim_id * 3 + 0];
    const int i1 = faces[prim_id * 3 + 1];
    const int i2 = faces[prim_id * 3 + 2];
    const float3 v0 = make_f3(vertices + i0 * 3);
    const float3 v1 = make_f3(vertices + i1 * 3);
    const float3 v2 = make_f3(vertices + i2 * 3);
    const float3 dv0 = read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dv1 = read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dv2 = read_vec3_or_zero(tangent_vertices, i2, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 o = make_f3(source + ray_idx * 3);
    const float3 r = make_f3(receiver + ray_idx * 3);
    const float3 d = sub3(r, o);
    const float3 do_t = read_vec3_or_zero(tangent_source, ray_idx, tangent_source_stride0, tangent_source_stride1);
    const float3 dr_t = read_vec3_or_zero(tangent_receiver, ray_idx, tangent_receiver_stride0, tangent_receiver_stride1);
    const float3 dd_t = sub3(dr_t, do_t);
    const float3 bary = bary3_from_tape(tape_bary, tape_bary_width, ray_idx);
    const float3 c0 = mul3(-1.f, d);
    const float solved_t = solve_columns(c0, e1, e2, sub3(o, v0)).x;
    const float3 vertex_tangent =
        add3(add3(mul3(bary.x, dv0), mul3(bary.y, dv1)), mul3(bary.z, dv2));
    const float3 rhs = sub3(add3(do_t, mul3(solved_t, dd_t)), vertex_tangent);
    const float tangent_t = solve_columns(c0, e1, e2, rhs).x;

    const float t = tape_t[ray_idx];
    const float inv_denom = 1.f / (1.f + t);
    const float s = sinf(t);
    const float c = cosf(t);
    const float real_dt = -s * inv_denom - c * inv_denom * inv_denom;
    const float imag_dt = c * inv_denom - s * inv_denom * inv_denom;
    tangent_field_real[ray_idx] = real_dt * tangent_t;
    tangent_field_imag[ray_idx] = imag_dt * tangent_t;
    tangent_path_length[ray_idx] = tangent_t;
}

} // namespace

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t) {
    (void)ray_tmax;
    at::Tensor grad_t_flat = grad_t.dim() == 1 ? grad_t : grad_t.select(1, 0);
    IntersectBackwardOutputs hit_grad = intersect_backward_t_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t_flat,
        grad_t_flat.stride(0),
        true,
        true,
        true,
        true);
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o,
        hit_grad.grad_ray_d,
        hit_grad.grad_ray_tmax,
    };
}

ReflectionBackwardOutputs reflection_chain_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor &image_sources,
    const at::Tensor *grad_t,
    const at::Tensor *grad_image_sources) {
    (void)ray_tmax;
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);
    if (max_bounces == 1 && grad_t != nullptr && grad_t->numel() != 0 && grad_image_sources == nullptr) {
        at::Tensor grad_t_flat = grad_t->dim() == 1 ? *grad_t : grad_t->select(1, 0);
        IntersectBackwardOutputs hit_grad = intersect_backward_t_cuda(
            vertices,
            faces,
            ray_o,
            ray_d,
            active,
            tape_prim_id.select(1, 0),
            tape_barycentric.select(1, 0),
            grad_t_flat,
            grad_t_flat.stride(0),
            true,
            true,
            true,
            true);
        return {
            hit_grad.grad_vertices,
            hit_grad.grad_ray_o,
            hit_grad.grad_ray_d,
            hit_grad.grad_ray_tmax,
        };
    }

    ReflectionBackwardOutputs out;
    out.grad_vertices = at::empty_like(vertices);
    out.grad_ray_o = at::empty_like(ray_o);
    out.grad_ray_d = at::empty_like(ray_d);
    out.grad_ray_tmax = at::empty({ray_count}, ray_o.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0) {
        return out;
    }

    at::Tensor origins = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    at::Tensor directions = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    at::Tensor image_states = at::empty({max_bounces, ray_count, 3}, ray_o.options());
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_chain_state_kernel<<<blocks, threads, 0, stream>>>(
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        image_sources.data_ptr<float>(),
        ray_count,
        max_bounces,
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        image_states.data_ptr<float>());
    reflection_chain_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(2)),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        image_states.data_ptr<float>(),
        grad_t == nullptr ? nullptr : grad_t->data_ptr<float>(),
        grad_t == nullptr ? 0 : static_cast<int>(grad_t->dim()),
        optional_stride(grad_t, 0),
        optional_stride(grad_t, 1),
        grad_image_sources == nullptr ? nullptr : grad_image_sources->data_ptr<float>(),
        optional_stride(grad_image_sources, 0),
        optional_stride(grad_image_sources, 1),
        optional_stride(grad_image_sources, 2),
        ray_count,
        max_bounces,
        out.grad_vertices.data_ptr<float>(),
        out.grad_ray_o.data_ptr<float>(),
        out.grad_ray_d.data_ptr<float>(),
        out.grad_ray_tmax.data_ptr<float>());
    return out;
}

ReflectionJvpOutputs reflection_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    IntersectJvpOutputs hit_jvp = intersect_jvp_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return {
        hit_jvp.tangent_t.reshape({ray_count, 1}),
        at::zeros_like(image_sources),
    };
}

ReflectionJvpOutputs reflection_chain_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);
    ReflectionJvpOutputs out;
    out.tangent_t = at::empty({ray_count, max_bounces}, ray_o.options());
    out.tangent_image_sources = at::empty_like(image_sources);
    if (ray_count == 0) {
        return out;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_chain_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(2)),
        tape_hit_points.data_ptr<float>(),
        tape_normals.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        tangent_ray_o == nullptr ? nullptr : tangent_ray_o->data_ptr<float>(),
        optional_stride(tangent_ray_o, 0),
        optional_stride(tangent_ray_o, 1),
        tangent_ray_d == nullptr ? nullptr : tangent_ray_d->data_ptr<float>(),
        optional_stride(tangent_ray_d, 0),
        optional_stride(tangent_ray_d, 1),
        image_sources.data_ptr<float>(),
        ray_count,
        max_bounces,
        out.tangent_t.data_ptr<float>(),
        out.tangent_image_sources.data_ptr<float>());
    return out;
}

ReflEpcBackwardOutputs refl_epc_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *grad_field_real,
    const at::Tensor *grad_field_imag,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    const int64_t ray_count = source.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    ReflEpcBackwardOutputs out;
    out.grad_vertices = need_grad_vertices ? at::empty_like(vertices) : at::empty({0, 3}, vertices.options());
    out.grad_source = need_grad_source ? at::empty_like(source) : at::empty({0, 3}, source.options());
    out.grad_receiver = need_grad_receiver ? at::empty_like(receiver) : at::empty({0, 3}, receiver.options());
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0 || (!need_grad_vertices && !need_grad_source && !need_grad_receiver)) {
        return out;
    }
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    refl_epc_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tape_t.data_ptr<float>(),
        grad_field_real == nullptr ? nullptr : grad_field_real->data_ptr<float>(),
        grad_field_imag == nullptr ? nullptr : grad_field_imag->data_ptr<float>(),
        grad_path_length == nullptr ? nullptr : grad_path_length->data_ptr<float>(),
        optional_stride(grad_field_real, 0),
        optional_stride(grad_field_imag, 0),
        optional_stride(grad_path_length, 0),
        ray_count,
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_source ? out.grad_source.data_ptr<float>() : nullptr,
        need_grad_receiver ? out.grad_receiver.data_ptr<float>() : nullptr);
    return out;
}

ReflEpcJvpOutputs refl_epc_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver) {
    const int64_t ray_count = source.size(0);
    ReflEpcJvpOutputs out;
    out.tangent_field_real = at::empty({ray_count}, source.options());
    out.tangent_field_imag = at::empty({ray_count}, source.options());
    out.tangent_path_length = at::empty({ray_count}, source.options());
    if (ray_count == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    refl_epc_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        optional_bool_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tape_t.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_source == nullptr ? nullptr : tangent_source->data_ptr<float>(),
        tangent_receiver == nullptr ? nullptr : tangent_receiver->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_source, 0),
        optional_stride(tangent_source, 1),
        optional_stride(tangent_receiver, 0),
        optional_stride(tangent_receiver, 1),
        ray_count,
        out.tangent_field_real.data_ptr<float>(),
        out.tangent_field_imag.data_ptr<float>(),
        out.tangent_path_length.data_ptr<float>());
    return out;
}

} // namespace raydn
