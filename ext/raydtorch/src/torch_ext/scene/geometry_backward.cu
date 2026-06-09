#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/common/math.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace raydtorch {

namespace {

__device__ void add_to3(float *base, int index, float3 value) {
    base[index * 3 + 0] += value.x;
    base[index * 3 + 1] += value.y;
    base[index * 3 + 2] += value.z;
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

__device__ float3 normal_from_edges(float3 e1, float3 e2, float *length_out) {
    const float3 q = cross3(e1, e2);
    float length = sqrtf(fmaxf(dot3(q, q), 1e-20f));
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

__global__ void intersect_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ grad_t,
    const float *__restrict__ grad_p,
    const float *__restrict__ grad_n,
    const float *__restrict__ grad_geo_n,
    const float *__restrict__ grad_uv,
    const float *__restrict__ grad_bary,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    float *__restrict__ grad_ray_tmax) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    grad_ray_tmax[ray_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        grad_ray_o[ray_idx * 3 + axis] = 0.f;
        grad_ray_d[ray_idx * 3 + axis] = 0.f;
    }
    if (!active[ray_idx])
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
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const float3 c0 = mul3(-1.f, d);
    float3 bary;
    if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        bary = make_float3(1.f - u - v, u, v);
    } else {
        bary = make_f3(tape_bary + ray_idx * 3);
    }

    float3 g_vertices0 = make_float3(0.f, 0.f, 0.f);
    float3 g_vertices1 = make_float3(0.f, 0.f, 0.f);
    float3 g_vertices2 = make_float3(0.f, 0.f, 0.f);

    const float3 gp = make_f3(grad_p + ray_idx * 3);
    const float3 gn = add3(
        make_f3(grad_n + ray_idx * 3),
        make_f3(grad_geo_n + ray_idx * 3));
    const float3 normal = normal_from_edges(e1, e2, nullptr);
    const float normal_length = sqrtf(fmaxf(dot3(cross3(e1, e2), cross3(e1, e2)), 1e-20f));
    const float3 gq = mul3(1.f / normal_length, sub3(gn, mul3(dot3(normal, gn), normal)));
    const float3 ge1_normal = cross3(e2, gq);
    const float3 ge2_normal = cross3(gq, e1);
    g_vertices0 = sub3(g_vertices0, add3(ge1_normal, ge2_normal));
    g_vertices1 = add3(g_vertices1, ge1_normal);
    g_vertices2 = add3(g_vertices2, ge2_normal);

    const float t_bar_from_p = dot3(gp, d);
    const float t = 0.f;
    (void)t;
    add_to3(grad_ray_o, ray_idx, gp);

    float gu = grad_uv[ray_idx * 2 + 0];
    float gv = grad_uv[ray_idx * 2 + 1];
    const float gb0 = grad_bary[ray_idx * 3 + 0];
    const float gb1 = grad_bary[ray_idx * 3 + 1];
    const float gb2 = grad_bary[ray_idx * 3 + 2];
    gu += -gb0 + gb1;
    gv += -gb0 + gb2;

    const float3 gy = make_float3(grad_t[ray_idx] + t_bar_from_p, gu, gv);
    const float3 lambda = solve_transpose_columns(c0, e1, e2, gy);

    add_to3(grad_ray_o, ray_idx, lambda);
    const float solved_t = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx * 3), v0)).x;
    add_to3(grad_ray_d, ray_idx, mul3(solved_t, add3(lambda, gp)));

    g_vertices0 = sub3(g_vertices0, mul3(bary.x, lambda));
    g_vertices1 = sub3(g_vertices1, mul3(bary.y, lambda));
    g_vertices2 = sub3(g_vertices2, mul3(bary.z, lambda));

    atomic_add3(grad_vertices, i0, g_vertices0);
    atomic_add3(grad_vertices, i1, g_vertices1);
    atomic_add3(grad_vertices, i2, g_vertices2);
}

__global__ void intersect_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_ray_o,
    const float *__restrict__ tangent_ray_d,
    int64_t ray_count,
    float *__restrict__ tangent_t,
    float *__restrict__ tangent_p,
    float *__restrict__ tangent_n,
    float *__restrict__ tangent_geo_n,
    float *__restrict__ tangent_uv,
    float *__restrict__ tangent_barycentric) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    for (int axis = 0; axis < 3; ++axis) {
        tangent_p[ray_idx * 3 + axis] = 0.f;
        tangent_n[ray_idx * 3 + axis] = 0.f;
        tangent_geo_n[ray_idx * 3 + axis] = 0.f;
        tangent_barycentric[ray_idx * 3 + axis] = 0.f;
    }
    tangent_t[ray_idx] = 0.f;
    tangent_uv[ray_idx * 2 + 0] = 0.f;
    tangent_uv[ray_idx * 2 + 1] = 0.f;
    if (!active[ray_idx])
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
    const float3 dv0 = make_f3(tangent_vertices + i0 * 3);
    const float3 dv1 = make_f3(tangent_vertices + i1 * 3);
    const float3 dv2 = make_f3(tangent_vertices + i2 * 3);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 de1 = sub3(dv1, dv0);
    const float3 de2 = sub3(dv2, dv0);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const float3 dd = make_f3(tangent_ray_d + ray_idx * 3);
    const float3 dorigin = make_f3(tangent_ray_o + ray_idx * 3);
    float3 bary;
    if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        bary = make_float3(1.f - u - v, u, v);
    } else {
        bary = make_f3(tape_bary + ray_idx * 3);
    }
    const float solved_t = solve_columns(
                               mul3(-1.f, d),
                               e1,
                               e2,
                               sub3(make_f3(ray_o + ray_idx * 3), v0))
                               .x;

    const float3 rhs = sub3(
        add3(dorigin, mul3(solved_t, dd)),
        add3(add3(mul3(bary.x, dv0), mul3(bary.y, dv1)), mul3(bary.z, dv2)));
    const float3 dy = solve_columns(mul3(-1.f, d), e1, e2, rhs);
    const float dt = dy.x;
    const float du = dy.y;
    const float dv = dy.z;
    const float3 dp = add3(dorigin, add3(mul3(dt, d), mul3(solved_t, dd)));
    const float3 dn = normal_jvp(e1, e2, de1, de2);

    tangent_t[ray_idx] = dt;
    tangent_uv[ray_idx * 2 + 0] = du;
    tangent_uv[ray_idx * 2 + 1] = dv;
    tangent_barycentric[ray_idx * 3 + 0] = -du - dv;
    tangent_barycentric[ray_idx * 3 + 1] = du;
    tangent_barycentric[ray_idx * 3 + 2] = dv;
    tangent_p[ray_idx * 3 + 0] = dp.x;
    tangent_p[ray_idx * 3 + 1] = dp.y;
    tangent_p[ray_idx * 3 + 2] = dp.z;
    tangent_n[ray_idx * 3 + 0] = dn.x;
    tangent_n[ray_idx * 3 + 1] = dn.y;
    tangent_n[ray_idx * 3 + 2] = dn.z;
    tangent_geo_n[ray_idx * 3 + 0] = dn.x;
    tangent_geo_n[ray_idx * 3 + 1] = dn.y;
    tangent_geo_n[ray_idx * 3 + 2] = dn.z;
}

} // namespace

IntersectBackwardOutputs intersect_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t,
    const at::Tensor &grad_p,
    const at::Tensor &grad_n,
    const at::Tensor &grad_geo_n,
    const at::Tensor &grad_uv,
    const at::Tensor &grad_barycentric) {
    (void)ray_tmax;
    const int64_t ray_count = ray_d.size(0);
    IntersectBackwardOutputs out;
    out.grad_vertices = at::zeros_like(vertices);
    out.grad_ray_o = at::zeros_like(ray_d);
    out.grad_ray_d = at::zeros_like(ray_d);
    out.grad_ray_tmax = at::zeros({ray_count}, ray_d.options());

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    intersect_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        active.data_ptr<bool>(),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        grad_t.data_ptr<float>(),
        grad_p.data_ptr<float>(),
        grad_n.data_ptr<float>(),
        grad_geo_n.data_ptr<float>(),
        grad_uv.data_ptr<float>(),
        grad_barycentric.data_ptr<float>(),
        ray_count,
        out.grad_vertices.data_ptr<float>(),
        out.grad_ray_o.data_ptr<float>(),
        out.grad_ray_d.data_ptr<float>(),
        out.grad_ray_tmax.data_ptr<float>());
    return out;
}

IntersectJvpOutputs intersect_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d) {
    const int64_t ray_count = ray_d.size(0);
    IntersectJvpOutputs out;
    out.tangent_t = at::zeros({ray_count}, vertices.options());
    out.tangent_p = at::zeros({ray_count, 3}, vertices.options());
    out.tangent_n = at::zeros({ray_count, 3}, vertices.options());
    out.tangent_geo_n = at::zeros({ray_count, 3}, vertices.options());
    out.tangent_uv = at::zeros({ray_count, 2}, vertices.options());
    out.tangent_barycentric = at::zeros({ray_count, 3}, vertices.options());

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    intersect_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        active.data_ptr<bool>(),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tangent_vertices.data_ptr<float>(),
        tangent_ray_o.data_ptr<float>(),
        tangent_ray_d.data_ptr<float>(),
        ray_count,
        out.tangent_t.data_ptr<float>(),
        out.tangent_p.data_ptr<float>(),
        out.tangent_n.data_ptr<float>(),
        out.tangent_geo_n.data_ptr<float>(),
        out.tangent_uv.data_ptr<float>(),
        out.tangent_barycentric.data_ptr<float>());
    return out;
}

} // namespace raydtorch
