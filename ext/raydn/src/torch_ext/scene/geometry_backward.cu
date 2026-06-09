#include <raydn/scene/geometry_kernels.h>
#include <raydn/common/math.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

namespace raydn {

namespace {

constexpr int64_t kRayFlagsGeometric = 0x01;
constexpr int64_t kRayFlagsShadingN = 0x02;
constexpr int64_t kRayFlagsUV = 0x04;
constexpr int64_t kRayFlagsAll = kRayFlagsGeometric | kRayFlagsShadingN | kRayFlagsUV;

const bool *optional_mask_ptr(const at::Tensor &active) {
    if (!active.defined() || active.numel() == 0)
        return nullptr;
    return active.data_ptr<bool>();
}

int64_t optional_stride(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 || tensor->dim() <= dim)
        return 0;
    return tensor->stride(dim);
}

void zero_float_tensor_async(const at::Tensor &tensor, cudaStream_t stream) {
    if (tensor.defined() && tensor.numel() > 0) {
        cudaMemsetAsync(tensor.data_ptr<float>(), 0, static_cast<size_t>(tensor.numel()) * sizeof(float), stream);
    }
}

__device__ float read_scalar_or_zero(const float *base, int64_t index, int64_t stride0) {
    return base == nullptr ? 0.f : base[index * stride0];
}

__device__ float2 read_vec2_or_zero(const float *base, int64_t index, int64_t stride0, int64_t stride1) {
    return base == nullptr ? make_float2(0.f, 0.f)
                           : make_float2(base[index * stride0 + 0 * stride1],
                                         base[index * stride0 + 1 * stride1]);
}

__device__ float3 read_vec3_or_zero(const float *base, int64_t index, int64_t stride0, int64_t stride1) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[index * stride0 + 0 * stride1],
                                         base[index * stride0 + 1 * stride1],
                                         base[index * stride0 + 2 * stride1]);
}

__device__ void add_to3(float *base, int index, float3 value) {
    base[index * 3 + 0] += value.x;
    base[index * 3 + 1] += value.y;
    base[index * 3 + 2] += value.z;
}

__device__ void atomic_add3_warp_labeled(float *base, int index, float3 value) {
    namespace cg = cooperative_groups;
    cg::coalesced_group active = cg::coalesced_threads();
    auto group = cg::labeled_partition(active, index);
    const float x = cg::reduce(group, value.x, cg::plus<float>());
    const float y = cg::reduce(group, value.y, cg::plus<float>());
    const float z = cg::reduce(group, value.z, cg::plus<float>());
    if (group.thread_rank() == 0) {
        atomic_add3(base, index, make_float3(x, y, z));
    }
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
    int64_t grad_t_stride0,
    int64_t grad_p_stride0,
    int64_t grad_p_stride1,
    int64_t grad_n_stride0,
    int64_t grad_n_stride1,
    int64_t grad_geo_n_stride0,
    int64_t grad_geo_n_stride1,
    int64_t grad_uv_stride0,
    int64_t grad_uv_stride1,
    int64_t grad_bary_stride0,
    int64_t grad_bary_stride1,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    float *__restrict__ grad_ray_tmax) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    if (grad_ray_tmax != nullptr)
        grad_ray_tmax[ray_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        if (grad_ray_o != nullptr)
            grad_ray_o[ray_idx * 3 + axis] = 0.f;
        if (grad_ray_d != nullptr)
            grad_ray_d[ray_idx * 3 + axis] = 0.f;
    }
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
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const float3 c0 = mul3(-1.f, d);
    float3 bary;
    if (tape_bary_width == 0) {
        const float3 y = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx * 3), v0));
        bary = make_float3(1.f - y.y - y.z, y.y, y.z);
    } else if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        bary = make_float3(1.f - u - v, u, v);
    } else {
        bary = make_f3(tape_bary + ray_idx * 3);
    }

    float3 g_vertices0 = make_float3(0.f, 0.f, 0.f);
    float3 g_vertices1 = make_float3(0.f, 0.f, 0.f);
    float3 g_vertices2 = make_float3(0.f, 0.f, 0.f);

    const float3 gp = read_vec3_or_zero(grad_p, ray_idx, grad_p_stride0, grad_p_stride1);
    const float3 gn = add3(
        read_vec3_or_zero(grad_n, ray_idx, grad_n_stride0, grad_n_stride1),
        read_vec3_or_zero(grad_geo_n, ray_idx, grad_geo_n_stride0, grad_geo_n_stride1));
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
    if (grad_ray_o != nullptr)
        add_to3(grad_ray_o, ray_idx, gp);

    const float2 guv = read_vec2_or_zero(grad_uv, ray_idx, grad_uv_stride0, grad_uv_stride1);
    float gu = guv.x;
    float gv = guv.y;
    const float3 gbary = read_vec3_or_zero(grad_bary, ray_idx, grad_bary_stride0, grad_bary_stride1);
    const float gb0 = gbary.x;
    const float gb1 = gbary.y;
    const float gb2 = gbary.z;
    gu += -gb0 + gb1;
    gv += -gb0 + gb2;

    const float3 gy = make_float3(read_scalar_or_zero(grad_t, ray_idx, grad_t_stride0) + t_bar_from_p, gu, gv);
    const float3 lambda = solve_transpose_columns(c0, e1, e2, gy);

    if (grad_ray_o != nullptr)
        add_to3(grad_ray_o, ray_idx, lambda);
    const float solved_t = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx * 3), v0)).x;
    if (grad_ray_d != nullptr)
        add_to3(grad_ray_d, ray_idx, mul3(solved_t, add3(lambda, gp)));

    g_vertices0 = sub3(g_vertices0, mul3(bary.x, lambda));
    g_vertices1 = sub3(g_vertices1, mul3(bary.y, lambda));
    g_vertices2 = sub3(g_vertices2, mul3(bary.z, lambda));

    if (grad_vertices != nullptr) {
        atomic_add3(grad_vertices, i0, g_vertices0);
        atomic_add3(grad_vertices, i1, g_vertices1);
        atomic_add3(grad_vertices, i2, g_vertices2);
    }
}

__global__ void intersect_backward_t_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ grad_t,
    int64_t grad_t_stride,
    float grad_t_constant,
    bool grad_t_is_constant,
    int64_t ray_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_ray_o,
    float *__restrict__ grad_ray_d,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    if (active != nullptr && !active[ray_idx])
        return;
    const int prim_id = tape_prim_id[ray_idx];
    if (prim_id < 0)
        return;
    const float gt = grad_t_is_constant ? grad_t_constant : grad_t[ray_idx * grad_t_stride];
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
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const float3 c0 = mul3(-1.f, d);
    const float3 lambda = solve_transpose_columns(c0, e1, e2, make_float3(gt, 0.f, 0.f));

    if (need_grad_ray_o) {
        add_to3(grad_ray_o, ray_idx, lambda);
    }
    if (need_grad_ray_d) {
        const float solved_t = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx * 3), v0)).x;
        add_to3(grad_ray_d, ray_idx, mul3(solved_t, lambda));
    }
    if (!need_grad_vertices) {
        return;
    }

    float3 bary;
    if (tape_bary_width == 0) {
        const float3 y = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx * 3), v0));
        bary = make_float3(1.f - y.y - y.z, y.y, y.z);
    } else if (tape_bary_width == 2) {
        const float u = tape_bary[ray_idx * 2 + 0];
        const float v = tape_bary[ray_idx * 2 + 1];
        bary = make_float3(1.f - u - v, u, v);
    } else {
        bary = make_f3(tape_bary + ray_idx * 3);
    }
    atomic_add3_warp_labeled(grad_vertices, i0, mul3(-bary.x, lambda));
    atomic_add3_warp_labeled(grad_vertices, i1, mul3(-bary.y, lambda));
    atomic_add3_warp_labeled(grad_vertices, i2, mul3(-bary.z, lambda));
}

__global__ void intersect_backward_t_vertices_coop_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    const int *__restrict__ tape_prim_id,
    const float *__restrict__ tape_bary,
    int tape_bary_width,
    const float *__restrict__ grad_t,
    int64_t grad_t_stride,
    int64_t vertex_count,
    int64_t ray_count,
    float *__restrict__ grad_vertices) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t thread_count = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = thread_index; idx < vertex_count * 3; idx += thread_count) {
        grad_vertices[idx] = 0.f;
    }
    grid.sync();

    for (int64_t ray_idx64 = thread_index; ray_idx64 < ray_count; ray_idx64 += thread_count) {
        const int ray_idx = static_cast<int>(ray_idx64);
        if (active != nullptr && !active[ray_idx])
            continue;
        const int prim_id = tape_prim_id[ray_idx];
        if (prim_id < 0)
            continue;
        const float gt = grad_t[ray_idx64 * grad_t_stride];
        if (gt == 0.f)
            continue;

        const int i0 = faces[prim_id * 3 + 0];
        const int i1 = faces[prim_id * 3 + 1];
        const int i2 = faces[prim_id * 3 + 2];
        const float3 v0 = make_f3(vertices + i0 * 3);
        const float3 v1 = make_f3(vertices + i1 * 3);
        const float3 v2 = make_f3(vertices + i2 * 3);
        const float3 e1 = sub3(v1, v0);
        const float3 e2 = sub3(v2, v0);
        const float3 d = make_f3(ray_d + ray_idx64 * 3);
        const float3 c0 = mul3(-1.f, d);
        const float3 lambda = solve_transpose_columns(c0, e1, e2, make_float3(gt, 0.f, 0.f));

        float3 bary;
        if (tape_bary_width == 0) {
            const float3 y = solve_columns(c0, e1, e2, sub3(make_f3(ray_o + ray_idx64 * 3), v0));
            bary = make_float3(1.f - y.y - y.z, y.y, y.z);
        } else if (tape_bary_width == 2) {
            const float u = tape_bary[ray_idx64 * 2 + 0];
            const float v = tape_bary[ray_idx64 * 2 + 1];
            bary = make_float3(1.f - u - v, u, v);
        } else {
            bary = make_f3(tape_bary + ray_idx64 * 3);
        }
        atomic_add3_warp_labeled(grad_vertices, i0, mul3(-bary.x, lambda));
        atomic_add3_warp_labeled(grad_vertices, i1, mul3(-bary.y, lambda));
        atomic_add3_warp_labeled(grad_vertices, i2, mul3(-bary.z, lambda));
    }
}

struct CoopLaunchConfig {
    bool supported = false;
    int max_blocks = 0;
};

CoopLaunchConfig coop_launch_config_for_device(int device) {
    static std::mutex config_mutex;
    static std::unordered_map<int, CoopLaunchConfig> configs;
    std::lock_guard<std::mutex> lock(config_mutex);
    auto it = configs.find(device);
    if (it != configs.end())
        return it->second;

    CoopLaunchConfig config;
    cudaDeviceProp prop{};
    cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status == cudaSuccess && prop.cooperativeLaunch) {
            constexpr int threads = 128;
        int active_blocks_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            intersect_backward_t_vertices_coop_kernel,
            threads,
            0);
        if (status == cudaSuccess && active_blocks_per_sm > 0) {
            config.supported = true;
            config.max_blocks = active_blocks_per_sm * prop.multiProcessorCount;
        }
    }
    if (status != cudaSuccess) {
        cudaGetLastError();
    }
    configs.emplace(device, config);
    return config;
}

bool launch_intersect_backward_t_vertices_coop(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t,
    int64_t grad_t_stride,
    at::Tensor &grad_vertices,
    cudaStream_t stream) {
    int device = vertices.get_device();
    CoopLaunchConfig config = coop_launch_config_for_device(device);
    if (!config.supported || config.max_blocks <= 0)
        return false;
    constexpr int threads = 128;
    int64_t vertex_count = vertices.size(0);
    int64_t ray_count = ray_d.size(0);
    const int64_t work_count = vertex_count > ray_count ? vertex_count : ray_count;
    if (work_count == 0) {
        return true;
    }
    int blocks = static_cast<int>((work_count + threads - 1) / threads);
    if (blocks > config.max_blocks)
        blocks = config.max_blocks;
    if (blocks <= 0)
        return false;

    int tape_bary_width =
        (!tape_barycentric.defined() || tape_barycentric.numel() == 0)
            ? 0
            : static_cast<int>(tape_barycentric.size(1));
    const bool *active_ptr = optional_mask_ptr(active);
    const float *vertices_ptr = vertices.data_ptr<float>();
    const int *faces_ptr = faces.data_ptr<int>();
    const float *ray_o_ptr = ray_o.data_ptr<float>();
    const float *ray_d_ptr = ray_d.data_ptr<float>();
    const int *tape_prim_id_ptr = tape_prim_id.data_ptr<int>();
    const float *tape_bary_ptr =
        tape_bary_width == 0 ? nullptr : tape_barycentric.data_ptr<float>();
    const float *grad_t_ptr = grad_t.data_ptr<float>();
    float *grad_vertices_ptr = grad_vertices.data_ptr<float>();
    void *kernel_args[] = {
        &vertices_ptr,
        &faces_ptr,
        &ray_o_ptr,
        &ray_d_ptr,
        &active_ptr,
        &tape_prim_id_ptr,
        &tape_bary_ptr,
        &tape_bary_width,
        &grad_t_ptr,
        &grad_t_stride,
        &vertex_count,
        &ray_count,
        &grad_vertices_ptr,
    };
    cudaError_t status = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(intersect_backward_t_vertices_coop_kernel),
        blocks,
        threads,
        kernel_args,
        0,
        stream);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
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
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_ray_o_stride0,
    int64_t tangent_ray_o_stride1,
    int64_t tangent_ray_d_stride0,
    int64_t tangent_ray_d_stride1,
    int64_t ray_count,
    float *__restrict__ tangent_t,
    float *__restrict__ tangent_p,
    float *__restrict__ tangent_n,
    float *__restrict__ tangent_geo_n,
    float *__restrict__ tangent_uv,
    float *__restrict__ tangent_barycentric,
    int64_t flags) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    const bool want_geometric = (flags & kRayFlagsGeometric) != 0;
    const bool want_shading = (flags & kRayFlagsShadingN) != 0;
    const bool want_uv = (flags & kRayFlagsUV) != 0;
    const bool want_normal = want_geometric || want_shading;
    if (want_geometric) {
        for (int axis = 0; axis < 3; ++axis) {
            tangent_p[ray_idx * 3 + axis] = 0.f;
            tangent_geo_n[ray_idx * 3 + axis] = 0.f;
            tangent_barycentric[ray_idx * 3 + axis] = 0.f;
        }
    }
    if (want_shading) {
        for (int axis = 0; axis < 3; ++axis) {
            tangent_n[ray_idx * 3 + axis] = 0.f;
        }
    }
    tangent_t[ray_idx] = 0.f;
    if (want_uv) {
        tangent_uv[ray_idx * 2 + 0] = 0.f;
        tangent_uv[ray_idx * 2 + 1] = 0.f;
    }
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
    const float3 de1 = sub3(dv1, dv0);
    const float3 de2 = sub3(dv2, dv0);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const float3 dd = read_vec3_or_zero(tangent_ray_d, ray_idx, tangent_ray_d_stride0, tangent_ray_d_stride1);
    const float3 dorigin = read_vec3_or_zero(tangent_ray_o, ray_idx, tangent_ray_o_stride0, tangent_ray_o_stride1);
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

    tangent_t[ray_idx] = dt;
    if (want_uv) {
        tangent_uv[ray_idx * 2 + 0] = du;
        tangent_uv[ray_idx * 2 + 1] = dv;
    }
    if (want_geometric) {
        const float3 dp = add3(dorigin, add3(mul3(dt, d), mul3(solved_t, dd)));
        tangent_barycentric[ray_idx * 3 + 0] = -du - dv;
        tangent_barycentric[ray_idx * 3 + 1] = du;
        tangent_barycentric[ray_idx * 3 + 2] = dv;
        tangent_p[ray_idx * 3 + 0] = dp.x;
        tangent_p[ray_idx * 3 + 1] = dp.y;
        tangent_p[ray_idx * 3 + 2] = dp.z;
    }
    if (want_normal) {
        const float3 dn = normal_jvp(e1, e2, de1, de2);
        if (want_shading) {
            tangent_n[ray_idx * 3 + 0] = dn.x;
            tangent_n[ray_idx * 3 + 1] = dn.y;
            tangent_n[ray_idx * 3 + 2] = dn.z;
        }
        if (want_geometric) {
            tangent_geo_n[ray_idx * 3 + 0] = dn.x;
            tangent_geo_n[ray_idx * 3 + 1] = dn.y;
            tangent_geo_n[ray_idx * 3 + 2] = dn.z;
        }
    }
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
    return intersect_backward_optional_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        &grad_t,
        &grad_p,
        &grad_n,
        &grad_geo_n,
        &grad_uv,
        &grad_barycentric,
        true,
        true,
        true,
        true);
}

IntersectBackwardOutputs intersect_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor *grad_t,
    const at::Tensor *grad_p,
    const at::Tensor *grad_n,
    const at::Tensor *grad_geo_n,
    const at::Tensor *grad_uv,
    const at::Tensor *grad_barycentric,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    (void)ray_tmax;
    const int64_t ray_count = ray_d.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    IntersectBackwardOutputs out;
    out.grad_vertices = need_grad_vertices ? at::empty(vertices.sizes(), vertices.options()) : at::Tensor();
    out.grad_ray_o = need_grad_ray_o ? at::empty(ray_d.sizes(), ray_d.options()) : at::Tensor();
    out.grad_ray_d = need_grad_ray_d ? at::empty(ray_d.sizes(), ray_d.options()) : at::Tensor();
    out.grad_ray_tmax = need_grad_ray_tmax ? at::empty({ray_count}, ray_d.options()) : at::Tensor();
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0 ||
        (!need_grad_vertices && !need_grad_ray_o && !need_grad_ray_d && !need_grad_ray_tmax)) {
        return out;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    intersect_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_mask_ptr(active),
        tape_prim_id.data_ptr<int>(),
        (!tape_barycentric.defined() || tape_barycentric.numel() == 0)
            ? nullptr
            : tape_barycentric.data_ptr<float>(),
        (!tape_barycentric.defined() || tape_barycentric.numel() == 0)
            ? 0
            : static_cast<int>(tape_barycentric.size(1)),
        grad_t == nullptr ? nullptr : grad_t->data_ptr<float>(),
        grad_p == nullptr ? nullptr : grad_p->data_ptr<float>(),
        grad_n == nullptr ? nullptr : grad_n->data_ptr<float>(),
        grad_geo_n == nullptr ? nullptr : grad_geo_n->data_ptr<float>(),
        grad_uv == nullptr ? nullptr : grad_uv->data_ptr<float>(),
        grad_barycentric == nullptr ? nullptr : grad_barycentric->data_ptr<float>(),
        optional_stride(grad_t, 0),
        optional_stride(grad_p, 0),
        optional_stride(grad_p, 1),
        optional_stride(grad_n, 0),
        optional_stride(grad_n, 1),
        optional_stride(grad_geo_n, 0),
        optional_stride(grad_geo_n, 1),
        optional_stride(grad_uv, 0),
        optional_stride(grad_uv, 1),
        optional_stride(grad_barycentric, 0),
        optional_stride(grad_barycentric, 1),
        ray_count,
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_ray_o ? out.grad_ray_o.data_ptr<float>() : nullptr,
        need_grad_ray_d ? out.grad_ray_d.data_ptr<float>() : nullptr,
        need_grad_ray_tmax ? out.grad_ray_tmax.data_ptr<float>() : nullptr);
    return out;
}

IntersectBackwardOutputs intersect_backward_t_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t,
    int64_t grad_t_stride,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    const int64_t ray_count = ray_d.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    IntersectBackwardOutputs out;
    out.grad_vertices = need_grad_vertices ? at::empty(vertices.sizes(), vertices.options()) : at::Tensor();
    out.grad_ray_o = need_grad_ray_o ? at::empty(ray_d.sizes(), ray_d.options()) : at::Tensor();
    out.grad_ray_d = need_grad_ray_d ? at::empty(ray_d.sizes(), ray_d.options()) : at::Tensor();
    out.grad_ray_tmax = need_grad_ray_tmax ? at::empty({ray_count}, ray_d.options()) : at::Tensor();
    if (need_grad_vertices && !need_grad_ray_o && !need_grad_ray_d && !need_grad_ray_tmax) {
        if (launch_intersect_backward_t_vertices_coop(
                vertices,
                faces,
                ray_o,
                ray_d,
                active,
                tape_prim_id,
                tape_barycentric,
                grad_t,
                grad_t_stride,
                out.grad_vertices,
                stream)) {
            return out;
        }
    }
    zero_float_tensor_async(out.grad_vertices, stream);
    zero_float_tensor_async(out.grad_ray_o, stream);
    zero_float_tensor_async(out.grad_ray_d, stream);
    zero_float_tensor_async(out.grad_ray_tmax, stream);

    if (ray_count == 0 || (!need_grad_vertices && !need_grad_ray_o && !need_grad_ray_d)) {
        return out;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    intersect_backward_t_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_mask_ptr(active),
        tape_prim_id.data_ptr<int>(),
        (!tape_barycentric.defined() || tape_barycentric.numel() == 0)
            ? nullptr
            : tape_barycentric.data_ptr<float>(),
        (!tape_barycentric.defined() || tape_barycentric.numel() == 0)
            ? 0
            : static_cast<int>(tape_barycentric.size(1)),
        grad_t.data_ptr<float>(),
        grad_t_stride,
        0.f,
        false,
        ray_count,
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_ray_o ? out.grad_ray_o.data_ptr<float>() : nullptr,
        need_grad_ray_d ? out.grad_ray_d.data_ptr<float>() : nullptr,
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d);
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
    return intersect_jvp_optional_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        &tangent_vertices,
        &tangent_ray_o,
        &tangent_ray_d,
        kRayFlagsAll);
}

IntersectJvpOutputs intersect_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    int64_t flags) {
    const int64_t ray_count = ray_d.size(0);
    const bool want_geometric = (flags & kRayFlagsGeometric) != 0;
    const bool want_shading = (flags & kRayFlagsShadingN) != 0;
    const bool want_uv = (flags & kRayFlagsUV) != 0;
    IntersectJvpOutputs out;
    out.tangent_t = at::empty({ray_count}, vertices.options());
    out.tangent_p = at::empty({want_geometric ? ray_count : 0, 3}, vertices.options());
    out.tangent_n = at::empty({want_shading ? ray_count : 0, 3}, vertices.options());
    out.tangent_geo_n = at::empty({want_geometric ? ray_count : 0, 3}, vertices.options());
    out.tangent_uv = at::empty({want_uv ? ray_count : 0, 2}, vertices.options());
    out.tangent_barycentric = at::empty({want_geometric ? ray_count : 0, 3}, vertices.options());
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    intersect_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_mask_ptr(active),
        tape_prim_id.data_ptr<int>(),
        tape_barycentric.data_ptr<float>(),
        static_cast<int>(tape_barycentric.size(1)),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_ray_o == nullptr ? nullptr : tangent_ray_o->data_ptr<float>(),
        tangent_ray_d == nullptr ? nullptr : tangent_ray_d->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_ray_o, 0),
        optional_stride(tangent_ray_o, 1),
        optional_stride(tangent_ray_d, 0),
        optional_stride(tangent_ray_d, 1),
        ray_count,
        out.tangent_t.data_ptr<float>(),
        want_geometric ? out.tangent_p.data_ptr<float>() : nullptr,
        want_shading ? out.tangent_n.data_ptr<float>() : nullptr,
        want_geometric ? out.tangent_geo_n.data_ptr<float>() : nullptr,
        want_uv ? out.tangent_uv.data_ptr<float>() : nullptr,
        want_geometric ? out.tangent_barycentric.data_ptr<float>() : nullptr,
        flags);
    return out;
}

} // namespace raydn
