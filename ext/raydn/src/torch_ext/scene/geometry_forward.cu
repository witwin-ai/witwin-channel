#include <raydn/scene/geometry_kernels.h>
#include <raydn/common/math.cuh>
#include <raydn/common/optix_context.h>
#include <raydn/scene/optix_intersect_params.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix_stubs.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void intersect_recompute_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const bool *__restrict__ active,
    int64_t ray_count,
    const float *__restrict__ optix_t,
    const int *__restrict__ optix_shape_id,
    const int *__restrict__ optix_local_prim_id,
    const int *__restrict__ optix_global_prim_id,
    const float *__restrict__ optix_bary_uv,
    float *__restrict__ out_p,
    float *__restrict__ out_n,
    float *__restrict__ out_geo_n,
    float *__restrict__ out_uv,
    float *__restrict__ out_bary,
    int *__restrict__ out_shape_id,
    int *__restrict__ out_prim_id,
    int *__restrict__ out_local_prim_id,
    int *__restrict__ out_global_prim_id,
    int64_t flags) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;

    const bool lane_active = active == nullptr || active[ray_idx];
    const float3 o = make_f3(ray_o + ray_idx * 3);
    const float3 d = make_f3(ray_d + ray_idx * 3);
    const int shape_id = lane_active ? optix_shape_id[ray_idx] : -1;
    const int local_prim_id = lane_active ? optix_local_prim_id[ray_idx] : -1;
    const int global_prim_id = lane_active ? optix_global_prim_id[ray_idx] : -1;
    const float hit_t = optix_t[ray_idx];
    const float u = optix_bary_uv[ray_idx * 2 + 0];
    const float v = optix_bary_uv[ray_idx * 2 + 1];
    const bool hit = global_prim_id >= 0;
    const bool want_geometric = (flags & kRayFlagsGeometric) != 0;
    const bool want_shading = (flags & kRayFlagsShadingN) != 0;
    const bool want_uv = (flags & kRayFlagsUV) != 0;

    if (want_geometric) {
        out_shape_id[ray_idx] = hit ? shape_id : -1;
        out_prim_id[ray_idx] = hit ? global_prim_id : -1;
        out_local_prim_id[ray_idx] = hit ? local_prim_id : -1;
        out_global_prim_id[ray_idx] = hit ? global_prim_id : -1;
        out_bary[ray_idx * 3 + 0] = hit ? 1.f - u - v : 0.f;
        out_bary[ray_idx * 3 + 1] = hit ? u : 0.f;
        out_bary[ray_idx * 3 + 2] = hit ? v : 0.f;
    }
    if (want_uv) {
        out_uv[ray_idx * 2 + 0] = hit ? u : 0.f;
        out_uv[ray_idx * 2 + 1] = hit ? v : 0.f;
    }

    if (want_geometric) {
        const float safe_t = hit ? hit_t : 0.f;
        const float3 p = add3(o, mul3(safe_t, d));
        out_p[ray_idx * 3 + 0] = hit ? p.x : 0.f;
        out_p[ray_idx * 3 + 1] = hit ? p.y : 0.f;
        out_p[ray_idx * 3 + 2] = hit ? p.z : 0.f;
    }

    float3 normal = make_float3(0.f, 0.f, 0.f);
    if ((want_geometric || want_shading) && hit) {
        const int i0 = faces[global_prim_id * 3 + 0];
        const int i1 = faces[global_prim_id * 3 + 1];
        const int i2 = faces[global_prim_id * 3 + 2];
        const float3 p0 = make_f3(vertices + i0 * 3);
        const float3 p1 = make_f3(vertices + i1 * 3);
        const float3 p2 = make_f3(vertices + i2 * 3);
        normal = cross3(sub3(p1, p0), sub3(p2, p0));
        const float inv_len = rsqrtf(fmaxf(dot3(normal, normal), 1e-20f));
        normal = mul3(inv_len, normal);
    }
    if (want_shading) {
        out_n[ray_idx * 3 + 0] = normal.x;
        out_n[ray_idx * 3 + 1] = normal.y;
        out_n[ray_idx * 3 + 2] = normal.z;
    }
    if (want_geometric) {
        out_geo_n[ray_idx * 3 + 0] = normal.x;
        out_geo_n[ray_idx * 3 + 1] = normal.y;
        out_geo_n[ray_idx * 3 + 2] = normal.z;
    }
}

struct CachedIntersectParams {
    OptixIntersectParams params;
    cudaStream_t stream = nullptr;
    at::Tensor buffer;
};

struct IntersectParamsCache {
    std::vector<CachedIntersectParams> entries;
    at::Tensor fallback_buffer;
};

bool same_intersect_params(const OptixIntersectParams &a, const OptixIntersectParams &b) {
    return a.traversable == b.traversable &&
           a.ray_o == b.ray_o &&
           a.ray_d == b.ray_d &&
           a.ray_tmax == b.ray_tmax &&
           a.active == b.active &&
           a.out_t == b.out_t &&
           a.out_shape_id == b.out_shape_id &&
           a.out_local_prim_id == b.out_local_prim_id &&
           a.out_global_prim_id == b.out_global_prim_id &&
           a.out_bary_uv == b.out_bary_uv &&
           a.face_offsets == b.face_offsets &&
           a.mesh_count == b.mesh_count &&
           a.ray_count == b.ray_count;
}

at::Tensor make_intersect_params_buffer(int device_index) {
    c10::cuda::CUDAGuard guard(device_index);
    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte);
    return at::empty({static_cast<int64_t>(sizeof(OptixIntersectParams))}, byte_options);
}

CUdeviceptr intersect_params_device_ptr(
    int device_index,
    const OptixIntersectParams &params,
    cudaStream_t stream) {
    constexpr size_t kMaxCachedParamBuffers = 128;
    static std::mutex buffer_mutex;
    static std::unordered_map<int, IntersectParamsCache> caches;

    std::lock_guard<std::mutex> lock(buffer_mutex);
    IntersectParamsCache &cache = caches[device_index];
    for (CachedIntersectParams &entry : cache.entries) {
        if (entry.stream == stream && same_intersect_params(entry.params, params)) {
            return reinterpret_cast<CUdeviceptr>(entry.buffer.data_ptr<uint8_t>());
        }
    }

    at::Tensor *buffer = nullptr;
    if (cache.entries.size() < kMaxCachedParamBuffers) {
        cache.entries.push_back(CachedIntersectParams{params, stream, make_intersect_params_buffer(device_index)});
        buffer = &cache.entries.back().buffer;
    } else {
        if (!cache.fallback_buffer.defined()) {
            cache.fallback_buffer = make_intersect_params_buffer(device_index);
        }
        buffer = &cache.fallback_buffer;
    }
    cuda_check(
        cudaMemcpyAsync(
            buffer->data_ptr<uint8_t>(),
            &params,
            sizeof(OptixIntersectParams),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(OptiX intersect params)");
    return reinterpret_cast<CUdeviceptr>(buffer->data_ptr<uint8_t>());
}

void launch_intersect_optix(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    at::Tensor &out_t,
    int *out_shape_id,
    int *out_local_prim_id,
    int *out_global_prim_id,
    float *out_bary_uv,
    cudaStream_t stream) {
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_intersect_pipeline(optix_entry);

    OptixIntersectParams params = {};
    params.traversable = scene.triangle_ias.traversable;
    params.ray_o = ray_o.data_ptr<float>();
    params.ray_d = ray_d.data_ptr<float>();
    params.ray_tmax =
        (!ray_tmax.defined() || ray_tmax.numel() == 0) ? nullptr : ray_tmax.data_ptr<float>();
    params.active = optional_mask_ptr(active);
    params.out_t = out_t.data_ptr<float>();
    params.out_shape_id = out_shape_id;
    params.out_local_prim_id = out_local_prim_id;
    params.out_global_prim_id = out_global_prim_id;
    params.out_bary_uv = out_bary_uv;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.mesh_count = static_cast<int32_t>(scene.meshes.size());
    params.ray_count = static_cast<int32_t>(ray_o.size(0));

    const CUdeviceptr params_device_ptr =
        intersect_params_device_ptr(static_cast<int>(scene.device_index), params, stream);

    raydn_OPTIX_CHECK(optixLaunch(
        optix_entry.intersect_pipeline,
        stream,
        params_device_ptr,
        sizeof(OptixIntersectParams),
        &optix_entry.intersect_sbt,
        static_cast<unsigned int>(ray_o.size(0)),
        1,
        1));
}

} // namespace

IntersectForwardOutputs intersect_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const at::Tensor &vertices = scene.global_vertices;
    const at::Tensor &faces = scene.global_faces;
    const int64_t ray_count = ray_o.size(0);
    auto fopts = vertices.options();
    auto iopts = faces.options();

    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    at::Tensor optix_bary_uv = at::empty({ray_count, 2}, fopts);
    out.p = at::empty({ray_count, 3}, fopts);
    out.n = at::empty({ray_count, 3}, fopts);
    out.geo_n = at::empty({ray_count, 3}, fopts);
    out.uv = at::empty({ray_count, 2}, fopts);
    out.barycentric = at::empty({ray_count, 3}, fopts);
    out.shape_id = at::empty({ray_count}, iopts);
    out.prim_id = at::empty({ray_count}, iopts);
    out.local_prim_id = at::empty({ray_count}, iopts);
    out.global_prim_id = at::empty({ray_count}, iopts);
    out.tape_prim_id = out.global_prim_id;
    out.tape_barycentric = out.barycentric;
    out.tape_t = out.t;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_intersect_optix(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        out.t,
        out.shape_id.data_ptr<int>(),
        out.local_prim_id.data_ptr<int>(),
        out.global_prim_id.data_ptr<int>(),
        optix_bary_uv.data_ptr<float>(),
        torch_ctx.stream);

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    intersect_recompute_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        optional_mask_ptr(active),
        ray_count,
        out.t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.local_prim_id.data_ptr<int>(),
        out.global_prim_id.data_ptr<int>(),
        optix_bary_uv.data_ptr<float>(),
        out.p.data_ptr<float>(),
        out.n.data_ptr<float>(),
        out.geo_n.data_ptr<float>(),
        out.uv.data_ptr<float>(),
        out.barycentric.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.prim_id.data_ptr<int>(),
        out.local_prim_id.data_ptr<int>(),
        out.global_prim_id.data_ptr<int>(),
        kRayFlagsAll);

    return out;
}

IntersectForwardOutputs intersect_forward_flags_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    int64_t flags) {
    const int64_t ray_count = ray_o.size(0);
    auto fopts = scene.global_vertices.options();
    auto iopts = scene.global_faces.options();
    const bool want_geometric = (flags & kRayFlagsGeometric) != 0;
    const bool want_shading = (flags & kRayFlagsShadingN) != 0;
    const bool want_uv = (flags & kRayFlagsUV) != 0;

    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    out.p = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.n = at::empty({want_shading ? ray_count : 0, 3}, fopts);
    out.geo_n = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.uv = at::empty({want_uv ? ray_count : 0, 2}, fopts);
    out.barycentric = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.shape_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.local_prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.global_prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.tape_prim_id = out.global_prim_id;
    out.tape_barycentric = out.barycentric;
    out.tape_t = out.t;

    at::Tensor optix_shape_id = want_geometric || want_shading || want_uv
        ? at::empty({ray_count}, iopts)
        : at::Tensor();
    at::Tensor optix_local_prim_id = want_geometric || want_shading || want_uv
        ? at::empty({ray_count}, iopts)
        : at::Tensor();
    at::Tensor optix_global_prim_id = want_geometric || want_shading || want_uv
        ? at::empty({ray_count}, iopts)
        : at::Tensor();
    at::Tensor optix_bary_uv = want_geometric || want_shading || want_uv
        ? at::empty({ray_count, 2}, fopts)
        : at::Tensor();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_intersect_optix(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        out.t,
        optix_shape_id.defined() ? optix_shape_id.data_ptr<int>() : nullptr,
        optix_local_prim_id.defined() ? optix_local_prim_id.data_ptr<int>() : nullptr,
        optix_global_prim_id.defined() ? optix_global_prim_id.data_ptr<int>() : nullptr,
        optix_bary_uv.defined() ? optix_bary_uv.data_ptr<float>() : nullptr,
        torch_ctx.stream);
    if (flags != 0) {
        const int threads = 128;
        const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
        intersect_recompute_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
            scene.global_vertices.data_ptr<float>(),
            scene.global_faces.data_ptr<int>(),
            ray_o.data_ptr<float>(),
            ray_d.data_ptr<float>(),
            optional_mask_ptr(active),
            ray_count,
            out.t.data_ptr<float>(),
            optix_shape_id.data_ptr<int>(),
            optix_local_prim_id.data_ptr<int>(),
            optix_global_prim_id.data_ptr<int>(),
            optix_bary_uv.data_ptr<float>(),
            want_geometric ? out.p.data_ptr<float>() : nullptr,
            want_shading ? out.n.data_ptr<float>() : nullptr,
            want_geometric ? out.geo_n.data_ptr<float>() : nullptr,
            want_uv ? out.uv.data_ptr<float>() : nullptr,
            want_geometric ? out.barycentric.data_ptr<float>() : nullptr,
            want_geometric ? out.shape_id.data_ptr<int>() : nullptr,
            want_geometric ? out.prim_id.data_ptr<int>() : nullptr,
            want_geometric ? out.local_prim_id.data_ptr<int>() : nullptr,
            want_geometric ? out.global_prim_id.data_ptr<int>() : nullptr,
            flags);
    }
    return out;
}

IntersectForwardOutputs intersect_forward_ad_flags_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    int64_t flags) {
    const int64_t ray_count = ray_o.size(0);
    auto fopts = scene.global_vertices.options();
    auto iopts = scene.global_faces.options();
    const bool want_geometric = (flags & kRayFlagsGeometric) != 0;
    const bool want_shading = (flags & kRayFlagsShadingN) != 0;
    const bool want_uv = (flags & kRayFlagsUV) != 0;

    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    out.p = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.n = at::empty({want_shading ? ray_count : 0, 3}, fopts);
    out.geo_n = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.uv = at::empty({want_uv ? ray_count : 0, 2}, fopts);
    out.barycentric = at::empty({want_geometric ? ray_count : 0, 3}, fopts);
    out.shape_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.local_prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);
    out.global_prim_id = at::empty({want_geometric ? ray_count : 0}, iopts);

    at::Tensor optix_shape_id = at::empty({ray_count}, iopts);
    at::Tensor optix_local_prim_id = at::empty({ray_count}, iopts);
    at::Tensor optix_global_prim_id = at::empty({ray_count}, iopts);
    at::Tensor optix_bary_uv = at::empty({ray_count, 2}, fopts);
    out.tape_prim_id = optix_global_prim_id;
    out.tape_barycentric = optix_bary_uv;
    out.tape_t = out.t;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_intersect_optix(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        out.t,
        optix_shape_id.data_ptr<int>(),
        optix_local_prim_id.data_ptr<int>(),
        optix_global_prim_id.data_ptr<int>(),
        optix_bary_uv.data_ptr<float>(),
        torch_ctx.stream);

    if (flags != 0) {
        const int threads = 128;
        const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
        intersect_recompute_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
            scene.global_vertices.data_ptr<float>(),
            scene.global_faces.data_ptr<int>(),
            ray_o.data_ptr<float>(),
            ray_d.data_ptr<float>(),
            optional_mask_ptr(active),
            ray_count,
            out.t.data_ptr<float>(),
            optix_shape_id.data_ptr<int>(),
            optix_local_prim_id.data_ptr<int>(),
            optix_global_prim_id.data_ptr<int>(),
            optix_bary_uv.data_ptr<float>(),
            want_geometric ? out.p.data_ptr<float>() : nullptr,
            want_shading ? out.n.data_ptr<float>() : nullptr,
            want_geometric ? out.geo_n.data_ptr<float>() : nullptr,
            want_uv ? out.uv.data_ptr<float>() : nullptr,
            want_geometric ? out.barycentric.data_ptr<float>() : nullptr,
            want_geometric ? out.shape_id.data_ptr<int>() : nullptr,
            want_geometric ? out.prim_id.data_ptr<int>() : nullptr,
            want_geometric ? out.local_prim_id.data_ptr<int>() : nullptr,
            want_geometric ? out.global_prim_id.data_ptr<int>() : nullptr,
            flags);
    }
    return out;
}

IntersectForwardOutputs intersect_forward_tape_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const int64_t ray_count = ray_o.size(0);
    auto fopts = scene.global_vertices.options();
    auto iopts = scene.global_faces.options();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    IntersectForwardOutputs out;
    out.t = at::empty({ray_count}, fopts);
    out.tape_prim_id = at::empty({ray_count}, iopts);
    out.tape_barycentric = at::Tensor();
    out.tape_t = out.t;

    launch_intersect_optix(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        out.t,
        nullptr,
        nullptr,
        out.tape_prim_id.data_ptr<int>(),
        nullptr,
        torch_ctx.stream);
    return out;
}

at::Tensor intersect_forward_t_only_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    const int64_t ray_count = ray_o.size(0);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    at::Tensor out_t = at::empty({ray_count}, scene.global_vertices.options());
    launch_intersect_optix(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        out_t,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        torch_ctx.stream);
    return out_t;
}

} // namespace raydn
