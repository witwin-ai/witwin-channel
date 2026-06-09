#include <raydn/edge/kernels.h>
#include <raydn/edge/optix_params.h>
#include <raydn/common/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix_stubs.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace raydn {

namespace {

__device__ float3 make_aos_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ float3 edge_start(
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    int edge) {
    return make_float3(p0_x[edge], p0_y[edge], p0_z[edge]);
}

__device__ float3 edge_vector(
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    int edge) {
    return make_float3(e1_x[edge], e1_y[edge], e1_z[edge]);
}

__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_edge_cache_dtype(const at::Tensor &tensor, at::ScalarType dtype, const char *name) {
    if (tensor.scalar_type() != dtype)
        throw std::runtime_error(
            std::string("edge cache tensor ") + name + " has dtype " +
            std::string(c10::toString(tensor.scalar_type())) + ", expected " +
            std::string(c10::toString(dtype)) + ".");
}

void require_edge_cache_dtypes(const SceneCache &scene) {
    require_edge_cache_dtype(scene.edge_v0, at::kInt, "edge_v0");
    require_edge_cache_dtype(scene.edge_v1, at::kInt, "edge_v1");
    require_edge_cache_dtype(scene.edge_shape_id, at::kInt, "edge_shape_id");
    require_edge_cache_dtype(scene.edge_local_id, at::kInt, "edge_local_id");
    require_edge_cache_dtype(scene.edge_p0_x, at::kFloat, "edge_p0_x");
    require_edge_cache_dtype(scene.edge_p0_y, at::kFloat, "edge_p0_y");
    require_edge_cache_dtype(scene.edge_p0_z, at::kFloat, "edge_p0_z");
    require_edge_cache_dtype(scene.edge_e1_x, at::kFloat, "edge_e1_x");
    require_edge_cache_dtype(scene.edge_e1_y, at::kFloat, "edge_e1_y");
    require_edge_cache_dtype(scene.edge_e1_z, at::kFloat, "edge_e1_z");
    require_edge_cache_dtype(scene.edge_mask, at::kByte, "edge_mask");
}

void fill_edge_tiers(const SceneCache &scene, EdgeOptixQueryParams &params) {
    if (scene.edge_accels.size() > static_cast<std::size_t>(EdgeOptixMaxTiers)) {
        throw std::runtime_error(
            "edge query has more GAS tiers than EdgeOptixMaxTiers; increase the OptiX params "
            "tier array size.");
    }

    params.tier_count = static_cast<int>(scene.edge_accels.size());
    if (params.tier_count <= 0) {
        return;
    }

    params.handle = static_cast<uint64_t>(scene.edge_accels.front().traversable);
    params.search_radius = scene.edge_accels.back().search_radius;
    for (int tier = 0; tier < params.tier_count; ++tier) {
        const OptixEdgeAccel &accel = scene.edge_accels[static_cast<std::size_t>(tier)];
        params.tier_handles[tier] = static_cast<uint64_t>(accel.traversable);
        params.tier_search_radii[tier] = accel.search_radius;
    }
}

void fill_edge_geometry_params(
    const SceneCache &scene,
    int64_t edge_count,
    EdgeOptixQueryParams &params) {
    fill_edge_tiers(scene, params);
    params.edge_p0_x = scene.edge_p0_x.data_ptr<float>();
    params.edge_p0_y = scene.edge_p0_y.data_ptr<float>();
    params.edge_p0_z = scene.edge_p0_z.data_ptr<float>();
    params.edge_e1_x = scene.edge_e1_x.data_ptr<float>();
    params.edge_e1_y = scene.edge_e1_y.data_ptr<float>();
    params.edge_e1_z = scene.edge_e1_z.data_ptr<float>();
    params.edge_mask = scene.edge_mask.data_ptr<uint8_t>();
    params.edge_count = static_cast<int>(edge_count);
}

__global__ void init_edge_point_outputs_kernel(
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d,
    bool *__restrict__ unresolved) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    distance[point_idx] = CUDART_INF_F;
    edge_t[point_idx] = 0.f;
    shape_id[point_idx] = -1;
    edge_id[point_idx] = -1;
    global_edge_id[point_idx] = -1;
    tape_edge_id[point_idx] = -1;
    tape_s[point_idx] = 0.f;
    unresolved[point_idx] = true;
    for (int axis = 0; axis < 3; ++axis) {
        edge_point[point_idx * 3 + axis] = 0.f;
        tape_d[point_idx * 3 + axis] = 0.f;
    }
}

__global__ void init_edge_point_public_outputs_kernel(
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    bool *__restrict__ unresolved) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    distance[point_idx] = CUDART_INF_F;
    edge_t[point_idx] = 0.f;
    shape_id[point_idx] = -1;
    edge_id[point_idx] = -1;
    global_edge_id[point_idx] = -1;
    unresolved[point_idx] = true;
    for (int axis = 0; axis < 3; ++axis) {
        edge_point[point_idx * 3 + axis] = 0.f;
    }
}

__global__ void split_vec3_to_soa_kernel(
    const float *__restrict__ value,
    int64_t count,
    int64_t stride0,
    int64_t stride1,
    float *__restrict__ x,
    float *__restrict__ y,
    float *__restrict__ z) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float *row = value + static_cast<int64_t>(idx) * stride0;
    x[idx] = row[0 * stride1];
    y[idx] = row[1 * stride1];
    z[idx] = row[2 * stride1];
}

__global__ void split_ray_vec3_to_soa_kernel(
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    int64_t count,
    int64_t ray_o_stride0,
    int64_t ray_o_stride1,
    int64_t ray_d_stride0,
    int64_t ray_d_stride1,
    float *__restrict__ query_x,
    float *__restrict__ query_y,
    float *__restrict__ query_z,
    float *__restrict__ ray_dx,
    float *__restrict__ ray_dy,
    float *__restrict__ ray_dz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float *origin = ray_o + static_cast<int64_t>(idx) * ray_o_stride0;
    const float *direction = ray_d + static_cast<int64_t>(idx) * ray_d_stride0;
    query_x[idx] = origin[0 * ray_o_stride1];
    query_y[idx] = origin[1 * ray_o_stride1];
    query_z[idx] = origin[2 * ray_o_stride1];
    ray_dx[idx] = direction[0 * ray_d_stride1];
    ray_dy[idx] = direction[1 * ray_d_stride1];
    ray_dz[idx] = direction[2 * ray_d_stride1];
}

__global__ void init_edge_ray_outputs_kernel(
    const bool *__restrict__ active,
    int64_t ray_count,
    float *__restrict__ distance,
    float *__restrict__ ray_t,
    float *__restrict__ point,
    float *__restrict__ edge_t,
    float *__restrict__ edge_point,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    bool *__restrict__ unresolved) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    distance[ray_idx] = CUDART_INF_F;
    ray_t[ray_idx] = 0.f;
    edge_t[ray_idx] = 0.f;
    shape_id[ray_idx] = -1;
    edge_id[ray_idx] = -1;
    global_edge_id[ray_idx] = -1;
    tape_edge_id[ray_idx] = -1;
    unresolved[ray_idx] = active == nullptr || active[ray_idx];
    for (int axis = 0; axis < 3; ++axis) {
        point[ray_idx * 3 + axis] = 0.f;
        edge_point[ray_idx * 3 + axis] = 0.f;
    }
}

__global__ void finalize_edge_ray_stage_kernel(
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    const float *__restrict__ ray_o,
    const float *__restrict__ ray_d,
    const int *__restrict__ stage_edge_id,
    const float *__restrict__ stage_distance_sq,
    const float *__restrict__ stage_ray_t,
    const float *__restrict__ stage_edge_t,
    const bool *__restrict__ stage_valid,
    int64_t ray_count,
    float *__restrict__ distance,
    float *__restrict__ ray_t,
    float *__restrict__ point,
    float *__restrict__ edge_t,
    float *__restrict__ edge_point,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    bool *__restrict__ unresolved) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count || !unresolved[ray_idx])
        return;
    const int edge = stage_edge_id[ray_idx];
    if (!stage_valid[ray_idx] || edge < 0)
        return;

    const float rt = stage_ray_t[ray_idx];
    const float s = fminf(1.f, fmaxf(0.f, stage_edge_t[ray_idx]));
    const float3 ro = make_aos_f3(ray_o + ray_idx * 3);
    const float3 rd = make_aos_f3(ray_d + ray_idx * 3);
    const float3 rp = add3(ro, mul3(rt, rd));
    const float3 a = edge_start(p0_x, p0_y, p0_z, edge);
    const float3 e = edge_vector(e1_x, e1_y, e1_z, edge);
    const float3 ep = add3(a, mul3(s, e));

    distance[ray_idx] = sqrtf(fmaxf(stage_distance_sq[ray_idx], 0.f));
    ray_t[ray_idx] = rt;
    edge_t[ray_idx] = s;
    shape_id[ray_idx] = edge_shape_id[edge];
    edge_id[ray_idx] = edge_local_id[edge];
    global_edge_id[ray_idx] = edge;
    tape_edge_id[ray_idx] = edge;
    point[ray_idx * 3 + 0] = rp.x;
    point[ray_idx * 3 + 1] = rp.y;
    point[ray_idx * 3 + 2] = rp.z;
    edge_point[ray_idx * 3 + 0] = ep.x;
    edge_point[ray_idx * 3 + 1] = ep.y;
    edge_point[ray_idx * 3 + 2] = ep.z;
    unresolved[ray_idx] = false;
}

void launch_edge_query(
    OptixDeviceContextEntry &optix_entry,
    cudaStream_t stream,
    const EdgeOptixQueryParams &params,
    EdgeOptixLaunchKind kind,
    int64_t query_count) {
    cuda_check(
        cudaMemcpyAsync(
            optix_entry.edge_params_buffer.data_ptr<uint8_t>(),
            &params,
            sizeof(EdgeOptixQueryParams),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge OptiX params)");
    raydn_OPTIX_CHECK(optixLaunch(
        edge_pipeline(optix_entry, kind),
        stream,
        reinterpret_cast<CUdeviceptr>(optix_entry.edge_params_buffer.data_ptr<uint8_t>()),
        sizeof(EdgeOptixQueryParams),
        &edge_sbt(optix_entry, kind),
        static_cast<unsigned int>(query_count),
        1,
        1));
}

} // namespace

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point) {
    require_edge_cache_dtypes(scene);
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(point.device()).dtype(at::kBool);

    EdgeForwardOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);
    out.tape_edge_id = at::empty({point_count}, iopts);
    out.tape_s = at::empty({point_count}, fopts);
    out.tape_d = at::empty({point_count, 3}, fopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({point_count}, bopts);
    init_edge_point_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point_count,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        out.tape_s.data_ptr<float>(),
        out.tape_d.data_ptr<float>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || point_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = at::empty({point_count}, fopts);
    at::Tensor query_y = at::empty({point_count}, fopts);
    at::Tensor query_z = at::empty({point_count}, fopts);
    split_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point.data_ptr<float>(),
        point_count,
        point.stride(0),
        point.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>());

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(point_count);
    params.edge_shape_id = scene.edge_shape_id.data_ptr<int>();
    params.edge_local_id = scene.edge_local_id.data_ptr<int>();
    params.write_point_outputs = 1;
    params.final_distance = out.distance.data_ptr<float>();
    params.final_edge_point = out.edge_point.data_ptr<float>();
    params.final_edge_t = out.edge_t.data_ptr<float>();
    params.final_shape_id = out.shape_id.data_ptr<int>();
    params.final_edge_id = out.edge_id.data_ptr<int>();
    params.final_global_edge_id = out.global_edge_id.data_ptr<int>();
    params.final_tape_edge_id = out.tape_edge_id.data_ptr<int>();
    params.final_tape_s = out.tape_s.data_ptr<float>();
    params.final_tape_d = out.tape_d.data_ptr<float>();
    params.final_unresolved = reinterpret_cast<uint8_t *>(unresolved.data_ptr<bool>());

    launch_edge_query(
        optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Point, point_count);
    return out;
}

EdgeForwardPublicOutputs edge_forward_noad_cuda(const SceneCache &scene, const at::Tensor &point) {
    require_edge_cache_dtypes(scene);
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(point.device()).dtype(at::kBool);

    EdgeForwardPublicOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({point_count}, bopts);
    init_edge_point_public_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point_count,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || point_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = at::empty({point_count}, fopts);
    at::Tensor query_y = at::empty({point_count}, fopts);
    at::Tensor query_z = at::empty({point_count}, fopts);
    split_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        point.data_ptr<float>(),
        point_count,
        point.stride(0),
        point.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>());

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(point_count);
    params.edge_shape_id = scene.edge_shape_id.data_ptr<int>();
    params.edge_local_id = scene.edge_local_id.data_ptr<int>();
    params.write_point_outputs = 1;
    params.final_distance = out.distance.data_ptr<float>();
    params.final_edge_point = out.edge_point.data_ptr<float>();
    params.final_edge_t = out.edge_t.data_ptr<float>();
    params.final_shape_id = out.shape_id.data_ptr<int>();
    params.final_edge_id = out.edge_id.data_ptr<int>();
    params.final_global_edge_id = out.global_edge_id.data_ptr<int>();
    params.final_unresolved = reinterpret_cast<uint8_t *>(unresolved.data_ptr<bool>());

    launch_edge_query(
        optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Point, point_count);
    return out;
}

EdgeRayForwardOutputs edge_ray_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active) {
    require_edge_cache_dtypes(scene);
    const int64_t ray_count = ray_o.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = ray_o.options();
    auto iopts = scene.edge_v0.options();
    auto bopts = at::TensorOptions().device(ray_o.device()).dtype(at::kBool);

    EdgeRayForwardOutputs out;
    out.distance = at::empty({ray_count}, fopts);
    out.ray_t = at::empty({ray_count}, fopts);
    out.point = at::empty({ray_count, 3}, fopts);
    out.edge_t = at::empty({ray_count}, fopts);
    out.edge_point = at::empty({ray_count, 3}, fopts);
    out.shape_id = at::empty({ray_count}, iopts);
    out.edge_id = at::empty({ray_count}, iopts);
    out.global_edge_id = at::empty({ray_count}, iopts);
    out.tape_edge_id = at::empty({ray_count}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    at::Tensor unresolved = at::empty({ray_count}, bopts);
    at::Tensor active_c = active.numel() == 0 ? active : active.contiguous();
    init_edge_ray_outputs_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        active_c.numel() == 0 ? nullptr : active_c.data_ptr<bool>(),
        ray_count,
        out.distance.data_ptr<float>(),
        out.ray_t.data_ptr<float>(),
        out.point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        unresolved.data_ptr<bool>());

    if (edge_count <= 0 || ray_count <= 0 || scene.edge_accels.empty())
        return out;

    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    ensure_edge_pipeline(optix_entry);

    at::Tensor query_x = at::empty({ray_count}, fopts);
    at::Tensor query_y = at::empty({ray_count}, fopts);
    at::Tensor query_z = at::empty({ray_count}, fopts);
    at::Tensor ray_dx = at::empty({ray_count}, fopts);
    at::Tensor ray_dy = at::empty({ray_count}, fopts);
    at::Tensor ray_dz = at::empty({ray_count}, fopts);
    split_ray_vec3_to_soa_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_count,
        ray_o.stride(0),
        ray_o.stride(1),
        ray_d.stride(0),
        ray_d.stride(1),
        query_x.data_ptr<float>(),
        query_y.data_ptr<float>(),
        query_z.data_ptr<float>(),
        ray_dx.data_ptr<float>(),
        ray_dy.data_ptr<float>(),
        ray_dz.data_ptr<float>());

    at::Tensor stage_edge_id = at::empty({ray_count}, iopts);
    at::Tensor stage_distance_sq = at::empty({ray_count}, fopts);
    at::Tensor stage_ray_t = at::empty({ray_count}, fopts);
    at::Tensor stage_edge_t = at::empty({ray_count}, fopts);
    at::Tensor stage_valid = at::empty({ray_count}, bopts);

    EdgeOptixQueryParams params = {};
    fill_edge_geometry_params(scene, edge_count, params);
    params.query_x = query_x.data_ptr<float>();
    params.query_y = query_y.data_ptr<float>();
    params.query_z = query_z.data_ptr<float>();
    params.ray_dx = ray_dx.data_ptr<float>();
    params.ray_dy = ray_dy.data_ptr<float>();
    params.ray_dz = ray_dz.data_ptr<float>();
    params.ray_tmax = ray_tmax.numel() == 0 ? nullptr : ray_tmax.data_ptr<float>();
    params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data_ptr<bool>());
    params.query_count = static_cast<int>(ray_count);
    params.out_edge_ids = stage_edge_id.data_ptr<int>();
    params.out_distance_sq = stage_distance_sq.data_ptr<float>();
    params.out_ray_t = stage_ray_t.data_ptr<float>();
    params.out_edge_t = stage_edge_t.data_ptr<float>();
    params.out_valid = reinterpret_cast<uint8_t *>(stage_valid.data_ptr<bool>());

    launch_edge_query(optix_entry, torch_ctx.stream, params, EdgeOptixLaunchKind::Ray, ray_count);
    finalize_edge_ray_stage_kernel<<<blocks, threads, 0, torch_ctx.stream>>>(
        scene.edge_p0_x.data_ptr<float>(),
        scene.edge_p0_y.data_ptr<float>(),
        scene.edge_p0_z.data_ptr<float>(),
        scene.edge_e1_x.data_ptr<float>(),
        scene.edge_e1_y.data_ptr<float>(),
        scene.edge_e1_z.data_ptr<float>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        stage_edge_id.data_ptr<int>(),
        stage_distance_sq.data_ptr<float>(),
        stage_ray_t.data_ptr<float>(),
        stage_edge_t.data_ptr<float>(),
        stage_valid.data_ptr<bool>(),
        ray_count,
        out.distance.data_ptr<float>(),
        out.ray_t.data_ptr<float>(),
        out.point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        unresolved.data_ptr<bool>());
    return out;
}

} // namespace raydn
