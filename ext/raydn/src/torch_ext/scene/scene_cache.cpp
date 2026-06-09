#include <raydn/scene/cache.h>
#include <raydn/common/optix_context.h>
#include <raydn/common/tensor_check.h>
#include <raydn/edge/bvh.h>
#include <raydn/scene/cache_kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace raydn {

namespace {
std::atomic<int64_t> next_handle{1};
std::mutex scenes_mutex;
std::unordered_map<int64_t, std::unique_ptr<SceneCache>> scenes;

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_optional_matrix4(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 4, name);
    if (tensor.size(0) != 0 && tensor.size(0) != 4)
        throw std::runtime_error(std::string(name) + " must be empty or have shape (4, 4).");
}

void compact_accel_if_smaller(
    OptixDeviceContext optix_context,
    cudaStream_t stream,
    at::TensorOptions byte_options,
    at::Tensor &gas_buffer,
    OptixTraversableHandle &traversable,
    const at::Tensor &compacted_size_buffer,
    const char *name) {
    uint64_t compacted_size = 0;
    cuda_check(
        cudaMemcpyAsync(
            &compacted_size,
            compacted_size_buffer.data_ptr<uint8_t>(),
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync(compacted GAS size)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(compacted GAS size)");
    if (compacted_size == 0 || compacted_size >= static_cast<uint64_t>(gas_buffer.numel()))
        return;
    if (compacted_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string(name) + ": compacted GAS size exceeds int64 range.");
    }

    at::Tensor source_buffer = gas_buffer;
    at::Tensor compacted_buffer =
        at::empty({static_cast<int64_t>(compacted_size)}, byte_options);
    OptixTraversableHandle compacted_traversable = 0;
    raydn_OPTIX_CHECK(optixAccelCompact(
        optix_context,
        stream,
        traversable,
        reinterpret_cast<CUdeviceptr>(compacted_buffer.data_ptr<uint8_t>()),
        static_cast<size_t>(compacted_size),
        &compacted_traversable));
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(GAS compaction)");
    gas_buffer = compacted_buffer;
    traversable = compacted_traversable;
}

OptixTriangleAccel build_triangle_accel(
    const MeshRecord &mesh,
    OptixDeviceContext optix_context,
    cudaStream_t stream) {
    OptixTriangleAccel accel;
    accel.vertex_buffer = mesh.vertices.contiguous();
    accel.index_buffer = mesh.faces.contiguous();

    CUdeviceptr vertex_buffer =
        reinterpret_cast<CUdeviceptr>(accel.vertex_buffer.data_ptr<float>());
    CUdeviceptr index_buffer =
        reinterpret_cast<CUdeviceptr>(accel.index_buffer.data_ptr<int>());
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.numVertices =
        static_cast<unsigned int>(accel.vertex_buffer.size(0));
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = index_buffer;
    build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(accel.index_buffer.size(0));
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (mesh.dynamic)
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    else
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes = {};
    raydn_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    at::TensorOptions byte_options =
        at::TensorOptions().device(mesh.vertices.device()).dtype(at::kByte);
    accel.gas_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    accel.gas_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
    at::Tensor compacted_size_buffer;
    OptixAccelEmitDesc compacted_size_emit = {};
    OptixAccelEmitDesc *emit_descs = nullptr;
    unsigned int emit_desc_count = 0;
    if (!mesh.dynamic) {
        compacted_size_buffer = at::empty({static_cast<int64_t>(sizeof(uint64_t))}, byte_options);
        compacted_size_emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        compacted_size_emit.result =
            reinterpret_cast<CUdeviceptr>(compacted_size_buffer.data_ptr<uint8_t>());
        emit_descs = &compacted_size_emit;
        emit_desc_count = 1;
    }

    raydn_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
        buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
        buffer_sizes.outputSizeInBytes,
        &accel.traversable,
        emit_descs,
        emit_desc_count));
    if (!mesh.dynamic) {
        compact_accel_if_smaller(
            optix_context,
            stream,
            byte_options,
            accel.gas_buffer,
            accel.traversable,
            compacted_size_buffer,
            "build_triangle_accel()");
    }

    return accel;
}

void write_identity_instance(OptixInstance &instance, unsigned int instance_id, OptixTraversableHandle traversable) {
    std::memset(&instance, 0, sizeof(instance));
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.instanceId = instance_id;
    instance.sbtOffset = 0;
    instance.visibilityMask = 255u;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = traversable;
}

void build_triangle_ias(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    if (scene.triangle_accels.empty())
        throw std::runtime_error("build_triangle_ias(): missing triangle acceleration structures.");

    std::vector<OptixInstance> instances(scene.triangle_accels.size());
    for (size_t mesh_index = 0; mesh_index < scene.triangle_accels.size(); ++mesh_index) {
        write_identity_instance(
            instances[mesh_index],
            static_cast<unsigned int>(mesh_index),
            scene.triangle_accels[mesh_index].traversable);
    }

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, scene.device_index)).dtype(at::kByte);
    scene.triangle_ias.instance_buffer =
        at::empty({static_cast<int64_t>(sizeof(OptixInstance) * instances.size())}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            scene.triangle_ias.instance_buffer.data_ptr<uint8_t>(),
            instances.data(),
            sizeof(OptixInstance) * instances.size(),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(triangle IAS instances)");

    CUdeviceptr instance_buffer =
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.instance_buffer.data_ptr<uint8_t>());
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = instance_buffer;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes = {};
    raydn_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    scene.triangle_ias.ias_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    scene.triangle_ias.ias_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);

    raydn_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.ias_temp_buffer.data_ptr<uint8_t>()),
        buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.ias_buffer.data_ptr<uint8_t>()),
        buffer_sizes.outputSizeInBytes,
        &scene.triangle_ias.traversable,
        nullptr,
        0));
}

void refresh_global_geometry(SceneCache &scene) {
    int64_t vertex_offset = 0;
    int64_t face_offset = 0;
    std::vector<int32_t> face_offsets;
    face_offsets.reserve(scene.meshes.size());
    for (size_t mesh_id = 0; mesh_id < scene.meshes.size(); ++mesh_id) {
        const MeshRecord &mesh = scene.meshes[mesh_id];
        if (vertex_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
            face_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Scene.build(): geometry exceeds int32 indexing limits.");
        }
        face_offsets.push_back(static_cast<int32_t>(face_offset));
        vertex_offset += mesh.vertices.size(0);
        face_offset += mesh.faces.size(0);
    }
    if (vertex_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
        face_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Scene.build(): geometry exceeds int32 indexing limits.");
    }

    at::TensorOptions fopts = scene.meshes[0].vertices.options();
    at::TensorOptions iopts = scene.meshes[0].faces.options();
    scene.global_vertices = at::empty({vertex_offset, 3}, fopts);
    scene.global_faces = at::empty({face_offset, 3}, iopts);
    scene.face_shape_id = at::empty({face_offset}, iopts);
    scene.face_local_id = at::empty({face_offset}, iopts);
    scene.face_offsets = at::empty({static_cast<int64_t>(face_offsets.size())}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cuda_check(
        cudaMemcpyAsync(
            scene.face_offsets.data_ptr<int>(),
            face_offsets.data(),
            sizeof(int32_t) * face_offsets.size(),
            cudaMemcpyHostToDevice,
            torch_ctx.stream),
        "cudaMemcpyAsync(face offsets)");

    vertex_offset = 0;
    face_offset = 0;
    for (int32_t mesh_id = 0; mesh_id < static_cast<int32_t>(scene.meshes.size()); ++mesh_id) {
        const MeshRecord &mesh = scene.meshes[mesh_id];
        pack_global_geometry_cuda(
            mesh.vertices,
            mesh.faces,
            static_cast<int32_t>(vertex_offset),
            static_cast<int32_t>(face_offset),
            mesh_id,
            scene.global_vertices,
            scene.global_faces,
            scene.face_shape_id,
            scene.face_local_id);
        vertex_offset += mesh.vertices.size(0);
        face_offset += mesh.faces.size(0);
    }

    const int64_t triangle_count = scene.global_faces.size(0);
    scene.tri_p0_x = at::empty({triangle_count}, fopts);
    scene.tri_p0_y = at::empty({triangle_count}, fopts);
    scene.tri_p0_z = at::empty({triangle_count}, fopts);
    scene.tri_e1_x = at::empty({triangle_count}, fopts);
    scene.tri_e1_y = at::empty({triangle_count}, fopts);
    scene.tri_e1_z = at::empty({triangle_count}, fopts);
    scene.tri_e2_x = at::empty({triangle_count}, fopts);
    scene.tri_e2_y = at::empty({triangle_count}, fopts);
    scene.tri_e2_z = at::empty({triangle_count}, fopts);
    scene.tri_fn_x = at::empty({triangle_count}, fopts);
    scene.tri_fn_y = at::empty({triangle_count}, fopts);
    scene.tri_fn_z = at::empty({triangle_count}, fopts);
    scene.tri_p0_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_e1_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_e2_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_fn_packed = at::empty({triangle_count, 4}, fopts);
    compute_triangle_soa_cuda(
        triangle_count,
        scene.global_vertices,
        scene.global_faces,
        scene.tri_p0_x,
        scene.tri_p0_y,
        scene.tri_p0_z,
        scene.tri_e1_x,
        scene.tri_e1_y,
        scene.tri_e1_z,
        scene.tri_e2_x,
        scene.tri_e2_y,
        scene.tri_e2_z,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.tri_p0_packed,
        scene.tri_e1_packed,
        scene.tri_e2_packed,
        scene.tri_fn_packed);
}

void update_triangle_accel(
    const MeshRecord &mesh,
    OptixTriangleAccel &accel,
    OptixDeviceContext optix_context,
    cudaStream_t stream) {
    accel.vertex_buffer = mesh.vertices.contiguous();
    CUdeviceptr vertex_buffer =
        reinterpret_cast<CUdeviceptr>(accel.vertex_buffer.data_ptr<float>());
    CUdeviceptr index_buffer =
        reinterpret_cast<CUdeviceptr>(accel.index_buffer.data_ptr<int>());
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.numVertices =
        static_cast<unsigned int>(accel.vertex_buffer.size(0));
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = index_buffer;
    build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(accel.index_buffer.size(0));
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes buffer_sizes = {};
    raydn_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    at::TensorOptions byte_options =
        at::TensorOptions().device(mesh.vertices.device()).dtype(at::kByte);
    size_t temp_bytes = buffer_sizes.tempUpdateSizeInBytes;
    if (accel.gas_buffer.numel() < static_cast<int64_t>(buffer_sizes.outputSizeInBytes)) {
        accel.gas_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        temp_bytes = buffer_sizes.tempSizeInBytes;
    }
    if (accel.gas_temp_buffer.numel() < static_cast<int64_t>(temp_bytes))
        accel.gas_temp_buffer = at::empty({static_cast<int64_t>(temp_bytes)}, byte_options);

    raydn_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
        temp_bytes,
        reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
        static_cast<size_t>(accel.gas_buffer.numel()),
        &accel.traversable,
        nullptr,
        0));
}

bool scene_has_dynamic_edges(const SceneCache &scene) {
    for (const MeshRecord &mesh : scene.meshes) {
        if (mesh.dynamic && mesh.edges_enabled)
            return true;
    }
    return false;
}

void build_edge_topology(SceneCache &scene) {
    std::vector<at::Tensor> edge_v0_parts;
    std::vector<at::Tensor> edge_v1_parts;
    std::vector<at::Tensor> edge_face0_parts;
    std::vector<at::Tensor> edge_face1_parts;
    std::vector<at::Tensor> edge_opposite_parts;
    std::vector<at::Tensor> edge_shape_id_parts;
    std::vector<at::Tensor> edge_local_id_parts;
    edge_v0_parts.reserve(scene.meshes.size());
    edge_v1_parts.reserve(scene.meshes.size());
    edge_face0_parts.reserve(scene.meshes.size());
    edge_face1_parts.reserve(scene.meshes.size());
    edge_opposite_parts.reserve(scene.meshes.size());
    edge_shape_id_parts.reserve(scene.meshes.size());
    edge_local_id_parts.reserve(scene.meshes.size());

    int32_t vertex_offset = 0;
    for (int32_t shape_id = 0; shape_id < static_cast<int32_t>(scene.meshes.size()); ++shape_id) {
        const MeshRecord &mesh = scene.meshes[shape_id];
        if (mesh.edges_enabled) {
            EdgeTopology topology = build_edge_topology_cuda(mesh.faces, vertex_offset, shape_id);
            if (topology.edge_v0.numel() > 0) {
                edge_v0_parts.push_back(topology.edge_v0);
                edge_v1_parts.push_back(topology.edge_v1);
                edge_face0_parts.push_back(topology.edge_face0);
                edge_face1_parts.push_back(topology.edge_face1);
                edge_opposite_parts.push_back(topology.edge_opposite);
                edge_shape_id_parts.push_back(topology.edge_shape_id);
                edge_local_id_parts.push_back(topology.edge_local_id);
            }
        }
        vertex_offset += static_cast<int32_t>(mesh.vertices.size(0));
    }

    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions iopts = at::TensorOptions().device(device).dtype(at::kInt);
    auto cat_or_empty = [&](std::vector<at::Tensor> &parts) {
        if (parts.empty())
            return at::empty({0}, iopts);
        return at::cat(parts, 0).contiguous();
    };
    scene.edge_v0 = cat_or_empty(edge_v0_parts);
    scene.edge_v1 = cat_or_empty(edge_v1_parts);
    scene.edge_face0 = cat_or_empty(edge_face0_parts);
    scene.edge_face1 = cat_or_empty(edge_face1_parts);
    scene.edge_opposite = cat_or_empty(edge_opposite_parts);
    scene.edge_shape_id = cat_or_empty(edge_shape_id_parts);
    scene.edge_local_id = cat_or_empty(edge_local_id_parts);
}

std::vector<float> compute_edge_search_radii(
    const EdgeSearchStats &stats) {
    if (!stats.has_edges)
        return {};

    const float dx = std::max(stats.max_x - stats.min_x, 0.0f);
    const float dy = std::max(stats.max_y - stats.min_y, 0.0f);
    const float dz = std::max(stats.max_z - stats.min_z, 0.0f);
    const float full_radius = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3f);
    const float edge_scale = std::max(stats.max_edge_length, full_radius * 1.0e-4f);

    std::vector<float> radii;
    radii.reserve(3);
    auto add_radius = [&](float radius) {
        if (std::isfinite(radius) && radius > 0.0f)
            radii.push_back(std::min(std::max(radius, 1.0e-5f), full_radius));
    };
    add_radius(edge_scale * 4.0f);
    add_radius(edge_scale * 34.0f);
    add_radius(full_radius);

    std::sort(radii.begin(), radii.end());
    std::vector<float> unique_radii;
    unique_radii.reserve(radii.size());
    for (float radius : radii) {
        if (unique_radii.empty() || radius > unique_radii.back() * 1.01f + 1.0e-6f)
            unique_radii.push_back(radius);
    }
    if (unique_radii.empty() || unique_radii.back() < full_radius * 0.999f)
        unique_radii.push_back(full_radius);
    else
        unique_radii.back() = full_radius;
    return unique_radii;
}

void refresh_edge_soa(SceneCache &scene) {
    const int64_t edge_count = scene.edge_v0.size(0);
    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions fopts = at::TensorOptions().device(device).dtype(at::kFloat);
    scene.edge_p0_x = at::empty({edge_count}, fopts);
    scene.edge_p0_y = at::empty({edge_count}, fopts);
    scene.edge_p0_z = at::empty({edge_count}, fopts);
    scene.edge_e1_x = at::empty({edge_count}, fopts);
    scene.edge_e1_y = at::empty({edge_count}, fopts);
    scene.edge_e1_z = at::empty({edge_count}, fopts);
    scene.edge_mask = at::ones({edge_count}, at::TensorOptions().device(device).dtype(at::kByte));
    compute_edge_soa_cuda(
        edge_count,
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
}

void build_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    refresh_edge_soa(scene);
    scene.edge_accels.clear();
    if (edge_count == 0) {
        scene.edge_accel = {};
        return;
    }

    const EdgeSearchStats stats = compute_edge_search_stats_cuda(
        edge_count,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
    std::vector<float> radii = compute_edge_search_radii(stats);
    scene.edge_accels.resize(radii.size());
    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions byte_options = at::TensorOptions().device(device).dtype(at::kByte);
    at::TensorOptions float_options = at::TensorOptions().device(device).dtype(at::kFloat);
    const bool compact_static_edges = !scene_has_dynamic_edges(scene);

    for (size_t gas_index = 0; gas_index < radii.size(); ++gas_index) {
        OptixEdgeAccel &accel = scene.edge_accels[gas_index];
        const float radius = radii[gas_index];
        accel.aabb_buffer = at::empty({edge_count, 6}, float_options);
        compute_edge_optix_aabbs_cuda(
            edge_count,
            scene.edge_p0_x,
            scene.edge_p0_y,
            scene.edge_p0_z,
            scene.edge_e1_x,
            scene.edge_e1_y,
            scene.edge_e1_z,
            radius,
            accel.aabb_buffer);

        CUdeviceptr aabb_buffer =
            reinterpret_cast<CUdeviceptr>(accel.aabb_buffer.data_ptr<float>());
        uint32_t input_flags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(edge_count);
        build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6;
        build_input.customPrimitiveArray.flags = &input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        if (compact_static_edges)
            accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes buffer_sizes = {};
        raydn_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, &accel_options, &build_input, 1, &buffer_sizes));

        accel.gas_temp_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
        accel.gas_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
        at::Tensor compacted_size_buffer;
        OptixAccelEmitDesc compacted_size_emit = {};
        OptixAccelEmitDesc *emit_descs = nullptr;
        unsigned int emit_desc_count = 0;
        if (compact_static_edges) {
            compacted_size_buffer = at::empty({static_cast<int64_t>(sizeof(uint64_t))}, byte_options);
            compacted_size_emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            compacted_size_emit.result =
                reinterpret_cast<CUdeviceptr>(compacted_size_buffer.data_ptr<uint8_t>());
            emit_descs = &compacted_size_emit;
            emit_desc_count = 1;
        }
        raydn_OPTIX_CHECK(optixAccelBuild(
            optix_context,
            stream,
            &accel_options,
            &build_input,
            1,
            reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
            buffer_sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
            buffer_sizes.outputSizeInBytes,
            &accel.traversable,
            emit_descs,
            emit_desc_count));
        if (compact_static_edges) {
            compact_accel_if_smaller(
                optix_context,
                stream,
                byte_options,
                accel.gas_buffer,
                accel.traversable,
                compacted_size_buffer,
                "build_edge_accel()");
        }
        accel.search_radius = radius;
    }
    scene.edge_accel = scene.edge_accels.back();
}

bool update_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    if (edge_count == 0) {
        scene.edge_accels.clear();
        scene.edge_accel = {};
        return true;
    }
    if (!scene_has_dynamic_edges(scene) || scene.edge_accels.empty())
        return false;

    refresh_edge_soa(scene);
    const EdgeSearchStats stats = compute_edge_search_stats_cuda(
        edge_count,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
    std::vector<float> radii = compute_edge_search_radii(stats);
    if (radii.size() != scene.edge_accels.size())
        return false;

    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions byte_options = at::TensorOptions().device(device).dtype(at::kByte);
    for (size_t gas_index = 0; gas_index < radii.size(); ++gas_index) {
        OptixEdgeAccel &accel = scene.edge_accels[gas_index];
        if (!accel.aabb_buffer.defined() || accel.aabb_buffer.size(0) != edge_count ||
            !accel.gas_buffer.defined()) {
            return false;
        }

        const float radius = radii[gas_index];
        compute_edge_optix_aabbs_cuda(
            edge_count,
            scene.edge_p0_x,
            scene.edge_p0_y,
            scene.edge_p0_z,
            scene.edge_e1_x,
            scene.edge_e1_y,
            scene.edge_e1_z,
            radius,
            accel.aabb_buffer);

        CUdeviceptr aabb_buffer =
            reinterpret_cast<CUdeviceptr>(accel.aabb_buffer.data_ptr<float>());
        uint32_t input_flags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(edge_count);
        build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6;
        build_input.customPrimitiveArray.flags = &input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes buffer_sizes = {};
        raydn_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, &accel_options, &build_input, 1, &buffer_sizes));
        const size_t temp_bytes = buffer_sizes.tempSizeInBytes;
        if (accel.gas_buffer.numel() < static_cast<int64_t>(buffer_sizes.outputSizeInBytes))
            return false;
        if (accel.gas_temp_buffer.numel() < static_cast<int64_t>(temp_bytes))
            accel.gas_temp_buffer = at::empty({static_cast<int64_t>(temp_bytes)}, byte_options);

        raydn_OPTIX_CHECK(optixAccelBuild(
            optix_context,
            stream,
            &accel_options,
            &build_input,
            1,
            reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
            temp_bytes,
            reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
            static_cast<size_t>(accel.gas_buffer.numel()),
            &accel.traversable,
            nullptr,
            0));
        accel.search_radius = radius;
    }
    scene.edge_accel = scene.edge_accels.back();
    return true;
}
} // namespace

SceneHandle::~SceneHandle() {
    if (owns_handle && handle != 0)
        destroy_scene(handle);
}

std::unique_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes) {
    if (meshes.empty())
        throw std::runtime_error("Scene.build(): at least one mesh is required.");

    const int64_t device_index = meshes[0].vertices.get_device();
    c10::cuda::CUDAGuard guard(static_cast<int>(device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != device_index)
        throw std::runtime_error("Scene.build(): current CUDA device does not match mesh tensors.");
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(device_index));

    for (const MeshRecord &mesh : meshes) {
        require_vec3f(mesh.vertices, "mesh.vertices");
        require_vec3i(mesh.faces, "mesh.faces");
        require_optional_matrix4(mesh.to_world_left, "mesh.to_world_left");
        require_optional_matrix4(mesh.to_world_right, "mesh.to_world_right");
        if (mesh.vertices.get_device() != device_index || mesh.faces.get_device() != device_index)
            throw std::runtime_error("Scene.build(): all tensors must be on the same CUDA device.");
        if (mesh.to_world_left.get_device() != device_index || mesh.to_world_right.get_device() != device_index)
            throw std::runtime_error("Scene.build(): transform tensors must be on the scene device.");
    }

    auto scene_unique = std::make_unique<SceneCache>();
    SceneCache *scene = scene_unique.get();
    scene->handle = next_handle.fetch_add(1);
    scene->device_index = device_index;
    scene->meshes = std::move(meshes);
    refresh_global_geometry(*scene);
    scene->triangle_accels.reserve(scene->meshes.size());
    for (const MeshRecord &mesh : scene->meshes)
        scene->triangle_accels.push_back(
            build_triangle_accel(mesh, optix_entry.optix_context, torch_ctx.stream));
    build_triangle_ias(*scene, optix_entry.optix_context, torch_ctx.stream);
    build_edge_topology(*scene);
    build_edge_accel(*scene, optix_entry.optix_context, torch_ctx.stream);
    return scene_unique;
}

int64_t create_scene(std::vector<MeshRecord> meshes) {
    auto scene = create_scene_cache(std::move(meshes));
    const int64_t handle = scene->handle;
    std::lock_guard<std::mutex> lock(scenes_mutex);
    scenes.emplace(handle, std::move(scene));
    return handle;
}

void destroy_scene(int64_t handle) {
    if (handle == 0)
        return;
    std::lock_guard<std::mutex> lock(scenes_mutex);
    scenes.erase(handle);
}

SceneCache &get_scene(int64_t handle) {
    std::lock_guard<std::mutex> lock(scenes_mutex);
    auto it = scenes.find(handle);
    if (it == scenes.end())
        throw std::runtime_error("Invalid RayDN scene handle.");
    return *it->second;
}

int64_t scene_version(int64_t handle) {
    return get_scene(handle).version;
}

int64_t scene_num_meshes(int64_t handle) {
    return static_cast<int64_t>(get_scene(handle).meshes.size());
}

int64_t scene_edge_count(int64_t handle) {
    return get_scene(handle).edge_v0.size(0);
}

void update_mesh_vertices(int64_t handle, int64_t mesh_id, at::Tensor vertices) {
    SceneCache &scene = get_scene(handle);
    if (mesh_id < 0 || mesh_id >= static_cast<int64_t>(scene.meshes.size()))
        throw std::runtime_error("update_mesh_vertices(): invalid mesh id.");
    MeshRecord &mesh = scene.meshes[mesh_id];
    if (!mesh.dynamic)
        throw std::runtime_error("update_mesh_vertices(): target mesh is not dynamic.");
    require_vec3f(vertices, "vertices");
    if (vertices.get_device() != scene.device_index)
        throw std::runtime_error("update_mesh_vertices(): vertices must stay on the scene device.");
    if (vertices.size(0) != mesh.vertices.size(0))
        throw std::runtime_error("update_mesh_vertices(): vertex count must stay unchanged.");
    mesh.vertices = vertices.contiguous();
    mesh.pending_update = true;
}

void sync_scene(int64_t handle) {
    SceneCache &scene = get_scene(handle);
    c10::cuda::CUDAGuard guard(static_cast<int>(scene.device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != scene.device_index)
        throw std::runtime_error("Scene.sync(): current CUDA device does not match scene tensors.");
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));

    bool changed = false;
    for (int64_t mesh_id = 0; mesh_id < static_cast<int64_t>(scene.meshes.size()); ++mesh_id) {
        MeshRecord &mesh = scene.meshes[mesh_id];
        if (!mesh.pending_update)
            continue;
        update_triangle_accel(
            mesh,
            scene.triangle_accels[mesh_id],
            optix_entry.optix_context,
            torch_ctx.stream);
        mesh.pending_update = false;
        changed = true;
    }
    if (changed) {
        refresh_global_geometry(scene);
        build_triangle_ias(scene, optix_entry.optix_context, torch_ctx.stream);
        if (!update_edge_accel(scene, optix_entry.optix_context, torch_ctx.stream))
            build_edge_accel(scene, optix_entry.optix_context, torch_ctx.stream);
        scene.version += 1;
        scene.edge_version += 1;
    }
}

int64_t scene_version(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_version(scene->handle);
}

int64_t scene_num_meshes(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_num_meshes(scene->handle);
}

int64_t scene_edge_count(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_edge_count(scene->handle);
}

void update_mesh_vertices(c10::intrusive_ptr<SceneHandle> scene, int64_t mesh_id, at::Tensor vertices) {
    update_mesh_vertices(scene->handle, mesh_id, std::move(vertices));
}

void sync_scene(c10::intrusive_ptr<SceneHandle> scene) {
    sync_scene(scene->handle);
}

} // namespace raydn
