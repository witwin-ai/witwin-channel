#include <raydn/scene/cache_kernels.h>
#include <raydn/common/math.cuh>
#include <raydn/common/optix_context.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

namespace raydn {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__forceinline__ __device__ uint64_t make_edge_key(int a, int b) {
    const uint32_t lo = static_cast<uint32_t>(a < b ? a : b);
    const uint32_t hi = static_cast<uint32_t>(a < b ? b : a);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

__forceinline__ __device__ int edge_key_v0(uint64_t key) {
    return static_cast<int>(key >> 32);
}

__forceinline__ __device__ int edge_key_v1(uint64_t key) {
    return static_cast<int>(key & 0xffffffffu);
}

__forceinline__ __device__ int candidate_face(int candidate) {
    return candidate / 3;
}

__forceinline__ __device__ int candidate_local_edge(int candidate) {
    return candidate - (candidate / 3) * 3;
}

__forceinline__ __device__ int candidate_opposite_vertex(
    const int *__restrict__ faces,
    int candidate) {
    const int face = candidate_face(candidate);
    const int local_edge = candidate_local_edge(candidate);
    const int opposite_corner = (local_edge + 2) % 3;
    return faces[face * 3 + opposite_corner];
}

__device__ int ordered_candidate(
    const int *__restrict__ candidates,
    int start,
    int count,
    int order) {
    int previous = -1;
    int selected = -1;
    for (int step = 0; step <= order; ++step) {
        int best = INT_MAX;
        for (int i = 0; i < count; ++i) {
            const int candidate = candidates[start + i];
            if (candidate > previous && candidate < best)
                best = candidate;
        }
        selected = best;
        previous = best;
    }
    return selected;
}

__forceinline__ __device__ void write_edge_topology_record(
    int out,
    int v0,
    int v1,
    int shape_id,
    int vertex_offset,
    const int *__restrict__ faces,
    int candidate0,
    int candidate1,
    int *__restrict__ edge_v0,
    int *__restrict__ edge_v1,
    int *__restrict__ edge_face0,
    int *__restrict__ edge_face1,
    int *__restrict__ edge_opposite,
    int *__restrict__ edge_shape_id,
    int *__restrict__ edge_local_id) {
    edge_v0[out] = v0;
    edge_v1[out] = v1;
    edge_face0[out] = candidate_face(candidate0);
    edge_face1[out] = candidate1 >= 0 ? candidate_face(candidate1) : -1;
    edge_opposite[out] = candidate_opposite_vertex(faces, candidate0) + vertex_offset;
    edge_shape_id[out] = shape_id;
    edge_local_id[out] = out;
}

__global__ void emit_edge_candidates_kernel(
    int face_count,
    const int *__restrict__ faces,
    uint64_t *__restrict__ keys,
    int *__restrict__ candidates) {
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= face_count)
        return;

    const int tri0 = faces[face * 3 + 0];
    const int tri1 = faces[face * 3 + 1];
    const int tri2 = faces[face * 3 + 2];
    const int tri[3] = {tri0, tri1, tri2};
    for (int local_edge = 0; local_edge < 3; ++local_edge) {
        const int out = face * 3 + local_edge;
        const int start_corner = local_edge;
        const int end_corner = (local_edge + 1) % 3;
        keys[out] = make_edge_key(tri[start_corner], tri[end_corner]);
        candidates[out] = out;
    }
}

__global__ void mark_edge_key_runs_kernel(
    int candidate_count,
    const uint64_t *__restrict__ sorted_keys,
    int *__restrict__ run_flags) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count)
        return;
    run_flags[idx] = (idx == 0 || sorted_keys[idx] != sorted_keys[idx - 1]) ? 1 : 0;
}

__global__ void fill_edge_run_starts_kernel(
    int candidate_count,
    const int *__restrict__ run_flags,
    const int *__restrict__ run_ids,
    int *__restrict__ run_starts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count || run_flags[idx] == 0)
        return;
    run_starts[run_ids[idx] - 1] = idx;
}

__global__ void compute_edge_output_counts_kernel(
    int unique_edge_count,
    int candidate_count,
    const int *__restrict__ run_starts,
    int *__restrict__ output_counts) {
    const int run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run >= unique_edge_count)
        return;
    const int start = run_starts[run];
    const int end = (run + 1 < unique_edge_count) ? run_starts[run + 1] : candidate_count;
    const int incident_count = end - start;
    output_counts[run] = (incident_count <= 1)
        ? 1
        : (incident_count * (incident_count - 1)) / 2;
}

__global__ void emit_edge_topology_outputs_kernel(
    int unique_edge_count,
    int candidate_count,
    int vertex_offset,
    int shape_id,
    const int *__restrict__ faces,
    const uint64_t *__restrict__ sorted_keys,
    const int *__restrict__ sorted_candidates,
    const int *__restrict__ run_starts,
    const int *__restrict__ output_offsets,
    int *__restrict__ edge_v0,
    int *__restrict__ edge_v1,
    int *__restrict__ edge_face0,
    int *__restrict__ edge_face1,
    int *__restrict__ edge_opposite,
    int *__restrict__ edge_shape_id,
    int *__restrict__ edge_local_id) {
    const int run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run >= unique_edge_count)
        return;

    const int start = run_starts[run];
    const int end = (run + 1 < unique_edge_count) ? run_starts[run + 1] : candidate_count;
    const int incident_count = end - start;
    const int out_start = (run == 0) ? 0 : output_offsets[run - 1];
    const uint64_t key = sorted_keys[start];
    const int v0 = edge_key_v0(key) + vertex_offset;
    const int v1 = edge_key_v1(key) + vertex_offset;

    if (incident_count <= 1) {
        const int candidate0 = sorted_candidates[start];
        write_edge_topology_record(
            out_start,
            v0,
            v1,
            shape_id,
            vertex_offset,
            faces,
            candidate0,
            -1,
            edge_v0,
            edge_v1,
            edge_face0,
            edge_face1,
            edge_opposite,
            edge_shape_id,
            edge_local_id);
        return;
    }

    int write_offset = 0;
    for (int i = 0; i < incident_count; ++i) {
        const int candidate0 = ordered_candidate(sorted_candidates, start, incident_count, i);
        for (int j = i + 1; j < incident_count; ++j) {
            const int candidate1 = ordered_candidate(sorted_candidates, start, incident_count, j);
            write_edge_topology_record(
                out_start + write_offset,
                v0,
                v1,
                shape_id,
                vertex_offset,
                faces,
                candidate0,
                candidate1,
                edge_v0,
                edge_v1,
                edge_face0,
                edge_face1,
                edge_opposite,
                edge_shape_id,
                edge_local_id);
            ++write_offset;
        }
    }
}

__global__ void compute_triangle_soa_kernel(
    int triangle_count,
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    float *__restrict__ p0_x,
    float *__restrict__ p0_y,
    float *__restrict__ p0_z,
    float *__restrict__ e1_x,
    float *__restrict__ e1_y,
    float *__restrict__ e1_z,
    float *__restrict__ e2_x,
    float *__restrict__ e2_y,
    float *__restrict__ e2_z,
    float *__restrict__ fn_x,
    float *__restrict__ fn_y,
    float *__restrict__ fn_z,
    float4 *__restrict__ p0_packed,
    float4 *__restrict__ e1_packed,
    float4 *__restrict__ e2_packed,
    float4 *__restrict__ fn_packed) {
    const int tri = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri >= triangle_count) {
        return;
    }

    const int i0 = faces[tri * 3 + 0];
    const int i1 = faces[tri * 3 + 1];
    const int i2 = faces[tri * 3 + 2];
    const float3 p0 = make_f3(vertices + i0 * 3);
    const float3 p1 = make_f3(vertices + i1 * 3);
    const float3 p2 = make_f3(vertices + i2 * 3);
    const float3 edge1 = sub3(p1, p0);
    const float3 edge2 = sub3(p2, p0);
    const float3 normal = cross3(edge1, edge2);

    p0_x[tri] = p0.x;
    p0_y[tri] = p0.y;
    p0_z[tri] = p0.z;
    e1_x[tri] = edge1.x;
    e1_y[tri] = edge1.y;
    e1_z[tri] = edge1.z;
    e2_x[tri] = edge2.x;
    e2_y[tri] = edge2.y;
    e2_z[tri] = edge2.z;
    fn_x[tri] = normal.x;
    fn_y[tri] = normal.y;
    fn_z[tri] = normal.z;
    p0_packed[tri] = make_float4(p0.x, p0.y, p0.z, 0.0f);
    e1_packed[tri] = make_float4(edge1.x, edge1.y, edge1.z, 0.0f);
    e2_packed[tri] = make_float4(edge2.x, edge2.y, edge2.z, 0.0f);
    fn_packed[tri] = make_float4(normal.x, normal.y, normal.z, 0.0f);
}

__global__ void pack_global_geometry_kernel(
    int vertex_count,
    int face_count,
    const float *__restrict__ mesh_vertices,
    const int *__restrict__ mesh_faces,
    int vertex_offset,
    int face_offset,
    int shape_id,
    float *__restrict__ global_vertices,
    int *__restrict__ global_faces,
    int *__restrict__ face_shape_id,
    int *__restrict__ face_local_id) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vertex_count) {
        const int src = idx * 3;
        const int dst = (vertex_offset + idx) * 3;
        global_vertices[dst + 0] = mesh_vertices[src + 0];
        global_vertices[dst + 1] = mesh_vertices[src + 1];
        global_vertices[dst + 2] = mesh_vertices[src + 2];
    }
    if (idx < face_count) {
        const int src = idx * 3;
        const int dst_face = face_offset + idx;
        const int dst = dst_face * 3;
        global_faces[dst + 0] = mesh_faces[src + 0] + vertex_offset;
        global_faces[dst + 1] = mesh_faces[src + 1] + vertex_offset;
        global_faces[dst + 2] = mesh_faces[src + 2] + vertex_offset;
        face_shape_id[dst_face] = shape_id;
        face_local_id[dst_face] = idx;
    }
}

__global__ void pack_global_vertex_tangent_kernel(
    int vertex_count,
    int vertex_offset,
    const float *__restrict__ mesh_tangent,
    float *__restrict__ global_tangent) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) {
        return;
    }
    const int src = idx * 3;
    const int dst = (vertex_offset + idx) * 3;
    global_tangent[dst + 0] = mesh_tangent[src + 0];
    global_tangent[dst + 1] = mesh_tangent[src + 1];
    global_tangent[dst + 2] = mesh_tangent[src + 2];
}

__global__ void zero_global_vertex_tangent_range_kernel(
    int vertex_count,
    int vertex_offset,
    float *__restrict__ global_tangent) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) {
        return;
    }
    const int dst = (vertex_offset + idx) * 3;
    global_tangent[dst + 0] = 0.0f;
    global_tangent[dst + 1] = 0.0f;
    global_tangent[dst + 2] = 0.0f;
}

__global__ void compute_edge_soa_kernel(
    int edge_count,
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    float *__restrict__ p0_x,
    float *__restrict__ p0_y,
    float *__restrict__ p0_z,
    float *__restrict__ e1_x,
    float *__restrict__ e1_y,
    float *__restrict__ e1_z) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edge_count) {
        return;
    }

    const int i0 = edge_v0[edge];
    const int i1 = edge_v1[edge];
    const float3 p0 = make_f3(vertices + i0 * 3);
    const float3 p1 = make_f3(vertices + i1 * 3);
    const float3 edge1 = sub3(p1, p0);

    p0_x[edge] = p0.x;
    p0_y[edge] = p0.y;
    p0_z[edge] = p0.z;
    e1_x[edge] = edge1.x;
    e1_y[edge] = edge1.y;
    e1_z[edge] = edge1.z;
}

__global__ void compute_edge_search_stats_kernel(
    int edge_count,
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    float *__restrict__ partials) {
    extern __shared__ float shared[];
    float *min_x = shared;
    float *min_y = min_x + blockDim.x;
    float *min_z = min_y + blockDim.x;
    float *max_x = min_z + blockDim.x;
    float *max_y = max_x + blockDim.x;
    float *max_z = max_y + blockDim.x;
    float *max_len = max_z + blockDim.x;

    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    float local_min_x = FLT_MAX;
    float local_min_y = FLT_MAX;
    float local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX;
    float local_max_y = -FLT_MAX;
    float local_max_z = -FLT_MAX;
    float local_max_len = 0.0f;
    if (edge < edge_count) {
        const float x0 = p0_x[edge];
        const float y0 = p0_y[edge];
        const float z0 = p0_z[edge];
        const float ex = e1_x[edge];
        const float ey = e1_y[edge];
        const float ez = e1_z[edge];
        const float x1 = x0 + ex;
        const float y1 = y0 + ey;
        const float z1 = z0 + ez;
        local_min_x = fminf(x0, x1);
        local_min_y = fminf(y0, y1);
        local_min_z = fminf(z0, z1);
        local_max_x = fmaxf(x0, x1);
        local_max_y = fmaxf(y0, y1);
        local_max_z = fmaxf(z0, z1);
        local_max_len = sqrtf(ex * ex + ey * ey + ez * ez);
    }

    const int lane = threadIdx.x;
    min_x[lane] = local_min_x;
    min_y[lane] = local_min_y;
    min_z[lane] = local_min_z;
    max_x[lane] = local_max_x;
    max_y[lane] = local_max_y;
    max_z[lane] = local_max_z;
    max_len[lane] = local_max_len;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            min_x[lane] = fminf(min_x[lane], min_x[lane + stride]);
            min_y[lane] = fminf(min_y[lane], min_y[lane + stride]);
            min_z[lane] = fminf(min_z[lane], min_z[lane + stride]);
            max_x[lane] = fmaxf(max_x[lane], max_x[lane + stride]);
            max_y[lane] = fmaxf(max_y[lane], max_y[lane + stride]);
            max_z[lane] = fmaxf(max_z[lane], max_z[lane + stride]);
            max_len[lane] = fmaxf(max_len[lane], max_len[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        float *out = partials + static_cast<int64_t>(blockIdx.x) * 7;
        out[0] = min_x[0];
        out[1] = min_y[0];
        out[2] = min_z[0];
        out[3] = max_x[0];
        out[4] = max_y[0];
        out[5] = max_z[0];
        out[6] = max_len[0];
    }
}

__global__ void finalize_edge_search_stats_kernel(
    int partial_count,
    const float *__restrict__ partials,
    float *__restrict__ out_stats) {
    extern __shared__ float shared[];
    float *min_x = shared;
    float *min_y = min_x + blockDim.x;
    float *min_z = min_y + blockDim.x;
    float *max_x = min_z + blockDim.x;
    float *max_y = max_x + blockDim.x;
    float *max_z = max_y + blockDim.x;
    float *max_len = max_z + blockDim.x;

    float local_min_x = FLT_MAX;
    float local_min_y = FLT_MAX;
    float local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX;
    float local_max_y = -FLT_MAX;
    float local_max_z = -FLT_MAX;
    float local_max_len = 0.0f;
    for (int block = threadIdx.x; block < partial_count; block += blockDim.x) {
        const float *row = partials + static_cast<int64_t>(block) * 7;
        local_min_x = fminf(local_min_x, row[0]);
        local_min_y = fminf(local_min_y, row[1]);
        local_min_z = fminf(local_min_z, row[2]);
        local_max_x = fmaxf(local_max_x, row[3]);
        local_max_y = fmaxf(local_max_y, row[4]);
        local_max_z = fmaxf(local_max_z, row[5]);
        local_max_len = fmaxf(local_max_len, row[6]);
    }

    const int lane = threadIdx.x;
    min_x[lane] = local_min_x;
    min_y[lane] = local_min_y;
    min_z[lane] = local_min_z;
    max_x[lane] = local_max_x;
    max_y[lane] = local_max_y;
    max_z[lane] = local_max_z;
    max_len[lane] = local_max_len;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            min_x[lane] = fminf(min_x[lane], min_x[lane + stride]);
            min_y[lane] = fminf(min_y[lane], min_y[lane + stride]);
            min_z[lane] = fminf(min_z[lane], min_z[lane + stride]);
            max_x[lane] = fmaxf(max_x[lane], max_x[lane + stride]);
            max_y[lane] = fmaxf(max_y[lane], max_y[lane + stride]);
            max_z[lane] = fmaxf(max_z[lane], max_z[lane + stride]);
            max_len[lane] = fmaxf(max_len[lane], max_len[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        out_stats[0] = min_x[0];
        out_stats[1] = min_y[0];
        out_stats[2] = min_z[0];
        out_stats[3] = max_x[0];
        out_stats[4] = max_y[0];
        out_stats[5] = max_z[0];
        out_stats[6] = max_len[0];
    }
}

void launch_require_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

} // namespace

EdgeTopology build_edge_topology_cuda(
    const at::Tensor &faces,
    int32_t vertex_offset,
    int32_t shape_id) {
    at::Tensor faces_contiguous = faces.contiguous();
    const int64_t face_count = faces_contiguous.size(0);
    launch_require_count(face_count, "build_edge_topology_cuda(face_count)");
    if (face_count > static_cast<int64_t>(std::numeric_limits<int>::max()) / 3) {
        throw std::runtime_error("build_edge_topology_cuda(): face count is too large.");
    }

    at::TensorOptions int_options = faces_contiguous.options().dtype(at::kInt);
    at::TensorOptions key_options = faces_contiguous.options().dtype(at::kLong);
    at::TensorOptions byte_options = faces_contiguous.options().dtype(at::kByte);
    auto make_empty = [&]() {
        EdgeTopology topology;
        topology.edge_v0 = at::empty({0}, int_options);
        topology.edge_v1 = at::empty({0}, int_options);
        topology.edge_face0 = at::empty({0}, int_options);
        topology.edge_face1 = at::empty({0}, int_options);
        topology.edge_opposite = at::empty({0}, int_options);
        topology.edge_shape_id = at::empty({0}, int_options);
        topology.edge_local_id = at::empty({0}, int_options);
        return topology;
    };

    if (face_count == 0) {
        return make_empty();
    }

    constexpr int block_size = 256;
    const int face_count_i = static_cast<int>(face_count);
    const int64_t candidate_count = face_count * 3;
    const int candidate_count_i = static_cast<int>(candidate_count);
    const int face_blocks = static_cast<int>((face_count + block_size - 1) / block_size);
    const int candidate_blocks =
        static_cast<int>((candidate_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cudaStream_t stream = torch_ctx.stream;

    at::Tensor keys_in = at::empty({candidate_count}, key_options);
    at::Tensor keys_out = at::empty({candidate_count}, key_options);
    at::Tensor candidates_in = at::empty({candidate_count}, int_options);
    at::Tensor candidates_out = at::empty({candidate_count}, int_options);
    auto *keys_in_ptr = reinterpret_cast<uint64_t *>(keys_in.data_ptr<int64_t>());
    auto *keys_out_ptr = reinterpret_cast<uint64_t *>(keys_out.data_ptr<int64_t>());

    emit_edge_candidates_kernel<<<face_blocks, block_size, 0, stream>>>(
        face_count_i,
        faces_contiguous.data_ptr<int>(),
        keys_in_ptr,
        candidates_in.data_ptr<int>());
    cuda_check(cudaGetLastError(), "emit_edge_candidates_kernel");

    size_t sort_temp_bytes = 0;
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_temp_bytes,
            keys_in_ptr,
            keys_out_ptr,
            candidates_in.data_ptr<int>(),
            candidates_out.data_ptr<int>(),
            candidate_count_i,
            0,
            64,
            stream),
        "cub::DeviceRadixSort::SortPairs(edge topology size)");
    at::Tensor sort_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(sort_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_temp_bytes,
            keys_in_ptr,
            keys_out_ptr,
            candidates_in.data_ptr<int>(),
            candidates_out.data_ptr<int>(),
            candidate_count_i,
            0,
            64,
            stream),
        "cub::DeviceRadixSort::SortPairs(edge topology)");

    at::Tensor run_flags = at::empty({candidate_count}, int_options);
    at::Tensor run_ids = at::empty({candidate_count}, int_options);
    mark_edge_key_runs_kernel<<<candidate_blocks, block_size, 0, stream>>>(
        candidate_count_i,
        keys_out_ptr,
        run_flags.data_ptr<int>());
    cuda_check(cudaGetLastError(), "mark_edge_key_runs_kernel");

    size_t scan_temp_bytes = 0;
    cuda_check(
        cub::DeviceScan::InclusiveSum(
            nullptr,
            scan_temp_bytes,
            run_flags.data_ptr<int>(),
            run_ids.data_ptr<int>(),
            candidate_count_i,
            stream),
        "cub::DeviceScan::InclusiveSum(edge runs size)");
    at::Tensor scan_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(scan_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceScan::InclusiveSum(
            scan_temp.data_ptr<uint8_t>(),
            scan_temp_bytes,
            run_flags.data_ptr<int>(),
            run_ids.data_ptr<int>(),
            candidate_count_i,
            stream),
        "cub::DeviceScan::InclusiveSum(edge runs)");

    int unique_edge_count = 0;
    cuda_check(
        cudaMemcpyAsync(
            &unique_edge_count,
            run_ids.data_ptr<int>() + candidate_count_i - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync(unique edge count)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(unique edge count)");
    if (unique_edge_count <= 0) {
        return make_empty();
    }
    launch_require_count(unique_edge_count, "build_edge_topology_cuda(unique_edge_count)");

    at::Tensor run_starts = at::empty({unique_edge_count}, int_options);
    at::Tensor output_counts = at::empty({unique_edge_count}, int_options);
    at::Tensor output_offsets = at::empty({unique_edge_count}, int_options);
    const int unique_blocks = (unique_edge_count + block_size - 1) / block_size;
    fill_edge_run_starts_kernel<<<candidate_blocks, block_size, 0, stream>>>(
        candidate_count_i,
        run_flags.data_ptr<int>(),
        run_ids.data_ptr<int>(),
        run_starts.data_ptr<int>());
    cuda_check(cudaGetLastError(), "fill_edge_run_starts_kernel");
    compute_edge_output_counts_kernel<<<unique_blocks, block_size, 0, stream>>>(
        unique_edge_count,
        candidate_count_i,
        run_starts.data_ptr<int>(),
        output_counts.data_ptr<int>());
    cuda_check(cudaGetLastError(), "compute_edge_output_counts_kernel");

    size_t output_scan_temp_bytes = 0;
    cuda_check(
        cub::DeviceScan::InclusiveSum(
            nullptr,
            output_scan_temp_bytes,
            output_counts.data_ptr<int>(),
            output_offsets.data_ptr<int>(),
            unique_edge_count,
            stream),
        "cub::DeviceScan::InclusiveSum(edge output offsets size)");
    if (output_scan_temp_bytes > static_cast<size_t>(scan_temp.numel())) {
        scan_temp = at::empty(
            {std::max<int64_t>(1, static_cast<int64_t>(output_scan_temp_bytes))},
            byte_options);
    }
    cuda_check(
        cub::DeviceScan::InclusiveSum(
            scan_temp.data_ptr<uint8_t>(),
            output_scan_temp_bytes,
            output_counts.data_ptr<int>(),
            output_offsets.data_ptr<int>(),
            unique_edge_count,
            stream),
        "cub::DeviceScan::InclusiveSum(edge output offsets)");

    int edge_count = 0;
    cuda_check(
        cudaMemcpyAsync(
            &edge_count,
            output_offsets.data_ptr<int>() + unique_edge_count - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync(edge topology output count)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(edge topology output count)");
    if (edge_count <= 0) {
        return make_empty();
    }
    launch_require_count(edge_count, "build_edge_topology_cuda(edge_count)");

    EdgeTopology topology;
    topology.edge_v0 = at::empty({edge_count}, int_options);
    topology.edge_v1 = at::empty({edge_count}, int_options);
    topology.edge_face0 = at::empty({edge_count}, int_options);
    topology.edge_face1 = at::empty({edge_count}, int_options);
    topology.edge_opposite = at::empty({edge_count}, int_options);
    topology.edge_shape_id = at::empty({edge_count}, int_options);
    topology.edge_local_id = at::empty({edge_count}, int_options);

    emit_edge_topology_outputs_kernel<<<unique_blocks, block_size, 0, stream>>>(
        unique_edge_count,
        candidate_count_i,
        vertex_offset,
        shape_id,
        faces_contiguous.data_ptr<int>(),
        keys_out_ptr,
        candidates_out.data_ptr<int>(),
        run_starts.data_ptr<int>(),
        output_offsets.data_ptr<int>(),
        topology.edge_v0.data_ptr<int>(),
        topology.edge_v1.data_ptr<int>(),
        topology.edge_face0.data_ptr<int>(),
        topology.edge_face1.data_ptr<int>(),
        topology.edge_opposite.data_ptr<int>(),
        topology.edge_shape_id.data_ptr<int>(),
        topology.edge_local_id.data_ptr<int>());
    cuda_check(cudaGetLastError(), "emit_edge_topology_outputs_kernel");

    return topology;
}

void pack_global_geometry_cuda(
    const at::Tensor &mesh_vertices,
    const at::Tensor &mesh_faces,
    int32_t vertex_offset,
    int32_t face_offset,
    int32_t shape_id,
    at::Tensor &global_vertices,
    at::Tensor &global_faces,
    at::Tensor &face_shape_id,
    at::Tensor &face_local_id) {
    const int64_t vertex_count = mesh_vertices.size(0);
    const int64_t face_count = mesh_faces.size(0);
    const int64_t launch_count = std::max(vertex_count, face_count);
    launch_require_count(launch_count, "pack_global_geometry_cuda()");
    if (launch_count == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((launch_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    pack_global_geometry_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(vertex_count),
        static_cast<int>(face_count),
        mesh_vertices.data_ptr<float>(),
        mesh_faces.data_ptr<int>(),
        vertex_offset,
        face_offset,
        shape_id,
        global_vertices.data_ptr<float>(),
        global_faces.data_ptr<int>(),
        face_shape_id.data_ptr<int>(),
        face_local_id.data_ptr<int>());
    cuda_check(cudaGetLastError(), "pack_global_geometry_kernel");
}

void pack_global_vertex_tangent_cuda(
    const at::Tensor &mesh_tangent,
    int64_t vertex_offset,
    int64_t vertex_count,
    at::Tensor &global_tangent) {
    launch_require_count(vertex_count, "pack_global_vertex_tangent_cuda()");
    if (vertex_count == 0) {
        return;
    }
    if (vertex_offset < 0 ||
        vertex_offset > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        vertex_count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("pack_global_vertex_tangent_cuda(): vertex range exceeds int32.");
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((vertex_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    pack_global_vertex_tangent_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(vertex_count),
        static_cast<int>(vertex_offset),
        mesh_tangent.data_ptr<float>(),
        global_tangent.data_ptr<float>());
    cuda_check(cudaGetLastError(), "pack_global_vertex_tangent_kernel");
}

void zero_global_vertex_tangent_range_cuda(
    int64_t vertex_offset,
    int64_t vertex_count,
    at::Tensor &global_tangent) {
    launch_require_count(vertex_count, "zero_global_vertex_tangent_range_cuda()");
    if (vertex_count == 0) {
        return;
    }
    if (vertex_offset < 0 ||
        vertex_offset > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        vertex_count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("zero_global_vertex_tangent_range_cuda(): vertex range exceeds int32.");
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((vertex_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    zero_global_vertex_tangent_range_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(vertex_count),
        static_cast<int>(vertex_offset),
        global_tangent.data_ptr<float>());
    cuda_check(cudaGetLastError(), "zero_global_vertex_tangent_range_kernel");
}

void compute_triangle_soa_cuda(
    int64_t triangle_count,
    const at::Tensor &vertices,
    const at::Tensor &faces,
    at::Tensor &tri_p0_x,
    at::Tensor &tri_p0_y,
    at::Tensor &tri_p0_z,
    at::Tensor &tri_e1_x,
    at::Tensor &tri_e1_y,
    at::Tensor &tri_e1_z,
    at::Tensor &tri_e2_x,
    at::Tensor &tri_e2_y,
    at::Tensor &tri_e2_z,
    at::Tensor &tri_fn_x,
    at::Tensor &tri_fn_y,
    at::Tensor &tri_fn_z,
    at::Tensor &tri_p0_packed,
    at::Tensor &tri_e1_packed,
    at::Tensor &tri_e2_packed,
    at::Tensor &tri_fn_packed) {
    launch_require_count(triangle_count, "compute_triangle_soa_cuda()");
    if (triangle_count == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((triangle_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_triangle_soa_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(triangle_count),
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        tri_p0_x.data_ptr<float>(),
        tri_p0_y.data_ptr<float>(),
        tri_p0_z.data_ptr<float>(),
        tri_e1_x.data_ptr<float>(),
        tri_e1_y.data_ptr<float>(),
        tri_e1_z.data_ptr<float>(),
        tri_e2_x.data_ptr<float>(),
        tri_e2_y.data_ptr<float>(),
        tri_e2_z.data_ptr<float>(),
        tri_fn_x.data_ptr<float>(),
        tri_fn_y.data_ptr<float>(),
        tri_fn_z.data_ptr<float>(),
        reinterpret_cast<float4 *>(tri_p0_packed.data_ptr<float>()),
        reinterpret_cast<float4 *>(tri_e1_packed.data_ptr<float>()),
        reinterpret_cast<float4 *>(tri_e2_packed.data_ptr<float>()),
        reinterpret_cast<float4 *>(tri_fn_packed.data_ptr<float>()));
}

void compute_edge_soa_cuda(
    int64_t edge_count,
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    at::Tensor &edge_p0_x,
    at::Tensor &edge_p0_y,
    at::Tensor &edge_p0_z,
    at::Tensor &edge_e1_x,
    at::Tensor &edge_e1_y,
    at::Tensor &edge_e1_z) {
    launch_require_count(edge_count, "compute_edge_soa_cuda()");
    if (edge_count == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((edge_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_edge_soa_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(edge_count),
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>());
}

EdgeSearchStats compute_edge_search_stats_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z) {
    launch_require_count(edge_count, "compute_edge_search_stats_cuda()");
    EdgeSearchStats stats;
    if (edge_count == 0) {
        return stats;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((edge_count + block_size - 1) / block_size);
    at::Tensor partials = at::empty({block_count, 7}, edge_p0_x.options());
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_edge_search_stats_kernel<<<
        block_count,
        block_size,
        sizeof(float) * block_size * 7,
        torch_ctx.stream>>>(
        static_cast<int>(edge_count),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>(),
        partials.data_ptr<float>());
    cuda_check(cudaGetLastError(), "compute_edge_search_stats_kernel");

    at::Tensor stats_gpu = at::empty({7}, edge_p0_x.options());
    constexpr int finalize_block_size = 256;
    finalize_edge_search_stats_kernel<<<
        1,
        finalize_block_size,
        sizeof(float) * finalize_block_size * 7,
        torch_ctx.stream>>>(
        block_count,
        partials.data_ptr<float>(),
        stats_gpu.data_ptr<float>());
    cuda_check(cudaGetLastError(), "finalize_edge_search_stats_kernel");

    at::Tensor stats_cpu = stats_gpu.cpu();
    const float *values = stats_cpu.data_ptr<float>();
    stats.has_edges = true;
    stats.min_x = values[0];
    stats.min_y = values[1];
    stats.min_z = values[2];
    stats.max_x = values[3];
    stats.max_y = values[4];
    stats.max_z = values[5];
    stats.max_edge_length = values[6];
    return stats;
}

} // namespace raydn
