#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <vector>

namespace {

constexpr int kDiffractionBlockSize = 256;

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

__global__ void diffraction_state_wi_kernel(
    const float *__restrict__ state_edge_pos,
    const float *__restrict__ state_src,
    float *__restrict__ state_wi,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const float *edge_pos = state_edge_pos + state * 3;
        const float *src = state_src + state * 3;
        float dx = edge_pos[0] - src[0];
        float dy = edge_pos[1] - src[1];
        float dz = edge_pos[2] - src[2];
        const float norm = sqrtf(dx * dx + dy * dy + dz * dz);
        const float scale = norm > 1.0e-6f ? 1.0f / norm : 1.0e6f;

        float *out = state_wi + state * 3;
        out[0] = dx * scale;
        out[1] = dy * scale;
        out[2] = dz * scale;
    }
}

__global__ void diffraction_state_pack_kernel(
    const int *__restrict__ edge_indices,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ line_min,
    const float *__restrict__ line_max,
    const float *__restrict__ n0,
    const float *__restrict__ n1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const float *__restrict__ exterior_angle,
    const float *__restrict__ tx,
    const float *__restrict__ tx_power,
    int *__restrict__ state_edge_index,
    float *__restrict__ state_edge_pos,
    float *__restrict__ state_edge_dir,
    float *__restrict__ state_line_min,
    float *__restrict__ state_line_max,
    float *__restrict__ state_n0,
    float *__restrict__ state_n1,
    int *__restrict__ state_face0,
    int *__restrict__ state_face1,
    float *__restrict__ state_exterior_angle,
    float *__restrict__ state_src,
    float *__restrict__ state_src_power,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float tx_x = tx[0];
    const float tx_y = tx[1];
    const float tx_z = tx[2];
    const float power = tx_power[0];
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const int edge = edge_indices[state];
        state_edge_index[state] = edge;
        state_line_min[state] = line_min[edge];
        state_line_max[state] = line_max[edge];
        state_face0[state] = face0[edge];
        state_face1[state] = face1[edge];
        state_exterior_angle[state] = exterior_angle[edge];
        state_src_power[state] = power;

        const int64_t edge_base = static_cast<int64_t>(edge) * 3;
        const int64_t state_base = state * 3;
        state_edge_pos[state_base + 0] = edge_pos[edge_base + 0];
        state_edge_pos[state_base + 1] = edge_pos[edge_base + 1];
        state_edge_pos[state_base + 2] = edge_pos[edge_base + 2];
        state_edge_dir[state_base + 0] = edge_dir[edge_base + 0];
        state_edge_dir[state_base + 1] = edge_dir[edge_base + 1];
        state_edge_dir[state_base + 2] = edge_dir[edge_base + 2];
        state_n0[state_base + 0] = n0[edge_base + 0];
        state_n0[state_base + 1] = n0[edge_base + 1];
        state_n0[state_base + 2] = n0[edge_base + 2];
        state_n1[state_base + 0] = n1[edge_base + 0];
        state_n1[state_base + 1] = n1[edge_base + 1];
        state_n1[state_base + 2] = n1[edge_base + 2];
        state_src[state_base + 0] = tx_x;
        state_src[state_base + 1] = tx_y;
        state_src[state_base + 2] = tx_z;
    }
}

__device__ __forceinline__ float3 load_vec3(const float *data, int64_t index) {
    const float *ptr = data + index * 3;
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ __forceinline__ void store_vec3(float *data, int64_t index, float3 value) {
    float *ptr = data + index * 3;
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    const float xz = __fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.z, b.z));
    return __fadd_rn(xz, __fmul_rn(a.y, b.y));
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float norm3(float3 value) {
    return sqrtf(dot3(value, value));
}

__device__ __forceinline__ float3 normalize3(float3 value, float eps) {
    const float norm = norm3(value);
    const float denom = fmaxf(norm, eps);
    return make_float3(
        __fdiv_rn(value.x, denom),
        __fdiv_rn(value.y, denom),
        __fdiv_rn(value.z, denom));
}

__device__ __forceinline__ float signf_like_torch(float value) {
    return (value > 0.0f) ? 1.0f : ((value < 0.0f) ? -1.0f : 0.0f);
}

__device__ __forceinline__ float unsigned_angle(float3 a, float3 b, float3 axis) {
    const float3 cross = cross3(a, b);
    const float signed_norm = signf_like_torch(dot3(cross, axis)) * norm3(cross);
    float angle = atan2f(signed_norm, dot3(a, b));
    return angle < 0.0f ? angle + 6.28318530717958647692f : angle;
}

__device__ __forceinline__ int opposite_vertex(const int *faces, int face, int shared0, int shared1) {
    const int *tri = faces + static_cast<int64_t>(face) * 3;
    const int v0 = tri[0];
    const int v1 = tri[1];
    const int v2 = tri[2];
    if (v0 != shared0 && v0 != shared1) {
        return v0;
    }
    if (v1 != shared0 && v1 != shared1) {
        return v1;
    }
    return v2;
}

__global__ void diffraction_edge_geometry_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ face_normals,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    bool *__restrict__ selected,
    float *__restrict__ edge_pos,
    float *__restrict__ edge_dir,
    float *__restrict__ lengths,
    float *__restrict__ line_min,
    float *__restrict__ line_max,
    float *__restrict__ n0,
    float *__restrict__ n1,
    float *__restrict__ exterior_angle,
    int64_t edge_count,
    float plane_tol) {
    constexpr float edge_epsilon = 1.0e-6f;
    constexpr float normal_cos_tol = 1.0f - 1.0e-5f;
    constexpr float two_pi = 6.28318530717958647692f;

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        const int v0 = edge_v0[edge];
        const int v1 = edge_v1[edge];
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const bool boundary = valid0 && !valid1;
        const bool interior = valid0 && valid1;
        const int safe0 = f0 >= 0 ? f0 : 0;
        const int safe1 = f1 >= 0 ? f1 : 0;

        const float3 start = load_vec3(vertices, v0);
        const float3 end = load_vec3(vertices, v1);
        const float3 vector = sub3(end, start);
        const float length = fmaxf(norm3(vector), 1.0e-12f);
        const float3 dir = make_float3(
            __fdiv_rn(vector.x, length),
            __fdiv_rn(vector.y, length),
            __fdiv_rn(vector.z, length));
        const float half_length = 0.5f * length;

        const float3 n0_cand = normalize3(load_vec3(face_normals, safe0), edge_epsilon);
        const float3 n1_cand = normalize3(load_vec3(face_normals, safe1), edge_epsilon);
        const float3 to1 = normalize3(cross3(n0_cand, dir), edge_epsilon);
        const float3 tn1 = normalize3(cross3(n1_cand, dir), edge_epsilon);
        const float3 to2 = normalize3(cross3(n1_cand, dir), edge_epsilon);
        const float3 tn2 = normalize3(cross3(n0_cand, dir), edge_epsilon);
        const bool choose_first = unsigned_angle(to1, tn1, dir) < unsigned_angle(to2, tn2, dir);
        const float3 ordered_n0 = choose_first ? n0_cand : n1_cand;
        const float3 ordered_n1 = choose_first ? n1_cand : n0_cand;
        float3 out_n0 = interior ? ordered_n0 : n0_cand;
        float3 out_n1 = interior ? ordered_n1 : n1_cand;
        if (f1 < 0) {
            out_n1 = mul3(n0_cand, -1.0f);
        }
        const float output_normal_dot = dot3(out_n0, out_n1);
        const float output_clamped_neg_dot = fminf(fmaxf(-output_normal_dot, -1.0f), 1.0f);
        const float output_interior_angle = acosf(output_clamped_neg_dot);
        const float out_exterior_angle = interior ? (two_pi - output_interior_angle) : two_pi;

        bool coplanar = false;
        if (interior) {
            const float selected_normal_dot = dot3(n0_cand, n1_cand);
            const bool aligned = fabsf(selected_normal_dot) >= normal_cos_tol;
            const int opp0 = opposite_vertex(faces, safe0, v0, v1);
            const int opp1 = opposite_vertex(faces, safe1, v0, v1);
            const float3 point_a = load_vec3(vertices, opp0);
            const float3 point_b = load_vec3(vertices, opp1);
            const float plane_dist_a = fabsf(dot3(sub3(point_a, start), n0_cand));
            const float plane_dist_b = fabsf(dot3(sub3(point_b, start), n0_cand));
            coplanar = aligned && plane_dist_a <= plane_tol && plane_dist_b <= plane_tol;
        }
        const float selected_normal_dot = dot3(n0_cand, n1_cand);
        const bool selected_wedge_angle = boundary || (interior && selected_normal_dot < 1.0f);
        selected[edge] =
            (interior || boundary) && !coplanar && length > edge_epsilon && selected_wedge_angle;

        store_vec3(edge_pos, edge, mul3(add3(start, end), 0.5f));
        store_vec3(edge_dir, edge, dir);
        lengths[edge] = length;
        line_min[edge] = -half_length;
        line_max[edge] = half_length;
        store_vec3(n0, edge, out_n0);
        store_vec3(n1, edge, out_n1);
        exterior_angle[edge] = out_exterior_angle;
    }
}

__device__ __forceinline__ int find_root_const(const int *__restrict__ parent, int x) {
    int p = parent[x];
    while (p != parent[p]) {
        p = parent[p];
    }
    return p;
}

__device__ int find_root_mutable(int *__restrict__ parent, int x) {
    int p = parent[x];
    while (p != parent[p]) {
        const int gp = parent[p];
        parent[x] = gp;
        x = gp;
        p = parent[x];
    }
    return p;
}

__global__ void init_parent_kernel(int *__restrict__ parent, int count) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        parent[idx] = idx;
    }
}

__global__ void compress_parent_kernel(int *__restrict__ parent, int count) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        parent[idx] = find_root_mutable(parent, idx);
    }
}

__global__ void surface_group_union_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ face_normals,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    int *__restrict__ parent,
    int *__restrict__ changed,
    int64_t edge_count,
    float plane_tol) {
    constexpr float edge_epsilon = 1.0e-6f;
    constexpr float normal_cos_tol = 1.0f - 1.0e-5f;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        if (f0 < 0 || f1 < 0) {
            continue;
        }

        const int v0 = edge_v0[edge];
        const int v1 = edge_v1[edge];
        const float3 n0 = normalize3(load_vec3(face_normals, f0), edge_epsilon);
        const float3 n1 = normalize3(load_vec3(face_normals, f1), edge_epsilon);
        const float normal_dot = dot3(n0, n1);
        if (fabsf(normal_dot) < normal_cos_tol) {
            continue;
        }

        const float3 plane_point = load_vec3(vertices, v0);
        const int opp0 = opposite_vertex(faces, f0, v0, v1);
        const int opp1 = opposite_vertex(faces, f1, v0, v1);
        const float plane_dist_a = fabsf(dot3(sub3(load_vec3(vertices, opp0), plane_point), n0));
        const float plane_dist_b = fabsf(dot3(sub3(load_vec3(vertices, opp1), plane_point), n0));
        if (plane_dist_a > plane_tol || plane_dist_b > plane_tol) {
            continue;
        }

        while (true) {
            int root0 = find_root_mutable(parent, f0);
            int root1 = find_root_mutable(parent, f1);
            if (root0 == root1) {
                break;
            }
            const int low = root0 < root1 ? root0 : root1;
            const int high = root0 < root1 ? root1 : root0;
            const int old = atomicMin(parent + high, low);
            if (old != low) {
                *changed = 1;
            }
            if (old == high || old == low) {
                break;
            }
        }
    }
}

__global__ void count_surface_group_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_count,
    int *__restrict__ max_count,
    int64_t edge_count,
    int face_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_count[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            root_count[root0] += 1;
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                root_count[root1] += 1;
            }
        }
    }
    int local_max = 0;
    for (int face = 0; face < face_count; ++face) {
        local_max = root_count[face] > local_max ? root_count[face] : local_max;
    }
    max_count[0] = local_max;
}

__global__ void fill_int_kernel(int *__restrict__ data, int value, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        data[idx] = value;
    }
}

__global__ void fill_surface_group_root_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_cursor,
    int *__restrict__ root_indices,
    int64_t edge_count,
    int face_count,
    int max_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_cursor[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int edge_i = static_cast<int>(edge);
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            const int slot = root_cursor[root0]++;
            if (slot < max_count) {
                root_indices[static_cast<int64_t>(root0) * max_count + slot] = edge_i;
            }
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                const int slot = root_cursor[root1]++;
                if (slot < max_count) {
                    root_indices[static_cast<int64_t>(root1) * max_count + slot] = edge_i;
                }
            }
        }
    }
}

__global__ void emit_surface_group_face_rows_kernel(
    const int *__restrict__ parent,
    const int *__restrict__ root_count,
    const int *__restrict__ root_indices,
    int *__restrict__ counts,
    int *__restrict__ indices,
    int face_count,
    int max_count) {
    const int stride = blockDim.x * gridDim.x;
    for (int face = blockIdx.x * blockDim.x + threadIdx.x; face < face_count; face += stride) {
        const int root = find_root_const(parent, face);
        const int count = root_count[root];
        counts[face] = count;
        for (int slot = 0; slot < count && slot < max_count; ++slot) {
            indices[static_cast<int64_t>(face) * max_count + slot] =
                root_indices[static_cast<int64_t>(root) * max_count + slot];
        }
    }
}

__global__ void count_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ count,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int local_count = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            ++local_count;
        }
    }
    count[0] = local_count;
}

__global__ void fill_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ indices,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int cursor = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            indices[cursor++] = static_cast<int>(edge);
        }
    }
}

}  // namespace

at::Tensor cn_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src) {
    check_tensor(state_edge_pos, "state_edge_pos", at::kFloat, 2);
    check_tensor(state_src, "state_src", at::kFloat, 2);
    TORCH_CHECK(state_edge_pos.size(1) == 3, "state_edge_pos must have shape (N, 3)");
    TORCH_CHECK(state_src.size(1) == 3, "state_src must have shape (N, 3)");
    TORCH_CHECK(state_src.size(0) == state_edge_pos.size(0), "state_src must match state_edge_pos");

    const int64_t state_count = state_edge_pos.size(0);
    auto state_wi = at::empty({state_count, 3}, state_edge_pos.options());
    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(state_edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_wi_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            state_edge_pos.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_wi.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state_wi;
}

at::Tensor cn_mc_selected_edge_indices_cuda(at::Tensor selected) {
    check_tensor(selected, "selected", at::kBool, 1);
    const int64_t edge_count = selected.size(0);
    auto int_options = selected.options().dtype(at::kInt);
    auto count_tensor = at::empty({1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(selected.get_device()).stream();
    count_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        count_tensor.data_ptr<int>(),
        edge_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int host_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_count,
        count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_count >= 0, "selected edge count must be non-negative");

    auto indices = at::empty({host_count}, int_options);
    if (host_count > 0) {
        fill_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
            selected.data_ptr<bool>(),
            indices.data_ptr<int>(),
            edge_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return indices;
}

std::vector<at::Tensor> cn_mc_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power) {
    check_tensor(edge_indices, "edge_indices", at::kInt, 1);
    check_tensor(edge_pos, "edge_pos", at::kFloat, 2);
    check_tensor(edge_dir, "edge_dir", at::kFloat, 2);
    check_tensor(line_min, "line_min", at::kFloat, 1);
    check_tensor(line_max, "line_max", at::kFloat, 1);
    check_tensor(n0, "n0", at::kFloat, 2);
    check_tensor(n1, "n1", at::kFloat, 2);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(exterior_angle, "exterior_angle", at::kFloat, 1);
    check_tensor(tx, "tx", at::kFloat, 1);
    check_tensor(tx_power, "tx_power", at::kFloat, 0);
    TORCH_CHECK(edge_pos.size(1) == 3, "edge_pos must have shape (N, 3)");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge count");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge count");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge count");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge count");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge count");
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(edge_pos.get_device() == edge_indices.get_device(), "edge tensors must be on the same device");

    const int64_t state_count = edge_indices.size(0);
    auto int_options = edge_indices.options();
    auto float_options = edge_pos.options();
    auto state_edge_index = at::empty({state_count}, int_options);
    auto state_edge_pos = at::empty({state_count, 3}, float_options);
    auto state_edge_dir = at::empty({state_count, 3}, float_options);
    auto state_line_min = at::empty({state_count}, float_options);
    auto state_line_max = at::empty({state_count}, float_options);
    auto state_n0 = at::empty({state_count, 3}, float_options);
    auto state_n1 = at::empty({state_count, 3}, float_options);
    auto state_face0 = at::empty({state_count}, int_options);
    auto state_face1 = at::empty({state_count}, int_options);
    auto state_exterior_angle = at::empty({state_count}, float_options);
    auto state_src = at::empty({state_count, 3}, float_options);
    auto state_src_power = at::empty({state_count}, float_options);

    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_pack_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            edge_indices.data_ptr<int>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            exterior_angle.data_ptr<float>(),
            tx.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_line_min.data_ptr<float>(),
            state_line_max.data_ptr<float>(),
            state_n0.data_ptr<float>(),
            state_n1.data_ptr<float>(),
            state_face0.data_ptr<int>(),
            state_face1.data_ptr<int>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        state_edge_index,
        state_edge_pos,
        state_edge_dir,
        state_line_min,
        state_line_max,
        state_n0,
        state_n1,
        state_face0,
        state_face1,
        state_exterior_angle,
        state_src,
        state_src_power,
    };
}

std::vector<at::Tensor> cn_mc_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");

    const int64_t edge_count = edge_v0.size(0);
    auto bool_options = edge_v0.options().dtype(at::kBool);
    auto float_options = vertices.options();
    auto selected = at::empty({edge_count}, bool_options);
    auto edge_pos = at::empty({edge_count, 3}, float_options);
    auto edge_dir = at::empty({edge_count, 3}, float_options);
    auto lengths = at::empty({edge_count}, float_options);
    auto line_min = at::empty({edge_count}, float_options);
    auto line_max = at::empty({edge_count}, float_options);
    auto n0 = at::empty({edge_count, 3}, float_options);
    auto n1 = at::empty({edge_count, 3}, float_options);
    auto exterior_angle = at::empty({edge_count}, float_options);

    if (edge_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const int block_count = static_cast<int>((edge_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_edge_geometry_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            face_normals.data_ptr<float>(),
            edge_v0.data_ptr<int>(),
            edge_v1.data_ptr<int>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            selected.data_ptr<bool>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            lengths.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            exterior_angle.data_ptr<float>(),
            edge_count,
            static_cast<float>(plane_tol));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        selected,
        edge_pos,
        edge_dir,
        lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    };
}

std::vector<at::Tensor> cn_mc_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(selected, "selected", at::kBool, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");
    TORCH_CHECK(selected.size(0) == edge_v0.size(0), "selected must match edge_v0");

    const int face_count = static_cast<int>(faces.size(0));
    const int64_t edge_count = edge_v0.size(0);
    auto int_options = faces.options();
    auto parent = at::empty({face_count}, int_options);
    auto root_count = at::empty({face_count}, int_options);
    auto root_cursor = at::empty({face_count}, int_options);
    auto max_count_tensor = at::empty({1}, int_options);
    auto changed = at::empty({1}, int_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int face_blocks = static_cast<int>((face_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
    const int edge_blocks = static_cast<int>((edge_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
    if (face_count > 0) {
        init_parent_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            face_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int host_changed = 0;
    int iteration = 0;
    constexpr int kMaxUnionIterations = 512;
    do {
        host_changed = 0;
        C10_CUDA_CHECK(cudaMemsetAsync(changed.data_ptr<int>(), 0, sizeof(int), stream));
        if (edge_count > 0) {
            surface_group_union_kernel<<<edge_blocks, kDiffractionBlockSize, 0, stream>>>(
                vertices.data_ptr<float>(),
                faces.data_ptr<int>(),
                face_normals.data_ptr<float>(),
                edge_v0.data_ptr<int>(),
                edge_v1.data_ptr<int>(),
                face0.data_ptr<int>(),
                face1.data_ptr<int>(),
                parent.data_ptr<int>(),
                changed.data_ptr<int>(),
                edge_count,
                static_cast<float>(plane_tol));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (face_count > 0) {
            compress_parent_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
                parent.data_ptr<int>(),
                face_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &host_changed,
            changed.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        ++iteration;
    } while (host_changed != 0 && iteration < kMaxUnionIterations);
    TORCH_CHECK(iteration < kMaxUnionIterations, "surface group union did not converge");

    count_surface_group_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_count.data_ptr<int>(),
        max_count_tensor.data_ptr<int>(),
        edge_count,
        face_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int host_max_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_max_count,
        max_count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_max_count >= 0, "surface group candidate max count must be non-negative");

    auto counts = at::empty({face_count}, int_options);
    auto indices = at::empty({face_count, host_max_count}, int_options);
    auto root_indices = at::empty({face_count, host_max_count}, int_options);
    const int64_t table_count = static_cast<int64_t>(face_count) * host_max_count;
    if (table_count > 0) {
        const int table_blocks = static_cast<int>((table_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            root_indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    fill_surface_group_root_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_cursor.data_ptr<int>(),
        root_indices.data_ptr<int>(),
        edge_count,
        face_count,
        host_max_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (face_count > 0) {
        emit_surface_group_face_rows_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            root_count.data_ptr<int>(),
            root_indices.data_ptr<int>(),
            counts.data_ptr<int>(),
            indices.data_ptr<int>(),
            face_count,
            host_max_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {counts, indices};
}
