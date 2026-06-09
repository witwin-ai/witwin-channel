#include <raydn/diffraction/builder.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace raydn {
namespace {

constexpr int kBlockSize = 256;
constexpr float kEps = 1.0e-6f;
constexpr float kSmallEps = 1.0e-6f;
constexpr float kHalfPiMinusOffset = 1.52079632679f;

struct Vec3 {
    float x;
    float y;
    float z;
};

__device__ __forceinline__ Vec3 make_v3(float x, float y, float z) {
    return Vec3{x, y, z};
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
    return make_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
    return make_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 mul(Vec3 a, float s) {
    return make_v3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
    return make_v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float squared_norm(Vec3 v) {
    return dot(v, v);
}

__device__ __forceinline__ float norm(Vec3 v) {
    return sqrtf(fmaxf(squared_norm(v), 0.0f));
}

__device__ __forceinline__ Vec3 normalize(Vec3 v, Vec3 fallback) {
    const float n = norm(v);
    if (n <= kEps) {
        return fallback;
    }
    return mul(v, 1.0f / n);
}

__device__ __forceinline__ float clampf_local(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

__device__ __forceinline__ Vec3 project_to_wedge_plane(Vec3 vec, Vec3 edge_dir) {
    const float denom = fmaxf(norm(edge_dir), kEps);
    const Vec3 edge_hat = mul(edge_dir, 1.0f / denom);
    return sub(vec, mul(edge_hat, dot(vec, edge_hat)));
}

__device__ __forceinline__ Vec3 surface_tangent_from_hit(Vec3 ray_dir, Vec3 surface_normal) {
    const Vec3 tangent = sub(ray_dir, mul(surface_normal, dot(ray_dir, surface_normal)));
    if (norm(tangent) > kEps) {
        return normalize(tangent, make_v3(1.0f, 0.0f, 0.0f));
    }
    const Vec3 fallback_x = cross(surface_normal, make_v3(1.0f, 0.0f, 0.0f));
    const Vec3 fallback_y = cross(surface_normal, make_v3(0.0f, 1.0f, 0.0f));
    const Vec3 fallback = norm(fallback_x) > kEps ? fallback_x : fallback_y;
    return normalize(fallback, make_v3(1.0f, 0.0f, 0.0f));
}

__device__ __forceinline__ Vec3 silhouette_viewpoint(
    Vec3 hit_p,
    Vec3 shading_normal,
    Vec3 geometric_normal,
    Vec3 ray_dir) {
    const Vec3 resolved_geometric = norm(geometric_normal) > kEps ? geometric_normal : shading_normal;
    const Vec3 surface_normal = dot(ray_dir, resolved_geometric) > 0.0f
        ? mul(resolved_geometric, -1.0f)
        : resolved_geometric;
    const Vec3 tangent = surface_tangent_from_hit(ray_dir, surface_normal);
    const Vec3 d = add(
        mul(surface_normal, cosf(kHalfPiMinusOffset)),
        mul(tangent, sinf(kHalfPiMinusOffset)));
    return add(hit_p, mul(d, 0.1f));
}

__device__ __forceinline__ bool wedge_exterior(
    Vec3 direction_from_edge,
    Vec3 edge_dir,
    Vec3 n0,
    Vec3 nn) {
    const Vec3 direction_proj = project_to_wedge_plane(direction_from_edge, edge_dir);
    const float signed_distance_0 = dot(direction_proj, n0);
    const float signed_distance_n = dot(direction_proj, nn);
    return (norm(direction_proj) > kSmallEps) &&
        ((signed_distance_0 >= -kSmallEps) || (signed_distance_n >= -kSmallEps));
}

__device__ __forceinline__ int best_edge_for_hit(
    Vec3 tx_pos,
    Vec3 ray_dir,
    Vec3 hit_p,
    Vec3 hit_n,
    Vec3 hit_geo_n,
    int prim_idx,
    const int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    int n_triangles,
    const float *edge_pos,
    const float *edge_dir,
    const float *edge_n0,
    const float *edge_nn,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges) {
    if (prim_idx < 0 || prim_idx >= n_triangles) {
        return -1;
    }

    const Vec3 viewpoint = silhouette_viewpoint(hit_p, hit_n, hit_geo_n, ray_dir);
    int best_edge_idx = -1;
    float best_distance = 1.0e30f;
    const int candidate_count = triangle_edge_count[prim_idx];
    const int row_base = prim_idx * max_triangle_edge_slots;
    for (int slot = 0; slot < max_triangle_edge_slots; ++slot) {
        if (slot >= candidate_count) {
            break;
        }
        const int edge_idx = triangle_edge_indices[row_base + slot];
        if (edge_idx < 0 || edge_idx >= n_edges) {
            continue;
        }
        const Vec3 edge_point0 = make_v3(edge_pos[edge_idx * 3 + 0],
                                         edge_pos[edge_idx * 3 + 1],
                                         edge_pos[edge_idx * 3 + 2]);
        const Vec3 edge_direction = make_v3(edge_dir[edge_idx * 3 + 0],
                                            edge_dir[edge_idx * 3 + 1],
                                            edge_dir[edge_idx * 3 + 2]);
        const Vec3 edge_hat = normalize(edge_direction, make_v3(0.0f, 0.0f, 1.0f));
        const float ell = clampf_local(
            dot(sub(viewpoint, edge_point0), edge_hat),
            edge_line_min[edge_idx],
            edge_line_max[edge_idx]);
        const Vec3 edge_point = add(edge_point0, mul(edge_hat, ell));
        const Vec3 n0 = make_v3(edge_n0[edge_idx * 3 + 0],
                                edge_n0[edge_idx * 3 + 1],
                                edge_n0[edge_idx * 3 + 2]);
        const Vec3 nn = make_v3(edge_nn[edge_idx * 3 + 0],
                                edge_nn[edge_idx * 3 + 1],
                                edge_nn[edge_idx * 3 + 2]);
        const bool flip = dot(ray_dir, n0) > 0.0f;
        const Vec3 oriented_n0 = flip ? nn : n0;
        const Vec3 oriented_nn = flip ? n0 : nn;
        if (!wedge_exterior(sub(tx_pos, edge_point), edge_direction, oriented_n0, oriented_nn)) {
            continue;
        }
        const Vec3 view_vec = sub(viewpoint, edge_point);
        const bool face0_front = dot(view_vec, n0) > kEps;
        const bool face1_front = dot(view_vec, nn) > kEps;
        const bool is_boundary_edge = edge_adjacent_face1[edge_idx] < 0;
        const bool silhouette_edge = is_boundary_edge || (face0_front != face1_front);
        if (!silhouette_edge) {
            continue;
        }
        const float candidate_distance = squared_norm(sub(viewpoint, edge_point));
        if (candidate_distance < best_distance) {
            best_distance = candidate_distance;
            best_edge_idx = edge_idx;
        }
    }
    return best_edge_idx;
}

__global__ void discover_edges_kernel(
    const float *tx_pos,
    const float *ray_dir,
    const int *prim_index,
    const float *hit_p,
    const float *hit_n,
    const float *hit_geo_n,
    int n_hits,
    const int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    int n_triangles,
    const float *edge_pos,
    const float *edge_dir,
    const float *edge_n0,
    const float *edge_nn,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges,
    int *out_seen_edge_mask) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_hits) {
        return;
    }
    const Vec3 tx = make_v3(tx_pos[0], tx_pos[1], tx_pos[2]);
    const int edge_idx = best_edge_for_hit(
        tx,
        make_v3(ray_dir[tid * 3 + 0], ray_dir[tid * 3 + 1], ray_dir[tid * 3 + 2]),
        make_v3(hit_p[tid * 3 + 0], hit_p[tid * 3 + 1], hit_p[tid * 3 + 2]),
        make_v3(hit_n[tid * 3 + 0], hit_n[tid * 3 + 1], hit_n[tid * 3 + 2]),
        make_v3(hit_geo_n[tid * 3 + 0], hit_geo_n[tid * 3 + 1], hit_geo_n[tid * 3 + 2]),
        prim_index[tid],
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots,
        n_triangles,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_nn,
        edge_line_min,
        edge_line_max,
        edge_adjacent_face1,
        n_edges);
    if (edge_idx >= 0 && edge_idx < n_edges) {
        atomicExch(out_seen_edge_mask + edge_idx, 1);
    }
}

} // namespace

void diffraction_discover_edges_cuda(
    const at::Tensor &tx_pos,
    const at::Tensor &ray_dir,
    const at::Tensor &prim_index,
    const at::Tensor &hit_p,
    const at::Tensor &hit_n,
    const at::Tensor &hit_geo_n,
    const at::Tensor &triangle_edge_count,
    const at::Tensor &triangle_edge_indices,
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &edge_n0,
    const at::Tensor &edge_nn,
    const at::Tensor &edge_line_min,
    const at::Tensor &edge_line_max,
    const at::Tensor &edge_adjacent_face1,
    at::Tensor &out_seen_edge_mask) {
    const int n_hits = static_cast<int>(ray_dir.size(0));
    const int n_edges = static_cast<int>(edge_pos.size(0));
    if (n_hits <= 0 || n_edges <= 0) {
        return;
    }
    const dim3 block(kBlockSize);
    const dim3 grid((n_hits + kBlockSize - 1) / kBlockSize);
    discover_edges_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tx_pos.data_ptr<float>(),
        ray_dir.data_ptr<float>(),
        prim_index.data_ptr<int>(),
        hit_p.data_ptr<float>(),
        hit_n.data_ptr<float>(),
        hit_geo_n.data_ptr<float>(),
        n_hits,
        triangle_edge_count.data_ptr<int>(),
        triangle_edge_indices.data_ptr<int>(),
        static_cast<int>(triangle_edge_indices.size(1)),
        static_cast<int>(triangle_edge_count.size(0)),
        edge_pos.data_ptr<float>(),
        edge_dir.data_ptr<float>(),
        edge_n0.data_ptr<float>(),
        edge_nn.data_ptr<float>(),
        edge_line_min.data_ptr<float>(),
        edge_line_max.data_ptr<float>(),
        edge_adjacent_face1.data_ptr<int>(),
        n_edges,
        out_seen_edge_mask.data_ptr<int>());
}

} // namespace raydn
