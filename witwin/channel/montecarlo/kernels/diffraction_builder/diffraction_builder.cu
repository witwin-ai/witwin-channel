#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <diffraction_builder/diffraction_builder.h>

#include <cmath>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

constexpr int BLOCK_SIZE = 256;
constexpr float EPS = 1.0e-6f;
constexpr float SMALL_EPS = 1.0e-6f;
constexpr float HALF_PI_MINUS_OFFSET = 1.52079632679f;

struct Vec3 {
    float x;
    float y;
    float z;
};

__device__ __forceinline__ Vec3 make_vec3(float x, float y, float z) {
    return Vec3{x, y, z};
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 mul(Vec3 a, float s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
    return make_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float squared_norm(Vec3 v) {
    return dot(v, v);
}

__device__ __forceinline__ float norm(Vec3 v) {
    return sqrtf(fmaxf(squared_norm(v), 0.0f));
}

__device__ __forceinline__ Vec3 normalize(Vec3 v, Vec3 fallback) {
    float n = norm(v);
    if (n <= EPS) {
        return fallback;
    }
    return mul(v, 1.0f / n);
}

__device__ __forceinline__ float clampf_local(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

__device__ __forceinline__ Vec3 project_to_wedge_plane(Vec3 vec, Vec3 edge_dir) {
    float denom = fmaxf(norm(edge_dir), EPS);
    Vec3 edge_hat = mul(edge_dir, 1.0f / denom);
    return sub(vec, mul(edge_hat, dot(vec, edge_hat)));
}

__device__ __forceinline__ Vec3 surface_tangent_from_hit(Vec3 ray_dir, Vec3 surface_normal) {
    Vec3 tangent = sub(ray_dir, mul(surface_normal, dot(ray_dir, surface_normal)));
    if (norm(tangent) > EPS) {
        return normalize(tangent, make_vec3(1.0f, 0.0f, 0.0f));
    }
    Vec3 fallback_x = cross(surface_normal, make_vec3(1.0f, 0.0f, 0.0f));
    Vec3 fallback_y = cross(surface_normal, make_vec3(0.0f, 1.0f, 0.0f));
    Vec3 fallback = norm(fallback_x) > EPS ? fallback_x : fallback_y;
    return normalize(fallback, make_vec3(1.0f, 0.0f, 0.0f));
}

__device__ __forceinline__ Vec3 silhouette_viewpoint(
    Vec3 hit_p,
    Vec3 shading_normal,
    Vec3 geometric_normal,
    Vec3 ray_dir
) {
    Vec3 resolved_geometric = norm(geometric_normal) > EPS ? geometric_normal : shading_normal;
    Vec3 surface_normal = dot(ray_dir, resolved_geometric) > 0.0f
        ? mul(resolved_geometric, -1.0f)
        : resolved_geometric;
    Vec3 tangent = surface_tangent_from_hit(ray_dir, surface_normal);
    Vec3 d = add(
        mul(surface_normal, cosf(HALF_PI_MINUS_OFFSET)),
        mul(tangent, sinf(HALF_PI_MINUS_OFFSET))
    );
    return add(hit_p, mul(d, 0.1f));
}

__device__ __forceinline__ bool wedge_exterior(
    Vec3 direction_from_edge,
    Vec3 edge_dir,
    Vec3 n0,
    Vec3 nn
) {
    Vec3 direction_proj = project_to_wedge_plane(direction_from_edge, edge_dir);
    float signed_distance_0 = dot(direction_proj, n0);
    float signed_distance_n = dot(direction_proj, nn);
    return (norm(direction_proj) > SMALL_EPS)
        && ((signed_distance_0 >= -SMALL_EPS) || (signed_distance_n >= -SMALL_EPS));
}

__device__ __forceinline__ unsigned int hash_uniform_bits(unsigned int index, int stream, int seed) {
    unsigned int resolved_seed = static_cast<unsigned int>(seed);
    unsigned int stream_value = static_cast<unsigned int>(stream + 1);
    unsigned int value =
        index * 747796405u
        + (resolved_seed + 1u) * 2891336453u
        + stream_value * 277803737u;
    value = (value ^ (value >> 16)) * 2246822519u;
    value = (value ^ (value >> 13)) * 3266489917u;
    value = value ^ (value >> 16);
    return value;
}

__device__ __forceinline__ float hash_uniform(unsigned int index, int stream, int seed) {
    unsigned int bits = hash_uniform_bits(index, stream, seed) & 0x00FFFFFFu;
    return static_cast<float>(bits) / 16777216.0f;
}

__global__ void diffraction_sample_slots_kernel(
    const unsigned int *sample_index,
    const float *cdf,
    unsigned int *out_slots,
    int n_samples,
    int n_states,
    float total_length_scalar,
    int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) {
        return;
    }
    if (n_states <= 0) {
        out_slots[tid] = 0u;
        return;
    }
    float sample_u = hash_uniform(sample_index[tid], 601, seed) * total_length_scalar;
    int lo = 0;
    int hi = n_states - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (cdf[mid] < sample_u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out_slots[tid] = static_cast<unsigned int>(lo);
}

__global__ void diffraction_best_edge_kernel(
    float tx_x,
    float tx_y,
    float tx_z,
    const float *ray_dir_x,
    const float *ray_dir_y,
    const float *ray_dir_z,
    const float *hit_p_x,
    const float *hit_p_y,
    const float *hit_p_z,
    const float *hit_n_x,
    const float *hit_n_y,
    const float *hit_n_z,
    const float *hit_geo_n_x,
    const float *hit_geo_n_y,
    const float *hit_geo_n_z,
    const int *hit_mask,
    const unsigned int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges,
    int n_rays,
    int *out_best_edge_idx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays || hit_mask[tid] == 0) {
        if (tid < n_rays) {
            out_best_edge_idx[tid] = -1;
        }
        return;
    }

    Vec3 tx_pos = make_vec3(tx_x, tx_y, tx_z);
    Vec3 ray_dir = make_vec3(ray_dir_x[tid], ray_dir_y[tid], ray_dir_z[tid]);
    Vec3 hit_p = make_vec3(hit_p_x[tid], hit_p_y[tid], hit_p_z[tid]);
    Vec3 hit_n = make_vec3(hit_n_x[tid], hit_n_y[tid], hit_n_z[tid]);
    Vec3 hit_geo_n = make_vec3(hit_geo_n_x[tid], hit_geo_n_y[tid], hit_geo_n_z[tid]);
    Vec3 viewpoint = silhouette_viewpoint(hit_p, hit_n, hit_geo_n, ray_dir);

    int best_edge_idx = -1;
    float best_distance = 1.0e30f;
    unsigned int candidate_count = triangle_edge_count[tid];
    int row_base = tid * max_triangle_edge_slots;
    for (int slot = 0; slot < max_triangle_edge_slots; ++slot) {
        if (slot >= static_cast<int>(candidate_count)) {
            break;
        }
        int edge_idx = triangle_edge_indices[row_base + slot];
        if (edge_idx < 0 || edge_idx >= n_edges) {
            continue;
        }

        Vec3 edge_pos = make_vec3(edge_pos_x[edge_idx], edge_pos_y[edge_idx], edge_pos_z[edge_idx]);
        Vec3 edge_dir = make_vec3(edge_dir_x[edge_idx], edge_dir_y[edge_idx], edge_dir_z[edge_idx]);
        Vec3 edge_hat = normalize(edge_dir, make_vec3(0.0f, 0.0f, 1.0f));
        float ell = clampf_local(
            dot(sub(viewpoint, edge_pos), edge_hat),
            edge_line_min[edge_idx],
            edge_line_max[edge_idx]
        );
        Vec3 edge_point = add(edge_pos, mul(edge_hat, ell));
        Vec3 n0 = make_vec3(edge_n0_x[edge_idx], edge_n0_y[edge_idx], edge_n0_z[edge_idx]);
        Vec3 nn = make_vec3(edge_nn_x[edge_idx], edge_nn_y[edge_idx], edge_nn_z[edge_idx]);
        bool flip = dot(ray_dir, n0) > 0.0f;
        Vec3 oriented_n0 = flip ? nn : n0;
        Vec3 oriented_nn = flip ? n0 : nn;
        if (!wedge_exterior(sub(tx_pos, edge_point), edge_dir, oriented_n0, oriented_nn)) {
            continue;
        }
        Vec3 view_vec = sub(viewpoint, edge_point);
        bool face0_front = dot(view_vec, n0) > EPS;
        bool face1_front = dot(view_vec, nn) > EPS;
        bool is_boundary_edge = edge_adjacent_face1[edge_idx] < 0;
        bool silhouette_edge = is_boundary_edge || (face0_front != face1_front);
        if (!silhouette_edge) {
            continue;
        }
        float candidate_distance = squared_norm(sub(viewpoint, edge_point));
        if (candidate_distance < best_distance) {
            best_distance = candidate_distance;
            best_edge_idx = edge_idx;
        }
    }
    out_best_edge_idx[tid] = best_edge_idx;
}

__device__ __forceinline__ int best_edge_for_hit(
    Vec3 tx_pos,
    Vec3 ray_dir,
    Vec3 hit_p,
    Vec3 hit_n,
    Vec3 hit_geo_n,
    int prim_idx,
    const unsigned int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    int n_triangles,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges
) {
    if (prim_idx < 0 || prim_idx >= n_triangles) {
        return -1;
    }
    Vec3 viewpoint = silhouette_viewpoint(hit_p, hit_n, hit_geo_n, ray_dir);
    int best_edge_idx = -1;
    float best_distance = 1.0e30f;
    unsigned int candidate_count = triangle_edge_count[prim_idx];
    int row_base = prim_idx * max_triangle_edge_slots;
    for (int slot = 0; slot < max_triangle_edge_slots; ++slot) {
        if (slot >= static_cast<int>(candidate_count)) {
            break;
        }
        int edge_idx = triangle_edge_indices[row_base + slot];
        if (edge_idx < 0 || edge_idx >= n_edges) {
            continue;
        }
        Vec3 edge_pos = make_vec3(edge_pos_x[edge_idx], edge_pos_y[edge_idx], edge_pos_z[edge_idx]);
        Vec3 edge_dir = make_vec3(edge_dir_x[edge_idx], edge_dir_y[edge_idx], edge_dir_z[edge_idx]);
        Vec3 edge_hat = normalize(edge_dir, make_vec3(0.0f, 0.0f, 1.0f));
        float ell = clampf_local(
            dot(sub(viewpoint, edge_pos), edge_hat),
            edge_line_min[edge_idx],
            edge_line_max[edge_idx]
        );
        Vec3 edge_point = add(edge_pos, mul(edge_hat, ell));
        Vec3 n0 = make_vec3(edge_n0_x[edge_idx], edge_n0_y[edge_idx], edge_n0_z[edge_idx]);
        Vec3 nn = make_vec3(edge_nn_x[edge_idx], edge_nn_y[edge_idx], edge_nn_z[edge_idx]);
        bool flip = dot(ray_dir, n0) > 0.0f;
        Vec3 oriented_n0 = flip ? nn : n0;
        Vec3 oriented_nn = flip ? n0 : nn;
        if (!wedge_exterior(sub(tx_pos, edge_point), edge_dir, oriented_n0, oriented_nn)) {
            continue;
        }
        Vec3 view_vec = sub(viewpoint, edge_point);
        bool face0_front = dot(view_vec, n0) > EPS;
        bool face1_front = dot(view_vec, nn) > EPS;
        bool is_boundary_edge = edge_adjacent_face1[edge_idx] < 0;
        bool silhouette_edge = is_boundary_edge || (face0_front != face1_front);
        if (!silhouette_edge) {
            continue;
        }
        float candidate_distance = squared_norm(sub(viewpoint, edge_point));
        if (candidate_distance < best_distance) {
            best_distance = candidate_distance;
            best_edge_idx = edge_idx;
        }
    }
    return best_edge_idx;
}

__global__ void diffraction_discover_edges_kernel(
    float tx_x,
    float tx_y,
    float tx_z,
    const float *ray_dir_x,
    const float *ray_dir_y,
    const float *ray_dir_z,
    const int *prim_index,
    const float *hit_p_x,
    const float *hit_p_y,
    const float *hit_p_z,
    const float *hit_n_x,
    const float *hit_n_y,
    const float *hit_n_z,
    const float *hit_geo_n_x,
    const float *hit_geo_n_y,
    const float *hit_geo_n_z,
    int n_hits,
    const unsigned int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    int n_triangles,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges,
    unsigned int *out_seen_edge_mask
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_hits) {
        return;
    }
    Vec3 tx_pos = make_vec3(tx_x, tx_y, tx_z);
    Vec3 ray_dir = make_vec3(ray_dir_x[tid], ray_dir_y[tid], ray_dir_z[tid]);
    Vec3 hit_p = make_vec3(hit_p_x[tid], hit_p_y[tid], hit_p_z[tid]);
    Vec3 hit_n = make_vec3(hit_n_x[tid], hit_n_y[tid], hit_n_z[tid]);
    Vec3 hit_geo_n = make_vec3(hit_geo_n_x[tid], hit_geo_n_y[tid], hit_geo_n_z[tid]);
    int edge_idx = best_edge_for_hit(
        tx_pos,
        ray_dir,
        hit_p,
        hit_n,
        hit_geo_n,
        prim_index[tid],
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots,
        n_triangles,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_line_min,
        edge_line_max,
        edge_adjacent_face1,
        n_edges
    );
    if (edge_idx >= 0 && edge_idx < n_edges) {
        atomicExch(&out_seen_edge_mask[edge_idx], 1u);
    }
}

__global__ void diffraction_build_state_arrays_kernel(
    const unsigned int *edge_idx,
    int n_states,
    float tx_x,
    float tx_y,
    float tx_z,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face0,
    const int *edge_adjacent_face1,
    int *out_edge_index,
    float *out_edge_pos_x,
    float *out_edge_pos_y,
    float *out_edge_pos_z,
    float *out_edge_dir_x,
    float *out_edge_dir_y,
    float *out_edge_dir_z,
    float *out_n0_x,
    float *out_n0_y,
    float *out_n0_z,
    float *out_nn_x,
    float *out_nn_y,
    float *out_nn_z,
    float *out_wedge_n,
    float *out_line_min,
    float *out_line_max,
    float *out_source_pos_x,
    float *out_source_pos_y,
    float *out_source_pos_z,
    int *out_adjacent_face0,
    int *out_adjacent_face1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) {
        return;
    }
    unsigned int edge_slot = edge_idx[tid];
    out_edge_index[tid] = static_cast<int>(edge_slot);
    out_edge_pos_x[tid] = edge_pos_x[edge_slot];
    out_edge_pos_y[tid] = edge_pos_y[edge_slot];
    out_edge_pos_z[tid] = edge_pos_z[edge_slot];
    out_edge_dir_x[tid] = edge_dir_x[edge_slot];
    out_edge_dir_y[tid] = edge_dir_y[edge_slot];
    out_edge_dir_z[tid] = edge_dir_z[edge_slot];
    out_n0_x[tid] = edge_n0_x[edge_slot];
    out_n0_y[tid] = edge_n0_y[edge_slot];
    out_n0_z[tid] = edge_n0_z[edge_slot];
    out_nn_x[tid] = edge_nn_x[edge_slot];
    out_nn_y[tid] = edge_nn_y[edge_slot];
    out_nn_z[tid] = edge_nn_z[edge_slot];
    out_wedge_n[tid] = edge_wedge_n[edge_slot];
    out_line_min[tid] = edge_line_min[edge_slot];
    out_line_max[tid] = edge_line_max[edge_slot];
    out_source_pos_x[tid] = tx_x;
    out_source_pos_y[tid] = tx_y;
    out_source_pos_z[tid] = tx_z;
    out_adjacent_face0[tid] = edge_adjacent_face0[edge_slot];
    out_adjacent_face1[tid] = edge_adjacent_face1[edge_slot];
}

} // namespace

void monte_carlo_diffraction_sample_slots(
    const unsigned int *sample_index,
    const float *cdf,
    unsigned int *out_slots,
    int n_samples,
    int n_states,
    float total_length_scalar,
    int seed
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    diffraction_sample_slots_kernel<<<grid_size, BLOCK_SIZE>>>(
        sample_index,
        cdf,
        out_slots,
        n_samples,
        n_states,
        total_length_scalar,
        seed
    );
    throw_cuda(cudaGetLastError(), "diffraction_sample_slots_kernel launch");
}

void monte_carlo_diffraction_best_edge_indices(
    float tx_x,
    float tx_y,
    float tx_z,
    const float *ray_dir_x,
    const float *ray_dir_y,
    const float *ray_dir_z,
    const float *hit_p_x,
    const float *hit_p_y,
    const float *hit_p_z,
    const float *hit_n_x,
    const float *hit_n_y,
    const float *hit_n_z,
    const float *hit_geo_n_x,
    const float *hit_geo_n_y,
    const float *hit_geo_n_z,
    const int *hit_mask,
    const unsigned int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges,
    int n_rays,
    int *out_best_edge_idx
) {
    if (n_rays <= 0) {
        return;
    }
    int grid_size = (n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;
    diffraction_best_edge_kernel<<<grid_size, BLOCK_SIZE>>>(
        tx_x,
        tx_y,
        tx_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
        hit_p_x,
        hit_p_y,
        hit_p_z,
        hit_n_x,
        hit_n_y,
        hit_n_z,
        hit_geo_n_x,
        hit_geo_n_y,
        hit_geo_n_z,
        hit_mask,
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_line_min,
        edge_line_max,
        edge_adjacent_face1,
        n_edges,
        n_rays,
        out_best_edge_idx
    );
    throw_cuda(cudaGetLastError(), "diffraction_best_edge_kernel launch");
}

void monte_carlo_diffraction_discover_edges(
    float tx_x,
    float tx_y,
    float tx_z,
    const float *ray_dir_x,
    const float *ray_dir_y,
    const float *ray_dir_z,
    const int *prim_index,
    const float *hit_p_x,
    const float *hit_p_y,
    const float *hit_p_z,
    const float *hit_n_x,
    const float *hit_n_y,
    const float *hit_n_z,
    const float *hit_geo_n_x,
    const float *hit_geo_n_y,
    const float *hit_geo_n_z,
    int n_hits,
    const unsigned int *triangle_edge_count,
    const int *triangle_edge_indices,
    int max_triangle_edge_slots,
    int n_triangles,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face1,
    int n_edges,
    unsigned int *out_seen_edge_mask
) {
    if (n_hits <= 0 || n_edges <= 0) {
        return;
    }
    int grid_size = (n_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;
    diffraction_discover_edges_kernel<<<grid_size, BLOCK_SIZE>>>(
        tx_x,
        tx_y,
        tx_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
        prim_index,
        hit_p_x,
        hit_p_y,
        hit_p_z,
        hit_n_x,
        hit_n_y,
        hit_n_z,
        hit_geo_n_x,
        hit_geo_n_y,
        hit_geo_n_z,
        n_hits,
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots,
        n_triangles,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_line_min,
        edge_line_max,
        edge_adjacent_face1,
        n_edges,
        out_seen_edge_mask
    );
    throw_cuda(cudaGetLastError(), "diffraction_discover_edges_kernel launch");
}

void monte_carlo_diffraction_build_state_arrays(
    const unsigned int *edge_idx,
    int n_states,
    float tx_x,
    float tx_y,
    float tx_z,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const int *edge_adjacent_face0,
    const int *edge_adjacent_face1,
    int *out_edge_index,
    float *out_edge_pos_x,
    float *out_edge_pos_y,
    float *out_edge_pos_z,
    float *out_edge_dir_x,
    float *out_edge_dir_y,
    float *out_edge_dir_z,
    float *out_n0_x,
    float *out_n0_y,
    float *out_n0_z,
    float *out_nn_x,
    float *out_nn_y,
    float *out_nn_z,
    float *out_wedge_n,
    float *out_line_min,
    float *out_line_max,
    float *out_source_pos_x,
    float *out_source_pos_y,
    float *out_source_pos_z,
    int *out_adjacent_face0,
    int *out_adjacent_face1
) {
    if (n_states <= 0) {
        return;
    }
    int grid_size = (n_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
    diffraction_build_state_arrays_kernel<<<grid_size, BLOCK_SIZE>>>(
        edge_idx,
        n_states,
        tx_x,
        tx_y,
        tx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_wedge_n,
        edge_line_min,
        edge_line_max,
        edge_adjacent_face0,
        edge_adjacent_face1,
        out_edge_index,
        out_edge_pos_x,
        out_edge_pos_y,
        out_edge_pos_z,
        out_edge_dir_x,
        out_edge_dir_y,
        out_edge_dir_z,
        out_n0_x,
        out_n0_y,
        out_n0_z,
        out_nn_x,
        out_nn_y,
        out_nn_z,
        out_wedge_n,
        out_line_min,
        out_line_max,
        out_source_pos_x,
        out_source_pos_y,
        out_source_pos_z,
        out_adjacent_face0,
        out_adjacent_face1
    );
    throw_cuda(cudaGetLastError(), "diffraction_build_state_arrays_kernel launch");
}

} // namespace witwin::channel::native_ext
