#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace raydn {

struct EdgeSearchStats {
    bool has_edges = false;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    float max_z = 0.0f;
    float max_edge_length = 0.0f;
};

struct EdgeTopology {
    at::Tensor edge_v0;
    at::Tensor edge_v1;
    at::Tensor edge_face0;
    at::Tensor edge_face1;
    at::Tensor edge_opposite;
    at::Tensor edge_shape_id;
    at::Tensor edge_local_id;
};

EdgeTopology build_edge_topology_cuda(
    const at::Tensor &faces,
    int32_t vertex_offset,
    int32_t shape_id);

void pack_global_geometry_cuda(
    const at::Tensor &mesh_vertices,
    const at::Tensor &mesh_faces,
    int32_t vertex_offset,
    int32_t face_offset,
    int32_t shape_id,
    at::Tensor &global_vertices,
    at::Tensor &global_faces,
    at::Tensor &face_shape_id,
    at::Tensor &face_local_id);

void pack_global_vertex_tangent_cuda(
    const at::Tensor &mesh_tangent,
    int64_t vertex_offset,
    int64_t vertex_count,
    at::Tensor &global_tangent);

void zero_global_vertex_tangent_range_cuda(
    int64_t vertex_offset,
    int64_t vertex_count,
    at::Tensor &global_tangent);

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
    at::Tensor &tri_fn_packed);

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
    at::Tensor &edge_e1_z);

EdgeSearchStats compute_edge_search_stats_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z);

} // namespace raydn
