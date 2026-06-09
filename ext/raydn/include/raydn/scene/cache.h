#pragma once

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <optix.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace raydn {

struct MeshRecord {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor to_world_left;
    at::Tensor to_world_right;
    bool use_face_normals = false;
    bool edges_enabled = true;
    bool dynamic = false;
    bool pending_update = false;
};

struct OptixTriangleAccel {
    at::Tensor vertex_buffer;
    at::Tensor index_buffer;
    at::Tensor gas_buffer;
    at::Tensor gas_temp_buffer;
    OptixTraversableHandle traversable = 0;
};

struct OptixEdgeAccel {
    at::Tensor aabb_buffer;
    at::Tensor gas_buffer;
    at::Tensor gas_temp_buffer;
    OptixTraversableHandle traversable = 0;
    float search_radius = 0.0f;
};

struct OptixInstanceAccel {
    at::Tensor instance_buffer;
    at::Tensor ias_buffer;
    at::Tensor ias_temp_buffer;
    OptixTraversableHandle traversable = 0;
};

struct SceneCache {
    int64_t handle = 0;
    int64_t version = 1;
    int64_t edge_version = 1;
    int64_t device_index = 0;
    std::vector<MeshRecord> meshes;
    std::vector<OptixTriangleAccel> triangle_accels;
    OptixInstanceAccel triangle_ias;
    at::Tensor global_vertices;
    at::Tensor global_faces;
    at::Tensor face_offsets;
    at::Tensor face_shape_id;
    at::Tensor face_local_id;
    at::Tensor tri_p0_x;
    at::Tensor tri_p0_y;
    at::Tensor tri_p0_z;
    at::Tensor tri_e1_x;
    at::Tensor tri_e1_y;
    at::Tensor tri_e1_z;
    at::Tensor tri_e2_x;
    at::Tensor tri_e2_y;
    at::Tensor tri_e2_z;
    at::Tensor tri_fn_x;
    at::Tensor tri_fn_y;
    at::Tensor tri_fn_z;
    at::Tensor edge_v0;
    at::Tensor edge_v1;
    at::Tensor edge_face0;
    at::Tensor edge_face1;
    at::Tensor edge_shape_id;
    at::Tensor edge_local_id;
    at::Tensor edge_p0_x;
    at::Tensor edge_p0_y;
    at::Tensor edge_p0_z;
    at::Tensor edge_e1_x;
    at::Tensor edge_e1_y;
    at::Tensor edge_e1_z;
    at::Tensor edge_mask;
    at::Tensor edge_opposite;
    OptixEdgeAccel edge_accel;
    std::vector<OptixEdgeAccel> edge_accels;
    at::Tensor tri_p0_packed;
    at::Tensor tri_e1_packed;
    at::Tensor tri_e2_packed;
    at::Tensor tri_fn_packed;
};

struct SceneHandle : torch::CustomClassHolder {
    explicit SceneHandle(int64_t handle_, bool owns_handle_ = true)
        : handle(handle_), owns_handle(owns_handle_) {}
    ~SceneHandle();
    int64_t handle = 0;
    bool owns_handle = true;
};

std::unique_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes);
int64_t create_scene(std::vector<MeshRecord> meshes);
void destroy_scene(int64_t handle);
SceneCache &get_scene(int64_t handle);
c10::intrusive_ptr<SceneHandle> create_scene_cache_from_flat(
    std::vector<at::Tensor> vertices,
    std::vector<at::Tensor> faces,
    std::vector<at::Tensor> uv,
    std::vector<at::Tensor> face_uv,
    std::vector<at::Tensor> to_world_left,
    std::vector<at::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags);
int64_t scene_version(int64_t handle);
int64_t scene_num_meshes(int64_t handle);
int64_t scene_edge_count(int64_t handle);
void update_mesh_vertices(int64_t handle, int64_t mesh_id, at::Tensor vertices);
void sync_scene(int64_t handle);
int64_t scene_version(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_num_meshes(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_edge_count(c10::intrusive_ptr<SceneHandle> scene);
void update_mesh_vertices(c10::intrusive_ptr<SceneHandle> scene, int64_t mesh_id, at::Tensor vertices);
void sync_scene(c10::intrusive_ptr<SceneHandle> scene);
std::vector<at::Tensor> split_scene_vertex_grad(c10::intrusive_ptr<SceneHandle> scene, at::Tensor grad_vertices);
at::Tensor pack_scene_vertex_tangents(
    c10::intrusive_ptr<SceneHandle> scene,
    std::vector<c10::optional<at::Tensor>> tangents);

} // namespace raydn
