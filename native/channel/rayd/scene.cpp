#include "resource.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

const rayd::torch::SceneEdgeRecordsResult &RayDSceneResource::edge_records() {
    if (!edge_records_.has_value()) {
        edge_records_.emplace(rayd::torch::scene_edge_records(resource_));
    }
    return *edge_records_;
}

std::shared_ptr<RayDSceneResource> channel_rayd_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags) {
    const size_t mesh_count = vertices.size();
    if (mesh_count == 0)
        throw std::runtime_error("rayd_scene_create requires at least one mesh");
    if (faces.size() != mesh_count ||
        uv.size() != mesh_count ||
        face_uv.size() != mesh_count ||
        to_world_left.size() != mesh_count ||
        to_world_right.size() != mesh_count ||
        mesh_flags.size() != mesh_count) {
        throw std::runtime_error("rayd_scene_create input lists must have the same length");
    }

    std::vector<rayd::torch::MeshInput> meshes;
    meshes.reserve(mesh_count);
    for (size_t index = 0; index < mesh_count; ++index) {
        const int64_t flags = mesh_flags[index];
        meshes.push_back({
            std::move(vertices[index]),
            std::move(faces[index]),
            std::move(uv[index]),
            std::move(face_uv[index]),
            std::move(to_world_left[index]),
            std::move(to_world_right[index]),
            (flags & 1) != 0,
            (flags & 2) != 0,
            (flags & 4) != 0,
        });
    }
    return std::make_shared<RayDSceneResource>(
        rayd::torch::create_scene(std::move(meshes)));
}

pybind11::tuple channel_rayd_scene_edge_records(RayDSceneResource &scene) {
    const rayd::torch::SceneEdgeRecordsResult &out = scene.edge_records();
    return pybind11::make_tuple(
        out.global_vertices,
        out.global_faces,
        out.tri_fn_x,
        out.tri_fn_y,
        out.tri_fn_z,
        out.edge_v0,
        out.edge_v1,
        out.edge_face0_global,
        out.edge_face1_global,
        out.edge_shape_id,
        out.edge_local_id,
        out.edge_opposite);
}
