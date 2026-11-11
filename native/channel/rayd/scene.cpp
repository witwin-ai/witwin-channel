// Copyright Xingyu Chen.
// Implements scene native integration.

#include "resource.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

rayd::torch::SceneTraceBackend parse_trace_backend(const std::string &name) {
    if (name == "auto")
        return rayd::torch::SceneTraceBackend::Auto;
    if (name == "optix")
        return rayd::torch::SceneTraceBackend::Optix;
    if (name == "cuda")
        return rayd::torch::SceneTraceBackend::Cuda;
    throw std::invalid_argument(
        "trace backend must be 'auto', 'optix' or 'cuda', got '" + name + "'");
}

}  // namespace

std::string RayDSceneResource::trace_backend() const {
    switch (rayd::torch::scene_trace_backend(resource_)) {
    case rayd::torch::SceneTraceBackend::Optix:
        return "optix";
    case rayd::torch::SceneTraceBackend::Cuda:
        return "cuda";
    case rayd::torch::SceneTraceBackend::Auto:
        break;
    }
    throw std::runtime_error("RayD reported an unresolved scene trace backend");
}

const rayd::torch::SceneEdgeRecordsResult &RayDSceneResource::edge_records() {
    if (!edge_records_.has_value()) {
        edge_records_.emplace(rayd::torch::scene_edge_records(resource_));
    }
    return *edge_records_;
}

void RayDSceneResource::update_vertices(
    const std::vector<int64_t> &mesh_indices,
    const std::vector<at::Tensor> &vertices) {
    if (mesh_indices.size() != vertices.size())
        throw std::runtime_error("rayd_scene_update_vertices needs one vertex tensor per mesh index");
    edge_records_.reset();
    for (size_t index = 0; index < mesh_indices.size(); ++index)
        rayd::torch::update_mesh_vertices(resource_, mesh_indices[index], vertices[index]);
    rayd::torch::sync_scene(resource_);
}

void channel_rayd_scene_update_vertices(
    RayDSceneResource &scene,
    std::vector<int64_t> mesh_indices,
    std::vector<torch::Tensor> vertices) {
    scene.update_vertices(mesh_indices, vertices);
}

std::shared_ptr<RayDSceneResource> channel_rayd_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags,
    const std::string &trace_backend) {
    rayd::torch::SceneOptions options;
    options.trace_backend = parse_trace_backend(trace_backend);
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
        rayd::torch::create_scene(std::move(meshes), options));
}

pybind11::dict channel_runtime_probe(int64_t device_index, bool probe_optix) {
    pybind11::dict probe;
    int driver_version = 0;
    const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
    probe["cuda_driver_version"] = driver_status == cudaSuccess ? driver_version : 0;
    if (!probe_optix)
        return probe;
    const rayd::torch::OptixAvailability optix =
        rayd::torch::optix_availability(static_cast<int>(device_index));
    probe["optix_available"] = optix.available;
    probe["optix_reason"] = optix.reason;
    return probe;
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
