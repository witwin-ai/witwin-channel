#include "bridge.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using channel_native::rayd_bridge::raydn_scene_create_fn;
using channel_native::rayd_bridge::raydn_scene_destroy_fn;
using channel_native::rayd_bridge::raydn_scene_edge_records_fn;

struct RaydnSceneOwner {
    int64_t scene_handle = 0;
    std::uintptr_t module_handle = 0;
};

void destroy_raydn_scene_capsule(PyObject *capsule) {
    void *raw = PyCapsule_GetPointer(capsule, "channel_native.raydn_scene");
    auto *owner = reinterpret_cast<RaydnSceneOwner *>(raw);
    if (owner == nullptr)
        return;
    try {
        if (owner->scene_handle != 0)
            raydn_scene_destroy_fn(owner->module_handle)(owner->scene_handle);
    } catch (...) {
    }
    delete owner;
}

}  // namespace

pybind11::tuple cn_raydn_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags,
    std::uintptr_t raydn_module_handle) {
    const size_t mesh_count = vertices.size();
    if (mesh_count == 0)
        throw std::runtime_error("raydn_scene_create requires at least one mesh");
    if (faces.size() != mesh_count ||
        uv.size() != mesh_count ||
        face_uv.size() != mesh_count ||
        to_world_left.size() != mesh_count ||
        to_world_right.size() != mesh_count ||
        mesh_flags.size() != mesh_count) {
        throw std::runtime_error("raydn_scene_create input lists must have the same length");
    }
    int64_t scene_handle = raydn_scene_create_fn(raydn_module_handle)(
        vertices.data(),
        faces.data(),
        uv.data(),
        face_uv.data(),
        to_world_left.data(),
        to_world_right.data(),
        mesh_flags.data(),
        static_cast<int64_t>(mesh_count));
    auto *owner = new RaydnSceneOwner{scene_handle, raydn_module_handle};
    pybind11::capsule capsule(owner, "channel_native.raydn_scene", destroy_raydn_scene_capsule);
    return pybind11::make_tuple(scene_handle, capsule);
}

pybind11::tuple cn_raydn_scene_edge_records(
    int64_t scene_handle,
    std::uintptr_t raydn_module_handle) {
    constexpr int64_t kOutputCount = 12;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_edge_records_fn(raydn_module_handle)(
        scene_handle,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN scene edge records returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}
