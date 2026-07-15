#include <torch/extension.h>

#include "path_block.h"
#include "tensor_checks.h"
#include <rayd/torch/integration.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

extern "C" void channel_native_diffraction_discover_edges(
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, at::Tensor *);
extern "C" void channel_native_diffraction_discover_edges_counted(
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    at::Tensor *);

namespace {

using VisibilityForwardFn = void (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    at::Tensor *,
    at::Tensor *);

using SceneCreateFn = int64_t (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const int64_t *,
    int64_t);

using SceneDestroyFn = void (*)(int64_t);

using SceneEdgeRecordsFn = int64_t (*)(
    int64_t,
    at::Tensor *,
    int64_t);

using IntersectForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    at::Tensor *,
    int64_t);

using TraceReflectionsForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    at::Tensor *,
    int64_t);

using ReflectionEpcPathsForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    at::Tensor *,
    int64_t);

using IntersectBackwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    bool,
    bool,
    bool,
    bool,
    at::Tensor *,
    int64_t);

using IntersectJvpFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    at::Tensor *,
    int64_t);

using TraceReflectionsBackwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

using TraceReflectionsJvpFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

using ReflectionEpcPathsBackwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    bool,
    bool,
    bool,
    at::Tensor *,
    int64_t);

using ReflectionEpcPathsJvpFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

using SceneFaceNormalsBackwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

using SceneFaceNormalsJvpFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

using ReflectionAccumulationForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    double,
    double,
    double,
    double,
    int64_t,
    int64_t,
    double,
    double,
    bool,
    bool,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    at::Tensor *,
    int64_t);

using DiffractionDiscoverEdgesFn = void (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *);

using DiffractionDiscoverEdgesCountedFn = void (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *);

using DiffractionPathsOrder1ForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    at::Tensor *,
    int64_t);

using DiffractionAccumulationForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    double,
    double,
    double,
    double,
    int64_t,
    int64_t,
    double,
    double,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

VisibilityForwardFn raydn_visibility_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_visibility_forward;
}

SceneCreateFn raydn_scene_create_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_scene_create;
}

SceneDestroyFn raydn_scene_destroy_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_scene_destroy;
}

SceneEdgeRecordsFn raydn_scene_edge_records_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_scene_edge_records;
}

IntersectForwardFn raydn_intersect_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_intersect_forward;
}

TraceReflectionsForwardFn raydn_trace_reflections_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_trace_reflections_forward;
}

ReflectionEpcPathsForwardFn raydn_reflection_epc_paths_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_reflection_epc_paths_forward;
}

ReflectionAccumulationForwardFn raydn_reflection_accumulation_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_reflection_accumulation_forward;
}

IntersectBackwardFn raydn_intersect_backward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_intersect_backward;
}

IntersectJvpFn raydn_intersect_jvp_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_intersect_jvp;
}

TraceReflectionsForwardFn raydn_trace_reflections_forward_tape_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_trace_reflections_forward_tape;
}

TraceReflectionsBackwardFn raydn_trace_reflections_backward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_trace_reflections_backward;
}

TraceReflectionsJvpFn raydn_trace_reflections_jvp_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_trace_reflections_jvp;
}

ReflectionEpcPathsBackwardFn raydn_reflection_epc_paths_backward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_reflection_epc_paths_backward;
}

ReflectionEpcPathsJvpFn raydn_reflection_epc_paths_jvp_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_reflection_epc_paths_jvp;
}

SceneFaceNormalsBackwardFn raydn_scene_face_normals_backward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_scene_face_normals_backward;
}

SceneFaceNormalsJvpFn raydn_scene_face_normals_jvp_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_scene_face_normals_jvp;
}

DiffractionDiscoverEdgesFn raydn_diffraction_discover_edges_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &channel_native_diffraction_discover_edges;
}

DiffractionDiscoverEdgesCountedFn raydn_diffraction_discover_edges_counted_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &channel_native_diffraction_discover_edges_counted;
}

DiffractionPathsOrder1ForwardFn raydn_diffraction_paths_order1_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_diffraction_paths_order1_forward;
}

DiffractionAccumulationForwardFn raydn_diffraction_accumulation_forward_fn(std::uintptr_t module_handle) {
    (void) module_handle;
    return &rayd_torch_native_diffraction_accumulation_forward;
}

const at::Tensor *optional_tensor(pybind11::object value, at::Tensor &storage) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>();
    if (!storage.defined())
        return nullptr;
    return &storage;
}

pybind11::object tensor_or_none(const at::Tensor &tensor) {
    if (!tensor.defined())
        return pybind11::none();
    return pybind11::cast(tensor);
}

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

} // namespace


std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
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
    at::Tensor tx_power,
    int64_t tx_power_index);
std::vector<at::Tensor> cn_coupled_rd_prepare_cuda(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max);
pybind11::dict cn_coupled_rd_finalize_cuda(
    at::Tensor prefix_active,
    at::Tensor suffix_visible,
    at::Tensor epc_path_length,
    at::Tensor resolved_face,
    at::Tensor edge_id,
    at::Tensor reflection_point,
    at::Tensor reflection_normal,
    at::Tensor edge_point,
    at::Tensor edge_direction,
    at::Tensor receiver,
    bool reverse);
at::Tensor cn_coupled_active_mask_cuda(at::Tensor lhs, at::Tensor rhs);
PathBlockTuple cn_path_diffraction_block_cuda(
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor field_x_re,
    at::Tensor field_x_im,
    at::Tensor field_y_re,
    at::Tensor field_y_im,
    at::Tensor field_z_re,
    at::Tensor field_z_im,
    int64_t tx_index);
PathBlockTuple cn_path_finalize_blocks_cuda(
    std::vector<at::Tensor> valid_blocks,
    std::vector<at::Tensor> tx_id_blocks,
    std::vector<at::Tensor> rx_id_blocks,
    std::vector<at::Tensor> depth_blocks,
    std::vector<at::Tensor> component_id_blocks,
    std::vector<at::Tensor> primitive_id_blocks,
    std::vector<at::Tensor> edge_id_blocks,
    std::vector<at::Tensor> path_length_blocks,
    std::vector<at::Tensor> delay_blocks,
    std::vector<at::Tensor> path_gain_blocks,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth);

namespace {

using channel_native::check_flat_tensor;
using channel_native::check_vec3_table;

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

pybind11::tuple cn_bdpt_intersect_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 10;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_intersect_forward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        flags,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN intersect forward returned an unexpected output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_bdpt_visibility_forward(
    int64_t scene_handle,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
    raydn_visibility_forward_fn(raydn_module_handle)(
        scene_handle,
        &start,
        &end,
        active_ptr,
        &visible,
        &blocker_prim,
        &tape_t);
    return pybind11::make_tuple(visible, blocker_prim, tape_t);
}

pybind11::tuple cn_raydn_trace_reflections_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_forward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        max_bounces,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection trace returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    pybind11::object active,
    torch::Tensor expected_prim_ids,
    torch::Tensor direct_plane_points,
    torch::Tensor direct_plane_normals,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_forward_fn(raydn_module_handle)(
        scene_handle,
        &source,
        &receiver,
        active_ptr,
        &expected_prim_ids,
        &direct_plane_points,
        &direct_plane_normals,
        &surface_group_id,
        &surface_group_size,
        &surface_group_members,
        max_bounces,
        visibility_ignore_mode,
        1.0e-3,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection EPC path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_intersect_backward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    pybind11::object grad_t,
    pybind11::object grad_p,
    pybind11::object grad_n,
    pybind11::object grad_geo_n,
    pybind11::object grad_uv,
    pybind11::object grad_barycentric,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor grad_t_storage;
    at::Tensor grad_p_storage;
    at::Tensor grad_n_storage;
    at::Tensor grad_geo_n_storage;
    at::Tensor grad_uv_storage;
    at::Tensor grad_barycentric_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *grad_t_ptr = optional_tensor(std::move(grad_t), grad_t_storage);
    const at::Tensor *grad_p_ptr = optional_tensor(std::move(grad_p), grad_p_storage);
    const at::Tensor *grad_n_ptr = optional_tensor(std::move(grad_n), grad_n_storage);
    const at::Tensor *grad_geo_n_ptr = optional_tensor(std::move(grad_geo_n), grad_geo_n_storage);
    const at::Tensor *grad_uv_ptr = optional_tensor(std::move(grad_uv), grad_uv_storage);
    const at::Tensor *grad_barycentric_ptr =
        optional_tensor(std::move(grad_barycentric), grad_barycentric_storage);
    constexpr int64_t kOutputCount = 4;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_intersect_backward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        grad_t_ptr,
        grad_p_ptr,
        grad_n_ptr,
        grad_geo_n_ptr,
        grad_uv_ptr,
        grad_barycentric_ptr,
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d,
        need_grad_ray_tmax,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN intersect backward returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_intersect_jvp(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    int64_t flags,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *tangent_vertices_ptr =
        optional_tensor(std::move(tangent_vertices), tangent_vertices_storage);
    const at::Tensor *tangent_ray_o_ptr =
        optional_tensor(std::move(tangent_ray_o), tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d_ptr =
        optional_tensor(std::move(tangent_ray_d), tangent_ray_d_storage);
    constexpr int64_t kOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_intersect_jvp_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        tangent_vertices_ptr,
        tangent_ray_o_ptr,
        tangent_ray_d_ptr,
        flags,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN intersect jvp returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_forward_tape(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 9;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_forward_tape_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        max_bounces,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN tape reflection trace returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_backward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    torch::Tensor image_sources,
    pybind11::object grad_t,
    pybind11::object grad_image_sources,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor grad_t_storage;
    at::Tensor grad_image_sources_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *grad_t_ptr = optional_tensor(std::move(grad_t), grad_t_storage);
    const at::Tensor *grad_image_sources_ptr =
        optional_tensor(std::move(grad_image_sources), grad_image_sources_storage);
    constexpr int64_t kOutputCount = 4;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_backward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        &tape_hit_points,
        &tape_normals,
        &image_sources,
        grad_t_ptr,
        grad_image_sources_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection backward returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_jvp(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    torch::Tensor image_sources,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *tangent_vertices_ptr =
        optional_tensor(std::move(tangent_vertices), tangent_vertices_storage);
    const at::Tensor *tangent_ray_o_ptr =
        optional_tensor(std::move(tangent_ray_o), tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d_ptr =
        optional_tensor(std::move(tangent_ray_d), tangent_ray_d_storage);
    constexpr int64_t kOutputCount = 2;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_jvp_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        &tape_hit_points,
        &tape_normals,
        tangent_vertices_ptr,
        tangent_ray_o_ptr,
        tangent_ray_d_ptr,
        &image_sources,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection jvp returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_backward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object grad_points,
    pybind11::object grad_normals,
    pybind11::object grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver,
    std::uintptr_t raydn_module_handle) {
    at::Tensor grad_points_storage;
    at::Tensor grad_normals_storage;
    at::Tensor grad_path_length_storage;
    const at::Tensor *grad_points_ptr =
        optional_tensor(std::move(grad_points), grad_points_storage);
    const at::Tensor *grad_normals_ptr =
        optional_tensor(std::move(grad_normals), grad_normals_storage);
    const at::Tensor *grad_path_length_ptr =
        optional_tensor(std::move(grad_path_length), grad_path_length_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_backward_fn(raydn_module_handle)(
        scene_handle,
        &source,
        &receiver,
        &sequence,
        &plane_points,
        &plane_normals,
        &valid,
        &bounce_count,
        grad_points_ptr,
        grad_normals_ptr,
        grad_path_length_ptr,
        need_grad_vertices,
        need_grad_source,
        need_grad_receiver,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection EPC paths backward returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_jvp(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object tangent_vertices,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver,
    std::uintptr_t raydn_module_handle) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_source_storage;
    at::Tensor tangent_receiver_storage;
    const at::Tensor *tangent_vertices_ptr =
        optional_tensor(std::move(tangent_vertices), tangent_vertices_storage);
    const at::Tensor *tangent_source_ptr =
        optional_tensor(std::move(tangent_source), tangent_source_storage);
    const at::Tensor *tangent_receiver_ptr =
        optional_tensor(std::move(tangent_receiver), tangent_receiver_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_jvp_fn(raydn_module_handle)(
        scene_handle,
        &source,
        &receiver,
        &sequence,
        &plane_points,
        &plane_normals,
        &valid,
        &bounce_count,
        tangent_vertices_ptr,
        tangent_source_ptr,
        tangent_receiver_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection EPC paths jvp returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

at::Tensor cn_raydn_scene_face_normals_backward(
    int64_t scene_handle,
    torch::Tensor grad_face_normals,
    std::uintptr_t raydn_module_handle) {
    constexpr int64_t kOutputCount = 1;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_face_normals_backward_fn(raydn_module_handle)(
        scene_handle,
        &grad_face_normals,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN scene face normals backward returned an invalid output count");
    return outputs[0];
}

at::Tensor cn_raydn_scene_face_normals_jvp(
    int64_t scene_handle,
    torch::Tensor tangent_vertices,
    std::uintptr_t raydn_module_handle) {
    constexpr int64_t kOutputCount = 1;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_face_normals_jvp_fn(raydn_module_handle)(
        scene_handle,
        &tangent_vertices,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN scene face normals jvp returned an invalid output count");
    return outputs[0];
}

pybind11::tuple cn_bdpt_reflection_accumulation_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    torch::Tensor active,
    torch::Tensor tx,
    torch::Tensor tx_pol,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool streaming_los_enabled,
    std::uintptr_t raydn_module_handle) {
    if (accumulation_strategy == 4) {
        if (procedural_sample_count <= 0)
            throw std::runtime_error("streaming_planar requires a positive procedural_sample_count");
        if (tx.dim() != 2 || tx.size(0) != 1 || tx.size(1) != 3)
            throw std::runtime_error("streaming_planar expects one transmitter position");

        const auto dopts = ray_o.options().dtype(at::kDouble);
        at::Tensor index = at::arange(procedural_sample_count, dopts);
        at::Tensor azimuth = index / 1.618033988749894848204586834365638118;
        at::Tensor azimuth_u = azimuth - at::floor(azimuth);
        at::Tensor phi = azimuth_u * (2.0 * 3.141592653589793238462643383279502884);
        at::Tensor elevation = procedural_sample_count == 1
            ? at::zeros_like(index)
            : index / static_cast<double>(procedural_sample_count - 1);
        at::Tensor z = 1.0 - 2.0 * elevation;
        at::Tensor radial = at::sqrt(at::clamp_min(1.0 - z * z, 0.0));
        at::Tensor x = radial * at::cos(phi);
        at::Tensor y = radial * at::sin(phi);
        ray_d = at::stack({x, y, z}, 1).to(ray_o.scalar_type()).contiguous();

        at::Tensor pole = radial <= 1.0e-6;
        at::Tensor safe_radial = at::clamp_min(radial, 1.0e-6);
        at::Tensor pole_x = at::where(z >= 0.0, at::ones_like(z), -at::ones_like(z));
        at::Tensor pol_x = at::where(pole, pole_x, z * x / safe_radial);
        at::Tensor pol_y = at::where(pole, at::zeros_like(z), z * y / safe_radial);
        at::Tensor pol_z = at::where(pole, at::zeros_like(z), -radial);
        tx_pol = at::stack({pol_x, pol_y, pol_z}, 1).to(ray_o.scalar_type()).contiguous();

        ray_o = tx.select(0, 0).reshape({1, 3})
                    .expand({procedural_sample_count, 3}).contiguous();
        tx = ray_o;
        ray_tmax = at::empty({0}, ray_o.options());
        active = at::ones({procedural_sample_count}, ray_o.options().dtype(at::kBool));
        accumulation_strategy = 1;
        procedural_sample_count = 0;
    }
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_accumulation_forward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        &active,
        &tx,
        &tx_pol,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        max_bounces,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        wavelength,
        solid_angle_per_ray,
        collect_wedges,
        collect_wedge_prefixes,
        wedge_capacity,
        wedge_sample_stride,
        accumulation_strategy,
        compact_min_samples,
        staged_min_samples_per_cell,
        procedural_sample_count,
        streaming_los_enabled,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection accumulation returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

torch::Tensor cn_bdpt_diffraction_discover_edges(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

torch::Tensor cn_bdpt_diffraction_discover_edges_counted(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor hit_count,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_counted_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &hit_count,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

pybind11::tuple cn_raydn_diffraction_paths_order1_forward(
    int64_t scene_handle,
    torch::Tensor tx_pos,
    torch::Tensor rx_pos,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    at::Tensor tx_pol = at::zeros_like(tx_pos);
    if (tx_pol.numel() != 0)
        tx_pol.select(1, 2).fill_(1.0);
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
        scene_handle,
        &tx_pos,
        &tx_pol,
        &rx_pos,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        state_limit,
        capacity,
        wavelength,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction order-1 path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::dict cn_path_diffraction_paths_order1(
    int64_t scene_handle,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor selected,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor line_min,
    torch::Tensor line_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor exterior_angle,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    check_vec3_table(tx_positions, "tx_positions");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(rx_positions, "rx_positions");
    check_flat_tensor(selected, "selected", at::kBool);
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_flat_tensor(line_min, "line_min", at::kFloat);
    check_flat_tensor(line_max, "line_max", at::kFloat);
    check_vec3_table(n0, "n0");
    check_vec3_table(n1, "n1");
    check_flat_tensor(face0, "face0", at::kInt);
    check_flat_tensor(face1, "face1", at::kInt);
    check_flat_tensor(exterior_angle, "exterior_angle", at::kFloat);
    check_flat_tensor(material_eta_r, "material_eta_r", at::kFloat);
    check_flat_tensor(material_sigma, "material_sigma", at::kFloat);
    check_flat_tensor(material_mu_r, "material_mu_r", at::kFloat);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(selected.size(0) == edge_pos.size(0), "selected must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge_pos");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge_pos");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge_pos");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge_pos");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge_pos");
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    const int device = tx_positions.get_device();
    for (const auto& tensor : {
             tx_power,
             rx_positions,
             selected,
             edge_pos,
             edge_dir,
             line_min,
             line_max,
             n0,
             n1,
             face0,
             face1,
             exterior_angle,
             material_gain,
             material_valid,
         }) {
        TORCH_CHECK(tensor.get_device() == device, "diffraction path tensors must share one CUDA device");
    }

    const int64_t tx_count = tx_positions.size(0);
    PathBlockLists blocks;
    blocks.reserve(static_cast<size_t>(tx_count));

    constexpr int64_t kOutputCount = 18;
    for (int64_t tx_index = 0; tx_index < tx_count; ++tx_index) {
        at::Tensor tx = tx_positions.select(0, tx_index);
        std::vector<at::Tensor> states = cn_deterministic_diffraction_state_pack_selected_cuda(
            selected,
            edge_pos,
            edge_dir,
            line_min,
            line_max,
            n0,
            n1,
            face0,
            face1,
            exterior_angle,
            tx,
            tx_power,
            tx_index);
        TORCH_CHECK(states.size() == 12, "diffraction state pack must return 12 tensors");

        at::Tensor tx_view = tx_positions.narrow(0, tx_index, 1);
        at::Tensor tx_pol = at::zeros_like(tx_view);
        if (tx_pol.numel() != 0)
            tx_pol.select(1, 2).fill_(1.0);
        const int64_t state_limit = states[0].size(0);
        const int64_t capacity = rx_positions.size(0) * state_limit;
        std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
        const int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
            scene_handle,
            &tx_view,
            &tx_pol,
            &rx_positions,
            // The packed state table keeps one row per edge; the selection
            // mask must gate the launch so deselected (e.g. merged duplicate)
            // records never emit paths.
            &selected,
            &states[0],
            &states[1],
            &states[2],
            &states[3],
            &states[4],
            &states[5],
            &states[6],
            &states[7],
            &states[8],
            &states[9],
            &states[10],
            &states[11],
            &material_eta_r,
            &material_sigma,
            &material_mu_r,
            &material_gain,
            &material_valid,
            state_limit,
            capacity,
            wavelength,
            outputs.data(),
            kOutputCount);
        TORCH_CHECK(output_count == kOutputCount, "RayDN order-1 diffraction path export returned an unexpected output count");
        PathBlockTuple block = cn_path_diffraction_block_cuda(
            outputs[1],
            outputs[3],
            outputs[4],
            outputs[5],
            outputs[8],
            outputs[9],
            outputs[10],
            outputs[11],
            outputs[12],
            outputs[13],
            outputs[14],
            tx_index);
        blocks.append(block);
    }

    return path_block_dict(cn_path_finalize_blocks_cuda(
        blocks.valid,
        blocks.tx_id,
        blocks.rx_id,
        blocks.depth,
        blocks.component_id,
        blocks.primitive_id,
        blocks.edge_id,
        blocks.path_length,
        blocks.delay,
        blocks.path_gain,
        -1,
        tx_count,
        1));
}

pybind11::tuple cn_bdpt_diffraction_accumulation_forward(
    int64_t scene_handle,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    pybind11::object state_wi,
    pybind11::object state_d0,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    int64_t recursive_state_limit,
    pybind11::object recursive_active,
    pybind11::object recursive_state_edge_index,
    pybind11::object recursive_state_edge_pos,
    pybind11::object recursive_state_edge_dir,
    pybind11::object recursive_state_edge_t_min,
    pybind11::object recursive_state_edge_t_max,
    pybind11::object recursive_state_n0,
    pybind11::object recursive_state_n1,
    pybind11::object recursive_state_prim0,
    pybind11::object recursive_state_prim1,
    pybind11::object recursive_state_exterior_angle,
    int64_t export_tape,
    pybind11::object sample_state_index,
    pybind11::object sample_edge_weight,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor state_wi_storage;
    at::Tensor state_d0_storage;
    at::Tensor recursive_active_storage;
    at::Tensor recursive_state_edge_index_storage;
    at::Tensor recursive_state_edge_pos_storage;
    at::Tensor recursive_state_edge_dir_storage;
    at::Tensor recursive_state_edge_t_min_storage;
    at::Tensor recursive_state_edge_t_max_storage;
    at::Tensor recursive_state_n0_storage;
    at::Tensor recursive_state_n1_storage;
    at::Tensor recursive_state_prim0_storage;
    at::Tensor recursive_state_prim1_storage;
    at::Tensor recursive_state_exterior_angle_storage;
    at::Tensor sample_state_index_storage;
    at::Tensor sample_edge_weight_storage;

    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *state_wi_ptr = optional_tensor(std::move(state_wi), state_wi_storage);
    const at::Tensor *state_d0_ptr = optional_tensor(std::move(state_d0), state_d0_storage);
    const at::Tensor *recursive_active_ptr = optional_tensor(std::move(recursive_active), recursive_active_storage);
    const at::Tensor *recursive_state_edge_index_ptr =
        optional_tensor(std::move(recursive_state_edge_index), recursive_state_edge_index_storage);
    const at::Tensor *recursive_state_edge_pos_ptr =
        optional_tensor(std::move(recursive_state_edge_pos), recursive_state_edge_pos_storage);
    const at::Tensor *recursive_state_edge_dir_ptr =
        optional_tensor(std::move(recursive_state_edge_dir), recursive_state_edge_dir_storage);
    const at::Tensor *recursive_state_edge_t_min_ptr =
        optional_tensor(std::move(recursive_state_edge_t_min), recursive_state_edge_t_min_storage);
    const at::Tensor *recursive_state_edge_t_max_ptr =
        optional_tensor(std::move(recursive_state_edge_t_max), recursive_state_edge_t_max_storage);
    const at::Tensor *recursive_state_n0_ptr = optional_tensor(std::move(recursive_state_n0), recursive_state_n0_storage);
    const at::Tensor *recursive_state_n1_ptr = optional_tensor(std::move(recursive_state_n1), recursive_state_n1_storage);
    const at::Tensor *recursive_state_prim0_ptr =
        optional_tensor(std::move(recursive_state_prim0), recursive_state_prim0_storage);
    const at::Tensor *recursive_state_prim1_ptr =
        optional_tensor(std::move(recursive_state_prim1), recursive_state_prim1_storage);
    const at::Tensor *recursive_state_exterior_angle_ptr =
        optional_tensor(std::move(recursive_state_exterior_angle), recursive_state_exterior_angle_storage);
    const at::Tensor *sample_state_index_ptr =
        optional_tensor(std::move(sample_state_index), sample_state_index_storage);
    const at::Tensor *sample_edge_weight_ptr =
        optional_tensor(std::move(sample_edge_weight), sample_edge_weight_storage);

    constexpr int64_t kOutputCount = 19;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_accumulation_forward_fn(raydn_module_handle)(
        scene_handle,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        state_wi_ptr,
        state_d0_ptr,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        state_limit,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        grid_cell_area,
        wavelength,
        direct_samples,
        keller_samples,
        suffix_samples,
        seed,
        max_order,
        recursive_state_limit,
        recursive_active_ptr,
        recursive_state_edge_index_ptr,
        recursive_state_edge_pos_ptr,
        recursive_state_edge_dir_ptr,
        recursive_state_edge_t_min_ptr,
        recursive_state_edge_t_max_ptr,
        recursive_state_n0_ptr,
        recursive_state_n1_ptr,
        recursive_state_prim0_ptr,
        recursive_state_prim1_ptr,
        recursive_state_exterior_angle_ptr,
        export_tape,
        sample_state_index_ptr,
        sample_edge_weight_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction accumulation returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::dict cn_raydn_coupled_rd_geometry_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor face_id,
    torch::Tensor plane_point,
    torch::Tensor plane_normal,
    torch::Tensor edge_id,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_t_min,
    torch::Tensor edge_t_max,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    bool reverse,
    std::uintptr_t raydn_module_handle) {
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    check_flat_tensor(face_id, "face_id", at::kInt);
    check_vec3_table(plane_point, "plane_point");
    check_vec3_table(plane_normal, "plane_normal");
    check_flat_tensor(edge_id, "edge_id", at::kInt);
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_flat_tensor(edge_t_min, "edge_t_min", at::kFloat);
    check_flat_tensor(edge_t_max, "edge_t_max", at::kFloat);
    check_flat_tensor(surface_group_id, "surface_group_id", at::kInt);
    check_flat_tensor(surface_group_size, "surface_group_size", at::kInt);
    check_flat_tensor(surface_group_members, "surface_group_members", at::kInt);
    const int64_t count = source.size(0);
    TORCH_CHECK(receiver.size(0) == count, "receiver must match source rows");
    TORCH_CHECK(face_id.size(0) == count && edge_id.size(0) == count,
                "face_id and edge_id must match source rows");
    for (const auto &tensor : {plane_point, plane_normal, edge_pos, edge_dir})
        TORCH_CHECK(tensor.size(0) == count, "coupled geometry vector tables must match source rows");
    TORCH_CHECK(edge_t_min.size(0) == count && edge_t_max.size(0) == count,
                "edge bounds must match source rows");
    TORCH_CHECK(surface_group_size.numel() > 0,
                "surface_group_size must contain at least one group");
    TORCH_CHECK(surface_group_members.numel() % surface_group_size.numel() == 0,
                "surface_group_members must be padded by group count");

    // D->R is the reciprocal R->D problem with endpoints exchanged. The
    // output interaction sequence is reversed again by the finalize kernel.
    at::Tensor epc_source = reverse ? receiver : source;
    at::Tensor epc_receiver = reverse ? source : receiver;
    std::vector<at::Tensor> prepared = cn_coupled_rd_prepare_cuda(
        epc_source,
        epc_receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max);
    TORCH_CHECK(prepared.size() == 4, "coupled R-D prepare returned an unexpected tensor count");
    at::Tensor candidate_active = prepared[0];
    at::Tensor diffraction_point = prepared[1];
    at::Tensor expected_faces = face_id.reshape({count, 1}).contiguous();
    at::Tensor direct_plane_points = plane_point.reshape({count, 1, 3}).contiguous();
    at::Tensor direct_plane_normals = plane_normal.reshape({count, 1, 3}).contiguous();

    constexpr int64_t kEpcOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kEpcOutputCount)> epc;
    const int64_t epc_output_count = raydn_reflection_epc_paths_forward_fn(raydn_module_handle)(
        scene_handle,
        &epc_source,
        &diffraction_point,
        &candidate_active,
        &expected_faces,
        &direct_plane_points,
        &direct_plane_normals,
        &surface_group_id,
        &surface_group_size,
        &surface_group_members,
        1,
        1,
        1.0e-3,
        epc.data(),
        kEpcOutputCount);
    TORCH_CHECK(epc_output_count == kEpcOutputCount,
                "RayDN reflection EPC returned an unexpected tensor count for coupled R-D geometry");

    at::Tensor prefix_active = cn_coupled_active_mask_cuda(candidate_active, epc[0]);
    at::Tensor suffix_visible;
    at::Tensor suffix_blocker;
    at::Tensor suffix_tape_t;
    raydn_visibility_forward_fn(raydn_module_handle)(
        scene_handle,
        &diffraction_point,
        &epc_receiver,
        &prefix_active,
        &suffix_visible,
        &suffix_blocker,
        &suffix_tape_t);
    at::Tensor resolved_face = epc[2].select(1, 0).contiguous();
    at::Tensor reflection_position = epc[4].select(1, 0).contiguous();
    at::Tensor reflection_normal = epc[5].select(1, 0).contiguous();
    pybind11::dict out = cn_coupled_rd_finalize_cuda(
        prefix_active,
        suffix_visible,
        epc[1].contiguous(),
        resolved_face,
        edge_id,
        reflection_position,
        reflection_normal,
        diffraction_point,
        edge_dir,
        epc_receiver,
        reverse);
    out["candidate_active"] = candidate_active;
    out["virtual_source"] = prepared[2];
    out["predicted_reflection_position"] = prepared[3];
    out["suffix_blocker_primitive"] = suffix_blocker;
    return out;
}
