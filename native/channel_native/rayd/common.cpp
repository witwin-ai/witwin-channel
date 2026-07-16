#include "bridge.h"

#include <rayd/torch/integration.h>

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

namespace channel_native::rayd_bridge {

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

}  // namespace channel_native::rayd_bridge
