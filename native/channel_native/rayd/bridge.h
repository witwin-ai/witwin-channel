#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace channel_native::rayd_bridge {

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

VisibilityForwardFn raydn_visibility_forward_fn(std::uintptr_t module_handle);
SceneCreateFn raydn_scene_create_fn(std::uintptr_t module_handle);
SceneDestroyFn raydn_scene_destroy_fn(std::uintptr_t module_handle);
SceneEdgeRecordsFn raydn_scene_edge_records_fn(std::uintptr_t module_handle);
IntersectForwardFn raydn_intersect_forward_fn(std::uintptr_t module_handle);
TraceReflectionsForwardFn raydn_trace_reflections_forward_fn(std::uintptr_t module_handle);
ReflectionEpcPathsForwardFn raydn_reflection_epc_paths_forward_fn(std::uintptr_t module_handle);
ReflectionAccumulationForwardFn raydn_reflection_accumulation_forward_fn(std::uintptr_t module_handle);
IntersectBackwardFn raydn_intersect_backward_fn(std::uintptr_t module_handle);
IntersectJvpFn raydn_intersect_jvp_fn(std::uintptr_t module_handle);
TraceReflectionsForwardFn raydn_trace_reflections_forward_tape_fn(std::uintptr_t module_handle);
TraceReflectionsBackwardFn raydn_trace_reflections_backward_fn(std::uintptr_t module_handle);
TraceReflectionsJvpFn raydn_trace_reflections_jvp_fn(std::uintptr_t module_handle);
ReflectionEpcPathsBackwardFn raydn_reflection_epc_paths_backward_fn(std::uintptr_t module_handle);
ReflectionEpcPathsJvpFn raydn_reflection_epc_paths_jvp_fn(std::uintptr_t module_handle);
SceneFaceNormalsBackwardFn raydn_scene_face_normals_backward_fn(std::uintptr_t module_handle);
SceneFaceNormalsJvpFn raydn_scene_face_normals_jvp_fn(std::uintptr_t module_handle);
DiffractionDiscoverEdgesFn raydn_diffraction_discover_edges_fn(std::uintptr_t module_handle);
DiffractionDiscoverEdgesCountedFn raydn_diffraction_discover_edges_counted_fn(std::uintptr_t module_handle);
DiffractionPathsOrder1ForwardFn raydn_diffraction_paths_order1_forward_fn(std::uintptr_t module_handle);
DiffractionAccumulationForwardFn raydn_diffraction_accumulation_forward_fn(std::uintptr_t module_handle);

const at::Tensor *optional_tensor(pybind11::object value, at::Tensor &storage);
pybind11::object tensor_or_none(const at::Tensor &tensor);

}  // namespace channel_native::rayd_bridge
