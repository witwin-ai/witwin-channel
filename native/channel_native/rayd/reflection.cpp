#include "resource.h"

#include <cstdint>
#include <stdexcept>

pybind11::tuple cn_rayd_trace_reflections_forward(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    rayd::torch::ReflectionTraceResult out = rayd::torch::trace_reflections_forward(
        scene.resource(),
        rayd::torch::ReflectionTraceRequest{
            {ray_o, ray_d, ray_tmax, optional_tensor(active)}, max_bounces});
    return pybind11::make_tuple(out.valid, out.t, out.prim_ids);
}

pybind11::tuple cn_rayd_reflection_epc_paths_forward(
    RayDSceneResource &scene,
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
    int64_t visibility_ignore_mode) {
    rayd::torch::ReflectionEpcRequest request;
    request.source = source;
    request.receiver = receiver;
    request.active = optional_tensor(active);
    request.expected_prim_ids = expected_prim_ids;
    request.direct_plane_points = direct_plane_points;
    request.direct_plane_normals = direct_plane_normals;
    request.surface_group_id = surface_group_id;
    request.surface_group_size = surface_group_size;
    request.surface_group_members = surface_group_members;
    request.max_bounces = max_bounces;
    request.visibility_ignore_mode = visibility_ignore_mode;
    request.plane_tolerance = 1.0e-3;
    rayd::torch::ReflectionEpcResult out =
        rayd::torch::reflection_epc_paths_forward(scene.resource(), request);
    return pybind11::make_tuple(
        out.valid,
        out.path_length,
        out.resolved_prim_ids,
        out.surface_group_ids,
        out.hit_positions,
        out.normals);
}

pybind11::tuple cn_rayd_trace_reflections_forward_tape(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    rayd::torch::ReflectionTraceTapeResult out =
        rayd::torch::trace_reflections_forward_tape(
            scene.resource(),
            rayd::torch::ReflectionTraceRequest{
                {ray_o, ray_d, ray_tmax, optional_tensor(active)}, max_bounces});
    return pybind11::make_tuple(
        out.valid,
        out.t,
        out.image_sources,
        out.prim_ids,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_hit_points,
        out.tape_normals,
        out.active_ctx);
}

pybind11::tuple cn_rayd_trace_reflections_backward(
    RayDSceneResource &scene,
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
    pybind11::object grad_image_sources) {
    rayd::torch::ReflectionTraceBackwardRequest request;
    request.rays = {ray_o, ray_d, ray_tmax, optional_tensor(active)};
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.tape_hit_points = tape_hit_points;
    request.tape_normals = tape_normals;
    request.image_sources = image_sources;
    request.grad_t = optional_tensor(grad_t);
    request.grad_image_sources = optional_tensor(grad_image_sources);
    rayd::torch::ReflectionTraceBackwardResult out =
        rayd::torch::trace_reflections_backward(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_ray_o),
        tensor_or_none(out.grad_ray_d),
        tensor_or_none(out.grad_ray_tmax));
}

pybind11::tuple cn_rayd_trace_reflections_jvp(
    RayDSceneResource &scene,
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
    torch::Tensor image_sources) {
    rayd::torch::ReflectionTraceJvpRequest request;
    request.ray_o = ray_o;
    request.ray_d = ray_d;
    request.active = optional_tensor(active);
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.tape_hit_points = tape_hit_points;
    request.tape_normals = tape_normals;
    request.tangent_vertices = optional_tensor(tangent_vertices);
    request.tangent_ray_o = optional_tensor(tangent_ray_o);
    request.tangent_ray_d = optional_tensor(tangent_ray_d);
    request.image_sources = image_sources;
    rayd::torch::ReflectionTraceJvpResult out =
        rayd::torch::trace_reflections_jvp(scene.resource(), request);
    return pybind11::make_tuple(out.tangent_t, out.tangent_image_sources);
}

pybind11::tuple cn_rayd_reflection_epc_paths_backward(
    RayDSceneResource &scene,
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
    bool need_grad_receiver) {
    rayd::torch::ReflectionEpcBackwardRequest request;
    request.source = source;
    request.receiver = receiver;
    request.sequence = sequence;
    request.plane_points = plane_points;
    request.plane_normals = plane_normals;
    request.valid = valid;
    request.bounce_count = bounce_count;
    request.grad_points = optional_tensor(grad_points);
    request.grad_normals = optional_tensor(grad_normals);
    request.grad_path_length = optional_tensor(grad_path_length);
    request.need_grad_vertices = need_grad_vertices;
    request.need_grad_source = need_grad_source;
    request.need_grad_receiver = need_grad_receiver;
    rayd::torch::ReflectionEpcBackwardResult out =
        rayd::torch::reflection_epc_paths_backward(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_source),
        tensor_or_none(out.grad_receiver));
}

pybind11::tuple cn_rayd_reflection_epc_paths_jvp(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object tangent_vertices,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver) {
    rayd::torch::ReflectionEpcJvpRequest request;
    request.source = source;
    request.receiver = receiver;
    request.sequence = sequence;
    request.plane_points = plane_points;
    request.plane_normals = plane_normals;
    request.valid = valid;
    request.bounce_count = bounce_count;
    request.tangent_vertices = optional_tensor(tangent_vertices);
    request.tangent_source = optional_tensor(tangent_source);
    request.tangent_receiver = optional_tensor(tangent_receiver);
    rayd::torch::ReflectionEpcJvpResult out =
        rayd::torch::reflection_epc_paths_jvp(scene.resource(), request);
    return pybind11::make_tuple(
        out.tangent_points, out.tangent_normals, out.tangent_path_length);
}

at::Tensor cn_rayd_scene_face_normals_backward(
    RayDSceneResource &scene,
    torch::Tensor grad_face_normals) {
    return rayd::torch::scene_face_normals_backward(
        scene.resource(), grad_face_normals);
}

at::Tensor cn_rayd_scene_face_normals_jvp(
    RayDSceneResource &scene,
    torch::Tensor tangent_vertices) {
    return rayd::torch::scene_face_normals_jvp(
        scene.resource(), tangent_vertices);
}
