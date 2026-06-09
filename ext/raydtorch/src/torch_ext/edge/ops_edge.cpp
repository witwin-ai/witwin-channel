#include <raydtorch/edge/kernels.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <torch/extension.h>

namespace raydtorch {

py::tuple nearest_edge_forward_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    SceneCache &scene = get_scene(scene_handle);
    EdgeForwardOutputs out = edge_forward_cuda(scene, point);
    return py::make_tuple(
        out.distance,
        out.edge_point,
        out.edge_t,
        out.shape_id,
        out.edge_id,
        out.global_edge_id,
        out.tape_edge_id,
        out.tape_s,
        out.tape_d);
}

py::tuple nearest_edge_forward_noad_op(int64_t scene_handle, at::Tensor point) {
    require_vec3f(point, "point");
    SceneCache &scene = get_scene(scene_handle);
    EdgeForwardPublicOutputs out = edge_forward_noad_cuda(scene, point);
    return py::make_tuple(
        out.distance,
        out.edge_point,
        out.edge_t,
        out.shape_id,
        out.edge_id,
        out.global_edge_id);
}

py::tuple nearest_edge_backward_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    at::Tensor grad_distance,
    at::Tensor grad_edge_point,
        at::Tensor grad_edge_t) {
    SceneCache &scene = get_scene(scene_handle);
    EdgeBackwardOutputs out = edge_backward_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        grad_distance.contiguous(),
        grad_edge_point.contiguous(),
        grad_edge_t.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_point);
}

py::tuple nearest_edge_ray_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    if (ray_d.size(0) != ray_o.size(0) || ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_d, ray_tmax, and active must match ray_o batch size.");
    SceneCache &scene = get_scene(scene_handle);
    EdgeRayForwardOutputs out = edge_ray_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
    return py::make_tuple(
        out.distance,
        out.ray_t,
        out.point,
        out.edge_t,
        out.edge_point,
        out.shape_id,
        out.edge_id,
        out.global_edge_id,
        out.tape_edge_id);
}

py::tuple nearest_edge_jvp_op(
    int64_t scene_handle,
    at::Tensor point,
    at::Tensor tape_edge_id,
    at::Tensor tape_s,
    at::Tensor tape_d,
    at::Tensor tangent_vertices,
        at::Tensor tangent_point) {
    SceneCache &scene = get_scene(scene_handle);
    EdgeJvpOutputs out = edge_jvp_cuda(
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        tangent_vertices.contiguous(),
        tangent_point.contiguous());
    return py::make_tuple(out.tangent_distance, out.tangent_edge_point, out.tangent_edge_t);
}

void bind_edge_ops(py::module_ &m) {
    m.def("nearest_edge_forward", &nearest_edge_forward_op);
    m.def("nearest_edge_forward_noad", &nearest_edge_forward_noad_op);
    m.def("nearest_edge_ray_forward", &nearest_edge_ray_forward_op);
    m.def("nearest_edge_backward", &nearest_edge_backward_op);
    m.def("nearest_edge_jvp", &nearest_edge_jvp_op);
}

} // namespace raydtorch
