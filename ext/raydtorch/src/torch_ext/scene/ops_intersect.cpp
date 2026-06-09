#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <torch/extension.h>

namespace raydtorch {

py::tuple intersect_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out = intersect_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_t);
}

py::tuple intersection_public_tuple(const IntersectForwardOutputs &out) {
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id);
}

py::tuple intersect_forward_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out =
        intersect_forward_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    return intersection_public_tuple(out);
}

py::tuple intersect_forward_ad_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out =
        intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_t);
}

py::tuple intersect_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t,
    at::Tensor grad_p,
    at::Tensor grad_n,
    at::Tensor grad_geo_n,
        at::Tensor grad_uv,
        at::Tensor grad_barycentric) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.contiguous(),
        grad_p.contiguous(),
        grad_n.contiguous(),
        grad_geo_n.contiguous(),
        grad_uv.contiguous(),
        grad_barycentric.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
        at::Tensor tangent_vertices,
        at::Tensor tangent_ray_o,
        at::Tensor tangent_ray_d) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectJvpOutputs out = intersect_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices.contiguous(),
        tangent_ray_o.contiguous(),
        tangent_ray_d.contiguous());
    return py::make_tuple(
        out.tangent_t,
        out.tangent_p,
        out.tangent_n,
        out.tangent_geo_n,
        out.tangent_uv,
        out.tangent_barycentric);
}

void bind_intersect_ops(py::module_ &m) {
    m.def("intersect_forward", &intersect_forward_op);
    m.def("intersect_forward_flags", &intersect_forward_flags_op);
    m.def("intersect_forward_ad_flags", &intersect_forward_ad_flags_op);
    m.def("intersect_backward", &intersect_backward_op);
    m.def("intersect_jvp", &intersect_jvp_op);
}

} // namespace raydtorch
