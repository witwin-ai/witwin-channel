#include "resource.h"
#include "../tensor_checks.h"

#include <cstdint>
#include <vector>

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
std::vector<at::Tensor> cn_coupled_dd_prepare_cuda(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor edge1_pos,
    at::Tensor edge1_dir,
    at::Tensor edge1_t_min,
    at::Tensor edge1_t_max,
    at::Tensor edge2_pos,
    at::Tensor edge2_dir,
    at::Tensor edge2_t_min,
    at::Tensor edge2_t_max);
pybind11::dict cn_coupled_dd_finalize_cuda(
    at::Tensor prefix_active,
    at::Tensor edge1_id,
    at::Tensor edge2_id,
    at::Tensor edge1_point,
    at::Tensor edge2_point,
    at::Tensor source,
    at::Tensor receiver);

namespace {

using channel_native::check_flat_tensor;
using channel_native::check_vec3_table;

}  // namespace

pybind11::tuple cn_rayd_intersect_forward(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags) {
    rayd::torch::IntersectResult out = rayd::torch::intersect_forward(
        scene.resource(),
        rayd::torch::RayBatch{ray_o, ray_d, ray_tmax, optional_tensor(active)},
        flags);
    return pybind11::make_tuple(
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

pybind11::tuple cn_rayd_visibility_forward(
    RayDSceneResource &scene,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active) {
    rayd::torch::VisibilityResult out = rayd::torch::visibility_forward(
        scene.resource(),
        rayd::torch::VisibilityRequest{start, end, optional_tensor(active)});
    return pybind11::make_tuple(out.visible, out.blocker_prim, out.tape_t);
}


pybind11::tuple cn_rayd_intersect_backward(
    RayDSceneResource &scene,
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
    bool need_grad_ray_tmax) {
    rayd::torch::IntersectBackwardRequest request;
    request.rays = {ray_o, ray_d, ray_tmax, optional_tensor(active)};
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.grad_t = optional_tensor(grad_t);
    request.grad_p = optional_tensor(grad_p);
    request.grad_n = optional_tensor(grad_n);
    request.grad_geo_n = optional_tensor(grad_geo_n);
    request.grad_uv = optional_tensor(grad_uv);
    request.grad_barycentric = optional_tensor(grad_barycentric);
    request.need_grad_vertices = need_grad_vertices;
    request.need_grad_ray_o = need_grad_ray_o;
    request.need_grad_ray_d = need_grad_ray_d;
    request.need_grad_ray_tmax = need_grad_ray_tmax;
    rayd::torch::IntersectBackwardResult out =
        rayd::torch::intersect_backward(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_ray_o),
        tensor_or_none(out.grad_ray_d),
        tensor_or_none(out.grad_ray_tmax));
}

pybind11::tuple cn_rayd_intersect_jvp(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    int64_t flags) {
    rayd::torch::IntersectJvpRequest request;
    request.ray_o = ray_o;
    request.ray_d = ray_d;
    request.active = optional_tensor(active);
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.tangent_vertices = optional_tensor(tangent_vertices);
    request.tangent_ray_o = optional_tensor(tangent_ray_o);
    request.tangent_ray_d = optional_tensor(tangent_ray_d);
    request.flags = flags;
    rayd::torch::IntersectJvpResult out =
        rayd::torch::intersect_jvp(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.tangent_t),
        tensor_or_none(out.tangent_p),
        tensor_or_none(out.tangent_n),
        tensor_or_none(out.tangent_geo_n),
        tensor_or_none(out.tangent_uv),
        tensor_or_none(out.tangent_barycentric));
}

pybind11::dict cn_coupled_rd_geometry_forward(
    RayDSceneResource &scene,
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
    bool reverse) {
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

    rayd::torch::ReflectionEpcRequest epc_request;
    epc_request.source = epc_source;
    epc_request.receiver = diffraction_point;
    epc_request.active = candidate_active;
    epc_request.expected_prim_ids = expected_faces;
    epc_request.direct_plane_points = direct_plane_points;
    epc_request.direct_plane_normals = direct_plane_normals;
    epc_request.surface_group_id = surface_group_id;
    epc_request.surface_group_size = surface_group_size;
    epc_request.surface_group_members = surface_group_members;
    epc_request.max_bounces = 1;
    epc_request.visibility_ignore_mode = 1;
    epc_request.plane_tolerance = 1.0e-3;
    rayd::torch::ReflectionEpcResult epc =
        rayd::torch::reflection_epc_paths_forward(scene.resource(), epc_request);

    at::Tensor prefix_active = cn_coupled_active_mask_cuda(candidate_active, epc.valid);
    rayd::torch::VisibilityResult suffix = rayd::torch::visibility_forward(
        scene.resource(),
        rayd::torch::VisibilityRequest{
            diffraction_point, epc_receiver, prefix_active});
    at::Tensor resolved_face = epc.resolved_prim_ids.select(1, 0).contiguous();
    at::Tensor reflection_position = epc.hit_positions.select(1, 0).contiguous();
    at::Tensor reflection_normal = epc.normals.select(1, 0).contiguous();
    pybind11::dict out = cn_coupled_rd_finalize_cuda(
        prefix_active,
        suffix.visible,
        epc.path_length.contiguous(),
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
    out["suffix_blocker_primitive"] = suffix.blocker_prim;
    return out;
}

pybind11::dict cn_coupled_dd_geometry_forward(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor edge1_id,
    torch::Tensor edge1_pos,
    torch::Tensor edge1_dir,
    torch::Tensor edge1_t_min,
    torch::Tensor edge1_t_max,
    torch::Tensor edge2_id,
    torch::Tensor edge2_pos,
    torch::Tensor edge2_dir,
    torch::Tensor edge2_t_min,
    torch::Tensor edge2_t_max) {
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    check_flat_tensor(edge1_id, "edge1_id", at::kInt);
    check_vec3_table(edge1_pos, "edge1_pos");
    check_vec3_table(edge1_dir, "edge1_dir");
    check_flat_tensor(edge1_t_min, "edge1_t_min", at::kFloat);
    check_flat_tensor(edge1_t_max, "edge1_t_max", at::kFloat);
    check_flat_tensor(edge2_id, "edge2_id", at::kInt);
    check_vec3_table(edge2_pos, "edge2_pos");
    check_vec3_table(edge2_dir, "edge2_dir");
    check_flat_tensor(edge2_t_min, "edge2_t_min", at::kFloat);
    check_flat_tensor(edge2_t_max, "edge2_t_max", at::kFloat);
    const int64_t count = source.size(0);
    TORCH_CHECK(receiver.size(0) == count, "receiver must match source rows");
    TORCH_CHECK(edge1_id.size(0) == count && edge2_id.size(0) == count,
                "edge ids must match source rows");
    for (const auto &tensor : {edge1_pos, edge1_dir, edge2_pos, edge2_dir})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction vector tables must match source rows");
    for (const auto &tensor : {edge1_t_min, edge1_t_max, edge2_t_min, edge2_t_max})
        TORCH_CHECK(tensor.size(0) == count, "edge bounds must match source rows");

    // Prepare the two-edge Fermat point pair (Q1 on e1, Q2 on e2).
    std::vector<at::Tensor> prepared = cn_coupled_dd_prepare_cuda(
        source,
        receiver,
        edge1_pos,
        edge1_dir,
        edge1_t_min,
        edge1_t_max,
        edge2_pos,
        edge2_dir,
        edge2_t_min,
        edge2_t_max);
    TORCH_CHECK(prepared.size() == 3,
                "coupled D-D prepare returned an unexpected tensor count");
    at::Tensor candidate_active = prepared[0];
    at::Tensor q1 = prepared[1];
    at::Tensor q2 = prepared[2];

    // Three RayD segment visibility queries (tx->Q1, Q1->Q2, Q2->rx), each
    // gated by the candidate mask and ANDed into the row validity.
    rayd::torch::VisibilityResult leg1 = rayd::torch::visibility_forward(
        scene.resource(), {source, q1, candidate_active});
    rayd::torch::VisibilityResult leg2 = rayd::torch::visibility_forward(
        scene.resource(), {q1, q2, candidate_active});
    rayd::torch::VisibilityResult leg3 = rayd::torch::visibility_forward(
        scene.resource(), {q2, receiver, candidate_active});

    at::Tensor prefix_active = cn_coupled_active_mask_cuda(candidate_active, leg1.visible);
    prefix_active = cn_coupled_active_mask_cuda(prefix_active, leg2.visible);
    prefix_active = cn_coupled_active_mask_cuda(prefix_active, leg3.visible);

    pybind11::dict out = cn_coupled_dd_finalize_cuda(
        prefix_active,
        edge1_id,
        edge2_id,
        q1,
        q2,
        source,
        receiver);
    out["candidate_active"] = candidate_active;
    return out;
}
