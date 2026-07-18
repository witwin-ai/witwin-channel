#include "bridge.h"
#include "../tensor_checks.h"

#include <array>
#include <cstdint>
#include <stdexcept>
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
using channel_native::rayd_bridge::optional_tensor;
using channel_native::rayd_bridge::raydn_intersect_backward_fn;
using channel_native::rayd_bridge::raydn_intersect_forward_fn;
using channel_native::rayd_bridge::raydn_intersect_jvp_fn;
using channel_native::rayd_bridge::raydn_reflection_epc_paths_forward_fn;
using channel_native::rayd_bridge::raydn_visibility_forward_fn;
using channel_native::rayd_bridge::tensor_or_none;

}  // namespace

pybind11::tuple cn_bdpt_intersect_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 10;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_intersect_forward_fn()(
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
    pybind11::object active) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
    raydn_visibility_forward_fn()(
        scene_handle,
        &start,
        &end,
        active_ptr,
        &visible,
        &blocker_prim,
        &tape_t);
    return pybind11::make_tuple(visible, blocker_prim, tape_t);
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
    bool need_grad_ray_tmax) {
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
    int64_t output_count = raydn_intersect_backward_fn()(
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
    int64_t flags) {
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
    int64_t output_count = raydn_intersect_jvp_fn()(
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

    constexpr int64_t kEpcOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kEpcOutputCount)> epc;
    const int64_t epc_output_count = raydn_reflection_epc_paths_forward_fn()(
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
    raydn_visibility_forward_fn()(
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

pybind11::dict cn_raydn_coupled_dd_geometry_forward(
    int64_t scene_handle,
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

    // Three RayDN segment visibility queries (tx->Q1, Q1->Q2, Q2->rx), each
    // gated by the candidate mask and ANDed into the row validity.
    at::Tensor visible_leg1;
    at::Tensor blocker_leg1;
    at::Tensor tape_leg1;
    raydn_visibility_forward_fn()(
        scene_handle, &source, &q1, &candidate_active, &visible_leg1, &blocker_leg1, &tape_leg1);
    at::Tensor visible_leg2;
    at::Tensor blocker_leg2;
    at::Tensor tape_leg2;
    raydn_visibility_forward_fn()(
        scene_handle, &q1, &q2, &candidate_active, &visible_leg2, &blocker_leg2, &tape_leg2);
    at::Tensor visible_leg3;
    at::Tensor blocker_leg3;
    at::Tensor tape_leg3;
    raydn_visibility_forward_fn()(
        scene_handle, &q2, &receiver, &candidate_active, &visible_leg3, &blocker_leg3, &tape_leg3);

    at::Tensor prefix_active = cn_coupled_active_mask_cuda(candidate_active, visible_leg1);
    prefix_active = cn_coupled_active_mask_cuda(prefix_active, visible_leg2);
    prefix_active = cn_coupled_active_mask_cuda(prefix_active, visible_leg3);

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
