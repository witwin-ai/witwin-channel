// Copyright Xingyu Chen.
// Implements geometry native integration.

#include "resource.h"
#include "../tensor_checks.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

pybind11::dict channel_coupled_chain_finalize_cuda(
    at::Tensor prefix_valid,
    at::Tensor suffix_valid,
    at::Tensor prefix_length,
    at::Tensor suffix_length,
    at::Tensor prefix_prims,
    at::Tensor suffix_prims,
    at::Tensor prefix_hits,
    at::Tensor suffix_hits,
    at::Tensor prefix_normals,
    at::Tensor suffix_normals,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor edge_point,
    at::Tensor edge_id,
    double min_segment_length);
at::Tensor channel_coupled_active_mask_cuda(at::Tensor lhs, at::Tensor rhs);
at::Tensor channel_coupled_active_either_mask_cuda(at::Tensor lhs, at::Tensor rhs, at::Tensor other);
std::vector<at::Tensor> channel_coupled_dd_prepare_cuda(
    at::Tensor edge1_id,
    at::Tensor edge2_id,
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
pybind11::dict channel_coupled_dd_finalize_cuda(
    at::Tensor prefix_active,
    at::Tensor edge1_id,
    at::Tensor edge2_id,
    at::Tensor edge1_point,
    at::Tensor edge2_point,
    at::Tensor source,
    at::Tensor receiver);

namespace {

using channel::check_flat_tensor;
using channel::check_vec3_table;

rayd::torch::SegmentPenetrationPolicy segment_penetration_policy(
    int64_t policy) {
    switch (policy) {
    case 0:
        return rayd::torch::SegmentPenetrationPolicy::EnumeratedFullDistance;
    case 1:
        return rayd::torch::SegmentPenetrationPolicy::MonteCarloTargetInset;
    default:
        TORCH_CHECK(false, "segment penetration policy must be 0 or 1");
    }
}

std::int32_t segment_penetration_failure_bit(int64_t failure_bit) {
    TORCH_CHECK(
        failure_bit > 0 &&
            failure_bit <= std::numeric_limits<std::int32_t>::max(),
        "segment penetration failure_bit must fit in positive int32");
    const auto bit = static_cast<std::uint32_t>(failure_bit);
    TORCH_CHECK(
        (bit & (bit - 1u)) == 0u,
        "segment penetration failure_bit must contain exactly one bit");
    return static_cast<std::int32_t>(bit);
}

rayd::torch::SegmentPenetrationRequest segment_penetration_request(
    RayDSceneResource &scene,
    torch::Tensor origins,
    torch::Tensor targets,
    pybind11::object input_active,
    bool input_active_any,
    int64_t hit_capacity,
    int64_t policy,
    double scene_diagonal,
    torch::Tensor capacity_failure_record,
    int64_t failure_bit) {
    TORCH_CHECK(
        capacity_failure_record.is_cuda() &&
            capacity_failure_record.scalar_type() == torch::kInt &&
            capacity_failure_record.dim() == 1 &&
            capacity_failure_record.size(0) == 2 &&
            capacity_failure_record.is_contiguous(),
        "capacity_failure_record must be a contiguous CUDA int32 tensor of shape (2,)");
    // A segment with more crossings than hit_capacity is beyond the requested
    // depth: RayD drops that row and counts it in record[1]; the batch stays valid.
    return {
        scene.resource(),
        std::move(origins),
        std::move(targets),
        optional_tensor(input_active),
        input_active_any,
        hit_capacity,
        segment_penetration_policy(policy),
        scene_diagonal,
        capacity_failure_record.narrow(0, 0, 1),
        segment_penetration_failure_bit(failure_bit),
        rayd::torch::SegmentPenetrationOverflowPolicy::DropRow,
        capacity_failure_record.narrow(0, 1, 1),
    };
}

rayd::torch::SegmentPenetrationTapeResult segment_penetration_tape(
    torch::Tensor valid,
    torch::Tensor num_hits,
    torch::Tensor reached_target,
    torch::Tensor overflow,
    torch::Tensor distance,
    torch::Tensor direction,
    torch::Tensor hit_t,
    torch::Tensor position,
    torch::Tensor normal,
    torch::Tensor geometric_normal,
    torch::Tensor global_primitive_id,
    torch::Tensor tape_primitive_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_restart_epsilon,
    torch::Tensor tape_restart_branch,
    torch::Tensor tape_restart_tie_mask,
    torch::Tensor tape_direction_denominator_branch) {
    return {
        {
            std::move(valid),
            std::move(num_hits),
            std::move(reached_target),
            std::move(overflow),
            std::move(distance),
            std::move(direction),
            std::move(hit_t),
            std::move(position),
            std::move(normal),
            std::move(geometric_normal),
            std::move(global_primitive_id),
        },
        std::move(tape_primitive_id),
        std::move(tape_barycentric),
        std::move(tape_restart_epsilon),
        std::move(tape_restart_branch),
        std::move(tape_restart_tie_mask),
        std::move(tape_direction_denominator_branch),
    };
}

pybind11::tuple pack_segment_penetration_result(
    const rayd::torch::SegmentPenetrationResult &out) {
    return pybind11::make_tuple(
        out.valid,
        out.num_hits,
        out.reached_target,
        out.overflow,
        out.distance,
        out.direction,
        out.t,
        out.position,
        out.normal,
        out.geometric_normal,
        out.global_primitive_id);
}

pybind11::tuple pack_segment_penetration_tape(
    const rayd::torch::SegmentPenetrationTapeResult &out) {
    return pybind11::make_tuple(
        out.result.valid,
        out.result.num_hits,
        out.result.reached_target,
        out.result.overflow,
        out.result.distance,
        out.result.direction,
        out.result.t,
        out.result.position,
        out.result.normal,
        out.result.geometric_normal,
        out.result.global_primitive_id,
        out.tape_primitive_id,
        out.tape_barycentric,
        out.tape_restart_epsilon,
        out.tape_restart_branch,
        out.tape_restart_tie_mask,
        out.tape_direction_denominator_branch);
}

}  // namespace

pybind11::tuple channel_rayd_segment_penetration_forward(
    RayDSceneResource &scene,
    torch::Tensor origins,
    torch::Tensor targets,
    pybind11::object input_active,
    bool input_active_any,
    int64_t hit_capacity,
    int64_t policy,
    double scene_diagonal,
    torch::Tensor capacity_failure_record,
    int64_t failure_bit) {
    const auto request = segment_penetration_request(
        scene,
        std::move(origins),
        std::move(targets),
        std::move(input_active),
        input_active_any,
        hit_capacity,
        policy,
        scene_diagonal,
        std::move(capacity_failure_record),
        failure_bit);
    return pack_segment_penetration_result(
        rayd::torch::segment_penetration_forward(request));
}

pybind11::tuple channel_rayd_segment_penetration_forward_tape(
    RayDSceneResource &scene,
    torch::Tensor origins,
    torch::Tensor targets,
    pybind11::object input_active,
    bool input_active_any,
    int64_t hit_capacity,
    int64_t policy,
    double scene_diagonal,
    torch::Tensor capacity_failure_record,
    int64_t failure_bit) {
    const auto request = segment_penetration_request(
        scene,
        std::move(origins),
        std::move(targets),
        std::move(input_active),
        input_active_any,
        hit_capacity,
        policy,
        scene_diagonal,
        std::move(capacity_failure_record),
        failure_bit);
    return pack_segment_penetration_tape(
        rayd::torch::segment_penetration_forward_tape(request));
}

pybind11::tuple channel_rayd_segment_penetration_backward(
    RayDSceneResource &scene,
    torch::Tensor origins,
    torch::Tensor targets,
    pybind11::object input_active,
    bool input_active_any,
    int64_t hit_capacity,
    int64_t policy,
    double scene_diagonal,
    torch::Tensor capacity_failure_record,
    int64_t failure_bit,
    torch::Tensor valid,
    torch::Tensor num_hits,
    torch::Tensor reached_target,
    torch::Tensor overflow,
    torch::Tensor distance,
    torch::Tensor direction,
    torch::Tensor hit_t,
    torch::Tensor position,
    torch::Tensor normal,
    torch::Tensor geometric_normal,
    torch::Tensor global_primitive_id,
    torch::Tensor tape_primitive_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_restart_epsilon,
    torch::Tensor tape_restart_branch,
    torch::Tensor tape_restart_tie_mask,
    torch::Tensor tape_direction_denominator_branch,
    pybind11::object grad_distance,
    pybind11::object grad_direction,
    pybind11::object grad_t,
    pybind11::object grad_position,
    pybind11::object grad_normal,
    pybind11::object grad_geometric_normal,
    bool need_grad_vertices,
    bool need_grad_origins,
    bool need_grad_targets) {
    const auto primal = segment_penetration_request(
        scene,
        std::move(origins),
        std::move(targets),
        std::move(input_active),
        input_active_any,
        hit_capacity,
        policy,
        scene_diagonal,
        std::move(capacity_failure_record),
        failure_bit);
    const auto tape = segment_penetration_tape(
        std::move(valid),
        std::move(num_hits),
        std::move(reached_target),
        std::move(overflow),
        std::move(distance),
        std::move(direction),
        std::move(hit_t),
        std::move(position),
        std::move(normal),
        std::move(geometric_normal),
        std::move(global_primitive_id),
        std::move(tape_primitive_id),
        std::move(tape_barycentric),
        std::move(tape_restart_epsilon),
        std::move(tape_restart_branch),
        std::move(tape_restart_tie_mask),
        std::move(tape_direction_denominator_branch));
    const rayd::torch::SegmentPenetrationBackwardRequest request{
        primal,
        tape,
        optional_tensor(grad_distance),
        optional_tensor(grad_direction),
        optional_tensor(grad_t),
        optional_tensor(grad_position),
        optional_tensor(grad_normal),
        optional_tensor(grad_geometric_normal),
        need_grad_vertices,
        need_grad_origins,
        need_grad_targets,
    };
    const auto out = rayd::torch::segment_penetration_backward(request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_origins),
        tensor_or_none(out.grad_targets));
}

pybind11::tuple channel_rayd_segment_penetration_jvp(
    RayDSceneResource &scene,
    torch::Tensor origins,
    torch::Tensor targets,
    pybind11::object input_active,
    bool input_active_any,
    int64_t hit_capacity,
    int64_t policy,
    double scene_diagonal,
    torch::Tensor capacity_failure_record,
    int64_t failure_bit,
    torch::Tensor valid,
    torch::Tensor num_hits,
    torch::Tensor reached_target,
    torch::Tensor overflow,
    torch::Tensor distance,
    torch::Tensor direction,
    torch::Tensor hit_t,
    torch::Tensor position,
    torch::Tensor normal,
    torch::Tensor geometric_normal,
    torch::Tensor global_primitive_id,
    torch::Tensor tape_primitive_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_restart_epsilon,
    torch::Tensor tape_restart_branch,
    torch::Tensor tape_restart_tie_mask,
    torch::Tensor tape_direction_denominator_branch,
    pybind11::object tangent_vertices,
    pybind11::object tangent_origins,
    pybind11::object tangent_targets) {
    const auto primal = segment_penetration_request(
        scene,
        std::move(origins),
        std::move(targets),
        std::move(input_active),
        input_active_any,
        hit_capacity,
        policy,
        scene_diagonal,
        std::move(capacity_failure_record),
        failure_bit);
    const auto tape = segment_penetration_tape(
        std::move(valid),
        std::move(num_hits),
        std::move(reached_target),
        std::move(overflow),
        std::move(distance),
        std::move(direction),
        std::move(hit_t),
        std::move(position),
        std::move(normal),
        std::move(geometric_normal),
        std::move(global_primitive_id),
        std::move(tape_primitive_id),
        std::move(tape_barycentric),
        std::move(tape_restart_epsilon),
        std::move(tape_restart_branch),
        std::move(tape_restart_tie_mask),
        std::move(tape_direction_denominator_branch));
    const rayd::torch::SegmentPenetrationJvpRequest request{
        primal,
        tape,
        optional_tensor(tangent_vertices),
        optional_tensor(tangent_origins),
        optional_tensor(tangent_targets),
    };
    const auto out = rayd::torch::segment_penetration_jvp(request);
    return pybind11::make_tuple(
        out.tangent_distance,
        out.tangent_direction,
        out.tangent_t,
        out.tangent_position,
        out.tangent_normal,
        out.tangent_geometric_normal);
}

pybind11::tuple channel_rayd_visibility_forward(
    RayDSceneResource &scene,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active) {
    rayd::torch::VisibilityResult out = rayd::torch::visibility_forward(
        scene.resource(),
        rayd::torch::VisibilityRequest{start, end, optional_tensor(active)});
    return pybind11::make_tuple(out.visible, out.blocker_prim, out.tape_t);
}

namespace {

// One half of an interaction chain: the endpoint to the diffraction point through `faces`
// (EPC), or a plain segment when the half has no reflection. Only active rows are traced.
struct ChainLeg {
    at::Tensor valid;
    at::Tensor path_length;
    at::Tensor prims;
    at::Tensor hits;
    at::Tensor normals;
};

ChainLeg trace_chain_leg(RayDSceneResource &scene, const at::Tensor &endpoint, const at::Tensor &point,
                         const at::Tensor &active, const at::Tensor &faces, const at::Tensor &plane_points,
                         const at::Tensor &plane_normals, const at::Tensor &surface_group_id,
                         const at::Tensor &surface_group_size, const at::Tensor &surface_group_members,
                         double min_segment_length) {
    const int64_t count = endpoint.size(0);
    const int64_t depth = faces.size(1);
    ChainLeg leg;
    if (depth == 0) {
        // The segment is traced from the diffraction point to the endpoint, so a chain and its
        // reverse trace the same segment the same way.
        rayd::torch::VisibilityResult visibility =
            rayd::torch::visibility_forward(scene.resource(), rayd::torch::VisibilityRequest{point, endpoint, active});
        leg.valid = visibility.visible;
        leg.path_length = at::empty({0}, point.options());
        leg.prims = at::empty({count, 0}, faces.options());
        leg.hits = at::empty({count, 0, 3}, point.options());
        leg.normals = at::empty({count, 0, 3}, point.options());
        return leg;
    }
    rayd::torch::ReflectionEpcRequest request;
    request.source = endpoint;
    request.receiver = point;
    request.active = active;
    request.expected_prim_ids = faces;
    request.direct_plane_points = plane_points;
    request.direct_plane_normals = plane_normals;
    request.surface_group_id = surface_group_id;
    request.surface_group_size = surface_group_size;
    request.surface_group_members = surface_group_members;
    request.max_bounces = depth;
    request.visibility_ignore_mode = 1;
    request.plane_tolerance = 1.0e-3;
    request.min_segment_length = min_segment_length;
    // The leg ends at the diffraction point, an interaction, so the minimum covers its last segment too.
    request.receiver_is_interaction = true;
    rayd::torch::ReflectionEpcResult epc = rayd::torch::reflection_epc_paths_forward(scene.resource(), request);
    leg.valid = epc.valid;
    leg.path_length = epc.path_length.contiguous();
    leg.prims = epc.resolved_prim_ids.contiguous();
    leg.hits = epc.hit_positions.contiguous();
    leg.normals = epc.normals.contiguous();
    return leg;
}

void check_chain_half(const at::Tensor &faces, const at::Tensor &plane_points, const at::Tensor &plane_normals,
                      int64_t count, const char *name) {
    TORCH_CHECK(faces.is_cuda() && faces.scalar_type() == at::kInt && faces.dim() == 2 && faces.is_contiguous(),
                name, " faces must be a contiguous CUDA int32 (N, D) tensor");
    TORCH_CHECK(faces.size(0) == count, name, " faces must match source rows");
    for (const at::Tensor *tensor : {&plane_points, &plane_normals}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->dim() == 3 &&
                        tensor->is_contiguous() && tensor->size(0) == count && tensor->size(1) == faces.size(1) &&
                        tensor->size(2) == 3,
                    name, " planes must be contiguous CUDA float32 (N, D, 3) tensors matching faces");
    }
}

}  // namespace

pybind11::dict channel_coupled_chain_geometry_forward(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor edge_point,
    torch::Tensor edge_id,
    torch::Tensor prefix_faces,
    torch::Tensor prefix_plane_points,
    torch::Tensor prefix_plane_normals,
    torch::Tensor suffix_faces,
    torch::Tensor suffix_plane_points,
    torch::Tensor suffix_plane_normals,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    double min_segment_length) {
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    check_vec3_table(edge_point, "edge_point");
    check_flat_tensor(edge_id, "edge_id", at::kInt);
    check_flat_tensor(surface_group_id, "surface_group_id", at::kInt);
    check_flat_tensor(surface_group_size, "surface_group_size", at::kInt);
    check_flat_tensor(surface_group_members, "surface_group_members", at::kInt);
    const int64_t count = source.size(0);
    TORCH_CHECK(receiver.size(0) == count && edge_point.size(0) == count && edge_id.size(0) == count,
                "chain candidate tables must match source rows");
    check_chain_half(prefix_faces, prefix_plane_points, prefix_plane_normals, count, "prefix");
    check_chain_half(suffix_faces, suffix_plane_points, suffix_plane_normals, count, "suffix");
    TORCH_CHECK(prefix_faces.size(1) + suffix_faces.size(1) >= 1, "a chain needs at least one reflection");
    TORCH_CHECK(prefix_faces.size(1) + suffix_faces.size(1) + 1 <= 8, "chain depth exceeds 8");
    TORCH_CHECK(surface_group_size.numel() > 0, "surface_group_size must contain at least one group");
    TORCH_CHECK(surface_group_members.numel() % surface_group_size.numel() == 0,
                "surface_group_members must be padded by group count");
    TORCH_CHECK(std::isfinite(min_segment_length) && min_segment_length >= 0.0,
                "min_segment_length must be finite and non-negative");

    // Each leg is traced only for rows whose previous leg is valid.
    const at::Tensor all_rows = at::ones({count}, source.options().dtype(at::kBool));
    ChainLeg prefix = trace_chain_leg(scene, source, edge_point, all_rows, prefix_faces, prefix_plane_points,
                                      prefix_plane_normals, surface_group_id, surface_group_size,
                                      surface_group_members, min_segment_length);
    ChainLeg suffix = trace_chain_leg(scene, receiver, edge_point, prefix.valid, suffix_faces, suffix_plane_points,
                                      suffix_plane_normals, surface_group_id, surface_group_size,
                                      surface_group_members, min_segment_length);
    return channel_coupled_chain_finalize_cuda(
        prefix.valid, suffix.valid, prefix.path_length, suffix.path_length, prefix.prims, suffix.prims, prefix.hits,
        suffix.hits, prefix.normals, suffix.normals, source, receiver, edge_point, edge_id, min_segment_length);
}

pybind11::dict channel_coupled_dd_geometry_forward(
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
    std::vector<at::Tensor> prepared = channel_coupled_dd_prepare_cuda(
        edge1_id,
        edge2_id,
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
    TORCH_CHECK(prepared.size() == 4,
                "coupled D-D prepare returned an unexpected tensor count");
    at::Tensor candidate_active = prepared[0];
    at::Tensor q1 = prepared[1];
    at::Tensor q2 = prepared[2];

    // RayD segment visibility of the three legs, each traced only for candidates whose
    // previous legs are visible. The outer legs are traced from the diffraction point to the
    // endpoint and the middle leg in both directions (visible when either trace is), so a
    // candidate and its reverse (endpoints and edges exchanged) trace the same segments.
    rayd::torch::VisibilityResult leg1 = rayd::torch::visibility_forward(
        scene.resource(), {q1, source, candidate_active});
    at::Tensor prefix_active = channel_coupled_active_mask_cuda(candidate_active, leg1.visible);
    rayd::torch::VisibilityResult leg2 = rayd::torch::visibility_forward(
        scene.resource(), {q1, q2, prefix_active});
    rayd::torch::VisibilityResult leg2_reverse = rayd::torch::visibility_forward(
        scene.resource(), {q2, q1, prefix_active});
    prefix_active = channel_coupled_active_either_mask_cuda(prefix_active, leg2.visible, leg2_reverse.visible);
    rayd::torch::VisibilityResult leg3 = rayd::torch::visibility_forward(
        scene.resource(), {q2, receiver, prefix_active});
    prefix_active = channel_coupled_active_mask_cuda(prefix_active, leg3.visible);

    pybind11::dict out = channel_coupled_dd_finalize_cuda(
        prefix_active,
        edge1_id,
        edge2_id,
        q1,
        q2,
        source,
        receiver);
    out["candidate_active"] = candidate_active;
    out["certificate_rejection_count"] = prepared[3];
    return out;
}
