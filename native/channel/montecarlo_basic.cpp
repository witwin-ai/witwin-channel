// Copyright Xingyu Chen.
// Implements montecarlo basic native integration.

#include <torch/extension.h>
#include "tensor_checks.h"
#include <cmath>
#include <vector>
#include "kernels/reflection_receivers.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering);
at::Tensor channel_mc_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
at::Tensor channel_mc_store_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    int64_t tx_index);
at::Tensor channel_mc_store_scaled_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    at::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
at::Tensor channel_mc_sample_directions_cuda(int64_t count, at::Tensor reference);
at::Tensor channel_mc_diffraction_discover_edges_cuda(
    at::Tensor tx_pos,
    at::Tensor ray_dir,
    at::Tensor prim_index,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_geo_n,
    at::Tensor triangle_edge_count,
    at::Tensor triangle_edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor edge_line_min,
    at::Tensor edge_line_max,
    at::Tensor edge_adjacent_face1);
at::Tensor channel_mc_diffraction_discover_edges_counted_cuda(
    at::Tensor tx_pos,
    at::Tensor ray_dir,
    at::Tensor prim_index,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_geo_n,
    at::Tensor hit_count,
    at::Tensor triangle_edge_count,
    at::Tensor triangle_edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor edge_line_min,
    at::Tensor edge_line_max,
    at::Tensor edge_adjacent_face1);

namespace {

void check_mc_diffraction_discovery_inputs(
    const at::Tensor &tx_pos,
    const at::Tensor &ray_dir,
    const at::Tensor &prim_index,
    const at::Tensor &hit_p,
    const at::Tensor &hit_n,
    const at::Tensor &hit_geo_n,
    const at::Tensor *hit_count,
    const at::Tensor &triangle_edge_count,
    const at::Tensor &triangle_edge_indices,
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &edge_n0,
    const at::Tensor &edge_n1,
    const at::Tensor &edge_line_min,
    const at::Tensor &edge_line_max,
    const at::Tensor &edge_adjacent_face1) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
    check_tensor(tx_pos, "tx_pos", at::kFloat, 1);
    check_vec3_table(ray_dir, "ray_dir");
    check_flat_tensor(prim_index, "prim_index", at::kInt);
    check_vec3_table(hit_p, "hit_p");
    check_vec3_table(hit_n, "hit_n");
    check_vec3_table(hit_geo_n, "hit_geo_n");
    check_flat_tensor(triangle_edge_count, "triangle_edge_count", at::kInt);
    check_tensor(triangle_edge_indices, "triangle_edge_indices", at::kInt, 2);
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_vec3_table(edge_n0, "edge_n0");
    check_vec3_table(edge_n1, "edge_n1");
    check_flat_tensor(edge_line_min, "edge_line_min", at::kFloat);
    check_flat_tensor(edge_line_max, "edge_line_max", at::kFloat);
    check_flat_tensor(edge_adjacent_face1, "edge_adjacent_face1", at::kInt);
    if (hit_count != nullptr) {
        check_flat_tensor(*hit_count, "hit_count", at::kInt);
        TORCH_CHECK(hit_count->numel() == 1, "hit_count must contain one element");
    }
    TORCH_CHECK(tx_pos.numel() == 3, "tx_pos must have shape (3,)");
    const int64_t capacity = ray_dir.size(0);
    for (const auto &tensor : {prim_index, hit_p, hit_n, hit_geo_n}) {
        TORCH_CHECK(tensor.size(0) == capacity,
                    "ray hit tensors must match ray_dir capacity");
    }
    TORCH_CHECK(triangle_edge_indices.size(0) == triangle_edge_count.size(0),
                "triangle edge tables must have matching face rows");
    const int64_t edge_count = edge_pos.size(0);
    for (const auto &tensor : {edge_dir, edge_n0, edge_n1, edge_line_min,
                               edge_line_max, edge_adjacent_face1}) {
        TORCH_CHECK(tensor.size(0) == edge_count,
                    "edge tensors must match edge_pos rows");
    }
    const int device = ray_dir.get_device();
    for (const auto &tensor : {tx_pos, prim_index, hit_p, hit_n, hit_geo_n,
                               triangle_edge_count, triangle_edge_indices,
                               edge_pos, edge_dir, edge_n0, edge_n1,
                               edge_line_min, edge_line_max,
                               edge_adjacent_face1}) {
        TORCH_CHECK(tensor.get_device() == device,
                    "MC diffraction discovery tensors must share one CUDA device");
    }
    if (hit_count != nullptr) {
        TORCH_CHECK(hit_count->get_device() == device,
                    "hit_count must share the discovery CUDA device");
    }
}

}  // namespace
at::Tensor channel_mc_los_component_maps_cuda(at::Tensor los);
at::Tensor channel_mc_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_los_path_gain_backward_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor grad_output,
    double frequency_hz,
    at::Tensor tx_pol);
at::Tensor channel_mc_los_path_gain_jvp_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor tx_tangent,
    at::Tensor power_tangent,
    at::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz,
    double frequency_tangent,
    at::Tensor tx_pol);
at::Tensor channel_mc_los_component_maps_adjoint_cuda(
    at::Tensor grad_maps,
    at::Tensor visible);
at::Tensor channel_mc_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index);
std::tuple<at::Tensor, at::Tensor> channel_mc_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
at::Tensor channel_mc_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z);
at::Tensor channel_mc_receiver_grid_points_cuda(
    at::Tensor reference,
    int64_t rows,
    int64_t cols,
    double origin_x,
    double origin_y,
    double origin_z,
    double x_axis_x,
    double x_axis_y,
    double x_axis_z,
    double y_axis_x,
    double y_axis_y,
    double y_axis_z,
    double spacing0,
    double spacing1);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_mc_scattering_pair_legs_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers);
at::Tensor channel_mc_scattering_pair_sum_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers, at::Tensor visible, at::Tensor value_te,
    at::Tensor value_tm, at::Tensor deposited);
std::tuple<at::Tensor, at::Tensor> channel_mc_scattering_pair_sum_backward_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers, at::Tensor visible, at::Tensor grad);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_reflection_rays_cuda(
    at::Tensor tx_positions, at::Tensor bounds, bool use_bounds, int64_t sample_count, int64_t seed);
at::Tensor channel_mc_scene_bounds_cuda(at::Tensor vertices);
void channel_mc_reflection_final_segment_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor material_valid,
    const ReflectionReceiverArgs& receivers, double wavelength, bool first_block,
    at::Tensor final_o, at::Tensor final_d, at::Tensor final_tmax, at::Tensor final_offset,
    at::Tensor final_active);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_mc_reflection_append_final_cuda(
    at::Tensor trace_valid, at::Tensor trace_t, at::Tensor trace_prim, at::Tensor final_valid,
    at::Tensor final_t, at::Tensor final_prim, at::Tensor final_offset);
at::Tensor channel_mc_slab_reflection_accumulate_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, at::Tensor tx_pol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_slab_reflection_accumulate_backward_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, double wavelength_dfreq, at::Tensor tx_pol);
at::Tensor channel_mc_slab_reflection_accumulate_jvp_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness,
    int64_t contribution_depth, const ReflectionReceiverArgs& receivers, double wavelength,
    at::Tensor solid_angle, double wavelength_tangent, at::Tensor tx_pol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id);
at::Tensor channel_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src);
at::Tensor channel_mc_utd_diffraction_tape_accumulate_cuda(
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,double,double,double,double,double,
    int64_t,int64_t,double,double,int64_t,double,at::Tensor,const PointTapeTensors&);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_utd_diffraction_tape_accumulate_backward_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness,
    at::Tensor grad_output,
    bool need_materials, bool need_source, bool need_frequency,
    int64_t axis, double plane,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed,
    double total_edge_length, double wavelength_dfreq, at::Tensor tx_pol,
    const PointTapeTensors& point_tape);
at::Tensor channel_mc_utd_diffraction_tape_accumulate_jvp_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness, at::Tensor tangent_source,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness, bool has_tangent_source,
    int64_t axis, double plane,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed,
    double total_edge_length, double wavelength_tangent, at::Tensor tx_pol,
    const PointTapeTensors& point_tape);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_point_diffraction_tape_cuda(
    at::Tensor sample_state, at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min,
    at::Tensor t_max, at::Tensor n0, at::Tensor nn, at::Tensor source, at::Tensor points,
    int64_t point_begin, int64_t point_count, int64_t seed, double point_radius);
at::Tensor channel_mc_selected_edge_indices_cuda(at::Tensor selected);
std::vector<at::Tensor> channel_mc_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
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
    at::Tensor tx_power);
std::vector<at::Tensor> channel_mc_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol);
std::vector<at::Tensor> channel_mc_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol);

torch::Tensor channel_mc_sample_directions(int64_t count, torch::Tensor reference) {
    return channel_mc_sample_directions_cuda(count, reference);
}

torch::Tensor channel_mc_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z) {
    return channel_mc_pack_vec3_cuda(x, y, z);
}

torch::Tensor channel_mc_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    return channel_mc_component_map_buffer_cuda(reference, tx_count, dim0, dim1);
}

torch::Tensor channel_mc_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index) {
    return channel_mc_store_component_map_cuda(maps, source, tx_index);
}

torch::Tensor channel_mc_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index) {
    return channel_mc_store_scaled_component_map_cuda(maps, source, scale_values, tx_index, scale_index);
}

torch::Tensor channel_mc_los_component_maps(torch::Tensor los) {
    return channel_mc_los_component_maps_cuda(los);
}

torch::Tensor channel_mc_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols) {
    return channel_mc_los_component_maps_from_matrix_cuda(los, rows, cols);
}

pybind11::tuple channel_mc_los_path_gain_backward(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor grad_output,
    double frequency_hz,
    torch::Tensor tx_pol) {
    auto [grad_tx, grad_power, grad_rx, grad_frequency] =
        channel_mc_los_path_gain_backward_cuda(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            frequency_hz,
            tx_pol);
    pybind11::tuple out(4);
    out[0] = grad_tx;
    out[1] = grad_power;
    out[2] = grad_rx;
    out[3] = grad_frequency;
    return out;
}

torch::Tensor channel_mc_los_path_gain_jvp(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor tx_tangent,
    torch::Tensor power_tangent,
    torch::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz,
    double frequency_tangent,
    torch::Tensor tx_pol) {
    return channel_mc_los_path_gain_jvp_cuda(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        has_tx_tangent,
        has_power_tangent,
        has_rx_tangent,
        frequency_hz,
        frequency_tangent,
        tx_pol);
}

torch::Tensor channel_mc_los_component_maps_adjoint(
    torch::Tensor grad_maps,
    torch::Tensor visible) {
    return channel_mc_los_component_maps_adjoint_cuda(grad_maps, visible);
}

torch::Tensor channel_mc_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index) {
    return channel_mc_apply_los_visibility_cuda(maps, los, visible, tx_index);
}

pybind11::dict channel_mc_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    auto [start, active] = channel_mc_los_visibility_inputs_cuda(tx_positions, tx_index, rx_count);
    pybind11::dict out;
    out["start"] = start;
    out["active"] = active;
    return out;
}

torch::Tensor channel_mc_receiver_grid_points(
    torch::Tensor reference,
    int64_t rows,
    int64_t cols,
    double origin_x,
    double origin_y,
    double origin_z,
    double x_axis_x,
    double x_axis_y,
    double x_axis_z,
    double y_axis_x,
    double y_axis_y,
    double y_axis_z,
    double spacing0,
    double spacing1) {
    return channel_mc_receiver_grid_points_cuda(
        reference,
        rows,
        cols,
        origin_x,
        origin_y,
        origin_z,
        x_axis_x,
        x_axis_y,
        x_axis_z,
        y_axis_x,
        y_axis_y,
        y_axis_z,
        spacing0,
        spacing1);
}

namespace {

void check_reflection_trace(
    const torch::Tensor& ray_o, const torch::Tensor& ray_d, const torch::Tensor& trace_valid,
    const torch::Tensor& trace_t, const torch::Tensor& trace_prim, const torch::Tensor& face_normals) {
    TORCH_CHECK(ray_o.is_cuda() && ray_d.is_cuda(), "reflection rays must be CUDA tensors");
    TORCH_CHECK(ray_o.scalar_type() == at::kFloat && ray_d.scalar_type() == at::kFloat,
                "reflection rays must be float32");
    TORCH_CHECK(ray_o.dim() == 2 && ray_o.size(1) == 3 && ray_d.sizes() == ray_o.sizes(),
                "reflection rays must have shape (N, 3)");
    TORCH_CHECK(trace_valid.scalar_type() == at::kBool && trace_valid.dim() == 2,
                "trace_valid must be a rank-2 bool tensor");
    TORCH_CHECK(trace_t.scalar_type() == at::kFloat && trace_t.sizes() == trace_valid.sizes(),
                "trace_t must match trace_valid");
    TORCH_CHECK(trace_prim.scalar_type() == at::kInt && trace_prim.sizes() == trace_valid.sizes(),
                "trace_prim must match trace_valid");
    TORCH_CHECK(trace_valid.size(0) == ray_o.size(0), "trace batch must match rays");
    TORCH_CHECK(face_normals.scalar_type() == at::kFloat && face_normals.dim() == 2 && face_normals.size(1) == 3,
                "face_normals must have shape (F, 3)");
}

ReflectionReceiverArgs reflection_receivers(
    int64_t axis, double plane_position, double coord0_min, double coord0_max, double coord1_min,
    double coord1_max, int64_t resolution0, int64_t resolution1, double cell_area,
    const torch::Tensor& points, double point_radius, const torch::Tensor& reference) {
    ReflectionReceiverArgs args;
    TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat && points.dim() == 2 &&
                    points.size(1) == 3 && points.get_device() == reference.get_device(),
                "points must be a float32 CUDA tensor of shape (P, 3)");
    if (points.size(0) > 0) {
        TORCH_CHECK(std::isfinite(point_radius) && point_radius != 0.0,
                    "point_radius must be finite and non-zero");
        args.points = points.contiguous();
        args.point_radius = point_radius;
        return args;
    }
    TORCH_CHECK(axis >= 0 && axis <= 2, "axis must be 0, 1, or 2");
    TORCH_CHECK(resolution0 > 0 && resolution1 > 0, "grid resolution must be positive");
    TORCH_CHECK(cell_area > 0.0, "cell_area must be positive");
    args.axis = axis;
    args.plane_position = plane_position;
    args.coord0_min = coord0_min;
    args.coord0_max = coord0_max;
    args.coord1_min = coord1_min;
    args.coord1_max = coord1_max;
    args.resolution0 = resolution0;
    args.resolution1 = resolution1;
    args.cell_area = cell_area;
    return args;
}

void check_solid_angle(const torch::Tensor& solid_angle, const torch::Tensor& reference) {
    TORCH_CHECK(solid_angle.is_cuda() && solid_angle.scalar_type() == at::kFloat &&
                    solid_angle.numel() == 1 && solid_angle.get_device() == reference.get_device(),
                "solid_angle must be a float32 CUDA tensor with one element");
}

void check_tx_pol(const torch::Tensor& tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
}

}  // namespace

pybind11::dict channel_mc_reflection_rays(
    torch::Tensor tx_positions, pybind11::object bounds, int64_t sample_count, int64_t seed) {
    TORCH_CHECK(tx_positions.is_cuda() && tx_positions.scalar_type() == at::kFloat &&
                    tx_positions.dim() == 2 && tx_positions.size(1) == 3,
                "tx_positions must be a float32 CUDA tensor of shape (T, 3)");
    TORCH_CHECK(sample_count > 0, "sample_count must be positive");
    torch::Tensor bounds_tensor;
    const bool use_bounds = !bounds.is_none();
    if (use_bounds) {
        bounds_tensor = bounds.cast<torch::Tensor>();
        TORCH_CHECK(bounds_tensor.is_cuda() && bounds_tensor.scalar_type() == at::kFloat &&
                        bounds_tensor.numel() == 6 &&
                        bounds_tensor.get_device() == tx_positions.get_device(),
                    "bounds must be a float32 CUDA tensor with 6 elements");
        bounds_tensor = bounds_tensor.contiguous();
    }
    auto [ray_o, ray_d, active, solid_angle] = channel_mc_reflection_rays_cuda(
        tx_positions.contiguous(), bounds_tensor, use_bounds, sample_count, seed);
    pybind11::dict out;
    out["ray_o"] = ray_o;
    out["ray_d"] = ray_d;
    out["active"] = active;
    out["solid_angle"] = solid_angle;
    return out;
}

torch::Tensor channel_mc_scene_bounds(torch::Tensor vertices) {
    TORCH_CHECK(vertices.is_cuda() && vertices.scalar_type() == at::kFloat && vertices.dim() == 2 &&
                    vertices.size(1) == 3 && vertices.size(0) > 0,
                "vertices must be a non-empty float32 CUDA tensor of shape (V, 3)");
    return channel_mc_scene_bounds_cuda(vertices.contiguous());
}

void channel_mc_reflection_final_segment(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid, torch::Tensor trace_t,
    torch::Tensor trace_prim, torch::Tensor face_normals, torch::Tensor material_valid,
    int64_t axis, double plane_position, double coord0_min, double coord0_max,
    double coord1_min, double coord1_max, int64_t resolution0, int64_t resolution1,
    double cell_area, torch::Tensor points, double point_radius, double wavelength,
    bool first_block, torch::Tensor final_o, torch::Tensor final_d, torch::Tensor final_tmax,
    torch::Tensor final_offset, torch::Tensor final_active) {
    check_reflection_trace(ray_o, ray_d, trace_valid, trace_t, trace_prim, face_normals);
    TORCH_CHECK(material_valid.scalar_type() == at::kBool &&
                    material_valid.numel() == face_normals.size(0),
                "material_valid must cover every face");
    const int64_t rays = ray_o.size(0);
    for (const auto* tensor : {&final_o, &final_d})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->is_contiguous() &&
                        tensor->sizes() == ray_o.sizes(),
                    "final_o and final_d must be contiguous float32 (N, 3) tensors");
    for (const auto* tensor : {&final_tmax, &final_offset})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->is_contiguous() &&
                        tensor->dim() == 1 && tensor->size(0) == rays,
                    "final_tmax and final_offset must be contiguous float32 (N,) tensors");
    TORCH_CHECK(final_active.is_cuda() && final_active.scalar_type() == at::kBool &&
                    final_active.is_contiguous() && final_active.dim() == 1 && final_active.size(0) == rays,
                "final_active must be a contiguous bool (N,) tensor");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    const ReflectionReceiverArgs receivers = reflection_receivers(
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max, resolution0,
        resolution1, cell_area, points, point_radius, ray_o);
    channel_mc_reflection_final_segment_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(), trace_t.contiguous(),
        trace_prim.contiguous(), face_normals.contiguous(), material_valid.contiguous(), receivers,
        wavelength, first_block, final_o, final_d, final_tmax, final_offset, final_active);
}

pybind11::tuple channel_mc_reflection_append_final(
    torch::Tensor trace_valid, torch::Tensor trace_t, torch::Tensor trace_prim,
    torch::Tensor final_valid, torch::Tensor final_t, torch::Tensor final_prim,
    torch::Tensor final_offset) {
    TORCH_CHECK(trace_valid.is_cuda() && trace_valid.scalar_type() == at::kBool && trace_valid.dim() == 2,
                "trace_valid must be a rank-2 bool CUDA tensor");
    TORCH_CHECK(trace_t.scalar_type() == at::kFloat && trace_t.sizes() == trace_valid.sizes(),
                "trace_t must match trace_valid");
    TORCH_CHECK(trace_prim.scalar_type() == at::kInt && trace_prim.sizes() == trace_valid.sizes(),
                "trace_prim must match trace_valid");
    const int64_t rays = trace_valid.size(0);
    TORCH_CHECK(final_valid.scalar_type() == at::kBool && final_valid.numel() == rays,
                "final_valid must have one entry per ray");
    TORCH_CHECK(final_t.scalar_type() == at::kFloat && final_t.numel() == rays,
                "final_t must have one entry per ray");
    TORCH_CHECK(final_prim.scalar_type() == at::kInt && final_prim.numel() == rays,
                "final_prim must have one entry per ray");
    TORCH_CHECK(final_offset.scalar_type() == at::kFloat && final_offset.numel() == rays,
                "final_offset must have one entry per ray");
    auto [valid, t, prim] = channel_mc_reflection_append_final_cuda(
        trace_valid.contiguous(), trace_t.contiguous(), trace_prim.contiguous(),
        final_valid.contiguous(), final_t.contiguous(), final_prim.contiguous(),
        final_offset.contiguous());
    pybind11::tuple out(3);
    out[0] = valid;
    out[1] = t;
    out[2] = prim;
    return out;
}

torch::Tensor channel_mc_slab_reflection_accumulate(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    torch::Tensor solid_angle, double cell_area, torch::Tensor tx_pol,
    torch::Tensor points, double point_radius) {
    check_reflection_trace(ray_o, ray_d, trace_valid, trace_t, trace_prim, face_normals);
    TORCH_CHECK(eta_r.sizes() == sigma.sizes() && eta_r.sizes() == gain.sizes() &&
                eta_r.sizes() == material_valid.sizes() && eta_r.sizes() == thickness.sizes(),
                "per-face material tensors must have matching shape");
    TORCH_CHECK(eta_r.numel() == face_normals.size(0), "materials must cover every face");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    check_solid_angle(solid_angle, ray_o);
    check_tx_pol(tx_pol);
    const ReflectionReceiverArgs receivers = reflection_receivers(
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max, resolution0,
        resolution1, cell_area, points, point_radius, ray_o);
    return channel_mc_slab_reflection_accumulate_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(), trace_t.contiguous(),
        trace_prim.contiguous(), face_normals.contiguous(), eta_r.contiguous(), sigma.contiguous(),
        gain.contiguous(), material_valid.contiguous(), thickness.contiguous(), contribution_depth,
        receivers, wavelength, solid_angle.contiguous(), tx_pol.contiguous());
}

pybind11::tuple channel_mc_slab_reflection_accumulate_backward(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    torch::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    torch::Tensor solid_angle, double cell_area, double wavelength_dfreq,
    torch::Tensor tx_pol, torch::Tensor points, double point_radius) {
    check_reflection_trace(ray_o, ray_d, trace_valid, trace_t, trace_prim, face_normals);
    check_solid_angle(solid_angle, ray_o);
    check_tx_pol(tx_pol);
    const ReflectionReceiverArgs receivers = reflection_receivers(
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max, resolution0,
        resolution1, cell_area, points, point_radius, ray_o);
    auto [grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_frequency] =
        channel_mc_slab_reflection_accumulate_backward_cuda(
            ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(),
            trace_t.contiguous(), trace_prim.contiguous(), face_normals.contiguous(),
            eta_r.contiguous(), sigma.contiguous(), gain.contiguous(),
            material_valid.contiguous(), thickness.contiguous(), grad_output,
            need_materials, need_frequency, contribution_depth, receivers, wavelength,
            solid_angle.contiguous(), wavelength_dfreq, tx_pol.contiguous());
    pybind11::tuple out(5);
    out[0] = grad_eta_r;
    out[1] = grad_sigma;
    out[2] = grad_gain;
    out[3] = grad_thickness;
    out[4] = grad_frequency;
    return out;
}

torch::Tensor channel_mc_slab_reflection_accumulate_jvp(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    torch::Tensor tangent_eta_r, torch::Tensor tangent_sigma,
    torch::Tensor tangent_gain, torch::Tensor tangent_thickness,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    torch::Tensor solid_angle, double cell_area, double wavelength_tangent,
    torch::Tensor tx_pol, torch::Tensor points, double point_radius) {
    check_reflection_trace(ray_o, ray_d, trace_valid, trace_t, trace_prim, face_normals);
    check_solid_angle(solid_angle, ray_o);
    check_tx_pol(tx_pol);
    const ReflectionReceiverArgs receivers = reflection_receivers(
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max, resolution0,
        resolution1, cell_area, points, point_radius, ray_o);
    return channel_mc_slab_reflection_accumulate_jvp_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(),
        trace_t.contiguous(), trace_prim.contiguous(), face_normals.contiguous(),
        eta_r.contiguous(), sigma.contiguous(), gain.contiguous(),
        material_valid.contiguous(), thickness.contiguous(),
        has_tangent_eta_r ? tangent_eta_r.contiguous() : tangent_eta_r,
        has_tangent_sigma ? tangent_sigma.contiguous() : tangent_sigma,
        has_tangent_gain ? tangent_gain.contiguous() : tangent_gain,
        has_tangent_thickness ? tangent_thickness.contiguous() : tangent_thickness,
        has_tangent_eta_r, has_tangent_sigma, has_tangent_gain,
        has_tangent_thickness, contribution_depth, receivers, wavelength,
        solid_angle.contiguous(), wavelength_tangent, tx_pol.contiguous());
}

torch::Tensor channel_mc_diffraction_discover_edges(
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
    torch::Tensor edge_n1,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1) {
    check_mc_diffraction_discovery_inputs(
        tx_pos, ray_dir, prim_index, hit_p, hit_n, hit_geo_n, nullptr,
        triangle_edge_count, triangle_edge_indices, edge_pos, edge_dir,
        edge_n0, edge_n1, edge_line_min, edge_line_max, edge_adjacent_face1);
    return channel_mc_diffraction_discover_edges_cuda(
        tx_pos, ray_dir, prim_index, hit_p, hit_n, hit_geo_n,
        triangle_edge_count, triangle_edge_indices, edge_pos, edge_dir,
        edge_n0, edge_n1, edge_line_min, edge_line_max, edge_adjacent_face1);
}

torch::Tensor channel_mc_diffraction_discover_edges_counted(
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
    torch::Tensor edge_n1,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1) {
    check_mc_diffraction_discovery_inputs(
        tx_pos, ray_dir, prim_index, hit_p, hit_n, hit_geo_n, &hit_count,
        triangle_edge_count, triangle_edge_indices, edge_pos, edge_dir,
        edge_n0, edge_n1, edge_line_min, edge_line_max, edge_adjacent_face1);
    return channel_mc_diffraction_discover_edges_counted_cuda(
        tx_pos, ray_dir, prim_index, hit_p, hit_n, hit_geo_n, hit_count,
        triangle_edge_count, triangle_edge_indices, edge_pos, edge_dir,
        edge_n0, edge_n1, edge_line_min, edge_line_max, edge_adjacent_face1);
}

torch::Tensor channel_mc_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src) {
    return channel_mc_diffraction_state_wi_cuda(state_edge_pos, state_src);
}

namespace {

// Point-receiver tape arguments of the diffraction tape ops; an empty `points` table is the
// grid tape.
PointTapeTensors point_tape_tensors(
    const torch::Tensor& tape_lane, const torch::Tensor& points, double point_radius,
    int64_t edge_samples, const torch::Tensor& tape_active) {
    PointTapeTensors tensors;
    TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat && points.dim() == 2 &&
                    points.size(1) == 3,
                "points must be a float32 CUDA tensor of shape (P, 3)");
    if (points.size(0) == 0) return tensors;
    TORCH_CHECK(tape_lane.is_cuda() && tape_lane.scalar_type() == at::kInt &&
                    tape_lane.numel() == tape_active.numel(),
                "tape_lane must be an int32 CUDA tensor with one entry per tape row");
    TORCH_CHECK(std::isfinite(point_radius) && point_radius != 0.0,
                "point_radius must be finite and non-zero");
    TORCH_CHECK(edge_samples > 0, "edge_samples must be positive for a point tape");
    tensors.lane = tape_lane.contiguous();
    tensors.points = points.contiguous();
    tensors.point_radius = point_radius;
    tensors.edge_samples = edge_samples;
    return tensors;
}

}  // namespace

pybind11::tuple channel_mc_point_diffraction_tape(
    torch::Tensor sample_state, torch::Tensor edge_pos, torch::Tensor edge_dir,
    torch::Tensor t_min, torch::Tensor t_max, torch::Tensor n0, torch::Tensor nn,
    torch::Tensor source, torch::Tensor points, int64_t point_begin, int64_t point_count,
    int64_t seed, double point_radius) {
    TORCH_CHECK(sample_state.is_cuda() && sample_state.scalar_type() == at::kInt &&
                    sample_state.dim() == 1,
                "sample_state must be a rank-1 int32 CUDA tensor");
    TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat && points.dim() == 2 &&
                    points.size(1) == 3,
                "points must be a float32 CUDA tensor of shape (P, 3)");
    TORCH_CHECK(point_begin >= 0 && point_count >= 0 && point_begin + point_count <= points.size(0),
                "point range is out of bounds");
    TORCH_CHECK(std::isfinite(point_radius) && point_radius != 0.0,
                "point_radius must be finite and non-zero");
    for (const auto* tensor : {&edge_pos, &edge_dir, &n0, &nn, &source})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->dim() == 2 &&
                        tensor->size(1) == 3 && tensor->size(0) == edge_pos.size(0),
                    "edge state vectors must be float32 CUDA tensors of shape (S, 3)");
    for (const auto* tensor : {&t_min, &t_max})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat &&
                        tensor->numel() == edge_pos.size(0),
                    "edge parameter bounds must have one entry per state");
    auto out = channel_mc_point_diffraction_tape_cuda(
        sample_state.contiguous(), edge_pos.contiguous(), edge_dir.contiguous(), t_min.contiguous(),
        t_max.contiguous(), n0.contiguous(), nn.contiguous(), source.contiguous(),
        points.contiguous(), point_begin, point_count, seed, point_radius);
    return pybind11::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out),
                                std::get<3>(out), std::get<4>(out), std::get<5>(out),
                                std::get<6>(out), std::get<7>(out));
}

torch::Tensor channel_mc_utd_diffraction_tape_accumulate(
    torch::Tensor tape_active,torch::Tensor tape_state,torch::Tensor tape_cell,torch::Tensor tape_u,
    torch::Tensor edge_pos,torch::Tensor edge_dir,torch::Tensor t_min,torch::Tensor t_max,
    torch::Tensor n0,torch::Tensor nn,torch::Tensor prim0,torch::Tensor prim1,
    torch::Tensor exterior_angle,torch::Tensor source,torch::Tensor source_power,
    torch::Tensor eta_r,torch::Tensor sigma,torch::Tensor mu_r,torch::Tensor gain,
    torch::Tensor material_valid,torch::Tensor thickness,int64_t axis,double plane,double c0min,double c0max,
    double c1min,double c1max,int64_t r0,int64_t r1,double wavelength,double cell_area,int64_t seed,double total_edge_length,
    torch::Tensor tx_pol,
    torch::Tensor tape_lane, torch::Tensor points, double point_radius, int64_t edge_samples) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_utd_diffraction_tape_accumulate_cuda(
        tape_active,tape_state,tape_cell,tape_u,edge_pos,edge_dir,t_min,t_max,n0,nn,prim0,prim1,
        exterior_angle,source,source_power,eta_r,sigma,mu_r,gain,material_valid,thickness,axis,plane,
        c0min,c0max,c1min,c1max,r0,r1,wavelength,cell_area,seed,total_edge_length,tx_pol.contiguous(),
        point_tape_tensors(tape_lane, points, point_radius, edge_samples, tape_active));
}

pybind11::tuple channel_mc_utd_diffraction_tape_accumulate_backward(
    torch::Tensor tape_active, torch::Tensor tape_state, torch::Tensor tape_cell,
    torch::Tensor tape_u, torch::Tensor edge_pos, torch::Tensor edge_dir,
    torch::Tensor t_min, torch::Tensor t_max, torch::Tensor n0, torch::Tensor nn,
    torch::Tensor prim0, torch::Tensor prim1, torch::Tensor exterior_angle,
    torch::Tensor source, torch::Tensor source_power, torch::Tensor eta_r,
    torch::Tensor sigma, torch::Tensor mu_r, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    torch::Tensor grad_output,
    bool need_materials, bool need_source, bool need_frequency,
    int64_t axis, double plane, int64_t r0, int64_t r1, double wavelength,
    double cell_area, int64_t seed, double total_edge_length,
    double wavelength_dfreq, torch::Tensor tx_pol,
    torch::Tensor tape_lane, torch::Tensor points, double point_radius, int64_t edge_samples) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    auto gradients = channel_mc_utd_diffraction_tape_accumulate_backward_cuda(
        tape_active, tape_state, tape_cell, tape_u, edge_pos, edge_dir, t_min,
        t_max, n0, nn, prim0, prim1, exterior_angle, source, source_power,
        eta_r, sigma, mu_r, gain, material_valid, thickness, grad_output,
        need_materials, need_source, need_frequency, axis, plane, r0, r1,
        wavelength, cell_area, seed, total_edge_length, wavelength_dfreq,
        tx_pol.contiguous(), point_tape_tensors(tape_lane, points, point_radius, edge_samples, tape_active));
    return pybind11::make_tuple(
        std::get<0>(gradients), std::get<1>(gradients), std::get<2>(gradients),
        std::get<3>(gradients), std::get<4>(gradients), std::get<5>(gradients));
}

torch::Tensor channel_mc_utd_diffraction_tape_accumulate_jvp(
    torch::Tensor tape_active, torch::Tensor tape_state, torch::Tensor tape_cell,
    torch::Tensor tape_u, torch::Tensor edge_pos, torch::Tensor edge_dir,
    torch::Tensor t_min, torch::Tensor t_max, torch::Tensor n0, torch::Tensor nn,
    torch::Tensor prim0, torch::Tensor prim1, torch::Tensor exterior_angle,
    torch::Tensor source, torch::Tensor source_power, torch::Tensor eta_r,
    torch::Tensor sigma, torch::Tensor mu_r, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    torch::Tensor tangent_eta_r, torch::Tensor tangent_sigma,
    torch::Tensor tangent_gain, torch::Tensor tangent_thickness,
    torch::Tensor tangent_source,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness, bool has_tangent_source,
    int64_t axis, double plane, int64_t r0, int64_t r1, double wavelength,
    double cell_area, int64_t seed, double total_edge_length,
    double wavelength_tangent, torch::Tensor tx_pol,
    torch::Tensor tape_lane, torch::Tensor points, double point_radius, int64_t edge_samples) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_utd_diffraction_tape_accumulate_jvp_cuda(
        tape_active, tape_state, tape_cell, tape_u, edge_pos, edge_dir, t_min,
        t_max, n0, nn, prim0, prim1, exterior_angle, source, source_power,
        eta_r, sigma, mu_r, gain, material_valid, thickness, tangent_eta_r,
        tangent_sigma, tangent_gain, tangent_thickness, tangent_source,
        has_tangent_eta_r, has_tangent_sigma, has_tangent_gain,
        has_tangent_thickness, has_tangent_source, axis, plane, r0, r1,
        wavelength, cell_area, seed, total_edge_length, wavelength_tangent,
        tx_pol.contiguous(), point_tape_tensors(tape_lane, points, point_radius, edge_samples, tape_active));
}

torch::Tensor channel_mc_selected_edge_indices(torch::Tensor selected) {
    return channel_mc_selected_edge_indices_cuda(selected);
}

pybind11::tuple channel_mc_diffraction_state_pack(
    torch::Tensor edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor line_min,
    torch::Tensor line_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor exterior_angle,
    torch::Tensor tx,
    torch::Tensor tx_power) {
    auto states = channel_mc_diffraction_state_pack_cuda(
        edge_indices,
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
        tx_power);
    pybind11::tuple out(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        out[i] = states[i];
    }
    return out;
}

pybind11::tuple channel_mc_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol) {
    auto geometry = channel_mc_diffraction_edge_geometry_cuda(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        plane_tol);
    pybind11::tuple out(geometry.size());
    for (size_t i = 0; i < geometry.size(); ++i) {
        out[i] = geometry[i];
    }
    return out;
}

pybind11::tuple channel_mc_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol) {
    auto candidates = channel_mc_surface_group_edge_candidates_cuda(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        plane_tol);
    pybind11::tuple out(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        out[i] = candidates[i];
    }
    return out;
}

pybind11::dict channel_mc_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id) {
    auto [face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid] =
        channel_mc_face_material_tensors_cuda(
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            face_material_id);

    pybind11::dict out;
    out["eps_r"] = face_eps_r;
    out["sigma_e"] = face_sigma_e;
    out["mu_r"] = face_mu_r;
    out["gain"] = face_gain;
    out["valid"] = face_valid;
    return out;
}

pybind11::dict channel_mc_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power, transmission_power,
        scattering_power] =
        channel_mc_finalize_component_maps_cuda(
            los, reflection, diffraction, transmission, scattering);

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los_power"] = los_power;
    out["reflection_power"] = reflection_power;
    out["diffraction_power"] = diffraction_power;
    out["transmission_power"] = transmission_power;
    out["scattering_power"] = scattering_power;
    return out;
}

namespace {

// Validates the shared sample, table and receiver tables of the MC scattering pair ops.
void check_scattering_pairs(
    const torch::Tensor& points, const torch::Tensor& normals, const torch::Tensor& frame_t1,
    const torch::Tensor& frame_t2, const torch::Tensor& wi_local, const torch::Tensor& incident,
    const torch::Tensor& epsilon, const torch::Tensor& material, const torch::Tensor& material_slot,
    const torch::Tensor& table_offset, const torch::Tensor& table_dims, const torch::Tensor& f_te,
    const torch::Tensor& f_tm, const torch::Tensor& receivers) {
    const int64_t samples = points.size(0);
    for (const auto* tensor : {&points, &normals, &frame_t1, &frame_t2, &wi_local, &receivers})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->dim() == 2 &&
                        tensor->size(1) == 3 && tensor->is_contiguous() &&
                        tensor->get_device() == points.get_device(),
                    "scattering pair vectors must be contiguous float32 CUDA (N, 3) tensors");
    for (const auto* tensor : {&normals, &frame_t1, &frame_t2, &wi_local})
        TORCH_CHECK(tensor->size(0) == samples, "scattering sample tables must share N");
    for (const auto* tensor : {&incident, &epsilon})
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kFloat && tensor->is_contiguous() &&
                        tensor->numel() == samples,
                    "incident and epsilon must be contiguous float32 (N,) tensors");
    TORCH_CHECK(material.is_cuda() && material.scalar_type() == at::kInt && material.is_contiguous() &&
                    material.numel() == samples,
                "material must be a contiguous int32 (N,) tensor");
    TORCH_CHECK(material_slot.is_cuda() && material_slot.scalar_type() == at::kInt &&
                    material_slot.is_contiguous(),
                "material_slot must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(table_offset.is_cuda() && table_offset.scalar_type() == at::kLong &&
                    table_offset.is_contiguous() && table_dims.is_cuda() &&
                    table_dims.scalar_type() == at::kInt && table_dims.is_contiguous() &&
                    table_dims.dim() == 2 && table_dims.size(1) == 4 &&
                    table_dims.size(0) == table_offset.numel(),
                "table_offset (S,) int64 and table_dims (S, 4) int32 must describe the same slots");
    TORCH_CHECK(f_te.is_cuda() && f_te.scalar_type() == at::kFloat && f_te.is_contiguous() &&
                    f_tm.sizes() == f_te.sizes() && f_tm.scalar_type() == at::kFloat &&
                    f_tm.is_contiguous(),
                "f_te and f_tm must be contiguous float32 tables of one shape");
}

void check_pair_mask(const torch::Tensor& visible, const torch::Tensor& points,
                     const torch::Tensor& receivers) {
    TORCH_CHECK(visible.is_cuda() && visible.scalar_type() == at::kBool && visible.is_contiguous() &&
                    visible.numel() == points.size(0) * receivers.size(0),
                "visible must be a contiguous bool tensor with one entry per (receiver, sample)");
}

}  // namespace

pybind11::tuple channel_mc_scattering_pair_legs(
    torch::Tensor points, torch::Tensor normals, torch::Tensor frame_t1, torch::Tensor frame_t2,
    torch::Tensor wi_local, torch::Tensor incident, torch::Tensor epsilon, torch::Tensor material,
    torch::Tensor material_slot, torch::Tensor table_offset, torch::Tensor table_dims,
    torch::Tensor f_te, torch::Tensor f_tm, torch::Tensor receivers) {
    check_scattering_pairs(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material,
                           material_slot, table_offset, table_dims, f_te, f_tm, receivers);
    auto out = channel_mc_scattering_pair_legs_cuda(
        points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material, material_slot,
        table_offset, table_dims, f_te, f_tm, receivers);
    return pybind11::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out));
}

torch::Tensor channel_mc_scattering_pair_sum(
    torch::Tensor points, torch::Tensor normals, torch::Tensor frame_t1, torch::Tensor frame_t2,
    torch::Tensor wi_local, torch::Tensor incident, torch::Tensor epsilon, torch::Tensor material,
    torch::Tensor material_slot, torch::Tensor table_offset, torch::Tensor table_dims,
    torch::Tensor f_te, torch::Tensor f_tm, torch::Tensor receivers, torch::Tensor visible,
    torch::Tensor value_te, torch::Tensor value_tm, torch::Tensor deposited) {
    check_scattering_pairs(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material,
                           material_slot, table_offset, table_dims, f_te, f_tm, receivers);
    check_pair_mask(visible, points, receivers);
    TORCH_CHECK(value_te.sizes() == f_te.sizes() && value_tm.sizes() == f_te.sizes() &&
                    value_te.scalar_type() == at::kFloat && value_tm.scalar_type() == at::kFloat &&
                    value_te.is_contiguous() && value_tm.is_contiguous(),
                "value tables must match f_te");
    TORCH_CHECK(deposited.is_cuda() && deposited.scalar_type() == at::kLong && deposited.numel() <= 1,
                "deposited must be an empty or one-element int64 CUDA tensor");
    return channel_mc_scattering_pair_sum_cuda(
        points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material, material_slot,
        table_offset, table_dims, f_te, f_tm, receivers, visible, value_te, value_tm, deposited);
}

pybind11::tuple channel_mc_scattering_pair_sum_backward(
    torch::Tensor points, torch::Tensor normals, torch::Tensor frame_t1, torch::Tensor frame_t2,
    torch::Tensor wi_local, torch::Tensor incident, torch::Tensor epsilon, torch::Tensor material,
    torch::Tensor material_slot, torch::Tensor table_offset, torch::Tensor table_dims,
    torch::Tensor f_te, torch::Tensor f_tm, torch::Tensor receivers, torch::Tensor visible,
    torch::Tensor grad) {
    check_scattering_pairs(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material,
                           material_slot, table_offset, table_dims, f_te, f_tm, receivers);
    check_pair_mask(visible, points, receivers);
    TORCH_CHECK(grad.is_cuda() && grad.scalar_type() == at::kFloat && grad.numel() == receivers.size(0),
                "grad must be a float32 CUDA tensor with one entry per receiver");
    auto out = channel_mc_scattering_pair_sum_backward_cuda(
        points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material, material_slot,
        table_offset, table_dims, f_te, f_tm, receivers, visible, grad.contiguous());
    return pybind11::make_tuple(std::get<0>(out), std::get<1>(out));
}
