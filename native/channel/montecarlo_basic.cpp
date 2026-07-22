#include <torch/extension.h>
#include "tensor_checks.h"
#include <vector>

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
std::tuple<at::Tensor, at::Tensor> channel_mc_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host);
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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
at::Tensor channel_mc_sionna_reflection_accumulate_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, at::Tensor tx_pol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_sionna_reflection_accumulate_backward_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, double wavelength_dfreq,
    at::Tensor tx_pol);
at::Tensor channel_mc_sionna_reflection_accumulate_jvp_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, double wavelength_tangent,
    at::Tensor tx_pol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id);
at::Tensor channel_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src);
at::Tensor channel_mc_sionna_diffraction_tape_accumulate_cuda(
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,double,double,double,double,double,
    int64_t,int64_t,double,double,int64_t,double,at::Tensor);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_sionna_diffraction_tape_accumulate_backward_cuda(
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
    double total_edge_length, double wavelength_dfreq, at::Tensor tx_pol);
at::Tensor channel_mc_sionna_diffraction_tape_accumulate_jvp_cuda(
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
    double total_edge_length, double wavelength_tangent, at::Tensor tx_pol);
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

pybind11::dict channel_mc_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers) {
    TORCH_CHECK(flat_positions.size() % 3 == 0, "flat_positions must contain xyz triples");
    TORCH_CHECK(flat_positions.size() / 3 == powers.size(), "powers must match flat_positions");
    std::vector<float> positions_host;
    std::vector<float> power_host;
    positions_host.reserve(flat_positions.size());
    power_host.reserve(powers.size());
    for (auto item : flat_positions) {
        positions_host.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    for (auto item : powers) {
        power_host.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    auto [positions, power] = channel_mc_transmitter_tensors_cuda(positions_host, power_host);
    pybind11::dict out;
    out["positions"] = positions;
    out["power"] = power;
    return out;
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

pybind11::dict channel_mc_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    auto [ray_o, ray_tmax, active, tx_pol] =
        channel_mc_reflection_launch_inputs_cuda(tx_positions, tx_index, sample_count);
    pybind11::dict out;
    out["ray_o"] = ray_o;
    out["ray_tmax"] = ray_tmax;
    out["active"] = active;
    out["tx_pol"] = tx_pol;
    return out;
}

torch::Tensor channel_mc_sionna_reflection_accumulate(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, torch::Tensor tx_pol) {
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
    TORCH_CHECK(eta_r.sizes() == sigma.sizes() && eta_r.sizes() == gain.sizes() &&
                eta_r.sizes() == material_valid.sizes() && eta_r.sizes() == thickness.sizes(),
                "per-face material tensors must have matching shape");
    TORCH_CHECK(eta_r.numel() == face_normals.size(0), "materials must cover every face");
    TORCH_CHECK(axis >= 0 && axis <= 2, "axis must be 0, 1, or 2");
    TORCH_CHECK(resolution0 > 0 && resolution1 > 0, "grid resolution must be positive");
    TORCH_CHECK(wavelength > 0.0 && cell_area > 0.0, "wavelength and cell_area must be positive");
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_sionna_reflection_accumulate_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(), trace_t.contiguous(),
        trace_prim.contiguous(), face_normals.contiguous(), eta_r.contiguous(), sigma.contiguous(),
        gain.contiguous(), material_valid.contiguous(), thickness.contiguous(), contribution_depth,
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max,
        resolution0, resolution1, wavelength, solid_angle_per_ray, cell_area,
        tx_pol.contiguous());
}

pybind11::tuple channel_mc_sionna_reflection_accumulate_backward(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    torch::Tensor grad_output,
    bool need_materials, bool need_frequency,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area, double wavelength_dfreq,
    torch::Tensor tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    auto [grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_frequency] =
        channel_mc_sionna_reflection_accumulate_backward_cuda(
            ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(),
            trace_t.contiguous(), trace_prim.contiguous(), face_normals.contiguous(),
            eta_r.contiguous(), sigma.contiguous(), gain.contiguous(),
            material_valid.contiguous(), thickness.contiguous(), grad_output,
            need_materials, need_frequency, contribution_depth, axis, plane_position,
            coord0_min, coord0_max, coord1_min, coord1_max,
            resolution0, resolution1, wavelength, solid_angle_per_ray, cell_area,
            wavelength_dfreq, tx_pol.contiguous());
    pybind11::tuple out(5);
    out[0] = grad_eta_r;
    out[1] = grad_sigma;
    out[2] = grad_gain;
    out[3] = grad_thickness;
    out[4] = grad_frequency;
    return out;
}

torch::Tensor channel_mc_sionna_reflection_accumulate_jvp(
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
    double solid_angle_per_ray, double cell_area, double wavelength_tangent,
    torch::Tensor tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_sionna_reflection_accumulate_jvp_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(),
        trace_t.contiguous(), trace_prim.contiguous(), face_normals.contiguous(),
        eta_r.contiguous(), sigma.contiguous(), gain.contiguous(),
        material_valid.contiguous(), thickness.contiguous(),
        has_tangent_eta_r ? tangent_eta_r.contiguous() : tangent_eta_r,
        has_tangent_sigma ? tangent_sigma.contiguous() : tangent_sigma,
        has_tangent_gain ? tangent_gain.contiguous() : tangent_gain,
        has_tangent_thickness ? tangent_thickness.contiguous() : tangent_thickness,
        has_tangent_eta_r, has_tangent_sigma, has_tangent_gain,
        has_tangent_thickness, contribution_depth, axis, plane_position,
        coord0_min, coord0_max, coord1_min, coord1_max,
        resolution0, resolution1, wavelength, solid_angle_per_ray, cell_area,
        wavelength_tangent, tx_pol.contiguous());
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

torch::Tensor channel_mc_sionna_diffraction_tape_accumulate(
    torch::Tensor tape_active,torch::Tensor tape_state,torch::Tensor tape_cell,torch::Tensor tape_u,
    torch::Tensor edge_pos,torch::Tensor edge_dir,torch::Tensor t_min,torch::Tensor t_max,
    torch::Tensor n0,torch::Tensor nn,torch::Tensor prim0,torch::Tensor prim1,
    torch::Tensor exterior_angle,torch::Tensor source,torch::Tensor source_power,
    torch::Tensor eta_r,torch::Tensor sigma,torch::Tensor mu_r,torch::Tensor gain,
    torch::Tensor material_valid,torch::Tensor thickness,int64_t axis,double plane,double c0min,double c0max,
    double c1min,double c1max,int64_t r0,int64_t r1,double wavelength,double cell_area,int64_t seed,double total_edge_length,
    torch::Tensor tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_sionna_diffraction_tape_accumulate_cuda(
        tape_active,tape_state,tape_cell,tape_u,edge_pos,edge_dir,t_min,t_max,n0,nn,prim0,prim1,
        exterior_angle,source,source_power,eta_r,sigma,mu_r,gain,material_valid,thickness,axis,plane,
        c0min,c0max,c1min,c1max,r0,r1,wavelength,cell_area,seed,total_edge_length,tx_pol.contiguous());
}

pybind11::tuple channel_mc_sionna_diffraction_tape_accumulate_backward(
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
    double wavelength_dfreq, torch::Tensor tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    auto gradients = channel_mc_sionna_diffraction_tape_accumulate_backward_cuda(
        tape_active, tape_state, tape_cell, tape_u, edge_pos, edge_dir, t_min,
        t_max, n0, nn, prim0, prim1, exterior_angle, source, source_power,
        eta_r, sigma, mu_r, gain, material_valid, thickness, grad_output,
        need_materials, need_source, need_frequency, axis, plane, r0, r1,
        wavelength, cell_area, seed, total_edge_length, wavelength_dfreq,
        tx_pol.contiguous());
    return pybind11::make_tuple(
        std::get<0>(gradients), std::get<1>(gradients), std::get<2>(gradients),
        std::get<3>(gradients), std::get<4>(gradients), std::get<5>(gradients));
}

torch::Tensor channel_mc_sionna_diffraction_tape_accumulate_jvp(
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
    double wavelength_tangent, torch::Tensor tx_pol) {
    TORCH_CHECK(tx_pol.is_cuda() && tx_pol.scalar_type() == at::kFloat && tx_pol.numel() == 3,
                "tx_pol must be a float32 CUDA tensor with 3 elements");
    return channel_mc_sionna_diffraction_tape_accumulate_jvp_cuda(
        tape_active, tape_state, tape_cell, tape_u, edge_pos, edge_dir, t_min,
        t_max, n0, nn, prim0, prim1, exterior_angle, source, source_power,
        eta_r, sigma, mu_r, gain, material_valid, thickness, tangent_eta_r,
        tangent_sigma, tangent_gain, tangent_thickness, tangent_source,
        has_tangent_eta_r, has_tangent_sigma, has_tangent_gain,
        has_tangent_thickness, has_tangent_source, axis, plane, r0, r1,
        wavelength, cell_area, seed, total_edge_length, wavelength_tangent,
        tx_pol.contiguous());
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
