#include <torch/extension.h>
#include <vector>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction);
at::Tensor cn_mc_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
at::Tensor cn_mc_store_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    int64_t tx_index);
at::Tensor cn_mc_store_scaled_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    at::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
at::Tensor cn_mc_sample_directions_cuda(int64_t count, at::Tensor reference);
at::Tensor cn_mc_los_component_maps_cuda(at::Tensor los);
std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_mc_los_path_gain_backward_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor grad_output,
    double frequency_hz);
at::Tensor cn_mc_los_path_gain_jvp_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor tx_tangent,
    at::Tensor power_tangent,
    at::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz);
at::Tensor cn_mc_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index);
std::tuple<at::Tensor, at::Tensor> cn_mc_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
std::tuple<at::Tensor, at::Tensor> cn_mc_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host);
at::Tensor cn_mc_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z);
at::Tensor cn_mc_receiver_grid_points_cuda(
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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id);
at::Tensor cn_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src);
at::Tensor cn_mc_selected_edge_indices_cuda(at::Tensor selected);
std::vector<at::Tensor> cn_mc_diffraction_state_pack_cuda(
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
std::vector<at::Tensor> cn_mc_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol);
std::vector<at::Tensor> cn_mc_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol);

torch::Tensor cn_mc_sample_directions(int64_t count, torch::Tensor reference) {
    return cn_mc_sample_directions_cuda(count, reference);
}

pybind11::dict cn_mc_transmitter_tensors(
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
    auto [positions, power] = cn_mc_transmitter_tensors_cuda(positions_host, power_host);
    pybind11::dict out;
    out["positions"] = positions;
    out["power"] = power;
    return out;
}

torch::Tensor cn_mc_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z) {
    return cn_mc_pack_vec3_cuda(x, y, z);
}

torch::Tensor cn_mc_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    return cn_mc_component_map_buffer_cuda(reference, tx_count, dim0, dim1);
}

torch::Tensor cn_mc_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index) {
    return cn_mc_store_component_map_cuda(maps, source, tx_index);
}

torch::Tensor cn_mc_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index) {
    return cn_mc_store_scaled_component_map_cuda(maps, source, scale_values, tx_index, scale_index);
}

torch::Tensor cn_mc_los_component_maps(torch::Tensor los) {
    return cn_mc_los_component_maps_cuda(los);
}

pybind11::tuple cn_mc_los_path_gain_backward(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor grad_output,
    double frequency_hz) {
    auto [grad_tx, grad_power, grad_rx] = cn_mc_los_path_gain_backward_cuda(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        frequency_hz);
    pybind11::tuple out(3);
    out[0] = grad_tx;
    out[1] = grad_power;
    out[2] = grad_rx;
    return out;
}

torch::Tensor cn_mc_los_path_gain_jvp(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor tx_tangent,
    torch::Tensor power_tangent,
    torch::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz) {
    return cn_mc_los_path_gain_jvp_cuda(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        has_tx_tangent,
        has_power_tangent,
        has_rx_tangent,
        frequency_hz);
}

torch::Tensor cn_mc_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index) {
    return cn_mc_apply_los_visibility_cuda(maps, los, visible, tx_index);
}

pybind11::dict cn_mc_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    auto [start, active] = cn_mc_los_visibility_inputs_cuda(tx_positions, tx_index, rx_count);
    pybind11::dict out;
    out["start"] = start;
    out["active"] = active;
    return out;
}

torch::Tensor cn_mc_receiver_grid_points(
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
    return cn_mc_receiver_grid_points_cuda(
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

pybind11::dict cn_mc_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    auto [ray_o, ray_tmax, active, tx_pol] =
        cn_mc_reflection_launch_inputs_cuda(tx_positions, tx_index, sample_count);
    pybind11::dict out;
    out["ray_o"] = ray_o;
    out["ray_tmax"] = ray_tmax;
    out["active"] = active;
    out["tx_pol"] = tx_pol;
    return out;
}

torch::Tensor cn_mc_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src) {
    return cn_mc_diffraction_state_wi_cuda(state_edge_pos, state_src);
}

torch::Tensor cn_mc_selected_edge_indices(torch::Tensor selected) {
    return cn_mc_selected_edge_indices_cuda(selected);
}

pybind11::tuple cn_mc_diffraction_state_pack(
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
    auto states = cn_mc_diffraction_state_pack_cuda(
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

pybind11::tuple cn_mc_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol) {
    auto geometry = cn_mc_diffraction_edge_geometry_cuda(
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

pybind11::tuple cn_mc_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol) {
    auto candidates = cn_mc_surface_group_edge_candidates_cuda(
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

pybind11::dict cn_mc_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id) {
    auto [face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid] =
        cn_mc_face_material_tensors_cuda(
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

pybind11::dict cn_mc_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction) {
    auto [path_gain, los_power, reflection_power, diffraction_power] =
        cn_mc_finalize_component_maps_cuda(los, reflection, diffraction);

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los_power"] = los_power;
    out["reflection_power"] = reflection_power;
    out["diffraction_power"] = diffraction_power;
    return out;
}
