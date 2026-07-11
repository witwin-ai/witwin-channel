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
at::Tensor cn_mc_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols);
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
at::Tensor cn_mc_sionna_reflection_accumulate_cuda(
    at::Tensor ray_o, at::Tensor ray_d, at::Tensor trace_valid, at::Tensor trace_t,
    at::Tensor trace_prim, at::Tensor face_normals, at::Tensor eta_r, at::Tensor sigma,
    at::Tensor gain, at::Tensor material_valid, at::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id);
at::Tensor cn_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src);
at::Tensor cn_mc_sionna_diffraction_tape_accumulate_cuda(
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,double,double,double,double,double,
    int64_t,int64_t,double,double,int64_t,double);
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

torch::Tensor cn_mc_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols) {
    return cn_mc_los_component_maps_from_matrix_cuda(los, rows, cols);
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

torch::Tensor cn_mc_sionna_reflection_accumulate(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area) {
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
    return cn_mc_sionna_reflection_accumulate_cuda(
        ray_o.contiguous(), ray_d.contiguous(), trace_valid.contiguous(), trace_t.contiguous(),
        trace_prim.contiguous(), face_normals.contiguous(), eta_r.contiguous(), sigma.contiguous(),
        gain.contiguous(), material_valid.contiguous(), thickness.contiguous(), contribution_depth,
        axis, plane_position, coord0_min, coord0_max, coord1_min, coord1_max,
        resolution0, resolution1, wavelength, solid_angle_per_ray, cell_area);
}

torch::Tensor cn_mc_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src) {
    return cn_mc_diffraction_state_wi_cuda(state_edge_pos, state_src);
}

torch::Tensor cn_mc_sionna_diffraction_tape_accumulate(
    torch::Tensor tape_active,torch::Tensor tape_state,torch::Tensor tape_cell,torch::Tensor tape_u,
    torch::Tensor edge_pos,torch::Tensor edge_dir,torch::Tensor t_min,torch::Tensor t_max,
    torch::Tensor n0,torch::Tensor nn,torch::Tensor prim0,torch::Tensor prim1,
    torch::Tensor exterior_angle,torch::Tensor source,torch::Tensor source_power,
    torch::Tensor eta_r,torch::Tensor sigma,torch::Tensor mu_r,torch::Tensor gain,
    torch::Tensor material_valid,torch::Tensor thickness,int64_t axis,double plane,double c0min,double c0max,
    double c1min,double c1max,int64_t r0,int64_t r1,double wavelength,double cell_area,int64_t seed,double total_edge_length) {
    return cn_mc_sionna_diffraction_tape_accumulate_cuda(
        tape_active,tape_state,tape_cell,tape_u,edge_pos,edge_dir,t_min,t_max,n0,nn,prim0,prim1,
        exterior_angle,source,source_power,eta_r,sigma,mu_r,gain,material_valid,thickness,axis,plane,
        c0min,c0max,c1min,c1max,r0,r1,wavelength,cell_area,seed,total_edge_length);
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
