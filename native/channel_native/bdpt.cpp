#include <torch/extension.h>

#include <array>
#include <vector>

namespace {

pybind11::tuple tensor_vector_to_tuple(const std::vector<at::Tensor>& tensors) {
    pybind11::tuple out(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        out[i] = tensors[i];
    }
    return out;
}

pybind11::dict subpath_state_to_dict(const std::vector<at::Tensor>& tensors, const char* name) {
    TORCH_CHECK(tensors.size() == 19, name, " native result must contain 19 tensors");
    static constexpr std::array<const char*, 19> kFields = {
        "origin", "direction", "throughput_real", "throughput_imag", "pdf_forward",
        "pdf_reverse", "depth", "component_mask", "primitive_id", "edge_id",
        "tx_id", "rx_id", "grid_linear_id", "valid", "path_length",
        "field_real", "field_imag", "source_power", "event_type",
    };
    pybind11::dict out;
    for (size_t index = 0; index < kFields.size(); ++index) {
        out[kFields[index]] = tensors[index];
    }
    return out;
}

at::Tensor tensor_from_dict(const pybind11::dict& values, const char* field) {
    TORCH_CHECK(values.contains(field), "missing tensor field: ", field);
    return pybind11::cast<at::Tensor>(values[pybind11::str(field)]);
}

pybind11::dict connection_samples_to_dict(
    const std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>& tensors) {
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        light_depth,
        sensor_depth,
        path_length_m] = tensors;
    pybind11::dict out;
    out["topology"] = topology;
    out["contribution"] = contribution;
    out["pdf"] = pdf;
    out["mis_weight"] = mis_weight;
    out["component_id"] = component_id;
    out["valid"] = valid;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["grid_linear_id"] = grid_linear_id;
    out["light_depth"] = light_depth;
    out["sensor_depth"] = sensor_depth;
    out["path_length_m"] = path_length_m;
    return out;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_launch_state_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed);
at::Tensor cn_bdpt_sample_directions_cuda(int64_t count, at::Tensor reference, int64_t seed);
std::vector<at::Tensor> cn_bdpt_empty_subpath_state_cuda(at::Tensor reference);
std::vector<at::Tensor> cn_bdpt_light_endpoint_subpath_state_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor launch_tx_id,
    at::Tensor light_seed);
std::vector<at::Tensor> cn_bdpt_sensor_endpoint_subpath_state_cuda(
    at::Tensor rx_positions, at::Tensor rx_polarization);
std::vector<at::Tensor> cn_bdpt_reflected_light_subpath_state_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_pdf_forward,
    at::Tensor light_pdf_reverse,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_rx_id,
    at::Tensor light_grid_linear_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor hit_t,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
    double frequency_hz);
std::vector<at::Tensor> cn_bdpt_transmitted_light_subpath_state_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_pdf_forward,
    at::Tensor light_pdf_reverse,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_rx_id,
    at::Tensor light_grid_linear_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor hit_t,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz);

at::Tensor cn_bdpt_mis_weights_cuda(
    at::Tensor pdf,
    at::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_endpoint_connection_samples_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_pdf_forward,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_pdf_reverse,
    at::Tensor sensor_depth,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_grid_linear_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths);
std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_bdpt_endpoint_connection_visibility_inputs_cuda(
    at::Tensor light_origin,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor sensor_origin,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    int64_t sample_count);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy);
void cn_bdpt_filter_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor valid,
    at::Tensor visible);
int64_t cn_bdpt_count_valid_connection_samples_cuda(at::Tensor valid);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_compact_connection_samples_cuda(
    at::Tensor topology,
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor grid_linear_id,
    at::Tensor light_depth,
    at::Tensor sensor_depth,
    at::Tensor path_length_m,
    int64_t max_paths);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_concat_connection_samples_cuda(
    std::vector<at::Tensor> topologies,
    std::vector<at::Tensor> contributions,
    std::vector<at::Tensor> pdfs,
    std::vector<at::Tensor> mis_weights,
    std::vector<at::Tensor> component_ids,
    std::vector<at::Tensor> valids,
    std::vector<at::Tensor> tx_ids,
    std::vector<at::Tensor> rx_ids,
    std::vector<at::Tensor> grid_linear_ids,
    std::vector<at::Tensor> light_depths,
    std::vector<at::Tensor> sensor_depths,
    std::vector<at::Tensor> path_lengths_m);
at::Tensor cn_bdpt_connection_variance_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_diffraction_connection_samples_from_tape_cuda(
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t mode_id,
    double beta,
    int64_t strategy_count);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_diffraction_point_connection_samples_cuda(
    at::Tensor rx_positions,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t seed,
    double wavelength,
    int64_t mode_id,
    double beta,
    int64_t strategy_count);
at::Tensor cn_bdpt_zero_matrix_cuda(at::Tensor reference, int64_t rows, int64_t cols);
at::Tensor cn_core_pack_int2_cuda(at::Tensor x, at::Tensor y);
int64_t cn_core_diffraction_edge_count_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    bool vertical_only,
    double vertical_ratio,
    bool boundary_half_plane,
    double plane_tol);
std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_bdpt_point_component_power_cuda(
    at::Tensor path_gain,
    bool include_los);
at::Tensor cn_bdpt_store_point_component_column_cuda(
    at::Tensor target,
    at::Tensor source,
    int64_t rx_index);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_finalize_point_components_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz,
    at::Tensor tx_pol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering);
at::Tensor cn_bdpt_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
at::Tensor cn_bdpt_store_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    int64_t tx_index);
at::Tensor cn_bdpt_store_scaled_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    at::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
std::tuple<at::Tensor, at::Tensor> cn_bdpt_transmitter_tensors_cuda(
    const std::vector<float>& positions_host,
    const std::vector<float>& power_host);
at::Tensor cn_bdpt_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z);
at::Tensor cn_bdpt_los_component_maps_cuda(at::Tensor los);
at::Tensor cn_bdpt_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols);
at::Tensor cn_bdpt_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index);
std::tuple<at::Tensor, at::Tensor> cn_bdpt_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
at::Tensor cn_bdpt_receiver_grid_points_cuda(
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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
at::Tensor cn_bdpt_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src);
at::Tensor cn_bdpt_selected_edge_indices_cuda(at::Tensor selected);
std::vector<at::Tensor> cn_bdpt_diffraction_state_pack_cuda(
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
std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_cuda(
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
    at::Tensor tx_power,
    int64_t tx_power_index);
std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
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
    at::Tensor tx_power,
    int64_t tx_power_index);
std::vector<at::Tensor> cn_bdpt_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol);
std::vector<at::Tensor> cn_bdpt_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_face_material_tensors_from_host_cuda(
    const std::vector<float>& material_eps_r,
    const std::vector<float>& material_sigma_e,
    const std::vector<float>& material_mu_r,
    const std::vector<int>& face_material_id);

pybind11::dict cn_bdpt_launch_state(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed) {
    auto [tx_id, sample_id, stream_id, light_seed] =
        cn_bdpt_launch_state_cuda(reference, tx_count, samples, sample_streams, seed);
    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["sample_id"] = sample_id;
    out["stream_id"] = stream_id;
    out["light_seed"] = light_seed;
    return out;
}

pybind11::dict cn_bdpt_empty_subpath_state(torch::Tensor reference) {
    return subpath_state_to_dict(cn_bdpt_empty_subpath_state_cuda(reference), "bdpt_empty_subpath_state");
}

pybind11::dict cn_bdpt_endpoint_subpath_state(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    torch::Tensor launch_tx_id,
    torch::Tensor light_seed) {
    pybind11::dict out;
    out["light"] = subpath_state_to_dict(
        cn_bdpt_light_endpoint_subpath_state_cuda(
            tx_positions, tx_power, tx_polarization, launch_tx_id, light_seed),
        "bdpt_light_endpoint_subpath_state");
    out["sensor"] = subpath_state_to_dict(
        cn_bdpt_sensor_endpoint_subpath_state_cuda(rx_positions, rx_polarization),
        "bdpt_sensor_endpoint_subpath_state");
    return out;
}

pybind11::dict cn_bdpt_subpath_intersection_inputs(pybind11::dict subpath) {
    at::Tensor ray_o = tensor_from_dict(subpath, "origin");
    at::Tensor ray_d = tensor_from_dict(subpath, "direction");
    at::Tensor active = tensor_from_dict(subpath, "valid");
    TORCH_CHECK(ray_o.is_cuda(), "subpath.origin must be a CUDA tensor");
    TORCH_CHECK(ray_d.is_cuda(), "subpath.direction must be a CUDA tensor");
    TORCH_CHECK(active.is_cuda(), "subpath.valid must be a CUDA tensor");
    TORCH_CHECK(ray_o.scalar_type() == at::kFloat, "subpath.origin must be float32");
    TORCH_CHECK(ray_d.scalar_type() == at::kFloat, "subpath.direction must be float32");
    TORCH_CHECK(active.scalar_type() == at::kBool, "subpath.valid must be bool");
    TORCH_CHECK(ray_o.dim() == 2 && ray_o.size(1) == 3, "subpath.origin must have shape (N, 3)");
    TORCH_CHECK(ray_d.sizes() == ray_o.sizes(), "subpath.direction must match origin");
    TORCH_CHECK(active.dim() == 1 && active.size(0) == ray_o.size(0), "subpath.valid must match origin");
    TORCH_CHECK(ray_d.get_device() == ray_o.get_device(), "subpath.direction must share origin device");
    TORCH_CHECK(active.get_device() == ray_o.get_device(), "subpath.valid must share origin device");
    pybind11::dict out;
    out["ray_o"] = ray_o;
    out["ray_d"] = ray_d;
    out["ray_tmax"] = at::empty({0}, ray_o.options());
    out["active"] = active;
    return out;
}

pybind11::dict cn_bdpt_reflected_light_subpath_state(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor material_thickness,
    double frequency_hz) {
    return subpath_state_to_dict(
        cn_bdpt_reflected_light_subpath_state_cuda(
            tensor_from_dict(light, "origin"),
            tensor_from_dict(light, "direction"),
            tensor_from_dict(light, "throughput_real"),
            tensor_from_dict(light, "throughput_imag"),
            tensor_from_dict(light, "pdf_forward"),
            tensor_from_dict(light, "pdf_reverse"),
            tensor_from_dict(light, "depth"),
            tensor_from_dict(light, "component_mask"),
            tensor_from_dict(light, "tx_id"),
            tensor_from_dict(light, "rx_id"),
            tensor_from_dict(light, "grid_linear_id"),
            tensor_from_dict(light, "valid"),
            tensor_from_dict(light, "path_length"),
            tensor_from_dict(light, "field_real"),
            tensor_from_dict(light, "field_imag"),
            tensor_from_dict(light, "source_power"),
            tensor_from_dict(intersection, "t"),
            tensor_from_dict(intersection, "p"),
            tensor_from_dict(intersection, "n"),
            tensor_from_dict(intersection, "global_prim_id"),
            material_gain,
            material_valid,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
            frequency_hz),
        "bdpt_reflected_light_subpath_state");
}

pybind11::dict cn_bdpt_transmitted_light_subpath_state(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor face_material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz) {
    return subpath_state_to_dict(
        cn_bdpt_transmitted_light_subpath_state_cuda(
            tensor_from_dict(light, "origin"),
            tensor_from_dict(light, "direction"),
            tensor_from_dict(light, "throughput_real"),
            tensor_from_dict(light, "throughput_imag"),
            tensor_from_dict(light, "pdf_forward"),
            tensor_from_dict(light, "pdf_reverse"),
            tensor_from_dict(light, "depth"),
            tensor_from_dict(light, "component_mask"),
            tensor_from_dict(light, "tx_id"),
            tensor_from_dict(light, "rx_id"),
            tensor_from_dict(light, "grid_linear_id"),
            tensor_from_dict(light, "valid"),
            tensor_from_dict(light, "path_length"),
            tensor_from_dict(light, "field_real"),
            tensor_from_dict(light, "field_imag"),
            tensor_from_dict(light, "source_power"),
            tensor_from_dict(intersection, "t"),
            tensor_from_dict(intersection, "p"),
            tensor_from_dict(intersection, "n"),
            tensor_from_dict(intersection, "global_prim_id"),
            face_material_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_hz),
        "bdpt_transmitted_light_subpath_state");
}

torch::Tensor cn_bdpt_mis_weights(
    torch::Tensor pdf,
    torch::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta) {
    return cn_bdpt_mis_weights_cuda(pdf, strategy_pdf_sum, mode_id, beta);
}

pybind11::dict cn_bdpt_endpoint_connection_samples(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths) {
    return connection_samples_to_dict(cn_bdpt_endpoint_connection_samples_cuda(
        tensor_from_dict(light, "origin"),
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(light, "source_power"),
        tensor_from_dict(light, "pdf_forward"),
        tensor_from_dict(light, "depth"),
        tensor_from_dict(light, "component_mask"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "path_length"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "field_real"),
        tensor_from_dict(sensor, "pdf_reverse"),
        tensor_from_dict(sensor, "depth"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "grid_linear_id"),
        tensor_from_dict(sensor, "valid"),
        frequency_hz,
        samples_per_tx,
        mode_id,
        beta,
        strategy_count,
        max_paths));
}

pybind11::dict cn_bdpt_accumulate_connection_samples(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy) {
    auto [path_gain, los, reflection, diffraction, transmission, scattering] = cn_bdpt_accumulate_connection_samples_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        tx_count,
        rx_count,
        accumulation_strategy);
    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los"] = los;
    out["reflection"] = reflection;
    out["diffraction"] = diffraction;
    out["transmission"] = transmission;
    out["scattering"] = scattering;
    return out;
}

pybind11::dict cn_bdpt_filter_connection_samples(pybind11::dict samples, torch::Tensor visible) {
    cn_bdpt_filter_connection_samples_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "pdf"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "valid"),
        visible);
    return samples;
}

int64_t cn_bdpt_count_valid_connection_samples(pybind11::dict samples) {
    return cn_bdpt_count_valid_connection_samples_cuda(tensor_from_dict(samples, "valid"));
}

pybind11::dict cn_bdpt_compact_connection_samples(pybind11::dict samples, int64_t max_paths) {
    return connection_samples_to_dict(cn_bdpt_compact_connection_samples_cuda(
        tensor_from_dict(samples, "topology"),
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "pdf"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "grid_linear_id"),
        tensor_from_dict(samples, "light_depth"),
        tensor_from_dict(samples, "sensor_depth"),
        tensor_from_dict(samples, "path_length_m"),
        max_paths));
}

pybind11::dict cn_bdpt_concat_connection_samples(pybind11::sequence samples) {
    TORCH_CHECK(pybind11::len(samples) > 0, "bdpt_concat_connection_samples requires at least one sample block");
    const auto count = static_cast<size_t>(pybind11::len(samples));
    std::vector<at::Tensor> topologies;
    std::vector<at::Tensor> contributions;
    std::vector<at::Tensor> pdfs;
    std::vector<at::Tensor> mis_weights;
    std::vector<at::Tensor> component_ids;
    std::vector<at::Tensor> valids;
    std::vector<at::Tensor> tx_ids;
    std::vector<at::Tensor> rx_ids;
    std::vector<at::Tensor> grid_linear_ids;
    std::vector<at::Tensor> light_depths;
    std::vector<at::Tensor> sensor_depths;
    std::vector<at::Tensor> path_lengths_m;
    topologies.reserve(count);
    contributions.reserve(count);
    pdfs.reserve(count);
    mis_weights.reserve(count);
    component_ids.reserve(count);
    valids.reserve(count);
    tx_ids.reserve(count);
    rx_ids.reserve(count);
    grid_linear_ids.reserve(count);
    light_depths.reserve(count);
    sensor_depths.reserve(count);
    path_lengths_m.reserve(count);
    for (auto item : samples) {
        auto block = pybind11::cast<pybind11::dict>(item);
        topologies.push_back(tensor_from_dict(block, "topology"));
        contributions.push_back(tensor_from_dict(block, "contribution"));
        pdfs.push_back(tensor_from_dict(block, "pdf"));
        mis_weights.push_back(tensor_from_dict(block, "mis_weight"));
        component_ids.push_back(tensor_from_dict(block, "component_id"));
        valids.push_back(tensor_from_dict(block, "valid"));
        tx_ids.push_back(tensor_from_dict(block, "tx_id"));
        rx_ids.push_back(tensor_from_dict(block, "rx_id"));
        grid_linear_ids.push_back(tensor_from_dict(block, "grid_linear_id"));
        light_depths.push_back(tensor_from_dict(block, "light_depth"));
        sensor_depths.push_back(tensor_from_dict(block, "sensor_depth"));
        path_lengths_m.push_back(tensor_from_dict(block, "path_length_m"));
    }
    return connection_samples_to_dict(cn_bdpt_concat_connection_samples_cuda(
        std::move(topologies),
        std::move(contributions),
        std::move(pdfs),
        std::move(mis_weights),
        std::move(component_ids),
        std::move(valids),
        std::move(tx_ids),
        std::move(rx_ids),
        std::move(grid_linear_ids),
        std::move(light_depths),
        std::move(sensor_depths),
        std::move(path_lengths_m)));
}

torch::Tensor cn_bdpt_connection_variance(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx) {
    return cn_bdpt_connection_variance_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "valid"),
        tx_count,
        rx_count,
        samples_per_tx);
}

torch::Tensor cn_bdpt_sample_directions(int64_t count, torch::Tensor reference, int64_t seed) {
    return cn_bdpt_sample_directions_cuda(count, reference, seed);
}

pybind11::dict cn_bdpt_endpoint_connection_visibility_inputs(
    pybind11::dict light,
    pybind11::dict sensor,
    int64_t sample_count) {
    auto [start, end, active] = cn_bdpt_endpoint_connection_visibility_inputs_cuda(
        tensor_from_dict(light, "origin"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "valid"),
        sample_count);
    pybind11::dict out;
    out["start"] = start;
    out["end"] = end;
    out["active"] = active;
    return out;
}

pybind11::dict cn_bdpt_diffraction_connection_samples_from_tape(
    pybind11::dict tape,
    pybind11::tuple states,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    TORCH_CHECK(pybind11::len(states) == 12, "states must contain 12 tensors");
    return connection_samples_to_dict(cn_bdpt_diffraction_connection_samples_from_tape_cuda(
        tensor_from_dict(tape, "active"),
        tensor_from_dict(tape, "state_idx"),
        tensor_from_dict(tape, "cell"),
        tensor_from_dict(tape, "material_idx"),
        tensor_from_dict(tape, "edge_u"),
        pybind11::cast<torch::Tensor>(states[0]),
        pybind11::cast<torch::Tensor>(states[1]),
        pybind11::cast<torch::Tensor>(states[2]),
        pybind11::cast<torch::Tensor>(states[3]),
        pybind11::cast<torch::Tensor>(states[4]),
        pybind11::cast<torch::Tensor>(states[9]),
        pybind11::cast<torch::Tensor>(states[10]),
        pybind11::cast<torch::Tensor>(states[11]),
        material_gain,
        material_valid,
        tx_index,
        state_count,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        grid_cell_area,
        wavelength,
        direct_samples,
        keller_samples,
        mode_id,
        beta,
        strategy_count));
}

pybind11::dict cn_bdpt_diffraction_point_connection_samples(
    torch::Tensor rx_positions,
    pybind11::tuple states,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t seed,
    double wavelength,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    TORCH_CHECK(pybind11::len(states) == 12, "states must contain 12 tensors");
    auto out = cn_bdpt_diffraction_point_connection_samples_cuda(
        rx_positions,
        pybind11::cast<torch::Tensor>(states[0]),
        pybind11::cast<torch::Tensor>(states[1]),
        pybind11::cast<torch::Tensor>(states[2]),
        pybind11::cast<torch::Tensor>(states[3]),
        pybind11::cast<torch::Tensor>(states[4]),
        pybind11::cast<torch::Tensor>(states[7]),
        pybind11::cast<torch::Tensor>(states[8]),
        pybind11::cast<torch::Tensor>(states[9]),
        pybind11::cast<torch::Tensor>(states[10]),
        pybind11::cast<torch::Tensor>(states[11]),
        material_gain,
        material_valid,
        tx_index,
        state_count,
        direct_samples,
        keller_samples,
        seed,
        wavelength,
        mode_id,
        beta,
        strategy_count);
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        light_depth,
        sensor_depth,
        path_length_m,
        source_start,
        source_end,
        target_start,
        target_end,
        visibility_active] = out;
    pybind11::dict result;
    result["samples"] = connection_samples_to_dict(std::make_tuple(
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        light_depth,
        sensor_depth,
        path_length_m));
    result["source_start"] = source_start;
    result["source_end"] = source_end;
    result["target_start"] = target_start;
    result["target_end"] = target_end;
    result["visibility_active"] = visibility_active;
    return result;
}

torch::Tensor cn_bdpt_zero_matrix(torch::Tensor reference, int64_t rows, int64_t cols) {
    return cn_bdpt_zero_matrix_cuda(reference, rows, cols);
}

torch::Tensor cn_core_pack_int2(torch::Tensor x, torch::Tensor y) {
    return cn_core_pack_int2_cuda(x, y);
}

int64_t cn_core_diffraction_edge_count(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    bool vertical_only,
    double vertical_ratio,
    bool boundary_half_plane,
    double plane_tol) {
    return cn_core_diffraction_edge_count_cuda(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        vertical_only,
        vertical_ratio,
        boundary_half_plane,
        plane_tol);
}

pybind11::dict cn_bdpt_point_component_power(torch::Tensor path_gain, bool include_los) {
    auto [los, reflection, diffraction] = cn_bdpt_point_component_power_cuda(path_gain, include_los);
    pybind11::dict out;
    out["los"] = los;
    out["reflection"] = reflection;
    out["diffraction"] = diffraction;
    return out;
}

torch::Tensor cn_bdpt_store_point_component_column(
    torch::Tensor target,
    torch::Tensor source,
    int64_t rx_index) {
    return cn_bdpt_store_point_component_column_cuda(target, source, rx_index);
}

pybind11::dict cn_bdpt_finalize_point_components(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power, transmission_power,
        scattering_power] =
        cn_bdpt_finalize_point_components_cuda(
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

pybind11::dict cn_bdpt_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz,
    torch::Tensor tx_pol) {
    auto [tx_id, rx_id, path_length, delay, path_gain, path_gain_matrix] =
        cn_bdpt_los_export_cuda(tx_positions, tx_power, rx_positions, frequency_hz, tx_pol);

    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["path_gain"] = path_gain;
    out["path_gain_matrix"] = path_gain_matrix;
    return out;
}

pybind11::dict cn_bdpt_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power, transmission_power,
        scattering_power] =
        cn_bdpt_finalize_component_maps_cuda(
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

torch::Tensor cn_bdpt_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    return cn_bdpt_component_map_buffer_cuda(reference, tx_count, dim0, dim1);
}

torch::Tensor cn_bdpt_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index) {
    return cn_bdpt_store_component_map_cuda(maps, source, tx_index);
}

torch::Tensor cn_bdpt_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index) {
    return cn_bdpt_store_scaled_component_map_cuda(maps, source, scale_values, tx_index, scale_index);
}

pybind11::dict cn_bdpt_transmitter_tensors(
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
    auto [positions, power] = cn_bdpt_transmitter_tensors_cuda(positions_host, power_host);
    pybind11::dict out;
    out["positions"] = positions;
    out["power"] = power;
    return out;
}

torch::Tensor cn_bdpt_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z) {
    return cn_bdpt_pack_vec3_cuda(x, y, z);
}

torch::Tensor cn_bdpt_los_component_maps(torch::Tensor los) {
    return cn_bdpt_los_component_maps_cuda(los);
}

torch::Tensor cn_bdpt_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols) {
    return cn_bdpt_los_component_maps_from_matrix_cuda(los, rows, cols);
}

torch::Tensor cn_bdpt_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index) {
    return cn_bdpt_apply_los_visibility_cuda(maps, los, visible, tx_index);
}

pybind11::dict cn_bdpt_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    auto [start, active] = cn_bdpt_los_visibility_inputs_cuda(tx_positions, tx_index, rx_count);
    pybind11::dict out;
    out["start"] = start;
    out["active"] = active;
    return out;
}

torch::Tensor cn_bdpt_receiver_grid_points(
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
    return cn_bdpt_receiver_grid_points_cuda(
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

pybind11::dict cn_bdpt_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    auto [ray_o, ray_tmax, active, tx_pol, tx_id, light_seed] =
        cn_bdpt_reflection_launch_inputs_cuda(tx_positions, tx_index, sample_count);
    pybind11::dict out;
    out["ray_o"] = ray_o;
    out["ray_tmax"] = ray_tmax;
    out["active"] = active;
    out["tx_pol"] = tx_pol;
    out["tx_id"] = tx_id;
    out["light_seed"] = light_seed;
    return out;
}

torch::Tensor cn_bdpt_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src) {
    return cn_bdpt_diffraction_state_wi_cuda(state_edge_pos, state_src);
}

torch::Tensor cn_bdpt_selected_edge_indices(torch::Tensor selected) {
    return cn_bdpt_selected_edge_indices_cuda(selected);
}

pybind11::tuple cn_bdpt_diffraction_state_pack(
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
    return tensor_vector_to_tuple(cn_bdpt_diffraction_state_pack_cuda(
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
        tx_power));
}

pybind11::tuple cn_deterministic_diffraction_state_pack(
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
    torch::Tensor tx_power,
    int64_t tx_power_index) {
    return tensor_vector_to_tuple(cn_deterministic_diffraction_state_pack_cuda(
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
        tx_power,
        tx_power_index));
}

pybind11::tuple cn_deterministic_diffraction_state_pack_selected(
    torch::Tensor selected,
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
    torch::Tensor tx_power,
    int64_t tx_power_index) {
    return tensor_vector_to_tuple(cn_deterministic_diffraction_state_pack_selected_cuda(
        selected,
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
        tx_power,
        tx_power_index));
}

pybind11::tuple cn_bdpt_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol) {
    return tensor_vector_to_tuple(cn_bdpt_diffraction_edge_geometry_cuda(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        plane_tol));
}

pybind11::tuple cn_bdpt_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol) {
    return tensor_vector_to_tuple(cn_bdpt_surface_group_edge_candidates_cuda(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        plane_tol));
}

pybind11::dict cn_bdpt_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id) {
    auto [face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid] =
        cn_bdpt_face_material_tensors_cuda(
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

pybind11::dict cn_bdpt_face_material_tensors_from_host(
    pybind11::sequence material_eps_r,
    pybind11::sequence material_sigma_e,
    pybind11::sequence material_mu_r,
    pybind11::sequence face_material_id) {
    TORCH_CHECK(material_sigma_e.size() == material_eps_r.size(), "material_sigma_e must match material_eps_r");
    TORCH_CHECK(material_mu_r.size() == material_eps_r.size(), "material_mu_r must match material_eps_r");
    std::vector<float> eps_r;
    std::vector<float> sigma_e;
    std::vector<float> mu_r;
    std::vector<int> face_ids;
    eps_r.reserve(material_eps_r.size());
    sigma_e.reserve(material_sigma_e.size());
    mu_r.reserve(material_mu_r.size());
    face_ids.reserve(face_material_id.size());
    for (auto item : material_eps_r) {
        eps_r.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    for (auto item : material_sigma_e) {
        sigma_e.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    for (auto item : material_mu_r) {
        mu_r.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    for (auto item : face_material_id) {
        face_ids.push_back(static_cast<int>(pybind11::cast<int64_t>(item)));
    }

    auto [face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid] =
        cn_bdpt_face_material_tensors_from_host_cuda(eps_r, sigma_e, mu_r, face_ids);

    pybind11::dict out;
    out["eps_r"] = face_eps_r;
    out["sigma_e"] = face_sigma_e;
    out["mu_r"] = face_mu_r;
    out["gain"] = face_gain;
    out["valid"] = face_valid;
    return out;
}
