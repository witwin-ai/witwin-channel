#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

pybind11::dict cn_build_info();
pybind11::dict cn_bdpt_launch_state(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed);
pybind11::dict cn_bdpt_empty_subpath_state(torch::Tensor reference);
pybind11::dict cn_bdpt_endpoint_subpath_state(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    torch::Tensor launch_tx_id,
    torch::Tensor light_seed);
pybind11::dict cn_bdpt_subpath_intersection_inputs(pybind11::dict subpath);
pybind11::dict cn_bdpt_reflected_light_subpath_state(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor material_thickness,
    double frequency_hz);
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
    double frequency_hz);
pybind11::dict cn_em_layer_stack_eval(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz);
pybind11::dict cn_bdpt_endpoint_connection_samples(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths);
pybind11::dict cn_bdpt_endpoint_connection_visibility_inputs(
    pybind11::dict light,
    pybind11::dict sensor,
    int64_t sample_count);
pybind11::dict cn_bdpt_accumulate_connection_samples(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy);
pybind11::dict cn_bdpt_filter_connection_samples(pybind11::dict samples, torch::Tensor visible);
int64_t cn_bdpt_count_valid_connection_samples(pybind11::dict samples);
pybind11::dict cn_bdpt_compact_connection_samples(pybind11::dict samples, int64_t max_paths);
pybind11::dict cn_bdpt_concat_connection_samples(pybind11::sequence samples);
torch::Tensor cn_bdpt_connection_variance(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx);
torch::Tensor cn_bdpt_sample_directions(int64_t count, torch::Tensor reference, int64_t seed);
torch::Tensor cn_bdpt_mis_weights(
    torch::Tensor pdf,
    torch::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta);
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
    int64_t strategy_count);
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
    int64_t strategy_count);
torch::Tensor cn_bdpt_zero_matrix(torch::Tensor reference, int64_t rows, int64_t cols);
torch::Tensor cn_core_pack_int2(torch::Tensor x, torch::Tensor y);
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
    double plane_tol);
pybind11::dict cn_bdpt_point_component_power(torch::Tensor path_gain, bool include_los);
torch::Tensor cn_bdpt_store_point_component_column(
    torch::Tensor target,
    torch::Tensor source,
    int64_t rx_index);
pybind11::dict cn_bdpt_finalize_point_components(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
pybind11::dict cn_bdpt_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz);
pybind11::dict cn_bdpt_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
torch::Tensor cn_bdpt_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
torch::Tensor cn_bdpt_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index);
torch::Tensor cn_bdpt_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
pybind11::dict cn_bdpt_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers);
torch::Tensor cn_bdpt_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z);
torch::Tensor cn_bdpt_los_component_maps(torch::Tensor los);
torch::Tensor cn_bdpt_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols);
torch::Tensor cn_bdpt_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index);
pybind11::dict cn_bdpt_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
pybind11::tuple cn_raydn_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_raydn_scene_edge_records(
    int64_t scene_handle,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_bdpt_intersect_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_bdpt_visibility_forward(
    int64_t scene_handle,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_raydn_trace_reflections_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_raydn_reflection_epc_paths_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    pybind11::object active,
    torch::Tensor expected_prim_ids,
    torch::Tensor direct_plane_points,
    torch::Tensor direct_plane_normals,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    std::uintptr_t raydn_module_handle);
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
    bool reverse,
    std::uintptr_t raydn_module_handle);
pybind11::dict cn_field_free_space(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    double frequency_hz);
pybind11::dict cn_field_project_complex3(
    torch::Tensor field_vector,
    torch::Tensor direction,
    torch::Tensor rx_polarization);
pybind11::dict cn_field_reflection_sequence(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    torch::Tensor thickness,
    double frequency_hz);
pybind11::dict cn_field_transmission_sequence(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor interaction_material_id,
    torch::Tensor interaction_valid,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz);
pybind11::dict cn_field_coupled_rd(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor reflection_position,
    torch::Tensor reflection_normal,
    torch::Tensor edge_position,
    torch::Tensor edge_direction,
    torch::Tensor edge_n0,
    torch::Tensor edge_n1,
    torch::Tensor exterior_angle,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor reflection_eps_r,
    torch::Tensor reflection_sigma_e,
    torch::Tensor reflection_mu_r,
    torch::Tensor reflection_gain,
    torch::Tensor reflection_thickness,
    torch::Tensor wedge_eps_r0,
    torch::Tensor wedge_sigma_e0,
    torch::Tensor wedge_mu_r0,
    torch::Tensor wedge_gain0,
    torch::Tensor wedge_thickness0,
    torch::Tensor wedge_eps_r1,
    torch::Tensor wedge_sigma_e1,
    torch::Tensor wedge_mu_r1,
    torch::Tensor wedge_gain1,
    torch::Tensor wedge_thickness1,
    double frequency_hz,
    bool reverse);
pybind11::tuple cn_bdpt_reflection_accumulation_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    torch::Tensor active,
    torch::Tensor tx,
    torch::Tensor tx_pol,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool streaming_los_enabled,
    std::uintptr_t raydn_module_handle);
torch::Tensor cn_bdpt_diffraction_discover_edges(
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
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle);
torch::Tensor cn_bdpt_diffraction_discover_edges_counted(
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
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_raydn_diffraction_paths_order1_forward(
    int64_t scene_handle,
    torch::Tensor tx_pos,
    torch::Tensor rx_pos,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    std::uintptr_t raydn_module_handle);
pybind11::dict cn_path_diffraction_paths_order1(
    int64_t scene_handle,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
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
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    double wavelength,
    std::uintptr_t raydn_module_handle);
pybind11::tuple cn_bdpt_diffraction_accumulation_forward(
    int64_t scene_handle,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    pybind11::object state_wi,
    pybind11::object state_d0,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
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
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    int64_t recursive_state_limit,
    pybind11::object recursive_active,
    pybind11::object recursive_state_edge_index,
    pybind11::object recursive_state_edge_pos,
    pybind11::object recursive_state_edge_dir,
    pybind11::object recursive_state_edge_t_min,
    pybind11::object recursive_state_edge_t_max,
    pybind11::object recursive_state_n0,
    pybind11::object recursive_state_n1,
    pybind11::object recursive_state_prim0,
    pybind11::object recursive_state_prim1,
    pybind11::object recursive_state_exterior_angle,
    int64_t export_tape,
    pybind11::object sample_state_index,
    pybind11::object sample_edge_weight,
    std::uintptr_t raydn_module_handle);
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
    double spacing1);
pybind11::dict cn_bdpt_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
torch::Tensor cn_bdpt_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src);
torch::Tensor cn_bdpt_selected_edge_indices(torch::Tensor selected);
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
    torch::Tensor tx_power);
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
    int64_t tx_power_index);
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
    int64_t tx_power_index);
pybind11::tuple cn_bdpt_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol);
pybind11::tuple cn_bdpt_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol);
pybind11::dict cn_bdpt_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id);
pybind11::dict cn_bdpt_face_material_tensors_from_host(
    pybind11::sequence material_eps_r,
    pybind11::sequence material_sigma_e,
    pybind11::sequence material_mu_r,
    pybind11::sequence face_material_id);
pybind11::dict cn_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz);
torch::Tensor cn_path_concat_vec3(pybind11::sequence blocks);
pybind11::dict cn_path_los_visibility_inputs(
    torch::Tensor tx_positions,
    torch::Tensor rx_positions,
    torch::Tensor tx_id,
    torch::Tensor rx_id);
pybind11::dict cn_path_filter_los(
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor path_length,
    torch::Tensor delay,
    torch::Tensor path_gain,
    torch::Tensor visible);
pybind11::dict cn_path_reflection_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor face_gain,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz);
pybind11::dict cn_path_filter_block(
    pybind11::dict block,
    torch::Tensor visible0,
    torch::Tensor visible1);
pybind11::dict cn_path_diffraction_block(
    pybind11::sequence raydn_output,
    int64_t tx_index);
pybind11::dict cn_path_merge_blocks(
    pybind11::sequence blocks,
    int64_t tx_count,
    int64_t max_depth);
pybind11::dict cn_path_finalize_blocks(
    pybind11::dict los,
    pybind11::dict reflection,
    pybind11::dict diffraction,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth);
pybind11::dict cn_mc_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
torch::Tensor cn_mc_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
torch::Tensor cn_mc_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index);
torch::Tensor cn_mc_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
torch::Tensor cn_mc_sample_directions(int64_t count, torch::Tensor reference);
pybind11::dict cn_mc_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers);
torch::Tensor cn_mc_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z);
torch::Tensor cn_mc_los_component_maps(torch::Tensor los);
torch::Tensor cn_mc_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols);
pybind11::tuple cn_mc_los_path_gain_backward(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor grad_output,
    double frequency_hz);
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
    double frequency_hz);
torch::Tensor cn_mc_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index);
pybind11::dict cn_mc_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
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
    double spacing1);
pybind11::dict cn_mc_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
torch::Tensor cn_mc_sionna_reflection_accumulate(
    torch::Tensor ray_o, torch::Tensor ray_d, torch::Tensor trace_valid,
    torch::Tensor trace_t, torch::Tensor trace_prim, torch::Tensor face_normals,
    torch::Tensor eta_r, torch::Tensor sigma, torch::Tensor gain,
    torch::Tensor material_valid, torch::Tensor thickness,
    int64_t contribution_depth, int64_t axis, double plane_position,
    double coord0_min, double coord0_max, double coord1_min, double coord1_max,
    int64_t resolution0, int64_t resolution1, double wavelength,
    double solid_angle_per_ray, double cell_area);
torch::Tensor cn_mc_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src);
torch::Tensor cn_mc_sionna_diffraction_tape_accumulate(
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,double,double,double,double,double,
    int64_t,int64_t,double,double,int64_t,double);
torch::Tensor cn_mc_selected_edge_indices(torch::Tensor selected);
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
    torch::Tensor tx_power);
pybind11::tuple cn_mc_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol);
pybind11::tuple cn_mc_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol);
pybind11::dict cn_mc_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id);
pybind11::dict cn_deterministic_los_field(
    torch::Tensor path_gain,
    torch::Tensor path_length_m,
    double frequency_hz);
pybind11::dict cn_deterministic_diffraction_vector_field(
    torch::Tensor x_re,
    torch::Tensor x_im,
    torch::Tensor y_re,
    torch::Tensor y_im,
    torch::Tensor z_re,
    torch::Tensor z_im);
pybind11::dict cn_deterministic_reflection_field(
    torch::Tensor tx_position,
    torch::Tensor rx_position,
    torch::Tensor hit_position,
    torch::Tensor normal,
    torch::Tensor tx_power,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    double frequency_hz);
pybind11::dict cn_deterministic_reflection_sequence_field(
    torch::Tensor tx_position,
    torch::Tensor rx_position,
    torch::Tensor hit_positions,
    torch::Tensor normals,
    torch::Tensor tx_power,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    double frequency_hz);
torch::Tensor cn_deterministic_delay_to_path_length(torch::Tensor delay_s);
torch::Tensor cn_deterministic_pack_complex(torch::Tensor field_real, torch::Tensor field_imag);
torch::Tensor cn_deterministic_phase_from_field(torch::Tensor field_real, torch::Tensor field_imag);
pybind11::dict cn_deterministic_zero_field_phase(torch::Tensor reference);
torch::Tensor cn_deterministic_phase_from_length(torch::Tensor path_length_m, double frequency_hz);
pybind11::dict cn_deterministic_field_from_power_phase(torch::Tensor path_gain, torch::Tensor phase_rad);
pybind11::dict cn_deterministic_los_topology_block(
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor path_gain,
    torch::Tensor visible,
    double frequency_hz,
    int64_t sequence_width);
pybind11::dict cn_deterministic_los_topology_block_all_visible(
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor path_gain,
    double frequency_hz,
    int64_t sequence_width);
pybind11::dict cn_deterministic_topology_default_fields(torch::Tensor reference);
pybind11::dict cn_deterministic_pad_topology_sequences(
    torch::Tensor depth,
    torch::Tensor primitive_id,
    torch::Tensor material_id,
    torch::Tensor interaction_position,
    torch::Tensor interaction_normal,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    int64_t width);
pybind11::dict cn_deterministic_topology_base_fields(
    torch::Tensor rx_id,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor path_gain,
    int64_t tx_index,
    int64_t component_id,
    torch::Tensor depth_source,
    int64_t depth_value,
    torch::Tensor primitive_source,
    int64_t primitive_value,
    torch::Tensor edge_source,
    int64_t edge_value);
torch::Tensor cn_deterministic_repeat_range(torch::Tensor reference, int64_t start, int64_t end, int64_t repeats);
torch::Tensor cn_deterministic_face_anchor_points(torch::Tensor vertices, torch::Tensor faces);
pybind11::dict cn_deterministic_reflection_epc_input_batch(
    torch::Tensor tx,
    torch::Tensor rx_positions,
    torch::Tensor sequences,
    torch::Tensor tri_a,
    torch::Tensor normals,
    int64_t rx_start,
    int64_t rx_end);
torch::Tensor cn_deterministic_face_sequence_chunk(
    torch::Tensor reference,
    int64_t face_count,
    int64_t depth,
    int64_t start,
    int64_t end,
    bool adjacent_distinct);
torch::Tensor cn_deterministic_mapped_face_sequence_chunk(
    torch::Tensor face_ids,
    int64_t depth,
    int64_t start,
    int64_t end,
    bool adjacent_distinct);
pybind11::dict cn_deterministic_reflection_order1_compact(
    torch::Tensor visible,
    torch::Tensor epc_faces,
    torch::Tensor epc_hits,
    torch::Tensor epc_normals,
    torch::Tensor sequence_batch,
    torch::Tensor rx_indices,
    torch::Tensor tx,
    torch::Tensor rx_positions,
    torch::Tensor tx_power,
    int64_t tx_index,
    torch::Tensor face_eps_r,
    torch::Tensor face_sigma_e,
    torch::Tensor face_mu_r,
    torch::Tensor face_gain,
    torch::Tensor face_material_id,
    bool grouped_export);
pybind11::dict cn_deterministic_reflection_sequence_compact(
    torch::Tensor visible,
    torch::Tensor epc_sequences,
    torch::Tensor epc_hits,
    torch::Tensor epc_normals,
    torch::Tensor rx_indices,
    torch::Tensor tx,
    torch::Tensor rx_positions,
    torch::Tensor tx_power,
    int64_t tx_index,
    torch::Tensor face_eps_r,
    torch::Tensor face_sigma_e,
    torch::Tensor face_mu_r,
    torch::Tensor face_gain,
    torch::Tensor face_material_id,
    int64_t max_count);
pybind11::dict cn_deterministic_diffraction_order1_compact(
    torch::Tensor valid,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor edge_id,
    torch::Tensor delay,
    torch::Tensor x_re,
    torch::Tensor x_im,
    torch::Tensor y_re,
    torch::Tensor y_im,
    torch::Tensor z_re,
    torch::Tensor z_im,
    torch::Tensor interaction_position);
torch::Tensor cn_deterministic_normalize_vec3(torch::Tensor values, double eps);
torch::Tensor cn_deterministic_reflect_points(
    torch::Tensor points,
    torch::Tensor plane_points,
    torch::Tensor normals);
pybind11::dict cn_deterministic_concat_topology_blocks(pybind11::sequence blocks, int64_t sequence_width);
pybind11::dict cn_deterministic_gather_topology_block(
    pybind11::dict block,
    torch::Tensor order,
    int64_t max_count,
    int64_t sequence_width);
pybind11::dict cn_deterministic_face_groups(
    torch::Tensor tri_a,
    torch::Tensor normals,
    torch::Tensor surface_ids,
    double quantization);
pybind11::dict cn_deterministic_surface_face_groups(torch::Tensor surface_ids);
torch::Tensor cn_deterministic_sort_order(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor primitive_sequence);
pybind11::dict cn_deterministic_accumulate_flat(
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor component_id,
    torch::Tensor path_gain,
    torch::Tensor field_real,
    torch::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent);
pybind11::dict cn_deterministic_component_counts(torch::Tensor component_id);
int64_t cn_deterministic_selected_edge_count(torch::Tensor edge_id);

PYBIND11_MODULE(_channel_native, module) {
    module.doc() = "Channel Native Torch/CUDA extension.";
    module.def("build_info", &cn_build_info, "Return Channel Native build metadata.");
    module.def(
        "bdpt_launch_state",
        &cn_bdpt_launch_state,
        "Generate deterministic BDPT launch-state tensors.");
    module.def(
        "bdpt_empty_subpath_state",
        &cn_bdpt_empty_subpath_state,
        "Allocate the empty BDPT subpath-state tensor schema with native CUDA storage.");
    module.def(
        "bdpt_endpoint_subpath_state",
        &cn_bdpt_endpoint_subpath_state,
        "Generate BDPT light and sensor endpoint subpaths with native CUDA.");
    module.def(
        "bdpt_subpath_intersection_inputs",
        &cn_bdpt_subpath_intersection_inputs,
        "Expose BDPT subpath state as RayDN intersection ray inputs with native tensor storage.");
    module.def(
        "bdpt_reflected_light_subpath_state",
        &cn_bdpt_reflected_light_subpath_state,
        "Propagate BDPT light subpaths through native RayDN hit geometry with CUDA reflection.");
    module.def(
        "bdpt_transmitted_light_subpath_state",
        &cn_bdpt_transmitted_light_subpath_state,
        "Continue BDPT light subpaths through thin_sheet walls with CUDA layer-stack transmission.");
    module.def(
        "em_layer_stack_eval",
        &cn_em_layer_stack_eval,
        "Evaluate shared em/ layer-stack reflection/transmission coefficients for parity tests.");
    module.def(
        "bdpt_endpoint_connection_samples",
        &cn_bdpt_endpoint_connection_samples,
        "Connect BDPT endpoint subpaths and emit native connection samples.");
    module.def(
        "bdpt_endpoint_connection_visibility_inputs",
        &cn_bdpt_endpoint_connection_visibility_inputs,
        "Build RayDN visibility inputs for BDPT endpoint connection samples.");
    module.def(
        "bdpt_accumulate_connection_samples",
        &cn_bdpt_accumulate_connection_samples,
        "Accumulate BDPT connection samples into component matrices.");
    module.def(
        "bdpt_filter_connection_samples",
        &cn_bdpt_filter_connection_samples,
        "Apply a native visibility mask to BDPT connection samples.");
    module.def(
        "bdpt_count_valid_connection_samples",
        &cn_bdpt_count_valid_connection_samples,
        "Count valid BDPT connection samples with CUDA.");
    module.def(
        "bdpt_compact_connection_samples",
        &cn_bdpt_compact_connection_samples,
        "Compact valid BDPT connection samples for export.");
    module.def(
        "bdpt_concat_connection_samples",
        &cn_bdpt_concat_connection_samples,
        "Concatenate BDPT connection sample blocks with CUDA.");
    module.def(
        "bdpt_connection_variance",
        &cn_bdpt_connection_variance,
        "Compute BDPT connection-sample first/second-moment variance.");
    module.def(
        "bdpt_mis_weights",
        &cn_bdpt_mis_weights,
        "Evaluate BDPT MIS weights with a CUDA kernel.");
    module.def(
        "bdpt_diffraction_connection_samples_from_tape",
        &cn_bdpt_diffraction_connection_samples_from_tape,
        "Replay RayDN diffraction tape into BDPT native connection samples.");
    module.def(
        "bdpt_diffraction_point_connection_samples",
        &cn_bdpt_diffraction_point_connection_samples,
        "Sample point-receiver diffraction paths into BDPT native connection samples and visibility segments.");
    module.def(
        "bdpt_zero_matrix",
        &cn_bdpt_zero_matrix,
        "Allocate and zero a BDPT float matrix with a CUDA kernel.");
    module.def(
        "core_pack_int2",
        &cn_core_pack_int2,
        "Pack two int32 CUDA vectors into an Nx2 tensor.");
    module.def(
        "core_diffraction_edge_count",
        &cn_core_diffraction_edge_count,
        "Count selected diffraction edges with native CUDA policy logic.");
    module.def(
        "bdpt_point_component_power",
        &cn_bdpt_point_component_power,
        "Reduce point-receiver BDPT component powers with CUDA kernels.");
    module.def(
        "bdpt_store_point_component_column",
        &cn_bdpt_store_point_component_column,
        "Store a one-cell BDPT component map into a point-receiver component matrix.");
    module.def(
        "bdpt_finalize_point_components",
        &cn_bdpt_finalize_point_components,
        "Fuse BDPT point-receiver component matrices and reduce component powers.");
    module.def(
        "bdpt_los_export",
        &cn_bdpt_los_export,
        "Export BDPT direct LoS connections from CUDA tensors.");
    module.def(
        "bdpt_finalize_component_maps",
        &cn_bdpt_finalize_component_maps,
        "Fuse BDPT component maps and component power reductions.");
    module.def(
        "bdpt_component_map_buffer",
        &cn_bdpt_component_map_buffer,
        "Allocate a zero-filled BDPT component map buffer.");
    module.def(
        "bdpt_store_component_map",
        &cn_bdpt_store_component_map,
        "Store one BDPT component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "bdpt_store_scaled_component_map",
        &cn_bdpt_store_scaled_component_map,
        "Store one scaled BDPT component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "bdpt_sample_directions",
        &cn_bdpt_sample_directions,
        "Generate seeded BDPT sphere sample directions with a fused CUDA kernel.");
    module.def(
        "bdpt_transmitter_tensors",
        &cn_bdpt_transmitter_tensors,
        "Create BDPT transmitter/host position CUDA tensors through native code.");
    module.def(
        "bdpt_pack_vec3",
        &cn_bdpt_pack_vec3,
        "Pack three CUDA float vectors into an interleaved BDPT vec3 tensor.");
    module.def(
        "bdpt_los_component_maps",
        &cn_bdpt_los_component_maps,
        "Convert BDPT LoS gains to public component-map grid layout with a CUDA kernel.");
    module.def(
        "bdpt_los_component_maps_from_matrix",
        &cn_bdpt_los_component_maps_from_matrix,
        "Convert a BDPT LoS path-gain matrix to public component-map grid layout with a CUDA kernel.");
    module.def(
        "bdpt_apply_los_visibility",
        &cn_bdpt_apply_los_visibility,
        "Apply RayDN LoS visibility to one BDPT transmitter component map with a CUDA kernel.");
    module.def(
        "bdpt_los_visibility_inputs",
        &cn_bdpt_los_visibility_inputs,
        "Prepare RayDN LoS visibility start and active tensors for BDPT with a CUDA kernel.");
    module.def(
        "raydn_scene_create",
        &cn_raydn_scene_create,
        "Create a RayDN native scene through the direct C bridge.");
    module.def(
        "raydn_scene_edge_records",
        &cn_raydn_scene_edge_records,
        "Read RayDN native scene edge records through the direct C bridge.");
    module.def(
        "bdpt_intersect_forward",
        &cn_bdpt_intersect_forward,
        "Call RayDN ray/scene intersection through the native C bridge.");
    module.def(
        "bdpt_visibility_forward",
        &cn_bdpt_visibility_forward,
        "Call RayDN segment visibility through the native C bridge.");
    module.def(
        "raydn_trace_reflections_forward",
        &cn_raydn_trace_reflections_forward,
        "Call RayDN multibounce reflection chain tracing through the native C bridge.");
    module.def(
        "raydn_reflection_epc_paths_forward",
        &cn_raydn_reflection_epc_paths_forward,
        "Call RayDN reflection EPC path export through the native C bridge.");
    module.def(
        "raydn_coupled_rd_geometry_forward",
        &cn_raydn_coupled_rd_geometry_forward,
        "Construct one-reflection/one-diffraction geometry with RayDN EPC and visibility; no field coefficient is evaluated.");
    module.def(
        "field_free_space",
        &cn_field_free_space,
        "Evaluate the canonical complex3 free-space field and receiver projection.");
    module.def(
        "field_project_complex3",
        &cn_field_project_complex3,
        "Project a world-Cartesian complex3 field onto a receiver polarization.");
    module.def(
        "field_reflection_sequence",
        &cn_field_reflection_sequence,
        "Transport a canonical complex3 field through a finite-slab reflection sequence.");
    module.def(
        "field_transmission_sequence",
        &cn_field_transmission_sequence,
        "Transport a canonical complex3 field through a thin_sheet transmission sequence.");
    module.def(
        "field_coupled_rd",
        &cn_field_coupled_rd,
        "Transport a canonical complex3 field through coupled R-D or D-R events.");
    module.def(
        "bdpt_reflection_accumulation_forward",
        &cn_bdpt_reflection_accumulation_forward,
        "Call RayDN reflection accumulation through the native C bridge.");
    module.def(
        "bdpt_diffraction_discover_edges",
        &cn_bdpt_diffraction_discover_edges,
        "Call RayDN diffraction edge discovery through the native C bridge.");
    module.def(
        "bdpt_diffraction_discover_edges_counted",
        &cn_bdpt_diffraction_discover_edges_counted,
        "Call RayDN counted diffraction edge discovery through the native C bridge.");
    module.def(
        "raydn_diffraction_paths_order1_forward",
        &cn_raydn_diffraction_paths_order1_forward,
        "Call RayDN diffraction order-1 path export through the native C bridge.");
    module.def(
        "bdpt_diffraction_accumulation_forward",
        &cn_bdpt_diffraction_accumulation_forward,
        "Call RayDN diffraction accumulation through the native C bridge.");
    module.def(
        "bdpt_receiver_grid_points",
        &cn_bdpt_receiver_grid_points,
        "Generate BDPT receiver grid points with a CUDA kernel.");
    module.def(
        "bdpt_reflection_launch_inputs",
        &cn_bdpt_reflection_launch_inputs,
        "Prepare RayDN reflection launch tensors for BDPT with a CUDA kernel.");
    module.def(
        "bdpt_diffraction_state_wi",
        &cn_bdpt_diffraction_state_wi,
        "Compute BDPT diffraction incident directions with a CUDA kernel.");
    module.def(
        "bdpt_selected_edge_indices",
        &cn_bdpt_selected_edge_indices,
        "Compact a selected BDPT edge mask into deterministic int32 indices with native CUDA kernels.");
    module.def(
        "bdpt_diffraction_state_pack",
        &cn_bdpt_diffraction_state_pack,
        "Pack BDPT diffraction edge states with a fused CUDA kernel.");
    module.def(
        "deterministic_diffraction_state_pack",
        &cn_deterministic_diffraction_state_pack,
        "Pack deterministic diffraction edge states from a transmitter power vector with a fused CUDA kernel.");
    module.def(
        "deterministic_diffraction_state_pack_selected",
        &cn_deterministic_diffraction_state_pack_selected,
        "Pack fixed-capacity selected diffraction edge states without host count compaction.");
    module.def(
        "bdpt_diffraction_edge_geometry",
        &cn_bdpt_diffraction_edge_geometry,
        "Build BDPT diffraction edge geometry tensors with a fused CUDA kernel.");
    module.def(
        "bdpt_surface_group_edge_candidates",
        &cn_bdpt_surface_group_edge_candidates,
        "Build BDPT surface-group diffraction edge candidate tables with native CUDA kernels.");
    module.def(
        "bdpt_face_material_tensors",
        &cn_bdpt_face_material_tensors,
        "Expand BDPT material parameters to per-face tensors with a fused CUDA kernel.");
    module.def(
        "bdpt_face_material_tensors_from_host",
        &cn_bdpt_face_material_tensors_from_host,
        "Create and expand BDPT material parameters from host scalar inputs with native CUDA.");
    module.def(
        "path_los_export",
        &cn_path_los_export,
        "Export empty-space LoS paths from CUDA tensors.");
    module.def(
        "path_concat_vec3",
        &cn_path_concat_vec3,
        "Concatenate path vec3 storage blocks with a CUDA kernel.");
    module.def(
        "path_los_visibility_inputs",
        &cn_path_los_visibility_inputs,
        "Build LoS visibility start/end/active tensors with a CUDA kernel.");
    module.def(
        "path_filter_los",
        &cn_path_filter_los,
        "Compact visible LoS path records with native CUDA kernels.");
    module.def(
        "path_reflection_candidates",
        &cn_path_reflection_candidates,
        "Generate first-order reflection path candidates and visibility segments with native CUDA kernels.");
    module.def(
        "path_filter_block",
        &cn_path_filter_block,
        "Compact path records by two visibility masks with native CUDA kernels.");
    module.def(
        "path_diffraction_block",
        &cn_path_diffraction_block,
        "Convert RayDN order-1 diffraction outputs into a compact path block with native CUDA kernels.");
    module.def(
        "path_diffraction_paths_order1",
        &cn_path_diffraction_paths_order1,
        "Build compact order-1 diffraction path blocks through native CUDA/RayDN without Python tensor loops.");
    module.def(
        "path_merge_blocks",
        &cn_path_merge_blocks,
        "Merge and stable-sort path blocks with native CUDA kernels.");
    module.def(
        "path_finalize_blocks",
        &cn_path_finalize_blocks,
        "Concatenate, stable-sort, cap, and gather path records with native CUDA kernels.");
    module.def(
        "deterministic_component_counts",
        &cn_deterministic_component_counts,
        "Count deterministic path components with a CUDA kernel.");
    module.def(
        "deterministic_selected_edge_count",
        &cn_deterministic_selected_edge_count,
        "Count distinct deterministic edge ids with native CUDA kernels.");
    module.def(
        "mc_finalize_component_maps",
        &cn_mc_finalize_component_maps,
        "Fuse MC component map total and component power reductions.");
    module.def(
        "mc_zero_matrix",
        &cn_bdpt_zero_matrix,
        "Allocate and zero an MC float matrix with a CUDA kernel.");
    module.def(
        "mc_point_component_power",
        &cn_bdpt_point_component_power,
        "Reduce point-receiver MC component powers with CUDA kernels.");
    module.def(
        "mc_component_map_buffer",
        &cn_mc_component_map_buffer,
        "Allocate a zero-filled MC component map buffer.");
    module.def(
        "mc_store_component_map",
        &cn_mc_store_component_map,
        "Store one source component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "mc_store_scaled_component_map",
        &cn_mc_store_scaled_component_map,
        "Store one scaled source component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "mc_sample_directions",
        &cn_mc_sample_directions,
        "Generate MC sphere sample directions with a fused CUDA kernel.");
    module.def(
        "mc_transmitter_tensors",
        &cn_mc_transmitter_tensors,
        "Create transmitter position and power CUDA tensors through native code.");
    module.def(
        "mc_pack_vec3",
        &cn_mc_pack_vec3,
        "Pack three CUDA float vectors into an interleaved vec3 tensor.");
    module.def(
        "mc_los_component_maps",
        &cn_mc_los_component_maps,
        "Convert MC LoS gains to public component-map grid layout with a CUDA kernel.");
    module.def(
        "mc_los_component_maps_from_matrix",
        &cn_mc_los_component_maps_from_matrix,
        "Convert flattened MC LoS matrices to public component-map grid layout with a CUDA kernel.");
    module.def(
        "mc_los_path_gain_backward",
        &cn_mc_los_path_gain_backward,
        "Compute LoS path-gain VJP gradients with native CUDA kernels.");
    module.def(
        "mc_los_path_gain_jvp",
        &cn_mc_los_path_gain_jvp,
        "Compute LoS path-gain JVP with a native CUDA kernel.");
    module.def(
        "mc_apply_los_visibility",
        &cn_mc_apply_los_visibility,
        "Apply RayDN LoS visibility to one transmitter component map with a CUDA kernel.");
    module.def(
        "mc_los_visibility_inputs",
        &cn_mc_los_visibility_inputs,
        "Prepare RayDN LoS visibility start and active tensors with a CUDA kernel.");
    module.def(
        "mc_receiver_grid_points",
        &cn_mc_receiver_grid_points,
        "Generate receiver grid points with a CUDA kernel.");
    module.def(
        "mc_reflection_launch_inputs",
        &cn_mc_reflection_launch_inputs,
        "Prepare RayDN reflection launch tensors with a CUDA kernel.");
    module.def(
        "mc_sionna_reflection_accumulate",
        &cn_mc_sionna_reflection_accumulate,
        "Accumulate finite-thickness Sionna/ITU specular reflections from RayDN traces.");
    module.def(
        "mc_diffraction_state_wi",
        &cn_mc_diffraction_state_wi,
        "Compute MC diffraction incident directions with a CUDA kernel.");
    module.def(
        "mc_sionna_diffraction_tape_accumulate",
        &cn_mc_sionna_diffraction_tape_accumulate,
        "Evaluate full UTD power for valid Keller-cone diffraction samples.");
    module.def(
        "mc_selected_edge_indices",
        &cn_mc_selected_edge_indices,
        "Compact a selected edge mask into deterministic int32 indices with native CUDA kernels.");
    module.def(
        "mc_diffraction_state_pack",
        &cn_mc_diffraction_state_pack,
        "Pack MC diffraction edge states with a fused CUDA kernel.");
    module.def(
        "mc_diffraction_edge_geometry",
        &cn_mc_diffraction_edge_geometry,
        "Build MC diffraction edge geometry tensors with a fused CUDA kernel.");
    module.def(
        "mc_surface_group_edge_candidates",
        &cn_mc_surface_group_edge_candidates,
        "Build MC surface-group diffraction edge candidate tables with native CUDA kernels.");
    module.def(
        "mc_face_material_tensors",
        &cn_mc_face_material_tensors,
        "Expand material parameters to per-face tensors with a fused CUDA kernel.");
    module.def(
        "deterministic_los_field",
        &cn_deterministic_los_field,
        "Evaluate deterministic LoS/free-space scalar complex fields with a CUDA kernel.");
    module.def(
        "deterministic_diffraction_vector_field",
        &cn_deterministic_diffraction_vector_field,
        "Convert RayDN diffraction vector components into scalar deterministic fields with a CUDA kernel.");
    module.def(
        "deterministic_reflection_field",
        &cn_deterministic_reflection_field,
        "Evaluate deterministic scalar reflection complex fields with a CUDA kernel.");
    module.def(
        "deterministic_reflection_sequence_field",
        &cn_deterministic_reflection_sequence_field,
        "Evaluate deterministic multi-bounce scalar reflection complex fields with a CUDA kernel.");
    module.def(
        "deterministic_delay_to_path_length",
        &cn_deterministic_delay_to_path_length,
        "Convert deterministic path delays to path lengths with a CUDA kernel.");
    module.def(
        "deterministic_pack_complex",
        &cn_deterministic_pack_complex,
        "Pack deterministic real/imag field components into complex storage with a CUDA kernel.");
    module.def(
        "deterministic_phase_from_field",
        &cn_deterministic_phase_from_field,
        "Compute deterministic path phase from real/imag field components with a CUDA kernel.");
    module.def(
        "deterministic_zero_field_phase",
        &cn_deterministic_zero_field_phase,
        "Allocate zero deterministic field and phase storage with native CUDA.");
    module.def(
        "deterministic_phase_from_length",
        &cn_deterministic_phase_from_length,
        "Compute deterministic free-space phase from path length with a CUDA kernel.");
    module.def(
        "deterministic_field_from_power_phase",
        &cn_deterministic_field_from_power_phase,
        "Compute deterministic scalar complex field components from path power and phase with a CUDA kernel.");
    module.def(
        "deterministic_los_topology_block",
        &cn_deterministic_los_topology_block,
        "Compact and fill deterministic LoS topology records with a CUDA kernel.");
    module.def(
        "deterministic_los_topology_block_all_visible",
        &cn_deterministic_los_topology_block_all_visible,
        "Fill deterministic LoS topology records without a Python visibility mask.");
    module.def(
        "deterministic_topology_default_fields",
        &cn_deterministic_topology_default_fields,
        "Fill deterministic topology default extension fields with native CUDA.");
    module.def(
        "deterministic_pad_topology_sequences",
        &cn_deterministic_pad_topology_sequences,
        "Pad deterministic topology sequence fields with native CUDA.");
    module.def(
        "deterministic_topology_base_fields",
        &cn_deterministic_topology_base_fields,
        "Fill deterministic topology base path fields with native CUDA.");
    module.def(
        "deterministic_repeat_range",
        &cn_deterministic_repeat_range,
        "Generate repeated deterministic int32 range indices with native CUDA.");
    module.def(
        "deterministic_face_anchor_points",
        &cn_deterministic_face_anchor_points,
        "Gather deterministic face anchor points with native CUDA.");
    module.def(
        "deterministic_reflection_epc_input_batch",
        &cn_deterministic_reflection_epc_input_batch,
        "Build deterministic reflection EPC input batches with native CUDA.");
    module.def(
        "deterministic_face_sequence_chunk",
        &cn_deterministic_face_sequence_chunk,
        "Generate deterministic face sequence chunks with native CUDA.");
    module.def(
        "deterministic_mapped_face_sequence_chunk",
        &cn_deterministic_mapped_face_sequence_chunk,
        "Generate mapped deterministic face sequence chunks with native CUDA.");
    module.def(
        "deterministic_reflection_order1_compact",
        &cn_deterministic_reflection_order1_compact,
        "Compact deterministic first-order reflection EPC outputs with native CUDA.");
    module.def(
        "deterministic_reflection_sequence_compact",
        &cn_deterministic_reflection_sequence_compact,
        "Compact deterministic multi-bounce reflection EPC outputs with native CUDA.");
    module.def(
        "deterministic_diffraction_order1_compact",
        &cn_deterministic_diffraction_order1_compact,
        "Compact deterministic first-order diffraction RayDN outputs with native CUDA.");
    module.def(
        "deterministic_normalize_vec3",
        &cn_deterministic_normalize_vec3,
        "Normalize deterministic vec3 rows with native CUDA.");
    module.def(
        "deterministic_reflect_points",
        &cn_deterministic_reflect_points,
        "Reflect deterministic points about planes with native CUDA.");
    module.def(
        "deterministic_concat_topology_blocks",
        &cn_deterministic_concat_topology_blocks,
        "Concatenate full deterministic topology blocks with native CUDA copies.");
    module.def(
        "deterministic_gather_topology_block",
        &cn_deterministic_gather_topology_block,
        "Gather and truncate deterministic topology rows with native CUDA.");
    module.def(
        "deterministic_face_groups",
        &cn_deterministic_face_groups,
        "Group deterministic faces by canonical native CUDA plane keys.");
    module.def(
        "deterministic_surface_face_groups",
        &cn_deterministic_surface_face_groups,
        "Group deterministic faces by native CUDA surface ids.");
    module.def(
        "deterministic_sort_order",
        &cn_deterministic_sort_order,
        "Stable-sort deterministic topology rows by native CUDA path keys.");
    module.def(
        "deterministic_accumulate_flat",
        &cn_deterministic_accumulate_flat,
        "Accumulate deterministic flat path fields into per-component maps with a CUDA kernel.");
}
