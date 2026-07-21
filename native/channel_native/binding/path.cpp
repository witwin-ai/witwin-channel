#include <torch/extension.h>
#include <pybind11/stl.h>

#include "registry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

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
pybind11::tuple cn_deterministic_diffraction_state_capacity_select(
    torch::Tensor failure_state,
    torch::Tensor active,
    torch::Tensor edge_index,
    torch::Tensor edge_position,
    torch::Tensor edge_direction,
    torch::Tensor edge_t_min,
    torch::Tensor edge_t_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor prim0,
    torch::Tensor prim1,
    torch::Tensor exterior_angle,
    torch::Tensor source,
    torch::Tensor source_power,
    int64_t state_capacity);
pybind11::dict cn_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz,
    torch::Tensor tx_pol);
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
    pybind11::sequence rayd_output,
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
pybind11::dict cn_deterministic_los_field(
    torch::Tensor path_gain,
    torch::Tensor path_length_m,
    double frequency_hz);
// ISB boundary taper (ADR-017) LoS member; both ops are launched only when the
// DEFAULT-OFF isb_boundary_taper switch is on.
torch::Tensor cn_los_silhouette_clearance(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor box_min,
    torch::Tensor box_max,
    double wavelength,
    double width);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cn_los_taper_apply(
    torch::Tensor field_vector,
    torch::Tensor coefficient,
    torch::Tensor path_field,
    torch::Tensor path_gain,
    torch::Tensor tau);
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
pybind11::dict cn_deterministic_reflection_candidate_capacity_block(
    torch::Tensor failure_state,
    torch::Tensor visible,
    torch::Tensor epc_sequences,
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
    bool grouped_export,
    int64_t candidate_capacity);
pybind11::dict cn_enumerated_transmission_topology_pack(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor num_hits,
    torch::Tensor reached_target,
    torch::Tensor overflow,
    torch::Tensor distance,
    torch::Tensor position,
    torch::Tensor normal,
    torch::Tensor global_primitive_id,
    torch::Tensor face_material_id,
    torch::Tensor geometry_mode_id,
    int64_t tx_count,
    int64_t rx_count);
pybind11::dict cn_enumerated_transmission_topology_pack_backward(
    torch::Tensor topology_valid,
    torch::Tensor hit_valid,
    std::optional<torch::Tensor> grad_path_length_m,
    std::optional<torch::Tensor> grad_delay_s,
    std::optional<torch::Tensor> grad_interaction_position,
    std::optional<torch::Tensor> grad_interaction_normal,
    std::optional<torch::Tensor> grad_interaction_positions,
    std::optional<torch::Tensor> grad_interaction_normals);
pybind11::dict cn_enumerated_transmission_topology_pack_jvp(
    torch::Tensor topology_valid,
    torch::Tensor hit_valid,
    std::optional<torch::Tensor> tangent_distance,
    std::optional<torch::Tensor> tangent_position,
    std::optional<torch::Tensor> tangent_normal);
pybind11::dict cn_enumerated_capacity_failure_sanitize(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor material_id,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor interaction_type,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor field_direction,
    torch::Tensor interaction_position,
    torch::Tensor interaction_normal,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor path_gain,
    torch::Tensor path_field,
    torch::Tensor field_xyz,
    torch::Tensor coefficient);
torch::Tensor cn_enumerated_capacity_failure_vector_sanitize(
    torch::Tensor failure_state,
    torch::Tensor values);
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
pybind11::dict cn_deterministic_diffraction_order1_capacity_block(
    torch::Tensor failure_state,
    torch::Tensor count,
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
    torch::Tensor interaction_position,
    int64_t output_capacity);
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
pybind11::dict cn_coupled_candidate_capacity_block(
    torch::Tensor failure_state,
    torch::Tensor representative_faces,
    torch::Tensor selected_edges,
    int64_t tx_count,
    int64_t rx_count,
    int64_t rx_id_offset,
    int64_t candidate_capacity,
    int64_t candidate_limit);
torch::Tensor cn_deterministic_sort_order(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor primitive_sequence);
pybind11::dict cn_enumerated_canonical_capacity_select(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor primitive_sequence,
    torch::Tensor path_length_m,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t max_paths,
    int64_t max_paths_scope);
pybind11::dict cn_deterministic_capacity_finalize(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);
pybind11::dict cn_evaluated_paths_canonical_capacity_gather(
    torch::Tensor failure_state,
    torch::Tensor selected_row_index,
    torch::Tensor selected_valid,
    torch::Tensor num_selected,
    torch::Tensor num_paths,
    torch::Tensor candidate_valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor material_id,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor interaction_type,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor field_direction,
    torch::Tensor interaction_position,
    torch::Tensor interaction_normal,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor path_gain,
    torch::Tensor path_field,
    torch::Tensor field_xyz,
    torch::Tensor coefficient,
    int64_t num_tx,
    int64_t num_rx);
pybind11::dict cn_evaluated_paths_canonical_capacity_gather_backward(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor selected_row_index,
    std::optional<torch::Tensor> grad_path_length_m,
    std::optional<torch::Tensor> grad_delay_s,
    std::optional<torch::Tensor> grad_field_direction,
    std::optional<torch::Tensor> grad_interaction_position,
    std::optional<torch::Tensor> grad_interaction_normal,
    std::optional<torch::Tensor> grad_interaction_positions,
    std::optional<torch::Tensor> grad_interaction_normals,
    std::optional<torch::Tensor> grad_path_gain,
    std::optional<torch::Tensor> grad_path_field,
    std::optional<torch::Tensor> grad_field_xyz,
    std::optional<torch::Tensor> grad_coefficient,
    int64_t candidate_count,
    int64_t sequence_width);
pybind11::dict cn_evaluated_paths_canonical_capacity_gather_jvp(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor selected_row_index,
    std::optional<torch::Tensor> tangent_path_length_m,
    std::optional<torch::Tensor> tangent_delay_s,
    std::optional<torch::Tensor> tangent_field_direction,
    std::optional<torch::Tensor> tangent_interaction_position,
    std::optional<torch::Tensor> tangent_interaction_normal,
    std::optional<torch::Tensor> tangent_interaction_positions,
    std::optional<torch::Tensor> tangent_interaction_normals,
    std::optional<torch::Tensor> tangent_path_gain,
    std::optional<torch::Tensor> tangent_path_field,
    std::optional<torch::Tensor> tangent_field_xyz,
    std::optional<torch::Tensor> tangent_coefficient,
    int64_t candidate_count,
    int64_t sequence_width);
pybind11::dict cn_evaluated_paths_capacity_pack(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor material_id,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor interaction_type,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor field_direction,
    torch::Tensor interaction_position,
    torch::Tensor interaction_normal,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor path_gain,
    torch::Tensor path_field,
    torch::Tensor field_xyz,
    torch::Tensor coefficient,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);
pybind11::dict cn_evaluated_paths_capacity_pack_backward(
    torch::Tensor valid,
    torch::Tensor selected_row_index,
    std::optional<torch::Tensor> grad_path_length_m,
    std::optional<torch::Tensor> grad_delay_s,
    std::optional<torch::Tensor> grad_field_direction,
    std::optional<torch::Tensor> grad_interaction_position,
    std::optional<torch::Tensor> grad_interaction_normal,
    std::optional<torch::Tensor> grad_interaction_positions,
    std::optional<torch::Tensor> grad_interaction_normals,
    std::optional<torch::Tensor> grad_path_gain,
    std::optional<torch::Tensor> grad_path_field,
    std::optional<torch::Tensor> grad_field_xyz,
    std::optional<torch::Tensor> grad_coefficient,
    int64_t candidate_count,
    int64_t sequence_width);
pybind11::dict cn_evaluated_paths_capacity_pack_jvp(
    torch::Tensor valid,
    torch::Tensor selected_row_index,
    std::optional<torch::Tensor> tangent_path_length_m,
    std::optional<torch::Tensor> tangent_delay_s,
    std::optional<torch::Tensor> tangent_field_direction,
    std::optional<torch::Tensor> tangent_interaction_position,
    std::optional<torch::Tensor> tangent_interaction_normal,
    std::optional<torch::Tensor> tangent_interaction_positions,
    std::optional<torch::Tensor> tangent_interaction_normals,
    std::optional<torch::Tensor> tangent_path_gain,
    std::optional<torch::Tensor> tangent_path_field,
    std::optional<torch::Tensor> tangent_field_xyz,
    std::optional<torch::Tensor> tangent_coefficient,
    int64_t candidate_count,
    int64_t sequence_width);
pybind11::dict cn_deterministic_path_table_capacity_pack(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor primitive_id,
    torch::Tensor edge_id,
    torch::Tensor material_id,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor path_length_m,
    torch::Tensor delay_s,
    torch::Tensor field_direction,
    torch::Tensor interaction_position,
    torch::Tensor interaction_normal,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor path_gain,
    torch::Tensor path_field,
    torch::Tensor field_xyz,
    torch::Tensor coefficient,
    torch::Tensor num_paths,
    torch::Tensor overflow,
    bool include_fields,
    int64_t pair_count,
    int64_t path_capacity_per_pair);
pybind11::dict cn_deterministic_path_table_capacity_pack_backward(
    torch::Tensor valid,
    bool include_fields,
    std::optional<torch::Tensor> grad_path_length_m,
    std::optional<torch::Tensor> grad_delay_s,
    std::optional<torch::Tensor> grad_path_gain,
    std::optional<torch::Tensor> grad_interaction_position,
    std::optional<torch::Tensor> grad_interaction_normal,
    std::optional<torch::Tensor> grad_interaction_positions,
    std::optional<torch::Tensor> grad_interaction_normals,
    std::optional<torch::Tensor> grad_field_real,
    std::optional<torch::Tensor> grad_field_imag,
    std::optional<torch::Tensor> grad_coefficient,
    std::optional<torch::Tensor> grad_field_xyz,
    std::optional<torch::Tensor> grad_field_direction,
    int64_t sequence_width);
pybind11::dict cn_deterministic_path_table_capacity_pack_jvp(
    torch::Tensor valid,
    bool include_fields,
    std::optional<torch::Tensor> tangent_path_length_m,
    std::optional<torch::Tensor> tangent_delay_s,
    std::optional<torch::Tensor> tangent_field_direction,
    std::optional<torch::Tensor> tangent_interaction_position,
    std::optional<torch::Tensor> tangent_interaction_normal,
    std::optional<torch::Tensor> tangent_interaction_positions,
    std::optional<torch::Tensor> tangent_interaction_normals,
    std::optional<torch::Tensor> tangent_path_gain,
    std::optional<torch::Tensor> tangent_path_field,
    std::optional<torch::Tensor> tangent_field_xyz,
    std::optional<torch::Tensor> tangent_coefficient,
    int64_t sequence_width);
pybind11::dict cn_deterministic_diffraction_pair_reduce(
    torch::Tensor failure_state,
    torch::Tensor reported_count,
    torch::Tensor valid,
    torch::Tensor x_re,
    torch::Tensor x_im,
    torch::Tensor y_re,
    torch::Tensor y_im,
    torch::Tensor z_re,
    torch::Tensor z_im,
    int64_t pair_count,
    int64_t state_capacity);
pybind11::dict cn_deterministic_diffraction_pair_reduce_backward(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor field_xyz,
    std::optional<torch::Tensor> grad_field_xyz,
    std::optional<torch::Tensor> grad_power,
    int64_t pair_count,
    int64_t state_capacity);
pybind11::dict cn_deterministic_diffraction_pair_reduce_jvp(
    torch::Tensor failure_state,
    torch::Tensor valid,
    torch::Tensor field_xyz,
    std::optional<torch::Tensor> tangent_x_re,
    std::optional<torch::Tensor> tangent_x_im,
    std::optional<torch::Tensor> tangent_y_re,
    std::optional<torch::Tensor> tangent_y_im,
    std::optional<torch::Tensor> tangent_z_re,
    std::optional<torch::Tensor> tangent_z_im,
    int64_t pair_count,
    int64_t state_capacity);
pybind11::dict cn_path_result_capacity_pack(
    torch::Tensor failure_state,
    torch::Tensor overflow,
    torch::Tensor valid,
    torch::Tensor num_paths,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor component_id,
    torch::Tensor edge_id,
    torch::Tensor primitive_sequence,
    torch::Tensor material_sequence,
    torch::Tensor interaction_type,
    torch::Tensor delay_s,
    torch::Tensor field_direction,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor coefficient,
    torch::Tensor field_xyz,
    torch::Tensor tx_positions,
    torch::Tensor rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);
pybind11::dict cn_path_result_capacity_pack_backward(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor interaction_type,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_positions,
    torch::Tensor rx_positions,
    std::optional<torch::Tensor> grad_a,
    std::optional<torch::Tensor> grad_tau,
    std::optional<torch::Tensor> grad_theta_t,
    std::optional<torch::Tensor> grad_phi_t,
    std::optional<torch::Tensor> grad_theta_r,
    std::optional<torch::Tensor> grad_phi_r,
    std::optional<torch::Tensor> grad_position,
    std::optional<torch::Tensor> grad_normal,
    std::optional<torch::Tensor> grad_field_xyz,
    std::optional<torch::Tensor> grad_field_direction,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);
pybind11::dict cn_path_result_capacity_pack_jvp(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor depth,
    torch::Tensor interaction_type,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_positions,
    torch::Tensor rx_positions,
    std::optional<torch::Tensor> tangent_delay_s,
    std::optional<torch::Tensor> tangent_field_direction,
    std::optional<torch::Tensor> tangent_interaction_positions,
    std::optional<torch::Tensor> tangent_interaction_normals,
    std::optional<torch::Tensor> tangent_coefficient,
    std::optional<torch::Tensor> tangent_field_xyz,
    std::optional<torch::Tensor> tangent_tx_positions,
    std::optional<torch::Tensor> tangent_rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair);
pybind11::dict cn_deterministic_accumulate_flat(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor component_id,
    torch::Tensor path_gain,
    torch::Tensor field_real,
    torch::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain);
pybind11::dict cn_deterministic_accumulate_flat_fwd64(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor component_id,
    torch::Tensor path_gain,
    torch::Tensor field_real,
    torch::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain);
pybind11::dict cn_deterministic_accumulate_flat_backward(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor component_id,
    torch::Tensor component_field_real,
    torch::Tensor component_field_imag,
    torch::Tensor field_total_real,
    torch::Tensor field_total_imag,
    torch::Tensor power_total,
    pybind11::object grad_power_total,
    pybind11::object grad_field_total_real,
    pybind11::object grad_field_total_imag,
    pybind11::object grad_component_power,
    pybind11::object grad_component_field_real,
    pybind11::object grad_component_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain);
pybind11::dict cn_deterministic_accumulate_flat_jvp(
    torch::Tensor valid,
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor component_id,
    torch::Tensor component_field_real,
    torch::Tensor component_field_imag,
    torch::Tensor power_total,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_field_real,
    pybind11::object tangent_field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent,
    int64_t scattering_combine_domain);
pybind11::dict cn_deterministic_component_counts(torch::Tensor component_id);
int64_t cn_deterministic_selected_edge_count(torch::Tensor edge_id);

void register_path_core(pybind11::module_ &module) {
    module.def(
        "core_pack_int2",
        &cn_core_pack_int2,
        "Pack two int32 CUDA vectors into an Nx2 tensor.");
    module.def(
        "core_diffraction_edge_count",
        &cn_core_diffraction_edge_count,
        "Count selected diffraction edges with native CUDA policy logic.");
}

void register_path_diffraction_state(pybind11::module_ &module) {
    module.def(
        "deterministic_diffraction_state_pack",
        &cn_deterministic_diffraction_state_pack,
        "Pack deterministic diffraction edge states from a transmitter power vector with a fused CUDA kernel.");
    module.def(
        "deterministic_diffraction_state_pack_selected",
        &cn_deterministic_diffraction_state_pack_selected,
        "Pack fixed-capacity selected diffraction edge states without host count compaction.");
    module.def(
        "deterministic_diffraction_state_capacity_select",
        &cn_deterministic_diffraction_state_capacity_select,
        "Stably select diffraction states into an explicit CUDA capacity block.");
}

void register_path(pybind11::module_ &module) {
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
        "Convert RayD order-1 diffraction outputs into a compact path block with native CUDA kernels.");
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
}

void register_path_deterministic(pybind11::module_ &module) {
    module.def(
        "deterministic_los_field",
        &cn_deterministic_los_field,
        "Evaluate deterministic LoS/free-space scalar complex fields with a CUDA kernel.");
    module.def(
        "los_silhouette_clearance",
        &cn_los_silhouette_clearance,
        "ISB boundary taper (ADR-017): per-(source,target) C1 clearance membership "
        "factor tau against the nearest occluding box silhouette with a CUDA kernel.");
    module.def(
        "los_taper_apply",
        &cn_los_taper_apply,
        "ISB boundary taper (ADR-017): scale a LoS field bundle by the per-row "
        "clearance factor tau with a CUDA kernel.");
    module.def(
        "deterministic_diffraction_vector_field",
        &cn_deterministic_diffraction_vector_field,
        "Convert RayD diffraction vector components into scalar deterministic fields with a CUDA kernel.");
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
        "deterministic_reflection_candidate_capacity_block",
        &cn_deterministic_reflection_candidate_capacity_block,
        "Stably gather visible reflection EPC rows into fixed CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("visible"),
        pybind11::arg("epc_sequences"),
        pybind11::arg("epc_hits"),
        pybind11::arg("epc_normals"),
        pybind11::arg("sequence_batch"),
        pybind11::arg("rx_indices"),
        pybind11::arg("tx"),
        pybind11::arg("rx_positions"),
        pybind11::arg("tx_power"),
        pybind11::arg("tx_index"),
        pybind11::arg("face_eps_r"),
        pybind11::arg("face_sigma_e"),
        pybind11::arg("face_mu_r"),
        pybind11::arg("face_gain"),
        pybind11::arg("face_material_id"),
        pybind11::arg("grouped_export"),
        pybind11::arg("candidate_capacity"));
    module.def(
        "enumerated_transmission_topology_pack",
        &cn_enumerated_transmission_topology_pack,
        "Pack fixed RayD segment hits into inert-capacity transmission topology.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("num_hits"),
        pybind11::arg("reached_target"),
        pybind11::arg("overflow"),
        pybind11::arg("distance"),
        pybind11::arg("position"),
        pybind11::arg("normal"),
        pybind11::arg("global_primitive_id"),
        pybind11::arg("face_material_id"),
        pybind11::arg("geometry_mode_id"),
        pybind11::arg("tx_count"),
        pybind11::arg("rx_count"));
    module.def(
        "enumerated_transmission_topology_pack_backward",
        &cn_enumerated_transmission_topology_pack_backward,
        "Propagate transmission topology cotangents to RayD penetration geometry.",
        pybind11::arg("topology_valid"),
        pybind11::arg("hit_valid"),
        pybind11::arg("grad_path_length_m"),
        pybind11::arg("grad_delay_s"),
        pybind11::arg("grad_interaction_position"),
        pybind11::arg("grad_interaction_normal"),
        pybind11::arg("grad_interaction_positions"),
        pybind11::arg("grad_interaction_normals"));
    module.def(
        "enumerated_transmission_topology_pack_jvp",
        &cn_enumerated_transmission_topology_pack_jvp,
        "Propagate RayD penetration tangents through transmission topology packing.",
        pybind11::arg("topology_valid"),
        pybind11::arg("hit_valid"),
        pybind11::arg("tangent_distance"),
        pybind11::arg("tangent_position"),
        pybind11::arg("tangent_normal"));
    module.def(
        "enumerated_capacity_failure_sanitize",
        &cn_enumerated_capacity_failure_sanitize,
        "Sanitize complete enumerated rows before terminal capacity failure observation.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("primitive_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("material_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("material_sequence"),
        pybind11::arg("interaction_type"),
        pybind11::arg("path_length_m"),
        pybind11::arg("delay_s"),
        pybind11::arg("field_direction"),
        pybind11::arg("interaction_position"),
        pybind11::arg("interaction_normal"),
        pybind11::arg("interaction_positions"),
        pybind11::arg("interaction_normals"),
        pybind11::arg("path_gain"),
        pybind11::arg("path_field"),
        pybind11::arg("field_xyz"),
        pybind11::arg("coefficient"));
    module.def(
        "enumerated_capacity_failure_vector_sanitize",
        &cn_enumerated_capacity_failure_vector_sanitize,
        "Sanitize the enumerated diffraction vector sidecar before terminal failure.",
        pybind11::arg("failure_state"),
        pybind11::arg("values"));
    module.def(
        "deterministic_diffraction_order1_compact",
        &cn_deterministic_diffraction_order1_compact,
        "Compact deterministic first-order diffraction RayD outputs with native CUDA.");
    module.def(
        "deterministic_diffraction_order1_capacity_block",
        &cn_deterministic_diffraction_order1_capacity_block,
        "Stably gather one RayD diffraction exporter chunk into a fixed CUDA capacity block.");
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
        "coupled_candidate_capacity_block",
        &cn_coupled_candidate_capacity_block,
        "Generate the fixed-capacity coupled R-D/D-R/D-D candidate axes.",
        pybind11::arg("failure_state"),
        pybind11::arg("representative_faces"),
        pybind11::arg("selected_edges"),
        pybind11::arg("tx_count"),
        pybind11::arg("rx_count"),
        pybind11::arg("rx_id_offset"),
        pybind11::arg("candidate_capacity"),
        pybind11::arg("candidate_limit"));
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
        "enumerated_canonical_capacity_select",
        &cn_enumerated_canonical_capacity_select,
        "Select canonical enumerated rows into a fixed compact-prefix CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("primitive_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("path_length_m"),
        pybind11::arg("pair_count"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("max_paths"),
        pybind11::arg("max_paths_scope"));
    module.def(
        "deterministic_capacity_finalize",
        &cn_deterministic_capacity_finalize,
        "Stably finalize deterministic rows into pair-major fixed CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("pair_count"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("path_capacity_per_pair"));
    module.def(
        "evaluated_paths_canonical_capacity_gather",
        &cn_evaluated_paths_canonical_capacity_gather,
        "Gather canonical selected rows into a fixed compact-prefix CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("selected_row_index"),
        pybind11::arg("selected_valid"),
        pybind11::arg("num_selected"),
        pybind11::arg("num_paths"),
        pybind11::arg("candidate_valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("primitive_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("material_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("material_sequence"),
        pybind11::arg("interaction_type"),
        pybind11::arg("path_length_m"),
        pybind11::arg("delay_s"),
        pybind11::arg("field_direction"),
        pybind11::arg("interaction_position"),
        pybind11::arg("interaction_normal"),
        pybind11::arg("interaction_positions"),
        pybind11::arg("interaction_normals"),
        pybind11::arg("path_gain"),
        pybind11::arg("path_field"),
        pybind11::arg("field_xyz"),
        pybind11::arg("coefficient"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"));
    module.def(
        "evaluated_paths_canonical_capacity_gather_backward",
        &cn_evaluated_paths_canonical_capacity_gather_backward,
        "Scatter canonical gather cotangents to unique evaluated-path source rows.");
    module.def(
        "evaluated_paths_canonical_capacity_gather_jvp",
        &cn_evaluated_paths_canonical_capacity_gather_jvp,
        "Gather evaluated-path tangents into canonical compact-prefix capacity.");
    module.def(
        "evaluated_paths_capacity_pack",
        &cn_evaluated_paths_capacity_pack,
        "Pack complete evaluated paths into stable pair-major CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("primitive_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("material_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("material_sequence"),
        pybind11::arg("interaction_type"),
        pybind11::arg("path_length_m"),
        pybind11::arg("delay_s"),
        pybind11::arg("field_direction"),
        pybind11::arg("interaction_position"),
        pybind11::arg("interaction_normal"),
        pybind11::arg("interaction_positions"),
        pybind11::arg("interaction_normals"),
        pybind11::arg("path_gain"),
        pybind11::arg("path_field"),
        pybind11::arg("field_xyz"),
        pybind11::arg("coefficient"),
        pybind11::arg("pair_count"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("path_capacity_per_pair"));
    module.def(
        "evaluated_paths_capacity_pack_backward",
        &cn_evaluated_paths_capacity_pack_backward,
        "Scatter capacity cotangents to unique evaluated-path source rows.");
    module.def(
        "evaluated_paths_capacity_pack_jvp",
        &cn_evaluated_paths_capacity_pack_jvp,
        "Gather evaluated-path tangents into stable pair-major capacity.");
    module.def(
        "deterministic_path_table_capacity_pack",
        &cn_deterministic_path_table_capacity_pack,
        "Pack a dormant deterministic PathTable into fixed pair-major CUDA capacity.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("primitive_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("material_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("material_sequence"),
        pybind11::arg("path_length_m"),
        pybind11::arg("delay_s"),
        pybind11::arg("field_direction"),
        pybind11::arg("interaction_position"),
        pybind11::arg("interaction_normal"),
        pybind11::arg("interaction_positions"),
        pybind11::arg("interaction_normals"),
        pybind11::arg("path_gain"),
        pybind11::arg("path_field"),
        pybind11::arg("field_xyz"),
        pybind11::arg("coefficient"),
        pybind11::arg("num_paths"),
        pybind11::arg("overflow"),
        pybind11::arg("include_fields"),
        pybind11::arg("pair_count"),
        pybind11::arg("path_capacity_per_pair"));
    module.def(
        "deterministic_path_table_capacity_pack_backward",
        &cn_deterministic_path_table_capacity_pack_backward,
        "Propagate fixed-capacity PathTable cotangents without compacting rows.",
        pybind11::arg("valid"),
        pybind11::arg("include_fields"),
        pybind11::arg("grad_path_length_m"),
        pybind11::arg("grad_delay_s"),
        pybind11::arg("grad_path_gain"),
        pybind11::arg("grad_interaction_position"),
        pybind11::arg("grad_interaction_normal"),
        pybind11::arg("grad_interaction_positions"),
        pybind11::arg("grad_interaction_normals"),
        pybind11::arg("grad_field_real"),
        pybind11::arg("grad_field_imag"),
        pybind11::arg("grad_coefficient"),
        pybind11::arg("grad_field_xyz"),
        pybind11::arg("grad_field_direction"),
        pybind11::arg("sequence_width"));
    module.def(
        "deterministic_path_table_capacity_pack_jvp",
        &cn_deterministic_path_table_capacity_pack_jvp,
        "Propagate fixed-capacity PathTable tangents without compacting rows.",
        pybind11::arg("valid"),
        pybind11::arg("include_fields"),
        pybind11::arg("tangent_path_length_m"),
        pybind11::arg("tangent_delay_s"),
        pybind11::arg("tangent_field_direction"),
        pybind11::arg("tangent_interaction_position"),
        pybind11::arg("tangent_interaction_normal"),
        pybind11::arg("tangent_interaction_positions"),
        pybind11::arg("tangent_interaction_normals"),
        pybind11::arg("tangent_path_gain"),
        pybind11::arg("tangent_path_field"),
        pybind11::arg("tangent_field_xyz"),
        pybind11::arg("tangent_coefficient"),
        pybind11::arg("sequence_width"));
    module.def(
        "deterministic_diffraction_pair_reduce",
        &cn_deterministic_diffraction_pair_reduce,
        "Reduce fixed source-lane diffraction fields in source-state order.",
        pybind11::arg("failure_state"),
        pybind11::arg("reported_count"),
        pybind11::arg("valid"),
        pybind11::arg("x_re"),
        pybind11::arg("x_im"),
        pybind11::arg("y_re"),
        pybind11::arg("y_im"),
        pybind11::arg("z_re"),
        pybind11::arg("z_im"),
        pybind11::arg("pair_count"),
        pybind11::arg("state_capacity"));
    module.def(
        "deterministic_diffraction_pair_reduce_backward",
        &cn_deterministic_diffraction_pair_reduce_backward,
        "Native fixed-valid diffraction pair-reduction VJP companion.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("field_xyz"),
        pybind11::arg("grad_field_xyz"),
        pybind11::arg("grad_power"),
        pybind11::arg("pair_count"),
        pybind11::arg("state_capacity"));
    module.def(
        "deterministic_diffraction_pair_reduce_jvp",
        &cn_deterministic_diffraction_pair_reduce_jvp,
        "Native fixed-valid diffraction pair-reduction JVP companion.",
        pybind11::arg("failure_state"),
        pybind11::arg("valid"),
        pybind11::arg("field_xyz"),
        pybind11::arg("tangent_x_re"),
        pybind11::arg("tangent_x_im"),
        pybind11::arg("tangent_y_re"),
        pybind11::arg("tangent_y_im"),
        pybind11::arg("tangent_z_re"),
        pybind11::arg("tangent_z_im"),
        pybind11::arg("pair_count"),
        pybind11::arg("state_capacity"));
    module.def(
        "path_result_capacity_pack",
        &cn_path_result_capacity_pack,
        "Pack evaluated capacity rows into canonical PathResult storage.",
        pybind11::arg("failure_state"),
        pybind11::arg("overflow"),
        pybind11::arg("valid"),
        pybind11::arg("num_paths"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("depth"),
        pybind11::arg("component_id"),
        pybind11::arg("edge_id"),
        pybind11::arg("primitive_sequence"),
        pybind11::arg("material_sequence"),
        pybind11::arg("interaction_type"),
        pybind11::arg("delay_s"),
        pybind11::arg("field_direction"),
        pybind11::arg("interaction_positions"),
        pybind11::arg("interaction_normals"),
        pybind11::arg("coefficient"),
        pybind11::arg("field_xyz"),
        pybind11::arg("tx_positions"),
        pybind11::arg("rx_positions"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("path_capacity_per_pair"));
    module.def(
        "path_result_capacity_pack_backward",
        &cn_path_result_capacity_pack_backward,
        "Native fixed-topology PathResult capacity VJP companion.");
    module.def(
        "path_result_capacity_pack_jvp",
        &cn_path_result_capacity_pack_jvp,
        "Native fixed-topology PathResult capacity JVP companion.");
    module.def(
        "deterministic_accumulate_flat",
        &cn_deterministic_accumulate_flat,
        "Accumulate deterministic flat path fields into per-component maps with a CUDA kernel.",
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("component_id"),
        pybind11::arg("path_gain"),
        pybind11::arg("field_real"),
        pybind11::arg("field_imag"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("coherent"),
        pybind11::arg("scattering_combine_domain") = 0);
    module.def(
        "deterministic_accumulate_flat_fwd64",
        &cn_deterministic_accumulate_flat_fwd64,
        "Float64 flat accumulation forward for the strict gradcheck AD path.",
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("component_id"),
        pybind11::arg("path_gain"),
        pybind11::arg("field_real"),
        pybind11::arg("field_imag"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("coherent"),
        pybind11::arg("scattering_combine_domain") = 0);
    module.def(
        "deterministic_accumulate_flat_backward",
        &cn_deterministic_accumulate_flat_backward,
        "Fixed-gate VJP of the flat accumulation (per-path field and power cotangents).",
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("component_id"),
        pybind11::arg("component_field_real"),
        pybind11::arg("component_field_imag"),
        pybind11::arg("field_total_real"),
        pybind11::arg("field_total_imag"),
        pybind11::arg("power_total"),
        pybind11::arg("grad_power_total"),
        pybind11::arg("grad_field_total_real"),
        pybind11::arg("grad_field_total_imag"),
        pybind11::arg("grad_component_power"),
        pybind11::arg("grad_component_field_real"),
        pybind11::arg("grad_component_field_imag"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("coherent"),
        pybind11::arg("scattering_combine_domain") = 0);
    module.def(
        "deterministic_accumulate_flat_jvp",
        &cn_deterministic_accumulate_flat_jvp,
        "Fixed-gate JVP of the flat accumulation (per-path field and power tangents).",
        pybind11::arg("valid"),
        pybind11::arg("tx_id"),
        pybind11::arg("rx_id"),
        pybind11::arg("component_id"),
        pybind11::arg("component_field_real"),
        pybind11::arg("component_field_imag"),
        pybind11::arg("power_total"),
        pybind11::arg("tangent_path_gain"),
        pybind11::arg("tangent_field_real"),
        pybind11::arg("tangent_field_imag"),
        pybind11::arg("num_tx"),
        pybind11::arg("num_rx"),
        pybind11::arg("coherent"),
        pybind11::arg("scattering_combine_domain") = 0);
}
