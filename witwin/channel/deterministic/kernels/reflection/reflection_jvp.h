#pragma once

namespace witwin::channel::native_ext {

// =========================================================================
// Reflection Accumulation Forward-Mode JVP Kernel
//
// Propagates tangent vectors through reflection-chain EPC +
// Fresnel + Jones transport + point-source field + scatter pipeline.
//
// Given tangent perturbations of geometry (image source, plane points/
// normals, rx positions) and material parameters, produces tangent
// perturbations of the per-receiver output vector field.
// =========================================================================
void reflection_accumulate_jvp(
    // Per-pair indexing [n_pairs]
    const int*   path_idx,
    const int*   rx_idx,
    const int*   valid_mask,

    // Primal path data [n_paths]
    const float* image_source_x, const float* image_source_y, const float* image_source_z,

    // Primal per-bounce slot data [chain_depth * n_paths]
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,

    // Primal receiver positions [n_rx]
    const float* rx_x, const float* rx_y, const float* rx_z,

    // TX polarization (uniform)
    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    // Tangent path data [n_paths]
    const float* t_image_source_x, const float* t_image_source_y, const float* t_image_source_z,

    // Tangent per-bounce slot data [chain_depth * n_paths]
    const float* t_slot_plane_point_x, const float* t_slot_plane_point_y, const float* t_slot_plane_point_z,
    const float* t_slot_plane_normal_x, const float* t_slot_plane_normal_y, const float* t_slot_plane_normal_z,

    // Tangent receiver positions [n_rx]
    const float* t_rx_x, const float* t_rx_y, const float* t_rx_z,

    // Tangent output [n_rx]
    float* t_out_vec_x_re, float* t_out_vec_x_im,
    float* t_out_vec_y_re, float* t_out_vec_y_im,
    float* t_out_vec_z_re, float* t_out_vec_z_im,

    // Scalars
    int n_pairs, int n_paths, int chain_depth,
    float k, float omega
);

void reflection_accumulate_f_weight_jvp(
    // Per-pair indexing [n_pairs]
    const int*   path_idx,
    const int*   rx_idx,
    const int*   valid_mask,

    // Primal path data [n_paths]
    const float* image_source_x, const float* image_source_y, const float* image_source_z,

    // Primal per-bounce primary slot data [chain_depth * n_paths]
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,

    // Primal transition descriptors [chain_depth * n_pairs]
    const int*   transition_support_valid,
    const int*   transition_primary_side,
    const float* transition_edge_distance,
    const float* transition_edge_v0_x, const float* transition_edge_v0_y, const float* transition_edge_v0_z,
    const float* transition_edge_v1_x, const float* transition_edge_v1_y, const float* transition_edge_v1_z,
    const int*   adjacent_valid,
    const float* adjacent_plane_point_x, const float* adjacent_plane_point_y, const float* adjacent_plane_point_z,
    const float* adjacent_plane_normal_x, const float* adjacent_plane_normal_y, const float* adjacent_plane_normal_z,
    const float* adjacent_eta_r,
    const float* adjacent_mu_r,
    const float* adjacent_sigma,
    const float* adjacent_gain,

    // Primal receiver positions [n_rx]
    const float* rx_x, const float* rx_y, const float* rx_z,

    // TX polarization (uniform)
    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    // Tangent path data [n_paths]
    const float* t_image_source_x, const float* t_image_source_y, const float* t_image_source_z,

    // Tangent per-bounce primary slot data [chain_depth * n_paths]
    const float* t_slot_plane_point_x, const float* t_slot_plane_point_y, const float* t_slot_plane_point_z,
    const float* t_slot_plane_normal_x, const float* t_slot_plane_normal_y, const float* t_slot_plane_normal_z,

    // Tangent transition descriptors [chain_depth * n_pairs]
    const float* t_transition_edge_distance,
    const float* t_transition_edge_v0_x, const float* t_transition_edge_v0_y, const float* t_transition_edge_v0_z,
    const float* t_transition_edge_v1_x, const float* t_transition_edge_v1_y, const float* t_transition_edge_v1_z,

    // Tangent adjacent slot data [chain_depth * n_pairs]
    const float* t_adjacent_plane_point_x, const float* t_adjacent_plane_point_y, const float* t_adjacent_plane_point_z,
    const float* t_adjacent_plane_normal_x, const float* t_adjacent_plane_normal_y, const float* t_adjacent_plane_normal_z,

    // Tangent receiver positions [n_rx]
    const float* t_rx_x, const float* t_rx_y, const float* t_rx_z,

    // Tangent output [n_rx]
    float* t_out_vec_x_re, float* t_out_vec_x_im,
    float* t_out_vec_y_re, float* t_out_vec_y_im,
    float* t_out_vec_z_re, float* t_out_vec_z_im,

    // Scalars
    int n_pairs, int n_paths, int chain_depth,
    float k, float omega
);

} // namespace witwin::channel::native_ext
