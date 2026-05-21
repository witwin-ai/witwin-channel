#pragma once

namespace witwin::channel::native_ext {

// =========================================================================
// Reflection Accumulation Forward Mega-Kernel -- host launcher
//
// Fuses exact path calculation (EPC) + Fresnel + Jones transport + point-source field +
// scatter-reduce into a single kernel launch. Visibility stays in Python.
//
// Per-bounce slot data uses slot-major layout:
//   index for (slot s, path p) = s * n_paths + p
//
// All float*/int* arguments are CUDA device pointers.
// =========================================================================
void reflection_accumulate_forward(
    // Per-pair indexing [n_pairs]
    const int*   path_idx,       // index into path arrays
    const int*   rx_idx,         // index into receiver arrays
    const int*   valid_mask,     // from visibility check (1=valid, 0=invalid)

    // Path data [n_paths]
    const float* image_source_x, const float* image_source_y, const float* image_source_z,

    // Per-bounce slot data [chain_depth * n_paths], slot-major
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,

    // Pre-resolved material per bounce [chain_depth * n_paths]
    const float* slot_eta_r,
    const float* slot_mu_r,
    const float* slot_sigma,
    const float* slot_gain,

    // Receiver positions [n_rx]
    const float* rx_x, const float* rx_y, const float* rx_z,

    // TX polarization direction (uniform scalars)
    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    // Output: atomically accumulated per-receiver vector field [n_rx, 6]
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,

    // Scalars
    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega     // = 2*pi*c/wavelength
);

void reflection_accumulate_f_weight_forward(
    // Per-pair indexing [n_pairs]
    const int*   path_idx,
    const int*   rx_idx,
    const int*   valid_mask,

    // Path data [n_paths]
    const float* image_source_x, const float* image_source_y, const float* image_source_z,

    // Primary per-bounce slot data [chain_depth * n_paths], slot-major
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r,
    const float* slot_mu_r,
    const float* slot_sigma,
    const float* slot_gain,

    // F-weight transition descriptors [chain_depth * n_pairs], slot-major
    const int*   transition_support_valid,
    const int*   transition_primary_side,
    const float* transition_edge_distance,
    const int*   adjacent_valid,
    const float* adjacent_plane_point_x, const float* adjacent_plane_point_y, const float* adjacent_plane_point_z,
    const float* adjacent_plane_normal_x, const float* adjacent_plane_normal_y, const float* adjacent_plane_normal_z,
    const float* adjacent_eta_r,
    const float* adjacent_mu_r,
    const float* adjacent_sigma,
    const float* adjacent_gain,

    // Receiver positions [n_rx]
    const float* rx_x, const float* rx_y, const float* rx_z,

    // TX position (uniform scalars)
    float tx_pos_x, float tx_pos_y, float tx_pos_z,

    // TX polarization direction (uniform scalars)
    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    // Output: atomically accumulated per-receiver vector field [n_rx, 6]
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,

    // Scalars
    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega
);

void reflection_epc_targets_forward(
    const int* path_idx,
    const float* image_source_x, const float* image_source_y, const float* image_source_z,
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,
    const float* target_x, const float* target_y, const float* target_z,
    float tx_pol_x, float tx_pol_y, float tx_pol_z,
    int* out_geom_valid,
    float* out_tx_pos_x, float* out_tx_pos_y, float* out_tx_pos_z,
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,
    float* out_hit_x,
    float* out_hit_y,
    float* out_hit_z,
    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega
);

} // namespace witwin::channel::native_ext
