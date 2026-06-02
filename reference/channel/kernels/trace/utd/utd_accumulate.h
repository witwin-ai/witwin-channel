#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// Forward-declare MaterialParams (defined in utd_types.h)
struct MaterialParams;

// =========================================================================
// UTD Forward Mega-Kernel — host launcher
//
// Fuses field evaluation + scatter_reduce into a single kernel launch.
// All float* pointers are CUDA device pointers (SoA layout).
// =========================================================================
void utd_accumulate_forward(
    // Per-pair index arrays  [n_pairs]
    const int*      state_index,
    const int*      rx_index,
    const int*      ownership_code,
    // State SoA arrays  [n_states]
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    edge_line_min, const float* edge_line_max,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    // Receiver positions  [n_rx]
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    // Output (atomically accumulated) — [n_rx] per field
    float*          direct_re,     float* direct_im,
    float*          multi_re,      float* multi_im,
    float*          direct_vec_x_re, float* direct_vec_x_im,
    float*          direct_vec_y_re, float* direct_vec_y_im,
    float*          direct_vec_z_re, float* direct_vec_z_im,
    float*          multi_vec_x_re,  float* multi_vec_x_im,
    float*          multi_vec_y_re,  float* multi_vec_y_im,
    float*          multi_vec_z_re,  float* multi_vec_z_im,
    // Scalars
    int n_pairs,
    float k,
    MaterialParams material
);

// =========================================================================
// UTD Forward Tiled Mega-Kernel -- host launcher
//
// Enumerates ``local_state x local_receiver`` inside CUDA from compact tile
// descriptors rather than receiving explicit per-pair state/rx arrays.
// Optional ``valid_mask`` gates tile-local pairs after Python-side visibility.
// =========================================================================
void utd_accumulate_tiled_forward(
    // Per-tile compact descriptors
    const int*      state_index,     // [n_local_states]
    const int*      rx_index,        // [n_local_receivers]
    const int*      valid_mask,      // [n_local_states * n_local_receivers] or nullptr
    const int*      ownership_code,  // [n_states]
    // State SoA arrays  [n_states]
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    edge_line_min, const float* edge_line_max,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    // Receiver positions  [n_rx]
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    // Output (atomically accumulated) -- [n_rx] per field
    float*          direct_re,     float* direct_im,
    float*          multi_re,      float* multi_im,
    float*          direct_vec_x_re, float* direct_vec_x_im,
    float*          direct_vec_y_re, float* direct_vec_y_im,
    float*          direct_vec_z_re, float* direct_vec_z_im,
    float*          multi_vec_x_re,  float* multi_vec_x_im,
    float*          multi_vec_y_re,  float* multi_vec_y_im,
    float*          multi_vec_z_re,  float* multi_vec_z_im,
    // Scalars
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material
);

void utd_accumulate_tiled_forward_slots(
    const int*      state_index,
    const int*      rx_index,
    const int*      valid_mask,
    const int*      ownership_code,
    const float* const* state_slots,
    const float*    rx_x,
    const float*    rx_y,
    const float*    rx_z,
    float*          direct_re,
    float*          direct_im,
    float*          multi_re,
    float*          multi_im,
    float*          direct_vec_x_re,
    float*          direct_vec_x_im,
    float*          direct_vec_y_re,
    float*          direct_vec_y_im,
    float*          direct_vec_z_re,
    float*          direct_vec_z_im,
    float*          multi_vec_x_re,
    float*          multi_vec_x_im,
    float*          multi_vec_y_re,
    float*          multi_vec_y_im,
    float*          multi_vec_z_re,
    float*          multi_vec_z_im,
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material
);

// =========================================================================
// UTD Tiled Vector-Power Forward Kernel -- host launcher
//
// Same tiled replay as ``utd_accumulate_tiled_forward`` but additionally
// accumulates matched-isotropic vector power plus the coherent scalar replay
// obtained by projecting each pair's vector field onto the requested receiver
// polarization basis, along with an exact valid-pair counter. RadioMapMonitor
// uses this on no-grad coherent matched-isotropic replays after Python-side
// visibility/support masking.
// =========================================================================
void utd_accumulate_tiled_vector_power_forward(
    // Per-tile compact descriptors
    const int*      state_index,     // [n_local_states]
    const int*      rx_index,        // [n_local_receivers]
    const int*      valid_mask,      // [n_local_states * n_local_receivers] or nullptr
    const int*      ownership_code,  // [n_states]
    // State SoA arrays  [n_states]
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    edge_line_min, const float* edge_line_max,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    // Receiver positions  [n_rx]
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    // Output (atomically accumulated) -- [n_rx] per field, plus matched
    // vector power and one exact valid-pair counter.
    float*          direct_re,     float* direct_im,
    float*          multi_re,      float* multi_im,
    float*          direct_vec_x_re, float* direct_vec_x_im,
    float*          direct_vec_y_re, float* direct_vec_y_im,
    float*          direct_vec_z_re, float* direct_vec_z_im,
    float*          multi_vec_x_re,  float* multi_vec_x_im,
    float*          multi_vec_y_re,  float* multi_vec_y_im,
    float*          multi_vec_z_re,  float* multi_vec_z_im,
    float*          matched_power,
    float*          valid_pair_count,
    // Scalars
    int n_local_states,
    int n_local_receivers,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material
);

void utd_accumulate_tiled_vector_power_forward_slots(
    const int*      state_index,
    const int*      rx_index,
    const int*      valid_mask,
    const int*      ownership_code,
    const float* const* state_slots,
    const float*    rx_x,
    const float*    rx_y,
    const float*    rx_z,
    float*          direct_re,
    float*          direct_im,
    float*          multi_re,
    float*          multi_im,
    float*          direct_vec_x_re,
    float*          direct_vec_x_im,
    float*          direct_vec_y_re,
    float*          direct_vec_y_im,
    float*          direct_vec_z_re,
    float*          direct_vec_z_im,
    float*          multi_vec_x_re,
    float*          multi_vec_x_im,
    float*          multi_vec_y_re,
    float*          multi_vec_y_im,
    float*          multi_vec_z_re,
    float*          multi_vec_z_im,
    float*          matched_power,
    float*          valid_pair_count,
    int n_local_states,
    int n_local_receivers,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material
);

// =========================================================================
// UTD Scalar-Power Forward Kernel -- host launcher
//
// Consumes one compact state per pair, scalarizes the vector contribution
// against a receiver polarization basis, and atomically accumulates both the
// coherent scalar field and incoherent power per receiver.
// =========================================================================
void utd_accumulate_scalar_power_forward(
    // Compact pair-state arrays [n_pairs]
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    edge_line_min, const float* edge_line_max,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    // Per-pair output indices and target positions [n_pairs]
    const int*      output_rx_index,
    const float*    pair_rx_x,     const float* pair_rx_y,    const float* pair_rx_z,
    // Output (atomically accumulated) [n_rx] plus a single valid-pair counter
    float*          coherent_re,
    float*          coherent_im,
    float*          power,
    float*          valid_pair_count,
    // Scalars
    int n_pairs,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material
);

void utd_accumulate_scalar_power_forward_slots(
    const float* const* state_slots,
    const int*      output_rx_index,
    const float*    pair_rx_x,
    const float*    pair_rx_y,
    const float*    pair_rx_z,
    float*          coherent_re,
    float*          coherent_im,
    float*          power,
    float*          valid_pair_count,
    int n_pairs,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material
);

// =========================================================================
// UTD Device Debug Kernel -- host launcher
//
// Evaluates one pair contribution on CUDA from a host-constructed PairInputs
// value and copies a compact debug record back to the host.
// =========================================================================
PairContributionDebug utd_debug_pair_device(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material
);

PairOutputs utd_debug_pair_outputs_device(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material
);

PairContributionDebug utd_debug_pair_from_state_slots(
    const float* const* state_slots,
    int state_index,
    float3a target,
    float k,
    MaterialParams material
);

// Legacy entry points without finite-edge bounds are retained only so older
// bindings keep compiling. They fail at runtime because explicit finite-edge
// bounds are required.
void utd_accumulate_forward(
    const int*      state_index,
    const int*      rx_index,
    const int*      ownership_code,
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    float*          direct_re,     float* direct_im,
    float*          multi_re,      float* multi_im,
    float*          direct_vec_x_re, float* direct_vec_x_im,
    float*          direct_vec_y_re, float* direct_vec_y_im,
    float*          direct_vec_z_re, float* direct_vec_z_im,
    float*          multi_vec_x_re,  float* multi_vec_x_im,
    float*          multi_vec_y_re,  float* multi_vec_y_im,
    float*          multi_vec_z_re,  float* multi_vec_z_im,
    int n_pairs,
    float k,
    MaterialParams material
);

void utd_accumulate_tiled_forward(
    const int*      state_index,
    const int*      rx_index,
    const int*      valid_mask,
    const int*      ownership_code,
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    float*          direct_re,     float* direct_im,
    float*          multi_re,      float* multi_im,
    float*          direct_vec_x_re, float* direct_vec_x_im,
    float*          direct_vec_y_re, float* direct_vec_y_im,
    float*          direct_vec_z_re, float* direct_vec_z_im,
    float*          multi_vec_x_re,  float* multi_vec_x_im,
    float*          multi_vec_y_re,  float* multi_vec_y_im,
    float*          multi_vec_z_re,  float* multi_vec_z_im,
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material
);

void utd_accumulate_scalar_power_forward(
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    const int*      output_rx_index,
    const float*    pair_rx_x,     const float* pair_rx_y,    const float* pair_rx_z,
    float*          coherent_re,
    float*          coherent_im,
    float*          power,
    float*          valid_pair_count,
    int n_pairs,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material
);

// =========================================================================
// UTD Backward Mega-Kernel — host launcher
//
// Reverse-mode VJP. Reads upstream gradients from output buffers and
// atomically accumulates gradients into state and rx gradient buffers.
// =========================================================================
void utd_accumulate_backward(
    // Per-pair index arrays  [n_pairs]
    const int*      state_index,
    const int*      rx_index,
    const int*      ownership_code,
    // State SoA arrays  [n_states]  (same as forward)
    const float*    edge_pos_x,   const float* edge_pos_y,   const float* edge_pos_z,
    const float*    edge_dir_x,   const float* edge_dir_y,   const float* edge_dir_z,
    const float*    n0_x,         const float* n0_y,         const float* n0_z,
    const float*    nn_x,         const float* nn_y,         const float* nn_z,
    const float*    wedge_n,
    const float*    source_pos_x, const float* source_pos_y, const float* source_pos_z,
    const float*    inc_field_re,  const float* inc_field_im,
    const float*    inc_nderiv_re, const float* inc_nderiv_im,
    const float*    r0_re,         const float* r0_im,
    const float*    rn_re,         const float* rn_im,
    const float*    inc_vec_x_re,  const float* inc_vec_x_im,
    const float*    inc_vec_y_re,  const float* inc_vec_y_im,
    const float*    inc_vec_z_re,  const float* inc_vec_z_im,
    const float*    inc_dvec_x_re, const float* inc_dvec_x_im,
    const float*    inc_dvec_y_re, const float* inc_dvec_y_im,
    const float*    inc_dvec_z_re, const float* inc_dvec_z_im,
    const float*    inc_jones_u_re, const float* inc_jones_u_im,
    const float*    inc_jones_v_re, const float* inc_jones_v_im,
    const float*    inc_djones_u_re, const float* inc_djones_u_im,
    const float*    inc_djones_v_re, const float* inc_djones_v_im,
    const float*    inc_basis_u_x, const float* inc_basis_u_y, const float* inc_basis_u_z,
    const float*    inc_basis_v_x, const float* inc_basis_v_y, const float* inc_basis_v_z,
    const float*    inc_basis_k_x, const float* inc_basis_k_y, const float* inc_basis_k_z,
    const float*    face0_op_m00_re, const float* face0_op_m00_im,
    const float*    face0_op_m01_re, const float* face0_op_m01_im,
    const float*    face0_op_m10_re, const float* face0_op_m10_im,
    const float*    face0_op_m11_re, const float* face0_op_m11_im,
    const float*    face1_op_m00_re, const float* face1_op_m00_im,
    const float*    face1_op_m01_re, const float* face1_op_m01_im,
    const float*    face1_op_m10_re, const float* face1_op_m10_im,
    const float*    face1_op_m11_re, const float* face1_op_m11_im,
    const float*    face0_eta_r,    const float* face0_sigma,
    const float*    face0_gain,     const float* face0_use_fresnel,
    const float*    face0_present,
    const float*    face1_eta_r,    const float* face1_sigma,
    const float*    face1_gain,     const float* face1_use_fresnel,
    const float*    face1_present,
    // Receiver positions  [n_rx]
    const float*    rx_x,          const float* rx_y,         const float* rx_z,
    // Upstream gradients (read-only) — [n_rx]
    const float*    grad_direct_re,  const float* grad_direct_im,
    const float*    grad_multi_re,   const float* grad_multi_im,
    const float*    grad_direct_vec_x_re, const float* grad_direct_vec_x_im,
    const float*    grad_direct_vec_y_re, const float* grad_direct_vec_y_im,
    const float*    grad_direct_vec_z_re, const float* grad_direct_vec_z_im,
    const float*    grad_multi_vec_x_re,  const float* grad_multi_vec_x_im,
    const float*    grad_multi_vec_y_re,  const float* grad_multi_vec_y_im,
    const float*    grad_multi_vec_z_re,  const float* grad_multi_vec_z_im,
    // Output: gradient accumulators (atomicAdd) — state grads [n_states], rx grads [n_rx]
    float*          grad_edge_pos_x, float* grad_edge_pos_y, float* grad_edge_pos_z,
    float*          grad_edge_dir_x, float* grad_edge_dir_y, float* grad_edge_dir_z,
    float*          grad_n0_x,       float* grad_n0_y,       float* grad_n0_z,
    float*          grad_nn_x,       float* grad_nn_y,       float* grad_nn_z,
    float*          grad_wedge_n,
    float*          grad_source_pos_x, float* grad_source_pos_y, float* grad_source_pos_z,
    float*          grad_inc_field_re,  float* grad_inc_field_im,
    float*          grad_inc_nderiv_re, float* grad_inc_nderiv_im,
    float*          grad_r0_re,         float* grad_r0_im,
    float*          grad_rn_re,         float* grad_rn_im,
    float*          grad_inc_vec_x_re,  float* grad_inc_vec_x_im,
    float*          grad_inc_vec_y_re,  float* grad_inc_vec_y_im,
    float*          grad_inc_vec_z_re,  float* grad_inc_vec_z_im,
    float*          grad_inc_dvec_x_re, float* grad_inc_dvec_x_im,
    float*          grad_inc_dvec_y_re, float* grad_inc_dvec_y_im,
    float*          grad_inc_dvec_z_re, float* grad_inc_dvec_z_im,
    float*          grad_face0_op_m00_re, float* grad_face0_op_m00_im,
    float*          grad_face0_op_m01_re, float* grad_face0_op_m01_im,
    float*          grad_face0_op_m10_re, float* grad_face0_op_m10_im,
    float*          grad_face0_op_m11_re, float* grad_face0_op_m11_im,
    float*          grad_face1_op_m00_re, float* grad_face1_op_m00_im,
    float*          grad_face1_op_m01_re, float* grad_face1_op_m01_im,
    float*          grad_face1_op_m10_re, float* grad_face1_op_m10_im,
    float*          grad_face1_op_m11_re, float* grad_face1_op_m11_im,
    float*          grad_face0_eta_r, float* grad_face0_sigma, float* grad_face0_gain,
    float*          grad_face1_eta_r, float* grad_face1_sigma, float* grad_face1_gain,
    float*          grad_rx_x, float* grad_rx_y, float* grad_rx_z,
    // Scalars
    int n_pairs,
    float k,
    MaterialParams material
);

} // namespace witwin::channel::native_ext
