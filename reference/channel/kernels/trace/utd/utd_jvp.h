#pragma once

#include <trace/utd/utd_types.h>

namespace witwin::channel::native_ext {

// =========================================================================
// UTD Forward-Mode JVP (Jacobian-Vector Product) Mega-Kernel
//
// Given tangent vectors for all differentiable state inputs, propagates
// them through the forward computation to produce tangent vectors for
// the outputs. This is the dual of the VJP (backward) kernel.
//
// Usage in DrJit AD: dr.CustomOp.forward() calls this to compute
// output tangents from input tangents, enabling forward-mode AD.
//
// The kernel reads the same state SoA + rx arrays as the forward kernel,
// plus corresponding tangent arrays for each differentiable input.
// It atomically accumulates output tangents into tangent output buffers.
// =========================================================================

void utd_accumulate_jvp(
    // Per-pair index arrays [n_pairs]
    const int*      state_index,
    const int*      rx_index,
    const int*      ownership_code,

    // State SoA arrays (primal) [n_states]
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

    // Receiver positions (primal) [n_rx]
    const float*    rx_x, const float* rx_y, const float* rx_z,

    // --- Tangent inputs (same layout as primals) ---
    // State tangents [n_states]
    const float*    t_edge_pos_x, const float* t_edge_pos_y, const float* t_edge_pos_z,
    const float*    t_edge_dir_x, const float* t_edge_dir_y, const float* t_edge_dir_z,
    const float*    t_n0_x, const float* t_n0_y, const float* t_n0_z,
    const float*    t_nn_x, const float* t_nn_y, const float* t_nn_z,
    const float*    t_wedge_n,
    const float*    t_source_pos_x, const float* t_source_pos_y, const float* t_source_pos_z,
    const float*    t_inc_field_re, const float* t_inc_field_im,
    const float*    t_inc_nderiv_re, const float* t_inc_nderiv_im,
    const float*    t_r0_re, const float* t_r0_im,
    const float*    t_rn_re, const float* t_rn_im,
    const float*    t_inc_vec_x_re, const float* t_inc_vec_x_im,
    const float*    t_inc_vec_y_re, const float* t_inc_vec_y_im,
    const float*    t_inc_vec_z_re, const float* t_inc_vec_z_im,
    const float*    t_inc_dvec_x_re, const float* t_inc_dvec_x_im,
    const float*    t_inc_dvec_y_re, const float* t_inc_dvec_y_im,
    const float*    t_inc_dvec_z_re, const float* t_inc_dvec_z_im,
    const float*    t_face0_op_m00_re, const float* t_face0_op_m00_im,
    const float*    t_face0_op_m01_re, const float* t_face0_op_m01_im,
    const float*    t_face0_op_m10_re, const float* t_face0_op_m10_im,
    const float*    t_face0_op_m11_re, const float* t_face0_op_m11_im,
    const float*    t_face1_op_m00_re, const float* t_face1_op_m00_im,
    const float*    t_face1_op_m01_re, const float* t_face1_op_m01_im,
    const float*    t_face1_op_m10_re, const float* t_face1_op_m10_im,
    const float*    t_face1_op_m11_re, const float* t_face1_op_m11_im,
    const float*    t_face0_eta_r, const float* t_face0_sigma, const float* t_face0_gain,
    const float*    t_face1_eta_r, const float* t_face1_sigma, const float* t_face1_gain,
    // Receiver tangents [n_rx]
    const float*    t_rx_x, const float* t_rx_y, const float* t_rx_z,

    // --- Output tangents (atomically accumulated) [n_rx] ---
    float*          t_direct_vec_x_re, float* t_direct_vec_x_im,
    float*          t_direct_vec_y_re, float* t_direct_vec_y_im,
    float*          t_direct_vec_z_re, float* t_direct_vec_z_im,
    float*          t_multi_vec_x_re,  float* t_multi_vec_x_im,
    float*          t_multi_vec_y_re,  float* t_multi_vec_y_im,
    float*          t_multi_vec_z_re,  float* t_multi_vec_z_im,

    // Scalars
    int n_pairs,
    float k,
    MaterialParams material
);

} // namespace witwin::channel::native_ext
