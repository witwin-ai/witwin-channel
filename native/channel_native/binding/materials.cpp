#include <torch/extension.h>

#include "registry.h"

#include <cstdint>
#include <string>
#include <vector>

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
pybind11::dict cn_em_layer_stack_backward(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::sequence grad_outputs,
    bool need_cos_theta,
    bool need_layers,
    bool need_frequency);
pybind11::dict cn_em_layer_stack_jvp(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_cos_theta,
    pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency);
pybind11::dict cn_scattering_table_eval(
    torch::Tensor wi, torch::Tensor wo, torch::Tensor f_te, torch::Tensor f_tm);
torch::Tensor cn_scattering_table_pdf(
    torch::Tensor wi, torch::Tensor wo, torch::Tensor sample_density, bool reverse);
pybind11::dict cn_scattering_table_sample(
    torch::Tensor wi,
    torch::Tensor uniforms,
    torch::Tensor marginal_cdf,
    torch::Tensor conditional_cdf,
    torch::Tensor sample_density);
pybind11::dict cn_scattering_event_probabilities(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor cap_r_te,
    torch::Tensor cap_r_tm,
    torch::Tensor cap_t_te,
    torch::Tensor cap_t_tm,
    torch::Tensor rough_sigma_h_m,
    torch::Tensor scatter_model_id,
    double frequency_hz,
    double probability_floor);
pybind11::dict cn_scattering_patch_integral_eval(
    torch::Tensor patch_tris,
    torch::Tensor patch_uvs,
    torch::Tensor rows,
    torch::Tensor d_i,
    torch::Tensor d_o,
    torch::Tensor n_rows,
    torch::Tensor r_te,
    torch::Tensor r_tm,
    torch::Tensor pol_t,
    torch::Tensor pol_r,
    torch::Tensor r1_rows,
    torch::Tensor r2_rows,
    torch::Tensor centroids,
    torch::Tensor heights,
    torch::Tensor quad_a,
    torch::Tensor quad_b,
    torch::Tensor quad_w,
    double k0);
pybind11::dict cn_scattering_ensemble_eval(
    torch::Tensor wo_rows,
    torch::Tensor r2_rows,
    torch::Tensor cos_o_rows,
    torch::Tensor n_o,
    torch::Tensor t1r,
    torch::Tensor t2r,
    torch::Tensor wi_local,
    torch::Tensor cos_i,
    torch::Tensor r1,
    torch::Tensor a_te2,
    torch::Tensor a_tm2,
    torch::Tensor weights,
    torch::Tensor material_id,
    torch::Tensor backup_axis,
    torch::Tensor rx_pol,
    torch::Tensor rc_idx,
    torch::Tensor sc_idx,
    torch::Tensor fte_flat,
    torch::Tensor ftm_flat,
    torch::Tensor table_offset,
    torch::Tensor table_dims,
    torch::Tensor material_slot,
    double coef,
    double threshold);
pybind11::dict cn_scattering_ensemble_eval_backward(
    torch::Tensor wo_rows,
    torch::Tensor r2_rows,
    torch::Tensor cos_o_rows,
    torch::Tensor n_o,
    torch::Tensor t1r,
    torch::Tensor t2r,
    torch::Tensor wi_local,
    torch::Tensor cos_i,
    torch::Tensor r1,
    torch::Tensor a_te2,
    torch::Tensor a_tm2,
    torch::Tensor weights,
    torch::Tensor material_id,
    torch::Tensor backup_axis,
    torch::Tensor rx_pol,
    torch::Tensor rc_idx,
    torch::Tensor sc_idx,
    torch::Tensor fte_flat,
    torch::Tensor ftm_flat,
    torch::Tensor table_offset,
    torch::Tensor table_dims,
    torch::Tensor material_slot,
    double coef,
    double threshold,
    pybind11::object grad_gain,
    pybind11::object grad_amplitude,
    pybind11::object grad_length,
    bool need_grad_rows,
    bool need_grad_samples,
    bool need_grad_tables,
    bool need_grad_coef);
pybind11::dict cn_scattering_ensemble_eval_jvp(
    torch::Tensor wo_rows,
    torch::Tensor r2_rows,
    torch::Tensor cos_o_rows,
    torch::Tensor n_o,
    torch::Tensor t1r,
    torch::Tensor t2r,
    torch::Tensor wi_local,
    torch::Tensor cos_i,
    torch::Tensor r1,
    torch::Tensor a_te2,
    torch::Tensor a_tm2,
    torch::Tensor weights,
    torch::Tensor material_id,
    torch::Tensor backup_axis,
    torch::Tensor rx_pol,
    torch::Tensor rc_idx,
    torch::Tensor sc_idx,
    torch::Tensor fte_flat,
    torch::Tensor ftm_flat,
    torch::Tensor table_offset,
    torch::Tensor table_dims,
    torch::Tensor material_slot,
    double coef,
    double threshold,
    pybind11::object t_wo_rows,
    pybind11::object t_r2_rows,
    pybind11::object t_cos_o_rows,
    pybind11::object t_n_o,
    pybind11::object t_t1r,
    pybind11::object t_t2r,
    pybind11::object t_wi_local,
    pybind11::object t_cos_i,
    pybind11::object t_r1,
    pybind11::object t_a_te2,
    pybind11::object t_a_tm2,
    pybind11::object t_weights,
    pybind11::object t_fte_flat,
    pybind11::object t_ftm_flat,
    double tangent_coef);
pybind11::dict cn_scattering_patch_integral_eval_backward(
    torch::Tensor patch_tris,
    torch::Tensor patch_uvs,
    torch::Tensor rows,
    torch::Tensor d_i,
    torch::Tensor d_o,
    torch::Tensor n_rows,
    torch::Tensor r_te,
    torch::Tensor r_tm,
    torch::Tensor pol_t,
    torch::Tensor pol_r,
    torch::Tensor r1_rows,
    torch::Tensor r2_rows,
    torch::Tensor centroids,
    torch::Tensor heights,
    torch::Tensor quad_a,
    torch::Tensor quad_b,
    torch::Tensor quad_w,
    double k0,
    torch::Tensor grad_total,
    bool need_grad_heights,
    bool need_grad_jones,
    bool need_grad_geometry,
    bool need_grad_k0);
pybind11::dict cn_scattering_patch_integral_eval_jvp(
    torch::Tensor patch_tris,
    torch::Tensor patch_uvs,
    torch::Tensor rows,
    torch::Tensor d_i,
    torch::Tensor d_o,
    torch::Tensor n_rows,
    torch::Tensor r_te,
    torch::Tensor r_tm,
    torch::Tensor pol_t,
    torch::Tensor pol_r,
    torch::Tensor r1_rows,
    torch::Tensor r2_rows,
    torch::Tensor centroids,
    torch::Tensor heights,
    torch::Tensor quad_a,
    torch::Tensor quad_b,
    torch::Tensor quad_w,
    double k0,
    pybind11::object t_heights,
    pybind11::object t_r_te,
    pybind11::object t_r_tm,
    pybind11::object t_d_i,
    pybind11::object t_d_o,
    pybind11::object t_r1_rows,
    pybind11::object t_r2_rows,
    pybind11::object t_centroids,
    double tangent_k0);
pybind11::dict cn_scattering_table_eval_backward(
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object grad_out_f_te,
    pybind11::object grad_out_f_tm,
    bool need_grad_dirs,
    bool need_grad_tables);
pybind11::dict cn_scattering_table_eval_jvp(
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object t_wi,
    pybind11::object t_wo,
    pybind11::object t_f_te,
    pybind11::object t_f_tm);
pybind11::dict cn_kirchhoff_table_build_backward(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, at::Tensor grad_f_te, at::Tensor grad_f_tm,
    bool need_grad_rough, bool need_grad_layers, bool need_grad_frequency);
pybind11::dict cn_kirchhoff_table_build_jvp(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, pybind11::object t_layer_thickness_m,
    pybind11::object t_layer_eps_r, pybind11::object t_layer_sigma_e,
    double t_sigma_h, double t_corr_x, double t_corr_y, double t_frequency);

// ADR-021 Op A (scattering_chain_ensemble.cu / _ad.cu). NOTE: the argument set
// follows the committed float64 oracle tests/reference/chain_ensemble.py and the
// existing native op-1 convention (weights + L1,L2 spreading; frozen wi_local;
// explicit source/vertex/target endpoints), reconciling the plan-10a section 3.1
// sketch against the oracle per the "existing native op conventions win" rule
// (see the change report / open issues).
pybind11::dict cn_scattering_chain_ensemble_eval(
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions, at::Tensor c1_normals,
    at::Tensor c1_eps_r, at::Tensor c1_sigma_e, at::Tensor c1_mu_r,
    at::Tensor c1_gain, at::Tensor c1_thickness, at::Tensor c1_depth,
    at::Tensor c2_positions, at::Tensor c2_normals, at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e, at::Tensor c2_mu_r, at::Tensor c2_gain,
    at::Tensor c2_thickness, at::Tensor c2_depth, at::Tensor n_o, at::Tensor t1r,
    at::Tensor t2r, at::Tensor backup_axis, at::Tensor wi_local, at::Tensor cos_i,
    at::Tensor cos_o, at::Tensor d_i, at::Tensor d_o, at::Tensor l1, at::Tensor l2,
    at::Tensor weights, at::Tensor material_id, at::Tensor fte_flat,
    at::Tensor ftm_flat, at::Tensor table_offset, at::Tensor table_dims,
    at::Tensor material_slot, double coef, double threshold, double frequency_hz);
pybind11::dict cn_scattering_chain_ensemble_eval_backward(
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions, at::Tensor c1_normals,
    at::Tensor c1_eps_r, at::Tensor c1_sigma_e, at::Tensor c1_mu_r,
    at::Tensor c1_gain, at::Tensor c1_thickness, at::Tensor c1_depth,
    at::Tensor c2_positions, at::Tensor c2_normals, at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e, at::Tensor c2_mu_r, at::Tensor c2_gain,
    at::Tensor c2_thickness, at::Tensor c2_depth, at::Tensor n_o, at::Tensor t1r,
    at::Tensor t2r, at::Tensor backup_axis, at::Tensor wi_local, at::Tensor cos_i,
    at::Tensor cos_o, at::Tensor d_i, at::Tensor d_o, at::Tensor l1, at::Tensor l2,
    at::Tensor weights, at::Tensor material_id, at::Tensor fte_flat,
    at::Tensor ftm_flat, at::Tensor table_offset, at::Tensor table_dims,
    at::Tensor material_slot, double coef, double threshold, double frequency_hz,
    pybind11::object grad_gain, pybind11::object grad_amplitude,
    pybind11::object grad_length, bool need_grad_chain1, bool need_grad_chain2,
    bool need_grad_tables, bool need_grad_geometry, bool need_grad_coef,
    bool need_grad_frequency);
pybind11::dict cn_scattering_chain_ensemble_eval_jvp(
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions, at::Tensor c1_normals,
    at::Tensor c1_eps_r, at::Tensor c1_sigma_e, at::Tensor c1_mu_r,
    at::Tensor c1_gain, at::Tensor c1_thickness, at::Tensor c1_depth,
    at::Tensor c2_positions, at::Tensor c2_normals, at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e, at::Tensor c2_mu_r, at::Tensor c2_gain,
    at::Tensor c2_thickness, at::Tensor c2_depth, at::Tensor n_o, at::Tensor t1r,
    at::Tensor t2r, at::Tensor backup_axis, at::Tensor wi_local, at::Tensor cos_i,
    at::Tensor cos_o, at::Tensor d_i, at::Tensor d_o, at::Tensor l1, at::Tensor l2,
    at::Tensor weights, at::Tensor material_id, at::Tensor fte_flat,
    at::Tensor ftm_flat, at::Tensor table_offset, at::Tensor table_dims,
    at::Tensor material_slot, double coef, double threshold, double frequency_hz,
    pybind11::object tangent_c1_eps_r, pybind11::object tangent_c1_sigma_e,
    pybind11::object tangent_c1_gain, pybind11::object tangent_c1_thickness,
    pybind11::object tangent_c2_eps_r, pybind11::object tangent_c2_sigma_e,
    pybind11::object tangent_c2_gain, pybind11::object tangent_c2_thickness,
    pybind11::object tangent_f_te_flat, pybind11::object tangent_f_tm_flat,
    pybind11::object tangent_c1_positions, pybind11::object tangent_c1_normals,
    pybind11::object tangent_c2_positions, pybind11::object tangent_c2_normals,
    pybind11::object tangent_d_i, pybind11::object tangent_d_o,
    pybind11::object tangent_v_normal, pybind11::object tangent_l1,
    pybind11::object tangent_l2, pybind11::object tangent_cos_i,
    pybind11::object tangent_cos_o, double tangent_coef, double tangent_frequency);
// ADR-021 Op B (scattering_chain_realization.cu / _ad.cu, sibling change).
// Declared extern per plan-10a section 4; the definitions land with the sibling.
pybind11::dict cn_scattering_chain_realization_eval(
    at::Tensor patch_tris, at::Tensor patch_uvs, at::Tensor rows, at::Tensor d_i,
    at::Tensor d_o, at::Tensor n_rows, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions,
    at::Tensor c1_normals, at::Tensor c1_eps_r, at::Tensor c1_sigma_e,
    at::Tensor c1_mu_r, at::Tensor c1_gain, at::Tensor c1_thickness,
    at::Tensor c1_depth, at::Tensor c2_positions, at::Tensor c2_normals,
    at::Tensor c2_eps_r, at::Tensor c2_sigma_e, at::Tensor c2_mu_r,
    at::Tensor c2_gain, at::Tensor c2_thickness, at::Tensor c2_depth,
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor l1, at::Tensor l2,
    at::Tensor sp1, at::Tensor sp2, at::Tensor centroids, at::Tensor heights,
    at::Tensor cos_spec, at::Tensor material_id, at::Tensor layer_offset,
    at::Tensor layer_count, at::Tensor layer_thickness_m, at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e, at::Tensor layer_mu_r, at::Tensor quad_a,
    at::Tensor quad_b, at::Tensor quad_w, double k0, double frequency_hz);
pybind11::dict cn_scattering_chain_realization_eval_backward(
    at::Tensor patch_tris, at::Tensor patch_uvs, at::Tensor rows, at::Tensor d_i,
    at::Tensor d_o, at::Tensor n_rows, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions,
    at::Tensor c1_normals, at::Tensor c1_eps_r, at::Tensor c1_sigma_e,
    at::Tensor c1_mu_r, at::Tensor c1_gain, at::Tensor c1_thickness,
    at::Tensor c1_depth, at::Tensor c2_positions, at::Tensor c2_normals,
    at::Tensor c2_eps_r, at::Tensor c2_sigma_e, at::Tensor c2_mu_r,
    at::Tensor c2_gain, at::Tensor c2_thickness, at::Tensor c2_depth,
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor l1, at::Tensor l2,
    at::Tensor sp1, at::Tensor sp2, at::Tensor centroids, at::Tensor heights,
    at::Tensor cos_spec, at::Tensor material_id, at::Tensor layer_offset,
    at::Tensor layer_count, at::Tensor layer_thickness_m, at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e, at::Tensor layer_mu_r, at::Tensor quad_a,
    at::Tensor quad_b, at::Tensor quad_w, double k0, double frequency_hz,
    at::Tensor grad_total, pybind11::object grad_path_field,
    pybind11::object grad_path_gain, bool need_grad_heights, bool need_grad_layers,
    bool need_grad_chain1, bool need_grad_chain2, bool need_grad_geometry,
    bool need_grad_k0, bool need_grad_frequency);
pybind11::dict cn_scattering_chain_realization_eval_jvp(
    at::Tensor patch_tris, at::Tensor patch_uvs, at::Tensor rows, at::Tensor d_i,
    at::Tensor d_o, at::Tensor n_rows, at::Tensor source, at::Tensor vertex,
    at::Tensor target, at::Tensor c1_positions,
    at::Tensor c1_normals, at::Tensor c1_eps_r, at::Tensor c1_sigma_e,
    at::Tensor c1_mu_r, at::Tensor c1_gain, at::Tensor c1_thickness,
    at::Tensor c1_depth, at::Tensor c2_positions, at::Tensor c2_normals,
    at::Tensor c2_eps_r, at::Tensor c2_sigma_e, at::Tensor c2_mu_r,
    at::Tensor c2_gain, at::Tensor c2_thickness, at::Tensor c2_depth,
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor l1, at::Tensor l2,
    at::Tensor sp1, at::Tensor sp2, at::Tensor centroids, at::Tensor heights,
    at::Tensor cos_spec, at::Tensor material_id, at::Tensor layer_offset,
    at::Tensor layer_count, at::Tensor layer_thickness_m, at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e, at::Tensor layer_mu_r, at::Tensor quad_a,
    at::Tensor quad_b, at::Tensor quad_w, double k0, double frequency_hz,
    pybind11::object tangent_heights, pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r, pybind11::object tangent_layer_sigma_e,
    pybind11::object tangent_c1_eps_r, pybind11::object tangent_c1_sigma_e,
    pybind11::object tangent_c1_gain, pybind11::object tangent_c1_thickness,
    pybind11::object tangent_c2_eps_r, pybind11::object tangent_c2_sigma_e,
    pybind11::object tangent_c2_gain, pybind11::object tangent_c2_thickness,
    pybind11::object tangent_d_i, pybind11::object tangent_d_o,
    pybind11::object tangent_c1_positions, pybind11::object tangent_c1_normals,
    pybind11::object tangent_c2_positions, pybind11::object tangent_c2_normals,
    pybind11::object tangent_l1, pybind11::object tangent_l2,
    pybind11::object tangent_sp1, pybind11::object tangent_sp2,
    pybind11::object tangent_centroids, double tangent_k0,
    double tangent_frequency);

void register_materials(pybind11::module_ &module) {
    module.def(
        "em_layer_stack_eval",
        &cn_em_layer_stack_eval,
        "Evaluate shared em/ layer-stack reflection/transmission coefficients for parity tests.");
    module.def(
        "em_layer_stack_backward",
        &cn_em_layer_stack_backward,
        "Layer-stack VJP over cos_theta, CSR layer parameters and frequency (CUDA duals).");
    module.def(
        "em_layer_stack_jvp",
        &cn_em_layer_stack_jvp,
        "Layer-stack JVP over cos_theta, CSR layer parameters and frequency (CUDA duals).");
    module.def("scattering_table_eval", &cn_scattering_table_eval,
               "Evaluate a resident Kirchhoff BSDF table with native CUDA.");
    module.def("scattering_table_pdf", &cn_scattering_table_pdf,
               "Evaluate a resident Kirchhoff sampling PDF with native CUDA.");
    module.def("scattering_table_sample", &cn_scattering_table_sample,
               "Sample a resident Kirchhoff CDF table with native CUDA.");
    module.def("scattering_event_probabilities", &cn_scattering_event_probabilities,
               "Evaluate fused rough-surface event budgets with native CUDA.");
    module.def("scattering_ensemble_eval", &cn_scattering_ensemble_eval,
               "Evaluate the Kirchhoff ensemble scattering row physics with native CUDA (ADR-010 op 1).");
    module.def("scattering_patch_integral_eval", &cn_scattering_patch_integral_eval,
               "Evaluate the realization-coherent phase-screen patch integral with native CUDA (ADR-010 op 2).");
    module.def("scattering_ensemble_eval_backward", &cn_scattering_ensemble_eval_backward,
               "Fixed-topology VJP of the Kirchhoff ensemble scattering rows (rows, samples, tables, coef) (ADR-014).");
    module.def("scattering_ensemble_eval_jvp", &cn_scattering_ensemble_eval_jvp,
               "Fixed-topology JVP of the Kirchhoff ensemble scattering rows (rows, samples, tables, coef) (ADR-014).");
    module.def("scattering_patch_integral_eval_backward", &cn_scattering_patch_integral_eval_backward,
               "Fixed-topology VJP of the realization-coherent phase-screen patch integral (heights, jones, geometry, k0) (ADR-014).");
    module.def("scattering_patch_integral_eval_jvp", &cn_scattering_patch_integral_eval_jvp,
               "Fixed-topology JVP of the realization-coherent phase-screen patch integral (heights, jones, geometry, k0) (ADR-014).");
    module.def("scattering_table_eval_backward", &cn_scattering_table_eval_backward,
               "Fixed-topology VJP of the resident Kirchhoff BSDF table lookup (directions, tables) (ADR-015).");
    module.def("scattering_table_eval_jvp", &cn_scattering_table_eval_jvp,
               "Fixed-topology JVP of the resident Kirchhoff BSDF table lookup (directions, tables) (ADR-015).");
    module.def("scattering_chain_ensemble_eval", &cn_scattering_chain_ensemble_eval,
               "Evaluate the ADR-021 multi-bounce Kirchhoff ensemble scattering chain rows with native CUDA (Op A).");
    module.def("scattering_chain_ensemble_eval_backward", &cn_scattering_chain_ensemble_eval_backward,
               "Fixed-topology VJP of the ADR-021 ensemble scattering chain (chain1/chain2 materials, tables, coef, frequency).");
    module.def("scattering_chain_ensemble_eval_jvp", &cn_scattering_chain_ensemble_eval_jvp,
               "Fixed-topology JVP of the ADR-021 ensemble scattering chain rows (Op A).");
    module.def("scattering_chain_realization_eval", &cn_scattering_chain_realization_eval,
               "Evaluate the ADR-021 coherent phase-screen scattering chain rows with native CUDA (Op B).");
    module.def("scattering_chain_realization_eval_backward", &cn_scattering_chain_realization_eval_backward,
               "Fixed-topology VJP of the ADR-021 coherent scattering chain (heights, layers, chains, geometry, k0, frequency).");
    module.def("scattering_chain_realization_eval_jvp", &cn_scattering_chain_realization_eval_jvp,
               "Fixed-topology JVP of the ADR-021 coherent scattering chain rows (Op B).");
    module.def("kirchhoff_table_build_backward", &cn_kirchhoff_table_build_backward,
               "VJP of the offline Kirchhoff table build over roughness, layers and frequency (ADR-015).");
    module.def("kirchhoff_table_build_jvp", &cn_kirchhoff_table_build_jvp,
               "JVP of the offline Kirchhoff table build over roughness, layers and frequency (ADR-015).");
}
