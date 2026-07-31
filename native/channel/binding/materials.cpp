// Copyright Xingyu Chen.
// Implements materials native integration.

#include <torch/extension.h>
#include <rayd/integration.h>

#include "registry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

rayd::torch::LayerStackRequest layer_stack_request(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz) {
    return {
        std::move(cos_theta),
        std::move(material_id),
        std::move(layer_offset),
        std::move(layer_count),
        std::move(layer_thickness_m),
        std::move(layer_eps_r),
        std::move(layer_sigma_e),
        std::move(layer_mu_r),
        frequency_hz};
}

std::optional<at::Tensor> optional_tensor(pybind11::handle value) {
    if (value.is_none())
        return std::nullopt;
    return pybind11::cast<at::Tensor>(value);
}

pybind11::dict layer_stack_result_dict(
    const rayd::torch::LayerStackResult &result) {
    pybind11::dict out;
    out["r_te_real"] = result.r_te_real;
    out["r_te_imag"] = result.r_te_imag;
    out["r_tm_real"] = result.r_tm_real;
    out["r_tm_imag"] = result.r_tm_imag;
    out["t_te_real"] = result.t_te_real;
    out["t_te_imag"] = result.t_te_imag;
    out["t_tm_real"] = result.t_tm_real;
    out["t_tm_imag"] = result.t_tm_imag;
    out["cap_R_te"] = result.cap_r_te;
    out["cap_R_tm"] = result.cap_r_tm;
    out["cap_T_te"] = result.cap_t_te;
    out["cap_T_tm"] = result.cap_t_tm;
    return out;
}

}  // namespace

pybind11::dict channel_em_layer_stack_eval(
    torch::Tensor cos_theta,
    torch::Tensor material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz) {
    return layer_stack_result_dict(rayd::torch::em_layer_stack_eval(
        layer_stack_request(
            std::move(cos_theta),
            std::move(material_id),
            std::move(layer_offset),
            std::move(layer_count),
            std::move(layer_thickness_m),
            std::move(layer_eps_r),
            std::move(layer_sigma_e),
            std::move(layer_mu_r),
            frequency_hz)));
}

pybind11::dict channel_em_layer_stack_backward(
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
    bool need_frequency) {
    TORCH_CHECK(
        grad_outputs.size() == 12,
        "grad_outputs must carry the twelve stack output cotangents");
    rayd::torch::LayerStackBackwardRequest request;
    request.primal = layer_stack_request(
        std::move(cos_theta),
        std::move(material_id),
        std::move(layer_offset),
        std::move(layer_count),
        std::move(layer_thickness_m),
        std::move(layer_eps_r),
        std::move(layer_sigma_e),
        std::move(layer_mu_r),
        frequency_hz);
    for (std::size_t field = 0; field < request.grad_outputs.size(); ++field)
        request.grad_outputs[field] = optional_tensor(grad_outputs[field]);
    request.need_cos_theta = need_cos_theta;
    request.need_layers = need_layers;
    request.need_frequency = need_frequency;
    const auto result = rayd::torch::em_layer_stack_backward(request);
    pybind11::dict out;
    out["grad_cos_theta"] = result.grad_cos_theta;
    out["grad_layer_thickness_m"] = result.grad_layer_thickness_m;
    out["grad_layer_eps_r"] = result.grad_layer_eps_r;
    out["grad_layer_sigma_e"] = result.grad_layer_sigma_e;
    out["grad_frequency"] = result.grad_frequency;
    return out;
}

pybind11::dict channel_em_layer_stack_jvp(
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
    double tangent_frequency) {
    rayd::torch::LayerStackJvpRequest request;
    request.primal = layer_stack_request(
        std::move(cos_theta),
        std::move(material_id),
        std::move(layer_offset),
        std::move(layer_count),
        std::move(layer_thickness_m),
        std::move(layer_eps_r),
        std::move(layer_sigma_e),
        std::move(layer_mu_r),
        frequency_hz);
    request.tangent_cos_theta = optional_tensor(tangent_cos_theta);
    request.tangent_layer_thickness_m = optional_tensor(tangent_layer_thickness);
    request.tangent_layer_eps_r = optional_tensor(tangent_layer_eps_r);
    request.tangent_layer_sigma_e = optional_tensor(tangent_layer_sigma_e);
    request.tangent_frequency = tangent_frequency;
    return layer_stack_result_dict(rayd::torch::em_layer_stack_jvp(request));
}
namespace {

rayd::torch::ScatteringTableEvalRequest scattering_table_eval_request(
    torch::Tensor valid,
    torch::Tensor wi,
    torch::Tensor wo,
    torch::Tensor f_te,
    torch::Tensor f_tm) {
    return {
        std::move(valid), std::move(wi), std::move(wo), std::move(f_te),
        std::move(f_tm)};
}

pybind11::dict scattering_table_eval_result_dict(
    const rayd::torch::ScatteringTableEvalResult &result) {
    pybind11::dict out;
    out["f_te"] = result.f_te;
    out["f_tm"] = result.f_tm;
    return out;
}

rayd::torch::ScatteringEnsembleEvalRequest scattering_ensemble_request(
    torch::Tensor valid,
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
    double threshold) {
    rayd::torch::ScatteringEnsembleEvalRequest request;
    request.valid = std::move(valid);
    request.wo_rows = std::move(wo_rows);
    request.r2_rows = std::move(r2_rows);
    request.cos_o_rows = std::move(cos_o_rows);
    request.n_o = std::move(n_o);
    request.t1r = std::move(t1r);
    request.t2r = std::move(t2r);
    request.wi_local = std::move(wi_local);
    request.cos_i = std::move(cos_i);
    request.r1 = std::move(r1);
    request.a_te2 = std::move(a_te2);
    request.a_tm2 = std::move(a_tm2);
    request.weights = std::move(weights);
    request.material_id = std::move(material_id);
    request.backup_axis = std::move(backup_axis);
    request.rx_pol = std::move(rx_pol);
    request.rc_idx = std::move(rc_idx);
    request.sc_idx = std::move(sc_idx);
    request.f_te_flat = std::move(fte_flat);
    request.f_tm_flat = std::move(ftm_flat);
    request.table_offset = std::move(table_offset);
    request.table_dims = std::move(table_dims);
    request.material_slot = std::move(material_slot);
    request.coefficient = coef;
    request.threshold = threshold;
    return request;
}

pybind11::dict scattering_ensemble_result_dict(
    const rayd::torch::ScatteringEnsembleEvalResult &result) {
    pybind11::dict out;
    out["gain"] = result.gain;
    out["amplitude"] = result.amplitude;
    out["length"] = result.length;
    out["keep"] = result.keep;
    return out;
}

rayd::torch::ScatteringPatchIntegralEvalRequest scattering_patch_request(
    torch::Tensor valid,
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
    double k0) {
    rayd::torch::ScatteringPatchIntegralEvalRequest request;
    request.valid = std::move(valid);
    request.patch_tris = std::move(patch_tris);
    request.patch_uvs = std::move(patch_uvs);
    request.rows = std::move(rows);
    request.d_i = std::move(d_i);
    request.d_o = std::move(d_o);
    request.n_rows = std::move(n_rows);
    request.r_te = std::move(r_te);
    request.r_tm = std::move(r_tm);
    request.pol_t = std::move(pol_t);
    request.pol_r = std::move(pol_r);
    request.r1_rows = std::move(r1_rows);
    request.r2_rows = std::move(r2_rows);
    request.centroids = std::move(centroids);
    request.heights = std::move(heights);
    request.quad_a = std::move(quad_a);
    request.quad_b = std::move(quad_b);
    request.quad_w = std::move(quad_w);
    request.k0 = k0;
    return request;
}

pybind11::dict scattering_patch_result_dict(
    const rayd::torch::ScatteringPatchIntegralEvalResult &result) {
    pybind11::dict out;
    out["total"] = result.total;
    out["integral"] = result.integral;
    out["row_value"] = result.row_value;
    return out;
}

#define CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS()                                    \
    std::move(c1_positions), std::move(c1_normals),                         \
        std::move(c1_eps_r), std::move(c1_sigma_e), std::move(c1_mu_r),     \
        std::move(c1_gain), std::move(c1_thickness), std::move(c1_depth),   \
        std::move(c2_positions), std::move(c2_normals),                     \
        std::move(c2_eps_r), std::move(c2_sigma_e), std::move(c2_mu_r),     \
        std::move(c2_gain), std::move(c2_thickness), std::move(c2_depth)

rayd::torch::ScatteringChainEnsembleEvalRequest scattering_chain_ensemble_request(
    torch::Tensor valid, torch::Tensor tx_pol, torch::Tensor rx_pol,
    torch::Tensor source,
    torch::Tensor vertex, torch::Tensor target, torch::Tensor c1_positions,
    torch::Tensor c1_normals, torch::Tensor c1_eps_r, torch::Tensor c1_sigma_e,
    torch::Tensor c1_mu_r, torch::Tensor c1_gain, torch::Tensor c1_thickness,
    torch::Tensor c1_depth, torch::Tensor c2_positions, torch::Tensor c2_normals,
    torch::Tensor c2_eps_r, torch::Tensor c2_sigma_e, torch::Tensor c2_mu_r,
    torch::Tensor c2_gain, torch::Tensor c2_thickness, torch::Tensor c2_depth,
    torch::Tensor n_o, torch::Tensor t1r, torch::Tensor t2r,
    torch::Tensor backup_axis, torch::Tensor wi_local, torch::Tensor cos_i,
    torch::Tensor cos_o, torch::Tensor d_i, torch::Tensor d_o, torch::Tensor l1,
    torch::Tensor l2, torch::Tensor weights, torch::Tensor material_id,
    torch::Tensor fte_flat, torch::Tensor ftm_flat, torch::Tensor table_offset,
    torch::Tensor table_dims, torch::Tensor material_slot, double coef,
    double threshold, double frequency_hz) {
    return {
        std::move(valid), std::move(tx_pol), std::move(rx_pol), std::move(source),
        std::move(vertex), std::move(target), CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS(),
        std::move(n_o), std::move(t1r), std::move(t2r), std::move(backup_axis),
        std::move(wi_local), std::move(cos_i), std::move(cos_o), std::move(d_i),
        std::move(d_o), std::move(l1), std::move(l2), std::move(weights),
        std::move(material_id), std::move(fte_flat), std::move(ftm_flat),
        std::move(table_offset), std::move(table_dims), std::move(material_slot),
        coef, threshold, frequency_hz};
}

pybind11::dict scattering_chain_ensemble_result_dict(
    const rayd::torch::ScatteringChainEnsembleEvalResult &result) {
    pybind11::dict out;
    out["gain"] = result.gain;
    out["amplitude"] = result.amplitude;
    out["length"] = result.length;
    out["keep"] = result.keep;
    return out;
}

rayd::torch::ScatteringChainRealizationEvalRequest scattering_chain_realization_request(
    torch::Tensor valid, torch::Tensor patch_tris, torch::Tensor patch_uvs,
    torch::Tensor rows,
    torch::Tensor d_i, torch::Tensor d_o, torch::Tensor n_rows,
    torch::Tensor source, torch::Tensor vertex, torch::Tensor target,
    torch::Tensor c1_positions, torch::Tensor c1_normals, torch::Tensor c1_eps_r,
    torch::Tensor c1_sigma_e, torch::Tensor c1_mu_r, torch::Tensor c1_gain,
    torch::Tensor c1_thickness, torch::Tensor c1_depth,
    torch::Tensor c2_positions, torch::Tensor c2_normals, torch::Tensor c2_eps_r,
    torch::Tensor c2_sigma_e, torch::Tensor c2_mu_r, torch::Tensor c2_gain,
    torch::Tensor c2_thickness, torch::Tensor c2_depth, torch::Tensor tx_pol,
    torch::Tensor rx_pol, torch::Tensor l1, torch::Tensor l2, torch::Tensor sp1,
    torch::Tensor sp2, torch::Tensor centroids, torch::Tensor heights,
    torch::Tensor cos_spec, torch::Tensor material_id, torch::Tensor layer_offset,
    torch::Tensor layer_count, torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r, torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r, torch::Tensor quad_a, torch::Tensor quad_b,
    torch::Tensor quad_w, double k0, double frequency_hz) {
    return {
        std::move(valid), std::move(patch_tris), std::move(patch_uvs),
        std::move(rows),
        std::move(d_i), std::move(d_o), std::move(n_rows), std::move(source),
        std::move(vertex), std::move(target), CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS(),
        std::move(tx_pol), std::move(rx_pol), std::move(l1), std::move(l2),
        std::move(sp1), std::move(sp2), std::move(centroids), std::move(heights),
        std::move(cos_spec), std::move(material_id), std::move(layer_offset),
        std::move(layer_count), std::move(layer_thickness_m),
        std::move(layer_eps_r), std::move(layer_sigma_e), std::move(layer_mu_r),
        std::move(quad_a), std::move(quad_b), std::move(quad_w), k0,
        frequency_hz};
}

pybind11::dict scattering_chain_realization_result_dict(
    const rayd::torch::ScatteringChainRealizationEvalResult &result) {
    pybind11::dict out;
    out["total"] = result.total;
    out["path_field"] = result.path_field;
    out["path_gain"] = result.path_gain;
    out["integral"] = result.integral;
    out["row_value"] = result.row_value;
    return out;
}

}  // namespace

pybind11::dict channel_scattering_table_eval(
    torch::Tensor valid,
    torch::Tensor wi,
    torch::Tensor wo,
    torch::Tensor f_te,
    torch::Tensor f_tm) {
    return scattering_table_eval_result_dict(rayd::torch::scattering_table_eval(
        scattering_table_eval_request(
            std::move(valid), std::move(wi), std::move(wo), std::move(f_te),
            std::move(f_tm))));
}

torch::Tensor channel_scattering_table_pdf(
    torch::Tensor valid,
    torch::Tensor wi,
    torch::Tensor wo,
    torch::Tensor sample_density,
    bool reverse) {
    rayd::torch::ScatteringTablePdfRequest request;
    request.valid = std::move(valid);
    request.wi = std::move(wi);
    request.wo = std::move(wo);
    request.sample_density = std::move(sample_density);
    request.reverse = reverse;
    return rayd::torch::scattering_table_pdf(request).pdf;
}

pybind11::dict channel_scattering_table_sample(
    torch::Tensor valid,
    torch::Tensor wi,
    torch::Tensor uniforms,
    torch::Tensor marginal_cdf,
    torch::Tensor conditional_cdf,
    torch::Tensor sample_density) {
    rayd::torch::ScatteringTableSampleRequest request;
    request.valid = std::move(valid);
    request.wi = std::move(wi);
    request.uniforms = std::move(uniforms);
    request.marginal_cdf = std::move(marginal_cdf);
    request.conditional_cdf = std::move(conditional_cdf);
    request.sample_density = std::move(sample_density);
    const auto result = rayd::torch::scattering_table_sample(request);
    pybind11::dict out;
    out["wo"] = result.wo;
    out["pdf_forward"] = result.pdf_forward;
    out["pdf_reverse"] = result.pdf_reverse;
    return out;
}

pybind11::dict channel_scattering_event_probabilities(
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

#define CHANNEL_SCATTERING_ENSEMBLE_PRIMAL_ARGS()                                \
    std::move(valid), std::move(wo_rows), std::move(r2_rows),                \
        std::move(cos_o_rows),                                                \
        std::move(n_o), std::move(t1r), std::move(t2r),                     \
        std::move(wi_local), std::move(cos_i), std::move(r1),               \
        std::move(a_te2), std::move(a_tm2), std::move(weights),             \
        std::move(material_id), std::move(backup_axis), std::move(rx_pol),  \
        std::move(rc_idx), std::move(sc_idx), std::move(fte_flat),          \
        std::move(ftm_flat), std::move(table_offset),                       \
        std::move(table_dims), std::move(material_slot), coef, threshold

#define CHANNEL_SCATTERING_PATCH_PRIMAL_ARGS()                                   \
    std::move(valid), std::move(patch_tris), std::move(patch_uvs),           \
        std::move(rows),                                                      \
        std::move(d_i), std::move(d_o), std::move(n_rows),                  \
        std::move(r_te), std::move(r_tm), std::move(pol_t),                 \
        std::move(pol_r), std::move(r1_rows), std::move(r2_rows),           \
        std::move(centroids), std::move(heights), std::move(quad_a),        \
        std::move(quad_b), std::move(quad_w), k0

#define CHANNEL_SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS()                          \
    std::move(valid), std::move(tx_pol), std::move(rx_pol),                  \
        std::move(source),                                                    \
        std::move(vertex), std::move(target),                               \
        CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS(),                                   \
        std::move(n_o), std::move(t1r), std::move(t2r),                     \
        std::move(backup_axis), std::move(wi_local), std::move(cos_i),      \
        std::move(cos_o), std::move(d_i), std::move(d_o), std::move(l1),    \
        std::move(l2), std::move(weights), std::move(material_id),          \
        std::move(fte_flat), std::move(ftm_flat), std::move(table_offset),  \
        std::move(table_dims), std::move(material_slot), coef, threshold,   \
        frequency_hz

#define CHANNEL_SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS()                       \
    std::move(valid), std::move(patch_tris), std::move(patch_uvs),           \
        std::move(rows),                                                      \
        std::move(d_i), std::move(d_o), std::move(n_rows),                  \
        std::move(source), std::move(vertex), std::move(target),            \
        CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS(),                                   \
        std::move(tx_pol), std::move(rx_pol), std::move(l1), std::move(l2), \
        std::move(sp1), std::move(sp2), std::move(centroids),               \
        std::move(heights), std::move(cos_spec), std::move(material_id),    \
        std::move(layer_offset), std::move(layer_count),                    \
        std::move(layer_thickness_m), std::move(layer_eps_r),               \
        std::move(layer_sigma_e), std::move(layer_mu_r),                    \
        std::move(quad_a), std::move(quad_b), std::move(quad_w), k0,        \
        frequency_hz

pybind11::dict channel_scattering_ensemble_eval(
    torch::Tensor valid,
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
    double threshold) {
    return scattering_ensemble_result_dict(rayd::torch::scattering_ensemble_eval(
        scattering_ensemble_request(CHANNEL_SCATTERING_ENSEMBLE_PRIMAL_ARGS())));
}
pybind11::dict channel_scattering_ensemble_eval_backward(
    torch::Tensor valid,
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
    bool need_grad_coef) {
    rayd::torch::ScatteringEnsembleEvalBackwardRequest request;
    request.primal = scattering_ensemble_request(
        CHANNEL_SCATTERING_ENSEMBLE_PRIMAL_ARGS());
    request.grad_gain = optional_tensor(grad_gain);
    request.grad_amplitude = optional_tensor(grad_amplitude);
    request.grad_length = optional_tensor(grad_length);
    request.need_grad_rows = need_grad_rows;
    request.need_grad_samples = need_grad_samples;
    request.need_grad_tables = need_grad_tables;
    request.need_grad_coefficient = need_grad_coef;
    const auto result = rayd::torch::scattering_ensemble_eval_backward(request);
    pybind11::dict out;
    out["grad_wo_rows"] = result.grad_wo_rows;
    out["grad_r2_rows"] = result.grad_r2_rows;
    out["grad_cos_o_rows"] = result.grad_cos_o_rows;
    out["grad_n_o"] = result.grad_n_o;
    out["grad_t1r"] = result.grad_t1r;
    out["grad_t2r"] = result.grad_t2r;
    out["grad_wi_local"] = result.grad_wi_local;
    out["grad_cos_i"] = result.grad_cos_i;
    out["grad_r1"] = result.grad_r1;
    out["grad_a_te2"] = result.grad_a_te2;
    out["grad_a_tm2"] = result.grad_a_tm2;
    out["grad_weights"] = result.grad_weights;
    out["grad_f_te"] = result.grad_f_te;
    out["grad_f_tm"] = result.grad_f_tm;
    out["grad_coef"] = result.grad_coefficient;
    return out;
}
pybind11::dict channel_scattering_ensemble_eval_jvp(
    torch::Tensor valid,
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
    double tangent_coef) {
    rayd::torch::ScatteringEnsembleEvalJvpRequest request;
    request.primal = scattering_ensemble_request(
        CHANNEL_SCATTERING_ENSEMBLE_PRIMAL_ARGS());
    request.tangent_wo_rows = optional_tensor(t_wo_rows);
    request.tangent_r2_rows = optional_tensor(t_r2_rows);
    request.tangent_cos_o_rows = optional_tensor(t_cos_o_rows);
    request.tangent_n_o = optional_tensor(t_n_o);
    request.tangent_t1r = optional_tensor(t_t1r);
    request.tangent_t2r = optional_tensor(t_t2r);
    request.tangent_wi_local = optional_tensor(t_wi_local);
    request.tangent_cos_i = optional_tensor(t_cos_i);
    request.tangent_r1 = optional_tensor(t_r1);
    request.tangent_a_te2 = optional_tensor(t_a_te2);
    request.tangent_a_tm2 = optional_tensor(t_a_tm2);
    request.tangent_weights = optional_tensor(t_weights);
    request.tangent_f_te_flat = optional_tensor(t_fte_flat);
    request.tangent_f_tm_flat = optional_tensor(t_ftm_flat);
    request.tangent_coefficient = tangent_coef;
    const auto result = rayd::torch::scattering_ensemble_eval_jvp(request);
    pybind11::dict out;
    out["tangent_gain"] = result.tangent_gain;
    out["tangent_amplitude"] = result.tangent_amplitude;
    out["tangent_length"] = result.tangent_length;
    return out;
}

pybind11::dict channel_scattering_patch_integral_eval(
    torch::Tensor valid,
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
    double k0) {
    return scattering_patch_result_dict(
        rayd::torch::scattering_patch_integral_eval(scattering_patch_request(
            CHANNEL_SCATTERING_PATCH_PRIMAL_ARGS())));
}

pybind11::dict channel_scattering_patch_integral_eval_backward(
    torch::Tensor valid,
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
    bool need_grad_k0) {
    rayd::torch::ScatteringPatchIntegralEvalBackwardRequest request;
    request.primal = scattering_patch_request(
        CHANNEL_SCATTERING_PATCH_PRIMAL_ARGS());
    request.grad_total = std::move(grad_total);
    request.need_grad_heights = need_grad_heights;
    request.need_grad_jones = need_grad_jones;
    request.need_grad_geometry = need_grad_geometry;
    request.need_grad_k0 = need_grad_k0;
    const auto result =
        rayd::torch::scattering_patch_integral_eval_backward(request);
    pybind11::dict out;
    out["grad_heights"] = result.grad_heights;
    out["grad_r_te"] = result.grad_r_te;
    out["grad_r_tm"] = result.grad_r_tm;
    out["grad_d_i"] = result.grad_d_i;
    out["grad_d_o"] = result.grad_d_o;
    out["grad_r1_rows"] = result.grad_r1_rows;
    out["grad_r2_rows"] = result.grad_r2_rows;
    out["grad_centroids"] = result.grad_centroids;
    out["grad_k0"] = result.grad_k0;
    return out;
}
pybind11::dict channel_scattering_patch_integral_eval_jvp(
    torch::Tensor valid,
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
    double tangent_k0) {
    rayd::torch::ScatteringPatchIntegralEvalJvpRequest request;
    request.primal = scattering_patch_request(
        CHANNEL_SCATTERING_PATCH_PRIMAL_ARGS());
    request.tangent_heights = optional_tensor(t_heights);
    request.tangent_r_te = optional_tensor(t_r_te);
    request.tangent_r_tm = optional_tensor(t_r_tm);
    request.tangent_d_i = optional_tensor(t_d_i);
    request.tangent_d_o = optional_tensor(t_d_o);
    request.tangent_r1_rows = optional_tensor(t_r1_rows);
    request.tangent_r2_rows = optional_tensor(t_r2_rows);
    request.tangent_centroids = optional_tensor(t_centroids);
    request.tangent_k0 = tangent_k0;
    const auto result = rayd::torch::scattering_patch_integral_eval_jvp(request);
    pybind11::dict out;
    out["tangent_total"] = result.tangent_total;
    return out;
}
pybind11::dict channel_scattering_table_eval_backward(
    at::Tensor valid,
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object grad_out_f_te,
    pybind11::object grad_out_f_tm,
    bool need_grad_dirs,
    bool need_grad_tables) {
    rayd::torch::ScatteringTableEvalBackwardRequest request;
    request.primal = scattering_table_eval_request(
        std::move(valid), std::move(wi), std::move(wo), std::move(f_te),
        std::move(f_tm));
    request.grad_f_te = optional_tensor(grad_out_f_te);
    request.grad_f_tm = optional_tensor(grad_out_f_tm);
    request.need_grad_directions = need_grad_dirs;
    request.need_grad_tables = need_grad_tables;
    const auto result = rayd::torch::scattering_table_eval_backward(request);
    pybind11::dict out;
    out["grad_wi"] = result.grad_wi;
    out["grad_wo"] = result.grad_wo;
    out["grad_f_te"] = result.grad_f_te;
    out["grad_f_tm"] = result.grad_f_tm;
    return out;
}
pybind11::dict channel_scattering_table_eval_jvp(
    at::Tensor valid,
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object t_wi,
    pybind11::object t_wo,
    pybind11::object t_f_te,
    pybind11::object t_f_tm) {
    rayd::torch::ScatteringTableEvalJvpRequest request;
    request.primal = scattering_table_eval_request(
        std::move(valid), std::move(wi), std::move(wo), std::move(f_te),
        std::move(f_tm));
    request.tangent_wi = optional_tensor(t_wi);
    request.tangent_wo = optional_tensor(t_wo);
    request.tangent_f_te = optional_tensor(t_f_te);
    request.tangent_f_tm = optional_tensor(t_f_tm);
    const auto result = rayd::torch::scattering_table_eval_jvp(request);
    pybind11::dict out;
    out["tangent_f_te"] = result.tangent_f_te;
    out["tangent_f_tm"] = result.tangent_f_tm;
    return out;
}
pybind11::dict channel_kirchhoff_table_build_backward(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, at::Tensor grad_f_te, at::Tensor grad_f_tm,
    bool need_grad_rough, bool need_grad_layers, bool need_grad_frequency);
pybind11::dict channel_kirchhoff_table_build_jvp(
    at::Tensor s_te, at::Tensor s_tm, at::Tensor a_te, at::Tensor a_tm,
    at::Tensor r_diff_te, at::Tensor r_diff_tm, at::Tensor cos_i,
    at::Tensor phi_i, at::Tensor cos_o, at::Tensor phi_o,
    at::Tensor layer_thickness_m, at::Tensor layer_eps_r, at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r, double sigma_h, double corr_x, double corr_y,
    double frequency_hz, pybind11::object t_layer_thickness_m,
    pybind11::object t_layer_eps_r, pybind11::object t_layer_sigma_e,
    double t_sigma_h, double t_corr_x, double t_corr_y, double t_frequency);

pybind11::dict channel_scattering_chain_ensemble_eval(
    at::Tensor valid, at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source,
    at::Tensor vertex,
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
    at::Tensor material_slot, double coef, double threshold, double frequency_hz) {
    return scattering_chain_ensemble_result_dict(
        rayd::torch::scattering_chain_ensemble_eval(
            scattering_chain_ensemble_request(
                CHANNEL_SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS())));
}
pybind11::dict channel_scattering_chain_ensemble_eval_backward(
    at::Tensor valid, at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source,
    at::Tensor vertex,
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
    bool need_grad_frequency) {
    rayd::torch::ScatteringChainEnsembleEvalBackwardRequest request;
    request.primal = scattering_chain_ensemble_request(
        CHANNEL_SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS());
    request.grad_gain = optional_tensor(grad_gain);
    request.grad_amplitude = optional_tensor(grad_amplitude);
    request.grad_length = optional_tensor(grad_length);
    request.need_grad_chain1 = need_grad_chain1;
    request.need_grad_chain2 = need_grad_chain2;
    request.need_grad_tables = need_grad_tables;
    request.need_grad_geometry = need_grad_geometry;
    request.need_grad_coefficient = need_grad_coef;
    request.need_grad_frequency = need_grad_frequency;
    const auto result = rayd::torch::scattering_chain_ensemble_eval_backward(request);
    pybind11::dict out;
    out["grad_c1_eps_r"] = result.grad_c1_eps_r;
    out["grad_c1_sigma_e"] = result.grad_c1_sigma_e;
    out["grad_c1_gain"] = result.grad_c1_gain;
    out["grad_c1_thickness"] = result.grad_c1_thickness;
    out["grad_c2_eps_r"] = result.grad_c2_eps_r;
    out["grad_c2_sigma_e"] = result.grad_c2_sigma_e;
    out["grad_c2_gain"] = result.grad_c2_gain;
    out["grad_c2_thickness"] = result.grad_c2_thickness;
    out["grad_f_te"] = result.grad_f_te;
    out["grad_f_tm"] = result.grad_f_tm;
    out["grad_coef"] = result.grad_coefficient;
    out["grad_frequency"] = result.grad_frequency;
    return out;
}
pybind11::dict channel_scattering_chain_ensemble_eval_jvp(
    at::Tensor valid, at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor source,
    at::Tensor vertex,
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
    pybind11::object tangent_cos_o, double tangent_coef, double tangent_frequency) {
    rayd::torch::ScatteringChainEnsembleEvalJvpRequest request;
    request.primal = scattering_chain_ensemble_request(
        CHANNEL_SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS());
    request.tangent_c1_eps_r = optional_tensor(tangent_c1_eps_r);
    request.tangent_c1_sigma_e = optional_tensor(tangent_c1_sigma_e);
    request.tangent_c1_gain = optional_tensor(tangent_c1_gain);
    request.tangent_c1_thickness = optional_tensor(tangent_c1_thickness);
    request.tangent_c2_eps_r = optional_tensor(tangent_c2_eps_r);
    request.tangent_c2_sigma_e = optional_tensor(tangent_c2_sigma_e);
    request.tangent_c2_gain = optional_tensor(tangent_c2_gain);
    request.tangent_c2_thickness = optional_tensor(tangent_c2_thickness);
    request.tangent_f_te_flat = optional_tensor(tangent_f_te_flat);
    request.tangent_f_tm_flat = optional_tensor(tangent_f_tm_flat);
    request.tangent_c1_positions = optional_tensor(tangent_c1_positions);
    request.tangent_c1_normals = optional_tensor(tangent_c1_normals);
    request.tangent_c2_positions = optional_tensor(tangent_c2_positions);
    request.tangent_c2_normals = optional_tensor(tangent_c2_normals);
    request.tangent_d_i = optional_tensor(tangent_d_i);
    request.tangent_d_o = optional_tensor(tangent_d_o);
    request.tangent_vertex_normal = optional_tensor(tangent_v_normal);
    request.tangent_l1 = optional_tensor(tangent_l1);
    request.tangent_l2 = optional_tensor(tangent_l2);
    request.tangent_cos_i = optional_tensor(tangent_cos_i);
    request.tangent_cos_o = optional_tensor(tangent_cos_o);
    request.tangent_coefficient = tangent_coef;
    request.tangent_frequency = tangent_frequency;
    const auto result = rayd::torch::scattering_chain_ensemble_eval_jvp(request);
    pybind11::dict out;
    out["tangent_gain"] = result.tangent_gain;
    out["tangent_amplitude"] = result.tangent_amplitude;
    out["tangent_length"] = result.tangent_length;
    return out;
}

pybind11::dict channel_scattering_chain_realization_eval(
    at::Tensor valid, at::Tensor patch_tris, at::Tensor patch_uvs,
    at::Tensor rows, at::Tensor d_i,
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
    at::Tensor quad_b, at::Tensor quad_w, double k0, double frequency_hz) {
    return scattering_chain_realization_result_dict(
        rayd::torch::scattering_chain_realization_eval(
            scattering_chain_realization_request(
                CHANNEL_SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS())));
}
pybind11::dict channel_scattering_chain_realization_eval_backward(
    at::Tensor valid, at::Tensor patch_tris, at::Tensor patch_uvs,
    at::Tensor rows, at::Tensor d_i,
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
    bool need_grad_k0, bool need_grad_frequency) {
    rayd::torch::ScatteringChainRealizationEvalBackwardRequest request;
    request.primal = scattering_chain_realization_request(
        CHANNEL_SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS());
    request.grad_total = std::move(grad_total);
    request.grad_path_field = optional_tensor(grad_path_field);
    request.grad_path_gain = optional_tensor(grad_path_gain);
    request.need_grad_heights = need_grad_heights;
    request.need_grad_layers = need_grad_layers;
    request.need_grad_chain1 = need_grad_chain1;
    request.need_grad_chain2 = need_grad_chain2;
    request.need_grad_geometry = need_grad_geometry;
    request.need_grad_k0 = need_grad_k0;
    request.need_grad_frequency = need_grad_frequency;
    const auto result = rayd::torch::scattering_chain_realization_eval_backward(request);
    pybind11::dict out;
    out["grad_heights"] = result.grad_heights;
    out["grad_layer_thickness"] = result.grad_layer_thickness;
    out["grad_layer_eps_r"] = result.grad_layer_eps_r;
    out["grad_layer_sigma_e"] = result.grad_layer_sigma_e;
    out["grad_c1_eps_r"] = result.grad_c1_eps_r;
    out["grad_c1_sigma_e"] = result.grad_c1_sigma_e;
    out["grad_c1_gain"] = result.grad_c1_gain;
    out["grad_c1_thickness"] = result.grad_c1_thickness;
    out["grad_c2_eps_r"] = result.grad_c2_eps_r;
    out["grad_c2_sigma_e"] = result.grad_c2_sigma_e;
    out["grad_c2_gain"] = result.grad_c2_gain;
    out["grad_c2_thickness"] = result.grad_c2_thickness;
    out["grad_d_i"] = result.grad_d_i;
    out["grad_d_o"] = result.grad_d_o;
    out["grad_c1_positions"] = result.grad_c1_positions;
    out["grad_c1_normals"] = result.grad_c1_normals;
    out["grad_c2_positions"] = result.grad_c2_positions;
    out["grad_c2_normals"] = result.grad_c2_normals;
    out["grad_L1"] = result.grad_l1;
    out["grad_L2"] = result.grad_l2;
    out["grad_sp1"] = result.grad_sp1;
    out["grad_sp2"] = result.grad_sp2;
    out["grad_centroids"] = result.grad_centroids;
    out["grad_k0"] = result.grad_k0;
    out["grad_frequency"] = result.grad_frequency;
    return out;
}
pybind11::dict channel_scattering_chain_realization_eval_jvp(
    at::Tensor valid, at::Tensor patch_tris, at::Tensor patch_uvs,
    at::Tensor rows, at::Tensor d_i,
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
    double tangent_frequency) {
    rayd::torch::ScatteringChainRealizationEvalJvpRequest request;
    request.primal = scattering_chain_realization_request(
        CHANNEL_SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS());
    request.tangent_heights = optional_tensor(tangent_heights);
    request.tangent_layer_thickness = optional_tensor(tangent_layer_thickness);
    request.tangent_layer_eps_r = optional_tensor(tangent_layer_eps_r);
    request.tangent_layer_sigma_e = optional_tensor(tangent_layer_sigma_e);
    request.tangent_c1_eps_r = optional_tensor(tangent_c1_eps_r);
    request.tangent_c1_sigma_e = optional_tensor(tangent_c1_sigma_e);
    request.tangent_c1_gain = optional_tensor(tangent_c1_gain);
    request.tangent_c1_thickness = optional_tensor(tangent_c1_thickness);
    request.tangent_c2_eps_r = optional_tensor(tangent_c2_eps_r);
    request.tangent_c2_sigma_e = optional_tensor(tangent_c2_sigma_e);
    request.tangent_c2_gain = optional_tensor(tangent_c2_gain);
    request.tangent_c2_thickness = optional_tensor(tangent_c2_thickness);
    request.tangent_d_i = optional_tensor(tangent_d_i);
    request.tangent_d_o = optional_tensor(tangent_d_o);
    request.tangent_c1_positions = optional_tensor(tangent_c1_positions);
    request.tangent_c1_normals = optional_tensor(tangent_c1_normals);
    request.tangent_c2_positions = optional_tensor(tangent_c2_positions);
    request.tangent_c2_normals = optional_tensor(tangent_c2_normals);
    request.tangent_l1 = optional_tensor(tangent_l1);
    request.tangent_l2 = optional_tensor(tangent_l2);
    request.tangent_sp1 = optional_tensor(tangent_sp1);
    request.tangent_sp2 = optional_tensor(tangent_sp2);
    request.tangent_centroids = optional_tensor(tangent_centroids);
    request.tangent_k0 = tangent_k0;
    request.tangent_frequency = tangent_frequency;
    const auto result = rayd::torch::scattering_chain_realization_eval_jvp(request);
    pybind11::dict out;
    out["tangent_total"] = result.tangent_total;
    out["tangent_path_field"] = result.tangent_path_field;
    out["tangent_path_gain"] = result.tangent_path_gain;
    return out;
}

#undef CHANNEL_SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS
#undef CHANNEL_SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS
#undef CHANNEL_SCATTERING_PATCH_PRIMAL_ARGS
#undef CHANNEL_SCATTERING_ENSEMBLE_PRIMAL_ARGS
#undef CHANNEL_SCATTERING_CHAIN_MEDIA_ARGS

void register_materials(pybind11::module_ &module) {
    module.def(
        "em_layer_stack_eval",
        &channel_em_layer_stack_eval,
        "Evaluate shared em/ layer-stack reflection/transmission coefficients for parity tests.");
    module.def(
        "em_layer_stack_backward",
        &channel_em_layer_stack_backward,
        "Layer-stack VJP over cos_theta, CSR layer parameters and frequency (CUDA duals).");
    module.def(
        "em_layer_stack_jvp",
        &channel_em_layer_stack_jvp,
        "Layer-stack JVP over cos_theta, CSR layer parameters and frequency (CUDA duals).");
    module.def("scattering_table_eval", &channel_scattering_table_eval,
               "Evaluate a resident Kirchhoff BSDF table with native CUDA.");
    module.def("scattering_table_pdf", &channel_scattering_table_pdf,
               "Evaluate a resident Kirchhoff sampling PDF with native CUDA.");
    module.def("scattering_table_sample", &channel_scattering_table_sample,
               "Sample a resident Kirchhoff CDF table with native CUDA.");
    module.def("scattering_event_probabilities", &channel_scattering_event_probabilities,
               "Evaluate fused rough-surface event budgets with native CUDA.");
    module.def("scattering_ensemble_eval", &channel_scattering_ensemble_eval,
               "Evaluate the Kirchhoff ensemble scattering row physics with native CUDA (ADR-010 op 1).");
    module.def("scattering_patch_integral_eval", &channel_scattering_patch_integral_eval,
               "Evaluate the realization-coherent phase-screen patch integral with native CUDA (ADR-010 op 2).");
    module.def("scattering_ensemble_eval_backward", &channel_scattering_ensemble_eval_backward,
               "Fixed-topology VJP of the Kirchhoff ensemble scattering rows (rows, samples, tables, coef) (ADR-014).");
    module.def("scattering_ensemble_eval_jvp", &channel_scattering_ensemble_eval_jvp,
               "Fixed-topology JVP of the Kirchhoff ensemble scattering rows (rows, samples, tables, coef) (ADR-014).");
    module.def("scattering_patch_integral_eval_backward", &channel_scattering_patch_integral_eval_backward,
               "Fixed-topology VJP of the realization-coherent phase-screen patch integral (heights, jones, geometry, k0) (ADR-014).");
    module.def("scattering_patch_integral_eval_jvp", &channel_scattering_patch_integral_eval_jvp,
               "Fixed-topology JVP of the realization-coherent phase-screen patch integral (heights, jones, geometry, k0) (ADR-014).");
    module.def("scattering_table_eval_backward", &channel_scattering_table_eval_backward,
               "Fixed-topology VJP of the resident Kirchhoff BSDF table lookup (directions, tables) (ADR-015).");
    module.def("scattering_table_eval_jvp", &channel_scattering_table_eval_jvp,
               "Fixed-topology JVP of the resident Kirchhoff BSDF table lookup (directions, tables) (ADR-015).");
    module.def("scattering_chain_ensemble_eval", &channel_scattering_chain_ensemble_eval,
               "Evaluate the ADR-021 multi-bounce Kirchhoff ensemble scattering chain rows with native CUDA (Op A).");
    module.def("scattering_chain_ensemble_eval_backward", &channel_scattering_chain_ensemble_eval_backward,
               "Fixed-topology VJP of the ADR-021 ensemble scattering chain (chain1/chain2 materials, tables, coef, frequency).");
    module.def("scattering_chain_ensemble_eval_jvp", &channel_scattering_chain_ensemble_eval_jvp,
               "Fixed-topology JVP of the ADR-021 ensemble scattering chain rows (Op A).");
    module.def("scattering_chain_realization_eval", &channel_scattering_chain_realization_eval,
               "Evaluate the ADR-021 coherent phase-screen scattering chain rows with native CUDA (Op B).");
    module.def("scattering_chain_realization_eval_backward", &channel_scattering_chain_realization_eval_backward,
               "Fixed-topology VJP of the ADR-021 coherent scattering chain (heights, layers, chains, geometry, k0, frequency).");
    module.def("scattering_chain_realization_eval_jvp", &channel_scattering_chain_realization_eval_jvp,
               "Fixed-topology JVP of the ADR-021 coherent scattering chain rows (Op B).");
    module.def("kirchhoff_table_build_backward", &channel_kirchhoff_table_build_backward,
               "VJP of the offline Kirchhoff table build over roughness, layers and frequency (ADR-015).");
    module.def("kirchhoff_table_build_jvp", &channel_kirchhoff_table_build_jvp,
               "JVP of the offline Kirchhoff table build over roughness, layers and frequency (ADR-015).");
}
