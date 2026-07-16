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
}
