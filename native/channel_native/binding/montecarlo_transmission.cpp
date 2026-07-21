#include <torch/extension.h>

#include "registry.h"

#include <cstdint>
#include <optional>

pybind11::dict cn_mc_capacity_failure_component_maps_sanitize(
    torch::Tensor failure_state,
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
pybind11::dict cn_mc_capacity_failure_component_maps_sanitize_backward(
    torch::Tensor failure_state,
    torch::Tensor reference,
    std::optional<torch::Tensor> grad_los,
    std::optional<torch::Tensor> grad_reflection,
    std::optional<torch::Tensor> grad_diffraction,
    std::optional<torch::Tensor> grad_transmission,
    std::optional<torch::Tensor> grad_scattering);
pybind11::dict cn_mc_capacity_failure_component_maps_sanitize_jvp(
    torch::Tensor failure_state,
    torch::Tensor reference,
    std::optional<torch::Tensor> tangent_los,
    std::optional<torch::Tensor> tangent_reflection,
    std::optional<torch::Tensor> tangent_diffraction,
    std::optional<torch::Tensor> tangent_transmission,
    std::optional<torch::Tensor> tangent_scattering);

pybind11::dict cn_mc_transmission_wall_product(
    torch::Tensor valid, torch::Tensor num_hits, torch::Tensor reached_target,
    torch::Tensor direction, torch::Tensor normal,
    torch::Tensor global_primitive_id, torch::Tensor face_material_id,
    torch::Tensor geometry_mode_id, torch::Tensor layer_offset,
    torch::Tensor layer_count, torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r, torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r, torch::Tensor pair_polarization,
    torch::Tensor base_power, double frequency_hz,
    torch::Tensor capacity_failure_state, int64_t failure_bit);
pybind11::tuple cn_mc_transmission_wall_product_backward(
    torch::Tensor valid, torch::Tensor num_hits, torch::Tensor reached_target,
    torch::Tensor direction, torch::Tensor normal,
    torch::Tensor global_primitive_id, torch::Tensor face_material_id,
    torch::Tensor geometry_mode_id, torch::Tensor layer_offset,
    torch::Tensor layer_count, torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r, torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r, torch::Tensor pair_polarization,
    torch::Tensor base_power, double frequency_hz,
    torch::Tensor capacity_failure_state, int64_t failure_bit,
    pybind11::object grad_scaled_power,
    pybind11::object grad_transmittance);
pybind11::dict cn_mc_transmission_wall_product_jvp(
    torch::Tensor valid, torch::Tensor num_hits, torch::Tensor reached_target,
    torch::Tensor direction, torch::Tensor normal,
    torch::Tensor global_primitive_id, torch::Tensor face_material_id,
    torch::Tensor geometry_mode_id, torch::Tensor layer_offset,
    torch::Tensor layer_count, torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r, torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r, torch::Tensor pair_polarization,
    torch::Tensor base_power, double frequency_hz,
    torch::Tensor capacity_failure_state, int64_t failure_bit,
    pybind11::object tangent_direction, pybind11::object tangent_normal,
    pybind11::object tangent_layer_thickness_m,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    pybind11::object tangent_base_power,
    double tangent_frequency);

void register_montecarlo_transmission(pybind11::module_ &module) {
    module.def(
        "mc_capacity_failure_component_maps_sanitize",
        &cn_mc_capacity_failure_component_maps_sanitize,
        "Sanitize all MC Basic component maps through one capacity failure state.");
    module.def(
        "mc_capacity_failure_component_maps_sanitize_backward",
        &cn_mc_capacity_failure_component_maps_sanitize_backward,
        "Apply the MC Basic capacity-map sanitizer adjoint on CUDA.");
    module.def(
        "mc_capacity_failure_component_maps_sanitize_jvp",
        &cn_mc_capacity_failure_component_maps_sanitize_jvp,
        "Apply the MC Basic capacity-map sanitizer pushforward on CUDA.");
    module.def(
        "mc_transmission_wall_product",
        &cn_mc_transmission_wall_product,
        "Evaluate fixed-capacity straight-penetration wall products on CUDA.");
    module.def(
        "mc_transmission_wall_product_backward",
        &cn_mc_transmission_wall_product_backward,
        "Evaluate the fixed-topology wall-product VJP on CUDA.");
    module.def(
        "mc_transmission_wall_product_jvp",
        &cn_mc_transmission_wall_product_jvp,
        "Evaluate the fixed-topology wall-product JVP on CUDA.");
}
