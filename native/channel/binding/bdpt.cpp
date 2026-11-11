// Copyright Xingyu Chen.
// Implements bdpt native integration.

#include <torch/extension.h>
#include <pybind11/stl.h>

#include "../rayd/resource.h"
#include "registry.h"

#include <cstdint>
#include <string>
#include <vector>

pybind11::dict channel_bdpt_launch_state(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed);
pybind11::dict channel_bdpt_endpoint_subpath_state(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    torch::Tensor launch_tx_id,
    torch::Tensor light_seed);
pybind11::dict channel_bdpt_endpoint_connection_samples(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths);
pybind11::dict channel_bdpt_endpoint_connection_visibility_inputs(
    pybind11::dict light,
    pybind11::dict sensor,
    int64_t sample_count);
pybind11::dict channel_bdpt_accumulate_connection_samples(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy,
    int64_t combine_domain,
    torch::Tensor coeff_real,
    torch::Tensor coeff_imag);
pybind11::dict channel_bdpt_filter_connection_samples(pybind11::dict samples, torch::Tensor visible);
torch::Tensor channel_bdpt_count_valid_connection_samples(pybind11::dict samples);
pybind11::dict channel_bdpt_compact_connection_samples(pybind11::dict samples, int64_t max_paths);
pybind11::dict channel_bdpt_concat_connection_samples(pybind11::sequence samples);
torch::Tensor channel_bdpt_connection_variance(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx);
pybind11::dict channel_bdpt_wavefront_transmission(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    pybind11::object scatter,
    torch::Tensor base_transmission,
    torch::Tensor base_scattering,
    double frequency_hz,
    int64_t samples,
    int64_t max_depth,
    int64_t seed,
    int64_t mode_id,
    double beta,
    bool emit_mixed,
    int64_t max_scattering_order,
    double scene_diagonal,
    int64_t chunk_samples,
    int64_t pair_tile,
    int64_t rows_capacity);
pybind11::dict channel_bdpt_wavefront_jvp(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    pybind11::object scatter,
    torch::Tensor base_transmission,
    torch::Tensor base_scattering,
    double frequency_hz,
    int64_t samples,
    int64_t max_depth,
    int64_t seed,
    int64_t mode_id,
    double beta,
    bool emit_mixed,
    int64_t max_scattering_order,
    double scene_diagonal,
    int64_t chunk_samples,
    int64_t pair_tile,
    pybind11::dict tangents);
pybind11::dict channel_bdpt_wavefront_backward(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    pybind11::object scatter,
    torch::Tensor grad_transmission,
    torch::Tensor grad_scattering,
    double frequency_hz,
    int64_t samples,
    int64_t max_depth,
    int64_t seed,
    int64_t mode_id,
    double beta,
    bool emit_mixed,
    int64_t max_scattering_order,
    double scene_diagonal,
    int64_t chunk_samples,
    int64_t pair_tile,
    std::vector<std::string> needs);
torch::Tensor channel_bdpt_zero_matrix(torch::Tensor reference, int64_t rows, int64_t cols);
pybind11::dict channel_bdpt_finalize_point_components(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
pybind11::dict channel_bdpt_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
pybind11::dict channel_bdpt_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers);
torch::Tensor channel_bdpt_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols);
torch::Tensor channel_bdpt_receiver_grid_points(
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

pybind11::dict channel_bdpt_endpoint_connection_samples_backward_dispatch(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths,
    torch::Tensor grad_contribution,
    bool need_grad_field,
    bool need_grad_frequency,
    bool need_grad_tx_power);
pybind11::dict channel_bdpt_endpoint_connection_samples_jvp_dispatch(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_sensor_field_real,
    pybind11::object tangent_sensor_field_imag,
    double tangent_frequency,
    pybind11::object tangent_tx_power);
pybind11::dict channel_bdpt_accumulate_connection_samples_backward(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object grad_path_gain,
    pybind11::object grad_los,
    pybind11::object grad_reflection,
    pybind11::object grad_diffraction,
    pybind11::object grad_transmission,
    pybind11::object grad_scattering,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im,
    bool need_grad_contribution,
    bool need_grad_coeff);
pybind11::dict channel_bdpt_accumulate_connection_samples_jvp(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object tangent_contribution,
    pybind11::object tangent_coeff_real,
    pybind11::object tangent_coeff_imag,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im);
pybind11::dict channel_bdpt_finalize_point_components_backward_dispatch(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components);
pybind11::dict channel_bdpt_finalize_point_components_jvp_dispatch(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering);
pybind11::dict channel_bdpt_finalize_component_maps_backward_dispatch(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components);
pybind11::dict channel_bdpt_finalize_component_maps_jvp_dispatch(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering);

void register_bdpt_subpaths(pybind11::module_ &module) {
    module.def(
        "bdpt_launch_state",
        &channel_bdpt_launch_state,
        "Generate deterministic BDPT launch-state tensors.");
    module.def(
        "bdpt_endpoint_subpath_state",
        &channel_bdpt_endpoint_subpath_state,
        "Generate BDPT light and sensor endpoint subpaths with native CUDA.");
}

void register_bdpt_connections(pybind11::module_ &module) {
    module.def(
        "bdpt_endpoint_connection_samples",
        &channel_bdpt_endpoint_connection_samples,
        "Connect BDPT endpoint subpaths and emit native connection samples.");
    module.def(
        "bdpt_endpoint_connection_visibility_inputs",
        &channel_bdpt_endpoint_connection_visibility_inputs,
        "Build RayD visibility inputs for BDPT endpoint connection samples.");
    module.def(
        "bdpt_accumulate_connection_samples",
        &channel_bdpt_accumulate_connection_samples,
        "Accumulate BDPT connection samples into component matrices.",
        pybind11::arg("samples"),
        pybind11::arg("tx_count"),
        pybind11::arg("rx_count"),
        pybind11::arg("accumulation_strategy"),
        pybind11::arg("combine_domain") = 0,
        pybind11::arg("coeff_real") = torch::Tensor(),
        pybind11::arg("coeff_imag") = torch::Tensor());
    module.def(
        "bdpt_endpoint_connection_samples_backward",
        &channel_bdpt_endpoint_connection_samples_backward_dispatch,
        "ADR-022 VJP of BDPT endpoint connection samples (native CUDA companion).");
    module.def(
        "bdpt_endpoint_connection_samples_jvp",
        &channel_bdpt_endpoint_connection_samples_jvp_dispatch,
        "ADR-022 JVP of BDPT endpoint connection samples (native CUDA companion).");
    module.def(
        "bdpt_accumulate_connection_samples_backward",
        &channel_bdpt_accumulate_connection_samples_backward,
        "ADR-022 VJP of BDPT connection-sample accumulation (native CUDA companion).");
    module.def(
        "bdpt_accumulate_connection_samples_jvp",
        &channel_bdpt_accumulate_connection_samples_jvp,
        "ADR-022 JVP of BDPT connection-sample accumulation (native CUDA companion).");
    module.def(
        "bdpt_filter_connection_samples",
        &channel_bdpt_filter_connection_samples,
        "Apply a native visibility mask to BDPT connection samples.");
    module.def(
        "bdpt_count_valid_connection_samples",
        &channel_bdpt_count_valid_connection_samples,
        "Count valid BDPT connection samples with CUDA.");
    module.def(
        "bdpt_compact_connection_samples",
        &channel_bdpt_compact_connection_samples,
        "Compact valid BDPT connection samples for export.");
    module.def(
        "bdpt_concat_connection_samples",
        &channel_bdpt_concat_connection_samples,
        "Concatenate BDPT connection sample blocks with CUDA.");
    module.def(
        "bdpt_connection_variance",
        &channel_bdpt_connection_variance,
        "Compute BDPT connection-sample first/second-moment variance.");
    module.def(
        "bdpt_zero_matrix",
        &channel_bdpt_zero_matrix,
        "Allocate and zero a BDPT float matrix with a CUDA kernel.");
}

void register_bdpt_components(pybind11::module_ &module) {
    module.def(
        "bdpt_finalize_point_components",
        &channel_bdpt_finalize_point_components,
        "Fuse BDPT point-receiver component matrices and reduce component powers.");
    module.def(
        "bdpt_finalize_component_maps",
        &channel_bdpt_finalize_component_maps,
        "Fuse BDPT component maps and component power reductions.");
    module.def(
        "bdpt_finalize_point_components_backward",
        &channel_bdpt_finalize_point_components_backward_dispatch,
        "ADR-022 VJP of the BDPT point-receiver finalize map (native CUDA companion).");
    module.def(
        "bdpt_finalize_point_components_jvp",
        &channel_bdpt_finalize_point_components_jvp_dispatch,
        "ADR-022 JVP of the BDPT point-receiver finalize map (native CUDA companion).");
    module.def(
        "bdpt_finalize_component_maps_backward",
        &channel_bdpt_finalize_component_maps_backward_dispatch,
        "ADR-022 VJP of the BDPT component-map finalize (native CUDA companion).");
    module.def(
        "bdpt_finalize_component_maps_jvp",
        &channel_bdpt_finalize_component_maps_jvp_dispatch,
        "ADR-022 JVP of the BDPT component-map finalize (native CUDA companion).");
    module.def(
        "bdpt_wavefront_transmission",
        &channel_bdpt_wavefront_transmission,
        "Run the fused all-transmitter BDPT wavefront: mixed chains and rough-face scatter connections.");
    module.def(
        "bdpt_wavefront_jvp",
        &channel_bdpt_wavefront_jvp,
        "Run the BDPT wavefront with its forward-mode tangents in one pass.");
    module.def(
        "bdpt_wavefront_backward",
        &channel_bdpt_wavefront_backward,
        "Replay the BDPT wavefront and return the reverse-mode leaf gradients.");
    module.def(
        "bdpt_transmitter_tensors",
        &channel_bdpt_transmitter_tensors,
        "Create BDPT transmitter/host position CUDA tensors through native code.");
    module.def(
        "bdpt_los_component_maps_from_matrix",
        &channel_bdpt_los_component_maps_from_matrix,
        "Convert a BDPT LoS path-gain matrix to public component-map grid layout with a CUDA kernel.");
}

void register_bdpt_diffraction_support(pybind11::module_ &module) {
    module.def(
        "bdpt_receiver_grid_points",
        &channel_bdpt_receiver_grid_points,
        "Generate BDPT receiver grid points with a CUDA kernel.");
}

void register_bdpt_material_helpers(pybind11::module_ &module) {
}
