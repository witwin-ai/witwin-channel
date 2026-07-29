// Copyright Xingyu Chen.
// Implements bdpt native integration.

#include <torch/extension.h>

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
pybind11::dict channel_bdpt_empty_subpath_state(torch::Tensor reference);
pybind11::dict channel_bdpt_endpoint_subpath_state(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    torch::Tensor launch_tx_id,
    torch::Tensor light_seed);
pybind11::dict channel_bdpt_subpath_intersection_inputs(pybind11::dict subpath);
pybind11::dict channel_bdpt_reflected_light_subpath_state(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor material_thickness,
    double frequency_hz);
pybind11::dict channel_bdpt_transmitted_light_subpath_state(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor face_material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz);
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
int64_t channel_bdpt_count_valid_connection_samples(pybind11::dict samples);
pybind11::dict channel_bdpt_compact_connection_samples(pybind11::dict samples, int64_t max_paths);
pybind11::dict channel_bdpt_concat_connection_samples(pybind11::sequence samples);
torch::Tensor channel_bdpt_connection_variance(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx);
torch::Tensor channel_bdpt_sample_directions(int64_t count, torch::Tensor reference, int64_t seed);
torch::Tensor channel_bdpt_mis_weights(
    torch::Tensor pdf,
    torch::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta);
torch::Tensor channel_bdpt_zero_matrix(torch::Tensor reference, int64_t rows, int64_t cols);
pybind11::dict channel_bdpt_point_component_power(torch::Tensor path_gain, bool include_los);
torch::Tensor channel_bdpt_store_point_component_column(
    torch::Tensor target,
    torch::Tensor source,
    int64_t rx_index);
pybind11::dict channel_bdpt_finalize_point_components(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
pybind11::dict channel_bdpt_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz,
    torch::Tensor tx_pol);
pybind11::dict channel_bdpt_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering);
torch::Tensor channel_bdpt_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
torch::Tensor channel_bdpt_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index);
torch::Tensor channel_bdpt_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
pybind11::dict channel_bdpt_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers);
torch::Tensor channel_bdpt_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z);
torch::Tensor channel_bdpt_los_component_maps(torch::Tensor los);
torch::Tensor channel_bdpt_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols);
torch::Tensor channel_bdpt_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index);
pybind11::dict channel_bdpt_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
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
pybind11::dict channel_bdpt_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
pybind11::dict channel_bdpt_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id);
pybind11::dict channel_bdpt_face_material_tensors_from_host(
    pybind11::sequence material_eps_r,
    pybind11::sequence material_sigma_e,
    pybind11::sequence material_mu_r,
    pybind11::sequence face_material_id);

// BDPT AD companions. The dispatch wrappers unpack the
// subpath/intersection/light/sensor dicts; the accumulate pair unpacks the
// connection-sample dict directly. All are defined in bdpt.cpp.
pybind11::dict channel_bdpt_reflected_light_subpath_state_backward_dispatch(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor material_thickness,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_material,
    bool need_grad_field_in,
    bool need_grad_frequency);
pybind11::dict channel_bdpt_reflected_light_subpath_state_jvp_dispatch(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor material_thickness,
    double frequency_hz,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_light_throughput_real,
    pybind11::object tangent_light_throughput_imag);
pybind11::dict channel_bdpt_transmitted_light_subpath_state_backward_dispatch(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor face_material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_layers,
    bool need_grad_field_in,
    bool need_grad_frequency);
pybind11::dict channel_bdpt_transmitted_light_subpath_state_jvp_dispatch(
    pybind11::dict light,
    pybind11::dict intersection,
    torch::Tensor face_material_id,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_light_throughput_real,
    pybind11::object tangent_light_throughput_imag);
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
        "bdpt_empty_subpath_state",
        &channel_bdpt_empty_subpath_state,
        "Allocate the empty BDPT subpath-state tensor schema with native CUDA storage.");
    module.def(
        "bdpt_endpoint_subpath_state",
        &channel_bdpt_endpoint_subpath_state,
        "Generate BDPT light and sensor endpoint subpaths with native CUDA.");
    module.def(
        "bdpt_subpath_intersection_inputs",
        &channel_bdpt_subpath_intersection_inputs,
        "Expose BDPT subpath state as RayD intersection ray inputs with native tensor storage.");
    module.def(
        "bdpt_reflected_light_subpath_state",
        &channel_bdpt_reflected_light_subpath_state,
        "Propagate BDPT light subpaths through native RayD hit geometry with CUDA reflection.");
    module.def(
        "bdpt_transmitted_light_subpath_state",
        &channel_bdpt_transmitted_light_subpath_state,
        "Continue BDPT light subpaths through thin_sheet walls with CUDA layer-stack transmission.");
    module.def(
        "bdpt_reflected_light_subpath_state_backward",
        &channel_bdpt_reflected_light_subpath_state_backward_dispatch,
        "ADR-022 VJP of BDPT reflected light subpath advance (native CUDA companion).");
    module.def(
        "bdpt_reflected_light_subpath_state_jvp",
        &channel_bdpt_reflected_light_subpath_state_jvp_dispatch,
        "ADR-022 JVP of BDPT reflected light subpath advance (native CUDA companion).");
    module.def(
        "bdpt_transmitted_light_subpath_state_backward",
        &channel_bdpt_transmitted_light_subpath_state_backward_dispatch,
        "ADR-022 VJP of BDPT transmitted light subpath advance (native CUDA companion).");
    module.def(
        "bdpt_transmitted_light_subpath_state_jvp",
        &channel_bdpt_transmitted_light_subpath_state_jvp_dispatch,
        "ADR-022 JVP of BDPT transmitted light subpath advance (native CUDA companion).");
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
        "bdpt_mis_weights",
        &channel_bdpt_mis_weights,
        "Evaluate BDPT MIS weights with a CUDA kernel.");
    module.def(
        "bdpt_zero_matrix",
        &channel_bdpt_zero_matrix,
        "Allocate and zero a BDPT float matrix with a CUDA kernel.");
}

void register_bdpt_components(pybind11::module_ &module) {
    module.def(
        "bdpt_point_component_power",
        &channel_bdpt_point_component_power,
        "Reduce point-receiver BDPT component powers with CUDA kernels.");
    module.def(
        "bdpt_store_point_component_column",
        &channel_bdpt_store_point_component_column,
        "Store a one-cell BDPT component map into a point-receiver component matrix.");
    module.def(
        "bdpt_finalize_point_components",
        &channel_bdpt_finalize_point_components,
        "Fuse BDPT point-receiver component matrices and reduce component powers.");
    module.def(
        "bdpt_los_export",
        &channel_bdpt_los_export,
        "Export BDPT direct LoS connections from CUDA tensors.");
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
        "bdpt_component_map_buffer",
        &channel_bdpt_component_map_buffer,
        "Allocate a zero-filled BDPT component map buffer.");
    module.def(
        "bdpt_store_component_map",
        &channel_bdpt_store_component_map,
        "Store one BDPT component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "bdpt_store_scaled_component_map",
        &channel_bdpt_store_scaled_component_map,
        "Store one scaled BDPT component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "bdpt_sample_directions",
        &channel_bdpt_sample_directions,
        "Generate seeded BDPT sphere sample directions with a fused CUDA kernel.");
    module.def(
        "bdpt_transmitter_tensors",
        &channel_bdpt_transmitter_tensors,
        "Create BDPT transmitter/host position CUDA tensors through native code.");
    module.def(
        "bdpt_pack_vec3",
        &channel_bdpt_pack_vec3,
        "Pack three CUDA float vectors into an interleaved BDPT vec3 tensor.");
    module.def(
        "bdpt_los_component_maps",
        &channel_bdpt_los_component_maps,
        "Convert BDPT LoS gains to public component-map grid layout with a CUDA kernel.");
    module.def(
        "bdpt_los_component_maps_from_matrix",
        &channel_bdpt_los_component_maps_from_matrix,
        "Convert a BDPT LoS path-gain matrix to public component-map grid layout with a CUDA kernel.");
    module.def(
        "bdpt_apply_los_visibility",
        &channel_bdpt_apply_los_visibility,
        "Apply RayD LoS visibility to one BDPT transmitter component map with a CUDA kernel.");
    module.def(
        "bdpt_los_visibility_inputs",
        &channel_bdpt_los_visibility_inputs,
        "Prepare RayD LoS visibility start and active tensors for BDPT with a CUDA kernel.");
}

void register_bdpt_diffraction_support(pybind11::module_ &module) {
    module.def(
        "bdpt_receiver_grid_points",
        &channel_bdpt_receiver_grid_points,
        "Generate BDPT receiver grid points with a CUDA kernel.");
    module.def(
        "bdpt_reflection_launch_inputs",
        &channel_bdpt_reflection_launch_inputs,
        "Prepare RayD reflection launch tensors for BDPT with a CUDA kernel.");
}

void register_bdpt_material_helpers(pybind11::module_ &module) {
    module.def(
        "bdpt_face_material_tensors",
        &channel_bdpt_face_material_tensors,
        "Expand BDPT material parameters to per-face tensors with a fused CUDA kernel.");
    module.def(
        "bdpt_face_material_tensors_from_host",
        &channel_bdpt_face_material_tensors_from_host,
        "Create and expand BDPT material parameters from host scalar inputs with native CUDA.");
}
