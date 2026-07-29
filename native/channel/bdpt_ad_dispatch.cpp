// Copyright Xingyu Chen.
// Implements bdpt ad dispatch native integration.

#include <torch/extension.h>

#include <utility>

#include "bdpt_dict_helpers.h"

// These wrappers unpack BDPT subpath and connection dictionaries and dispatch
// to the flat AD kernels. Primal wrappers and pybind registration stay in
// bdpt.cpp and binding/bdpt.cpp. Both bridges share tensor_from_dict through
// bdpt_dict_helpers.h.

// Flat AD-kernel declarations consumed only by the accumulate wrappers below.
pybind11::dict channel_bdpt_accumulate_connection_samples_backward_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
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
pybind11::dict channel_bdpt_accumulate_connection_samples_jvp_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
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
    bool need_grad_coeff) {
    return channel_bdpt_accumulate_connection_samples_backward_cuda(
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        tx_count,
        rx_count,
        combine_domain,
        std::move(grad_path_gain),
        std::move(grad_los),
        std::move(grad_reflection),
        std::move(grad_diffraction),
        std::move(grad_transmission),
        std::move(grad_scattering),
        std::move(los_re),
        std::move(los_im),
        std::move(reflection_re),
        std::move(reflection_im),
        std::move(diffraction_re),
        std::move(diffraction_im),
        std::move(transmission_re),
        std::move(transmission_im),
        std::move(scattering_re),
        std::move(scattering_im),
        need_grad_contribution,
        need_grad_coeff);
}

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
    pybind11::object scattering_im) {
    return channel_bdpt_accumulate_connection_samples_jvp_cuda(
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        tx_count,
        rx_count,
        combine_domain,
        std::move(tangent_contribution),
        std::move(tangent_coeff_real),
        std::move(tangent_coeff_imag),
        std::move(los_re),
        std::move(los_im),
        std::move(reflection_re),
        std::move(reflection_im),
        std::move(diffraction_re),
        std::move(diffraction_im),
        std::move(transmission_re),
        std::move(transmission_im),
        std::move(scattering_re),
        std::move(scattering_im));
}

// ---------------------------------------------------------------------------
// BDPT fixed-topology derivative dispatch. The flat
// kernels live in bdpt_subpaths_ad.cu / bdpt_connect_ad.cu / bdpt_maps.cu and
// consume unpacked tensor tables; the dispatch wrappers below mirror the
// forward wrappers so the registered ABI symbols accept the same
// subpath/intersection/light/sensor dicts positionally and forward only what
// the flat kernel consumes.
// ---------------------------------------------------------------------------

pybind11::dict channel_bdpt_reflected_light_subpath_state_backward(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_material,
    bool need_grad_field_in,
    bool need_grad_frequency);
pybind11::dict channel_bdpt_reflected_light_subpath_state_jvp(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid,
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor material_thickness,
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
pybind11::dict channel_bdpt_transmitted_light_subpath_state_backward(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_real,
    pybind11::object grad_field_imag,
    pybind11::object grad_throughput_real,
    pybind11::object grad_throughput_imag,
    bool need_grad_layers,
    bool need_grad_field_in,
    bool need_grad_frequency);
pybind11::dict channel_bdpt_transmitted_light_subpath_state_jvp(
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_valid,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor hit_t,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor face_material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_light_throughput_real,
    pybind11::object tangent_light_throughput_imag);
pybind11::dict channel_bdpt_endpoint_connection_samples_backward_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    at::Tensor grad_contribution,
    bool need_grad_field,
    bool need_grad_frequency,
    bool need_grad_tx_power);
pybind11::dict channel_bdpt_endpoint_connection_samples_jvp_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_sensor_field_real,
    double tangent_frequency,
    pybind11::object tangent_tx_power);
pybind11::dict channel_bdpt_finalize_point_components_backward(
    at::Tensor los,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components);
pybind11::dict channel_bdpt_finalize_point_components_jvp(
    at::Tensor los,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering);
pybind11::dict channel_bdpt_finalize_component_maps_backward(
    at::Tensor los,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components);
pybind11::dict channel_bdpt_finalize_component_maps_jvp(
    at::Tensor los,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering);

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
    bool need_grad_frequency) {
    // The flat VJP consumes hit_t/hit_n/hit_global_prim_id (hit_p is unused).
    return channel_bdpt_reflected_light_subpath_state_backward(
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "throughput_imag"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(intersection, "t"),
        tensor_from_dict(intersection, "n"),
        tensor_from_dict(intersection, "global_prim_id"),
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency_hz,
        std::move(grad_field_real),
        std::move(grad_field_imag),
        std::move(grad_throughput_real),
        std::move(grad_throughput_imag),
        need_grad_material,
        need_grad_field_in,
        need_grad_frequency);
}

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
    pybind11::object tangent_light_throughput_imag) {
    // Positional slot 3 of the material tangents is the reflection gain tangent
    // (differentiable reflected material set = {eps_r, sigma_e, gain, thickness};
    // mu_r is frozen). The facade forwards its tangent value into this slot.
    return channel_bdpt_reflected_light_subpath_state_jvp(
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "throughput_imag"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(intersection, "t"),
        tensor_from_dict(intersection, "n"),
        tensor_from_dict(intersection, "global_prim_id"),
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency_hz,
        std::move(tangent_eps_r),
        std::move(tangent_sigma_e),
        std::move(tangent_gain),
        std::move(tangent_thickness),
        tangent_frequency,
        std::move(tangent_light_field_real),
        std::move(tangent_light_field_imag),
        std::move(tangent_light_throughput_real),
        std::move(tangent_light_throughput_imag));
}

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
    bool need_grad_frequency) {
    return channel_bdpt_transmitted_light_subpath_state_backward(
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "throughput_imag"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(intersection, "t"),
        tensor_from_dict(intersection, "n"),
        tensor_from_dict(intersection, "global_prim_id"),
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency_hz,
        std::move(grad_field_real),
        std::move(grad_field_imag),
        std::move(grad_throughput_real),
        std::move(grad_throughput_imag),
        need_grad_layers,
        need_grad_field_in,
        need_grad_frequency);
}

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
    pybind11::object tangent_light_throughput_imag) {
    return channel_bdpt_transmitted_light_subpath_state_jvp(
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "throughput_imag"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(intersection, "t"),
        tensor_from_dict(intersection, "n"),
        tensor_from_dict(intersection, "global_prim_id"),
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency_hz,
        std::move(tangent_layer_thickness),
        std::move(tangent_layer_eps_r),
        std::move(tangent_layer_sigma_e),
        tangent_frequency,
        std::move(tangent_light_field_real),
        std::move(tangent_light_field_imag),
        std::move(tangent_light_throughput_real),
        std::move(tangent_light_throughput_imag));
}

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
    bool need_grad_tx_power) {
    // MIS (mode_id/beta/strategy_count) is frozen for the connection VJP; the
    // field/frequency/tx-power adjoints do not read it, so the wrapper accepts
    // the forward's positional scalars and forwards only what the flat kernel
    // consumes. tx_count is the transmitter count; for the endpoint (direct)
    // strategy the light subpath has exactly one endpoint per transmitter with
    // tx_id = arange(tx_count), so tx_count == light row count, which is also
    // the shape grad_tx_power must take.
    (void)mode_id;
    (void)beta;
    (void)strategy_count;
    at::Tensor light_origin = tensor_from_dict(light, "origin");
    const int64_t tx_count = light_origin.size(0);
    return channel_bdpt_endpoint_connection_samples_backward_cuda(
        light_origin,
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(light, "source_power"),
        tensor_from_dict(light, "depth"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "path_length"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "field_real"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "valid"),
        frequency_hz,
        samples_per_tx,
        tx_count,
        max_paths,
        grad_contribution,
        need_grad_field,
        need_grad_frequency,
        need_grad_tx_power);
}

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
    pybind11::object tangent_tx_power) {
    // MIS scalars are frozen (see the VJP wrapper). The sensor imaginary field
    // never enters the forward, so its tangent is dropped (matching the VJP's
    // always-zero grad_sensor_field_imag). tx_count == light row count.
    (void)mode_id;
    (void)beta;
    (void)strategy_count;
    (void)tangent_sensor_field_imag;
    at::Tensor light_origin = tensor_from_dict(light, "origin");
    const int64_t tx_count = light_origin.size(0);
    return channel_bdpt_endpoint_connection_samples_jvp_cuda(
        light_origin,
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(light, "source_power"),
        tensor_from_dict(light, "depth"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "path_length"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "field_real"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "valid"),
        frequency_hz,
        samples_per_tx,
        tx_count,
        max_paths,
        std::move(tangent_light_field_real),
        std::move(tangent_light_field_imag),
        std::move(tangent_sensor_field_real),
        tangent_frequency,
        std::move(tangent_tx_power));
}

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
    bool need_grad_components) {
    // The finalize map is linear; its transpose needs only `los` for
    // shape/device. The primal component matrices mirror the forward call and
    // are dropped.
    (void)reflection;
    (void)diffraction;
    (void)transmission;
    (void)scattering;
    return channel_bdpt_finalize_point_components_backward(
        los,
        std::move(grad_path_gain),
        std::move(grad_los_power),
        std::move(grad_reflection_power),
        std::move(grad_diffraction_power),
        std::move(grad_transmission_power),
        std::move(grad_scattering_power),
        need_grad_components);
}

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
    pybind11::object tangent_scattering) {
    (void)reflection;
    (void)diffraction;
    (void)transmission;
    (void)scattering;
    return channel_bdpt_finalize_point_components_jvp(
        los,
        std::move(tangent_los),
        std::move(tangent_reflection),
        std::move(tangent_diffraction),
        std::move(tangent_transmission),
        std::move(tangent_scattering));
}

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
    bool need_grad_components) {
    (void)reflection;
    (void)diffraction;
    (void)transmission;
    (void)scattering;
    return channel_bdpt_finalize_component_maps_backward(
        los,
        std::move(grad_path_gain),
        std::move(grad_los_power),
        std::move(grad_reflection_power),
        std::move(grad_diffraction_power),
        std::move(grad_transmission_power),
        std::move(grad_scattering_power),
        need_grad_components);
}

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
    pybind11::object tangent_scattering) {
    (void)reflection;
    (void)diffraction;
    (void)transmission;
    (void)scattering;
    return channel_bdpt_finalize_component_maps_jvp(
        los,
        std::move(tangent_los),
        std::move(tangent_reflection),
        std::move(tangent_diffraction),
        std::move(tangent_transmission),
        std::move(tangent_scattering));
}
