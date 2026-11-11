// Copyright Xingyu Chen.
// Implements bdpt native integration.

#include <torch/extension.h>

#include <array>
#include <vector>

#include "bdpt_dict_helpers.h"

namespace {

pybind11::tuple tensor_vector_to_tuple(const std::vector<at::Tensor>& tensors) {
    pybind11::tuple out(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        out[i] = tensors[i];
    }
    return out;
}

pybind11::dict subpath_state_to_dict(const std::vector<at::Tensor>& tensors, const char* name) {
    TORCH_CHECK(tensors.size() == 19, name, " native result must contain 19 tensors");
    static constexpr std::array<const char*, 19> kFields = {
        "origin", "direction", "throughput_real", "throughput_imag", "pdf_forward",
        "pdf_reverse", "depth", "component_mask", "primitive_id", "edge_id",
        "tx_id", "rx_id", "grid_linear_id", "valid", "path_length",
        "field_real", "field_imag", "source_power", "event_type",
    };
    pybind11::dict out;
    for (size_t index = 0; index < kFields.size(); ++index) {
        out[kFields[index]] = tensors[index];
    }
    return out;
}

pybind11::dict connection_samples_to_dict(
    const std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>& tensors) {
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        light_depth,
        sensor_depth,
        path_length_m] = tensors;
    pybind11::dict out;
    out["topology"] = topology;
    out["contribution"] = contribution;
    out["pdf"] = pdf;
    out["mis_weight"] = mis_weight;
    out["component_id"] = component_id;
    out["valid"] = valid;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["grid_linear_id"] = grid_linear_id;
    out["light_depth"] = light_depth;
    out["sensor_depth"] = sensor_depth;
    out["path_length_m"] = path_length_m;
    return out;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_bdpt_launch_state_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed);
std::vector<at::Tensor> channel_bdpt_light_endpoint_subpath_state_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor launch_tx_id,
    at::Tensor light_seed);
std::vector<at::Tensor> channel_bdpt_sensor_endpoint_subpath_state_cuda(
    at::Tensor rx_positions, at::Tensor rx_polarization);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
channel_bdpt_endpoint_connection_samples_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_pdf_forward,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_pdf_reverse,
    at::Tensor sensor_depth,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_grid_linear_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_bdpt_endpoint_connection_visibility_inputs_cuda(
    at::Tensor light_origin,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor sensor_origin,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    int64_t sample_count);
std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor coeff_real,
    at::Tensor coeff_imag,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy,
    int64_t combine_domain);
void channel_bdpt_filter_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor valid,
    at::Tensor visible);
at::Tensor channel_bdpt_count_valid_connection_samples_cuda(at::Tensor valid);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
channel_bdpt_compact_connection_samples_cuda(
    at::Tensor topology,
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor grid_linear_id,
    at::Tensor light_depth,
    at::Tensor sensor_depth,
    at::Tensor path_length_m,
    int64_t max_paths);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
channel_bdpt_concat_connection_samples_cuda(
    std::vector<at::Tensor> topologies,
    std::vector<at::Tensor> contributions,
    std::vector<at::Tensor> pdfs,
    std::vector<at::Tensor> mis_weights,
    std::vector<at::Tensor> component_ids,
    std::vector<at::Tensor> valids,
    std::vector<at::Tensor> tx_ids,
    std::vector<at::Tensor> rx_ids,
    std::vector<at::Tensor> grid_linear_ids,
    std::vector<at::Tensor> light_depths,
    std::vector<at::Tensor> sensor_depths,
    std::vector<at::Tensor> path_lengths_m);
at::Tensor channel_bdpt_connection_variance_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx);
at::Tensor channel_bdpt_zero_matrix_cuda(at::Tensor reference, int64_t rows, int64_t cols);
at::Tensor channel_core_pack_int2_cuda(at::Tensor x, at::Tensor y);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_finalize_point_components_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering);
std::tuple<at::Tensor, at::Tensor> channel_bdpt_transmitter_tensors_cuda(
    const std::vector<float>& positions_host,
    const std::vector<float>& power_host);
at::Tensor channel_bdpt_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols);
at::Tensor channel_bdpt_receiver_grid_points_cuda(
    at::Tensor reference,
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
std::vector<at::Tensor> channel_deterministic_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index);
std::vector<at::Tensor> channel_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index);

pybind11::dict channel_bdpt_launch_state(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed) {
    auto [tx_id, sample_id, stream_id, light_seed] =
        channel_bdpt_launch_state_cuda(reference, tx_count, samples, sample_streams, seed);
    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["sample_id"] = sample_id;
    out["stream_id"] = stream_id;
    out["light_seed"] = light_seed;
    return out;
}

pybind11::dict channel_bdpt_endpoint_subpath_state(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    torch::Tensor launch_tx_id,
    torch::Tensor light_seed) {
    pybind11::dict out;
    out["light"] = subpath_state_to_dict(
        channel_bdpt_light_endpoint_subpath_state_cuda(
            tx_positions, tx_power, tx_polarization, launch_tx_id, light_seed),
        "bdpt_light_endpoint_subpath_state");
    out["sensor"] = subpath_state_to_dict(
        channel_bdpt_sensor_endpoint_subpath_state_cuda(rx_positions, rx_polarization),
        "bdpt_sensor_endpoint_subpath_state");
    return out;
}

pybind11::dict channel_bdpt_endpoint_connection_samples(
    pybind11::dict light,
    pybind11::dict sensor,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths) {
    return connection_samples_to_dict(channel_bdpt_endpoint_connection_samples_cuda(
        tensor_from_dict(light, "origin"),
        tensor_from_dict(light, "direction"),
        tensor_from_dict(light, "throughput_real"),
        tensor_from_dict(light, "field_real"),
        tensor_from_dict(light, "field_imag"),
        tensor_from_dict(light, "source_power"),
        tensor_from_dict(light, "pdf_forward"),
        tensor_from_dict(light, "depth"),
        tensor_from_dict(light, "component_mask"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(light, "path_length"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "field_real"),
        tensor_from_dict(sensor, "pdf_reverse"),
        tensor_from_dict(sensor, "depth"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "grid_linear_id"),
        tensor_from_dict(sensor, "valid"),
        frequency_hz,
        samples_per_tx,
        mode_id,
        beta,
        strategy_count,
        max_paths));
}

pybind11::dict channel_bdpt_accumulate_connection_samples(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy,
    int64_t combine_domain,
    torch::Tensor coeff_real,
    torch::Tensor coeff_imag) {
    auto [path_gain, los, reflection, diffraction, transmission, scattering,
          los_re, los_im, reflection_re, reflection_im, diffraction_re,
          diffraction_im, transmission_re, transmission_im, scattering_re,
          scattering_im] = channel_bdpt_accumulate_connection_samples_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        // Coherent coefficients arrive as .real/.imag strided views of the
        // natively-computed complex path field; the one-time layout copy to
        // dense planes belongs at this ABI boundary, not in the Python hot
        // path (mc hot-path layout-copy rule).
        coeff_real.defined() ? coeff_real.contiguous() : coeff_real,
        coeff_imag.defined() ? coeff_imag.contiguous() : coeff_imag,
        tx_count,
        rx_count,
        accumulation_strategy,
        combine_domain);
    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los"] = los;
    out["reflection"] = reflection;
    out["diffraction"] = diffraction;
    out["transmission"] = transmission;
    out["scattering"] = scattering;
    // Coherent BDPT accumulation: the coherent forward additionally returns the per
    // component phasor bin sums S_b so the accumulate backward reads them as
    // explicit args. The power domain leaves them undefined; only publish them
    // on the coherent branch so combine_domain == 0 stays byte-identical.
    if (combine_domain == 1) {
        out["los_re"] = los_re;
        out["los_im"] = los_im;
        out["reflection_re"] = reflection_re;
        out["reflection_im"] = reflection_im;
        out["diffraction_re"] = diffraction_re;
        out["diffraction_im"] = diffraction_im;
        out["transmission_re"] = transmission_re;
        out["transmission_im"] = transmission_im;
        out["scattering_re"] = scattering_re;
        out["scattering_im"] = scattering_im;
    }
    return out;
}

pybind11::dict channel_bdpt_filter_connection_samples(pybind11::dict samples, torch::Tensor visible) {
    channel_bdpt_filter_connection_samples_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "pdf"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "valid"),
        visible);
    return samples;
}

torch::Tensor channel_bdpt_count_valid_connection_samples(pybind11::dict samples) {
    return channel_bdpt_count_valid_connection_samples_cuda(tensor_from_dict(samples, "valid"));
}

pybind11::dict channel_bdpt_compact_connection_samples(pybind11::dict samples, int64_t max_paths) {
    return connection_samples_to_dict(channel_bdpt_compact_connection_samples_cuda(
        tensor_from_dict(samples, "topology"),
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "pdf"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "component_id"),
        tensor_from_dict(samples, "valid"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "grid_linear_id"),
        tensor_from_dict(samples, "light_depth"),
        tensor_from_dict(samples, "sensor_depth"),
        tensor_from_dict(samples, "path_length_m"),
        max_paths));
}

pybind11::dict channel_bdpt_concat_connection_samples(pybind11::sequence samples) {
    TORCH_CHECK(pybind11::len(samples) > 0, "bdpt_concat_connection_samples requires at least one sample block");
    const auto count = static_cast<size_t>(pybind11::len(samples));
    std::vector<at::Tensor> topologies;
    std::vector<at::Tensor> contributions;
    std::vector<at::Tensor> pdfs;
    std::vector<at::Tensor> mis_weights;
    std::vector<at::Tensor> component_ids;
    std::vector<at::Tensor> valids;
    std::vector<at::Tensor> tx_ids;
    std::vector<at::Tensor> rx_ids;
    std::vector<at::Tensor> grid_linear_ids;
    std::vector<at::Tensor> light_depths;
    std::vector<at::Tensor> sensor_depths;
    std::vector<at::Tensor> path_lengths_m;
    topologies.reserve(count);
    contributions.reserve(count);
    pdfs.reserve(count);
    mis_weights.reserve(count);
    component_ids.reserve(count);
    valids.reserve(count);
    tx_ids.reserve(count);
    rx_ids.reserve(count);
    grid_linear_ids.reserve(count);
    light_depths.reserve(count);
    sensor_depths.reserve(count);
    path_lengths_m.reserve(count);
    for (auto item : samples) {
        auto block = pybind11::cast<pybind11::dict>(item);
        topologies.push_back(tensor_from_dict(block, "topology"));
        contributions.push_back(tensor_from_dict(block, "contribution"));
        pdfs.push_back(tensor_from_dict(block, "pdf"));
        mis_weights.push_back(tensor_from_dict(block, "mis_weight"));
        component_ids.push_back(tensor_from_dict(block, "component_id"));
        valids.push_back(tensor_from_dict(block, "valid"));
        tx_ids.push_back(tensor_from_dict(block, "tx_id"));
        rx_ids.push_back(tensor_from_dict(block, "rx_id"));
        grid_linear_ids.push_back(tensor_from_dict(block, "grid_linear_id"));
        light_depths.push_back(tensor_from_dict(block, "light_depth"));
        sensor_depths.push_back(tensor_from_dict(block, "sensor_depth"));
        path_lengths_m.push_back(tensor_from_dict(block, "path_length_m"));
    }
    return connection_samples_to_dict(channel_bdpt_concat_connection_samples_cuda(
        std::move(topologies),
        std::move(contributions),
        std::move(pdfs),
        std::move(mis_weights),
        std::move(component_ids),
        std::move(valids),
        std::move(tx_ids),
        std::move(rx_ids),
        std::move(grid_linear_ids),
        std::move(light_depths),
        std::move(sensor_depths),
        std::move(path_lengths_m)));
}

torch::Tensor channel_bdpt_connection_variance(
    pybind11::dict samples,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx) {
    return channel_bdpt_connection_variance_cuda(
        tensor_from_dict(samples, "contribution"),
        tensor_from_dict(samples, "mis_weight"),
        tensor_from_dict(samples, "tx_id"),
        tensor_from_dict(samples, "rx_id"),
        tensor_from_dict(samples, "valid"),
        tx_count,
        rx_count,
        samples_per_tx);
}

pybind11::dict channel_bdpt_endpoint_connection_visibility_inputs(
    pybind11::dict light,
    pybind11::dict sensor,
    int64_t sample_count) {
    auto [start, end, active] = channel_bdpt_endpoint_connection_visibility_inputs_cuda(
        tensor_from_dict(light, "origin"),
        tensor_from_dict(light, "tx_id"),
        tensor_from_dict(light, "valid"),
        tensor_from_dict(sensor, "origin"),
        tensor_from_dict(sensor, "rx_id"),
        tensor_from_dict(sensor, "valid"),
        sample_count);
    pybind11::dict out;
    out["start"] = start;
    out["end"] = end;
    out["active"] = active;
    return out;
}

torch::Tensor channel_bdpt_zero_matrix(torch::Tensor reference, int64_t rows, int64_t cols) {
    return channel_bdpt_zero_matrix_cuda(reference, rows, cols);
}

torch::Tensor channel_core_pack_int2(torch::Tensor x, torch::Tensor y) {
    return channel_core_pack_int2_cuda(x, y);
}

pybind11::dict channel_bdpt_finalize_point_components(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power, transmission_power,
        scattering_power] =
        channel_bdpt_finalize_point_components_cuda(
            los, reflection, diffraction, transmission, scattering);
    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los_power"] = los_power;
    out["reflection_power"] = reflection_power;
    out["diffraction_power"] = diffraction_power;
    out["transmission_power"] = transmission_power;
    out["scattering_power"] = scattering_power;
    return out;
}

pybind11::dict channel_bdpt_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction,
    torch::Tensor transmission,
    torch::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power, transmission_power,
        scattering_power] =
        channel_bdpt_finalize_component_maps_cuda(
            los, reflection, diffraction, transmission, scattering);

    pybind11::dict out;
    out["path_gain"] = path_gain;
    out["los_power"] = los_power;
    out["reflection_power"] = reflection_power;
    out["diffraction_power"] = diffraction_power;
    out["transmission_power"] = transmission_power;
    out["scattering_power"] = scattering_power;
    return out;
}

pybind11::dict channel_bdpt_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers) {
    TORCH_CHECK(flat_positions.size() % 3 == 0, "flat_positions must contain xyz triples");
    TORCH_CHECK(flat_positions.size() / 3 == powers.size(), "powers must match flat_positions");
    std::vector<float> positions_host;
    std::vector<float> power_host;
    positions_host.reserve(flat_positions.size());
    power_host.reserve(powers.size());
    for (auto item : flat_positions) {
        positions_host.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    for (auto item : powers) {
        power_host.push_back(static_cast<float>(pybind11::cast<double>(item)));
    }
    auto [positions, power] = channel_bdpt_transmitter_tensors_cuda(positions_host, power_host);
    pybind11::dict out;
    out["positions"] = positions;
    out["power"] = power;
    return out;
}

torch::Tensor channel_bdpt_los_component_maps_from_matrix(torch::Tensor los, int64_t rows, int64_t cols) {
    return channel_bdpt_los_component_maps_from_matrix_cuda(los, rows, cols);
}

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
    double spacing1) {
    return channel_bdpt_receiver_grid_points_cuda(
        reference,
        rows,
        cols,
        origin_x,
        origin_y,
        origin_z,
        x_axis_x,
        x_axis_y,
        x_axis_z,
        y_axis_x,
        y_axis_y,
        y_axis_z,
        spacing0,
        spacing1);
}

pybind11::tuple channel_deterministic_diffraction_state_pack(
    torch::Tensor edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor line_min,
    torch::Tensor line_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor exterior_angle,
    torch::Tensor tx,
    torch::Tensor tx_power,
    int64_t tx_power_index) {
    return tensor_vector_to_tuple(channel_deterministic_diffraction_state_pack_cuda(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index));
}

pybind11::tuple channel_deterministic_diffraction_state_pack_selected(
    torch::Tensor selected,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor line_min,
    torch::Tensor line_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor exterior_angle,
    torch::Tensor tx,
    torch::Tensor tx_power,
    int64_t tx_power_index) {
    return tensor_vector_to_tuple(channel_deterministic_diffraction_state_pack_selected_cuda(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index));
}

