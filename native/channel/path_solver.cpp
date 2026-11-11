// Copyright Xingyu Chen.
// Implements path solver native integration.

#include <torch/extension.h>

#include "path_block.h"
#include "tensor_checks.h"

#include <tuple>
#include <vector>

using PathReflectionCandidateTuple = std::tuple<
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
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_path_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz,
    at::Tensor tx_pol);
at::Tensor channel_path_concat_vec3_cuda(std::vector<at::Tensor> blocks);
std::vector<at::Tensor> channel_path_pack_padded_cuda(
    at::Tensor tx_id, at::Tensor rx_id, at::Tensor depth, at::Tensor component_id,
    at::Tensor edge_id, at::Tensor interaction_type, at::Tensor primitive_sequence,
    at::Tensor material_sequence, at::Tensor positions, at::Tensor normals, at::Tensor coefficient,
    at::Tensor delay, at::Tensor field_xyz, at::Tensor field_direction, at::Tensor tx_positions,
    at::Tensor rx_positions, at::Tensor pair_offsets, int64_t width);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_path_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    at::Tensor tx_id,
    at::Tensor rx_id);
PathReflectionCandidateTuple channel_path_reflection_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor face_gain,
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz);

namespace {

std::vector<at::Tensor> tensor_sequence(pybind11::sequence sequence, const char* name) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(static_cast<size_t>(pybind11::len(sequence)));
    for (pybind11::handle item : sequence) {
        tensors.push_back(item.cast<at::Tensor>());
    }
    TORCH_CHECK(!tensors.empty(), name, " must not be empty");
    return tensors;
}

at::Tensor block_tensor(const pybind11::dict& block, const char* key) {
    TORCH_CHECK(block.contains(key), "path block missing ", key);
    return block[pybind11::str(key)].cast<at::Tensor>();
}

}  // namespace

pybind11::dict channel_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz,
    torch::Tensor tx_pol) {
    channel::check_tensor(tx_positions, "tx_positions", torch::kFloat32, 2);
    channel::check_tensor(tx_power, "tx_power", torch::kFloat32, 1);
    channel::check_tensor(rx_positions, "rx_positions", torch::kFloat32, 2);
    channel::check_tensor(tx_pol, "tx_pol", torch::kFloat32, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(tx_pol.sizes() == tx_positions.sizes(), "tx_pol must match tx_positions shape");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto [tx_id, rx_id, path_length, delay, path_gain, path_gain_matrix] =
        channel_path_los_export_cuda(tx_positions, tx_power, rx_positions, frequency_hz, tx_pol);

    pybind11::dict out;
    out["tx_id"] = tx_id;
    out["rx_id"] = rx_id;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["path_gain"] = path_gain;
    out["path_gain_matrix"] = path_gain_matrix;
    return out;
}

torch::Tensor channel_path_concat_vec3(pybind11::sequence blocks) {
    return channel_path_concat_vec3_cuda(tensor_sequence(blocks, "blocks"));
}

pybind11::dict channel_path_pack_padded(pybind11::dict rows, torch::Tensor tx_positions,
                                        torch::Tensor rx_positions, torch::Tensor pair_offsets,
                                        int64_t width) {
    auto field = [&rows](const char* name) {
        TORCH_CHECK(rows.contains(name), "path_pack_padded rows are missing ", name);
        return rows[name].cast<at::Tensor>();
    };
    auto packed = channel_path_pack_padded_cuda(
        field("tx_id"), field("rx_id"), field("depth"), field("component_id"), field("edge_id"),
        field("interaction_type"), field("primitive_sequence"), field("material_sequence"),
        field("interaction_positions"), field("interaction_normals"), field("coefficient"),
        field("delay_s"), field("field_xyz"), field("field_direction"), tx_positions, rx_positions,
        pair_offsets, width);
    static const char* const names[] = {
        "a",        "tau",    "theta_t",          "phi_t",           "theta_r",
        "phi_r",    "valid",  "interaction_type", "primitive_id",    "material_id",
        "position", "normal", "field_xyz",        "field_direction", "num_paths"};
    pybind11::dict out;
    for (size_t index = 0; index < packed.size(); ++index) {
        out[names[index]] = packed[index];
    }
    return out;
}

pybind11::dict channel_path_los_visibility_inputs(
    torch::Tensor tx_positions,
    torch::Tensor rx_positions,
    torch::Tensor tx_id,
    torch::Tensor rx_id) {
    auto [start, end, active] =
        channel_path_los_visibility_inputs_cuda(tx_positions, rx_positions, tx_id, rx_id);
    pybind11::dict out;
    out["start"] = start;
    out["end"] = end;
    out["active"] = active;
    return out;
}

