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
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_path_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    at::Tensor tx_id,
    at::Tensor rx_id);
PathBlockTuple channel_path_filter_los_cuda(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible);
PathReflectionCandidateTuple channel_path_reflection_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor face_gain,
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz);
PathBlockTuple channel_path_filter_block_cuda(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible0,
    at::Tensor visible1);
PathBlockTuple channel_path_diffraction_block_cuda(
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor field_x_re,
    at::Tensor field_x_im,
    at::Tensor field_y_re,
    at::Tensor field_y_im,
    at::Tensor field_z_re,
    at::Tensor field_z_im,
    int64_t tx_index);
PathBlockTuple channel_path_finalize_blocks_cuda(
    std::vector<at::Tensor> valid_blocks,
    std::vector<at::Tensor> tx_id_blocks,
    std::vector<at::Tensor> rx_id_blocks,
    std::vector<at::Tensor> depth_blocks,
    std::vector<at::Tensor> component_id_blocks,
    std::vector<at::Tensor> primitive_id_blocks,
    std::vector<at::Tensor> edge_id_blocks,
    std::vector<at::Tensor> path_length_blocks,
    std::vector<at::Tensor> delay_blocks,
    std::vector<at::Tensor> path_gain_blocks,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth);

namespace {

pybind11::dict reflection_candidate_dict(const PathReflectionCandidateTuple& candidates) {
    pybind11::dict out = path_block_dict(PathBlockTuple{
        std::get<0>(candidates),
        std::get<1>(candidates),
        std::get<2>(candidates),
        std::get<3>(candidates),
        std::get<4>(candidates),
        std::get<5>(candidates),
        std::get<6>(candidates),
        std::get<7>(candidates),
        std::get<8>(candidates),
        std::get<9>(candidates),
    });
    out["seg0_start"] = std::get<10>(candidates);
    out["seg0_end"] = std::get<11>(candidates);
    out["seg1_start"] = std::get<12>(candidates);
    out["seg1_end"] = std::get<13>(candidates);
    out["active"] = std::get<14>(candidates);
    return out;
}

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

void append_path_block(
    const pybind11::dict& block,
    PathBlockLists& blocks) {
    blocks.valid.push_back(block_tensor(block, "valid"));
    blocks.tx_id.push_back(block_tensor(block, "tx_id"));
    blocks.rx_id.push_back(block_tensor(block, "rx_id"));
    blocks.depth.push_back(block_tensor(block, "depth"));
    blocks.component_id.push_back(block_tensor(block, "component_id"));
    blocks.primitive_id.push_back(block_tensor(block, "primitive_id"));
    blocks.edge_id.push_back(block_tensor(block, "edge_id"));
    blocks.path_length.push_back(block_tensor(block, "path_length_m"));
    blocks.delay.push_back(block_tensor(block, "delay_s"));
    blocks.path_gain.push_back(block_tensor(block, "path_gain"));
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

pybind11::dict channel_path_filter_los(
    torch::Tensor tx_id,
    torch::Tensor rx_id,
    torch::Tensor path_length,
    torch::Tensor delay,
    torch::Tensor path_gain,
    torch::Tensor visible) {
    return path_block_dict(channel_path_filter_los_cuda(tx_id, rx_id, path_length, delay, path_gain, visible));
}

pybind11::dict channel_path_reflection_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor face_gain,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz) {
    return reflection_candidate_dict(channel_path_reflection_candidates_cuda(
        vertices,
        faces,
        face_normals,
        face_gain,
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz));
}

pybind11::dict channel_path_filter_block(
    pybind11::dict block,
    torch::Tensor visible0,
    torch::Tensor visible1) {
    return path_block_dict(channel_path_filter_block_cuda(
        block_tensor(block, "valid"),
        block_tensor(block, "tx_id"),
        block_tensor(block, "rx_id"),
        block_tensor(block, "depth"),
        block_tensor(block, "component_id"),
        block_tensor(block, "primitive_id"),
        block_tensor(block, "edge_id"),
        block_tensor(block, "path_length_m"),
        block_tensor(block, "delay_s"),
        block_tensor(block, "path_gain"),
        visible0,
        visible1));
}

pybind11::dict channel_path_diffraction_block(
    pybind11::sequence rayd_output,
    int64_t tx_index) {
    TORCH_CHECK(pybind11::len(rayd_output) == 18, "RayD diffraction path output must contain 18 tensors");
    return path_block_dict(channel_path_diffraction_block_cuda(
        rayd_output[1].cast<at::Tensor>(),
        rayd_output[3].cast<at::Tensor>(),
        rayd_output[4].cast<at::Tensor>(),
        rayd_output[5].cast<at::Tensor>(),
        rayd_output[8].cast<at::Tensor>(),
        rayd_output[9].cast<at::Tensor>(),
        rayd_output[10].cast<at::Tensor>(),
        rayd_output[11].cast<at::Tensor>(),
        rayd_output[12].cast<at::Tensor>(),
        rayd_output[13].cast<at::Tensor>(),
        rayd_output[14].cast<at::Tensor>(),
        tx_index));
}

pybind11::dict channel_path_merge_blocks(
    pybind11::sequence blocks,
    int64_t tx_count,
    int64_t max_depth) {
    const size_t block_count = static_cast<size_t>(pybind11::len(blocks));
    TORCH_CHECK(block_count > 0, "path_merge_blocks requires at least one block");
    PathBlockLists path_blocks;
    path_blocks.reserve(block_count);
    for (pybind11::handle item : blocks) {
        append_path_block(item.cast<pybind11::dict>(), path_blocks);
    }
    return path_block_dict(channel_path_finalize_blocks_cuda(
        path_blocks.valid,
        path_blocks.tx_id,
        path_blocks.rx_id,
        path_blocks.depth,
        path_blocks.component_id,
        path_blocks.primitive_id,
        path_blocks.edge_id,
        path_blocks.path_length,
        path_blocks.delay,
        path_blocks.path_gain,
        -1,
        tx_count,
        max_depth));
}

pybind11::dict channel_path_finalize_blocks(
    pybind11::dict los,
    pybind11::dict reflection,
    pybind11::dict diffraction,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth) {
    std::vector<at::Tensor> valid_blocks = {
        block_tensor(los, "valid"),
        block_tensor(reflection, "valid"),
        block_tensor(diffraction, "valid"),
    };
    std::vector<at::Tensor> tx_id_blocks = {
        block_tensor(los, "tx_id"),
        block_tensor(reflection, "tx_id"),
        block_tensor(diffraction, "tx_id"),
    };
    std::vector<at::Tensor> rx_id_blocks = {
        block_tensor(los, "rx_id"),
        block_tensor(reflection, "rx_id"),
        block_tensor(diffraction, "rx_id"),
    };
    std::vector<at::Tensor> depth_blocks = {
        block_tensor(los, "depth"),
        block_tensor(reflection, "depth"),
        block_tensor(diffraction, "depth"),
    };
    std::vector<at::Tensor> component_id_blocks = {
        block_tensor(los, "component_id"),
        block_tensor(reflection, "component_id"),
        block_tensor(diffraction, "component_id"),
    };
    std::vector<at::Tensor> primitive_id_blocks = {
        block_tensor(los, "primitive_id"),
        block_tensor(reflection, "primitive_id"),
        block_tensor(diffraction, "primitive_id"),
    };
    std::vector<at::Tensor> edge_id_blocks = {
        block_tensor(los, "edge_id"),
        block_tensor(reflection, "edge_id"),
        block_tensor(diffraction, "edge_id"),
    };
    std::vector<at::Tensor> path_length_blocks = {
        block_tensor(los, "path_length_m"),
        block_tensor(reflection, "path_length_m"),
        block_tensor(diffraction, "path_length_m"),
    };
    std::vector<at::Tensor> delay_blocks = {
        block_tensor(los, "delay_s"),
        block_tensor(reflection, "delay_s"),
        block_tensor(diffraction, "delay_s"),
    };
    std::vector<at::Tensor> path_gain_blocks = {
        block_tensor(los, "path_gain"),
        block_tensor(reflection, "path_gain"),
        block_tensor(diffraction, "path_gain"),
    };
    return path_block_dict(channel_path_finalize_blocks_cuda(
        valid_blocks,
        tx_id_blocks,
        rx_id_blocks,
        depth_blocks,
        component_id_blocks,
        primitive_id_blocks,
        edge_id_blocks,
        path_length_blocks,
        delay_blocks,
        path_gain_blocks,
        max_paths,
        tx_count,
        max_depth));
}
