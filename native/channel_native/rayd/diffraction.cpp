#include "bridge.h"
#include "../path_block.h"
#include "../tensor_checks.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_selected_cuda(
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
PathBlockTuple cn_path_diffraction_block_cuda(
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
PathBlockTuple cn_path_finalize_blocks_cuda(
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

using channel_native::check_flat_tensor;
using channel_native::check_vec3_table;
using channel_native::rayd_bridge::optional_tensor;
using channel_native::rayd_bridge::raydn_diffraction_accumulation_forward_fn;
using channel_native::rayd_bridge::raydn_diffraction_discover_edges_counted_fn;
using channel_native::rayd_bridge::raydn_diffraction_discover_edges_fn;
using channel_native::rayd_bridge::raydn_diffraction_paths_order1_forward_fn;

}  // namespace

torch::Tensor cn_bdpt_diffraction_discover_edges(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

torch::Tensor cn_bdpt_diffraction_discover_edges_counted(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor hit_count,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_counted_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &hit_count,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

pybind11::tuple cn_raydn_diffraction_paths_order1_forward(
    int64_t scene_handle,
    torch::Tensor tx_pos,
    torch::Tensor rx_pos,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    at::Tensor tx_pol = at::zeros_like(tx_pos);
    if (tx_pol.numel() != 0)
        tx_pol.select(1, 2).fill_(1.0);
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
        scene_handle,
        &tx_pos,
        &tx_pol,
        &rx_pos,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        state_limit,
        capacity,
        wavelength,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction order-1 path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::dict cn_path_diffraction_paths_order1(
    int64_t scene_handle,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
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
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    check_vec3_table(tx_positions, "tx_positions");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(rx_positions, "rx_positions");
    check_flat_tensor(selected, "selected", at::kBool);
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_flat_tensor(line_min, "line_min", at::kFloat);
    check_flat_tensor(line_max, "line_max", at::kFloat);
    check_vec3_table(n0, "n0");
    check_vec3_table(n1, "n1");
    check_flat_tensor(face0, "face0", at::kInt);
    check_flat_tensor(face1, "face1", at::kInt);
    check_flat_tensor(exterior_angle, "exterior_angle", at::kFloat);
    check_flat_tensor(material_eta_r, "material_eta_r", at::kFloat);
    check_flat_tensor(material_sigma, "material_sigma", at::kFloat);
    check_flat_tensor(material_mu_r, "material_mu_r", at::kFloat);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(selected.size(0) == edge_pos.size(0), "selected must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge_pos");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge_pos");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge_pos");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge_pos");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge_pos");
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    const int device = tx_positions.get_device();
    for (const auto& tensor : {
             tx_power,
             rx_positions,
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
             material_gain,
             material_valid,
         }) {
        TORCH_CHECK(tensor.get_device() == device, "diffraction path tensors must share one CUDA device");
    }

    const int64_t tx_count = tx_positions.size(0);
    PathBlockLists blocks;
    blocks.reserve(static_cast<size_t>(tx_count));

    constexpr int64_t kOutputCount = 18;
    for (int64_t tx_index = 0; tx_index < tx_count; ++tx_index) {
        at::Tensor tx = tx_positions.select(0, tx_index);
        std::vector<at::Tensor> states = cn_deterministic_diffraction_state_pack_selected_cuda(
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
            tx_index);
        TORCH_CHECK(states.size() == 12, "diffraction state pack must return 12 tensors");

        at::Tensor tx_view = tx_positions.narrow(0, tx_index, 1);
        at::Tensor tx_pol = at::zeros_like(tx_view);
        if (tx_pol.numel() != 0)
            tx_pol.select(1, 2).fill_(1.0);
        const int64_t state_limit = states[0].size(0);
        const int64_t capacity = rx_positions.size(0) * state_limit;
        std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
        const int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
            scene_handle,
            &tx_view,
            &tx_pol,
            &rx_positions,
            // The packed state table keeps one row per edge; the selection
            // mask must gate the launch so deselected (e.g. merged duplicate)
            // records never emit paths.
            &selected,
            &states[0],
            &states[1],
            &states[2],
            &states[3],
            &states[4],
            &states[5],
            &states[6],
            &states[7],
            &states[8],
            &states[9],
            &states[10],
            &states[11],
            &material_eta_r,
            &material_sigma,
            &material_mu_r,
            &material_gain,
            &material_valid,
            state_limit,
            capacity,
            wavelength,
            outputs.data(),
            kOutputCount);
        TORCH_CHECK(output_count == kOutputCount, "RayDN order-1 diffraction path export returned an unexpected output count");
        PathBlockTuple block = cn_path_diffraction_block_cuda(
            outputs[1],
            outputs[3],
            outputs[4],
            outputs[5],
            outputs[8],
            outputs[9],
            outputs[10],
            outputs[11],
            outputs[12],
            outputs[13],
            outputs[14],
            tx_index);
        blocks.append(block);
    }

    return path_block_dict(cn_path_finalize_blocks_cuda(
        blocks.valid,
        blocks.tx_id,
        blocks.rx_id,
        blocks.depth,
        blocks.component_id,
        blocks.primitive_id,
        blocks.edge_id,
        blocks.path_length,
        blocks.delay,
        blocks.path_gain,
        -1,
        tx_count,
        1));
}

pybind11::tuple cn_bdpt_diffraction_accumulation_forward(
    int64_t scene_handle,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    pybind11::object state_wi,
    pybind11::object state_d0,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    int64_t recursive_state_limit,
    pybind11::object recursive_active,
    pybind11::object recursive_state_edge_index,
    pybind11::object recursive_state_edge_pos,
    pybind11::object recursive_state_edge_dir,
    pybind11::object recursive_state_edge_t_min,
    pybind11::object recursive_state_edge_t_max,
    pybind11::object recursive_state_n0,
    pybind11::object recursive_state_n1,
    pybind11::object recursive_state_prim0,
    pybind11::object recursive_state_prim1,
    pybind11::object recursive_state_exterior_angle,
    int64_t export_tape,
    pybind11::object sample_state_index,
    pybind11::object sample_edge_weight,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor state_wi_storage;
    at::Tensor state_d0_storage;
    at::Tensor recursive_active_storage;
    at::Tensor recursive_state_edge_index_storage;
    at::Tensor recursive_state_edge_pos_storage;
    at::Tensor recursive_state_edge_dir_storage;
    at::Tensor recursive_state_edge_t_min_storage;
    at::Tensor recursive_state_edge_t_max_storage;
    at::Tensor recursive_state_n0_storage;
    at::Tensor recursive_state_n1_storage;
    at::Tensor recursive_state_prim0_storage;
    at::Tensor recursive_state_prim1_storage;
    at::Tensor recursive_state_exterior_angle_storage;
    at::Tensor sample_state_index_storage;
    at::Tensor sample_edge_weight_storage;

    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *state_wi_ptr = optional_tensor(std::move(state_wi), state_wi_storage);
    const at::Tensor *state_d0_ptr = optional_tensor(std::move(state_d0), state_d0_storage);
    const at::Tensor *recursive_active_ptr = optional_tensor(std::move(recursive_active), recursive_active_storage);
    const at::Tensor *recursive_state_edge_index_ptr =
        optional_tensor(std::move(recursive_state_edge_index), recursive_state_edge_index_storage);
    const at::Tensor *recursive_state_edge_pos_ptr =
        optional_tensor(std::move(recursive_state_edge_pos), recursive_state_edge_pos_storage);
    const at::Tensor *recursive_state_edge_dir_ptr =
        optional_tensor(std::move(recursive_state_edge_dir), recursive_state_edge_dir_storage);
    const at::Tensor *recursive_state_edge_t_min_ptr =
        optional_tensor(std::move(recursive_state_edge_t_min), recursive_state_edge_t_min_storage);
    const at::Tensor *recursive_state_edge_t_max_ptr =
        optional_tensor(std::move(recursive_state_edge_t_max), recursive_state_edge_t_max_storage);
    const at::Tensor *recursive_state_n0_ptr = optional_tensor(std::move(recursive_state_n0), recursive_state_n0_storage);
    const at::Tensor *recursive_state_n1_ptr = optional_tensor(std::move(recursive_state_n1), recursive_state_n1_storage);
    const at::Tensor *recursive_state_prim0_ptr =
        optional_tensor(std::move(recursive_state_prim0), recursive_state_prim0_storage);
    const at::Tensor *recursive_state_prim1_ptr =
        optional_tensor(std::move(recursive_state_prim1), recursive_state_prim1_storage);
    const at::Tensor *recursive_state_exterior_angle_ptr =
        optional_tensor(std::move(recursive_state_exterior_angle), recursive_state_exterior_angle_storage);
    const at::Tensor *sample_state_index_ptr =
        optional_tensor(std::move(sample_state_index), sample_state_index_storage);
    const at::Tensor *sample_edge_weight_ptr =
        optional_tensor(std::move(sample_edge_weight), sample_edge_weight_storage);

    constexpr int64_t kOutputCount = 19;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_accumulation_forward_fn(raydn_module_handle)(
        scene_handle,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        state_wi_ptr,
        state_d0_ptr,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        state_limit,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        grid_cell_area,
        wavelength,
        direct_samples,
        keller_samples,
        suffix_samples,
        seed,
        max_order,
        recursive_state_limit,
        recursive_active_ptr,
        recursive_state_edge_index_ptr,
        recursive_state_edge_pos_ptr,
        recursive_state_edge_dir_ptr,
        recursive_state_edge_t_min_ptr,
        recursive_state_edge_t_max_ptr,
        recursive_state_n0_ptr,
        recursive_state_n1_ptr,
        recursive_state_prim0_ptr,
        recursive_state_prim1_ptr,
        recursive_state_exterior_angle_ptr,
        export_tape,
        sample_state_index_ptr,
        sample_edge_weight_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction accumulation returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}
