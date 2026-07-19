#include "resource.h"
#include "../path_block.h"
#include "../tensor_checks.h"

#include <cstdint>
#include <vector>

extern "C" void channel_native_diffraction_discover_edges(
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, at::Tensor *);
extern "C" void channel_native_diffraction_discover_edges_counted(
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    const at::Tensor *, const at::Tensor *, const at::Tensor *, const at::Tensor *,
    at::Tensor *);

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

pybind11::tuple diffraction_path_tuple(const rayd::torch::DiffractionPathResult &result) {
    return pybind11::make_tuple(
        result.count, result.valid, result.tx_id, result.rx_id, result.order,
        result.edge0, result.edge1, result.edge2, result.delay,
        result.field_x_re, result.field_x_im, result.field_y_re, result.field_y_im,
        result.field_z_re, result.field_z_im, result.p0, result.p1, result.p2);
}

pybind11::tuple diffraction_accumulation_tuple(
    const rayd::torch::DiffractionAccumulationResult &result) {
    return pybind11::make_tuple(
        result.power, result.field_x_re, result.field_x_im, result.field_y_re,
        result.field_y_im, result.field_z_re, result.field_z_im,
        result.direct_count, result.keller_count, result.suffix_count,
        result.visibility_rejects, result.edge_visibility_rejects,
        result.utd_rejects, result.edge_uses, result.tape_active,
        result.tape_state_idx, result.tape_cell, result.tape_material_idx,
        result.tape_edge_u);
}

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
    torch::Tensor edge_adjacent_face1) {
    at::Tensor out;
    channel_native_diffraction_discover_edges(
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
    torch::Tensor edge_adjacent_face1) {
    at::Tensor out;
    channel_native_diffraction_discover_edges_counted(
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

pybind11::tuple cn_rayd_diffraction_paths_order1_forward(
    RayDSceneResource &scene,
    torch::Tensor tx_pos,
    torch::Tensor tx_pol,
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
    double isb_taper_width_scale) {
    // tx_pol is the scene transmitter polarization threaded from the caller
    // (R5 fix). The RayD UTD op consumes it as the incident field basis.
    TORCH_CHECK(
        tx_pol.sizes() == tx_pos.sizes(),
        "tx_pol must match tx_pos shape (N, 3)");
    rayd::torch::DiffractionPathConfig config{
        tx_pos,
        tx_pol,
        rx_pos,
        optional_tensor(active),
        {state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min,
         state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1,
         state_exterior_angle, state_src, state_src_power, std::nullopt,
         std::nullopt},
        {material_eta_r, material_sigma, material_mu_r, material_gain,
         material_valid},
        state_limit,
        capacity,
        wavelength,
        isb_taper_width_scale};
    return diffraction_path_tuple(
        rayd::torch::diffraction_paths_order1_forward(scene.resource(), config));
}

pybind11::dict cn_path_diffraction_paths_order1(
    RayDSceneResource &scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_polarizations,
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
    double wavelength) {
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(tx_polarizations, "tx_polarizations");
    TORCH_CHECK(
        tx_polarizations.sizes() == tx_positions.sizes(),
        "tx_polarizations must match tx_positions shape (N, 3)");
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
        // R5 fix: use the scene transmitter polarization threaded from the
        // caller instead of a fabricated z-axis vector.
        at::Tensor tx_pol = tx_polarizations.narrow(0, tx_index, 1);
        const int64_t state_limit = states[0].size(0);
        const int64_t capacity = rx_positions.size(0) * state_limit;
        rayd::torch::DiffractionPathConfig config{
            tx_view,
            tx_pol,
            rx_positions,
            // The packed state table keeps one row per edge; the selection
            // mask must gate the launch so deselected (e.g. merged duplicate)
            // records never emit paths.
            selected,
            {states[0], states[1], states[2], states[3], states[4], states[5],
             states[6], states[7], states[8], states[9], states[10], states[11],
             std::nullopt, std::nullopt},
            {material_eta_r, material_sigma, material_mu_r, material_gain,
             material_valid},
            state_limit,
            capacity,
            wavelength,
            // ADR-017 ISB taper is not wired through this legacy per-tx path
            // table (no live solver caller); pass 0 to keep the hard GO step.
            0.0};
        const rayd::torch::DiffractionPathResult output =
            rayd::torch::diffraction_paths_order1_forward(scene.resource(), config);
        PathBlockTuple block = cn_path_diffraction_block_cuda(
            output.valid,
            output.rx_id,
            output.order,
            output.edge0,
            output.delay,
            output.field_x_re,
            output.field_x_im,
            output.field_y_re,
            output.field_y_im,
            output.field_z_re,
            output.field_z_im,
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
    RayDSceneResource &scene,
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
    pybind11::object sample_edge_weight) {
    std::optional<rayd::torch::RecursiveDiffractionState> recursive_state;
    if (!recursive_state_edge_index.is_none()) {
        recursive_state.emplace(rayd::torch::RecursiveDiffractionState{
            optional_tensor(recursive_active),
            recursive_state_edge_index.cast<at::Tensor>(),
            recursive_state_edge_pos.cast<at::Tensor>(),
            recursive_state_edge_dir.cast<at::Tensor>(),
            recursive_state_edge_t_min.cast<at::Tensor>(),
            recursive_state_edge_t_max.cast<at::Tensor>(),
            recursive_state_n0.cast<at::Tensor>(),
            recursive_state_n1.cast<at::Tensor>(),
            recursive_state_prim0.cast<at::Tensor>(),
            recursive_state_prim1.cast<at::Tensor>(),
            recursive_state_exterior_angle.cast<at::Tensor>(),
            recursive_state_limit});
    }

    rayd::torch::DiffractionAccumulationConfig config{
        optional_tensor(active),
        {state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min,
         state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1,
         state_exterior_angle, state_src, state_src_power,
         optional_tensor(state_wi), optional_tensor(state_d0)},
        {material_eta_r, material_sigma, material_mu_r, material_gain,
         material_valid},
        state_limit,
        {grid_axis, grid_position, grid_coord0_min, grid_coord0_max,
         grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1,
         grid_cell_area},
        wavelength,
        direct_samples,
        keller_samples,
        suffix_samples,
        seed,
        max_order,
        std::move(recursive_state),
        export_tape != 0,
        optional_tensor(sample_state_index),
        optional_tensor(sample_edge_weight)};
    return diffraction_accumulation_tuple(
        rayd::torch::diffraction_accumulation_forward(scene.resource(), config));
}
