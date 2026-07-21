#include "resource.h"

#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kDiffractionStateCapacity = 4'194'304;

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

at::Tensor cn_diffraction_tx_visible_state_plan(
    RayDSceneResource &scene,
    torch::Tensor tx,
    torch::Tensor edge_position,
    torch::Tensor edge_direction,
    torch::Tensor edge_t_min,
    torch::Tensor edge_t_max) {
    TORCH_CHECK(
        edge_position.dim() >= 1,
        "edge_position must have at least one dimension");
    TORCH_CHECK(
        edge_position.size(0) <= kDiffractionStateCapacity,
        "diffraction transmitter-visible state capacity exceeds 4194304");
    rayd::torch::AxialEdgeVisibilityRequest request{
        tx,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        std::nullopt,
        {}};
    return rayd::torch::axial_edge_visibility_forward(scene.resource(), request)
        .any_visible;
}

pybind11::tuple cn_rayd_diffraction_paths_order1_forward(
    RayDSceneResource &scene,
    torch::Tensor tx_pos,
    torch::Tensor tx_pol,
    torch::Tensor rx_pos,
    torch::Tensor active,
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
        active,
        {state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min,
         state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1,
         state_exterior_angle, state_src, state_src_power, std::nullopt,
         std::nullopt},
        {material_eta_r, material_sigma, material_mu_r, material_gain,
         material_valid},
        state_limit,
        capacity,
        wavelength,
        isb_taper_width_scale,
        rayd::torch::DiffractionPathLayout::Compact};
    return diffraction_path_tuple(
        rayd::torch::diffraction_paths_order1_forward(scene.resource(), config));
}

pybind11::tuple cn_rayd_diffraction_sample_tape_forward(
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
