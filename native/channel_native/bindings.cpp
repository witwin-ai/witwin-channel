#include <torch/extension.h>

pybind11::dict cn_build_info();
pybind11::dict cn_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz);
pybind11::dict cn_mc_finalize_component_maps(
    torch::Tensor los,
    torch::Tensor reflection,
    torch::Tensor diffraction);
torch::Tensor cn_mc_component_map_buffer(
    torch::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1);
torch::Tensor cn_mc_store_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    int64_t tx_index);
torch::Tensor cn_mc_store_scaled_component_map(
    torch::Tensor maps,
    torch::Tensor source,
    torch::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index);
torch::Tensor cn_mc_sample_directions(int64_t count, torch::Tensor reference);
pybind11::dict cn_mc_transmitter_tensors(
    pybind11::sequence flat_positions,
    pybind11::sequence powers);
torch::Tensor cn_mc_pack_vec3(torch::Tensor x, torch::Tensor y, torch::Tensor z);
torch::Tensor cn_mc_los_component_maps(torch::Tensor los);
pybind11::tuple cn_mc_los_path_gain_backward(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor grad_output,
    double frequency_hz);
torch::Tensor cn_mc_los_path_gain_jvp(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor tx_tangent,
    torch::Tensor power_tangent,
    torch::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz);
torch::Tensor cn_mc_apply_los_visibility(
    torch::Tensor maps,
    torch::Tensor los,
    torch::Tensor visible,
    int64_t tx_index);
pybind11::dict cn_mc_los_visibility_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count);
torch::Tensor cn_mc_receiver_grid_points(
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
pybind11::dict cn_mc_reflection_launch_inputs(
    torch::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count);
torch::Tensor cn_mc_diffraction_state_wi(torch::Tensor state_edge_pos, torch::Tensor state_src);
torch::Tensor cn_mc_selected_edge_indices(torch::Tensor selected);
pybind11::tuple cn_mc_diffraction_state_pack(
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
    torch::Tensor tx_power);
pybind11::tuple cn_mc_diffraction_edge_geometry(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    double plane_tol);
pybind11::tuple cn_mc_surface_group_edge_candidates(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor face_normals,
    torch::Tensor edge_v0,
    torch::Tensor edge_v1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor selected,
    double plane_tol);
pybind11::dict cn_mc_face_material_tensors(
    torch::Tensor material_eps_r,
    torch::Tensor material_sigma_e,
    torch::Tensor material_mu_r,
    torch::Tensor face_material_id);

PYBIND11_MODULE(_channel_native, module) {
    module.doc() = "Channel Native Torch/CUDA extension.";
    module.def("build_info", &cn_build_info, "Return Channel Native build metadata.");
    module.def(
        "path_los_export",
        &cn_path_los_export,
        "Export empty-space LoS paths from CUDA tensors.");
    module.def(
        "mc_finalize_component_maps",
        &cn_mc_finalize_component_maps,
        "Fuse MC component map total and component power reductions.");
    module.def(
        "mc_component_map_buffer",
        &cn_mc_component_map_buffer,
        "Allocate a zero-filled MC component map buffer.");
    module.def(
        "mc_store_component_map",
        &cn_mc_store_component_map,
        "Store one source component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "mc_store_scaled_component_map",
        &cn_mc_store_scaled_component_map,
        "Store one scaled source component map into a transmitter slot with a CUDA kernel.");
    module.def(
        "mc_sample_directions",
        &cn_mc_sample_directions,
        "Generate MC sphere sample directions with a fused CUDA kernel.");
    module.def(
        "mc_transmitter_tensors",
        &cn_mc_transmitter_tensors,
        "Create transmitter position and power CUDA tensors through native code.");
    module.def(
        "mc_pack_vec3",
        &cn_mc_pack_vec3,
        "Pack three CUDA float vectors into an interleaved vec3 tensor.");
    module.def(
        "mc_los_component_maps",
        &cn_mc_los_component_maps,
        "Convert MC LoS gains to public component-map grid layout with a CUDA kernel.");
    module.def(
        "mc_los_path_gain_backward",
        &cn_mc_los_path_gain_backward,
        "Compute LoS path-gain VJP gradients with native CUDA kernels.");
    module.def(
        "mc_los_path_gain_jvp",
        &cn_mc_los_path_gain_jvp,
        "Compute LoS path-gain JVP with a native CUDA kernel.");
    module.def(
        "mc_apply_los_visibility",
        &cn_mc_apply_los_visibility,
        "Apply RayDN LoS visibility to one transmitter component map with a CUDA kernel.");
    module.def(
        "mc_los_visibility_inputs",
        &cn_mc_los_visibility_inputs,
        "Prepare RayDN LoS visibility start and active tensors with a CUDA kernel.");
    module.def(
        "mc_receiver_grid_points",
        &cn_mc_receiver_grid_points,
        "Generate receiver grid points with a CUDA kernel.");
    module.def(
        "mc_reflection_launch_inputs",
        &cn_mc_reflection_launch_inputs,
        "Prepare RayDN reflection launch tensors with a CUDA kernel.");
    module.def(
        "mc_diffraction_state_wi",
        &cn_mc_diffraction_state_wi,
        "Compute MC diffraction incident directions with a CUDA kernel.");
    module.def(
        "mc_selected_edge_indices",
        &cn_mc_selected_edge_indices,
        "Compact a selected edge mask into deterministic int32 indices with native CUDA kernels.");
    module.def(
        "mc_diffraction_state_pack",
        &cn_mc_diffraction_state_pack,
        "Pack MC diffraction edge states with a fused CUDA kernel.");
    module.def(
        "mc_diffraction_edge_geometry",
        &cn_mc_diffraction_edge_geometry,
        "Build MC diffraction edge geometry tensors with a fused CUDA kernel.");
    module.def(
        "mc_surface_group_edge_candidates",
        &cn_mc_surface_group_edge_candidates,
        "Build MC surface-group diffraction edge candidate tables with native CUDA kernels.");
    module.def(
        "mc_face_material_tensors",
        &cn_mc_face_material_tensors,
        "Expand material parameters to per-face tensors with a fused CUDA kernel.");
}
