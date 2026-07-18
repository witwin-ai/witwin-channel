#include <torch/extension.h>

#include "registry.h"

#include <cstdint>
#include <string>
#include <vector>

pybind11::tuple cn_raydn_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags);
pybind11::tuple cn_raydn_scene_edge_records(
    int64_t scene_handle);
pybind11::tuple cn_bdpt_intersect_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags);
pybind11::tuple cn_bdpt_visibility_forward(
    int64_t scene_handle,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active);
pybind11::tuple cn_raydn_trace_reflections_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces);
pybind11::tuple cn_raydn_reflection_epc_paths_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    pybind11::object active,
    torch::Tensor expected_prim_ids,
    torch::Tensor direct_plane_points,
    torch::Tensor direct_plane_normals,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode);
pybind11::tuple cn_raydn_intersect_backward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    pybind11::object grad_t,
    pybind11::object grad_p,
    pybind11::object grad_n,
    pybind11::object grad_geo_n,
    pybind11::object grad_uv,
    pybind11::object grad_barycentric,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax);
pybind11::tuple cn_raydn_intersect_jvp(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    int64_t flags);
pybind11::tuple cn_raydn_trace_reflections_forward_tape(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces);
pybind11::tuple cn_raydn_trace_reflections_backward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    torch::Tensor image_sources,
    pybind11::object grad_t,
    pybind11::object grad_image_sources);
pybind11::tuple cn_raydn_trace_reflections_jvp(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    torch::Tensor image_sources);
pybind11::tuple cn_raydn_reflection_epc_paths_backward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object grad_points,
    pybind11::object grad_normals,
    pybind11::object grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver);
pybind11::tuple cn_raydn_reflection_epc_paths_jvp(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object tangent_vertices,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver);
at::Tensor cn_raydn_scene_face_normals_backward(
    int64_t scene_handle,
    torch::Tensor grad_face_normals);
at::Tensor cn_raydn_scene_face_normals_jvp(
    int64_t scene_handle,
    torch::Tensor tangent_vertices);
pybind11::dict cn_raydn_coupled_rd_geometry_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor face_id,
    torch::Tensor plane_point,
    torch::Tensor plane_normal,
    torch::Tensor edge_id,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_t_min,
    torch::Tensor edge_t_max,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    bool reverse);
pybind11::dict cn_raydn_coupled_dd_geometry_forward(
    int64_t scene_handle,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor edge1_id,
    torch::Tensor edge1_pos,
    torch::Tensor edge1_dir,
    torch::Tensor edge1_t_min,
    torch::Tensor edge1_t_max,
    torch::Tensor edge2_id,
    torch::Tensor edge2_pos,
    torch::Tensor edge2_dir,
    torch::Tensor edge2_t_min,
    torch::Tensor edge2_t_max);
pybind11::tuple cn_bdpt_reflection_accumulation_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    torch::Tensor active,
    torch::Tensor tx,
    torch::Tensor tx_pol,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool streaming_los_enabled);
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
    torch::Tensor edge_adjacent_face1);
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
    torch::Tensor edge_adjacent_face1);
pybind11::tuple cn_raydn_diffraction_paths_order1_forward(
    int64_t scene_handle,
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
    double wavelength);
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
    pybind11::object sample_edge_weight);

void register_rayd_geometry(pybind11::module_ &module) {
    module.def(
        "raydn_scene_create",
        &cn_raydn_scene_create,
        "Create a RayDN native scene through the direct C bridge.");
    module.def(
        "raydn_scene_edge_records",
        &cn_raydn_scene_edge_records,
        "Read RayDN native scene edge records through the direct C bridge.");
    module.def(
        "bdpt_intersect_forward",
        &cn_bdpt_intersect_forward,
        "Call RayDN ray/scene intersection through the native C bridge.");
    module.def(
        "bdpt_visibility_forward",
        &cn_bdpt_visibility_forward,
        "Call RayDN segment visibility through the native C bridge.");
    module.def(
        "raydn_trace_reflections_forward",
        &cn_raydn_trace_reflections_forward,
        "Call RayDN multibounce reflection chain tracing through the native C bridge.");
    module.def(
        "raydn_reflection_epc_paths_forward",
        &cn_raydn_reflection_epc_paths_forward,
        "Call RayDN reflection EPC path export through the native C bridge.");
    module.def(
        "raydn_intersect_backward",
        &cn_raydn_intersect_backward,
        "Call the RayDN fixed-winner intersect VJP through the native C bridge.");
    module.def(
        "raydn_intersect_jvp",
        &cn_raydn_intersect_jvp,
        "Call the RayDN fixed-winner intersect JVP through the native C bridge.");
    module.def(
        "raydn_trace_reflections_forward_tape",
        &cn_raydn_trace_reflections_forward_tape,
        "Call RayDN tape-emitting reflection chain tracing through the native C bridge.");
    module.def(
        "raydn_trace_reflections_backward",
        &cn_raydn_trace_reflections_backward,
        "Call the RayDN fixed-winner reflection chain VJP through the native C bridge.");
    module.def(
        "raydn_trace_reflections_jvp",
        &cn_raydn_trace_reflections_jvp,
        "Call the RayDN fixed-winner reflection chain JVP through the native C bridge.");
    module.def(
        "raydn_reflection_epc_paths_backward",
        &cn_raydn_reflection_epc_paths_backward,
        "Call the RayDN fixed-winner reflection EPC paths geometry VJP through the native C bridge.");
    module.def(
        "raydn_reflection_epc_paths_jvp",
        &cn_raydn_reflection_epc_paths_jvp,
        "Call the RayDN fixed-winner reflection EPC paths geometry JVP through the native C bridge.");
    module.def(
        "raydn_scene_face_normals_backward",
        &cn_raydn_scene_face_normals_backward,
        "Call the RayDN face-normal table VJP through the native C bridge.");
    module.def(
        "raydn_scene_face_normals_jvp",
        &cn_raydn_scene_face_normals_jvp,
        "Call the RayDN face-normal table JVP through the native C bridge.");
    module.def(
        "raydn_coupled_rd_geometry_forward",
        &cn_raydn_coupled_rd_geometry_forward,
        "Construct one-reflection/one-diffraction geometry with RayDN EPC and visibility; no field coefficient is evaluated.");
    module.def(
        "raydn_coupled_dd_geometry_forward",
        &cn_raydn_coupled_dd_geometry_forward,
        "Construct two-edge (double) diffraction geometry with an alternating-projection Fermat solve and RayDN segment visibility; no field coefficient is evaluated.");
}

void register_rayd_accumulation(pybind11::module_ &module) {
    module.def(
        "bdpt_reflection_accumulation_forward",
        &cn_bdpt_reflection_accumulation_forward,
        "Call RayDN reflection accumulation through the native C bridge.");
    module.def(
        "bdpt_diffraction_discover_edges",
        &cn_bdpt_diffraction_discover_edges,
        "Call RayDN diffraction edge discovery through the native C bridge.");
    module.def(
        "bdpt_diffraction_discover_edges_counted",
        &cn_bdpt_diffraction_discover_edges_counted,
        "Call RayDN counted diffraction edge discovery through the native C bridge.");
    module.def(
        "raydn_diffraction_paths_order1_forward",
        &cn_raydn_diffraction_paths_order1_forward,
        "Call RayDN diffraction order-1 path export through the native C bridge.");
    module.def(
        "bdpt_diffraction_accumulation_forward",
        &cn_bdpt_diffraction_accumulation_forward,
        "Call RayDN diffraction accumulation through the native C bridge.");
}
