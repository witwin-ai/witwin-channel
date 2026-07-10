#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#if defined(_WIN32)
#define RAYDN_NATIVE_API __declspec(dllexport)
#else
#define RAYDN_NATIVE_API __attribute__((visibility("default")))
#endif

extern "C" RAYDN_NATIVE_API void raydn_native_visibility_forward(
    int64_t scene_handle,
    const at::Tensor *start,
    const at::Tensor *end,
    const at::Tensor *active,
    at::Tensor *visible,
    at::Tensor *blocker_prim,
    at::Tensor *tape_t);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_scene_create(
    const at::Tensor *vertices,
    const at::Tensor *faces,
    const at::Tensor *uv,
    const at::Tensor *face_uv,
    const at::Tensor *to_world_left,
    const at::Tensor *to_world_right,
    const int64_t *mesh_flags,
    int64_t mesh_count);

extern "C" RAYDN_NATIVE_API void raydn_native_scene_destroy(int64_t scene_handle);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_scene_edge_records(
    int64_t scene_handle,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_intersect_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    int64_t flags,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_trace_reflections_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    int64_t max_bounces,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_reflection_epc_paths_forward(
    int64_t scene_handle,
    const at::Tensor *source,
    const at::Tensor *receiver,
    const at::Tensor *active,
    const at::Tensor *expected_prim_ids,
    const at::Tensor *direct_plane_points,
    const at::Tensor *direct_plane_normals,
    const at::Tensor *surface_group_id,
    const at::Tensor *surface_group_size,
    const at::Tensor *surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_reflection_accumulation_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    const at::Tensor *tx,
    const at::Tensor *tx_pol,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
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
    bool streaming_los_enabled,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API void raydn_native_diffraction_discover_edges(
    const at::Tensor *tx_pos,
    const at::Tensor *ray_dir,
    const at::Tensor *prim_index,
    const at::Tensor *hit_p,
    const at::Tensor *hit_n,
    const at::Tensor *hit_geo_n,
    const at::Tensor *triangle_edge_count,
    const at::Tensor *triangle_edge_indices,
    const at::Tensor *edge_pos,
    const at::Tensor *edge_dir,
    const at::Tensor *edge_n0,
    const at::Tensor *edge_nn,
    const at::Tensor *edge_line_min,
    const at::Tensor *edge_line_max,
    const at::Tensor *edge_adjacent_face1,
    at::Tensor *out);

extern "C" RAYDN_NATIVE_API void raydn_native_diffraction_discover_edges_counted(
    const at::Tensor *tx_pos,
    const at::Tensor *ray_dir,
    const at::Tensor *prim_index,
    const at::Tensor *hit_p,
    const at::Tensor *hit_n,
    const at::Tensor *hit_geo_n,
    const at::Tensor *hit_count,
    const at::Tensor *triangle_edge_count,
    const at::Tensor *triangle_edge_indices,
    const at::Tensor *edge_pos,
    const at::Tensor *edge_dir,
    const at::Tensor *edge_n0,
    const at::Tensor *edge_nn,
    const at::Tensor *edge_line_min,
    const at::Tensor *edge_line_max,
    const at::Tensor *edge_adjacent_face1,
    at::Tensor *out);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_diffraction_accumulation_forward(
    int64_t scene_handle,
    const at::Tensor *active,
    const at::Tensor *state_edge_index,
    const at::Tensor *state_edge_pos,
    const at::Tensor *state_edge_dir,
    const at::Tensor *state_edge_t_min,
    const at::Tensor *state_edge_t_max,
    const at::Tensor *state_n0,
    const at::Tensor *state_n1,
    const at::Tensor *state_prim0,
    const at::Tensor *state_prim1,
    const at::Tensor *state_exterior_angle,
    const at::Tensor *state_src,
    const at::Tensor *state_src_power,
    const at::Tensor *state_wi,
    const at::Tensor *state_d0,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
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
    const at::Tensor *recursive_active,
    const at::Tensor *recursive_state_edge_index,
    const at::Tensor *recursive_state_edge_pos,
    const at::Tensor *recursive_state_edge_dir,
    const at::Tensor *recursive_state_edge_t_min,
    const at::Tensor *recursive_state_edge_t_max,
    const at::Tensor *recursive_state_n0,
    const at::Tensor *recursive_state_n1,
    const at::Tensor *recursive_state_prim0,
    const at::Tensor *recursive_state_prim1,
    const at::Tensor *recursive_state_exterior_angle,
    int64_t export_tape,
    const at::Tensor *sample_state_index,
    const at::Tensor *sample_edge_weight,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" RAYDN_NATIVE_API int64_t raydn_native_diffraction_paths_order1_forward(
    int64_t scene_handle,
    const at::Tensor *tx_pos,
    const at::Tensor *rx_pos,
    const at::Tensor *active,
    const at::Tensor *state_edge_index,
    const at::Tensor *state_edge_pos,
    const at::Tensor *state_edge_dir,
    const at::Tensor *state_edge_t_min,
    const at::Tensor *state_edge_t_max,
    const at::Tensor *state_n0,
    const at::Tensor *state_n1,
    const at::Tensor *state_prim0,
    const at::Tensor *state_prim1,
    const at::Tensor *state_exterior_angle,
    const at::Tensor *state_src,
    const at::Tensor *state_src_power,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    at::Tensor *outputs,
    int64_t output_capacity);
