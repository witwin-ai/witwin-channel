#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <optix.h>
#endif

namespace raydn {

/// Maximum reflection depth supported by the native EPC kernel.
constexpr int ReflEpcMaxBounces = 8;
/// visibility_ignore_mode value: ignore the exact reflector primitive on each segment.
constexpr int ReflEpcVisibilityIgnorePrimitive = 0;
/// visibility_ignore_mode value: ignore the reflector's whole surface group.
constexpr int ReflEpcVisibilityIgnoreSurfaceGroup = 1;

/// Launch parameters for the native EPC pipeline (flat SoA device pointers).
struct ReflEpcParams {
    OptixTraversableHandle primary_handle;   ///< Primary scene IAS handle.
    OptixTraversableHandle secondary_handle; ///< Secondary IAS handle (split scene).
    int split_mode;                          ///< 0 = single scene, nonzero = traverse both handles.

    // Scene-global triangles in edge-vector form (p0 + s*e1 + t*e2) with face normal fn.
    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;

    const int *face_offsets;  ///< Per-mesh face prefix-sum for globalizing primitive ids.
    int n_meshes;
    int n_triangles;

    // Expected reflector sequence and optional surface-group tables (see ReflEpcOptions).
    const int *expected_prim_ids;
    int expected_prim_count;
    const int *surface_group_id;
    int surface_group_id_count;
    const int *surface_group_size;
    int surface_group_count;
    const int *surface_group_members;
    int surface_max_group_size;
    int visibility_ignore_mode;  ///< One of the ReflEpcVisibilityIgnore* constants.
    const int *final_ignore_group_ids;
    int final_ignore_group_count;

    // Input rays, optional per-slot direct-plane override, and receiver positions.
    const float *ray_ox;
    const float *ray_oy;
    const float *ray_oz;
    const float *ray_dx;
    const float *ray_dy;
    const float *ray_dz;
    const float *ray_tmax;
    const float *direct_plane_point_x;
    const float *direct_plane_point_y;
    const float *direct_plane_point_z;
    const float *direct_plane_normal_x;
    const float *direct_plane_normal_y;
    const float *direct_plane_normal_z;
    const float *rx_x;
    const float *rx_y;
    const float *rx_z;
    int rx_count;
    const uint8_t *active_mask;
    int n_rays;
    int max_bounces;

    // Outputs: per-ray validity/length/blocking plus ray-major (n_rays * max_bounces) per-slot arrays.
    uint8_t *out_valid;
    int *out_bounce_count;
    float *out_path_length;
    float *out_point_x;
    float *out_point_y;
    float *out_point_z;
    int *out_trace_prim_ids;
    int *out_resolved_prim_ids;
    int *out_surface_group_ids;
    float *out_plane_normal_x;
    float *out_plane_normal_y;
    float *out_plane_normal_z;
    int *out_first_blocked_segment;
    int *out_first_blocked_prim;
    int *out_first_blocked_group;
};

} // namespace raydn
