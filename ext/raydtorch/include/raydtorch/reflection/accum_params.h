#pragma once

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#  include <vector_types.h>
#else
#  include <optix.h>
#  include <vector_types.h>
#endif

namespace raydtorch {

struct ReflAccumStagedValue {
    // xyzw = power, field_x_re, field_x_im, field_y_re
    float4 a;
    // xyzw = field_y_im, field_z_re, field_z_im, reflection_count
    float4 b;
};

/// Launch parameters for the native reflection-accumulation pipeline (flat SoA device pointers).
struct AccumParams {
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

    // Input rays and per-ray active mask.
    const float *ray_ox;
    const float *ray_oy;
    const float *ray_oz;
    const float *ray_dx;
    const float *ray_dy;
    const float *ray_dz;
    const float *ray_tmax;
    const uint8_t *active_mask;
    int n_rays;

    // Transmitter position and polarization (per ray or broadcast).
    const float *tx_x;
    const float *tx_y;
    const float *tx_z;
    const float *tx_pol_x;
    const float *tx_pol_y;
    const float *tx_pol_z;

    int max_bounces;
    float wavelength;
    float k;
    float solid_angle_per_ray;
    float cell_area;
    int seed;
    int rr_depth;
    float rr_prob;
    float stop_threshold;

    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;

    // Per-global-primitive material payload (see MaterialData).
    const float *material_eta_r;
    const float *material_sigma;
    const float *material_gain;
    const float *material_mu_r;
    const uint8_t *material_valid;
    int material_count;

    int collect_wedges;          ///< Record diffraction-wedge events when nonzero.
    int collect_wedge_prefixes;  ///< Also record each wedge's reflection prefix.
    int wedge_capacity;          ///< Capacity of the wedge-event output buffers.
    int wedge_sample_stride;     ///< Prefix wedge sampling stride.

    // Outputs: per-grid-cell power/field, total reflection count, and the wedge-event buffers.
    float *out_reflection_power;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    int *out_reflection_count;
    int *stage_cell;
    ReflAccumStagedValue *stage_value;
    int *out_wedge_count;
    int *out_wedge_ray_index;
    float *out_wedge_hit_x;
    float *out_wedge_hit_y;
    float *out_wedge_hit_z;
    float *out_wedge_normal_x;
    float *out_wedge_normal_y;
    float *out_wedge_normal_z;
    int *out_wedge_prim_id;
    float *out_wedge_dir_x;
    float *out_wedge_dir_y;
    float *out_wedge_dir_z;
    float *out_wedge_source_x;
    float *out_wedge_source_y;
    float *out_wedge_source_z;
    float *out_wedge_source_power;
    float *out_wedge_initial_dir_x;
    float *out_wedge_initial_dir_y;
    float *out_wedge_initial_dir_z;
    int *out_wedge_bounce_depth;
};

} // namespace raydtorch
