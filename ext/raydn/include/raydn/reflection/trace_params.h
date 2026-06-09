#pragma once

#include <optix.h>
#include <vector_types.h>

#include <cstdint>

namespace raydn {

struct ReflectionTraceParams {
    OptixTraversableHandle primary_handle = 0;
    OptixTraversableHandle secondary_handle = 0;
    int split_mode = 0;

    const float *tri_p0_x = nullptr;
    const float *tri_p0_y = nullptr;
    const float *tri_p0_z = nullptr;
    const float *tri_e1_x = nullptr;
    const float *tri_e1_y = nullptr;
    const float *tri_e1_z = nullptr;
    const float *tri_e2_x = nullptr;
    const float *tri_e2_y = nullptr;
    const float *tri_e2_z = nullptr;
    const float *tri_fn_x = nullptr;
    const float *tri_fn_y = nullptr;
    const float *tri_fn_z = nullptr;
    const float4 *tri_p0_packed = nullptr;
    const float4 *tri_e1_packed = nullptr;
    const float4 *tri_e2_packed = nullptr;
    const float4 *tri_fn_packed = nullptr;

    const int *face_offsets = nullptr;
    int n_meshes = 0;
    int n_triangles = 0;

    const float *ray_ox = nullptr;
    const float *ray_oy = nullptr;
    const float *ray_oz = nullptr;
    const float *ray_dx = nullptr;
    const float *ray_dy = nullptr;
    const float *ray_dz = nullptr;
    const float *ray_o_aos = nullptr;
    const float *ray_d_aos = nullptr;
    const float *ray_tmax = nullptr;
    const uint8_t *active_mask = nullptr;
    int n_rays = 0;
    int max_bounces = 0;
    int export_mode = 0;
    int return_trailing = 0;
    int output_layout = 0;  ///< 0 = ray-major [ray, bounce], 1 = bounce-major [bounce, ray].

    uint8_t *out_valid = nullptr;  ///< Ray-major [ray, bounce] valid mask.
    int *out_bounce_count = nullptr;
    int *out_shape_ids = nullptr;
    int *out_prim_ids = nullptr;
    int *out_global_prim_ids = nullptr;
    float *out_t = nullptr;
    float *out_bary_u = nullptr;
    float *out_bary_v = nullptr;
    float *out_bary = nullptr;
    float *out_hit_x = nullptr;
    float *out_hit_y = nullptr;
    float *out_hit_z = nullptr;
    float *out_hit = nullptr;
    float *out_norm_x = nullptr;
    float *out_norm_y = nullptr;
    float *out_norm_z = nullptr;
    float *out_norm = nullptr;
    float *out_img_x = nullptr;
    float *out_img_y = nullptr;
    float *out_img_z = nullptr;
    float *out_img = nullptr;
    float *out_trailing_t = nullptr;
    int *out_trailing_prim = nullptr;
    float *out_trailing_dir_x = nullptr;
    float *out_trailing_dir_y = nullptr;
    float *out_trailing_dir_z = nullptr;
    float *out_trailing_origin_x = nullptr;
    float *out_trailing_origin_y = nullptr;
    float *out_trailing_origin_z = nullptr;
};

} // namespace raydn
