#pragma once

namespace raydtorch {

/// \brief Deduplicate reflection paths that share the same reflector sequence and image source.
///
/// Reads the ray-major (n_rays * max_bounces) trace arrays, collapses duplicate paths,
/// and writes the kept paths plus a per-kept discovery_count and representative ray index
/// into the caller-owned out_* buffers. All pointers are flat device arrays.
///
/// \return Number of unique paths kept.
int reflection_dedup_gpu(
    int n_rays,
    int max_bounces,
    const int *bounce_count,
    const int *shape_ids,
    const int *prim_ids,
    const float *t,
    const float *bary_u,
    const float *bary_v,
    const float *hit_x,
    const float *hit_y,
    const float *hit_z,
    const float *norm_x,
    const float *norm_y,
    const float *norm_z,
    const float *img_x,
    const float *img_y,
    const float *img_z,
    const int *face_offsets,
    int n_meshes,
    const int *canonical_prim_table,
    int canonical_table_size,
    float image_source_tolerance,
    int *out_bounce_count,
    int *out_shape_ids,
    int *out_prim_ids,
    float *out_t,
    float *out_bary_u,
    float *out_bary_v,
    float *out_hit_x,
    float *out_hit_y,
    float *out_hit_z,
    float *out_norm_x,
    float *out_norm_y,
    float *out_norm_z,
    float *out_img_x,
    float *out_img_y,
    float *out_img_z,
    int *out_discovery_count,
    int *out_representative_ray_index);

} // namespace raydtorch
