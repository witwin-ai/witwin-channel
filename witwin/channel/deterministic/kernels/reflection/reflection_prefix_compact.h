#pragma once

namespace witwin::channel::native_ext {

void reflection_prefix_compact_representatives(
    const int* bounce_count,
    const int* discovery_count,
    const int* representative_ray_index,
    const int* global_prim_ids,
    const float* image_source_x,
    const float* image_source_y,
    const float* image_source_z,
    const int* canonical_prim_table,
    int canonical_table_size,
    int ray_count,
    int max_bounces,
    int depth,
    double image_source_tolerance,
    int* out_count,
    int* out_representative_chain_idx,
    int* out_discovery_count
);

} // namespace witwin::channel::native_ext
