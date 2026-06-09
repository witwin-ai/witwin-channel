#pragma once

#include <cstdint>

namespace raydtorch {

constexpr int EdgeOptixTopKMax = 16;
constexpr int EdgeOptixMaxTiers = 4;

struct EdgeOptixQueryParams {
    uint64_t handle = 0;
    uint64_t tier_handles[EdgeOptixMaxTiers] = {};
    float tier_search_radii[EdgeOptixMaxTiers] = {};
    int tier_count = 0;

    const float *edge_p0_x = nullptr;
    const float *edge_p0_y = nullptr;
    const float *edge_p0_z = nullptr;
    const float *edge_e1_x = nullptr;
    const float *edge_e1_y = nullptr;
    const float *edge_e1_z = nullptr;
    const uint8_t *edge_mask = nullptr;
    int edge_count = 0;
    float search_radius = 0.0f;

    const float *query_x = nullptr;
    const float *query_y = nullptr;
    const float *query_z = nullptr;
    const float *ray_dx = nullptr;
    const float *ray_dy = nullptr;
    const float *ray_dz = nullptr;
    const float *ray_tmax = nullptr;
    const uint8_t *active_mask = nullptr;
    int query_count = 0;
    int k = 0;

    const int *edge_shape_id = nullptr;
    const int *edge_local_id = nullptr;

    int *out_edge_ids = nullptr;
    float *out_distance_sq = nullptr;
    float *out_ray_t = nullptr;
    float *out_edge_t = nullptr;
    uint8_t *out_valid = nullptr;

    int write_point_outputs = 0;
    float *final_distance = nullptr;
    float *final_edge_point = nullptr;
    float *final_edge_t = nullptr;
    int *final_shape_id = nullptr;
    int *final_edge_id = nullptr;
    int *final_global_edge_id = nullptr;
    int *final_tape_edge_id = nullptr;
    float *final_tape_s = nullptr;
    float *final_tape_d = nullptr;
    uint8_t *final_unresolved = nullptr;
};

} // namespace raydtorch
