#pragma once

#include <cstdint>

namespace raydtorch {

/// Maximum edge samples per axial-edge visibility query.
constexpr int SegmentVisibilityMaxSamples = 16;

/// Launch parameters shared by the split segment-visibility pipelines; only the
/// fields relevant to the selected raygen entry are read.
struct SegmentVisibilityParams {
    uint64_t handle = 0;       ///< Traversable handle of the scene IAS.

    const int *face_offsets = nullptr; ///< Per-mesh face prefix-sum for globalizing ids.
    int n_meshes = 0;

    // Segment endpoints; end_b is the second endpoint for the segment-pair kind.
    const float *start_x = nullptr;
    const float *start_y = nullptr;
    const float *start_z = nullptr;
    const float *end_x = nullptr;
    const float *end_y = nullptr;
    const float *end_z = nullptr;
    const float *end_b_x = nullptr;
    const float *end_b_y = nullptr;
    const float *end_b_z = nullptr;

    // Axial-edge kind: edge line through end (= edge_pos) along edge_dir, sampled in [min, max].
    const float *edge_dir_x = nullptr;
    const float *edge_dir_y = nullptr;
    const float *edge_dir_z = nullptr;
    const float *edge_t_min = nullptr;
    const float *edge_t_max = nullptr;

    // Chain kind: flattened polyline points and per-chain length.
    const float *chain_point_x = nullptr;
    const float *chain_point_y = nullptr;
    const float *chain_point_z = nullptr;
    const int *chain_length = nullptr;
    int max_points = 0;       ///< Max points per chain (stride into the point arrays).
    int max_segments = 0;     ///< Max segments per chain.

    const int *ignore_prim_ids = nullptr; ///< Optional per-query primitives treated as non-occluding.
    int ignore_k = 0;         ///< Ignore-list stride per query (0 disables).
    const uint8_t *active_mask = nullptr;

    int n_rays = 0;
    int sample_count = 0;     ///< Edge samples used by the axial-edge kind.
    float sample_fractions[SegmentVisibilityMaxSamples] = {}; ///< Sample positions in [0, 1] along the edge line.

    uint8_t *out_visible = nullptr;   ///< Primary visibility output.
    uint8_t *out_visible_b = nullptr; ///< Second-endpoint output (segment-pair kind).
    int *out_first_blocked_segment = nullptr; ///< First occluded segment (chain kind).
    int *out_first_blocked_prim = nullptr;    ///< Primitive that blocked a segment query when requested.
};

} // namespace raydtorch
