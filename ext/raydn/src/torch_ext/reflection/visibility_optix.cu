#include <optix.h>
#include <optix_device.h>

#include <raydn/common/math.cuh>
#include <raydn/reflection/visibility_params.h>

namespace raydn {

extern "C" {
__constant__ SegmentVisibilityParams params;
}

namespace {

constexpr float kMinSegmentLength = 2e-5f;

static __forceinline__ __device__ bool is_active(unsigned int ray) {
    return params.active_mask == nullptr || params.active_mask[ray] != 0u;
}

static __forceinline__ __device__ float3 load_aos_vec3(const float *value,
                                                       unsigned int ray) {
    const unsigned int base = ray * 3u;
    return make_f3(value[base + 0u], value[base + 1u], value[base + 2u]);
}

static __forceinline__ __device__ float3 load_start(unsigned int ray) {
    if (params.start_aos != nullptr) {
        return load_aos_vec3(params.start_aos, ray);
    }
    return make_f3(params.start_x[ray], params.start_y[ray], params.start_z[ray]);
}

static __forceinline__ __device__ float3 load_end_a(unsigned int ray) {
    if (params.end_aos != nullptr) {
        return load_aos_vec3(params.end_aos, ray);
    }
    return make_f3(params.end_x[ray], params.end_y[ray], params.end_z[ray]);
}

static __forceinline__ __device__ float3 load_end_b(unsigned int ray) {
    if (params.end_b_aos != nullptr) {
        return load_aos_vec3(params.end_b_aos, ray);
    }
    return make_f3(params.end_b_x[ray], params.end_b_y[ray], params.end_b_z[ray]);
}

static __forceinline__ __device__ float3 load_chain_point(unsigned int chain,
                                                          int point_index) {
    const int slot = static_cast<int>(chain) * params.max_points + point_index;
    return make_f3(params.chain_point_x[slot],
                     params.chain_point_y[slot],
                     params.chain_point_z[slot]);
}

static __forceinline__ __device__ uint32_t trace_segment(float3 start,
                                                         float3 end,
                                                         bool active,
                                                         unsigned int ignore_base,
                                                         uint32_t *blocker_prim) {
    if (!active || params.handle == 0ull) {
        if (blocker_prim != nullptr) {
            *blocker_prim = 0xFFFFFFFFu;
        }
        return 0u;
    }

    float3 direction = end - start;
    const float length_sq = dot3(direction, direction);
    const float length = sqrtf(length_sq);
    if (length <= kMinSegmentLength) {
        if (blocker_prim != nullptr) {
            *blocker_prim = 0xFFFFFFFFu;
        }
        return 1u;
    }

    direction = (1.0f / length) * direction;
    const float3 origin = start + kRayBias * direction;
    const float tmax = fmaxf(length - 2.0f * kRayBias, 0.0f);

    uint32_t visible = 1u;
    uint32_t blocker = 0xFFFFFFFFu;
    const unsigned int ray_flags =
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        ((params.ignore_prim_ids == nullptr || params.ignore_k <= 0)
             ? OPTIX_RAY_FLAG_DISABLE_ANYHIT
             : 0u);
    optixTrace(static_cast<OptixTraversableHandle>(params.handle),
               origin,
               direction,
               kRayTMin,
               tmax,
               0.0f,
               255u,
               ray_flags,
               0,
               1,
               0,
               visible,
               blocker,
               ignore_base);
    if (blocker_prim != nullptr) {
        *blocker_prim = blocker;
    }
    return visible;
}

} // namespace

// OptiX programs for the native segment-visibility pipelines. Each split pipeline
// links one raygen entry and shares the anyhit/closesthit/miss programs.

/// Anyhit: skip occluders whose global primitive id is on this query's ignore list.
extern "C" __global__ void __anyhit__segment_visibility() {
    if (params.ignore_prim_ids == nullptr || params.ignore_k <= 0) {
        return;
    }

    const unsigned int ray = optixGetLaunchIndex().x;
    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int face_offset =
        (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
    const int global_prim = face_offset + static_cast<int>(optixGetPrimitiveIndex());
    const unsigned int ignore_base = optixGetPayload_2();

    for (int slot = 0; slot < params.ignore_k; ++slot) {
        const int ignored = params.ignore_prim_ids[ignore_base + slot];
        if (ignored == global_prim) {
            optixIgnoreIntersection();
            return;
        }
    }
}

/// Closest-hit: mark the segment occluded and record the blocking primitive's global id.
extern "C" __global__ void __closesthit__segment_visibility() {
    optixSetPayload_0(0u);
    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int face_offset =
        (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
    const int global_prim = face_offset + static_cast<int>(optixGetPrimitiveIndex());
    optixSetPayload_1(static_cast<unsigned int>(global_prim));
}

extern "C" __global__ void __miss__segment_visibility() {
    // Payload 0 is initialized to 1 by raygen and remains clear on miss.
}

/// Raygen (Segment kind): test one segment [start, end] and write its visibility.
extern "C" __global__ void __raygen__segment_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    uint32_t blocker = 0xFFFFFFFFu;
    const bool collect_blocker = params.out_first_blocked_prim != nullptr;
    const uint32_t visible =
        trace_segment(load_start(ray),
                      load_end_a(ray),
                      is_active(ray),
                      ray * params.ignore_k,
                      collect_blocker ? &blocker : nullptr);
    params.out_visible[ray] = visible != 0u ? 1u : 0u;
    if (collect_blocker) {
        params.out_first_blocked_prim[ray] =
            (visible == 0u && blocker != 0xFFFFFFFFu)
                ? static_cast<int>(blocker)
                : -1;
    }
    if (params.out_t != nullptr) {
        params.out_t[ray] = __uint_as_float(0x7f800000u);
    }
}

/// Raygen (SegmentPair kind): test two segments sharing the start point in one launch.
extern "C" __global__ void __raygen__segment_pair_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const bool active = is_active(ray);
    const float3 start = load_start(ray);
    const unsigned int ignore_base = ray * params.ignore_k;
    params.out_visible[ray] =
        trace_segment(start, load_end_a(ray), active, ignore_base, nullptr) != 0u ? 1u : 0u;
    params.out_visible_b[ray] =
        trace_segment(start, load_end_b(ray), active, ignore_base, nullptr) != 0u ? 1u : 0u;
}

/// Raygen (AxialEdge kind): test the source against sampled points along the edge line;
/// reports visible if any sample is unoccluded.
extern "C" __global__ void __raygen__axial_edge_visibility() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const bool active = is_active(ray);
    const float3 source = load_start(ray);
    const float3 edge_pos = load_end_a(ray);
    const float3 edge_dir = make_f3(params.edge_dir_x[ray],
                                      params.edge_dir_y[ray],
                                      params.edge_dir_z[ray]);
    const float line_min = params.edge_t_min[ray];
    const float line_max = params.edge_t_max[ray];
    const float span = fmaxf(line_max - line_min, 0.0f);
    uint32_t any_visible = 0u;

    #pragma unroll
    for (int i = 0; i < SegmentVisibilityMaxSamples; ++i) {
        if (i < params.sample_count) {
            const float t = line_min + params.sample_fractions[i] * span;
            const float3 sample = edge_pos + t * edge_dir;
            any_visible |= trace_segment(source, sample, active, 0u, nullptr);
        }
    }

    params.out_visible[ray] = any_visible != 0u ? 1u : 0u;
}

/// Raygen (SegmentChain kind): test every segment of one polyline, reporting overall
/// visibility and the first occluded segment/primitive.
extern "C" __global__ void __raygen__segment_chain_visibility() {
    const unsigned int chain = optixGetLaunchIndex().x;
    if (chain >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    if (!is_active(chain)) {
        params.out_visible[chain] = 0u;
        params.out_first_blocked_segment[chain] = -1;
        params.out_first_blocked_prim[chain] = -1;
        return;
    }

    int segment_count = params.chain_length != nullptr
        ? params.chain_length[chain]
        : params.max_segments;
    segment_count = max(0, min(segment_count, params.max_segments));

    uint32_t all_visible = 1u;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;

    for (int segment = 0; segment < segment_count; ++segment) {
        const float3 start = load_chain_point(chain, segment);
        const float3 end = load_chain_point(chain, segment + 1);
        const unsigned int ignore_base = params.ignore_k > 0
            ? (static_cast<unsigned int>(chain) * params.max_segments +
               static_cast<unsigned int>(segment)) *
                  static_cast<unsigned int>(params.ignore_k)
            : 0u;

        uint32_t blocker_prim = 0xFFFFFFFFu;
        const uint32_t visible =
            trace_segment(start, end, true, ignore_base, &blocker_prim);
        if (visible == 0u) {
            all_visible = 0u;
            first_blocked_segment = segment;
            first_blocked_prim = static_cast<int>(blocker_prim);
            break;
        }
    }

    params.out_visible[chain] = all_visible != 0u ? 1u : 0u;
    params.out_first_blocked_segment[chain] = first_blocked_segment;
    params.out_first_blocked_prim[chain] = first_blocked_prim;
}

} // namespace raydn
