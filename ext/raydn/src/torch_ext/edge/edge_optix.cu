#include <optix.h>
#include <optix_device.h>

#include <raydn/common/math.cuh>
#include <raydn/edge/optix_params.h>

namespace raydn {

extern "C" {
__constant__ EdgeOptixQueryParams params;
}

namespace {

constexpr float kInfiniteRayTMax = 1.0e8f;
constexpr float kPointProbeTMax = 1.0e-5f;
constexpr uint32_t kInvalidEdgeId = 0xffffffffu;

static __forceinline__ __device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __forceinline__ __device__ bool is_active(unsigned int query) {
    return params.active_mask == nullptr || params.active_mask[query] != 0u;
}

static __forceinline__ __device__ bool edge_visible(unsigned int edge) {
    return params.edge_mask == nullptr || params.edge_mask[edge] != 0u;
}

static __forceinline__ __device__ float3 load_query_point(unsigned int query) {
    return make_f3(params.query_x[query], params.query_y[query], params.query_z[query]);
}

static __forceinline__ __device__ float3 load_ray_direction(unsigned int query) {
    return make_f3(params.ray_dx[query], params.ray_dy[query], params.ray_dz[query]);
}

static __forceinline__ __device__ float3 load_edge_start(unsigned int edge) {
    return make_f3(params.edge_p0_x[edge], params.edge_p0_y[edge], params.edge_p0_z[edge]);
}

static __forceinline__ __device__ float3 load_edge_vector(unsigned int edge) {
    return make_f3(params.edge_e1_x[edge], params.edge_e1_y[edge], params.edge_e1_z[edge]);
}

static __forceinline__ __device__ float safe_search_radius() {
    return fmaxf(params.search_radius, 0.0f);
}

static __forceinline__ __device__ int active_tier_count() {
    if (params.tier_count <= 0) {
        return 1;
    }
    return params.tier_count < EdgeOptixMaxTiers ? params.tier_count : EdgeOptixMaxTiers;
}

static __forceinline__ __device__ uint64_t tier_handle(int tier) {
    return params.tier_count > 0 ? params.tier_handles[tier] : params.handle;
}

static __forceinline__ __device__ float tier_search_radius(int tier) {
    return params.tier_count > 0 ? fmaxf(params.tier_search_radii[tier], 0.0f)
                                 : safe_search_radius();
}

static __forceinline__ __device__ void point_segment_distance(float3 point,
                                                              float3 p0,
                                                              float3 e1,
                                                              float &edge_t,
                                                              float &distance_sq) {
    const float edge_length_sq = squared_norm(e1);
    edge_t = edge_length_sq > 1.0e-7f ? clamp01(dot3(point - p0, e1) / edge_length_sq) : 0.0f;
    const float3 edge_point = p0 + e1 * edge_t;
    distance_sq = squared_norm(point - edge_point);
}

static __forceinline__ __device__ void update_segment_best(float3 query_origin,
                                                           float3 query_edge,
                                                           float3 edge_origin,
                                                           float3 edge_vector,
                                                           float query_t,
                                                           float edge_t,
                                                           bool enabled,
                                                           float &best_distance_sq,
                                                           float &best_query_t,
                                                           float &best_edge_t) {
    if (!enabled) {
        return;
    }

    const float3 query_point = query_origin + query_edge * query_t;
    const float3 edge_point = edge_origin + edge_vector * edge_t;
    const float distance_sq = squared_norm(query_point - edge_point);
    if (distance_sq < best_distance_sq) {
        best_distance_sq = distance_sq;
        best_query_t = query_t;
        best_edge_t = edge_t;
    }
}

static __forceinline__ __device__ void segment_segment_distance(float3 query_origin,
                                                                float3 query_edge,
                                                                float3 edge_origin,
                                                                float3 edge_vector,
                                                                float &query_t,
                                                                float &edge_t,
                                                                float &distance_sq) {
    const float3 w0 = query_origin - edge_origin;
    const float3 query_end = query_origin + query_edge;
    const float3 edge_end = edge_origin + edge_vector;

    const float a = squared_norm(query_edge);
    const float b = dot3(query_edge, edge_vector);
    const float c = squared_norm(edge_vector);
    const float d = dot3(query_edge, w0);
    const float e = dot3(edge_vector, w0);
    const float det = a * c - b * b;

    float best_distance_sq = 3.4028234663852886e38f;
    float best_query_t = 0.0f;
    float best_edge_t = 0.0f;

    float candidate_edge_t = 0.0f;
    float candidate_distance_sq = 0.0f;
    point_segment_distance(query_origin, edge_origin, edge_vector,
                           candidate_edge_t, candidate_distance_sq);
    update_segment_best(query_origin, query_edge, edge_origin, edge_vector,
                        0.0f, candidate_edge_t, true,
                        best_distance_sq, best_query_t, best_edge_t);

    point_segment_distance(query_end, edge_origin, edge_vector,
                           candidate_edge_t, candidate_distance_sq);
    update_segment_best(query_origin, query_edge, edge_origin, edge_vector,
                        1.0f, candidate_edge_t, true,
                        best_distance_sq, best_query_t, best_edge_t);

    float candidate_query_t = 0.0f;
    point_segment_distance(edge_origin, query_origin, query_edge,
                           candidate_query_t, candidate_distance_sq);
    update_segment_best(query_origin, query_edge, edge_origin, edge_vector,
                        candidate_query_t, 0.0f, true,
                        best_distance_sq, best_query_t, best_edge_t);

    point_segment_distance(edge_end, query_origin, query_edge,
                           candidate_query_t, candidate_distance_sq);
    update_segment_best(query_origin, query_edge, edge_origin, edge_vector,
                        candidate_query_t, 1.0f, true,
                        best_distance_sq, best_query_t, best_edge_t);

    const bool interior = a > 1.0e-7f && c > 1.0e-7f && fabsf(det) > 1.0e-7f;
    if (interior) {
        const float query_t_line = (b * e - c * d) / det;
        const float edge_t_line = (a * e - b * d) / det;
        update_segment_best(query_origin, query_edge, edge_origin, edge_vector,
                            query_t_line,
                            edge_t_line,
                            query_t_line >= 0.0f && query_t_line <= 1.0f &&
                                edge_t_line >= 0.0f && edge_t_line <= 1.0f,
                            best_distance_sq, best_query_t, best_edge_t);
    }

    query_t = best_query_t;
    edge_t = best_edge_t;
    distance_sq = best_distance_sq;
}

static __forceinline__ __device__ void write_invalid(unsigned int query) {
    params.out_edge_ids[query] = -1;
    params.out_distance_sq[query] = 3.4028234663852886e38f;
    if (params.out_ray_t != nullptr) {
        params.out_ray_t[query] = 0.0f;
    }
    params.out_edge_t[query] = 0.0f;
    if (params.out_valid != nullptr) {
        params.out_valid[query] = 0u;
    }
}

static __forceinline__ __device__ void write_final_point_output(unsigned int query,
                                                                unsigned int edge,
                                                                float distance_sq,
                                                                float edge_t) {
    if (params.write_point_outputs == 0) {
        return;
    }

    const float s = clamp01(edge_t);
    const float3 p = load_query_point(query);
    const float3 a = load_edge_start(edge);
    const float3 e = load_edge_vector(edge);
    const float3 q = a + e * s;
    const float3 d = p - q;

    params.final_distance[query] = sqrtf(fmaxf(distance_sq, 0.0f));
    params.final_edge_t[query] = s;
    params.final_shape_id[query] = params.edge_shape_id[edge];
    params.final_edge_id[query] = params.edge_local_id[edge];
    params.final_global_edge_id[query] = static_cast<int>(edge);
    params.final_edge_point[query * 3 + 0] = q.x;
    params.final_edge_point[query * 3 + 1] = q.y;
    params.final_edge_point[query * 3 + 2] = q.z;
    if (params.final_tape_edge_id != nullptr) {
        params.final_tape_edge_id[query] = static_cast<int>(edge);
    }
    if (params.final_tape_s != nullptr) {
        params.final_tape_s[query] = s;
    }
    if (params.final_tape_d != nullptr) {
        params.final_tape_d[query * 3 + 0] = d.x;
        params.final_tape_d[query * 3 + 1] = d.y;
        params.final_tape_d[query * 3 + 2] = d.z;
    }
    if (params.final_unresolved != nullptr) {
        params.final_unresolved[query] = 0u;
    }
}

#ifndef RAYDN_EDGE_POINT_RAY_ONLY
static __forceinline__ __device__ void insert_topk_candidate(unsigned int query,
                                                             int edge_id,
                                                             float distance_sq,
                                                             float edge_t) {
    const int k = params.k;
    if (k <= 0 || k > EdgeOptixTopKMax) {
        return;
    }

    const int base = static_cast<int>(query) * k;
    if (distance_sq >= params.out_distance_sq[base + k - 1]) {
        return;
    }

    int insert = k - 1;
    while (insert > 0 && distance_sq < params.out_distance_sq[base + insert - 1]) {
        params.out_distance_sq[base + insert] = params.out_distance_sq[base + insert - 1];
        params.out_edge_ids[base + insert] = params.out_edge_ids[base + insert - 1];
        params.out_edge_t[base + insert] = params.out_edge_t[base + insert - 1];
        params.out_valid[base + insert] = params.out_valid[base + insert - 1];
        --insert;
    }

    params.out_distance_sq[base + insert] = distance_sq;
    params.out_edge_ids[base + insert] = edge_id;
    params.out_edge_t[base + insert] = edge_t;
    params.out_valid[base + insert] = 1u;
}

static __forceinline__ __device__ uint32_t get_topk_payload_id(int slot) {
    switch (slot) {
    case 0: return optixGetPayload_0();
    case 1: return optixGetPayload_1();
    case 2: return optixGetPayload_2();
    case 3: return optixGetPayload_3();
    case 4: return optixGetPayload_4();
    case 5: return optixGetPayload_5();
    case 6: return optixGetPayload_6();
    default: return optixGetPayload_7();
    }
}

static __forceinline__ __device__ uint32_t get_topk_payload_distance(int slot) {
    switch (slot) {
    case 0: return optixGetPayload_8();
    case 1: return optixGetPayload_9();
    case 2: return optixGetPayload_10();
    case 3: return optixGetPayload_11();
    case 4: return optixGetPayload_12();
    case 5: return optixGetPayload_13();
    case 6: return optixGetPayload_14();
    default: return optixGetPayload_15();
    }
}

static __forceinline__ __device__ void set_topk_payload_slot(int slot,
                                                             uint32_t edge_id,
                                                             uint32_t distance_sq) {
    switch (slot) {
    case 0:
        optixSetPayload_0(edge_id);
        optixSetPayload_8(distance_sq);
        break;
    case 1:
        optixSetPayload_1(edge_id);
        optixSetPayload_9(distance_sq);
        break;
    case 2:
        optixSetPayload_2(edge_id);
        optixSetPayload_10(distance_sq);
        break;
    case 3:
        optixSetPayload_3(edge_id);
        optixSetPayload_11(distance_sq);
        break;
    case 4:
        optixSetPayload_4(edge_id);
        optixSetPayload_12(distance_sq);
        break;
    case 5:
        optixSetPayload_5(edge_id);
        optixSetPayload_13(distance_sq);
        break;
    case 6:
        optixSetPayload_6(edge_id);
        optixSetPayload_14(distance_sq);
        break;
    default:
        optixSetPayload_7(edge_id);
        optixSetPayload_15(distance_sq);
        break;
    }
}

static __forceinline__ __device__ void insert_topk_payload_candidate(int edge_id,
                                                                     float distance_sq) {
    const int k = params.k;
    if (k <= 0 || k > 8) {
        return;
    }

    const uint32_t candidate_distance = __float_as_uint(distance_sq);
    if (distance_sq >= __uint_as_float(get_topk_payload_distance(k - 1))) {
        return;
    }

    int insert = k - 1;
    while (insert > 0 &&
           distance_sq < __uint_as_float(get_topk_payload_distance(insert - 1))) {
        set_topk_payload_slot(insert,
                              get_topk_payload_id(insert - 1),
                              get_topk_payload_distance(insert - 1));
        --insert;
    }

    set_topk_payload_slot(insert,
                          static_cast<uint32_t>(edge_id),
                          candidate_distance);
}
#endif

} // namespace

// OptiX programs for the custom-AABB edge backend. Each raygen launch handles one
// query (launch index x); intersection programs report the point/segment-to-edge
// distance, and the anyhit/closesthit programs keep the running nearest edge.

/// IntersectionAD for point queries: report the point-to-edge distance if within the search radius.
#ifndef RAYDN_EDGE_TOPK_ONLY
extern "C" __global__ void __intersection__edge_point() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    point_segment_distance(optixGetWorldRayOrigin(),
                           load_edge_start(edge),
                           load_edge_vector(edge),
                           edge_t,
                           distance_sq);
    const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
    if (distance <= optixGetRayTmax() && distance <= safe_search_radius()) {
        optixReportIntersection(distance, 0u, __float_as_uint(edge_t));
    }
}

/// IntersectionAD for ray queries: report the ray-to-edge closest approach within the search radius.
extern "C" __global__ void __intersection__edge_ray() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    const float trace_tmax = optixGetRayTmax();
    const float payload_radius = __uint_as_float(optixGetPayload_4());
    const float search_radius = payload_radius > 0.0f ? payload_radius : safe_search_radius();
    float query_t = 0.0f;
    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    segment_segment_distance(optixGetWorldRayOrigin(),
                             optixGetWorldRayDirection() * trace_tmax,
                             load_edge_start(edge),
                             load_edge_vector(edge),
                             query_t,
                             edge_t,
                             distance_sq);
    const float ray_t = query_t * trace_tmax;
    if (ray_t >= optixGetRayTmin() && ray_t <= trace_tmax &&
        sqrtf(fmaxf(distance_sq, 0.0f)) <= search_radius) {
        optixReportIntersection(ray_t,
                                0u,
                                __float_as_uint(distance_sq),
                                __float_as_uint(ray_t),
                                __float_as_uint(edge_t));
    }
}
#endif

/// IntersectionAD for top-k point queries: report every edge within the search radius for ranking.
#ifndef RAYDN_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __intersection__edge_topk_point() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    point_segment_distance(optixGetWorldRayOrigin(),
                           load_edge_start(edge),
                           load_edge_vector(edge),
                           edge_t,
                           distance_sq);
    const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
    if (distance <= safe_search_radius()) {
        optixReportIntersection(kPointProbeTMax * 0.5f,
                                0u,
                                __float_as_uint(distance_sq),
                                __float_as_uint(edge_t));
    }
}
#endif

/// Closest-hit for point queries: publish the winning edge id, distance, and edge parameter to payload.
#ifndef RAYDN_EDGE_TOPK_ONLY
extern "C" __global__ void __closesthit__edge_point() {
    const float distance = optixGetRayTmax();
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(distance * distance));
    optixSetPayload_2(optixGetAttribute_0());
    optixSetPayload_3(1u);
}

/// Anyhit for ray queries: keep the nearest edge so far in payload, then ignore the hit to continue.
extern "C" __global__ void __anyhit__edge_ray() {
    const float candidate_distance_sq = __uint_as_float(optixGetAttribute_0());
    const float best_distance_sq = __uint_as_float(optixGetPayload_1());
    if (candidate_distance_sq < best_distance_sq) {
        optixSetPayload_0(optixGetPrimitiveIndex());
        optixSetPayload_1(__float_as_uint(candidate_distance_sq));
        optixSetPayload_2(optixGetAttribute_1());
        optixSetPayload_3(optixGetAttribute_2());
    }
    optixIgnoreIntersection();
}
#endif

/// Anyhit for top-k point queries: insert the candidate into the per-query top-k (payload for k<=8,
/// global buffer otherwise), then ignore the hit to keep traversing.
#ifndef RAYDN_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __anyhit__edge_topk_point() {
    if (params.k <= 8) {
        insert_topk_payload_candidate(static_cast<int>(optixGetPrimitiveIndex()),
                                      __uint_as_float(optixGetAttribute_0()));
    } else {
        insert_topk_candidate(optixGetLaunchIndex().x,
                              static_cast<int>(optixGetPrimitiveIndex()),
                              __uint_as_float(optixGetAttribute_0()),
                              __uint_as_float(optixGetAttribute_1()));
    }
    optixIgnoreIntersection();
}
#endif

/// Miss program: no edge within range; outputs are left at their invalid defaults.
extern "C" __global__ void __miss__edge_query() {
}

/// Raygen for point queries: trace a degenerate ray from each query point and write the nearest edge.
#ifndef RAYDN_EDGE_TOPK_ONLY
extern "C" __global__ void __raygen__edge_point() {
    const unsigned int query = optixGetLaunchIndex().x;
    if (query >= static_cast<unsigned int>(params.query_count) || !is_active(query) ||
        params.edge_count <= 0) {
        if (params.write_point_outputs == 0) {
            write_invalid(query);
        }
        return;
    }

    const int tiers = active_tier_count();
    for (int tier = 0; tier < tiers; ++tier) {
        const uint64_t handle = tier_handle(tier);
        const float radius = tier_search_radius(tier);
        if (handle == 0ull || !(radius > 0.0f)) {
            continue;
        }

        uint32_t edge_id = kInvalidEdgeId;
        uint32_t distance_sq = __float_as_uint(3.4028234663852886e38f);
        uint32_t edge_t = __float_as_uint(0.0f);
        uint32_t valid = 0u;
        optixTrace(static_cast<OptixTraversableHandle>(handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   radius,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   0,
                   1,
                   0,
                   edge_id,
                   distance_sq,
                   edge_t,
                   valid);

        if (valid == 0u || edge_id == kInvalidEdgeId) {
            continue;
        }

        if (params.write_point_outputs != 0) {
            write_final_point_output(query,
                                     edge_id,
                                     __uint_as_float(distance_sq),
                                     __uint_as_float(edge_t));
        } else {
            params.out_edge_ids[query] = static_cast<int>(edge_id);
            params.out_distance_sq[query] = __uint_as_float(distance_sq);
            params.out_edge_t[query] = __uint_as_float(edge_t);
            if (params.out_ray_t != nullptr) {
                params.out_ray_t[query] = 0.0f;
            }
            if (params.out_valid != nullptr) {
                params.out_valid[query] = 1u;
            }
        }
        return;
    }

    if (params.write_point_outputs == 0) {
        write_invalid(query);
    }
}

/// Raygen for ray queries: trace each query ray (anyhit-enforced) and write the nearest edge.
extern "C" __global__ void __raygen__edge_ray() {
    const unsigned int query = optixGetLaunchIndex().x;
    if (query >= static_cast<unsigned int>(params.query_count) || !is_active(query) ||
        params.edge_count <= 0) {
        write_invalid(query);
        return;
    }

    float trace_tmax = params.ray_tmax != nullptr ? params.ray_tmax[query] : kInfiniteRayTMax;
    if (!(trace_tmax > 0.0f) || isinf(trace_tmax)) {
        trace_tmax = kInfiniteRayTMax;
    }

    const int tiers = active_tier_count();
    for (int tier = 0; tier < tiers; ++tier) {
        const uint64_t handle = tier_handle(tier);
        const float radius = tier_search_radius(tier);
        if (handle == 0ull || !(radius > 0.0f)) {
            continue;
        }

        uint32_t edge_id = kInvalidEdgeId;
        uint32_t distance_sq = __float_as_uint(3.4028234663852886e38f);
        uint32_t ray_t = __float_as_uint(0.0f);
        uint32_t edge_t = __float_as_uint(0.0f);
        uint32_t radius_bits = __float_as_uint(radius);
        optixTrace(static_cast<OptixTraversableHandle>(handle),
                   load_query_point(query),
                   load_ray_direction(query),
                   0.0f,
                   trace_tmax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   1,
                   1,
                   0,
                   edge_id,
                   distance_sq,
                   ray_t,
                   edge_t,
                   radius_bits);

        if (edge_id == kInvalidEdgeId) {
            continue;
        }

        params.out_edge_ids[query] = static_cast<int>(edge_id);
        params.out_distance_sq[query] = __uint_as_float(distance_sq);
        params.out_ray_t[query] = __uint_as_float(ray_t);
        params.out_edge_t[query] = __uint_as_float(edge_t);
        if (params.out_valid != nullptr) {
            params.out_valid[query] = 1u;
        }
        return;
    }

    write_invalid(query);
}
#endif

/// Raygen for top-k point queries: initialize the per-query top-k slots, trace, and emit the sorted neighbors.
#ifndef RAYDN_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __raygen__edge_topk_point() {
    const unsigned int query = optixGetLaunchIndex().x;
    const int k = params.k;
    if (query >= static_cast<unsigned int>(params.query_count) || k <= 0 || k > EdgeOptixTopKMax) {
        return;
    }

    const int base = static_cast<int>(query) * k;
    for (int slot = 0; slot < EdgeOptixTopKMax; ++slot) {
        if (slot < k) {
            params.out_edge_ids[base + slot] = -1;
            params.out_distance_sq[base + slot] = 3.4028234663852886e38f;
            params.out_edge_t[base + slot] = 0.0f;
            params.out_valid[base + slot] = 0u;
        }
    }

    if (!is_active(query) || params.handle == 0ull || params.edge_count <= 0) {
        return;
    }

    if (k <= 8) {
        uint32_t edge0 = kInvalidEdgeId;
        uint32_t edge1 = kInvalidEdgeId;
        uint32_t edge2 = kInvalidEdgeId;
        uint32_t edge3 = kInvalidEdgeId;
        uint32_t edge4 = kInvalidEdgeId;
        uint32_t edge5 = kInvalidEdgeId;
        uint32_t edge6 = kInvalidEdgeId;
        uint32_t edge7 = kInvalidEdgeId;
        uint32_t dist0 = __float_as_uint(3.4028234663852886e38f);
        uint32_t dist1 = dist0;
        uint32_t dist2 = dist0;
        uint32_t dist3 = dist0;
        uint32_t dist4 = dist0;
        uint32_t dist5 = dist0;
        uint32_t dist6 = dist0;
        uint32_t dist7 = dist0;

        optixTrace(static_cast<OptixTraversableHandle>(params.handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   kPointProbeTMax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   0,
                   1,
                   0,
                   edge0,
                   edge1,
                   edge2,
                   edge3,
                   edge4,
                   edge5,
                   edge6,
                   edge7,
                   dist0,
                   dist1,
                   dist2,
                   dist3,
                   dist4,
                   dist5,
                   dist6,
                   dist7);

        uint32_t edges[8] = { edge0, edge1, edge2, edge3, edge4, edge5, edge6, edge7 };
        uint32_t distances[8] = { dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7 };
        for (int slot = 0; slot < k; ++slot) {
            const int out_index = base + slot;
            const bool valid = edges[slot] != kInvalidEdgeId;
            params.out_edge_ids[out_index] = valid ? static_cast<int>(edges[slot]) : -1;
            params.out_distance_sq[out_index] = valid
                ? __uint_as_float(distances[slot])
                : 3.4028234663852886e38f;
            params.out_edge_t[out_index] = 0.0f;
            params.out_valid[out_index] = valid ? 1u : 0u;
        }
    } else {
        uint32_t dummy = 0u;
        optixTrace(static_cast<OptixTraversableHandle>(params.handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   kPointProbeTMax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   0,
                   1,
                   0,
                   dummy);
    }
}
#endif

} // namespace raydn
