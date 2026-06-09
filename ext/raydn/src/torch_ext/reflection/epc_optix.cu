#include <optix.h>
#include <optix_device.h>

#include <raydn/common/math.cuh>
#include <raydn/reflection/epc_params.h>

namespace raydn {

extern "C" {
extern __constant__ ReflEpcParams params;
}

namespace {

constexpr float kMinSegmentLength = 2e-5f;
constexpr float kEpcTolerance = 1e-4f;
constexpr unsigned int kInvalidPrim = 0xFFFFFFFFu;
constexpr unsigned int kTraceModeReflection = 0u;
constexpr unsigned int kTraceModeVisibility = 1u;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

struct VisibilityPayload {
    unsigned int visible = 1u;
    unsigned int blocker = kInvalidPrim;
};

static __forceinline__ __device__ void set_reflection_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.bary_u);
    optixSetPayload_3(payload.bary_v);
    optixSetPayload_4(payload.prim);
    optixSetPayload_5(payload.instance);
}

static __forceinline__ __device__ int global_prim_from_hit(int shape_id, int local_prim) {
    const int face_offset =
        (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
    return face_offset + local_prim;
}

static __forceinline__ __device__ bool has_surface_groups() {
    return params.surface_group_id != nullptr &&
           params.surface_group_size != nullptr &&
           params.surface_group_members != nullptr &&
           params.surface_group_count > 0 &&
           params.surface_max_group_size > 0;
}

static __forceinline__ __device__ int surface_group_for_prim(int prim) {
    if (!has_surface_groups() || prim < 0 || prim >= params.surface_group_id_count) {
        return -1;
    }
    const int group = params.surface_group_id[prim];
    return group >= 0 && group < params.surface_group_count ? group : -1;
}

static __forceinline__ __device__ int expected_prim_for_bounce(int slot) {
    if (params.expected_prim_ids == nullptr ||
        slot < 0 ||
        slot >= params.expected_prim_count) {
        return -1;
    }
    return params.expected_prim_ids[slot];
}

static __forceinline__ __device__ bool direct_plane_mode() {
    return params.direct_plane_point_x != nullptr &&
           params.direct_plane_point_y != nullptr &&
           params.direct_plane_point_z != nullptr &&
           params.direct_plane_normal_x != nullptr &&
           params.direct_plane_normal_y != nullptr &&
           params.direct_plane_normal_z != nullptr;
}

static __forceinline__ __device__ int final_ignore_group_for_ray(int ray_index) {
    if (params.final_ignore_group_ids == nullptr ||
        params.final_ignore_group_count <= 0) {
        return -1;
    }
    const int index = params.final_ignore_group_count == 1 ? 0 : ray_index;
    if (index < 0 || index >= params.final_ignore_group_count) {
        return -1;
    }
    return params.final_ignore_group_ids[index];
}

static __forceinline__ __device__ void trace_reflection_handle(OptixTraversableHandle handle,
                                                               float3 origin,
                                                               float3 direction,
                                                               float tmax,
                                                               HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(kRayTMax);
    payload.bary_u = 0u;
    payload.bary_v = 0u;
    payload.prim = 0u;
    payload.instance = kTraceModeReflection;
    if (handle == 0ull) {
        return;
    }

    optixTrace(handle,
               origin,
               direction,
               kRayTMin,
               tmax,
               0.0f,
               255u,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0,
               1,
               0,
               payload.hit,
               payload.t,
               payload.bary_u,
               payload.bary_v,
               payload.prim,
               payload.instance);
}

static __forceinline__ __device__ HitPayload choose_hit(const HitPayload &a,
                                                        const HitPayload &b) {
    if (a.hit == 0u) {
        return b;
    }
    if (b.hit == 0u) {
        return a;
    }
    return __uint_as_float(b.t) < __uint_as_float(a.t) ? b : a;
}

static __forceinline__ __device__ HitPayload trace_scene(float3 origin,
                                                         float3 direction,
                                                         float tmax) {
    HitPayload hit_primary;
    trace_reflection_handle(params.primary_handle, origin, direction, tmax, hit_primary);
    if (params.split_mode == 0) {
        return hit_primary;
    }
    HitPayload hit_secondary;
    trace_reflection_handle(params.secondary_handle, origin, direction, tmax, hit_secondary);
    return choose_hit(hit_primary, hit_secondary);
}

static __forceinline__ __device__ VisibilityPayload trace_visibility_handle(
    OptixTraversableHandle handle,
    float3 start,
    float3 end,
    unsigned int ignore0,
    unsigned int ignore1,
    unsigned int ignore2) {
    VisibilityPayload payload;
    if (handle == 0ull) {
        return payload;
    }

    float3 direction = end - start;
    const float length = length3(direction);
    if (length <= kMinSegmentLength) {
        return payload;
    }

    direction = (1.f / length) * direction;
    const float3 origin = start + kRayBias * direction;
    const float tmax = fmaxf(length - 2.f * kRayBias, 0.f);
    unsigned int mode = kTraceModeVisibility;
    const bool has_ignore =
        ignore0 != kInvalidPrim || ignore1 != kInvalidPrim || ignore2 != kInvalidPrim;
    const unsigned int ray_flags =
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        (has_ignore ? 0u : OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    optixTrace(handle,
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
               payload.visible,
               payload.blocker,
               ignore0,
               ignore1,
               ignore2,
               mode);
    return payload;
}

static __forceinline__ __device__ VisibilityPayload trace_visibility(float3 start,
                                                                     float3 end,
                                                                     int ignore0,
                                                                     int ignore1,
                                                                     int ignore2) {
    const unsigned int ignore0_u = ignore0 >= 0 ? static_cast<unsigned int>(ignore0)
                                                : kInvalidPrim;
    const unsigned int ignore1_u = ignore1 >= 0 ? static_cast<unsigned int>(ignore1)
                                                : kInvalidPrim;
    const unsigned int ignore2_u = ignore2 >= 0 ? static_cast<unsigned int>(ignore2)
                                                : kInvalidPrim;

    VisibilityPayload primary =
        trace_visibility_handle(params.primary_handle, start, end, ignore0_u, ignore1_u, ignore2_u);
    if (primary.visible == 0u || params.split_mode == 0) {
        return primary;
    }

    return trace_visibility_handle(params.secondary_handle, start, end, ignore0_u, ignore1_u, ignore2_u);
}

static __forceinline__ __device__ VisibilityPayload trace_visibility_primary(float3 start,
                                                                             float3 end,
                                                                             int ignore0,
                                                                             int ignore1,
                                                                             int ignore2) {
    const unsigned int ignore0_u = ignore0 >= 0 ? static_cast<unsigned int>(ignore0)
                                                : kInvalidPrim;
    const unsigned int ignore1_u = ignore1 >= 0 ? static_cast<unsigned int>(ignore1)
                                                : kInvalidPrim;
    const unsigned int ignore2_u = ignore2 >= 0 ? static_cast<unsigned int>(ignore2)
                                                : kInvalidPrim;

    return trace_visibility_handle(params.primary_handle, start, end, ignore0_u, ignore1_u, ignore2_u);
}

static __forceinline__ __device__ float3 load_triangle_p0(int prim) {
    return make_f3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
}

static __forceinline__ __device__ float3 load_triangle_e1(int prim) {
    return make_f3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
}

static __forceinline__ __device__ float3 load_triangle_e2(int prim) {
    return make_f3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
}

static __forceinline__ __device__ float3 load_triangle_normal(int prim) {
    return normalize3(make_f3(params.tri_fn_x[prim],
                                params.tri_fn_y[prim],
                                params.tri_fn_z[prim]));
}

static __forceinline__ __device__ bool point_inside_triangle(int prim, float3 point) {
    if (prim < 0 || prim >= params.n_triangles) {
        return false;
    }
    const float3 p0 = load_triangle_p0(prim);
    const float3 e1 = load_triangle_e1(prim);
    const float3 e2 = load_triangle_e2(prim);
    const float3 vp = point - p0;
    const float d00 = dot3(e1, e1);
    const float d01 = dot3(e1, e2);
    const float d11 = dot3(e2, e2);
    const float d20 = dot3(vp, e1);
    const float d21 = dot3(vp, e2);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= 1e-12f) {
        return false;
    }
    const float inv_denom = 1.f / denom;
    const float u = (d11 * d20 - d01 * d21) * inv_denom;
    const float v = (d00 * d21 - d01 * d20) * inv_denom;
    return u >= -kEpcTolerance &&
           v >= -kEpcTolerance &&
           u + v <= 1.f + kEpcTolerance;
}

static __forceinline__ __device__ bool point_inside_surface_group(int group,
                                                                  float3 point,
                                                                  int &resolved_prim) {
    resolved_prim = -1;
    if (!has_surface_groups() ||
        group < 0 ||
        group >= params.surface_group_count ||
        params.surface_max_group_size <= 0) {
        return false;
    }

    int member_count = params.surface_group_size[group];
    if (member_count < 0) {
        member_count = 0;
    }
    if (member_count > params.surface_max_group_size) {
        member_count = params.surface_max_group_size;
    }
    const int base = group * params.surface_max_group_size;
    for (int i = 0; i < member_count; ++i) {
        const int prim = params.surface_group_members[base + i];
        if (prim < 0) {
            continue;
        }
        if (point_inside_triangle(prim, point)) {
            resolved_prim = prim;
            return true;
        }
    }
    return false;
}

static __forceinline__ __device__ bool intersect_line_plane(float3 line_start,
                                                            float3 line_end,
                                                            float3 plane_point,
                                                            float3 plane_normal,
                                                            float3 &point) {
    const float3 direction = line_end - line_start;
    const float denom = dot3(direction, plane_normal);
    if (fabsf(denom) <= 1e-7f) {
        return false;
    }
    const float t = dot3(plane_point - line_start, plane_normal) / denom;
    if (!isfinite(t) || t < -kEpcTolerance || t > 1.f + kEpcTolerance) {
        return false;
    }
    point = line_start + t * direction;
    return isfinite(point.x) && isfinite(point.y) && isfinite(point.z);
}

static __forceinline__ __device__ void store_invalid(unsigned int ray_index,
                                                     int bounce_count,
                                                     int first_blocked_segment,
                                                     int first_blocked_prim,
                                                     int first_blocked_group) {
    params.out_valid[ray_index] = 0u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = __uint_as_float(0x7f800000u);
    params.out_first_blocked_segment[ray_index] = first_blocked_segment;
    params.out_first_blocked_prim[ray_index] = first_blocked_prim;
    params.out_first_blocked_group[ray_index] = first_blocked_group;
}

} // namespace

extern "C" {
__constant__ ReflEpcParams params;
}

// OptiX programs for the native EPC pipeline; one raygen launch per ray. The same
// programs serve reflection tracing and visibility checks, switched by payload 5.

/// Anyhit (visibility mode only): skip occluders on the ignore list (primitive or surface group).
extern "C" __global__ void __anyhit__reflection_epc() {
    if (optixGetPayload_5() != kTraceModeVisibility) {
        return;
    }

    const int shape_id = static_cast<int>(optixGetInstanceId());
    const int global_prim =
        global_prim_from_hit(shape_id, static_cast<int>(optixGetPrimitiveIndex()));
    const unsigned int ignore0 = optixGetPayload_2();
    const unsigned int ignore1 = optixGetPayload_3();
    const unsigned int ignore2 = optixGetPayload_4();
    const int candidate =
        params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup
            ? surface_group_for_prim(global_prim)
            : global_prim;
    if ((ignore0 != kInvalidPrim && candidate == static_cast<int>(ignore0)) ||
        (ignore1 != kInvalidPrim && candidate == static_cast<int>(ignore1)) ||
        (ignore2 != kInvalidPrim && candidate == static_cast<int>(ignore2))) {
        optixIgnoreIntersection();
    }
}

/// Closest-hit: in visibility mode record the blocker; otherwise pack the reflection hit into payload.
extern "C" __global__ void __closesthit__reflection_epc() {
    if (optixGetPayload_5() == kTraceModeVisibility) {
        const int shape_id = static_cast<int>(optixGetInstanceId());
        const int global_prim =
            global_prim_from_hit(shape_id, static_cast<int>(optixGetPrimitiveIndex()));
        optixSetPayload_0(0u);
        optixSetPayload_1(static_cast<unsigned int>(global_prim));
        return;
    }

    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 bary = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(bary.x);
    payload.bary_v = __float_as_uint(bary.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_reflection_payload(payload);
}

/// Miss: in reflection mode mark "no hit"; in visibility mode a miss means unoccluded.
extern "C" __global__ void __miss__reflection_epc() {
    if (optixGetPayload_5() != kTraceModeVisibility) {
        optixSetPayload_0(0u);
    }
}

/// Raygen: trace the expected reflector sequence, apply equivalent-path correction, and check
/// segment visibility to the receiver, writing per-slot reflection geometry and per-ray validity.
template <bool DirectOnly, bool PrimaryVisibilityOnly>
static __forceinline__ __device__ void run_reflection_epc_raygen() {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const int B = params.max_bounces;
    const int base = static_cast<int>(ray_index) * B;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = 0.f;
        params.out_point_y[slot] = 0.f;
        params.out_point_z[slot] = 0.f;
        params.out_trace_prim_ids[slot] = -1;
        params.out_resolved_prim_ids[slot] = -1;
        params.out_surface_group_ids[slot] = -1;
        params.out_plane_normal_x[slot] = 0.f;
        params.out_plane_normal_y[slot] = 0.f;
        params.out_plane_normal_z[slot] = 0.f;
    }

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        store_invalid(ray_index, 0, -1, -1, -1);
        return;
    }

    float3 origin = make_f3(params.ray_ox[ray_index],
                              params.ray_oy[ray_index],
                              params.ray_oz[ray_index]);
    const int rx_id = params.rx_count == 1 ? 0 : static_cast<int>(ray_index);
    const float3 receiver = make_f3(params.rx_x[rx_id],
                                      params.rx_y[rx_id],
                                      params.rx_z[rx_id]);

    float3 plane_points[ReflEpcMaxBounces];
    float3 plane_normals[ReflEpcMaxBounces];
    int trace_prim_ids[ReflEpcMaxBounces];
    int resolved_prim_ids[ReflEpcMaxBounces];
    int surface_group_ids[ReflEpcMaxBounces];
    float3 image_sources[ReflEpcMaxBounces + 1];
    float3 reflection_points[ReflEpcMaxBounces];
    image_sources[0] = origin;

    int bounce_count = 0;
    float3 image_source = origin;

    if (direct_plane_mode()) {
        for (int bounce = 0; bounce < B; ++bounce) {
            const int slot = base + bounce;
            const int expected_prim = expected_prim_for_bounce(slot);
            const int expected_group = surface_group_for_prim(expected_prim);
            if (expected_prim < 0 ||
                expected_prim >= params.n_triangles ||
                !isfinite(params.direct_plane_point_x[slot]) ||
                !isfinite(params.direct_plane_point_y[slot]) ||
                !isfinite(params.direct_plane_point_z[slot]) ||
                !isfinite(params.direct_plane_normal_x[slot]) ||
                !isfinite(params.direct_plane_normal_y[slot]) ||
                !isfinite(params.direct_plane_normal_z[slot])) {
                store_invalid(ray_index, bounce_count, -1, -1, -1);
                return;
            }

            const float3 plane_point =
                make_f3(params.direct_plane_point_x[slot],
                          params.direct_plane_point_y[slot],
                          params.direct_plane_point_z[slot]);
            const float3 plane_normal =
                normalize3(make_f3(params.direct_plane_normal_x[slot],
                                     params.direct_plane_normal_y[slot],
                                     params.direct_plane_normal_z[slot]));
            if (length3(plane_normal) <= 0.f) {
                store_invalid(ray_index, bounce_count, -1, -1, -1);
                return;
            }

            const float image_distance = dot3(image_source - plane_point, plane_normal);
            image_source = image_source - 2.f * image_distance * plane_normal;
            image_sources[bounce + 1] = image_source;
            plane_points[bounce] = plane_point;
            plane_normals[bounce] = plane_normal;
            trace_prim_ids[bounce] = expected_prim;
            resolved_prim_ids[bounce] = -1;
            surface_group_ids[bounce] = expected_group;
            ++bounce_count;
        }
    } else {
        if constexpr (DirectOnly) {
            store_invalid(ray_index, bounce_count, -1, -1, -1);
            return;
        } else {
        float3 trace_origin = origin;
        float3 trace_direction = normalize3(make_f3(params.ray_dx[ray_index],
                                                      params.ray_dy[ray_index],
                                                      params.ray_dz[ray_index]));

        for (int bounce = 0; bounce < B; ++bounce) {
            const float tmax_input =
                bounce == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kRayTMax;
            const float trace_tmax = isfinite(tmax_input) ? tmax_input : kRayTMax;
            const HitPayload hit = trace_scene(trace_origin, trace_direction, trace_tmax);
            if (hit.hit == 0u) {
                break;
            }

            const int shape_id = static_cast<int>(hit.instance);
            const int local_prim = static_cast<int>(hit.prim);
            const int global_prim = global_prim_from_hit(shape_id, local_prim);
            const int actual_group = surface_group_for_prim(global_prim);
            const int slot = base + bounce;
            const int expected_prim = expected_prim_for_bounce(slot);
            const int expected_group = surface_group_for_prim(expected_prim);
            const bool expected_matches =
                expected_prim < 0 ||
                (has_surface_groups()
                     ? (actual_group >= 0 && actual_group == expected_group)
                     : (global_prim == expected_prim));
            if (!expected_matches) {
                store_invalid(ray_index, bounce_count, -1, -1, -1);
                return;
            }
            const float bary_u = __uint_as_float(hit.bary_u);
            const float bary_v = __uint_as_float(hit.bary_v);
            const float t = __uint_as_float(hit.t);

            float3 hit_point = trace_origin + t * trace_direction;
            float3 geo_normal = make_f3(0.f, 0.f, 1.f);
            if (global_prim >= 0 && global_prim < params.n_triangles) {
                const float3 p0 = load_triangle_p0(global_prim);
                const float3 e1 = load_triangle_e1(global_prim);
                const float3 e2 = load_triangle_e2(global_prim);
                hit_point = p0 + bary_u * e1 + bary_v * e2;
                geo_normal = load_triangle_normal(global_prim);
            }
            if (dot3(trace_direction, geo_normal) > 0.f) {
                geo_normal = -1.f * geo_normal;
            }

            const float image_distance = dot3(image_source - hit_point, geo_normal);
            image_source = image_source - 2.f * image_distance * geo_normal;
            image_sources[bounce + 1] = image_source;
            plane_points[bounce] = hit_point;
            plane_normals[bounce] = geo_normal;
            trace_prim_ids[bounce] = global_prim;
            resolved_prim_ids[bounce] = -1;
            surface_group_ids[bounce] =
                expected_group >= 0 ? expected_group : actual_group;
            ++bounce_count;

            const float ray_dot_normal = dot3(trace_direction, geo_normal);
            trace_direction = normalize3(trace_direction - 2.f * ray_dot_normal * geo_normal);
            trace_origin = hit_point + kRayBias * trace_direction;
        }
        }
    }

    if (bounce_count != B) {
        store_invalid(ray_index, bounce_count, -1, -1, -1);
        return;
    }

    float3 endpoint = receiver;
    bool valid = true;
    for (int bounce = B - 1; bounce >= 0; --bounce) {
        float3 point;
        valid = valid &&
                intersect_line_plane(image_sources[bounce + 1],
                                     endpoint,
                                     plane_points[bounce],
                                     plane_normals[bounce],
                                     point);
        int resolved_prim = -1;
        if (valid && has_surface_groups() && surface_group_ids[bounce] >= 0) {
            valid = point_inside_surface_group(surface_group_ids[bounce],
                                               point,
                                               resolved_prim);
        } else if (valid) {
            const int expected_prim = expected_prim_for_bounce(base + bounce);
            const int containment_prim =
                expected_prim >= 0 ? expected_prim : trace_prim_ids[bounce];
            valid = point_inside_triangle(containment_prim, point);
            resolved_prim = valid ? containment_prim : -1;
        }
        resolved_prim_ids[bounce] = resolved_prim;
        reflection_points[bounce] = point;
        endpoint = point;
    }
    if (!valid) {
        store_invalid(ray_index, bounce_count, -1, -1, -1);
        return;
    }

    float path_length = 0.f;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;
    int first_blocked_group = -1;
    const int final_ignore_group =
        final_ignore_group_for_ray(static_cast<int>(ray_index));
    for (int segment = 0; segment <= B; ++segment) {
        const float3 start = segment == 0 ? origin : reflection_points[segment - 1];
        const float3 end = segment == B ? receiver : reflection_points[segment];
        path_length += length3(end - start);

        const bool ignore_surface_group =
            params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup &&
            has_surface_groups();
        const int ignore0 = segment > 0
                                ? (ignore_surface_group
                                       ? surface_group_ids[segment - 1]
                                       : resolved_prim_ids[segment - 1])
                                : -1;
        const int ignore1 = segment < B
                                ? (ignore_surface_group
                                       ? surface_group_ids[segment]
                                       : resolved_prim_ids[segment])
                                : -1;
        const int ignore2 =
            ignore_surface_group && segment == B ? final_ignore_group : -1;
        VisibilityPayload visibility;
        if constexpr (PrimaryVisibilityOnly) {
            visibility = trace_visibility_primary(start, end, ignore0, ignore1, ignore2);
        } else {
            visibility = trace_visibility(start, end, ignore0, ignore1, ignore2);
        }
        if (visibility.visible == 0u) {
            first_blocked_segment = segment;
            first_blocked_prim = static_cast<int>(visibility.blocker);
            first_blocked_group = surface_group_for_prim(first_blocked_prim);
            valid = false;
            break;
        }
    }

    if (!valid) {
        store_invalid(ray_index,
                      bounce_count,
                      first_blocked_segment,
                      first_blocked_prim,
                      first_blocked_group);
        return;
    }

    params.out_valid[ray_index] = 1u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = path_length;
    params.out_first_blocked_segment[ray_index] = -1;
    params.out_first_blocked_prim[ray_index] = -1;
    params.out_first_blocked_group[ray_index] = -1;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = reflection_points[bounce].x;
        params.out_point_y[slot] = reflection_points[bounce].y;
        params.out_point_z[slot] = reflection_points[bounce].z;
        params.out_trace_prim_ids[slot] = trace_prim_ids[bounce];
        params.out_resolved_prim_ids[slot] = resolved_prim_ids[bounce];
        params.out_surface_group_ids[slot] = surface_group_ids[bounce];
        params.out_plane_normal_x[slot] = plane_normals[bounce].x;
        params.out_plane_normal_y[slot] = plane_normals[bounce].y;
        params.out_plane_normal_z[slot] = plane_normals[bounce].z;
    }
}

extern "C" __global__ void __raygen__reflection_epc() {
    run_reflection_epc_raygen<false, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct() {
    run_reflection_epc_raygen<true, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct_primary() {
    run_reflection_epc_raygen<true, true>();
}

} // namespace raydn
