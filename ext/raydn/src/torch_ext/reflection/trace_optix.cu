#include <optix.h>
#include <optix_device.h>

#include <raydn/common/math.cuh>
#include <raydn/reflection/trace_params.h>

namespace raydn {

namespace {

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

static __forceinline__ __device__ void clear_payload(HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(kRayTMax);
    payload.bary_u = 0u;
    payload.bary_v = 0u;
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.bary_u);
    optixSetPayload_3(payload.bary_v);
    optixSetPayload_4(payload.prim);
    optixSetPayload_5(payload.instance);
}

static __forceinline__ __device__ void trace_handle(
    OptixTraversableHandle handle,
    float3 origin,
    float3 direction,
    float tmax,
    HitPayload &payload) {
    clear_payload(payload);
    if (handle == 0ull)
        return;

    optixTrace(
        handle,
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

static __forceinline__ __device__ HitPayload choose_hit(
    const HitPayload &a,
    const HitPayload &b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return __uint_as_float(b.t) < __uint_as_float(a.t) ? b : a;
}

} // namespace

extern "C" {
__constant__ ReflectionTraceParams params;
}

namespace {

static __forceinline__ __device__ int output_slot(unsigned int ray_index, int bounce) {
    if (params.output_layout != 0)
        return bounce * params.n_rays + static_cast<int>(ray_index);
    return static_cast<int>(ray_index) * params.max_bounces + bounce;
}

static __forceinline__ __device__ void write_empty_output_slot(unsigned int ray_index, int bounce) {
    const int slot = output_slot(ray_index, bounce);
    if (params.out_t != nullptr)
        params.out_t[slot] = kRayTMax;
    if (params.out_shape_ids != nullptr)
        params.out_shape_ids[slot] = -1;
    if (params.out_prim_ids != nullptr)
        params.out_prim_ids[slot] = -1;
    if (params.out_global_prim_ids != nullptr)
        params.out_global_prim_ids[slot] = -1;
    if (params.out_valid != nullptr)
        params.out_valid[ray_index * params.max_bounces + bounce] = 0u;
    if (params.out_bary != nullptr) {
        params.out_bary[slot * 3 + 0] = 0.0f;
        params.out_bary[slot * 3 + 1] = 0.0f;
        params.out_bary[slot * 3 + 2] = 0.0f;
    }
    if (params.out_hit != nullptr) {
        params.out_hit[slot * 3 + 0] = 0.0f;
        params.out_hit[slot * 3 + 1] = 0.0f;
        params.out_hit[slot * 3 + 2] = 0.0f;
    }
    if (params.out_norm != nullptr) {
        params.out_norm[slot * 3 + 0] = 0.0f;
        params.out_norm[slot * 3 + 1] = 0.0f;
        params.out_norm[slot * 3 + 2] = 0.0f;
    }
    if (params.out_img != nullptr) {
        params.out_img[slot * 3 + 0] = 0.0f;
        params.out_img[slot * 3 + 1] = 0.0f;
        params.out_img[slot * 3 + 2] = 0.0f;
    }
}

struct TriangleData {
    float3 p0;
    float3 e1;
    float3 e2;
    float3 fn;
};

static __forceinline__ __device__ float3 f3_from_f4(float4 value) {
    return make_f3(value.x, value.y, value.z);
}

static __forceinline__ __device__ TriangleData load_triangle_data(int prim) {
    TriangleData tri;
    if (params.tri_p0_packed != nullptr &&
        params.tri_e1_packed != nullptr &&
        params.tri_e2_packed != nullptr &&
        params.tri_fn_packed != nullptr) {
        tri.p0 = f3_from_f4(params.tri_p0_packed[prim]);
        tri.e1 = f3_from_f4(params.tri_e1_packed[prim]);
        tri.e2 = f3_from_f4(params.tri_e2_packed[prim]);
        tri.fn = f3_from_f4(params.tri_fn_packed[prim]);
        return tri;
    }

    tri.p0 = make_f3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
    tri.e1 = make_f3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
    tri.e2 = make_f3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
    tri.fn = make_f3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]);
    return tri;
}

} // namespace

extern "C" __global__ void __closesthit__reflection() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    const float2 bary = optixGetTriangleBarycentrics();
    payload.bary_u = __float_as_uint(bary.x);
    payload.bary_v = __float_as_uint(bary.y);
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__reflection() {
    optixSetPayload_0(0u);
}

extern "C" __global__ void __raygen__reflection_trace() {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        for (int bounce = 0; bounce < params.max_bounces; ++bounce)
            write_empty_output_slot(ray_index, bounce);
        if (params.out_bounce_count != nullptr)
            params.out_bounce_count[ray_index] = 0;
        return;
    }

    const int B = params.max_bounces;

    float3 origin;
    float3 direction;
    if (params.ray_o_aos != nullptr) {
        origin = make_f3(params.ray_o_aos + ray_index * 3);
        direction = make_f3(params.ray_d_aos + ray_index * 3);
    } else {
        origin = make_f3(
            params.ray_ox[ray_index],
            params.ray_oy[ray_index],
            params.ray_oz[ray_index]);
        direction = make_f3(
            params.ray_dx[ray_index],
            params.ray_dy[ray_index],
            params.ray_dz[ray_index]);
    }
    float3 image_source = origin;
    int bounce_count = 0;

    for (int bounce = 0; bounce < B; ++bounce) {
        const float tmax_input =
            bounce == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kRayTMax;
        const float trace_tmax = isfinite(tmax_input) ? tmax_input : kRayTMax;

        HitPayload hit_primary;
        trace_handle(params.primary_handle, origin, direction, trace_tmax, hit_primary);

        HitPayload hit = hit_primary;
        if (params.split_mode != 0) {
            HitPayload hit_secondary;
            trace_handle(params.secondary_handle, origin, direction, trace_tmax, hit_secondary);
            hit = choose_hit(hit_primary, hit_secondary);
        }

        if (hit.hit == 0u)
            break;

        const int shape_id = static_cast<int>(hit.instance);
        const int local_prim = static_cast<int>(hit.prim);
        const int face_offset =
            (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;

        const float t = __uint_as_float(hit.t);
        const float bary_u = __uint_as_float(hit.bary_u);
        const float bary_v = __uint_as_float(hit.bary_v);

        float3 hit_point = origin + t * direction;
        float3 geo_normal = make_f3(0.0f, 0.0f, 1.0f);

        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const TriangleData tri = load_triangle_data(global_prim);
            hit_point = tri.p0 + bary_u * tri.e1 + bary_v * tri.e2;
            geo_normal = normalize3(tri.fn);
        }

        if (dot3(direction, geo_normal) > 0.0f)
            geo_normal = -1.0f * geo_normal;

        const bool write_image_source =
            params.out_img != nullptr ||
            (params.out_img_x != nullptr &&
             params.out_img_y != nullptr &&
             params.out_img_z != nullptr);
        if (write_image_source) {
            const float image_distance = dot3(image_source - hit_point, geo_normal);
            image_source = image_source - 2.0f * image_distance * geo_normal;
        }

        const int slot = output_slot(ray_index, bounce);
        if (params.out_valid != nullptr)
            params.out_valid[ray_index * params.max_bounces + bounce] = 1u;
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = shape_id;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = local_prim;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = global_prim;
        if (params.out_t != nullptr)
            params.out_t[slot] = t;
        if (params.out_bary_u != nullptr)
            params.out_bary_u[slot] = bary_u;
        if (params.out_bary_v != nullptr)
            params.out_bary_v[slot] = bary_v;
        if (params.out_bary != nullptr) {
            params.out_bary[slot * 3 + 0] = 1.0f - bary_u - bary_v;
            params.out_bary[slot * 3 + 1] = bary_u;
            params.out_bary[slot * 3 + 2] = bary_v;
        }
        if (params.out_hit_x != nullptr)
            params.out_hit_x[slot] = hit_point.x;
        if (params.out_hit_y != nullptr)
            params.out_hit_y[slot] = hit_point.y;
        if (params.out_hit_z != nullptr)
            params.out_hit_z[slot] = hit_point.z;
        if (params.out_hit != nullptr) {
            params.out_hit[slot * 3 + 0] = hit_point.x;
            params.out_hit[slot * 3 + 1] = hit_point.y;
            params.out_hit[slot * 3 + 2] = hit_point.z;
        }
        if (params.out_norm_x != nullptr)
            params.out_norm_x[slot] = geo_normal.x;
        if (params.out_norm_y != nullptr)
            params.out_norm_y[slot] = geo_normal.y;
        if (params.out_norm_z != nullptr)
            params.out_norm_z[slot] = geo_normal.z;
        if (params.out_norm != nullptr) {
            params.out_norm[slot * 3 + 0] = geo_normal.x;
            params.out_norm[slot * 3 + 1] = geo_normal.y;
            params.out_norm[slot * 3 + 2] = geo_normal.z;
        }
        if (write_image_source) {
            if (params.out_img != nullptr) {
                params.out_img[slot * 3 + 0] = image_source.x;
                params.out_img[slot * 3 + 1] = image_source.y;
                params.out_img[slot * 3 + 2] = image_source.z;
            } else {
                params.out_img_x[slot] = image_source.x;
                params.out_img_y[slot] = image_source.y;
                params.out_img_z[slot] = image_source.z;
            }
        }

        const float dot_dn = dot3(direction, geo_normal);
        direction = direction - 2.0f * dot_dn * geo_normal;
        origin = hit_point + kRayBias * direction;
        bounce_count = bounce + 1;
    }

    for (int bounce = bounce_count; bounce < B; ++bounce)
        write_empty_output_slot(ray_index, bounce);

    if (bounce_count > 0 && params.return_trailing != 0) {
        if (params.out_trailing_dir_x != nullptr)
            params.out_trailing_dir_x[ray_index] = direction.x;
        if (params.out_trailing_dir_y != nullptr)
            params.out_trailing_dir_y[ray_index] = direction.y;
        if (params.out_trailing_dir_z != nullptr)
            params.out_trailing_dir_z[ray_index] = direction.z;
        if (params.out_trailing_origin_x != nullptr)
            params.out_trailing_origin_x[ray_index] = origin.x;
        if (params.out_trailing_origin_y != nullptr)
            params.out_trailing_origin_y[ray_index] = origin.y;
        if (params.out_trailing_origin_z != nullptr)
            params.out_trailing_origin_z[ray_index] = origin.z;

        HitPayload trailing_primary;
        trace_handle(params.primary_handle, origin, direction, kRayTMax, trailing_primary);

        HitPayload trailing = trailing_primary;
        if (params.split_mode != 0) {
            HitPayload trailing_secondary;
            trace_handle(params.secondary_handle, origin, direction, kRayTMax, trailing_secondary);
            trailing = choose_hit(trailing_primary, trailing_secondary);
        }

        if (trailing.hit != 0u) {
            const int shape_id = static_cast<int>(trailing.instance);
            const int local_prim = static_cast<int>(trailing.prim);
            const int face_offset =
                (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
            if (params.out_trailing_t != nullptr)
                params.out_trailing_t[ray_index] = __uint_as_float(trailing.t);
            if (params.out_trailing_prim != nullptr)
                params.out_trailing_prim[ray_index] = face_offset + local_prim;
        }
    }

    if (params.out_bounce_count != nullptr)
        params.out_bounce_count[ray_index] = bounce_count;
}

} // namespace raydn
