#include <raydn/scene/optix_intersect_params.h>

#include <optix_device.h>

extern "C" {
__constant__ raydn::OptixIntersectParams params;
}

extern "C" __global__ void __raygen__intersect() {
    const unsigned int ray_idx = optixGetLaunchIndex().x;
    if (ray_idx >= static_cast<unsigned int>(params.ray_count))
        return;

    float t = __uint_as_float(0x7f800000u);
    int shape_id = -1;
    int local_prim_id = -1;
    int global_prim_id = -1;
    float u = 0.f;
    float v = 0.f;

    if (params.active == nullptr || params.active[ray_idx]) {
        const float3 origin = make_float3(
            params.ray_o[ray_idx * 3 + 0],
            params.ray_o[ray_idx * 3 + 1],
            params.ray_o[ray_idx * 3 + 2]);
        const float3 direction = make_float3(
            params.ray_d[ray_idx * 3 + 0],
            params.ray_d[ray_idx * 3 + 1],
            params.ray_d[ray_idx * 3 + 2]);
        unsigned int p0 = __float_as_uint(t);
        unsigned int p1 = static_cast<unsigned int>(shape_id);
        unsigned int p2 = __float_as_uint(u);
        unsigned int p3 = __float_as_uint(v);
        unsigned int p4 = static_cast<unsigned int>(local_prim_id);
        const float trace_tmax =
            params.ray_tmax != nullptr ? params.ray_tmax[ray_idx] : __uint_as_float(0x7f7fffffu);
        optixTrace(
            params.traversable,
            origin,
            direction,
            1e-6f,
            trace_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            p0,
            p1,
            p2,
            p3,
            p4);
        t = __uint_as_float(p0);
        shape_id = static_cast<int>(p1);
        u = __uint_as_float(p2);
        v = __uint_as_float(p3);
        local_prim_id = static_cast<int>(p4);
        if (shape_id >= 0 && shape_id < params.mesh_count && local_prim_id >= 0) {
            global_prim_id = params.face_offsets[shape_id] + local_prim_id;
        }
    }

    if (params.out_t != nullptr)
        params.out_t[ray_idx] = t;
    if (params.out_shape_id != nullptr)
        params.out_shape_id[ray_idx] = shape_id;
    if (params.out_local_prim_id != nullptr)
        params.out_local_prim_id[ray_idx] = local_prim_id;
    if (params.out_global_prim_id != nullptr)
        params.out_global_prim_id[ray_idx] = global_prim_id;
    if (params.out_bary_uv != nullptr) {
        params.out_bary_uv[ray_idx * 2 + 0] = u;
        params.out_bary_uv[ray_idx * 2 + 1] = v;
    }
}

extern "C" __global__ void __miss__intersect() {
}

extern "C" __global__ void __closesthit__intersect() {
    const float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(optixGetInstanceId());
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(static_cast<unsigned int>(optixGetPrimitiveIndex()));
}
