#pragma once

#include <utd/utd_math.h>
#include <utd/utd_types.h>

namespace witwin::channel::native_ext::reflection_detail {

__device__ __forceinline__ float3a reflect_point_across_plane(
    float3a p,
    float3a plane_pt,
    float3a plane_n
) {
    float d = f3_dot(f3_sub(p, plane_pt), plane_n);
    return f3_sub(p, f3_mul(plane_n, 2.f * d));
}

__device__ __forceinline__ float3a reflect_direction(float3a d, float3a n) {
    return f3_sub(d, f3_mul(n, 2.f * f3_dot(d, n)));
}

__device__ __forceinline__ float3a project_polarization_to_ray(
    float3a tx_pol,
    float3a ray_dir
) {
    float3a ray_hat = safe_normalize(ray_dir, make_f3(0, 0, 1));
    float3a proj = f3_sub(tx_pol, f3_mul(ray_hat, f3_dot(tx_pol, ray_hat)));
    return safe_normalize(proj, stable_perp_basis(ray_hat, make_f3(0, 1, 0)));
}

} // namespace witwin::channel::native_ext::reflection_detail
