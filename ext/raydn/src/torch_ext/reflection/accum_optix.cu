#include <optix.h>
#include <optix_device.h>

#include <raydn/common/complex.cuh>
#include <raydn/common/math.cuh>
#include <raydn/reflection/accum_params.h>

namespace raydn {

extern "C" {
extern __constant__ AccumParams params;
}

namespace {

constexpr float kReflEps = 1e-6f;
constexpr float kEpsilon0 = 8.854187817e-12f;
constexpr float kSpeedOfLight = 299792458.0f;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

static __forceinline__ __device__ float3 fallback_axis(float3 direction) {
    return fabsf(direction.z) < 0.9f
               ? make_f3(0.f, 0.f, 1.f)
               : make_f3(0.f, 1.f, 0.f);
}

static __forceinline__ __device__ float3 stable_perpendicular(float3 direction,
                                                              float3 preferred) {
    const float3 dir = normalize3(direction);
    float3 projected = preferred - dot3(preferred, dir) * dir;
    if (dot3(projected, projected) > 1e-12f) {
        return normalize3(projected);
    }
    const float3 axis = fallback_axis(dir);
    projected = axis - dot3(axis, dir) * dir;
    return normalize3(projected);
}

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

static __forceinline__ __device__ void trace_handle(OptixTraversableHandle handle,
                                                    float3 origin,
                                                    float3 direction,
                                                    float tmax,
                                                    HitPayload &payload) {
    clear_payload(payload);
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
    trace_handle(params.primary_handle, origin, direction, tmax, hit_primary);
    if (params.split_mode == 0) {
        return hit_primary;
    }
    HitPayload hit_secondary;
    trace_handle(params.secondary_handle, origin, direction, tmax, hit_secondary);
    return choose_hit(hit_primary, hit_secondary);
}

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ void plane_coords(float3 value,
                                                    int axis,
                                                    float &coord0,
                                                    float &coord1) {
    if (axis == 0) {
        coord0 = value.y;
        coord1 = value.z;
    } else if (axis == 1) {
        coord0 = value.x;
        coord1 = value.z;
    } else {
        coord0 = value.x;
        coord1 = value.y;
    }
}

static __forceinline__ __device__ float3 axis_plane_point(int axis,
                                                          float position,
                                                          float coord0,
                                                          float coord1) {
    if (axis == 0) {
        return make_f3(position, coord0, coord1);
    }
    if (axis == 1) {
        return make_f3(coord0, position, coord1);
    }
    return make_f3(coord0, coord1, position);
}

static __forceinline__ __device__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __forceinline__ __device__ float uniform01(unsigned int ray_index,
                                                  unsigned int depth,
                                                  unsigned int seed) {
    const unsigned int h = hash_u32(ray_index ^ (depth * 0x9e3779b9u) ^ seed);
    return static_cast<float>(h & 0x00ffffffu) * (1.f / 16777216.f);
}

static __forceinline__ __device__ bool material_reflection_coefficients(int global_prim,
                                                                        float cos_theta,
                                                                        Complex &r_te,
                                                                        Complex &r_tm) {
    r_te = c_make(0.f, 0.f);
    r_tm = c_make(0.f, 0.f);
    if (global_prim < 0 || global_prim >= params.material_count ||
        params.material_valid == nullptr || params.material_valid[global_prim] == 0u) {
        return false;
    }

    const float eta_r = fmaxf(params.material_eta_r[global_prim], kReflEps);
    const float sigma = fmaxf(params.material_sigma[global_prim], 0.f);
    const float gain = params.material_gain[global_prim];
    const float mu_r = fmaxf(params.material_mu_r[global_prim], kReflEps);
    const float omega = fmaxf(2.f * kPi * kSpeedOfLight /
                                  fmaxf(params.wavelength, kReflEps),
                              kReflEps);
    const Complex eta = c_make(eta_r, -sigma / (omega * kEpsilon0));
    const Complex mu = c_make(mu_r, 0.f);
    const float cos_clamped = fminf(fmaxf(fabsf(cos_theta), kReflEps), 1.f);
    const float sin2 = fmaxf(0.f, 1.f - cos_clamped * cos_clamped);
    const Complex a = c_sqrt(c_sub(c_mul(mu, eta), c_make(sin2, 0.f)));
    const Complex mu_cos = c_make(mu_r * cos_clamped, 0.f);
    const Complex eta_cos = c_make(eta_r * cos_clamped, eta.i * cos_clamped);
    r_te = c_scale(c_div(c_sub(mu_cos, a), c_add(mu_cos, a)), gain);
    r_tm = c_scale(c_div(c_sub(eta_cos, a), c_add(eta_cos, a)), gain);
    if (!isfinite(r_te.r) || !isfinite(r_te.i)) {
        r_te = c_make(0.f, 0.f);
    }
    if (!isfinite(r_tm.r) || !isfinite(r_tm.i)) {
        r_tm = c_make(0.f, 0.f);
    }
    return c_abs2(r_te) > 0.f || c_abs2(r_tm) > 0.f;
}

static __forceinline__ __device__ Complex3 reflect_field_vector(Complex3 field,
                                                                float3 incident_dir,
                                                                float3 normal,
                                                                int global_prim,
                                                                float3 &reflected_dir) {
    const float3 incident_hat = normalize3(incident_dir);
    const float3 normal_hat = normalize3(normal);
    const float dot_dn = dot3(incident_hat, normal_hat);
    reflected_dir = normalize3(incident_hat - 2.f * dot_dn * normal_hat);

    float3 s_hat = cross(normal_hat, incident_hat);
    if (dot3(s_hat, s_hat) <= 1e-12f) {
        s_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        s_hat = normalize3(s_hat);
    }
    float3 p_in_hat = cross(s_hat, incident_hat);
    if (dot3(p_in_hat, p_in_hat) <= 1e-12f) {
        p_in_hat = stable_perpendicular(incident_hat, normal_hat);
    } else {
        p_in_hat = normalize3(p_in_hat);
    }
    float3 p_out_hat = cross(s_hat, reflected_dir);
    if (dot3(p_out_hat, p_out_hat) <= 1e-12f) {
        p_out_hat = stable_perpendicular(reflected_dir, normal_hat);
    } else {
        p_out_hat = normalize3(p_out_hat);
    }

    Complex r_te;
    Complex r_tm;
    const float cos_theta = fabsf(dot3(incident_hat, normal_hat));
    if (!material_reflection_coefficients(global_prim, cos_theta, r_te, r_tm)) {
        return c3_zero();
    }

    const Complex e_s = c3_dot_real(field, s_hat);
    const Complex e_p = c3_dot_real(field, p_in_hat);
    return c3_add(c3_scale_complex(s_hat, c_mul(r_te, e_s)),
                  c3_scale_complex(p_out_hat, c_mul(r_tm, e_p)));
}

static __forceinline__ __device__ void store_wedge_event(unsigned int ray_index,
                                                         int depth,
                                                         int global_prim,
                                                         float3 hit_point,
                                                         float3 normal,
                                                         float3 wi,
                                                         float3 source_point,
                                                         float src_power,
                                                         float3 d0) {
    if (params.collect_wedges == 0 || params.out_wedge_count == nullptr) {
        return;
    }
    if (depth > 0 && params.collect_wedge_prefixes == 0) {
        return;
    }
    float stored_source_power = src_power;
    const int sample_stride = max(params.wedge_sample_stride, 1);
    if (params.collect_wedge_prefixes != 0 && sample_stride > 1) {
        const unsigned int max_prefix_depth =
            static_cast<unsigned int>(max(params.max_bounces, 1));
        const unsigned int ordinal =
            ray_index * max_prefix_depth + static_cast<unsigned int>(depth);
        const unsigned int phase =
            static_cast<unsigned int>(params.seed) % static_cast<unsigned int>(sample_stride);
        if ((ordinal + phase) % static_cast<unsigned int>(sample_stride) != 0u) {
            return;
        }
        stored_source_power *= static_cast<float>(sample_stride);
    }

    const int slot = atomicAdd(params.out_wedge_count, 1);
    if (slot < 0 || slot >= params.wedge_capacity) {
        return;
    }

    params.out_wedge_ray_index[slot] = static_cast<int>(ray_index);
    params.out_wedge_hit_x[slot] = hit_point.x;
    params.out_wedge_hit_y[slot] = hit_point.y;
    params.out_wedge_hit_z[slot] = hit_point.z;
    params.out_wedge_normal_x[slot] = normal.x;
    params.out_wedge_normal_y[slot] = normal.y;
    params.out_wedge_normal_z[slot] = normal.z;
    params.out_wedge_prim_id[slot] = global_prim;
    params.out_wedge_dir_x[slot] = wi.x;
    params.out_wedge_dir_y[slot] = wi.y;
    params.out_wedge_dir_z[slot] = wi.z;
    params.out_wedge_source_x[slot] = source_point.x;
    params.out_wedge_source_y[slot] = source_point.y;
    params.out_wedge_source_z[slot] = source_point.z;
    params.out_wedge_source_power[slot] = stored_source_power;
    params.out_wedge_initial_dir_x[slot] = d0.x;
    params.out_wedge_initial_dir_y[slot] = d0.y;
    params.out_wedge_initial_dir_z[slot] = d0.z;
    params.out_wedge_bounce_depth[slot] = depth;
}

static __forceinline__ __device__ bool accumulate_plane(unsigned int ray_index,
                                                        int depth,
                                                        float3 origin,
                                                        float3 direction,
                                                        float blocker_t,
                                                        float3 image_source,
                                                        Complex3 field) {
    if (depth <= 0 || c3_power(field) <= 0.f) {
        return false;
    }
    const int axis = params.grid_axis;
    const float axis_dir = component(direction, axis);
    if (fabsf(axis_dir) <= kReflEps) {
        return false;
    }
    const float safe_axis_dir =
        axis_dir + (axis_dir >= 0.f ? kReflEps : -kReflEps);
    const float t_plane =
        (params.grid_position - component(origin, axis)) / safe_axis_dir;
    if (!(t_plane > kRayBias && t_plane < blocker_t)) {
        return false;
    }

    const float3 target = origin + t_plane * direction;
    float coord0 = 0.f;
    float coord1 = 0.f;
    plane_coords(target, axis, coord0, coord1);
    if (coord0 < params.grid_coord0_min || coord0 >= params.grid_coord0_max ||
        coord1 < params.grid_coord1_min || coord1 >= params.grid_coord1_max) {
        return false;
    }

    const float span0 = params.grid_coord0_max - params.grid_coord0_min;
    const float span1 = params.grid_coord1_max - params.grid_coord1_min;
    if (span0 <= 0.f || span1 <= 0.f ||
        params.grid_resolution0 <= 0 || params.grid_resolution1 <= 0) {
        return false;
    }

    const float u = (coord0 - params.grid_coord0_min) / span0;
    const float v = (coord1 - params.grid_coord1_min) / span1;
    const int ix = min(max(static_cast<int>(u * params.grid_resolution0), 0),
                       params.grid_resolution0 - 1);
    const int iy = min(max(static_cast<int>(v * params.grid_resolution1), 0),
                       params.grid_resolution1 - 1);
    const int cell = iy * params.grid_resolution0 + ix;

    const float3 target_plane =
        axis_plane_point(axis, params.grid_position, coord0, coord1);
    const float unfolded_distance = norm3(target_plane - image_source);
    const float fspl =
        params.wavelength / (4.f * kPi * fmaxf(unfolded_distance, kReflEps));
    const float cos_theta = fmaxf(fabsf(axis_dir), kReflEps);
    const float geometry_power_scale =
        params.solid_angle_per_ray / fmaxf(params.cell_area, kReflEps) *
        unfolded_distance * unfolded_distance / cos_theta;
    const float amplitude_scale =
        fspl * sqrtf(fmaxf(geometry_power_scale, 0.f));
    const float wave_k = fabsf(params.k) > kReflEps
                             ? params.k
                             : (2.f * kPi / fmaxf(params.wavelength, kReflEps));
    const Complex phase = c_exp_neg_i(wave_k * unfolded_distance);
    const Complex coeff = c_scale(phase, amplitude_scale);
    Complex3 contribution_field = c3_mul_complex(field, coeff);

    if (!isfinite(contribution_field.x.r) || !isfinite(contribution_field.x.i) ||
        !isfinite(contribution_field.y.r) || !isfinite(contribution_field.y.i) ||
        !isfinite(contribution_field.z.r) || !isfinite(contribution_field.z.i)) {
        return false;
    }

    const float contribution_power = c3_power(contribution_field);
    if (!(contribution_power > 0.f) || !isfinite(contribution_power)) {
        return false;
    }

    if (params.stage_cell != nullptr && params.stage_value != nullptr) {
        const long long stage_stride = static_cast<long long>(params.max_bounces) + 1ll;
        const long long slot = static_cast<long long>(ray_index) * stage_stride +
                               static_cast<long long>(depth);
        ReflAccumStagedValue value;
        value.a = make_float4(contribution_power,
                              contribution_field.x.r,
                              contribution_field.x.i,
                              contribution_field.y.r);
        value.b = make_float4(contribution_field.y.i,
                              contribution_field.z.r,
                              contribution_field.z.i,
                              1.0f);
        params.stage_cell[slot] = cell;
        params.stage_value[slot] = value;
        return true;
    }

    const WarpCellGroup cell_group = warp_cell_group(cell);
    atomic_add_same_cell(params.out_field_x_re, cell, contribution_field.x.r, cell_group);
    atomic_add_same_cell(params.out_field_x_im, cell, contribution_field.x.i, cell_group);
    atomic_add_same_cell(params.out_field_y_re, cell, contribution_field.y.r, cell_group);
    atomic_add_same_cell(params.out_field_y_im, cell, contribution_field.y.i, cell_group);
    atomic_add_same_cell(params.out_field_z_re, cell, contribution_field.z.r, cell_group);
    atomic_add_same_cell(params.out_field_z_im, cell, contribution_field.z.i, cell_group);
    atomic_add_same_cell(params.out_reflection_power, cell, contribution_power, cell_group);
    atomic_add_warp(params.out_reflection_count, 1);
    return true;
}

} // namespace

extern "C" {
__constant__ AccumParams params;
}

// OptiX programs for the native reflection-accumulation pipeline; one raygen launch per ray.

/// Closest-hit: pack the hit (t, barycentrics, primitive, instance) into the ray payload.
extern "C" __global__ void __closesthit__reflection_accumulation() {
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

/// Miss: mark the payload as "no hit".
extern "C" __global__ void __miss__reflection_accumulation() {
    optixSetPayload_0(0u);
}

/// Raygen: trace one ray through its reflections and scatter the resulting power/field
/// contributions into the accumulation grid (with Russian-roulette termination).
extern "C" __global__ void __raygen__reflection_accumulation() {
    const unsigned int ray_index = optixGetLaunchIndex().x;
    if (ray_index >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }
    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        return;
    }

    float3 origin = make_f3(params.ray_ox[ray_index],
                              params.ray_oy[ray_index],
                              params.ray_oz[ray_index]);
    float3 direction = normalize3(make_f3(params.ray_dx[ray_index],
                                            params.ray_dy[ray_index],
                                            params.ray_dz[ray_index]));
    const float3 d0 = direction;
    float3 image_source = make_f3(params.tx_x[ray_index],
                                    params.tx_y[ray_index],
                                    params.tx_z[ray_index]);
    float3 tx_polarization = make_f3(params.tx_pol_x[ray_index],
                                       params.tx_pol_y[ray_index],
                                       params.tx_pol_z[ray_index]);
    float3 transverse_polarization =
        tx_polarization - dot3(tx_polarization, direction) * direction;
    if (dot3(transverse_polarization, transverse_polarization) <= 1e-12f) {
        transverse_polarization = stable_perpendicular(direction, tx_polarization);
    } else {
        transverse_polarization = normalize3(transverse_polarization);
    }
    Complex3 field = c3_from_real(transverse_polarization);
    float path_length = 0.f;

    for (int depth = 0; depth <= params.max_bounces; ++depth) {
        const float tmax_input =
            depth == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kRayTMax;
        const float trace_tmax = isfinite(tmax_input) ? tmax_input : kRayTMax;
        const HitPayload hit = trace_scene(origin, direction, trace_tmax);
        const float blocker_t = hit.hit != 0u ? __uint_as_float(hit.t) : kRayTMax;

        accumulate_plane(ray_index,
                         depth,
                         origin,
                          direction,
                          blocker_t,
                          image_source,
                          field);

        if (hit.hit == 0u || depth >= params.max_bounces) {
            break;
        }

        const int shape_id = static_cast<int>(hit.instance);
        const int local_prim = static_cast<int>(hit.prim);
        const int face_offset =
            (shape_id >= 0 && shape_id < params.n_meshes) ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;
        const float bary_u = __uint_as_float(hit.bary_u);
        const float bary_v = __uint_as_float(hit.bary_v);

        float3 hit_point = origin + blocker_t * direction;
        float3 geo_normal = make_f3(0.f, 0.f, 1.f);
        if (global_prim >= 0 && global_prim < params.n_triangles) {
            hit_point = make_f3(
                params.tri_p0_x[global_prim] + bary_u * params.tri_e1_x[global_prim] +
                    bary_v * params.tri_e2_x[global_prim],
                params.tri_p0_y[global_prim] + bary_u * params.tri_e1_y[global_prim] +
                    bary_v * params.tri_e2_y[global_prim],
                params.tri_p0_z[global_prim] + bary_u * params.tri_e1_z[global_prim] +
                    bary_v * params.tri_e2_z[global_prim]);
            geo_normal = normalize3(make_f3(params.tri_fn_x[global_prim],
                                              params.tri_fn_y[global_prim],
                                              params.tri_fn_z[global_prim]));
        }
        if (dot3(direction, geo_normal) > 0.f) {
            geo_normal = -1.f * geo_normal;
        }

        float3 reflected_dir;
        const float src_power = c3_power(field) * params.solid_angle_per_ray;
        const Complex3 reflected_field =
            reflect_field_vector(field, direction, geo_normal, global_prim, reflected_dir);
        if (c3_power(reflected_field) <= 0.f) {
            break;
        }

        store_wedge_event(ray_index,
                          depth,
                          global_prim,
                          hit_point,
                          geo_normal,
                          direction,
                          image_source,
                          src_power,
                          d0);

        const float image_distance = dot3(image_source - hit_point, geo_normal);
        image_source = image_source - 2.f * image_distance * geo_normal;
        path_length += blocker_t;

        field = reflected_field;
        direction = reflected_dir;
        origin = hit_point + kRayBias * direction;

        const int next_depth = depth + 1;
        if (params.rr_depth > 0 && params.rr_prob < 1.f && next_depth >= params.rr_depth) {
            const float field_power = c3_power(field);
            const float continue_prob =
                fminf(fmaxf(field_power, 1e-8f), fmaxf(params.rr_prob, 1e-8f));
            if (uniform01(ray_index, static_cast<unsigned int>(next_depth),
                          static_cast<unsigned int>(params.seed)) >= continue_prob) {
                break;
            }
            const float rr_scale = rsqrtf(fmaxf(continue_prob, 1e-8f));
            field.x = c_scale(field.x, rr_scale);
            field.y = c_scale(field.y, rr_scale);
            field.z = c_scale(field.z, rr_scale);
        }

        if (params.stop_threshold > 0.f) {
            const float fspl =
                params.wavelength /
                (4.f * kPi * fmaxf(path_length, kReflEps));
            if (c3_power(field) * fspl * fspl <= params.stop_threshold) {
                break;
            }
        }
    }
}

} // namespace raydn
