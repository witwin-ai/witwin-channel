#include <raydtorch/diffraction/accum_ad.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <raydtorch/common/math.cuh>
#include <raydtorch/common/native_compat.h>

namespace raydtorch {

namespace {

constexpr float kDfrEps = 1e-6f;

static __forceinline__ __device__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __forceinline__ __device__ float uniform01(unsigned int lane,
                                                  unsigned int stream,
                                                  unsigned int seed) {
    const unsigned int h = hash_u32(lane ^ (stream * 0x9e3779b9u) ^ seed);
    return static_cast<float>(h & 0x00ffffffu) * (1.f / 16777216.f);
}

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ float3 stable_perpendicular(float3 axis,
                                                              float3 preferred) {
    float3 projected = preferred - dot3(preferred, axis) * axis;
    if (dot3(projected, projected) > 1e-12f) {
        return normalize3(projected);
    }
    const float3 fallback = fabsf(axis.z) < 0.9f
                                ? make_f3(0.f, 0.f, 1.f)
                                : make_f3(0.f, 1.f, 0.f);
    return normalize3(fallback - dot3(fallback, axis) * axis);
}

static __forceinline__ __device__ float3 normalize_jvp(float3 unit,
                                                       float norm,
                                                       float3 dot_v) {
    if (!(norm > kDfrEps) || !isfinite(norm)) {
        return make_f3(0.f, 0.f, 0.f);
    }
    return (1.f / norm) * (dot_v - dot3(unit, dot_v) * unit);
}

static __forceinline__ __device__ float3 grid_cell_center(
    const DfrDirectAccumADParams &params,
    int cell) {
    const int i = cell % params.grid_resolution0;
    const int j = cell / params.grid_resolution0;
    const float u = (static_cast<float>(i) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution0), 1.f);
    const float v = (static_cast<float>(j) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution1), 1.f);
    const float c0 = params.grid_coord0_min +
                     u * (params.grid_coord0_max - params.grid_coord0_min);
    const float c1 = params.grid_coord1_min +
                     v * (params.grid_coord1_max - params.grid_coord1_min);
    if (params.grid_axis == 0) {
        return make_f3(params.grid_position, c0, c1);
    }
    if (params.grid_axis == 1) {
        return make_f3(c0, params.grid_position, c1);
    }
    return make_f3(c0, c1, params.grid_position);
}

static __forceinline__ __device__ float3 grid_cell_center(
    const DfrChainAccumADParams &params,
    int cell) {
    const int i = cell % params.grid_resolution0;
    const int j = cell / params.grid_resolution0;
    const float u = (static_cast<float>(i) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution0), 1.f);
    const float v = (static_cast<float>(j) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution1), 1.f);
    const float c0 = params.grid_coord0_min +
                     u * (params.grid_coord0_max - params.grid_coord0_min);
    const float c1 = params.grid_coord1_min +
                     v * (params.grid_coord1_max - params.grid_coord1_min);
    if (params.grid_axis == 0) {
        return make_f3(params.grid_position, c0, c1);
    }
    if (params.grid_axis == 1) {
        return make_f3(c0, params.grid_position, c1);
    }
    return make_f3(c0, c1, params.grid_position);
}

struct DirectPrimal {
    int lane;
    int state_idx;
    int cell;
    int material_idx;
    int suffix_material_idx;
    bool is_keller;
    bool is_suffix;
    int sample_count;
    float edge_u;
    float3 edge_pos;
    float3 edge_dir_raw;
    float edge_dir_norm;
    float3 edge_dir;
    float edge_t_min;
    float edge_t_max;
    float edge_t;
    float edge_length;
    float3 edge_point;
    float3 source;
    float3 wi_raw;
    float wi_norm;
    float3 wi;
    float src_power;
    float exterior_angle;
    float material_gain;
    float suffix_material_gain;
    float suffix_reflection_gain;
    float suffix_fspl;
    float suffix_candidate_count;
    float suffix_outgoing_dist2;
    float suffix_ray_t;
    float wedge_scale;
    float3 target;
    float3 grid_target;
    float3 keller_ko;
    float3 suffix_p0;
    float3 suffix_normal;
    float suffix_normal_norm;
    float3 suffix_image_source;
    float3 suffix_ray_dir;
    float keller_ray_t;
    float keller_sin;
    float keller_cos;
    float source_dist2;
    float target_dist2;
    float contribution;
    float common_no_src;
    bool edge_length_active;
    bool wedge_active;
    bool material_active;
    bool suffix_material_active;
};

struct DfrTangent {
    float3 edge_pos;
    float3 edge_dir_raw;
    float edge_t_min;
    float edge_t_max;
    float3 source;
    float3 wi_raw;
    float src_power;
    float exterior_angle;
    float material_gain;
    float suffix_material_gain;
    float3 suffix_p0;
    float3 suffix_normal_raw;
};

struct ChainEventPrimal {
    int state_idx;
    int material_idx;
    bool material_active;
    float edge_u;
    float edge_t;
    float edge_length;
    float edge_dir_norm;
    float wedge_scale;
    float source_dist2;
    float target_dist2;
    float src_power;
    float exterior_angle;
    float material_gain;
    float contribution;
    bool edge_length_active;
    bool wedge_active;
    float3 edge_pos;
    float3 edge_dir_raw;
    float3 edge_dir;
    float3 edge_point;
    float3 source;
    float3 target;
};

struct ChainPrimal {
    int lane;
    int cell;
    int first_idx;
    int second_idx;
    int third_idx;
    int suffix_material_idx;
    bool is_keller;
    bool is_suffix;
    bool has_third;
    int sample_count;
    float wave_gain;
    float sample_norm;
    float contribution;
    float suffix_material_gain;
    float suffix_reflection_gain;
    float suffix_fspl;
    float suffix_candidate_count;
    float suffix_outgoing_dist2;
    float suffix_ray_t;
    float keller_ray_t;
    float keller_sin;
    float keller_cos;
    float keller_incident_norm;
    float suffix_normal_norm;
    bool suffix_material_active;
    float3 keller_incident;
    float3 keller_ko;
    float3 grid_target;
    float3 final_target;
    float3 suffix_p0;
    float3 suffix_normal;
    float3 suffix_image_source;
    float3 suffix_ray_dir;
    ChainEventPrimal first;
    ChainEventPrimal second;
    ChainEventPrimal third;
};

struct ChainEventTangent {
    float3 edge_pos;
    float3 edge_dir_raw;
    float edge_t_min;
    float edge_t_max;
    float3 source;
    float src_power;
    float exterior_angle;
    float material_gain;
};

struct ChainTangent {
    ChainEventTangent first;
    ChainEventTangent second;
    ChainEventTangent third;
    float suffix_material_gain;
    float3 suffix_p0;
    float3 suffix_normal_raw;
};

static __forceinline__ __device__ bool keller_target_from_state(
    const DfrDirectAccumADParams &params,
    int lane,
    float3 edge_point,
    float3 edge_dir,
    float3 wi,
    float3 &target,
    float3 &ko,
    float &ray_t,
    float &sin_theta,
    float &cos_theta) {
    const float axial = fminf(fmaxf(dot3(wi, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, wi);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    sincosf(2.f * kPi * uniform01(static_cast<unsigned int>(lane),
                                  1u,
                                  static_cast<unsigned int>(params.seed)),
            &sin_theta,
            &cos_theta);
    ko = normalize3(axial * edge_dir +
                    radial * (cos_theta * basis0 + sin_theta * basis1));
    const float denom = component(ko, params.grid_axis);
    if (fabsf(denom) <= kDfrEps) {
        return false;
    }
    ray_t = (params.grid_position - component(edge_point, params.grid_axis)) / denom;
    if (!(ray_t > kDfrRayBias) || !isfinite(ray_t)) {
        return false;
    }
    target = edge_point + ray_t * ko;
    return true;
}

static __forceinline__ __device__ float3 stable_perpendicular_jvp(
    float3 axis,
    float3 dot_axis,
    float3 preferred,
    float3 dot_preferred,
    float3 basis) {
    const float axis_dot_preferred = dot3(preferred, axis);
    const float3 projected = preferred - axis_dot_preferred * axis;
    const float projected_norm2 = dot3(projected, projected);
    float3 dot_projected;
    float projected_norm;
    if (projected_norm2 > 1e-12f) {
        dot_projected =
            dot_preferred -
            (dot3(dot_preferred, axis) + dot3(preferred, dot_axis)) * axis -
            axis_dot_preferred * dot_axis;
        projected_norm = sqrtf(fmaxf(projected_norm2, 0.f));
    } else {
        const float3 fallback = fabsf(axis.z) < 0.9f
                                    ? make_f3(0.f, 0.f, 1.f)
                                    : make_f3(0.f, 1.f, 0.f);
        const float fallback_dot_axis = dot3(fallback, axis);
        const float3 fallback_projected = fallback - fallback_dot_axis * axis;
        dot_projected =
            -1.f * (dot3(fallback, dot_axis) * axis + fallback_dot_axis * dot_axis);
        projected_norm = norm3(fallback_projected);
    }
    return normalize_jvp(basis, projected_norm, dot_projected);
}

static __forceinline__ __device__ float3 keller_target_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float3 dot_edge_point,
    float3 dot_edge_dir,
    float3 dot_wi_raw) {
    if (!p.is_keller) {
        return make_f3(0.f, 0.f, 0.f);
    }

    const float3 dot_wi = normalize_jvp(p.wi, p.wi_norm, dot_wi_raw);
    const float unclamped_axial = dot3(p.wi, p.edge_dir);
    const bool axial_active = unclamped_axial > -1.f && unclamped_axial < 1.f;
    const float axial = fminf(fmaxf(unclamped_axial, -1.f), 1.f);
    const float dot_axial = axial_active
                                ? dot3(dot_wi, p.edge_dir) + dot3(p.wi, dot_edge_dir)
                                : 0.f;
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float dot_radial =
        radial > kDfrEps ? (-(axial / radial) * dot_axial) : 0.f;

    const float3 basis0 = stable_perpendicular(p.edge_dir, p.wi);
    const float3 dot_basis0 =
        stable_perpendicular_jvp(p.edge_dir, dot_edge_dir, p.wi, dot_wi, basis0);
    const float3 basis1_raw = cross3(p.edge_dir, basis0);
    const float3 basis1 = normalize3(basis1_raw);
    const float basis1_norm = norm3(basis1_raw);
    const float3 dot_basis1_raw =
        cross3(dot_edge_dir, basis0) + cross3(p.edge_dir, dot_basis0);
    const float3 dot_basis1 = normalize_jvp(basis1, basis1_norm, dot_basis1_raw);

    const float3 radial_basis = p.keller_cos * basis0 + p.keller_sin * basis1;
    const float3 dot_radial_basis =
        p.keller_cos * dot_basis0 + p.keller_sin * dot_basis1;
    const float3 ko_raw = axial * p.edge_dir + radial * radial_basis;
    const float3 dot_ko_raw =
        dot_axial * p.edge_dir +
        axial * dot_edge_dir +
        dot_radial * radial_basis +
        radial * dot_radial_basis;
    const float ko_norm = norm3(ko_raw);
    const float3 dot_ko = normalize_jvp(p.keller_ko, ko_norm, dot_ko_raw);

    const float denom = component(p.keller_ko, params.grid_axis);
    const float dot_denom = component(dot_ko, params.grid_axis);
    const float numerator =
        params.grid_position - component(p.edge_point, params.grid_axis);
    const float dot_numerator = -component(dot_edge_point, params.grid_axis);
    const float dot_t =
        (dot_numerator * denom - numerator * dot_denom) /
        fmaxf(denom * denom, kDfrEps);
    return dot_edge_point + dot_t * p.keller_ko + p.keller_ray_t * dot_ko;
}

static __forceinline__ __device__ bool material_valid_for_prim(
    const DfrDirectAccumADParams &params,
    int prim) {
    return prim >= 0 &&
           prim < params.material_count &&
           params.material_gain != nullptr &&
           (params.material_valid == nullptr || params.material_valid[prim] != 0u);
}

static __forceinline__ __device__ bool material_valid_for_prim(
    const DfrChainAccumADParams &params,
    int prim) {
    return prim >= 0 &&
           prim < params.material_count &&
           params.material_gain != nullptr &&
           (params.material_valid == nullptr || params.material_valid[prim] != 0u);
}

static __forceinline__ __device__ int material_index_for_faces(
    const DfrChainAccumADParams &params,
    int face0_prim,
    int face1_prim) {
    if (material_valid_for_prim(params, face0_prim)) {
        return face0_prim;
    }
    if (face1_prim != face0_prim && material_valid_for_prim(params, face1_prim)) {
        return face1_prim;
    }
    return -1;
}

template <typename Params>
static __forceinline__ __device__ bool suffix_candidate_valid(
    const Params &params,
    int prim) {
    return prim >= 0 &&
           prim < params.n_triangles &&
           material_valid_for_prim(params, prim);
}

template <typename Params>
static __forceinline__ __device__ bool select_local_suffix_candidate(
    const Params &params,
    int face0_prim,
    int face1_prim,
    unsigned int lane,
    unsigned int stream,
    int &prim,
    float &candidate_count) {
    const bool face0_valid = suffix_candidate_valid(params, face0_prim);
    const bool face1_valid =
        suffix_candidate_valid(params, face1_prim) && face1_prim != face0_prim;
    const int count = (face0_valid ? 1 : 0) + (face1_valid ? 1 : 0);
    if (count <= 0) {
        return false;
    }
    const unsigned int candidate_hash = hash_u32(
        lane ^ (stream * 0x9e3779b9u) ^ static_cast<unsigned int>(params.seed));
    const int slot = static_cast<int>(
        candidate_hash % static_cast<unsigned int>(count));
    prim = (face0_valid && slot == 0) ? face0_prim : face1_prim;
    candidate_count = static_cast<float>(count);
    return true;
}

template <typename Params>
static __forceinline__ __device__ bool load_triangle(
    const Params &params,
    int prim,
    float3 &p0,
    float3 &e1,
    float3 &e2,
    float3 &normal,
    float *normal_norm = nullptr) {
    if (prim < 0 ||
        prim >= params.n_triangles ||
        params.tri_p0_x == nullptr ||
        params.tri_e1_x == nullptr ||
        params.tri_e2_x == nullptr ||
        params.tri_fn_x == nullptr) {
        return false;
    }
    p0 = make_f3(params.tri_p0_x[prim],
                   params.tri_p0_y[prim],
                   params.tri_p0_z[prim]);
    e1 = make_f3(params.tri_e1_x[prim],
                   params.tri_e1_y[prim],
                   params.tri_e1_z[prim]);
    e2 = make_f3(params.tri_e2_x[prim],
                   params.tri_e2_y[prim],
                   params.tri_e2_z[prim]);
    normal = make_f3(params.tri_fn_x[prim],
                       params.tri_fn_y[prim],
                       params.tri_fn_z[prim]);
    float norm = norm3(normal);
    if (norm <= 1e-6f) {
        normal = cross3(e1, e2);
        norm = norm3(normal);
    }
    if (norm <= 1e-6f) {
        return false;
    }
    if (normal_norm != nullptr) {
        *normal_norm = norm;
    }
    normal = (1.f / norm) * normal;
    return true;
}

template <typename Params>
static __forceinline__ __device__ bool intersect_reflection_triangle(
    const Params &params,
    float3 image_source,
    float3 target,
    int prim,
    float3 &reflection_point,
    float3 &normal,
    float3 &ray_dir,
    float &ray_t) {
    float3 p0;
    float3 e1;
    float3 e2;
    if (!load_triangle(params, prim, p0, e1, e2, normal)) {
        return false;
    }
    const float3 delta = target - image_source;
    const float dist = norm3(delta);
    if (!(dist > kDfrRayBias) || !isfinite(dist)) {
        return false;
    }
    ray_dir = (1.f / dist) * delta;
    const float3 h = cross3(ray_dir, e2);
    const float a = dot3(e1, h);
    if (fabsf(a) <= 1e-7f) {
        return false;
    }
    const float f = 1.f / a;
    const float3 s = image_source - p0;
    const float u = f * dot3(s, h);
    if (u < -1e-5f || u > 1.f + 1e-5f) {
        return false;
    }
    const float3 q = cross3(s, e1);
    const float v = f * dot3(ray_dir, q);
    if (v < -1e-5f || u + v > 1.f + 1e-5f) {
        return false;
    }
    ray_t = f * dot3(e2, q);
    if (!(ray_t > kDfrRayBias) || !(ray_t < dist - kDfrRayBias) || !isfinite(ray_t)) {
        return false;
    }
    reflection_point = image_source + ray_t * ray_dir;
    return true;
}

template <typename Params>
static __forceinline__ __device__ bool suffix_reflection_connection(
    const Params &params,
    float3 diff_point,
    float3 target,
    int face0_prim,
    int face1_prim,
    unsigned int lane,
    float3 &reflection_point,
    int &prim,
    float &reflection_gain,
    float &suffix_fspl,
    float &candidate_count,
    float3 &normal,
    float3 &plane_p0,
    float &normal_norm,
    float3 &image_source,
    float3 &ray_dir,
    float &ray_t,
    float &outgoing_dist2,
    float &material_gain,
    bool &material_active) {
    if (!select_local_suffix_candidate(params,
                                       face0_prim,
                                       face1_prim,
                                       lane,
                                       17u,
                                       prim,
                                       candidate_count)) {
        return false;
    }
    float3 p0;
    float3 e1;
    float3 e2;
    if (!load_triangle(params, prim, p0, e1, e2, normal, &normal_norm)) {
        return false;
    }
    plane_p0 = p0;
    const float plane_distance = dot3(diff_point - p0, normal);
    image_source = diff_point - 2.f * plane_distance * normal;
    if (!intersect_reflection_triangle(params,
                                       image_source,
                                       target,
                                       prim,
                                       reflection_point,
                                       normal,
                                       ray_dir,
                                       ray_t)) {
        return false;
    }

    const float3 incoming = reflection_point - diff_point;
    const float3 outgoing = target - reflection_point;
    const float incoming_dist = norm3(incoming);
    const float outgoing_dist = norm3(outgoing);
    if (!(incoming_dist > kDfrEps) || !(outgoing_dist > kDfrEps)) {
        return false;
    }
    const float3 incoming_hat = (1.f / incoming_dist) * incoming;
    const float3 oriented_normal =
        dot3(incoming_hat, normal) > 0.f ? (-1.f * normal) : normal;
    const float3 reflected_hat =
        incoming_hat - 2.f * dot3(incoming_hat, oriented_normal) * oriented_normal;
    const float3 outgoing_hat = (1.f / outgoing_dist) * outgoing;
    if (dot3(reflected_hat, outgoing_hat) <= 1.f - 1e-3f) {
        return false;
    }

    const float raw_gain = params.material_gain != nullptr ? params.material_gain[prim] : 1.f;
    material_gain = fmaxf(raw_gain, 0.f);
    material_active = raw_gain > 0.f && material_valid_for_prim(params, prim);
    reflection_gain = material_gain * material_gain;
    const float fspl = params.wavelength * (1.f / (4.f * kPi));
    outgoing_dist2 = outgoing_dist * outgoing_dist;
    suffix_fspl = (fspl * fspl) / fmaxf(outgoing_dist2, kDfrEps);
    return isfinite(reflection_gain) && isfinite(suffix_fspl);
}

static __forceinline__ __device__ float3 suffix_target_jvp(
    const DirectPrimal &p,
    const DfrTangent &tangent,
    float3 dot_edge_point,
    float &dot_suffix_fspl) {
    dot_suffix_fspl = 0.f;
    if (!p.is_suffix) {
        return make_f3(0.f, 0.f, 0.f);
    }

    const float3 dot_normal =
        normalize_jvp(p.suffix_normal, p.suffix_normal_norm, tangent.suffix_normal_raw);
    const float3 plane_delta = p.edge_point - p.suffix_p0;
    const float plane_distance = dot3(plane_delta, p.suffix_normal);
    const float dot_plane_distance =
        dot3(dot_edge_point - tangent.suffix_p0, p.suffix_normal) +
        dot3(plane_delta, dot_normal);
    const float3 dot_image_source =
        dot_edge_point -
        2.f * (dot_plane_distance * p.suffix_normal + plane_distance * dot_normal);
    const float3 ray_delta = p.grid_target - p.suffix_image_source;
    const float ray_dist = norm3(ray_delta);
    const float3 dot_ray_delta = -1.f * dot_image_source;
    const float3 dot_ray_dir =
        normalize_jvp(p.suffix_ray_dir, ray_dist, dot_ray_delta);
    const float denom = dot3(p.suffix_ray_dir, p.suffix_normal);
    const float dot_denom =
        dot3(dot_ray_dir, p.suffix_normal) + dot3(p.suffix_ray_dir, dot_normal);
    const float3 plane_to_image = p.suffix_p0 - p.suffix_image_source;
    const float numerator = dot3(plane_to_image, p.suffix_normal);
    const float dot_numerator =
        dot3(tangent.suffix_p0 - dot_image_source, p.suffix_normal) +
        dot3(plane_to_image, dot_normal);
    const float dot_ray_t =
        (dot_numerator * denom - numerator * dot_denom) /
        fmaxf(denom * denom, kDfrEps);
    const float3 dot_reflection_point =
        dot_image_source + dot_ray_t * p.suffix_ray_dir + p.suffix_ray_t * dot_ray_dir;

    if (p.suffix_outgoing_dist2 > kDfrEps && p.suffix_fspl != 0.f) {
        const float3 outgoing = p.grid_target - p.target;
        const float dot_outgoing_dist2 =
            2.f * dot3(outgoing, -1.f * dot_reflection_point);
        dot_suffix_fspl =
            p.suffix_fspl * (-(dot_outgoing_dist2 / p.suffix_outgoing_dist2));
    }
    return dot_reflection_point;
}

static __forceinline__ __device__ bool load_primal(
    const DfrDirectAccumADParams &params,
    int lane,
    DirectPrimal &p) {
    if (lane >= params.n_rays ||
        params.tape_active == nullptr ||
        params.tape_active[lane] == 0u) {
        return false;
    }

    p.lane = lane;
    p.state_idx = params.tape_state_idx[lane];
    p.cell = params.tape_cell[lane];
    p.material_idx = params.tape_material_idx != nullptr
                         ? params.tape_material_idx[lane]
                         : -1;
    p.edge_u = params.tape_edge_u != nullptr ? params.tape_edge_u[lane] : 0.f;
    if (p.state_idx < 0 || p.state_idx >= params.state_count ||
        p.cell < 0 ||
        p.cell >= params.grid_resolution0 * params.grid_resolution1) {
        return false;
    }
    p.is_keller = lane >= params.direct_samples &&
                  lane < params.direct_samples + params.keller_samples;
    p.is_suffix = lane >= params.direct_samples + params.keller_samples &&
                  lane < params.direct_samples + params.keller_samples + params.suffix_samples;
    p.sample_count = p.is_keller
                         ? params.keller_samples
                         : (p.is_suffix ? params.suffix_samples : params.direct_samples);
    if (p.sample_count <= 0) {
        return false;
    }

    p.edge_pos = make_f3(params.state_edge_pos_x[p.state_idx],
                           params.state_edge_pos_y[p.state_idx],
                           params.state_edge_pos_z[p.state_idx]);
    p.edge_dir_raw = make_f3(params.state_edge_dir_x[p.state_idx],
                               params.state_edge_dir_y[p.state_idx],
                               params.state_edge_dir_z[p.state_idx]);
    p.edge_dir_norm = norm3(p.edge_dir_raw);
    if (!(p.edge_dir_norm > kDfrEps) || !isfinite(p.edge_dir_norm)) {
        return false;
    }
    p.edge_dir = (1.f / p.edge_dir_norm) * p.edge_dir_raw;
    p.edge_t_min = params.state_edge_t_min[p.state_idx];
    p.edge_t_max = params.state_edge_t_max[p.state_idx];
    p.edge_t = p.edge_t_min + p.edge_u * (p.edge_t_max - p.edge_t_min);
    p.edge_length = fmaxf(p.edge_t_max - p.edge_t_min, 0.f);
    p.edge_length_active = (p.edge_t_max - p.edge_t_min) > 0.f;
    p.edge_point = p.edge_pos + p.edge_t * p.edge_dir;
    p.source = make_f3(params.state_src_x[p.state_idx],
                         params.state_src_y[p.state_idx],
                         params.state_src_z[p.state_idx]);
    p.wi_raw = make_f3(params.state_wi_x != nullptr ? params.state_wi_x[p.state_idx] : 0.f,
                         params.state_wi_y != nullptr ? params.state_wi_y[p.state_idx] : 0.f,
                         params.state_wi_z != nullptr ? params.state_wi_z[p.state_idx] : 0.f);
    p.wi_norm = norm3(p.wi_raw);
    p.wi = normalize3(p.wi_raw);
    p.src_power = params.state_src_power[p.state_idx];
    p.exterior_angle = params.state_exterior_angle[p.state_idx];
    const float exterior_clamped = fmaxf(p.exterior_angle, 0.25f * kPi);
    p.wedge_scale = fminf(exterior_clamped / (2.f * kPi), 2.f);
    p.wedge_active = p.exterior_angle > 0.25f * kPi &&
                     exterior_clamped / (2.f * kPi) < 2.f;
    p.material_gain = 1.f;
    p.material_active = false;
    if (material_valid_for_prim(params, p.material_idx)) {
        const float raw_gain = params.material_gain[p.material_idx];
        p.material_gain = fmaxf(raw_gain, 0.f);
        p.material_active = raw_gain > 0.f;
    }
    p.suffix_material_idx = -1;
    p.suffix_material_gain = 1.f;
    p.suffix_reflection_gain = 1.f;
    p.suffix_fspl = 1.f;
    p.suffix_candidate_count = 1.f;
    p.suffix_outgoing_dist2 = 1.f;
    p.suffix_ray_t = 0.f;
    p.suffix_material_active = false;
    p.grid_target = make_f3(0.f, 0.f, 0.f);
    p.suffix_p0 = make_f3(0.f, 0.f, 0.f);
    p.suffix_normal = make_f3(0.f, 0.f, 1.f);
    p.suffix_normal_norm = 1.f;
    p.suffix_image_source = make_f3(0.f, 0.f, 0.f);
    p.suffix_ray_dir = make_f3(0.f, 0.f, 0.f);
    p.keller_ko = make_f3(0.f, 0.f, 0.f);
    p.keller_ray_t = 0.f;
    p.keller_sin = 0.f;
    p.keller_cos = 1.f;
    if (p.is_keller) {
        if (!keller_target_from_state(params,
                                      lane,
                                      p.edge_point,
                                      p.edge_dir,
                                      p.wi,
                                      p.target,
                                      p.keller_ko,
                                      p.keller_ray_t,
                                      p.keller_sin,
                                      p.keller_cos)) {
            return false;
        }
    } else {
        p.target = grid_cell_center(params, p.cell);
    }
    p.grid_target = p.target;
    if (p.is_suffix) {
        const int prim0 = params.state_prim0 != nullptr
                              ? params.state_prim0[p.state_idx]
                              : -1;
        const int prim1 = params.state_prim1 != nullptr
                              ? params.state_prim1[p.state_idx]
                              : -1;
        if (!suffix_reflection_connection(params,
                                          p.edge_point,
                                          p.grid_target,
                                          prim0,
                                          prim1,
                                          static_cast<unsigned int>(lane),
                                          p.target,
                                          p.suffix_material_idx,
                                          p.suffix_reflection_gain,
                                          p.suffix_fspl,
                                           p.suffix_candidate_count,
                                           p.suffix_normal,
                                           p.suffix_p0,
                                           p.suffix_normal_norm,
                                           p.suffix_image_source,
                                          p.suffix_ray_dir,
                                          p.suffix_ray_t,
                                          p.suffix_outgoing_dist2,
                                          p.suffix_material_gain,
                                          p.suffix_material_active)) {
            return false;
        }
    }

    const float source_dist = fmaxf(norm3(p.edge_point - p.source), kDfrEps);
    const float target_dist = fmaxf(norm3(p.target - p.edge_point), kDfrEps);
    p.source_dist2 = source_dist * source_dist;
    p.target_dist2 = target_dist * target_dist;
    const float sample_norm =
        1.f / fmaxf(static_cast<float>(p.sample_count), 1.f);
    const float suffix_scale =
        p.is_suffix
            ? p.suffix_reflection_gain *
                  p.suffix_fspl *
                  fmaxf(p.suffix_candidate_count, 1.f)
            : 1.f;
    p.common_no_src = p.material_gain *
                      p.edge_length *
                      params.grid_cell_area *
                      p.wedge_scale *
                      sample_norm *
                      suffix_scale /
                      (p.source_dist2 * p.target_dist2);
    p.contribution = p.src_power * p.common_no_src;
    return p.contribution > 0.f && isfinite(p.contribution);
}

static __forceinline__ __device__ float read_or_zero(const float *ptr, int index) {
    return ptr != nullptr ? ptr[index] : 0.f;
}

static __forceinline__ __device__ float3 read_vec_or_zero(
    const float *x,
    const float *y,
    const float *z,
    int index) {
    return make_f3(read_or_zero(x, index),
                     read_or_zero(y, index),
                     read_or_zero(z, index));
}

static __forceinline__ __device__ void atomic_add_vec(
    float *x,
    float *y,
    float *z,
    int index,
    float3 value) {
    if (x != nullptr) {
        atomicAdd(x + index, value.x);
    }
    if (y != nullptr) {
        atomicAdd(y + index, value.y);
    }
    if (z != nullptr) {
        atomicAdd(z + index, value.z);
    }
}

static __forceinline__ __device__ bool chain_keller_target(
    const DfrChainAccumADParams &params,
    int lane,
    unsigned int stream,
    float3 incident_vec,
    float3 edge_point,
    float3 edge_dir,
    float3 &target,
    float3 &ko,
    float &ray_t,
    float &sin_theta,
    float &cos_theta) {
    const float3 incident = normalize3(incident_vec);
    const float axial = fminf(fmaxf(dot3(incident, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, incident);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    sincosf(2.f * kPi * uniform01(static_cast<unsigned int>(lane),
                                  stream,
                                  static_cast<unsigned int>(params.seed)),
            &sin_theta,
            &cos_theta);
    ko = normalize3(axial * edge_dir +
                    radial * (cos_theta * basis0 + sin_theta * basis1));
    const float denom = component(ko, params.grid_axis);
    if (fabsf(denom) <= kDfrEps) {
        return false;
    }
    ray_t = (params.grid_position - component(edge_point, params.grid_axis)) / denom;
    if (!(ray_t > kDfrRayBias) || !isfinite(ray_t)) {
        return false;
    }
    target = edge_point + ray_t * ko;
    return true;
}

static __forceinline__ __device__ float3 chain_keller_target_jvp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float3 dot_incident_raw,
    float3 dot_edge_point,
    float3 dot_edge_dir) {
    const float3 incident = normalize3(p.keller_incident);
    const float3 dot_incident =
        normalize_jvp(incident, p.keller_incident_norm, dot_incident_raw);
    const ChainEventPrimal &terminal = p.has_third ? p.third : p.second;
    const float unclamped_axial = dot3(incident, terminal.edge_dir);
    const bool axial_active = unclamped_axial > -1.f && unclamped_axial < 1.f;
    const float axial = fminf(fmaxf(unclamped_axial, -1.f), 1.f);
    const float dot_axial = axial_active
                                ? dot3(dot_incident, terminal.edge_dir) +
                                      dot3(incident, dot_edge_dir)
                                : 0.f;
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float dot_radial =
        radial > kDfrEps ? (-(axial / radial) * dot_axial) : 0.f;
    const float3 basis0 = stable_perpendicular(terminal.edge_dir, incident);
    const float3 dot_basis0 =
        stable_perpendicular_jvp(terminal.edge_dir,
                                 dot_edge_dir,
                                 incident,
                                 dot_incident,
                                 basis0);
    const float3 basis1_raw = cross3(terminal.edge_dir, basis0);
    const float3 basis1 = normalize3(basis1_raw);
    const float basis1_norm = norm3(basis1_raw);
    const float3 dot_basis1_raw =
        cross3(dot_edge_dir, basis0) + cross3(terminal.edge_dir, dot_basis0);
    const float3 dot_basis1 = normalize_jvp(basis1, basis1_norm, dot_basis1_raw);
    const float3 radial_basis = p.keller_cos * basis0 + p.keller_sin * basis1;
    const float3 dot_radial_basis =
        p.keller_cos * dot_basis0 + p.keller_sin * dot_basis1;
    const float3 ko_raw = axial * terminal.edge_dir + radial * radial_basis;
    const float3 dot_ko_raw =
        dot_axial * terminal.edge_dir +
        axial * dot_edge_dir +
        dot_radial * radial_basis +
        radial * dot_radial_basis;
    const float ko_norm = norm3(ko_raw);
    const float3 dot_ko = normalize_jvp(p.keller_ko, ko_norm, dot_ko_raw);
    const float denom = component(p.keller_ko, params.grid_axis);
    const float dot_denom = component(dot_ko, params.grid_axis);
    const float numerator =
        params.grid_position - component(terminal.edge_point, params.grid_axis);
    const float dot_numerator = -component(dot_edge_point, params.grid_axis);
    const float dot_t =
        (dot_numerator * denom - numerator * dot_denom) /
        fmaxf(denom * denom, kDfrEps);
    return dot_edge_point + dot_t * p.keller_ko + p.keller_ray_t * dot_ko;
}

static __forceinline__ __device__ float3 chain_suffix_target_jvp(
    const ChainPrimal &p,
    const ChainTangent &tangent,
    float3 terminal_point,
    float3 dot_terminal_point,
    float &dot_suffix_fspl) {
    dot_suffix_fspl = 0.f;
    if (!p.is_suffix) {
        return make_f3(0.f, 0.f, 0.f);
    }

    const float3 dot_normal =
        normalize_jvp(p.suffix_normal, p.suffix_normal_norm, tangent.suffix_normal_raw);
    const float3 plane_delta = terminal_point - p.suffix_p0;
    const float plane_distance = dot3(plane_delta, p.suffix_normal);
    const float dot_plane_distance =
        dot3(dot_terminal_point - tangent.suffix_p0, p.suffix_normal) +
        dot3(plane_delta, dot_normal);
    const float3 dot_image_source =
        dot_terminal_point -
        2.f * (dot_plane_distance * p.suffix_normal + plane_distance * dot_normal);
    const float3 ray_delta = p.grid_target - p.suffix_image_source;
    const float ray_dist = norm3(ray_delta);
    const float3 dot_ray_delta = -1.f * dot_image_source;
    const float3 dot_ray_dir =
        normalize_jvp(p.suffix_ray_dir, ray_dist, dot_ray_delta);
    const float denom = dot3(p.suffix_ray_dir, p.suffix_normal);
    const float dot_denom =
        dot3(dot_ray_dir, p.suffix_normal) + dot3(p.suffix_ray_dir, dot_normal);
    const float3 plane_to_image = p.suffix_p0 - p.suffix_image_source;
    const float numerator = dot3(plane_to_image, p.suffix_normal);
    const float dot_numerator =
        dot3(tangent.suffix_p0 - dot_image_source, p.suffix_normal) +
        dot3(plane_to_image, dot_normal);
    const float dot_ray_t =
        (dot_numerator * denom - numerator * dot_denom) /
        fmaxf(denom * denom, kDfrEps);
    const float3 dot_reflection_point =
        dot_image_source + dot_ray_t * p.suffix_ray_dir + p.suffix_ray_t * dot_ray_dir;

    if (p.suffix_outgoing_dist2 > kDfrEps && p.suffix_fspl != 0.f) {
        const float3 outgoing = p.grid_target - p.final_target;
        const float dot_outgoing_dist2 =
            2.f * dot3(outgoing, -1.f * dot_reflection_point);
        dot_suffix_fspl =
            p.suffix_fspl * (-(dot_outgoing_dist2 / p.suffix_outgoing_dist2));
    }
    return dot_reflection_point;
}

static __forceinline__ __device__ float chain_event_weight(
    const DfrChainAccumADParams &params,
    ChainEventPrimal &event,
    float3 source,
    float3 target,
    float src_power,
    int face0_prim,
    int face1_prim,
    float edge_t_min,
    float edge_t_max,
    float exterior_angle) {
    event.source = source;
    event.target = target;
    event.src_power = src_power;
    event.exterior_angle = exterior_angle;
    event.edge_length = fmaxf(edge_t_max - edge_t_min, 0.f);
    event.edge_length_active = (edge_t_max - edge_t_min) > 0.f;
    const float exterior_clamped = fmaxf(exterior_angle, 0.25f * kPi);
    event.wedge_scale = fminf(exterior_clamped / (2.f * kPi), 2.f);
    event.wedge_active = exterior_angle > 0.25f * kPi &&
                         exterior_clamped / (2.f * kPi) < 2.f;
    event.material_idx = material_index_for_faces(params, face0_prim, face1_prim);
    event.material_gain = 1.f;
    event.material_active = false;
    if (event.material_idx >= 0) {
        const float raw_gain = params.material_gain[event.material_idx];
        event.material_gain = fmaxf(raw_gain, 0.f);
        event.material_active = raw_gain > 0.f;
    }
    const float source_dist = fmaxf(norm3(event.edge_point - source), kDfrEps);
    const float target_dist = fmaxf(norm3(target - event.edge_point), kDfrEps);
    event.source_dist2 = source_dist * source_dist;
    event.target_dist2 = target_dist * target_dist;
    event.contribution =
        src_power *
        event.material_gain *
        event.edge_length *
        event.wedge_scale /
        (event.source_dist2 * event.target_dist2);
    return event.contribution;
}

static __forceinline__ __device__ bool load_chain_event_initial(
    const DfrChainAccumADParams &params,
    ChainEventPrimal &event,
    int idx,
    float u) {
    event.state_idx = idx;
    event.edge_u = u;
    event.edge_pos = make_f3(params.state_edge_pos_x[idx],
                               params.state_edge_pos_y[idx],
                               params.state_edge_pos_z[idx]);
    event.edge_dir_raw = make_f3(params.state_edge_dir_x[idx],
                                   params.state_edge_dir_y[idx],
                                   params.state_edge_dir_z[idx]);
    event.edge_dir_norm = norm3(event.edge_dir_raw);
    if (!(event.edge_dir_norm > kDfrEps) || !isfinite(event.edge_dir_norm)) {
        return false;
    }
    event.edge_dir = (1.f / event.edge_dir_norm) * event.edge_dir_raw;
    const float t_min = params.state_edge_t_min[idx];
    const float t_max = params.state_edge_t_max[idx];
    event.edge_t = t_min + u * (t_max - t_min);
    event.edge_point = event.edge_pos + event.edge_t * event.edge_dir;
    return true;
}

static __forceinline__ __device__ bool load_chain_event_recursive(
    const DfrChainAccumADParams &params,
    ChainEventPrimal &event,
    int idx,
    float u) {
    event.state_idx = idx;
    event.edge_u = u;
    event.edge_pos = make_f3(params.recursive_state_edge_pos_x[idx],
                               params.recursive_state_edge_pos_y[idx],
                               params.recursive_state_edge_pos_z[idx]);
    event.edge_dir_raw = make_f3(params.recursive_state_edge_dir_x[idx],
                                   params.recursive_state_edge_dir_y[idx],
                                   params.recursive_state_edge_dir_z[idx]);
    event.edge_dir_norm = norm3(event.edge_dir_raw);
    if (!(event.edge_dir_norm > kDfrEps) || !isfinite(event.edge_dir_norm)) {
        return false;
    }
    event.edge_dir = (1.f / event.edge_dir_norm) * event.edge_dir_raw;
    const float t_min = params.recursive_state_edge_t_min[idx];
    const float t_max = params.recursive_state_edge_t_max[idx];
    event.edge_t = t_min + u * (t_max - t_min);
    event.edge_point = event.edge_pos + event.edge_t * event.edge_dir;
    return true;
}

static __forceinline__ __device__ bool load_chain_primal(
    const DfrChainAccumADParams &params,
    int lane,
    ChainPrimal &p) {
    if (lane >= params.n_rays ||
        params.tape_active == nullptr ||
        params.tape_active[lane] == 0u) {
        return false;
    }
    p.lane = lane;
    p.cell = params.tape_cell != nullptr ? params.tape_cell[lane] : 0;
    if (p.cell < 0 ||
        p.cell >= params.grid_resolution0 * params.grid_resolution1 ||
        params.state_count <= 0 ||
        params.recursive_state_count <= 0 ||
        (params.max_order != 2 && params.max_order != 3)) {
        return false;
    }
    p.is_keller = lane >= params.direct_samples &&
                  lane < params.direct_samples + params.keller_samples;
    p.is_suffix = lane >= params.direct_samples + params.keller_samples &&
                  lane < params.direct_samples + params.keller_samples + params.suffix_samples;
    p.has_third = params.max_order == 3;
    p.sample_count = p.is_keller
                         ? params.keller_samples
                         : (p.is_suffix ? params.suffix_samples : params.direct_samples);
    if (p.sample_count <= 0) {
        return false;
    }
    p.first_idx = lane % params.state_count;
    const unsigned int second_hash = hash_u32(
        static_cast<unsigned int>(lane) ^
        (static_cast<unsigned int>(params.seed) * 0x9e3779b9u) ^
        0x51ed270bu);
    p.second_idx = static_cast<int>(
        second_hash % static_cast<unsigned int>(params.recursive_state_count));
    const unsigned int third_hash = hash_u32(
        static_cast<unsigned int>(lane) ^
        (static_cast<unsigned int>(params.seed) * 0x85ebca6bu) ^
        0xc2b2ae35u);
    p.third_idx = static_cast<int>(
        third_hash % static_cast<unsigned int>(params.recursive_state_count));

    const float first_u = uniform01(static_cast<unsigned int>(lane),
                                    0u,
                                    static_cast<unsigned int>(params.seed));
    const float second_u = uniform01(static_cast<unsigned int>(lane),
                                     2u,
                                     static_cast<unsigned int>(params.seed));
    const float third_u = uniform01(static_cast<unsigned int>(lane),
                                    4u,
                                    static_cast<unsigned int>(params.seed));
    if (!load_chain_event_initial(params, p.first, p.first_idx, first_u) ||
        !load_chain_event_recursive(params, p.second, p.second_idx, second_u)) {
        return false;
    }
    if (p.has_third &&
        !load_chain_event_recursive(params, p.third, p.third_idx, third_u)) {
        return false;
    }

    const float3 source = make_f3(params.state_src_x[p.first_idx],
                                    params.state_src_y[p.first_idx],
                                    params.state_src_z[p.first_idx]);
    p.grid_target = grid_cell_center(params, p.cell);
    p.final_target = p.grid_target;
    p.suffix_material_idx = -1;
    p.suffix_material_gain = 1.f;
    p.suffix_reflection_gain = 1.f;
    p.suffix_fspl = 1.f;
    p.suffix_candidate_count = 1.f;
    p.suffix_outgoing_dist2 = 1.f;
    p.suffix_ray_t = 0.f;
    p.suffix_normal_norm = 1.f;
    p.suffix_material_active = false;
    p.suffix_p0 = make_f3(0.f, 0.f, 0.f);
    p.suffix_normal = make_f3(0.f, 0.f, 1.f);
    p.suffix_image_source = make_f3(0.f, 0.f, 0.f);
    p.suffix_ray_dir = make_f3(0.f, 0.f, 0.f);
    p.keller_ko = make_f3(0.f, 0.f, 0.f);
    p.keller_ray_t = 0.f;
    p.keller_sin = 0.f;
    p.keller_cos = 1.f;
    p.keller_incident =
        p.has_third ? (p.third.edge_point - p.second.edge_point)
                    : (p.second.edge_point - p.first.edge_point);
    p.keller_incident_norm = norm3(p.keller_incident);
    if (p.is_keller) {
        const ChainEventPrimal &terminal = p.has_third ? p.third : p.second;
        if (!chain_keller_target(params,
                                 lane,
                                 7u + static_cast<unsigned int>(params.max_order),
                                 p.keller_incident,
                                 terminal.edge_point,
                                 terminal.edge_dir,
                                 p.final_target,
                                 p.keller_ko,
                                 p.keller_ray_t,
                                 p.keller_sin,
                                 p.keller_cos)) {
            return false;
        }
    }
    if (p.is_suffix) {
        const ChainEventPrimal &terminal = p.has_third ? p.third : p.second;
        const int face0_prim = p.has_third
                                   ? params.recursive_state_prim0[p.third_idx]
                                   : params.recursive_state_prim0[p.second_idx];
        const int face1_prim = p.has_third
                                   ? params.recursive_state_prim1[p.third_idx]
                                   : params.recursive_state_prim1[p.second_idx];
        if (!suffix_reflection_connection(params,
                                          terminal.edge_point,
                                          p.grid_target,
                                          face0_prim,
                                          face1_prim,
                                          static_cast<unsigned int>(lane),
                                          p.final_target,
                                          p.suffix_material_idx,
                                          p.suffix_reflection_gain,
                                          p.suffix_fspl,
                                          p.suffix_candidate_count,
                                          p.suffix_normal,
                                          p.suffix_p0,
                                          p.suffix_normal_norm,
                                          p.suffix_image_source,
                                          p.suffix_ray_dir,
                                          p.suffix_ray_t,
                                          p.suffix_outgoing_dist2,
                                          p.suffix_material_gain,
                                          p.suffix_material_active)) {
            return false;
        }
    }

    const float first_weight = chain_event_weight(
        params,
        p.first,
        source,
        p.second.edge_point,
        params.state_src_power[p.first_idx],
        params.state_prim0[p.first_idx],
        params.state_prim1[p.first_idx],
        params.state_edge_t_min[p.first_idx],
        params.state_edge_t_max[p.first_idx],
        params.state_exterior_angle[p.first_idx]);
    const float3 second_target = p.has_third ? p.third.edge_point : p.final_target;
    const float second_weight = chain_event_weight(
        params,
        p.second,
        p.first.edge_point,
        second_target,
        1.f,
        params.recursive_state_prim0[p.second_idx],
        params.recursive_state_prim1[p.second_idx],
        params.recursive_state_edge_t_min[p.second_idx],
        params.recursive_state_edge_t_max[p.second_idx],
        params.recursive_state_exterior_angle[p.second_idx]);
    float chain_weight = first_weight * second_weight;
    if (p.has_third) {
        const float third_weight = chain_event_weight(
            params,
            p.third,
            p.second.edge_point,
            p.final_target,
            1.f,
            params.recursive_state_prim0[p.third_idx],
            params.recursive_state_prim1[p.third_idx],
            params.recursive_state_edge_t_min[p.third_idx],
            params.recursive_state_edge_t_max[p.third_idx],
            params.recursive_state_exterior_angle[p.third_idx]);
        chain_weight *= third_weight;
    }
    const float wave_gain_per_event =
        (params.wavelength * (1.f / (4.f * kPi))) *
        (params.wavelength * (1.f / (4.f * kPi)));
    p.wave_gain = p.has_third ? wave_gain_per_event * wave_gain_per_event
                              : wave_gain_per_event;
    p.sample_norm = 1.f / fmaxf(static_cast<float>(p.sample_count), 1.f);
    const float suffix_scale =
        p.is_suffix
            ? p.suffix_reflection_gain *
                  p.suffix_fspl *
                  fmaxf(p.suffix_candidate_count, 1.f)
            : 1.f;
    p.contribution =
        chain_weight * p.wave_gain * params.grid_cell_area * p.sample_norm * suffix_scale;
    return p.contribution > 0.f && isfinite(p.contribution);
}

static __forceinline__ __device__ float3 chain_event_point_jvp(
    const ChainEventPrimal &event,
    const ChainEventTangent &tangent) {
    const float dot_t =
        (1.f - event.edge_u) * tangent.edge_t_min +
        event.edge_u * tangent.edge_t_max;
    const float3 dot_edge_dir =
        normalize_jvp(event.edge_dir, event.edge_dir_norm, tangent.edge_dir_raw);
    return tangent.edge_pos + dot_t * event.edge_dir + event.edge_t * dot_edge_dir;
}

static __forceinline__ __device__ float chain_event_weight_jvp(
    const ChainEventPrimal &event,
    const ChainEventTangent &tangent,
    float3 dot_edge_point,
    float3 dot_source,
    float3 dot_target) {
    float dot_weight = 0.f;
    if (event.src_power != 0.f) {
        dot_weight += event.contribution * (tangent.src_power / event.src_power);
    }
    if (event.material_gain != 0.f) {
        dot_weight += event.contribution * (tangent.material_gain / event.material_gain);
    }
    if (event.edge_length_active && event.edge_length != 0.f) {
        dot_weight +=
            event.contribution *
            ((tangent.edge_t_max - tangent.edge_t_min) / event.edge_length);
    }
    if (event.wedge_active && event.wedge_scale != 0.f) {
        dot_weight +=
            event.contribution *
            ((tangent.exterior_angle / (2.f * kPi)) / event.wedge_scale);
    }
    const float3 source_delta = event.edge_point - event.source;
    const float3 target_delta = event.target - event.edge_point;
    const float dot_source_dist2 =
        2.f * dot3(source_delta, dot_edge_point - dot_source);
    const float dot_target_dist2 =
        2.f * dot3(target_delta, dot_target - dot_edge_point);
    dot_weight +=
        event.contribution *
        (-(dot_source_dist2 / event.source_dist2) -
         (dot_target_dist2 / event.target_dist2));
    return dot_weight;
}

static __forceinline__ __device__ float chain_contribution_jvp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    const ChainTangent &tangent) {
    const float3 dot_first_point = chain_event_point_jvp(p.first, tangent.first);
    const float3 dot_second_point = chain_event_point_jvp(p.second, tangent.second);
    const float3 dot_third_point =
        p.has_third ? chain_event_point_jvp(p.third, tangent.third)
                    : make_f3(0.f, 0.f, 0.f);
    float3 dot_final_target = make_f3(0.f, 0.f, 0.f);
    float dot_suffix_fspl = 0.f;
    if (p.is_keller) {
        const ChainEventPrimal &terminal = p.has_third ? p.third : p.second;
        const ChainEventTangent &terminal_tangent =
            p.has_third ? tangent.third : tangent.second;
        const float3 dot_terminal_point =
            p.has_third ? dot_third_point : dot_second_point;
        const float3 dot_previous_point =
            p.has_third ? dot_second_point : dot_first_point;
        const float3 dot_terminal_dir =
            normalize_jvp(terminal.edge_dir,
                          terminal.edge_dir_norm,
                          terminal_tangent.edge_dir_raw);
        dot_final_target =
            chain_keller_target_jvp(params,
                                    p,
                                     dot_terminal_point - dot_previous_point,
                                     dot_terminal_point,
                                     dot_terminal_dir);
    } else if (p.is_suffix) {
        const ChainEventPrimal &terminal = p.has_third ? p.third : p.second;
        const float3 dot_terminal_point =
            p.has_third ? dot_third_point : dot_second_point;
        dot_final_target =
            chain_suffix_target_jvp(p,
                                    tangent,
                                    terminal.edge_point,
                                    dot_terminal_point,
                                    dot_suffix_fspl);
    }

    const float dot_first_weight = chain_event_weight_jvp(
        p.first,
        tangent.first,
        dot_first_point,
        tangent.first.source,
        dot_second_point);
    const float dot_second_weight = chain_event_weight_jvp(
        p.second,
        tangent.second,
        dot_second_point,
        dot_first_point,
        p.has_third ? dot_third_point : dot_final_target);
    float dot_chain_weight =
        dot_first_weight * p.second.contribution +
        p.first.contribution * dot_second_weight;
    float chain_weight = p.first.contribution * p.second.contribution;
    if (p.has_third) {
        const float dot_third_weight = chain_event_weight_jvp(
            p.third,
            tangent.third,
            dot_third_point,
            dot_second_point,
            dot_final_target);
        dot_chain_weight =
            dot_chain_weight * p.third.contribution +
            chain_weight * dot_third_weight;
        chain_weight *= p.third.contribution;
    }
    const float suffix_scale =
        p.is_suffix
            ? p.suffix_reflection_gain *
                  p.suffix_fspl *
                  fmaxf(p.suffix_candidate_count, 1.f)
            : 1.f;
    float dot_contribution =
        dot_chain_weight * p.wave_gain * params.grid_cell_area * p.sample_norm * suffix_scale;
    if (p.is_suffix && p.suffix_reflection_gain != 0.f) {
        const float dot_reflection_gain =
            p.suffix_material_active
                ? 2.f * p.suffix_material_gain * tangent.suffix_material_gain
                : 0.f;
        dot_contribution +=
            p.contribution * (dot_reflection_gain / p.suffix_reflection_gain);
    }
    if (p.is_suffix && p.suffix_fspl != 0.f) {
        dot_contribution += p.contribution * (dot_suffix_fspl / p.suffix_fspl);
    }
    (void) chain_weight;
    return dot_contribution;
}

static __forceinline__ __device__ ChainTangent chain_read_tangent(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p) {
    ChainTangent tangent = {};
    tangent.first.edge_pos =
        read_vec_or_zero(params.dot_state_edge_pos_x,
                         params.dot_state_edge_pos_y,
                         params.dot_state_edge_pos_z,
                         p.first_idx);
    tangent.first.edge_dir_raw =
        read_vec_or_zero(params.dot_state_edge_dir_x,
                         params.dot_state_edge_dir_y,
                         params.dot_state_edge_dir_z,
                         p.first_idx);
    tangent.first.edge_t_min = read_or_zero(params.dot_state_edge_t_min, p.first_idx);
    tangent.first.edge_t_max = read_or_zero(params.dot_state_edge_t_max, p.first_idx);
    tangent.first.source =
        read_vec_or_zero(params.dot_state_src_x,
                         params.dot_state_src_y,
                         params.dot_state_src_z,
                         p.first_idx);
    tangent.first.src_power = read_or_zero(params.dot_state_src_power, p.first_idx);
    tangent.first.exterior_angle =
        read_or_zero(params.dot_state_exterior_angle, p.first_idx);
    tangent.first.material_gain =
        (p.first.material_active && p.first.material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.first.material_idx)
            : 0.f;

    tangent.second.edge_pos =
        read_vec_or_zero(params.dot_recursive_state_edge_pos_x,
                         params.dot_recursive_state_edge_pos_y,
                         params.dot_recursive_state_edge_pos_z,
                         p.second_idx);
    tangent.second.edge_dir_raw =
        read_vec_or_zero(params.dot_recursive_state_edge_dir_x,
                         params.dot_recursive_state_edge_dir_y,
                         params.dot_recursive_state_edge_dir_z,
                         p.second_idx);
    tangent.second.edge_t_min =
        read_or_zero(params.dot_recursive_state_edge_t_min, p.second_idx);
    tangent.second.edge_t_max =
        read_or_zero(params.dot_recursive_state_edge_t_max, p.second_idx);
    tangent.second.exterior_angle =
        read_or_zero(params.dot_recursive_state_exterior_angle, p.second_idx);
    tangent.second.material_gain =
        (p.second.material_active && p.second.material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.second.material_idx)
            : 0.f;

    if (p.has_third) {
        tangent.third.edge_pos =
            read_vec_or_zero(params.dot_recursive_state_edge_pos_x,
                             params.dot_recursive_state_edge_pos_y,
                             params.dot_recursive_state_edge_pos_z,
                             p.third_idx);
        tangent.third.edge_dir_raw =
            read_vec_or_zero(params.dot_recursive_state_edge_dir_x,
                             params.dot_recursive_state_edge_dir_y,
                             params.dot_recursive_state_edge_dir_z,
                             p.third_idx);
        tangent.third.edge_t_min =
            read_or_zero(params.dot_recursive_state_edge_t_min, p.third_idx);
        tangent.third.edge_t_max =
            read_or_zero(params.dot_recursive_state_edge_t_max, p.third_idx);
        tangent.third.exterior_angle =
            read_or_zero(params.dot_recursive_state_exterior_angle, p.third_idx);
        tangent.third.material_gain =
            (p.third.material_active && p.third.material_idx >= 0)
                ? read_or_zero(params.dot_material_gain, p.third.material_idx)
                : 0.f;
    }
    tangent.suffix_material_gain =
        (p.suffix_material_active && p.suffix_material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.suffix_material_idx)
            : 0.f;
    if (p.is_suffix && p.suffix_material_idx >= 0) {
        tangent.suffix_p0 =
            read_vec_or_zero(params.dot_tri_p0_x,
                             params.dot_tri_p0_y,
                             params.dot_tri_p0_z,
                             p.suffix_material_idx);
        tangent.suffix_normal_raw =
            read_vec_or_zero(params.dot_tri_fn_x,
                             params.dot_tri_fn_y,
                             params.dot_tri_fn_z,
                             p.suffix_material_idx);
    }
    return tangent;
}

static __forceinline__ __device__ void add_chain_unit_vjp(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const ChainTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = chain_contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void chain_vjp_by_unit_jvps(
    const DfrChainAccumADParams &params,
    const ChainPrimal &p,
    float grad_contribution) {
    ChainTangent tangent = {};
    tangent.first.edge_pos = make_f3(1.f, 0.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_x, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_pos = make_f3(0.f, 1.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_y, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_pos = make_f3(0.f, 0.f, 1.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_z, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = make_f3(1.f, 0.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_x, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = make_f3(0.f, 1.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_y, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_dir_raw = make_f3(0.f, 0.f, 1.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_z, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_t_min = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_min, p.first_idx, tangent);
    tangent = {};
    tangent.first.edge_t_max = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_max, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = make_f3(1.f, 0.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_src_x, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = make_f3(0.f, 1.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_src_y, p.first_idx, tangent);
    tangent = {};
    tangent.first.source = make_f3(0.f, 0.f, 1.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_src_z, p.first_idx, tangent);
    tangent = {};
    tangent.first.src_power = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_src_power, p.first_idx, tangent);
    tangent = {};
    tangent.first.exterior_angle = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_state_exterior_angle, p.first_idx, tangent);
    if (p.first.material_active && p.first.material_idx >= 0) {
        tangent = {};
        tangent.first.material_gain = 1.f;
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_material_gain, p.first.material_idx, tangent);
    }

    tangent = {};
    tangent.second.edge_pos = make_f3(1.f, 0.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_x, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_pos = make_f3(0.f, 1.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_y, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_pos = make_f3(0.f, 0.f, 1.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_z, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = make_f3(1.f, 0.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_x, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = make_f3(0.f, 1.f, 0.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_y, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_dir_raw = make_f3(0.f, 0.f, 1.f);
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_z, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_t_min = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_t_min, p.second_idx, tangent);
    tangent = {};
    tangent.second.edge_t_max = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_t_max, p.second_idx, tangent);
    tangent = {};
    tangent.second.exterior_angle = 1.f;
    add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_exterior_angle, p.second_idx, tangent);
    if (p.second.material_active && p.second.material_idx >= 0) {
        tangent = {};
        tangent.second.material_gain = 1.f;
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_material_gain, p.second.material_idx, tangent);
    }

    if (p.has_third) {
        tangent = {};
        tangent.third.edge_pos = make_f3(1.f, 0.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_x, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_pos = make_f3(0.f, 1.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_y, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_pos = make_f3(0.f, 0.f, 1.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_pos_z, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = make_f3(1.f, 0.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_x, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = make_f3(0.f, 1.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_y, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_dir_raw = make_f3(0.f, 0.f, 1.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_dir_z, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_t_min = 1.f;
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_t_min, p.third_idx, tangent);
        tangent = {};
        tangent.third.edge_t_max = 1.f;
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_edge_t_max, p.third_idx, tangent);
        tangent = {};
        tangent.third.exterior_angle = 1.f;
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_recursive_state_exterior_angle, p.third_idx, tangent);
        if (p.third.material_active && p.third.material_idx >= 0) {
            tangent = {};
            tangent.third.material_gain = 1.f;
            add_chain_unit_vjp(params, p, grad_contribution, params.grad_material_gain, p.third.material_idx, tangent);
        }
    }
    if (p.suffix_material_active && p.suffix_material_idx >= 0) {
        tangent = {};
        tangent.suffix_material_gain = 1.f;
        add_chain_unit_vjp(params,
                           p,
                           grad_contribution,
                           params.grad_material_gain,
                           p.suffix_material_idx,
                           tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(1.f, 0.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(0.f, 1.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(0.f, 0.f, 1.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_z, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(1.f, 0.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(0.f, 1.f, 0.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(0.f, 0.f, 1.f);
        add_chain_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_z, p.suffix_material_idx, tangent);
    }
}

static __forceinline__ __device__ float contribution_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    const DfrTangent &tangent) {
    const float dot_edge_t =
        (1.f - p.edge_u) * tangent.edge_t_min +
        p.edge_u * tangent.edge_t_max;
    const float dot_edge_length =
        p.edge_length_active ? (tangent.edge_t_max - tangent.edge_t_min) : 0.f;
    const float dot_wedge =
        p.wedge_active ? tangent.exterior_angle / (2.f * kPi) : 0.f;
    const float3 dot_edge_dir =
        normalize_jvp(p.edge_dir, p.edge_dir_norm, tangent.edge_dir_raw);
    const float3 dot_edge_point =
        tangent.edge_pos + dot_edge_t * p.edge_dir + p.edge_t * dot_edge_dir;
    float dot_suffix_fspl = 0.f;
    const float3 dot_target =
        p.is_suffix
            ? suffix_target_jvp(p, tangent, dot_edge_point, dot_suffix_fspl)
            : keller_target_jvp(params, p, dot_edge_point, dot_edge_dir, tangent.wi_raw);

    float dot_contribution = p.common_no_src * tangent.src_power;
    if (p.material_gain != 0.f) {
        dot_contribution += p.contribution * (tangent.material_gain / p.material_gain);
    }
    if (p.is_suffix && p.suffix_reflection_gain != 0.f) {
        const float dot_reflection_gain =
            p.suffix_material_active
                ? 2.f * p.suffix_material_gain * tangent.suffix_material_gain
                : 0.f;
        dot_contribution +=
            p.contribution * (dot_reflection_gain / p.suffix_reflection_gain);
    }
    if (p.is_suffix && p.suffix_fspl != 0.f) {
        dot_contribution += p.contribution * (dot_suffix_fspl / p.suffix_fspl);
    }
    if (p.edge_length != 0.f) {
        dot_contribution += p.contribution * (dot_edge_length / p.edge_length);
    }
    if (p.wedge_scale != 0.f) {
        dot_contribution += p.contribution * (dot_wedge / p.wedge_scale);
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float dot_source_dist2 =
        2.f * dot3(source_delta, dot_edge_point - tangent.source);
    const float dot_target_dist2 =
        2.f * dot3(target_delta, dot_target - dot_edge_point);
    dot_contribution +=
        p.contribution *
        (-(dot_source_dist2 / p.source_dist2) -
         (dot_target_dist2 / p.target_dist2));
    return dot_contribution;
}

static __forceinline__ __device__ float direct_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p) {
    DfrTangent tangent = {};
    tangent.edge_pos =
        read_vec_or_zero(params.dot_state_edge_pos_x,
                         params.dot_state_edge_pos_y,
                         params.dot_state_edge_pos_z,
                         p.state_idx);
    tangent.edge_dir_raw =
        read_vec_or_zero(params.dot_state_edge_dir_x,
                         params.dot_state_edge_dir_y,
                         params.dot_state_edge_dir_z,
                         p.state_idx);
    tangent.edge_t_min = read_or_zero(params.dot_state_edge_t_min, p.state_idx);
    tangent.edge_t_max = read_or_zero(params.dot_state_edge_t_max, p.state_idx);
    tangent.source =
        read_vec_or_zero(params.dot_state_src_x,
                         params.dot_state_src_y,
                         params.dot_state_src_z,
                         p.state_idx);
    tangent.wi_raw =
        read_vec_or_zero(params.dot_state_wi_x,
                         params.dot_state_wi_y,
                         params.dot_state_wi_z,
                         p.state_idx);
    tangent.src_power = read_or_zero(params.dot_state_src_power, p.state_idx);
    tangent.exterior_angle =
        read_or_zero(params.dot_state_exterior_angle, p.state_idx);
    tangent.material_gain =
        (p.material_active && p.material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.material_idx)
            : 0.f;
    tangent.suffix_material_gain =
        (p.suffix_material_active && p.suffix_material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.suffix_material_idx)
            : 0.f;
    if (p.is_suffix && p.suffix_material_idx >= 0) {
        tangent.suffix_p0 =
            read_vec_or_zero(params.dot_tri_p0_x,
                             params.dot_tri_p0_y,
                             params.dot_tri_p0_z,
                             p.suffix_material_idx);
        tangent.suffix_normal_raw =
            read_vec_or_zero(params.dot_tri_fn_x,
                             params.dot_tri_fn_y,
                             params.dot_tri_fn_z,
                             p.suffix_material_idx);
    }
    return contribution_jvp(params, p, tangent);
}

static __forceinline__ __device__ void add_unit_vjp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void vjp_by_unit_jvps(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution) {
    DfrTangent tangent = {};
    tangent.edge_pos = make_f3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_x, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = make_f3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_y, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = make_f3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_z, p.state_idx, tangent);

    tangent = {};
    tangent.edge_dir_raw = make_f3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_x, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = make_f3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_y, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = make_f3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_z, p.state_idx, tangent);

    tangent = {};
    tangent.edge_t_min = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_min, p.state_idx, tangent);
    tangent = {};
    tangent.edge_t_max = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_max, p.state_idx, tangent);

    tangent = {};
    tangent.source = make_f3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_x, p.state_idx, tangent);
    tangent = {};
    tangent.source = make_f3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_y, p.state_idx, tangent);
    tangent = {};
    tangent.source = make_f3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_z, p.state_idx, tangent);

    tangent = {};
    tangent.wi_raw = make_f3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_x, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = make_f3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_y, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = make_f3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_z, p.state_idx, tangent);

    tangent = {};
    tangent.src_power = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_power, p.state_idx, tangent);
    tangent = {};
    tangent.exterior_angle = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_exterior_angle, p.state_idx, tangent);
    if (p.material_active && p.material_idx >= 0) {
        tangent = {};
        tangent.material_gain = 1.f;
        add_unit_vjp(params, p, grad_contribution, params.grad_material_gain, p.material_idx, tangent);
    }
    if (p.suffix_material_active && p.suffix_material_idx >= 0) {
        tangent = {};
        tangent.suffix_material_gain = 1.f;
        add_unit_vjp(params,
                     p,
                     grad_contribution,
                     params.grad_material_gain,
                     p.suffix_material_idx,
                     tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(1.f, 0.f, 0.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(0.f, 1.f, 0.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_p0 = make_f3(0.f, 0.f, 1.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_p0_z, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(1.f, 0.f, 0.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_x, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(0.f, 1.f, 0.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_y, p.suffix_material_idx, tangent);
        tangent = {};
        tangent.suffix_normal_raw = make_f3(0.f, 0.f, 1.f);
        add_unit_vjp(params, p, grad_contribution, params.grad_tri_fn_z, p.suffix_material_idx, tangent);
    }
}

__global__ void dfr_direct_accum_jvp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }
    const float dot_contribution = direct_jvp(params, p);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_direct_accum_vjp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }

    float grad_contribution =
        read_or_zero(params.grad_out_power, p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_or_zero(params.grad_out_field_x_re, p.cell) * 0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }

    if (p.is_keller || p.is_suffix) {
        vjp_by_unit_jvps(params, p, grad_contribution);
        return;
    }

    const float grad_src_power = grad_contribution * p.common_no_src;
    if (params.grad_state_src_power != nullptr) {
        atomicAdd(params.grad_state_src_power + p.state_idx, grad_src_power);
    }

    if (p.material_active &&
        p.material_idx >= 0 &&
        params.grad_material_gain != nullptr) {
        const float grad_gain =
            grad_contribution * p.contribution / fmaxf(p.material_gain, kDfrEps);
        atomicAdd(params.grad_material_gain + p.material_idx, grad_gain);
    }

    float grad_edge_length = 0.f;
    if (p.edge_length_active && p.edge_length > kDfrEps) {
        grad_edge_length = grad_contribution * p.contribution / p.edge_length;
    }
    if (p.wedge_active && params.grad_state_exterior_angle != nullptr) {
        const float grad_wedge =
            grad_contribution * p.contribution / fmaxf(p.wedge_scale, kDfrEps);
        atomicAdd(params.grad_state_exterior_angle + p.state_idx,
                  grad_wedge / (2.f * kPi));
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float3 d_contribution_d_edge =
        p.contribution *
        ((-2.f / p.source_dist2) * source_delta +
         (2.f / p.target_dist2) * target_delta);
    const float3 d_contribution_d_source =
        p.contribution * ((2.f / p.source_dist2) * source_delta);

    const float3 grad_edge_point = grad_contribution * d_contribution_d_edge;
    const float3 grad_source = grad_contribution * d_contribution_d_source;
    atomic_add_vec(params.grad_state_src_x,
                   params.grad_state_src_y,
                   params.grad_state_src_z,
                   p.state_idx,
                   grad_source);
    atomic_add_vec(params.grad_state_edge_pos_x,
                   params.grad_state_edge_pos_y,
                   params.grad_state_edge_pos_z,
                   p.state_idx,
                   grad_edge_point);

    const float grad_edge_t = dot3(grad_edge_point, p.edge_dir);
    if (params.grad_state_edge_t_min != nullptr) {
        atomicAdd(params.grad_state_edge_t_min + p.state_idx,
                  (1.f - p.edge_u) * grad_edge_t - grad_edge_length);
    }
    if (params.grad_state_edge_t_max != nullptr) {
        atomicAdd(params.grad_state_edge_t_max + p.state_idx,
                  p.edge_u * grad_edge_t + grad_edge_length);
    }

    const float3 grad_edge_dir = p.edge_t * grad_edge_point;
    const float3 grad_edge_dir_raw =
        (1.f / p.edge_dir_norm) *
        (grad_edge_dir - dot3(p.edge_dir, grad_edge_dir) * p.edge_dir);
    atomic_add_vec(params.grad_state_edge_dir_x,
                   params.grad_state_edge_dir_y,
                   params.grad_state_edge_dir_z,
                   p.state_idx,
                   grad_edge_dir_raw);
}

__global__ void dfr_chain_accum_jvp_kernel(DfrChainAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    const ChainTangent tangent = chain_read_tangent(params, p);
    const float dot_contribution = chain_contribution_jvp(params, p, tangent);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kDfrEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_chain_accum_vjp_kernel(DfrChainAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    ChainPrimal p;
    if (!load_chain_primal(params, lane, p)) {
        return;
    }
    float grad_contribution =
        read_or_zero(params.grad_out_power, p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kDfrEps) {
        grad_contribution +=
            read_or_zero(params.grad_out_field_x_re, p.cell) * 0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }
    chain_vjp_by_unit_jvps(params, p, grad_contribution);
}

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

template <typename Params, typename Kernel>
void launch_ad_kernel(const char *name,
                      Kernel kernel,
                      const Params &params) {
    if (params.n_rays <= 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch(name,
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error("dfr_direct_accum_ad_gpu(): failed to launch kernel");
}

} // namespace

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_jvp_kernel",
                     dfr_direct_accum_jvp_kernel,
                     params);
}

void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_vjp_kernel",
                     dfr_direct_accum_vjp_kernel,
                     params);
}

void dfr_chain_accum_jvp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_jvp_kernel",
                     dfr_chain_accum_jvp_kernel,
                     params);
}

void dfr_chain_accum_vjp_gpu(const DfrChainAccumADParams &params) {
    launch_ad_kernel("dfr_chain_accum_vjp_kernel",
                     dfr_chain_accum_vjp_kernel,
                     params);
}

} // namespace raydtorch

