#include <optix.h>
#include <optix_device.h>

#include <raydn/common/math.cuh>
#include <raydn/diffraction/accum_params.h>
#include <utd/utd_math.h>

namespace raydn {

extern "C" {
extern __constant__ DfrAccumParams params;
}

namespace {

constexpr float kDfrEps = 1e-6f;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

static __forceinline__ __device__ void clear_payload(HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(1e8f);
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.prim);
    optixSetPayload_3(payload.instance);
}

static __forceinline__ __device__ void trace_handle(OptixTraversableHandle handle,
                                                    float3 origin,
                                                    float3 direction,
                                                    float tmax,
                                                    HitPayload &payload) {
    clear_payload(payload);
    if (handle == 0ull || tmax <= kRayTMin) {
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
               payload.prim,
               payload.instance);
}

static __forceinline__ __device__ HitPayload choose_hit(HitPayload a, HitPayload b) {
    if (a.hit == 0u) {
        return b;
    }
    if (b.hit == 0u) {
        return a;
    }
    return __uint_as_float(a.t) <= __uint_as_float(b.t) ? a : b;
}

static __forceinline__ __device__ bool active_for_state(
    const uint8_t *mask,
    int width,
    int stride,
    int state_idx) {
    if (mask == nullptr) {
        return true;
    }
    const int active_idx = width == 1 ? 0 : state_idx;
    return mask[active_idx * stride] != 0u;
}

template <bool PrimaryOnly>
static __forceinline__ __device__ HitPayload trace_scene_impl(float3 origin,
                                                              float3 direction,
                                                              float tmax) {
    HitPayload primary;
    trace_handle(params.primary_handle, origin, direction, tmax, primary);
    if constexpr (PrimaryOnly) {
        return primary;
    } else if (params.split_mode == 0) {
        return primary;
    }
    HitPayload secondary;
    trace_handle(params.secondary_handle, origin, direction, tmax, secondary);
    return choose_hit(primary, secondary);
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool visible_segment_impl(float3 start, float3 end) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit =
        trace_scene_impl<PrimaryOnly>(start + kDfrRayBias * dir, dir, fmaxf(dist - 2.f * kDfrRayBias, 0.f));
    return hit.hit == 0u;
}

static __forceinline__ __device__ int global_primitive_id(const HitPayload &hit) {
    if (hit.hit == 0u) {
        return -1;
    }
    const int instance = static_cast<int>(hit.instance);
    if (params.face_offsets != nullptr &&
        instance >= 0 &&
        instance < params.n_meshes) {
        return params.face_offsets[instance] + static_cast<int>(hit.prim);
    }
    return static_cast<int>(hit.prim);
}


static __forceinline__ __device__ float3 face_normal_for_global_prim(int prim) {
    if (prim < 0 || prim >= params.n_triangles || params.tri_fn_x == nullptr) {
        return make_f3(0.f, 0.f, 0.f);
    }
    return normalize3(make_f3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]));
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool point_inside_one_ray_impl(float3 point, float3 ray_dir) {
    const HitPayload hit =
        trace_scene_impl<PrimaryOnly>(point + 1.0e-3f * ray_dir, ray_dir, 1.0e8f);
    if (hit.hit == 0u) {
        return false;
    }
    const float3 normal = face_normal_for_global_prim(global_primitive_id(hit));
    return dot3(normal, ray_dir) > 0.f;
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool point_inside_closed_mesh_robust_impl(float3 point) {
    const float3 d0 = normalize3(make_f3(0.81234133f, 0.52311241f, 0.25843197f));
    const float3 d1 = normalize3(make_f3(-0.37139068f, 0.60114462f, 0.70757474f));
    return point_inside_one_ray_impl<PrimaryOnly>(point, d0) &&
           point_inside_one_ray_impl<PrimaryOnly>(point, d1);
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool visible_segment_ignore_prim_impl(float3 start,
                                                                        float3 end,
                                                                        int ignore_prim) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit =
        trace_scene_impl<PrimaryOnly>(start + kDfrRayBias * dir, dir, fmaxf(dist - 2.f * kDfrRayBias, 0.f));
    if (hit.hit == 0u) {
        return true;
    }
    return global_primitive_id(hit) == ignore_prim;
}

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

static __forceinline__ __device__ float3 state_vec(const float *x,
                                                   const float *y,
                                                   const float *z,
                                                   int stride,
                                                   int idx) {
    const int offset = idx * stride;
    return make_f3(x[offset], y[offset], z[offset]);
}

static __forceinline__ __device__ float3 optional_state_vec(const float *x,
                                                            const float *y,
                                                            const float *z,
                                                            int stride,
                                                            int idx) {
    const int offset = idx * stride;
    return make_f3(x != nullptr ? x[offset] : 0.f,
                   y != nullptr ? y[offset] : 0.f,
                   z != nullptr ? z[offset] : 0.f);
}

static __forceinline__ __device__ float3 recursive_state_vec(const float *x,
                                                             const float *y,
                                                             const float *z,
                                                             int stride,
                                                             int idx) {
    const int offset = idx * stride;
    return make_f3(x[offset], y[offset], z[offset]);
}

static __forceinline__ __device__ float read_f32(const float *ptr, int stride, int idx) {
    return ptr[idx * stride];
}

static __forceinline__ __device__ int read_i32(const int *ptr, int stride, int idx) {
    return ptr[idx * stride];
}

static __forceinline__ __device__ uint8_t read_u8(const uint8_t *ptr, int stride, int idx) {
    return ptr[idx * stride];
}

static __forceinline__ __device__ int state_edge_index_at(int idx) {
    return read_i32(params.state_edge_index, params.state_edge_index_stride, idx);
}

static __forceinline__ __device__ float3 state_edge_pos_at(int idx) {
    return state_vec(params.state_edge_pos_x,
                     params.state_edge_pos_y,
                     params.state_edge_pos_z,
                     params.state_edge_pos_stride,
                     idx);
}

static __forceinline__ __device__ float3 state_edge_dir_at(int idx) {
    return state_vec(params.state_edge_dir_x,
                     params.state_edge_dir_y,
                     params.state_edge_dir_z,
                     params.state_edge_dir_stride,
                     idx);
}

static __forceinline__ __device__ float state_edge_t_min_at(int idx) {
    return read_f32(params.state_edge_t_min, params.state_edge_t_min_stride, idx);
}

static __forceinline__ __device__ float state_edge_t_max_at(int idx) {
    return read_f32(params.state_edge_t_max, params.state_edge_t_max_stride, idx);
}

static __forceinline__ __device__ int state_prim0_at(int idx) {
    return read_i32(params.state_prim0, params.state_prim0_stride, idx);
}

static __forceinline__ __device__ int state_prim1_at(int idx) {
    return read_i32(params.state_prim1, params.state_prim1_stride, idx);
}

static __forceinline__ __device__ float state_exterior_angle_at(int idx) {
    return read_f32(params.state_exterior_angle, params.state_exterior_angle_stride, idx);
}

static __forceinline__ __device__ float state_src_power_at(int idx) {
    return read_f32(params.state_src_power, params.state_src_power_stride, idx);
}

static __forceinline__ __device__ float3 state_src_at(int idx) {
    return state_vec(params.state_src_x,
                     params.state_src_y,
                     params.state_src_z,
                     params.state_src_stride,
                     idx);
}

static __forceinline__ __device__ float3 state_wi_at(int idx) {
    return optional_state_vec(params.state_wi_x,
                              params.state_wi_y,
                              params.state_wi_z,
                              params.state_wi_stride,
                              idx);
}

static __forceinline__ __device__ int recursive_state_edge_index_at(int idx) {
    return read_i32(params.recursive_state_edge_index, params.recursive_state_edge_index_stride, idx);
}

static __forceinline__ __device__ float3 recursive_state_edge_pos_at(int idx) {
    return recursive_state_vec(params.recursive_state_edge_pos_x,
                               params.recursive_state_edge_pos_y,
                               params.recursive_state_edge_pos_z,
                               params.recursive_state_edge_pos_stride,
                               idx);
}

static __forceinline__ __device__ float3 recursive_state_edge_dir_at(int idx) {
    return recursive_state_vec(params.recursive_state_edge_dir_x,
                               params.recursive_state_edge_dir_y,
                               params.recursive_state_edge_dir_z,
                               params.recursive_state_edge_dir_stride,
                               idx);
}

static __forceinline__ __device__ float recursive_state_edge_t_min_at(int idx) {
    return read_f32(params.recursive_state_edge_t_min, params.recursive_state_edge_t_min_stride, idx);
}

static __forceinline__ __device__ float recursive_state_edge_t_max_at(int idx) {
    return read_f32(params.recursive_state_edge_t_max, params.recursive_state_edge_t_max_stride, idx);
}

static __forceinline__ __device__ int recursive_state_prim0_at(int idx) {
    return read_i32(params.recursive_state_prim0, params.recursive_state_prim0_stride, idx);
}

static __forceinline__ __device__ int recursive_state_prim1_at(int idx) {
    return read_i32(params.recursive_state_prim1, params.recursive_state_prim1_stride, idx);
}

static __forceinline__ __device__ float recursive_state_exterior_angle_at(int idx) {
    return read_f32(params.recursive_state_exterior_angle, params.recursive_state_exterior_angle_stride, idx);
}

static __forceinline__ __device__ bool material_valid_at(int prim) {
    return params.material_valid == nullptr ||
           read_u8(params.material_valid, params.material_valid_stride, prim) != 0u;
}

static __forceinline__ __device__ float material_gain_at(int prim) {
    return read_f32(params.material_gain, params.material_gain_stride, prim);
}

static __forceinline__ __device__ float3 grid_cell_center(int cell) {
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

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ bool grid_cell_from_point(float3 point, int &cell) {
    float c0;
    float c1;
    if (params.grid_axis == 0) {
        c0 = point.y;
        c1 = point.z;
    } else if (params.grid_axis == 1) {
        c0 = point.x;
        c1 = point.z;
    } else {
        c0 = point.x;
        c1 = point.y;
    }
    if (c0 < params.grid_coord0_min || c0 >= params.grid_coord0_max ||
        c1 < params.grid_coord1_min || c1 >= params.grid_coord1_max) {
        return false;
    }
    const float u = (c0 - params.grid_coord0_min) /
                    fmaxf(params.grid_coord0_max - params.grid_coord0_min, kDfrEps);
    const float v = (c1 - params.grid_coord1_min) /
                    fmaxf(params.grid_coord1_max - params.grid_coord1_min, kDfrEps);
    const int i = min(max(static_cast<int>(u * params.grid_resolution0), 0),
                      params.grid_resolution0 - 1);
    const int j = min(max(static_cast<int>(v * params.grid_resolution1), 0),
                      params.grid_resolution1 - 1);
    cell = j * params.grid_resolution0 + i;
    return true;
}


namespace utd = witwin::channel::native_ext;

static __forceinline__ __device__ float first_order_diffraction_parameter(float3 source, float3 target, float3 edge_origin, float3 edge_dir);

static __forceinline__ __device__ utd::float3a to_utd(float3 value) {
    return utd::make_f3(value.x, value.y, value.z);
}

static __forceinline__ __device__ float shadow_decay_span_from_wedge_n(float wedge_n) {
    const float opening = fmaxf(2.f * kPi - wedge_n * kPi, 2.0e-3f);
    const float ratio = fminf(opening / kPi, 1.f);
    const float span = fmaxf((0.17f + 0.12f * ratio) * opening, 8.0e-3f);
    return fminf(span, 0.5f * opening);
}

static __forceinline__ __device__ utd::MaterialParams coherent_material_params() {
    utd::MaterialParams mat;
    mat.useFresnel = 0;
    mat.etaR = 0.f;
    mat.muR = 1.f;
    mat.sigma = 0.f;
    mat.gain = 1.f;
    mat.omega = params.omega;
    mat.txPolX = params.tx_pol_x;
    mat.txPolY = params.tx_pol_y;
    mat.txPolZ = params.tx_pol_z;
    return mat;
}

static __forceinline__ __device__ utd::PairInputs load_coherent_pair_inputs(int sIdx) {
    utd::PairInputs p;
    p.edgePos = utd::make_f3(params.utd_epx[sIdx], params.utd_epy[sIdx], params.utd_epz[sIdx]);
    p.edgeDir = utd::make_f3(params.utd_edx[sIdx], params.utd_edy[sIdx], params.utd_edz[sIdx]);
    p.n0 = utd::make_f3(params.utd_n0x[sIdx], params.utd_n0y[sIdx], params.utd_n0z[sIdx]);
    p.nn = utd::make_f3(params.utd_nnx[sIdx], params.utd_nny[sIdx], params.utd_nnz[sIdx]);
    p.wedgeN = params.utd_wn[sIdx];
    p.edgeLineMin = params.utd_elm[sIdx];
    p.edgeLineMax = params.utd_elx[sIdx];
    p.sourcePos = utd::make_f3(params.utd_spx[sIdx], params.utd_spy[sIdx], params.utd_spz[sIdx]);
    p.incidentField = utd::cplx(params.utd_ifr[sIdx], params.utd_ifi[sIdx]);
    p.incidentNormalDerivative = utd::cplx(params.utd_inr[sIdx], params.utd_ini[sIdx]);
    p.r0 = utd::cplx(params.utd_r0r[sIdx], params.utd_r0i[sIdx]);
    p.rn = utd::cplx(params.utd_rnr[sIdx], params.utd_rni[sIdx]);
    p.incidentVector = {utd::cplx(params.utd_vxr[sIdx], params.utd_vxi[sIdx]),
                        utd::cplx(params.utd_vyr[sIdx], params.utd_vyi[sIdx]),
                        utd::cplx(params.utd_vzr[sIdx], params.utd_vzi[sIdx])};
    p.incidentDerivativeVector = {utd::cplx(params.utd_dxr[sIdx], params.utd_dxi[sIdx]),
                                  utd::cplx(params.utd_dyr[sIdx], params.utd_dyi[sIdx]),
                                  utd::cplx(params.utd_dzr[sIdx], params.utd_dzi[sIdx])};
    p.incidentJones = {utd::cplx(params.utd_jur[sIdx], params.utd_jui[sIdx]),
                       utd::cplx(params.utd_jvr[sIdx], params.utd_jvi[sIdx])};
    p.incidentDerivativeJones = {utd::cplx(params.utd_djur[sIdx], params.utd_djui[sIdx]),
                                 utd::cplx(params.utd_djvr[sIdx], params.utd_djvi[sIdx])};
    p.incidentBasis = {utd::make_f3(params.utd_bux[sIdx], params.utd_buy[sIdx], params.utd_buz[sIdx]),
                       utd::make_f3(params.utd_bvx[sIdx], params.utd_bvy[sIdx], params.utd_bvz[sIdx]),
                       utd::make_f3(params.utd_bkx[sIdx], params.utd_bky[sIdx], params.utd_bkz[sIdx])};
    p.face0Operator = {utd::cplx(params.utd_f0m00r[sIdx], params.utd_f0m00i[sIdx]),
                       utd::cplx(params.utd_f0m01r[sIdx], params.utd_f0m01i[sIdx]),
                       utd::cplx(params.utd_f0m10r[sIdx], params.utd_f0m10i[sIdx]),
                       utd::cplx(params.utd_f0m11r[sIdx], params.utd_f0m11i[sIdx])};
    p.face1Operator = {utd::cplx(params.utd_f1m00r[sIdx], params.utd_f1m00i[sIdx]),
                       utd::cplx(params.utd_f1m01r[sIdx], params.utd_f1m01i[sIdx]),
                       utd::cplx(params.utd_f1m10r[sIdx], params.utd_f1m10i[sIdx]),
                       utd::cplx(params.utd_f1m11r[sIdx], params.utd_f1m11i[sIdx])};
    p.face0Material = {params.utd_f0er[sIdx], params.utd_f0mu[sIdx], params.utd_f0sg[sIdx],
                       params.utd_f0g[sIdx], params.utd_f0uf[sIdx], 1.f};
    p.face1Material = {params.utd_f1er[sIdx], params.utd_f1mu[sIdx], params.utd_f1sg[sIdx],
                       params.utd_f1g[sIdx], params.utd_f1uf[sIdx], 1.f};
    p.selectStationaryPoint = params.utd_select[sIdx];
    return p;
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool visible_segment_ignore_two_prims_impl(float3 start,
                                                                             float3 end,
                                                                             int ignore0,
                                                                             int ignore1) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit =
        trace_scene_impl<PrimaryOnly>(start + kDfrRayBias * dir, dir, fmaxf(dist - 2.f * kDfrRayBias, 0.f));
    if (hit.hit == 0u) {
        return true;
    }
    const int prim = global_primitive_id(hit);
    return prim == ignore0 || prim == ignore1;
}

template <bool PrimaryOnly>
static __forceinline__ __device__ bool coherent_visibility_and_support(utd::PairInputs state,
                                                                       float3 target_f3,
                                                                       bool selected_valid,
                                                                       bool selected_inside,
                                                                       float3 visibility_point_f3) {
    if (!selected_valid) {
        return false;
    }
    const utd::float3a target = to_utd(target_f3);
    const bool target_exterior = utd::wedge_exterior_mask(
        utd::f3_sub(target, state.edgePos), state.edgeDir, state.n0, state.nn);
    float phi, phiP, s, sP, sb;
    utd::compute_edge_geometry_3d(state.sourcePos, state.edgePos, state.edgeDir, state.n0,
                                  target, phi, phiP, s, sP, sb);
    const bool source_exterior = utd::wedge_exterior_mask(
        utd::f3_sub(state.sourcePos, state.edgePos), state.edgeDir, state.n0, state.nn);
    const bool base_valid = source_exterior && sP > utd::UTD_MIN_DISTANCE && s > utd::UTD_MIN_DISTANCE;
    if (!base_valid) {
        return false;
    }
    const float opening = fmaxf(2.f * kPi - state.wedgeN * kPi, 2.0e-3f);
    const float half_angle = 0.5f * opening;
    const bool wrap_boundary = phi >= 2.f * kPi - half_angle;
    const float shadow_boundary_distance = wrap_boundary ? 2.f * kPi - phi : phi - state.wedgeN * kPi;
    const bool selected_stationary = state.selectStationaryPoint > 0.5f;
    const float support_angle = selected_stationary ? half_angle : shadow_decay_span_from_wedge_n(state.wedgeN);
    bool shadow_completion = !target_exterior &&
                             shadow_boundary_distance >= 0.f &&
                             shadow_boundary_distance < support_angle;
    if (shadow_completion && point_inside_closed_mesh_robust_impl<PrimaryOnly>(target_f3)) {
        shadow_completion = false;
    }
    (void)selected_inside;
    (void)visibility_point_f3;
    return target_exterior || shadow_completion;
}

static __forceinline__ __device__ bool coherent_selected_visibility_point(utd::PairInputs original,
                                                                         float3 target,
                                                                         utd::PairInputs &selected,
                                                                         float3 &visibility_point) {
    selected = original;
    visibility_point = make_f3(original.edgePos.x, original.edgePos.y, original.edgePos.z);
    if (original.selectStationaryPoint <= 0.5f) {
        return true;
    }
    const float3 edge_dir = normalize3(make_f3(original.edgeDir.x, original.edgeDir.y, original.edgeDir.z));
    const float3 edge_pos = make_f3(original.edgePos.x, original.edgePos.y, original.edgePos.z);
    const float edge_length = original.edgeLineMax - original.edgeLineMin;
    const float3 edge_origin = edge_pos + original.edgeLineMin * edge_dir;
    const float parameter = first_order_diffraction_parameter(
        make_f3(original.sourcePos.x, original.sourcePos.y, original.sourcePos.z),
        target,
        edge_origin,
        edge_dir);
    if (!isfinite(parameter) || !(edge_length > kDfrEps)) {
        return false;
    }
    const float clamped_parameter = fminf(fmaxf(parameter, 0.f), edge_length);
    visibility_point = edge_origin + clamped_parameter * edge_dir;
    selected.edgePos = utd::make_f3(edge_origin.x + parameter * edge_dir.x,
                                    edge_origin.y + parameter * edge_dir.y,
                                    edge_origin.z + parameter * edge_dir.z);
    selected.edgeLineMin = -parameter;
    selected.edgeLineMax = edge_length - parameter;
    return parameter > 0.f && parameter < edge_length;
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_coherent_utd_lane(int state_idx, int cell) {
    utd::PairInputs original = load_coherent_pair_inputs(state_idx);
    const float3 target = grid_cell_center(cell);
    utd::PairInputs selected;
    float3 visibility_point;
    const bool selected_valid = coherent_selected_visibility_point(original, target, selected, visibility_point);
    if (params.prefilter_visibility != 0) {
        const int ignore0 = params.coherent_adjacent_face0 != nullptr ? params.coherent_adjacent_face0[state_idx] : -1;
        const int ignore1 = params.coherent_adjacent_face1 != nullptr ? params.coherent_adjacent_face1[state_idx] : -1;
        if (!visible_segment_ignore_two_prims_impl<PrimaryOnly>(visibility_point, target, ignore0, ignore1)) {
            if (params.collect_debug_counts != 0 && params.out_visibility_reject_count != nullptr) {
                atomicAdd(params.out_visibility_reject_count + cell, 1);
            }
            return;
        }
    }
    const bool selected_inside = selected.edgeLineMin < 0.f && selected.edgeLineMax > 0.f;
    if (!coherent_visibility_and_support<PrimaryOnly>(selected, target, selected_valid, selected_inside, visibility_point)) {
        if (params.collect_debug_counts != 0 && params.out_utd_reject_count != nullptr) {
            atomicAdd(params.out_utd_reject_count + cell, 1);
        }
        return;
    }
    const utd::PairOutputs out = utd::compute_pair_contribution(
        original, to_utd(target), params.k, coherent_material_params());
    const float norm = utd::cplx_abs_sqr(out.vectorField.x) +
                       utd::cplx_abs_sqr(out.vectorField.y) +
                       utd::cplx_abs_sqr(out.vectorField.z);
    if (!(norm > 0.f) || !isfinite(norm)) {
        if (params.collect_debug_counts != 0 && params.out_utd_reject_count != nullptr) {
            atomicAdd(params.out_utd_reject_count + cell, 1);
        }
        return;
    }
    const int owner = params.coherent_owner_code != nullptr ? params.coherent_owner_code[state_idx] : 0;
    if (params.coherent_stage_key != nullptr && params.coherent_stage_value != nullptr) {
        const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
        const int key = owner == utd::OWNERSHIP_MIXED ? grid_cell_count + cell : cell;
        const int slot = cell * params.state_count + state_idx;
        DfrCoherentStagedValue value;
        value.a = make_float4(out.vectorField.x.re,
                              out.vectorField.x.im,
                              out.vectorField.y.re,
                              out.vectorField.y.im);
        value.b = make_float4(out.vectorField.z.re,
                              out.vectorField.z.im,
                              1.0f,
                              0.0f);
        params.coherent_stage_key[slot] = key;
        params.coherent_stage_value[slot] = value;
        return;
    }
    if (owner == utd::OWNERSHIP_MIXED) {
        const WarpCellGroup cell_group = warp_cell_group(cell);
        atomic_add_same_cell(params.out_multi_field_x_re, cell, out.vectorField.x.re, cell_group);
        atomic_add_same_cell(params.out_multi_field_x_im, cell, out.vectorField.x.im, cell_group);
        atomic_add_same_cell(params.out_multi_field_y_re, cell, out.vectorField.y.re, cell_group);
        atomic_add_same_cell(params.out_multi_field_y_im, cell, out.vectorField.y.im, cell_group);
        atomic_add_same_cell(params.out_multi_field_z_re, cell, out.vectorField.z.re, cell_group);
        atomic_add_same_cell(params.out_multi_field_z_im, cell, out.vectorField.z.im, cell_group);
        if (params.out_multi_count != nullptr) atomic_add_same_cell(params.out_multi_count, cell, 1, cell_group);
    } else {
        const WarpCellGroup cell_group = warp_cell_group(cell);
        atomic_add_same_cell(params.out_direct_field_x_re, cell, out.vectorField.x.re, cell_group);
        atomic_add_same_cell(params.out_direct_field_x_im, cell, out.vectorField.x.im, cell_group);
        atomic_add_same_cell(params.out_direct_field_y_re, cell, out.vectorField.y.re, cell_group);
        atomic_add_same_cell(params.out_direct_field_y_im, cell, out.vectorField.y.im, cell_group);
        atomic_add_same_cell(params.out_direct_field_z_re, cell, out.vectorField.z.re, cell_group);
        atomic_add_same_cell(params.out_direct_field_z_im, cell, out.vectorField.z.im, cell_group);
        if (params.out_direct_count != nullptr) atomic_add_same_cell(params.out_direct_count, cell, 1, cell_group);
    }
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

static __forceinline__ __device__ bool keller_grid_hit_from_incident(float3 incident_vec,
                                                                     unsigned int lane,
                                                                     unsigned int stream,
                                                                     float3 edge_point,
                                                                     float3 edge_dir,
                                                                     float3 &target,
                                                                     int &cell) {
    const float3 incident = normalize3(incident_vec);
    const float axial = fminf(fmaxf(dot3(incident, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, incident);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    float s;
    float c;
    sincosf(2.f * kPi * uniform01(lane, stream, static_cast<unsigned int>(params.seed)), &s, &c);
    const float3 ko = normalize3(axial * edge_dir + radial * (c * basis0 + s * basis1));
    const float denom = component(ko, params.grid_axis);
    if (fabsf(denom) <= kDfrEps) {
        return false;
    }
    const float t = (params.grid_position - component(edge_point, params.grid_axis)) / denom;
    if (!(t > kDfrRayBias) || !isfinite(t)) {
        return false;
    }
    target = edge_point + t * ko;
    return grid_cell_from_point(target, cell);
}

static __forceinline__ __device__ bool keller_grid_hit(int state_idx,
                                                       unsigned int lane,
                                                       float3 edge_point,
                                                       float3 edge_dir,
                                                       float3 &target,
                                                       int &cell) {
    const float3 incident = state_wi_at(state_idx);
    return keller_grid_hit_from_incident(incident, lane, 1u, edge_point, edge_dir, target, cell);
}

static __forceinline__ __device__ int material_index_for_faces(int face0_prim,
                                                               int face1_prim) {
    if (params.material_gain == nullptr || params.material_count <= 0) {
        return -1;
    }
    int prim = face0_prim;
    if (prim < 0 || prim >= params.material_count || !material_valid_at(prim)) {
        prim = face1_prim;
    }
    if (prim < 0 || prim >= params.material_count || !material_valid_at(prim)) {
        return -1;
    }
    return prim;
}

static __forceinline__ __device__ float material_gain_for_faces(int face0_prim,
                                                                int face1_prim) {
    const int prim = material_index_for_faces(face0_prim, face1_prim);
    if (prim < 0) {
        return 1.f;
    }
    return fmaxf(material_gain_at(prim), 0.f);
}

static __forceinline__ __device__ float material_gain_for_state(int state_idx) {
    return material_gain_for_faces(state_prim0_at(state_idx),
                                   state_prim1_at(state_idx));
}

static __forceinline__ __device__ float material_gain_for_prim(int prim) {
    if (params.material_gain == nullptr ||
        prim < 0 ||
        prim >= params.material_count ||
        !material_valid_at(prim)) {
        return 1.f;
    }
    return fmaxf(material_gain_at(prim), 0.f);
}

static __forceinline__ __device__ bool suffix_candidate_valid(int prim) {
    return prim >= 0 &&
           prim < params.n_triangles &&
           prim < params.material_count &&
           material_valid_at(prim);
}

static __forceinline__ __device__ bool select_local_suffix_candidate(int face0_prim,
                                                                     int face1_prim,
                                                                     unsigned int lane,
                                                                     unsigned int stream,
                                                                     int &prim,
                                                                     float &candidate_count) {
    const bool face0_valid = suffix_candidate_valid(face0_prim);
    const bool face1_valid =
        suffix_candidate_valid(face1_prim) && face1_prim != face0_prim;
    const int count = (face0_valid ? 1 : 0) + (face1_valid ? 1 : 0);
    if (count <= 0) {
        return false;
    }
    const unsigned int candidate_hash = hash_u32(
        lane ^ (stream * 0x9e3779b9u) ^ static_cast<unsigned int>(params.seed));
    const int slot = static_cast<int>(candidate_hash % static_cast<unsigned int>(count));
    if (face0_valid && slot == 0) {
        prim = face0_prim;
    } else {
        prim = face1_prim;
    }
    candidate_count = static_cast<float>(count);
    return true;
}

static __forceinline__ __device__ bool load_triangle(int prim,
                                                     float3 &p0,
                                                     float3 &e1,
                                                     float3 &e2,
                                                     float3 &normal) {
    if (prim < 0 ||
        prim >= params.n_triangles ||
        params.tri_p0_x == nullptr ||
        params.tri_e1_x == nullptr ||
        params.tri_e2_x == nullptr ||
        params.tri_fn_x == nullptr) {
        return false;
    }
    p0 = make_f3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
    e1 = make_f3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
    e2 = make_f3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
    normal = make_f3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]);
    if (dot3(normal, normal) <= 1e-12f) {
        normal = cross3(e1, e2);
    }
    normal = normalize3(normal);
    return dot3(normal, normal) > 0.f;
}

static __forceinline__ __device__ bool intersect_reflection_triangle(float3 image_source,
                                                                     float3 target,
                                                                     int prim,
                                                                     float3 &reflection_point,
                                                                     float3 &normal) {
    float3 p0;
    float3 e1;
    float3 e2;
    if (!load_triangle(prim, p0, e1, e2, normal)) {
        return false;
    }
    const float3 delta = target - image_source;
    const float dist = norm3(delta);
    if (!(dist > kDfrRayBias) || !isfinite(dist)) {
        return false;
    }
    const float3 dir = (1.f / dist) * delta;
    const float3 h = cross3(dir, e2);
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
    const float v = f * dot3(dir, q);
    if (v < -1e-5f || u + v > 1.f + 1e-5f) {
        return false;
    }
    const float t = f * dot3(e2, q);
    if (!(t > kDfrRayBias) || !(t < dist - kDfrRayBias) || !isfinite(t)) {
        return false;
    }
    reflection_point = image_source + t * dir;
    return true;
}

static __forceinline__ __device__ bool suffix_reflection_connection(float3 diff_point,
                                                                    float3 target,
                                                                    int face0_prim,
                                                                    int face1_prim,
                                                                    unsigned int lane,
                                                                    unsigned int stream,
                                                                    float3 &reflection_point,
                                                                    int &prim,
                                                                    float &reflection_gain,
                                                                    float &suffix_fspl,
                                                                    float &candidate_count) {
    if (!select_local_suffix_candidate(face0_prim,
                                       face1_prim,
                                       lane,
                                       stream,
                                       prim,
                                       candidate_count)) {
        return false;
    }
    float3 p0;
    float3 e1;
    float3 e2;
    float3 normal;
    if (!load_triangle(prim, p0, e1, e2, normal)) {
        return false;
    }
    const float plane_distance = dot3(diff_point - p0, normal);
    const float3 image_source = diff_point - 2.f * plane_distance * normal;
    if (!intersect_reflection_triangle(image_source, target, prim, reflection_point, normal)) {
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

    const float gain = material_gain_for_prim(prim);
    reflection_gain = gain * gain;
    suffix_fspl = (params.wavelength * (1.f / (4.f * kPi))) *
                  (params.wavelength * (1.f / (4.f * kPi))) /
                  fmaxf(outgoing_dist * outgoing_dist, kDfrEps);
    if (!(isfinite(reflection_gain) && isfinite(suffix_fspl))) {
        return false;
    }
    return true;
}

static __forceinline__ __device__ float3 rotate_around_axis(float3 value,
                                                            float3 axis,
                                                            float angle) {
    const float c = cosf(angle);
    const float s = sinf(angle);
    return c * value + s * cross3(axis, value) +
           (1.f - c) * dot3(axis, value) * axis;
}

static __forceinline__ __device__ float first_order_diffraction_parameter(
    float3 source,
    float3 target,
    float3 edge_origin,
    float3 edge_dir) {
    const float3 zeta = normalize3(edge_dir);
    const float3 target_offset = target - edge_origin;
    const float3 source_offset = source - edge_origin;
    const float3 target_projection = dot3(target_offset, zeta) * zeta;
    const float3 source_projection = dot3(source_offset, zeta) * zeta;
    const float3 target_radial = target_offset - target_projection;
    const float3 source_radial = source_offset - source_projection;
    const float target_radial_norm = norm3(target_radial);
    const float source_radial_norm = norm3(source_radial);
    const float3 v1 = (1.f / fmaxf(target_radial_norm, kDfrEps)) * target_radial;
    const float3 v2 = (1.f / fmaxf(source_radial_norm, kDfrEps)) * source_radial;
    const float theta = kPi - acosf(fminf(fmaxf(dot3(v1, v2), -1.f), 1.f));

    const float3 raw_rotation_axis = cross3(source_radial, target_radial);
    const float rotation_axis_norm = norm3(raw_rotation_axis);
    const float3 rotation_axis = rotation_axis_norm > kDfrEps
        ? (1.f / rotation_axis_norm) * raw_rotation_axis
        : zeta;

    const float3 coplanar_target =
        rotate_around_axis(target_offset, rotation_axis, theta);
    const float3 source_to_target = coplanar_target - source_offset;
    const float source_to_target_norm = norm3(source_to_target);
    const float3 u0 =
        (1.f / fmaxf(source_to_target_norm, kDfrEps)) * source_to_target;
    const float3 u1 = cross3(source_offset, u0);
    const float3 u2 = cross3(zeta, u0);
    const float u2_norm = norm3(u2);
    const float sign = dot3(u1, u2) < 0.f ? -1.f : 1.f;
    return sign * norm3(u1) / fmaxf(u2_norm, kDfrEps);
}

static __forceinline__ __device__ float diffraction_weight(int state_idx,
                                                           float3 edge_point,
                                                           float3 target,
                                                           int sample_count) {
    const float3 source = state_src_at(state_idx);
    const float source_distance = fmaxf(norm3(edge_point - source), kDfrEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kDfrEps);
    const float edge_length = fmaxf(
        state_edge_t_max_at(state_idx) - state_edge_t_min_at(state_idx),
        0.f);
    const float exterior_angle =
        fmaxf(state_exterior_angle_at(state_idx), 0.25f * kPi);
    const float wedge_scale = fminf(exterior_angle / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_state(state_idx);
    const float sample_norm = 1.f / fmaxf(static_cast<float>(sample_count), 1.f);
    return state_src_power_at(state_idx) *
           material_gain *
           edge_length *
           params.grid_cell_area *
           wedge_scale *
           sample_norm /
           (source_distance * source_distance * target_distance * target_distance);
}

static __forceinline__ __device__ float chain_event_weight(float src_power,
                                                           int face0_prim,
                                                           int face1_prim,
                                                           float edge_t_min,
                                                           float edge_t_max,
                                                           float exterior_angle,
                                                           float3 source,
                                                           float3 edge_point,
                                                           float3 target) {
    const float source_distance = fmaxf(norm3(edge_point - source), kDfrEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kDfrEps);
    const float edge_length = fmaxf(edge_t_max - edge_t_min, 0.f);
    const float wedge_scale = fminf(fmaxf(exterior_angle, 0.25f * kPi) / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_faces(face0_prim, face1_prim);
    return src_power *
           material_gain *
           edge_length *
           wedge_scale /
           (source_distance * source_distance * target_distance * target_distance);
}

} // namespace

extern "C" {
__constant__ DfrAccumParams params;
}

extern "C" __global__ void __closesthit__diffraction_accumulation() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__diffraction_accumulation() {
    optixSetPayload_0(0u);
}

template <bool PrimaryOnly,
          bool IncludeCoherent,
          bool IncludeDirect,
          bool IncludeKeller,
          bool IncludeSuffix>
static __forceinline__ __device__ void run_diffraction_order1_accumulation_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0) {
        return;
    }

    const int direct_limit = IncludeDirect && (params.strategy_mask & RAYDN_DFR_DIRECT) != 0
        ? params.direct_samples
        : 0;
    const int keller_limit = IncludeKeller && (params.strategy_mask & RAYDN_DFR_KELLER) != 0
        ? params.keller_samples
        : 0;
    const int suffix_limit = IncludeSuffix && (params.strategy_mask & RAYDN_DFR_SUFFIX_REFL) != 0
        ? params.suffix_samples
        : 0;
    const int total_samples = direct_limit + keller_limit + suffix_limit;
    if (total_samples <= 0) {
        return;
    }
    const bool is_direct = IncludeDirect && static_cast<int>(lane) < direct_limit;
    const bool is_keller = IncludeKeller &&
        !is_direct && static_cast<int>(lane) < direct_limit + keller_limit;
    const bool is_suffix = IncludeSuffix &&
        static_cast<int>(lane) >= direct_limit + keller_limit &&
        static_cast<int>(lane) < total_samples;
    if (!is_direct && !is_keller && !is_suffix) {
        return;
    }

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>((lane / static_cast<unsigned int>(params.state_count)) %
                                static_cast<unsigned int>(grid_cell_count));
    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float edge_t = edge_t_min + edge_u * (edge_t_max - edge_t_min);
    if constexpr (IncludeCoherent) {
        if (params.coherent_utd_slot_count >= 84 && params.utd_epx != nullptr) {
            run_coherent_utd_lane<PrimaryOnly>(state_idx, cell);
            return;
        }
    }

    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 source = state_src_at(state_idx);
    float3 target = grid_cell_center(cell);
    if constexpr (IncludeKeller) {
        if (is_keller && !keller_grid_hit(state_idx, lane, edge_point, edge_dir, target, cell)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }

    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    float3 connection_target = target;
    if constexpr (IncludeSuffix) {
        if (is_suffix) {
            if (!suffix_reflection_connection(edge_point,
                                              target,
                                              state_prim0_at(state_idx),
                                              state_prim1_at(state_idx),
                                              lane,
                                              17u,
                                              connection_target,
                                              suffix_prim,
                                              suffix_reflection_gain,
                                              suffix_fspl,
                                              suffix_candidate_count)) {
                if (params.collect_debug_counts != 0) {
                    atomicAdd(params.out_utd_rejects, 1);
                }
                return;
            }
        }
    }

    const bool source_visible = visible_segment_impl<PrimaryOnly>(source, edge_point);
    bool target_visible = true;
    if constexpr (IncludeDirect || IncludeKeller) {
        if (is_direct || is_keller) {
            target_visible = visible_segment_impl<PrimaryOnly>(edge_point, target);
        }
    }
    if constexpr (IncludeSuffix) {
        if (is_suffix) {
            target_visible =
                visible_segment_ignore_prim_impl<PrimaryOnly>(edge_point, connection_target, suffix_prim) &&
                visible_segment_ignore_prim_impl<PrimaryOnly>(connection_target, target, suffix_prim);
        }
    }
    if (!source_visible || !target_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }

    const int strategy_sample_count =
        is_direct ? direct_limit : (is_keller ? keller_limit : suffix_limit);
    float contribution =
        diffraction_weight(state_idx, edge_point, connection_target, strategy_sample_count);
    if constexpr (IncludeSuffix) {
        if (is_suffix) {
            contribution *= suffix_reflection_gain *
                            suffix_fspl *
                            fmaxf(suffix_candidate_count, 1.f);
        }
    }
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = state_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(state_prim0_at(state_idx),
                                         state_prim1_at(state_idx));
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = edge_u;
        }
    }

    const float field_x_re = sqrtf(fmaxf(contribution, 0.f));
    if (params.stage_cell != nullptr && params.stage_value != nullptr) {
        params.stage_cell[lane] = cell;
        params.stage_value[lane] = make_float4(
            contribution,
            field_x_re,
            is_direct ? 1.0f : 0.0f,
            is_keller ? 1.0f : 0.0f);
        return;
    }

    const WarpCellGroup cell_group = warp_cell_group(cell);
    atomic_add_same_cell(params.out_power, cell, contribution, cell_group);
    atomic_add_same_cell(params.out_field_x_re, cell, field_x_re, cell_group);
    if (is_direct) {
        atomic_add_warp(params.out_direct_count, 1);
    } else {
        if constexpr (IncludeKeller) {
            if (is_keller) {
                atomic_add_warp(params.out_keller_count, 1);
            }
        }
        if constexpr (IncludeSuffix) {
            if (is_suffix) {
                atomic_add_warp(params.out_suffix_count, 1);
            }
        }
    }
    if (params.collect_edge_use != 0) {
        atomic_add_warp(params.out_edge_uses, 1);
    }
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation() {
    run_diffraction_order1_accumulation_raygen<false, false, true, true, true>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_primary() {
    run_diffraction_order1_accumulation_raygen<true, false, true, true, true>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_no_suffix() {
    run_diffraction_order1_accumulation_raygen<false, false, true, true, false>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_no_suffix_primary() {
    run_diffraction_order1_accumulation_raygen<true, false, true, true, false>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_suffix() {
    run_diffraction_order1_accumulation_raygen<false, false, false, false, true>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_suffix_primary() {
    run_diffraction_order1_accumulation_raygen<true, false, false, false, true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_order1_source_visibility_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.temp_visibility == nullptr) {
        return;
    }
    params.temp_visibility[lane] = 0u;

    const int direct_limit =
        (params.strategy_mask & RAYDN_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYDN_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYDN_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int total_samples = direct_limit + keller_limit + suffix_limit;
    if (total_samples <= 0 || static_cast<int>(lane) >= total_samples) {
        return;
    }

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        return;
    }

    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float edge_t = edge_t_min + edge_u * (edge_t_max - edge_t_min);
    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 source = state_src_at(state_idx);
    params.temp_visibility[lane] =
        visible_segment_impl<PrimaryOnly>(source, edge_point) ? 1u : 0u;
}

extern "C" __global__ void __raygen__diffraction_order1_source_visibility_primary() {
    run_diffraction_order1_source_visibility_raygen<true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_order1_no_suffix_target_accumulation_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYDN_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYDN_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int total_samples = direct_limit + keller_limit;
    if (total_samples <= 0 || static_cast<int>(lane) >= total_samples) {
        return;
    }
    if (params.temp_visibility != nullptr && params.temp_visibility[lane] == 0u) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }
    const bool is_direct = static_cast<int>(lane) < direct_limit;
    const bool is_keller =
        !is_direct && static_cast<int>(lane) < direct_limit + keller_limit;

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>((lane / static_cast<unsigned int>(params.state_count)) %
                                static_cast<unsigned int>(grid_cell_count));
    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float edge_t = edge_t_min + edge_u * (edge_t_max - edge_t_min);
    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    float3 target = grid_cell_center(cell);
    if (is_keller && !keller_grid_hit(state_idx, lane, edge_point, edge_dir, target, cell)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (!visible_segment_impl<PrimaryOnly>(edge_point, target)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }

    const int strategy_sample_count = is_direct ? direct_limit : keller_limit;
    const float contribution =
        diffraction_weight(state_idx, edge_point, target, strategy_sample_count);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }
    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = state_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(state_prim0_at(state_idx),
                                         state_prim1_at(state_idx));
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = edge_u;
        }
    }

    const float field_x_re = sqrtf(fmaxf(contribution, 0.f));
    if (params.stage_cell != nullptr && params.stage_value != nullptr) {
        params.stage_cell[lane] = cell;
        params.stage_value[lane] = make_float4(
            contribution,
            field_x_re,
            is_direct ? 1.0f : 0.0f,
            is_keller ? 1.0f : 0.0f);
        return;
    }

    const WarpCellGroup cell_group = warp_cell_group(cell);
    atomic_add_same_cell(params.out_power, cell, contribution, cell_group);
    atomic_add_same_cell(params.out_field_x_re, cell, field_x_re, cell_group);
    if (is_direct) {
        atomic_add_warp(params.out_direct_count, 1);
    } else {
        atomic_add_warp(params.out_keller_count, 1);
    }
    if (params.collect_edge_use != 0) {
        atomic_add_warp(params.out_edge_uses, 1);
    }
}

extern "C" __global__ void __raygen__diffraction_order1_no_suffix_target_accumulation_primary() {
    run_diffraction_order1_no_suffix_target_accumulation_raygen<true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_order1_suffix_first_visibility_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.temp_visibility == nullptr) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYDN_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYDN_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYDN_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int suffix_begin = direct_limit + keller_limit;
    const int total_samples = suffix_begin + suffix_limit;
    if (suffix_limit <= 0 ||
        static_cast<int>(lane) < suffix_begin ||
        static_cast<int>(lane) >= total_samples) {
        return;
    }
    if (params.temp_visibility[lane] == 0u) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        params.temp_visibility[lane] = 0u;
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>((lane / static_cast<unsigned int>(params.state_count)) %
                                static_cast<unsigned int>(grid_cell_count));
    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float edge_t = edge_t_min + edge_u * (edge_t_max - edge_t_min);
    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 target = grid_cell_center(cell);

    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    float3 connection_target = target;
    if (!suffix_reflection_connection(edge_point,
                                      target,
                                      state_prim0_at(state_idx),
                                      state_prim1_at(state_idx),
                                      lane,
                                      17u,
                                      connection_target,
                                      suffix_prim,
                                      suffix_reflection_gain,
                                      suffix_fspl,
                                      suffix_candidate_count)) {
        params.temp_visibility[lane] = 0u;
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    const bool visible =
        visible_segment_ignore_prim_impl<PrimaryOnly>(edge_point, connection_target, suffix_prim);
    params.temp_visibility[lane] = visible ? 1u : 0u;
    if (!visible && params.collect_debug_counts != 0) {
        atomicAdd(params.out_vis_rejects, 1);
    }
    (void)suffix_reflection_gain;
    (void)suffix_fspl;
    (void)suffix_candidate_count;
}

extern "C" __global__ void __raygen__diffraction_order1_suffix_first_visibility_primary() {
    run_diffraction_order1_suffix_first_visibility_raygen<true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_order1_suffix_target_accumulation_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYDN_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYDN_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYDN_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int suffix_begin = direct_limit + keller_limit;
    const int total_samples = suffix_begin + suffix_limit;
    if (suffix_limit <= 0 ||
        static_cast<int>(lane) < suffix_begin ||
        static_cast<int>(lane) >= total_samples) {
        return;
    }
    if (params.temp_visibility != nullptr && params.temp_visibility[lane] == 0u) {
        return;
    }

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>((lane / static_cast<unsigned int>(params.state_count)) %
                                static_cast<unsigned int>(grid_cell_count));
    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float edge_t = edge_t_min + edge_u * (edge_t_max - edge_t_min);
    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 target = grid_cell_center(cell);

    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    float3 connection_target = target;
    if (!suffix_reflection_connection(edge_point,
                                      target,
                                      state_prim0_at(state_idx),
                                      state_prim1_at(state_idx),
                                      lane,
                                      17u,
                                      connection_target,
                                      suffix_prim,
                                      suffix_reflection_gain,
                                      suffix_fspl,
                                      suffix_candidate_count)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (!visible_segment_ignore_prim_impl<PrimaryOnly>(connection_target, target, suffix_prim)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }

    float contribution =
        diffraction_weight(state_idx, edge_point, connection_target, suffix_limit);
    contribution *= suffix_reflection_gain *
                    suffix_fspl *
                    fmaxf(suffix_candidate_count, 1.f);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = state_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(state_prim0_at(state_idx),
                                         state_prim1_at(state_idx));
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = edge_u;
        }
    }

    const WarpCellGroup cell_group = warp_cell_group(cell);
    atomic_add_same_cell(params.out_power, cell, contribution, cell_group);
    atomic_add_same_cell(params.out_field_x_re, cell, sqrtf(fmaxf(contribution, 0.f)), cell_group);
    atomic_add_warp(params.out_suffix_count, 1);
    if (params.collect_edge_use != 0) {
        atomic_add_warp(params.out_edge_uses, 1);
    }
}

extern "C" __global__ void __raygen__diffraction_order1_suffix_target_accumulation_primary() {
    run_diffraction_order1_suffix_target_accumulation_raygen<true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_order1_coherent_accumulation_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0) {
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    const int cell = static_cast<int>(lane / static_cast<unsigned int>(params.state_count));
    if (cell < 0 || cell >= grid_cell_count) {
        return;
    }
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, state_idx)) {
        return;
    }
    if (params.coherent_utd_slot_count >= 84 && params.utd_epx != nullptr) {
        run_coherent_utd_lane<PrimaryOnly>(state_idx, cell);
        return;
    }

    const float3 edge_pos = state_edge_pos_at(state_idx);
    const float3 edge_dir = normalize3(state_edge_dir_at(state_idx));
    const float edge_t_min = state_edge_t_min_at(state_idx);
    const float edge_t_max = state_edge_t_max_at(state_idx);
    const float3 source = state_src_at(state_idx);
    const float3 target = grid_cell_center(cell);
    float edge_t = 0.5f * (edge_t_min + edge_t_max);
    float visibility_edge_t = edge_t;
    if (params.select_diffraction_point != 0) {
        const float edge_length = edge_t_max - edge_t_min;
        const float3 edge_origin = edge_pos + edge_t_min * edge_dir;
        const float parameter =
            first_order_diffraction_parameter(source, target, edge_origin, edge_dir);
        if (!isfinite(parameter) || !(edge_length > kDfrEps)) {
            if (params.collect_debug_counts != 0 &&
                params.out_utd_reject_count != nullptr) {
                atomicAdd(params.out_utd_reject_count + cell, 1);
            }
            return;
        }
        edge_t = edge_t_min + parameter;
        visibility_edge_t = edge_t_min + fminf(fmaxf(parameter, 0.f), edge_length);
    }
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 visibility_edge_point = edge_pos + visibility_edge_t * edge_dir;

    if (params.prefilter_visibility != 0) {
        const bool source_visible = visible_segment_impl<PrimaryOnly>(source, visibility_edge_point);
        const bool target_visible = visible_segment_impl<PrimaryOnly>(visibility_edge_point, target);
        if (!source_visible || !target_visible) {
            if (params.collect_debug_counts != 0 &&
                params.out_visibility_reject_count != nullptr) {
                atomicAdd(params.out_visibility_reject_count + cell, 1);
            }
            return;
        }
    }

    const float contribution =
        diffraction_weight(state_idx, edge_point, target, 1);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0 &&
            params.out_utd_reject_count != nullptr) {
            atomicAdd(params.out_utd_reject_count + cell, 1);
        }
        return;
    }

    const float source_distance = fmaxf(norm3(edge_point - source), kDfrEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kDfrEps);
    const float phase = -params.k * (source_distance + target_distance);
    const float amplitude = sqrtf(fmaxf(contribution, 0.f));
    const float field_re = amplitude * cosf(phase);
    const float field_im = amplitude * sinf(phase);
    const bool is_multi = params.state_prefix_depth != nullptr &&
                          params.state_prefix_depth[state_idx] > 0;

    if (params.coherent_stage_key != nullptr && params.coherent_stage_value != nullptr) {
        const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
        const int key = is_multi ? grid_cell_count + cell : cell;
        const int slot = cell * params.state_count + state_idx;
        DfrCoherentStagedValue value;
        value.a = make_float4(field_re, field_im, 0.0f, 0.0f);
        value.b = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        params.coherent_stage_key[slot] = key;
        params.coherent_stage_value[slot] = value;
        return;
    }

    if (is_multi) {
        const WarpCellGroup cell_group = warp_cell_group(cell);
        if (params.out_multi_field_x_re != nullptr) {
            atomic_add_same_cell(params.out_multi_field_x_re, cell, field_re, cell_group);
            atomic_add_same_cell(params.out_multi_field_x_im, cell, field_im, cell_group);
        }
        if (params.out_multi_count != nullptr) {
            atomic_add_same_cell(params.out_multi_count, cell, 1, cell_group);
        }
    } else {
        const WarpCellGroup cell_group = warp_cell_group(cell);
        if (params.out_direct_field_x_re != nullptr) {
            atomic_add_same_cell(params.out_direct_field_x_re, cell, field_re, cell_group);
            atomic_add_same_cell(params.out_direct_field_x_im, cell, field_im, cell_group);
        }
        if (params.out_direct_count != nullptr) {
            atomic_add_same_cell(params.out_direct_count, cell, 1, cell_group);
        }
    }
}

extern "C" __global__ void __raygen__diffraction_order1_coherent_accumulation() {
    run_diffraction_order1_coherent_accumulation_raygen<false>();
}

extern "C" __global__ void __raygen__diffraction_order1_coherent_accumulation_primary() {
    run_diffraction_order1_coherent_accumulation_raygen<true>();
}

template <bool PrimaryOnly>
static __forceinline__ __device__ void run_diffraction_chain_accumulation_raygen() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.recursive_state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0 ||
        (params.max_order != 2 && params.max_order != 3) ||
        (params.strategy_mask &
         (RAYDN_DFR_DIRECT | RAYDN_DFR_KELLER | RAYDN_DFR_SUFFIX_REFL)) == 0) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYDN_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYDN_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYDN_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int total_samples = direct_limit + keller_limit + suffix_limit;
    if (total_samples <= 0 || static_cast<int>(lane) >= total_samples) {
        return;
    }
    const bool is_direct = static_cast<int>(lane) < direct_limit;
    const bool is_keller =
        !is_direct && static_cast<int>(lane) < direct_limit + keller_limit;
    const bool is_suffix =
        static_cast<int>(lane) >= direct_limit + keller_limit &&
        static_cast<int>(lane) < total_samples;

    const int first_idx = static_cast<int>(
        lane % static_cast<unsigned int>(params.state_count));
    const unsigned int second_hash = hash_u32(
        lane ^ (static_cast<unsigned int>(params.seed) * 0x9e3779b9u) ^ 0x51ed270bu);
    const int second_idx = static_cast<int>(
        second_hash % static_cast<unsigned int>(params.recursive_state_count));
    int third_idx = -1;
    if (!active_for_state(params.active_mask, params.active_width, params.active_stride, first_idx)) {
        return;
    }
    if (!active_for_state(
            params.recursive_active_mask,
            params.recursive_active_width,
            params.recursive_active_stride,
            second_idx)) {
        return;
    }
    if (params.max_order == 3) {
        const unsigned int third_hash = hash_u32(
            lane ^ (static_cast<unsigned int>(params.seed) * 0x85ebca6bu) ^ 0xc2b2ae35u);
        third_idx = static_cast<int>(
            third_hash % static_cast<unsigned int>(params.recursive_state_count));
        if (!active_for_state(
                params.recursive_active_mask,
                params.recursive_active_width,
                params.recursive_active_stride,
                third_idx)) {
            return;
        }
    }

    const int first_edge_index = state_edge_index_at(first_idx);
    const int second_edge_index = recursive_state_edge_index_at(second_idx);
    const int third_edge_index =
        params.max_order == 3 ? recursive_state_edge_index_at(third_idx) : -1;
    if (first_edge_index == second_edge_index ||
        (params.max_order == 3 &&
         (first_edge_index == third_edge_index || second_edge_index == third_edge_index))) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>(
        (lane / static_cast<unsigned int>(params.state_count)) %
        static_cast<unsigned int>(grid_cell_count));
    const float first_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float second_u = uniform01(lane, 2u, static_cast<unsigned int>(params.seed));

    const float3 first_edge_pos = state_edge_pos_at(first_idx);
    const float3 first_edge_dir = normalize3(state_edge_dir_at(first_idx));
    const float first_t_min = state_edge_t_min_at(first_idx);
    const float first_t_max = state_edge_t_max_at(first_idx);
    const float first_t = first_t_min + first_u * (first_t_max - first_t_min);
    const float3 first_point = first_edge_pos + first_t * first_edge_dir;

    const float3 second_edge_pos = recursive_state_edge_pos_at(second_idx);
    const float3 second_edge_dir = normalize3(recursive_state_edge_dir_at(second_idx));
    const float second_t_min = recursive_state_edge_t_min_at(second_idx);
    const float second_t_max = recursive_state_edge_t_max_at(second_idx);
    const float second_t = second_t_min + second_u * (second_t_max - second_t_min);
    const float3 second_point = second_edge_pos + second_t * second_edge_dir;

    const float3 source = state_src_at(first_idx);
    const float3 target = grid_cell_center(cell);
    float3 third_point = second_point;
    float3 third_edge_dir = second_edge_dir;
    if (params.max_order == 3) {
        const float third_u = uniform01(lane, 4u, static_cast<unsigned int>(params.seed));
        const float3 third_edge_pos = recursive_state_edge_pos_at(third_idx);
        third_edge_dir = normalize3(recursive_state_edge_dir_at(third_idx));
        const float third_t_min = recursive_state_edge_t_min_at(third_idx);
        const float third_t_max = recursive_state_edge_t_max_at(third_idx);
        const float third_t = third_t_min + third_u * (third_t_max - third_t_min);
        third_point = third_edge_pos + third_t * third_edge_dir;
    }
    const float3 terminal_point = params.max_order == 3 ? third_point : second_point;
    const float3 terminal_edge_dir = params.max_order == 3 ? third_edge_dir : second_edge_dir;
    float3 final_target = target;
    if (is_keller) {
        const float3 terminal_incident =
            params.max_order == 3 ? (third_point - second_point) : (second_point - first_point);
        if (!keller_grid_hit_from_incident(terminal_incident,
                                           lane,
                                           7u + static_cast<unsigned int>(params.max_order),
                                           terminal_point,
                                           terminal_edge_dir,
                                           final_target,
                                           cell)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }
    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    if (is_suffix) {
        const int suffix_face0_prim =
            params.max_order == 3
                ? recursive_state_prim0_at(third_idx)
                : recursive_state_prim0_at(second_idx);
        const int suffix_face1_prim =
            params.max_order == 3
                ? recursive_state_prim1_at(third_idx)
                : recursive_state_prim1_at(second_idx);
        if (!suffix_reflection_connection(terminal_point,
                                          target,
                                          suffix_face0_prim,
                                          suffix_face1_prim,
                                          lane,
                                          23u + static_cast<unsigned int>(params.max_order),
                                          final_target,
                                          suffix_prim,
                                          suffix_reflection_gain,
                                          suffix_fspl,
                                          suffix_candidate_count)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }
    const bool source_visible = visible_segment_impl<PrimaryOnly>(source, first_point);
    const bool first_inter_edge_visible = visible_segment_impl<PrimaryOnly>(first_point, second_point);
    const bool second_inter_edge_visible =
        params.max_order == 3 ? visible_segment_impl<PrimaryOnly>(second_point, third_point) : true;
    const bool target_visible = is_suffix
        ? (visible_segment_ignore_prim_impl<PrimaryOnly>(terminal_point, final_target, suffix_prim) &&
           visible_segment_ignore_prim_impl<PrimaryOnly>(final_target, target, suffix_prim))
        : visible_segment_impl<PrimaryOnly>(terminal_point, final_target);
    if (!source_visible || !target_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }
    if (!first_inter_edge_visible || !second_inter_edge_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_edge_vis_rejects, 1);
        }
        return;
    }

    const float first_weight = chain_event_weight(
        state_src_power_at(first_idx),
        state_prim0_at(first_idx),
        state_prim1_at(first_idx),
        state_edge_t_min_at(first_idx),
        state_edge_t_max_at(first_idx),
        state_exterior_angle_at(first_idx),
        source,
        first_point,
        second_point);
    const float3 second_target = params.max_order == 3 ? third_point : final_target;
    const float second_weight = chain_event_weight(
        1.f,
        recursive_state_prim0_at(second_idx),
        recursive_state_prim1_at(second_idx),
        recursive_state_edge_t_min_at(second_idx),
        recursive_state_edge_t_max_at(second_idx),
        recursive_state_exterior_angle_at(second_idx),
        first_point,
        second_point,
        second_target);
    float chain_weight = first_weight * second_weight;
    if (params.max_order == 3) {
        const float third_weight = chain_event_weight(
            1.f,
            recursive_state_prim0_at(third_idx),
            recursive_state_prim1_at(third_idx),
            recursive_state_edge_t_min_at(third_idx),
            recursive_state_edge_t_max_at(third_idx),
            recursive_state_exterior_angle_at(third_idx),
            second_point,
            third_point,
            final_target);
        chain_weight *= third_weight;
    }
    const float wave_gain_per_event =
        (params.wavelength * (1.f / (4.f * kPi))) *
        (params.wavelength * (1.f / (4.f * kPi)));
    const float wave_gain =
        params.max_order == 3 ? wave_gain_per_event * wave_gain_per_event
                              : wave_gain_per_event;
    const int strategy_sample_count =
        is_direct ? direct_limit : (is_keller ? keller_limit : suffix_limit);
    const float sample_norm = 1.f / fmaxf(static_cast<float>(strategy_sample_count), 1.f);
    float contribution =
        chain_weight * wave_gain * params.grid_cell_area * sample_norm;
    if (is_suffix) {
        contribution *= suffix_reflection_gain *
                        suffix_fspl *
                        fmaxf(suffix_candidate_count, 1.f);
    }
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = first_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(state_prim0_at(first_idx),
                                         state_prim1_at(first_idx));
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = first_u;
        }
    }

    const WarpCellGroup cell_group = warp_cell_group(cell);
    atomic_add_same_cell(params.out_power, cell, contribution, cell_group);
    atomic_add_same_cell(params.out_field_x_re, cell, sqrtf(fmaxf(contribution, 0.f)), cell_group);
    if (is_direct) {
        atomic_add_warp(params.out_direct_count, 1);
    } else if (is_keller) {
        atomic_add_warp(params.out_keller_count, 1);
    } else {
        atomic_add_warp(params.out_suffix_count, 1);
    }
    if (params.collect_edge_use != 0) {
        atomic_add_warp(params.out_edge_uses, 1);
    }
}

extern "C" __global__ void __raygen__diffraction_chain_accumulation() {
    run_diffraction_chain_accumulation_raygen<false>();
}

extern "C" __global__ void __raygen__diffraction_chain_accumulation_primary() {
    run_diffraction_chain_accumulation_raygen<true>();
}

} // namespace raydn

