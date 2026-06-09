#include <optix.h>
#include <optix_device.h>

#include <raydtorch/common/math.cuh>
#include <raydtorch/diffraction/paths_params.h>

namespace raydtorch {

extern "C" {
extern __constant__ DfrPathParams params;
}

namespace {

constexpr float kPathEps = 1e-6f;
constexpr float kSpeedOfLight = 299792458.f;

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

template <bool SplitScene>
static __forceinline__ __device__ HitPayload trace_scene(float3 origin,
                                                         float3 direction,
                                                         float tmax) {
    HitPayload primary;
    trace_handle(params.primary_handle, origin, direction, tmax, primary);
    if (!SplitScene) {
        return primary;
    }
    HitPayload secondary;
    trace_handle(params.secondary_handle, origin, direction, tmax, secondary);
    return choose_hit(primary, secondary);
}

template <bool SplitScene>
static __forceinline__ __device__ bool visible_segment(float3 start, float3 end) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit =
        trace_scene<SplitScene>(
            start + kDfrRayBias * dir,
            dir,
            fmaxf(dist - 2.f * kDfrRayBias, 0.f));
    return hit.hit == 0u;
}

static __forceinline__ __device__ float3 state_vec(const float *x,
                                                   const float *y,
                                                   const float *z,
                                                   int idx) {
    return make_f3(x[idx], y[idx], z[idx]);
}

static __forceinline__ __device__ float material_gain_for_faces(int face0_prim,
                                                                int face1_prim) {
    if (params.material_gain == nullptr || params.material_count <= 0) {
        return 1.f;
    }
    int prim = face0_prim;
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        prim = face1_prim;
    }
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        return 1.f;
    }
    return fmaxf(params.material_gain[prim], 0.f);
}

static __forceinline__ __device__ bool state_active(int state_idx) {
    if (params.active_mask == nullptr) {
        return true;
    }
    const int active_idx = params.active_width == 1 ? 0 : state_idx;
    return params.active_mask[active_idx] != 0u;
}

static __forceinline__ __device__ float path_weight(int state_idx,
                                                    float3 edge_point,
                                                    float3 receiver) {
    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const float source_distance = fmaxf(norm3(edge_point - source), kPathEps);
    const float receiver_distance = fmaxf(norm3(receiver - edge_point), kPathEps);
    const float edge_length = fmaxf(
        params.state_edge_t_max[state_idx] - params.state_edge_t_min[state_idx],
        0.f);
    const float exterior_angle =
        fmaxf(params.state_exterior_angle[state_idx], 0.25f * kPi);
    const float wedge_scale = fminf(exterior_angle / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_faces(params.state_prim0[state_idx],
                                                       params.state_prim1[state_idx]);
    const float wave_gain = params.wavelength * (1.f / (4.f * kPi));
    return params.state_src_power[state_idx] *
           material_gain *
           edge_length *
           wedge_scale *
           wave_gain *
           wave_gain /
           (source_distance * source_distance * receiver_distance * receiver_distance);
}

static __forceinline__ __device__ void write_point(float *x,
                                                   float *y,
                                                   float *z,
                                                   int idx,
                                                   float3 value) {
    if (x == nullptr || y == nullptr || z == nullptr) {
        return;
    }
    x[idx] = value.x;
    y[idx] = value.y;
    z[idx] = value.z;
}

static __forceinline__ __device__ int reserve_path_slot() {
    const unsigned int mask = __activemask();
    unsigned int lane = 0u;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    const int leader = __ffs(mask) - 1;
    const int rank = __popc(mask & ((1u << lane) - 1u));
    const int count = __popc(mask);
    int base = 0;
    if (static_cast<int>(lane) == leader) {
        base = atomicAdd(params.out_count, count);
    }
    base = __shfl_sync(mask, base, leader);
    return base + rank;
}

} // namespace

extern "C" {
__constant__ DfrPathParams params;
}

extern "C" __global__ void __closesthit__diffraction_paths() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__diffraction_paths() {
    optixSetPayload_0(0u);
}

template <bool SplitScene>
static __forceinline__ __device__ void trace_paths_order1_impl() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.capacity <= 0 ||
        params.tx_count <= 0 ||
        params.rx_count <= 0 ||
        params.state_count <= 0 ||
        params.state_limit <= 0 ||
        params.max_order != 1 ||
        (params.strategy_mask & RAYDTORCH_DFR_DIRECT) == 0 ||
        params.receiver_model != RAYDTORCH_DFR_MATCHED_ISO) {
        return;
    }

    const int state_limit = params.state_limit;
    const int rx_count = params.rx_count;
    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(state_limit));
    const int pair_idx = static_cast<int>(lane / static_cast<unsigned int>(state_limit));
    const int rx_idx = pair_idx % rx_count;
    const int tx_idx = pair_idx / rx_count;
    if (tx_idx >= params.tx_count || state_idx >= params.state_count || !state_active(state_idx)) {
        return;
    }

    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const float3 edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const float3 edge_dir =
        normalize3(state_vec(params.state_edge_dir_x,
                             params.state_edge_dir_y,
                             params.state_edge_dir_z,
                             state_idx));
    const float mid_t = 0.5f * (params.state_edge_t_min[state_idx] +
                               params.state_edge_t_max[state_idx]);
    const float3 edge_point = edge_pos + mid_t * edge_dir;
    const float3 receiver =
        make_f3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);

    if (!isfinite(source.x) || !isfinite(source.y) || !isfinite(source.z) ||
        !isfinite(edge_point.x) || !isfinite(edge_point.y) || !isfinite(edge_point.z) ||
        !isfinite(receiver.x) || !isfinite(receiver.y) || !isfinite(receiver.z)) {
        return;
    }
    if (!visible_segment<SplitScene>(source, edge_point) ||
        !visible_segment<SplitScene>(edge_point, receiver)) {
        return;
    }

    const float contribution = path_weight(state_idx, edge_point, receiver);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        return;
    }

    const int out_idx = reserve_path_slot();
    if (out_idx < 0 || out_idx >= params.capacity) {
        return;
    }

    const float path_length = norm3(edge_point - source) + norm3(receiver - edge_point);
    float phase_s;
    float phase_c;
    sincosf(-params.k * path_length, &phase_s, &phase_c);
    const float amplitude = sqrtf(fmaxf(contribution, 0.f));

    params.out_valid[out_idx] = 1u;
    params.out_tx_id[out_idx] = tx_idx;
    params.out_rx_id[out_idx] = rx_idx;
    params.out_order[out_idx] = 1;
    params.out_edge0[out_idx] = params.state_edge_index[state_idx];
    params.out_delay[out_idx] = path_length / kSpeedOfLight;
    params.out_field_x_re[out_idx] = amplitude * phase_c;
    params.out_field_x_im[out_idx] = amplitude * phase_s;
    write_point(params.out_p0_x, params.out_p0_y, params.out_p0_z,
                out_idx, edge_point);
}

static __forceinline__ __device__ bool paths_order1_lane(unsigned int lane,
                                                         int &state_idx,
                                                         int &rx_idx,
                                                         int &tx_idx) {
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.capacity <= 0 ||
        params.tx_count <= 0 ||
        params.rx_count <= 0 ||
        params.state_count <= 0 ||
        params.state_limit <= 0 ||
        params.max_order != 1 ||
        (params.strategy_mask & RAYDTORCH_DFR_DIRECT) == 0 ||
        params.receiver_model != RAYDTORCH_DFR_MATCHED_ISO) {
        return false;
    }

    const int state_limit = params.state_limit;
    const int rx_count = params.rx_count;
    state_idx = static_cast<int>(lane % static_cast<unsigned int>(state_limit));
    const int pair_idx = static_cast<int>(lane / static_cast<unsigned int>(state_limit));
    rx_idx = pair_idx % rx_count;
    tx_idx = pair_idx / rx_count;
    return tx_idx < params.tx_count &&
           state_idx < params.state_count &&
           state_active(state_idx);
}

static __forceinline__ __device__ float3 paths_edge_point(int state_idx) {
    const float3 edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const float3 edge_dir =
        normalize3(state_vec(params.state_edge_dir_x,
                             params.state_edge_dir_y,
                             params.state_edge_dir_z,
                             state_idx));
    const float mid_t = 0.5f * (params.state_edge_t_min[state_idx] +
                               params.state_edge_t_max[state_idx]);
    return edge_pos + mid_t * edge_dir;
}

static __forceinline__ __device__ bool finite_paths_points(float3 source,
                                                           float3 edge_point,
                                                           float3 receiver) {
    return isfinite(source.x) && isfinite(source.y) && isfinite(source.z) &&
           isfinite(edge_point.x) && isfinite(edge_point.y) && isfinite(edge_point.z) &&
           isfinite(receiver.x) && isfinite(receiver.y) && isfinite(receiver.z);
}

static __forceinline__ __device__ void trace_paths_order1_source_visibility_primary_impl() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.temp_visibility == nullptr) {
        return;
    }
    params.temp_visibility[lane] = 0u;

    int state_idx = -1;
    int rx_idx = -1;
    int tx_idx = -1;
    if (!paths_order1_lane(lane, state_idx, rx_idx, tx_idx)) {
        return;
    }
    (void)rx_idx;
    (void)tx_idx;

    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const float3 edge_point = paths_edge_point(state_idx);
    if (!isfinite(source.x) || !isfinite(source.y) || !isfinite(source.z) ||
        !isfinite(edge_point.x) || !isfinite(edge_point.y) || !isfinite(edge_point.z)) {
        return;
    }

    params.temp_visibility[lane] =
        visible_segment<false>(source, edge_point) ? 1u : 0u;
}

static __forceinline__ __device__ void trace_paths_order1_target_export_primary_impl() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }
    if (params.temp_visibility != nullptr && params.temp_visibility[lane] == 0u) {
        return;
    }

    int state_idx = -1;
    int rx_idx = -1;
    int tx_idx = -1;
    if (!paths_order1_lane(lane, state_idx, rx_idx, tx_idx)) {
        return;
    }

    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const float3 edge_point = paths_edge_point(state_idx);
    const float3 receiver =
        make_f3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);

    if (!finite_paths_points(source, edge_point, receiver)) {
        return;
    }
    if (!visible_segment<false>(edge_point, receiver)) {
        return;
    }

    const float contribution = path_weight(state_idx, edge_point, receiver);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        return;
    }

    const int out_idx = reserve_path_slot();
    if (out_idx < 0 || out_idx >= params.capacity) {
        return;
    }

    const float path_length = norm3(edge_point - source) + norm3(receiver - edge_point);
    float phase_s;
    float phase_c;
    sincosf(-params.k * path_length, &phase_s, &phase_c);
    const float amplitude = sqrtf(fmaxf(contribution, 0.f));

    params.out_valid[out_idx] = 1u;
    params.out_tx_id[out_idx] = tx_idx;
    params.out_rx_id[out_idx] = rx_idx;
    params.out_order[out_idx] = 1;
    params.out_edge0[out_idx] = params.state_edge_index[state_idx];
    params.out_delay[out_idx] = path_length / kSpeedOfLight;
    params.out_field_x_re[out_idx] = amplitude * phase_c;
    params.out_field_x_im[out_idx] = amplitude * phase_s;
    write_point(params.out_p0_x, params.out_p0_y, params.out_p0_z,
                out_idx, edge_point);
}

extern "C" __global__ void __raygen__diffraction_paths_order1_primary() {
    trace_paths_order1_impl<false>();
}

extern "C" __global__ void __raygen__diffraction_paths_order1() {
    trace_paths_order1_impl<true>();
}

extern "C" __global__ void __raygen__diffraction_paths_order1_source_visibility_primary() {
    trace_paths_order1_source_visibility_primary_impl();
}

extern "C" __global__ void __raygen__diffraction_paths_order1_target_export_primary() {
    trace_paths_order1_target_export_primary_impl();
}

} // namespace raydtorch

