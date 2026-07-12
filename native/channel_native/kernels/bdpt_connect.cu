#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../field_transport.cuh"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace utd = witwin::channel::native_ext;
namespace transport = channel_native::field_transport;

constexpr float kLightSpeedMPerS = 299792458.0f;
constexpr float kPi = 3.14159265358979323846f;

// BDPT component_mask bits (contract section 1): 1=los, 2=reflection,
// 4=diffraction, 8=transmission, 16=scattering.
constexpr int kMaskReflection = 2;
constexpr int kMaskDiffraction = 4;
constexpr int kMaskTransmission = 8;
constexpr int kMaskScattering = 16;
// Connection-sample component ids follow core/path_topology.py:
// 0=los, 1=reflection, 2=diffraction, 5=transmission, 6=scattering.
constexpr int kComponentLos = 0;
constexpr int kComponentReflection = 1;
constexpr int kComponentDiffraction = 2;
constexpr int kComponentTransmission = 5;
constexpr int kComponentScattering = 6;

// Collapse a per-path component mask to the EXCLUSIVE path_class with the
// contract priority scattering > diffraction > transmission > reflection >
// los, so mixed paths are never double counted across component buckets.
__device__ int bdpt_component_from_mask(int mask) {
    if (mask & kMaskScattering) {
        return kComponentScattering;
    }
    if (mask & kMaskDiffraction) {
        return kComponentDiffraction;
    }
    if (mask & kMaskTransmission) {
        return kComponentTransmission;
    }
    if (mask & kMaskReflection) {
        return kComponentReflection;
    }
    return kComponentLos;
}

__device__ bool bdpt_component_accumulable(int component) {
    return component == kComponentLos || component == kComponentReflection ||
        component == kComponentDiffraction || component == kComponentTransmission ||
        component == kComponentScattering;
}

__device__ unsigned long long bdpt_splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ float bdpt_uniform01_from_u64(unsigned long long x) {
    return static_cast<float>((x >> 40) & 0xffffffULL) / 16777216.0f;
}

void check_float_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bool_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kBool, name, " must be bool");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_vec3_cuda(const at::Tensor& tensor, const char* name) {
    check_float_cuda(tensor, name, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

void check_same_device(const at::Tensor& tensor, const at::Tensor& reference, const char* name) {
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must share device");
}

void check_mis_args(int64_t mode_id, int64_t strategy_count) {
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1, or 2");
    TORCH_CHECK(strategy_count > 0, "strategy_count must be positive");
}

void check_diffraction_mis_args(
    int64_t mode_id,
    int64_t strategy_count,
    int64_t direct_samples,
    int64_t keller_samples) {
    check_mis_args(mode_id, strategy_count);
    const int64_t actual = (direct_samples > 0 ? 1 : 0) + (keller_samples > 0 ? 1 : 0);
    TORCH_CHECK(
        strategy_count == actual,
        "strategy_count must match enabled direct/Keller proposals");
    TORCH_CHECK(mode_id != 0 || actual == 1, "MIS none requires exactly one diffraction proposal");
}

__device__ float bdpt_connection_mis_weight_from_sums(
    float pdf,
    float balance_pdf_sum,
    float power_pdf_sum,
    int mode_id,
    float beta) {
    if (pdf <= 0.0f) {
        return 0.0f;
    }
    if (mode_id == 0) {
        return 1.0f;
    }
    if (mode_id == 1) {
        return pdf / fmaxf(balance_pdf_sum, 1.17549435e-38f);
    }
    return powf(pdf, beta) / fmaxf(power_pdf_sum, 1.17549435e-38f);
}

__device__ float bdpt_single_strategy_mis_weight(float pdf, int mode_id, float beta) {
    return bdpt_connection_mis_weight_from_sums(pdf, pdf, powf(pdf, beta), mode_id, beta);
}

__device__ float bdpt_diffraction_strategy_mis_weight(
    float pdf,
    float direct_pdf,
    float keller_pdf,
    int strategy_count,
    int mode_id,
    float beta) {
    if (strategy_count <= 1) {
        return bdpt_single_strategy_mis_weight(pdf, mode_id, beta);
    }
    const float balance_sum = fmaxf(direct_pdf, 0.0f) + fmaxf(keller_pdf, 0.0f);
    const float power_sum =
        powf(fmaxf(direct_pdf, 0.0f), beta) +
        powf(fmaxf(keller_pdf, 0.0f), beta);
    return bdpt_connection_mis_weight_from_sums(pdf, balance_sum, power_sum, mode_id, beta);
}

__device__ float bdpt_free_space_gain(float tx_power, float distance, float frequency_hz) {
    const float wavelength = kLightSpeedMPerS / fmaxf(frequency_hz, 1.0f);
    const float denom = 4.0f * kPi * fmaxf(distance, 1.0e-6f) / wavelength;
    return tx_power / fmaxf(denom * denom, 1.0e-30f);
}

__device__ float3 bdpt_make_float3(float x, float y, float z) {
    float3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

__device__ float3 bdpt_add3(float3 a, float3 b) {
    return bdpt_make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 bdpt_sub3(float3 a, float3 b) {
    return bdpt_make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 bdpt_scale3(float3 a, float s) {
    return bdpt_make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float bdpt_norm3(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ float3 bdpt_normalize3(float3 a) {
    const float inv = 1.0f / fmaxf(bdpt_norm3(a), 1.0e-12f);
    return bdpt_scale3(a, inv);
}

__device__ float3 bdpt_vec3_at(const float* values, int index) {
    const float* row = values + static_cast<int64_t>(index) * 3;
    return bdpt_make_float3(row[0], row[1], row[2]);
}

__device__ float3 bdpt_grid_cell_center(
    int cell,
    int grid_axis,
    float grid_position,
    float coord0_min,
    float coord0_max,
    float coord1_min,
    float coord1_max,
    int resolution0,
    int resolution1) {
    const int i = cell % resolution0;
    const int j = cell / resolution0;
    const float u = coord0_min + (static_cast<float>(i) + 0.5f) *
        (coord0_max - coord0_min) / fmaxf(static_cast<float>(resolution0), 1.0f);
    const float v = coord1_min + (static_cast<float>(j) + 0.5f) *
        (coord1_max - coord1_min) / fmaxf(static_cast<float>(resolution1), 1.0f);
    if (grid_axis == 0) {
        return bdpt_make_float3(grid_position, u, v);
    }
    if (grid_axis == 1) {
        return bdpt_make_float3(u, grid_position, v);
    }
    return bdpt_make_float3(u, v, grid_position);
}

__device__ float bdpt_diffraction_contribution(
    float src_power,
    float material_gain,
    float wavelength,
    float edge_measure_weight,
    float grid_cell_area,
    float exterior_angle,
    float3 source,
    float3 edge_point,
    float3 target) {
    const float source_distance = fmaxf(bdpt_norm3(bdpt_sub3(edge_point, source)), 1.0e-6f);
    const float target_distance = fmaxf(bdpt_norm3(bdpt_sub3(target, edge_point)), 1.0e-6f);
    const float wave = wavelength * (1.0f / (4.0f * kPi));
    const float wedge_scale = fminf(fmaxf(exterior_angle, 0.25f * kPi) / (2.0f * kPi), 2.0f);
    return src_power * material_gain * wave * wave * fmaxf(edge_measure_weight, 0.0f) *
        grid_cell_area * wedge_scale /
        (source_distance * source_distance * target_distance * target_distance);
}

__global__ void bdpt_mis_weights_kernel(
    int64_t count,
    const float* pdf,
    const float* strategy_pdf_sum,
    int mode_id,
    float beta,
    float* weights) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = pdf[index];
    float sum = strategy_pdf_sum[0];
    if (value <= 0.0f || sum <= 0.0f) {
        weights[index] = 0.0f;
        return;
    }
    if (mode_id == 0) {
        weights[index] = 1.0f;
    } else if (mode_id == 1) {
        weights[index] = value / fmaxf(sum, 1.17549435e-38f);
    } else {
        weights[index] = powf(value, beta) / fmaxf(sum, 1.17549435e-38f);
    }
}

__global__ void bdpt_endpoint_connection_samples_kernel(
    int64_t count,
    int64_t sensor_count,
    float frequency_hz,
    float inv_samples_per_tx,
    int mode_id,
    float beta,
    int strategy_count,
    const float* light_origin,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const float* light_pdf_forward,
    const int* light_depth,
    const int* light_component_mask,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const float* sensor_pdf_reverse,
    const int* sensor_depth,
    const int* sensor_rx_id,
    const int* sensor_grid_linear_id,
    const bool* sensor_valid,
    int* topology,
    float* contribution,
    float* pdf,
    float* mis_weight,
    int* component_id,
    bool* valid,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const int tx = light_tx_id[light_index];
    const int rx = sensor_rx_id[sensor_index];
    const int grid = sensor_grid_linear_id[sensor_index];
    const bool is_valid = light_valid[light_index] && sensor_valid[sensor_index] && tx >= 0 && rx >= 0;

    const float lx = light_origin[light_index * 3 + 0];
    const float ly = light_origin[light_index * 3 + 1];
    const float lz = light_origin[light_index * 3 + 2];
    const float sx = sensor_origin[sensor_index * 3 + 0];
    const float sy = sensor_origin[sensor_index * 3 + 1];
    const float sz = sensor_origin[sensor_index * 3 + 2];
    const float dx = sx - lx;
    const float dy = sy - ly;
    const float dz = sz - lz;
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1.0e-6f);
    const int light_path_depth = light_depth[light_index];
    const float dir_dot = light_path_depth > 0
        ? (dx * light_direction[light_index * 3 + 0] +
              dy * light_direction[light_index * 3 + 1] +
              dz * light_direction[light_index * 3 + 2]) /
            distance
        : 1.0f;
    const bool direction_valid = dir_dot > 0.0f;
    const bool row_valid = is_valid && direction_valid;
    // Proposal density excludes free-space geometry. The deterministic
    // endpoint connection has unit discrete mass; inverse-square spreading
    // belongs to the contribution, not to the sampling PDF.
    const float row_pdf = row_valid
        ? fmaxf(light_pdf_forward[light_index], 0.0f) *
            fmaxf(sensor_pdf_reverse[sensor_index], 0.0f)
        : 0.0f;
    // The free-space spreading acts over the unfolded path (light-subpath
    // prefix + connection segment), not the last segment alone.
    const float total_distance = distance + fmaxf(light_path_length[light_index], 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float amplitude = 1.0f /
        (2.0f * fmaxf(wave_number, 1.0e-12f) * fmaxf(total_distance, 1.0e-6f));
    const utd::Complex propagation = utd::cplx_mul_real(
        utd::cplx_exp_phase(transport::precise_neg_kd(wave_number, total_distance)),
        amplitude);
    const int64_t field_offset = light_index * 3;
    const utd::Complex3 incident_field = {
        utd::cplx(light_field_real[field_offset], light_field_imag[field_offset]),
        utd::cplx(light_field_real[field_offset + 1], light_field_imag[field_offset + 1]),
        utd::cplx(light_field_real[field_offset + 2], light_field_imag[field_offset + 2])};
    const utd::Complex3 received_field = utd::c3_scale(incident_field, propagation);
    const utd::float3a connection_direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    const int64_t sensor_field_offset = sensor_index * 3;
    const utd::float3a receiver_polarization = utd::make_f3(
        sensor_field_real[sensor_field_offset],
        sensor_field_real[sensor_field_offset + 1],
        sensor_field_real[sensor_field_offset + 2]);
    const utd::Complex coefficient = transport::project_receiver(
        received_field, connection_direction, receiver_polarization);
    const float coefficient_power = utd::cplx_abs_sqr(coefficient);
    const float row_contribution = row_valid
        ? light_source_power[light_index] * coefficient_power * inv_samples_per_tx
        : 0.0f;

    tx_id[index] = tx;
    rx_id[index] = rx;
    grid_linear_id[index] = grid;
    const int light_component = light_component_mask[light_index];
    const int sample_component = bdpt_component_from_mask(light_component);
    component_id[index] = sample_component;
    out_light_depth[index] = light_depth[light_index];
    out_sensor_depth[index] = sensor_depth[sensor_index];
    contribution[index] = row_contribution;
    pdf[index] = row_pdf;
    mis_weight[index] = row_valid ? bdpt_single_strategy_mis_weight(row_pdf, mode_id, beta) : 0.0f;
    valid[index] = row_valid;
    path_length_m[index] = total_distance;
    const int row = static_cast<int>(index * 4);
    topology[row + 0] = tx;
    topology[row + 1] = rx;
    topology[row + 2] = sample_component;
    topology[row + 3] = light_depth[light_index] + sensor_depth[sensor_index];
}

__global__ void bdpt_diffraction_connection_samples_from_tape_kernel(
    int64_t count,
    int tx_index,
    int state_count,
    int grid_resolution0,
    int grid_resolution1,
    int grid_axis,
    float grid_position,
    float grid_coord0_min,
    float grid_coord0_max,
    float grid_coord1_min,
    float grid_coord1_max,
    float grid_cell_area,
    float wavelength,
    int direct_samples,
    int keller_samples,
    int mode_id,
    float beta,
    int strategy_count,
    int material_count,
    const bool* tape_active,
    const int* tape_state_idx,
    const int* tape_cell,
    const int* tape_material_idx,
    const float* tape_edge_u,
    const int* state_edge_index,
    const float* state_edge_pos,
    const float* state_edge_dir,
    const float* state_edge_t_min,
    const float* state_edge_t_max,
    const float* state_exterior_angle,
    const float* state_src,
    const float* state_src_power,
    const float* material_gain,
    const bool* material_valid,
    int* topology,
    float* contribution,
    float* pdf,
    float* mis_weight,
    int* component_id,
    bool* valid,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* path_length_m) {
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= count) {
        return;
    }
    const int total_samples = direct_samples + keller_samples;
    const bool lane_direct = static_cast<int>(lane) < direct_samples;
    const bool lane_keller =
        !lane_direct && static_cast<int>(lane) < total_samples;
    const int strategy_samples = lane_direct ? direct_samples : (lane_keller ? keller_samples : 0);
    const int state_idx = tape_state_idx[lane];
    const int raydn_cell = tape_cell[lane];
    const int material_idx = tape_material_idx[lane];
    const int cell_count = grid_resolution0 * grid_resolution1;
    const bool row_valid = tape_active[lane] &&
        strategy_samples > 0 &&
        state_idx >= 0 &&
        state_idx < state_count &&
        raydn_cell >= 0 &&
        raydn_cell < cell_count &&
        material_idx >= 0 &&
        material_idx < material_count &&
        material_valid[material_idx];

    const int row = raydn_cell % grid_resolution0;
    const int col = raydn_cell / grid_resolution0;
    const int bdpt_cell = row_valid ? row * grid_resolution1 + col : -1;
    float row_contribution = 0.0f;
    float row_pdf = 0.0f;
    float direct_pdf = 0.0f;
    float keller_pdf = 0.0f;
    float row_path_length = 0.0f;
    if (row_valid) {
        const float t_min = state_edge_t_min[state_idx];
        const float t_max = state_edge_t_max[state_idx];
        const float edge_length = fmaxf(t_max - t_min, 0.0f);
        const float edge_t = t_min + tape_edge_u[lane] * (t_max - t_min);
        const float3 edge_origin = bdpt_vec3_at(state_edge_pos, state_idx);
        const float3 edge_dir = bdpt_normalize3(bdpt_vec3_at(state_edge_dir, state_idx));
        const float3 edge_point = bdpt_add3(edge_origin, bdpt_scale3(edge_dir, edge_t));
        const float3 source = bdpt_vec3_at(state_src, state_idx);
        const float3 target = bdpt_grid_cell_center(
            raydn_cell,
            grid_axis,
            grid_position,
            grid_coord0_min,
            grid_coord0_max,
            grid_coord1_min,
            grid_coord1_max,
            grid_resolution0,
            grid_resolution1);
        // The launch samples a state and a receiver cell in addition to the
        // continuous edge coordinate.  Preserve those discrete proposal
        // probabilities when reconstructing the exported sample; omitting
        // state_count * cell_count made the tape disagree with the map
        // accumulator by exactly that factor.
        const float discrete_domain =
            static_cast<float>(state_count) * static_cast<float>(cell_count);
        const float edge_measure_weight =
            edge_length * discrete_domain /
            fmaxf(static_cast<float>(strategy_samples), 1.0f);
        const float edge_pdf_base = 1.0f /
            fmaxf(edge_length * discrete_domain * grid_cell_area, 1.0e-30f);
        direct_pdf = direct_samples > 0 ? static_cast<float>(direct_samples) * edge_pdf_base : 0.0f;
        keller_pdf = keller_samples > 0 ? static_cast<float>(keller_samples) * edge_pdf_base : 0.0f;
        row_contribution = bdpt_diffraction_contribution(
            state_src_power[state_idx],
            material_gain[material_idx],
            wavelength,
            edge_measure_weight,
            grid_cell_area,
            state_exterior_angle[state_idx],
            source,
            edge_point,
            target);
        row_pdf = static_cast<float>(strategy_samples) * edge_pdf_base;
        row_path_length =
            bdpt_norm3(bdpt_sub3(edge_point, source)) +
            bdpt_norm3(bdpt_sub3(target, edge_point));
    }

    tx_id[lane] = tx_index;
    rx_id[lane] = bdpt_cell;
    grid_linear_id[lane] = bdpt_cell;
    component_id[lane] = 2;
    out_light_depth[lane] = 1;
    out_sensor_depth[lane] = 0;
    contribution[lane] = row_valid ? row_contribution : 0.0f;
    pdf[lane] = row_valid ? row_pdf : 0.0f;
    mis_weight[lane] = row_valid
        ? bdpt_diffraction_strategy_mis_weight(row_pdf, direct_pdf, keller_pdf, strategy_count, mode_id, beta)
        : 0.0f;
    valid[lane] = row_valid;
    path_length_m[lane] = row_path_length;
    const int64_t top = lane * 4;
    topology[top + 0] = tx_index;
    topology[top + 1] = bdpt_cell;
    topology[top + 2] = 2;
    topology[top + 3] = 1;
}

__global__ void bdpt_diffraction_point_connection_samples_kernel(
    int64_t count,
    int tx_index,
    int state_count,
    int rx_count,
    float wavelength,
    int direct_samples,
    int keller_samples,
    int mode_id,
    float beta,
    int strategy_count,
    int material_count,
    unsigned long long seed,
    const int* state_edge_index,
    const float* state_edge_pos,
    const float* state_edge_dir,
    const float* state_edge_t_min,
    const float* state_edge_t_max,
    const int* state_prim0,
    const int* state_prim1,
    const float* state_exterior_angle,
    const float* state_src,
    const float* state_src_power,
    const float* rx_positions,
    const float* material_gain,
    const bool* material_valid,
    int* topology,
    float* contribution,
    float* pdf,
    float* mis_weight,
    int* component_id,
    bool* valid,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* path_length_m,
    float* source_start,
    float* source_end,
    float* target_start,
    float* target_end,
    bool* visibility_active) {
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= count) {
        return;
    }
    const int sample_id = rx_count > 0 ? static_cast<int>(lane / rx_count) : 0;
    const int rx = rx_count > 0 ? static_cast<int>(lane % rx_count) : -1;
    const bool lane_direct = sample_id < direct_samples;
    const bool lane_keller = !lane_direct && sample_id < direct_samples + keller_samples;
    const int strategy_samples = lane_direct ? direct_samples : (lane_keller ? keller_samples : 0);
    const unsigned long long lane_seed =
        seed ^ (static_cast<unsigned long long>(lane) * 0xd1b54a32d192ed03ULL);
    const float u_state = bdpt_uniform01_from_u64(bdpt_splitmix64(lane_seed ^ 0x6a09e667f3bcc909ULL));
    const float u_edge = bdpt_uniform01_from_u64(bdpt_splitmix64(lane_seed ^ 0xbb67ae8584caa73bULL));
    int state_idx = state_count > 0 ? static_cast<int>(floorf(u_state * static_cast<float>(state_count))) : -1;
    if (state_idx >= state_count) {
        state_idx = state_count - 1;
    }
    const int prim0 = state_idx >= 0 ? state_prim0[state_idx] : -1;
    const int prim1 = state_idx >= 0 ? state_prim1[state_idx] : -1;
    const int material_idx = (prim0 >= 0 && prim0 < material_count) ? prim0 : prim1;
    const bool material_ok =
        material_idx >= 0 &&
        material_idx < material_count &&
        material_valid[material_idx];
    const bool row_valid =
        strategy_samples > 0 &&
        state_idx >= 0 &&
        state_idx < state_count &&
        rx >= 0 &&
        rx < rx_count &&
        material_ok;

    float3 source = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float3 edge_point = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float3 target = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float row_contribution = 0.0f;
    float row_pdf = 0.0f;
    float direct_pdf = 0.0f;
    float keller_pdf = 0.0f;
    float row_path_length = 0.0f;
    if (row_valid) {
        const float t_min = state_edge_t_min[state_idx];
        const float t_max = state_edge_t_max[state_idx];
        const float edge_length = fmaxf(t_max - t_min, 0.0f);
        const float edge_t = t_min + u_edge * (t_max - t_min);
        const float3 edge_origin = bdpt_vec3_at(state_edge_pos, state_idx);
        const float3 edge_dir = bdpt_normalize3(bdpt_vec3_at(state_edge_dir, state_idx));
        edge_point = bdpt_add3(edge_origin, bdpt_scale3(edge_dir, edge_t));
        source = bdpt_vec3_at(state_src, state_idx);
        target = bdpt_vec3_at(rx_positions, rx);
        const float edge_measure_weight =
            edge_length * fmaxf(static_cast<float>(state_count), 1.0f) /
            fmaxf(static_cast<float>(strategy_samples), 1.0f);
        const float edge_pdf_base =
            1.0f / fmaxf(static_cast<float>(state_count) * edge_length, 1.0e-30f);
        direct_pdf = direct_samples > 0 ? static_cast<float>(direct_samples) * edge_pdf_base : 0.0f;
        keller_pdf = keller_samples > 0 ? static_cast<float>(keller_samples) * edge_pdf_base : 0.0f;
        row_contribution = bdpt_diffraction_contribution(
            state_src_power[state_idx],
            material_gain[material_idx],
            wavelength,
            edge_measure_weight,
            1.0f,
            state_exterior_angle[state_idx],
            source,
            edge_point,
            target);
        row_pdf = static_cast<float>(strategy_samples) * edge_pdf_base;
        row_path_length =
            bdpt_norm3(bdpt_sub3(edge_point, source)) +
            bdpt_norm3(bdpt_sub3(target, edge_point));
    }

    tx_id[lane] = tx_index;
    rx_id[lane] = row_valid ? rx : -1;
    grid_linear_id[lane] = row_valid ? rx : -1;
    component_id[lane] = 2;
    out_light_depth[lane] = 1;
    out_sensor_depth[lane] = 0;
    contribution[lane] = row_valid ? row_contribution : 0.0f;
    pdf[lane] = row_valid ? row_pdf : 0.0f;
    mis_weight[lane] = row_valid
        ? bdpt_diffraction_strategy_mis_weight(row_pdf, direct_pdf, keller_pdf, strategy_count, mode_id, beta)
        : 0.0f;
    valid[lane] = row_valid;
    path_length_m[lane] = row_valid ? row_path_length : 0.0f;
    const int64_t top = lane * 4;
    topology[top + 0] = tx_index;
    topology[top + 1] = row_valid ? rx : -1;
    topology[top + 2] = 2;
    topology[top + 3] = 1;
    const int64_t vec = lane * 3;
    source_start[vec + 0] = source.x;
    source_start[vec + 1] = source.y;
    source_start[vec + 2] = source.z;
    source_end[vec + 0] = edge_point.x;
    source_end[vec + 1] = edge_point.y;
    source_end[vec + 2] = edge_point.z;
    target_start[vec + 0] = edge_point.x;
    target_start[vec + 1] = edge_point.y;
    target_start[vec + 2] = edge_point.z;
    target_end[vec + 0] = target.x;
    target_end[vec + 1] = target.y;
    target_end[vec + 2] = target.z;
    visibility_active[lane] = row_valid;
}

__global__ void bdpt_endpoint_connection_visibility_inputs_kernel(
    int64_t count,
    int64_t sensor_count,
    const float* light_origin,
    const int* light_tx_id,
    const bool* light_valid,
    const float* sensor_origin,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    float* start,
    float* end,
    bool* active) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const float* src = light_origin + light_index * 3;
    const float* dst = sensor_origin + sensor_index * 3;
    float* out_start = start + index * 3;
    float* out_end = end + index * 3;
    out_start[0] = src[0];
    out_start[1] = src[1];
    out_start[2] = src[2];
    out_end[0] = dst[0];
    out_end[1] = dst[1];
    out_end[2] = dst[2];
    active[index] = light_valid[light_index] && sensor_valid[sensor_index] &&
        light_tx_id[light_index] >= 0 && sensor_rx_id[sensor_index] >= 0;
}

__global__ void bdpt_accumulate_connection_samples_double_kernel(
    int64_t count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    double* path_gain,
    double* los,
    double* reflection,
    double* diffraction,
    double* transmission,
    double* scattering) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count || !bdpt_component_accumulable(component)) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double value = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
    atomicAdd(path_gain + out_index, value);
    if (component == kComponentLos) {
        atomicAdd(los + out_index, value);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection + out_index, value);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction + out_index, value);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission + out_index, value);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering + out_index, value);
    }
}

__global__ void bdpt_filter_connection_samples_kernel(
    int64_t count,
    const bool* visible,
    float* contribution,
    float* pdf,
    float* mis_weight,
    bool* valid) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const bool keep = valid[index] && visible[index];
    valid[index] = keep;
    if (!keep) {
        contribution[index] = 0.0f;
        pdf[index] = 0.0f;
        mis_weight[index] = 0.0f;
    }
}

__global__ void bdpt_compact_connection_samples_kernel(
    int64_t count,
    int64_t capacity,
    const int* topology,
    const float* contribution,
    const float* pdf,
    const float* mis_weight,
    const int* component_id,
    const bool* valid,
    const int* tx_id,
    const int* rx_id,
    const int* grid_linear_id,
    const int* light_depth,
    const int* sensor_depth,
    const float* path_length_m,
    int* compact_count,
    int* out_topology,
    float* out_contribution,
    float* out_pdf,
    float* out_mis_weight,
    int* out_component_id,
    bool* out_valid,
    int* out_tx_id,
    int* out_rx_id,
    int* out_grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* out_path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int slot = atomicAdd(compact_count, 1);
    if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
        return;
    }
    const int64_t src_row = index * 4;
    const int64_t dst_row = static_cast<int64_t>(slot) * 4;
    out_topology[dst_row + 0] = topology[src_row + 0];
    out_topology[dst_row + 1] = topology[src_row + 1];
    out_topology[dst_row + 2] = topology[src_row + 2];
    out_topology[dst_row + 3] = topology[src_row + 3];
    out_contribution[slot] = contribution[index];
    out_pdf[slot] = pdf[index];
    out_mis_weight[slot] = mis_weight[index];
    out_component_id[slot] = component_id[index];
    out_valid[slot] = true;
    out_tx_id[slot] = tx_id[index];
    out_rx_id[slot] = rx_id[index];
    out_grid_linear_id[slot] = grid_linear_id[index];
    out_light_depth[slot] = light_depth[index];
    out_sensor_depth[slot] = sensor_depth[index];
    out_path_length_m[slot] = path_length_m[index];
}

__global__ void bdpt_copy_connection_samples_kernel(
    int64_t count,
    int64_t dst_offset,
    const int* topology,
    const float* contribution,
    const float* pdf,
    const float* mis_weight,
    const int* component_id,
    const bool* valid,
    const int* tx_id,
    const int* rx_id,
    const int* grid_linear_id,
    const int* light_depth,
    const int* sensor_depth,
    const float* path_length_m,
    int* out_topology,
    float* out_contribution,
    float* out_pdf,
    float* out_mis_weight,
    int* out_component_id,
    bool* out_valid,
    int* out_tx_id,
    int* out_rx_id,
    int* out_grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* out_path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t dst = dst_offset + index;
    const int64_t src_row = index * 4;
    const int64_t dst_row = dst * 4;
    out_topology[dst_row + 0] = topology[src_row + 0];
    out_topology[dst_row + 1] = topology[src_row + 1];
    out_topology[dst_row + 2] = topology[src_row + 2];
    out_topology[dst_row + 3] = topology[src_row + 3];
    out_contribution[dst] = contribution[index];
    out_pdf[dst] = pdf[index];
    out_mis_weight[dst] = mis_weight[index];
    out_component_id[dst] = component_id[index];
    out_valid[dst] = valid[index];
    out_tx_id[dst] = tx_id[index];
    out_rx_id[dst] = rx_id[index];
    out_grid_linear_id[dst] = grid_linear_id[index];
    out_light_depth[dst] = light_depth[index];
    out_sensor_depth[dst] = sensor_depth[index];
    out_path_length_m[dst] = path_length_m[index];
}

__global__ void bdpt_count_valid_connection_samples_kernel(
    int64_t count,
    const bool* valid,
    int* compact_count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    atomicAdd(compact_count, 1);
}

__global__ void bdpt_compact_valid_connection_indices_kernel(
    int64_t count,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    int* compact_count,
    int* compact_indices) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count || !bdpt_component_accumulable(component)) {
        return;
    }
    const int slot = atomicAdd(compact_count, 1);
    compact_indices[slot] = static_cast<int>(index);
}

__global__ void bdpt_accumulate_connection_samples_compacted_kernel(
    int64_t capacity,
    const int* compact_count,
    const int* compact_indices,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    int64_t rx_count,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t compact_linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact_linear >= capacity || compact_linear >= static_cast<int64_t>(compact_count[0])) {
        return;
    }
    const int index = compact_indices[compact_linear];
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const float value = contribution[index] * mis_weight[index];
    atomicAdd(path_gain + out_index, value);
    if (component == kComponentLos) {
        atomicAdd(los + out_index, value);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection + out_index, value);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction + out_index, value);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission + out_index, value);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering + out_index, value);
    }
}

__global__ void bdpt_accumulate_connection_samples_staged_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_index >= out_count) {
        return;
    }
    const int out_tx = static_cast<int>(out_index / rx_count);
    const int out_rx = static_cast<int>(out_index - static_cast<int64_t>(out_tx) * rx_count);
    double path_sum = 0.0;
    double los_sum = 0.0;
    double reflection_sum = 0.0;
    double diffraction_sum = 0.0;
    double transmission_sum = 0.0;
    double scattering_sum = 0.0;
    for (int64_t index = 0; index < sample_count; ++index) {
        if (!valid[index] || tx_id[index] != out_tx || rx_id[index] != out_rx) {
            continue;
        }
        const int component = component_id[index];
        if (!bdpt_component_accumulable(component)) {
            continue;
        }
        const double value = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
        path_sum += value;
        if (component == kComponentLos) {
            los_sum += value;
        } else if (component == kComponentReflection) {
            reflection_sum += value;
        } else if (component == kComponentDiffraction) {
            diffraction_sum += value;
        } else if (component == kComponentTransmission) {
            transmission_sum += value;
        } else if (component == kComponentScattering) {
            scattering_sum += value;
        }
    }
    path_gain[out_index] = static_cast<float>(path_sum);
    los[out_index] = static_cast<float>(los_sum);
    reflection[out_index] = static_cast<float>(reflection_sum);
    diffraction[out_index] = static_cast<float>(diffraction_sum);
    transmission[out_index] = static_cast<float>(transmission_sum);
    scattering[out_index] = static_cast<float>(scattering_sum);
}

__global__ void bdpt_cast_connection_accumulation_kernel(
    int64_t count,
    const double* path_gain_sum,
    const double* los_sum,
    const double* reflection_sum,
    const double* diffraction_sum,
    const double* transmission_sum,
    const double* scattering_sum,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    path_gain[index] = static_cast<float>(path_gain_sum[index]);
    los[index] = static_cast<float>(los_sum[index]);
    reflection[index] = static_cast<float>(reflection_sum[index]);
    diffraction[index] = static_cast<float>(diffraction_sum[index]);
    transmission[index] = static_cast<float>(transmission_sum[index]);
    scattering[index] = static_cast<float>(scattering_sum[index]);
}

__global__ void bdpt_connection_variance_accum_double_kernel(
    int64_t count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const bool* valid,
    int64_t rx_count,
    double samples_per_tx,
    double* sum,
    double* sum_square_unweighted,
    int* sample_count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    if (tx < 0 || rx < 0 || rx >= rx_count) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double weighted = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
    const double unweighted = weighted * samples_per_tx;
    atomicAdd(sum + out_index, weighted);
    atomicAdd(sum_square_unweighted + out_index, unweighted * unweighted);
    atomicAdd(sample_count + out_index, 1);
}

__global__ void bdpt_connection_variance_finalize_double_kernel(
    int64_t count,
    const double* sum,
    const double* sum_square_unweighted,
    const int* sample_count,
    double samples_per_tx,
    float* variance) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int n = sample_count[index];
    if (n <= 0 || samples_per_tx <= 0.0) {
        variance[index] = 0.0f;
        return;
    }
    const double mean = sum[index];
    const double ex2 = sum_square_unweighted[index] / samples_per_tx;
    const double variance_value = fmax(ex2 - mean * mean, 0.0) / samples_per_tx;
    variance[index] = variance_value <= 1.0e-30 ? 0.0f : static_cast<float>(variance_value);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
allocate_connection_samples(const at::Tensor& reference, int64_t count) {
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto bool_options = reference.options().dtype(at::kBool);
    return {
        at::empty({count, 4}, int_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, int_options),
        at::empty({count}, bool_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, float_options),
    };
}

void zero_double_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<double>(), 0, tensor.numel() * sizeof(double), stream));
}

void zero_int_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<int>(), 0, tensor.numel() * sizeof(int), stream));
}

void zero_float_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<float>(), 0, tensor.numel() * sizeof(float), stream));
}

}  // namespace

at::Tensor cn_bdpt_mis_weights_cuda(
    at::Tensor pdf,
    at::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta) {
    check_float_cuda(pdf, "pdf", 1);
    check_float_cuda(strategy_pdf_sum, "strategy_pdf_sum", 0);
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1, or 2");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    auto weights = at::empty_like(pdf);
    int64_t count = pdf.numel();
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(pdf.get_device()).stream();
        bdpt_mis_weights_kernel<<<blocks, threads, 0, stream>>>(
            count,
            pdf.data_ptr<float>(),
            strategy_pdf_sum.data_ptr<float>(),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            weights.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return weights;
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_endpoint_connection_samples_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_pdf_forward,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_pdf_reverse,
    at::Tensor sensor_depth,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_grid_linear_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_float_cuda(light_throughput_real, "light_throughput_real", 1);
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_float_cuda(light_pdf_forward, "light_pdf_forward", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_component_mask, "light_component_mask", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_float_cuda(sensor_pdf_reverse, "sensor_pdf_reverse", 1);
    check_int_cuda(sensor_depth, "sensor_depth", 1);
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_int_cuda(sensor_grid_linear_id, "sensor_grid_linear_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_mis_args(mode_id, strategy_count);
    TORCH_CHECK(strategy_count == 1, "endpoint connections support exactly one strategy");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    TORCH_CHECK(light_direction.size(0) == light_count, "light_direction must match light count");
    check_same_device(light_direction, light_origin, "light_direction");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&light_throughput_real, "light_throughput_real"),
             std::pair<const at::Tensor*, const char*>(&light_field_real, "light_field_real"),
             std::pair<const at::Tensor*, const char*>(&light_field_imag, "light_field_imag"),
             std::pair<const at::Tensor*, const char*>(&light_source_power, "light_source_power"),
             std::pair<const at::Tensor*, const char*>(&light_pdf_forward, "light_pdf_forward"),
             std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),
             std::pair<const at::Tensor*, const char*>(&light_component_mask, "light_component_mask"),
             std::pair<const at::Tensor*, const char*>(&light_tx_id, "light_tx_id"),
             std::pair<const at::Tensor*, const char*>(&light_valid, "light_valid"),
             std::pair<const at::Tensor*, const char*>(&light_path_length, "light_path_length"),
         }) {
        TORCH_CHECK(pair.first->size(0) == light_count, pair.second, " must match light count");
        check_same_device(*pair.first, light_origin, pair.second);
    }
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&sensor_pdf_reverse, "sensor_pdf_reverse"),
             std::pair<const at::Tensor*, const char*>(&sensor_field_real, "sensor_field_real"),
             std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),
             std::pair<const at::Tensor*, const char*>(&sensor_rx_id, "sensor_rx_id"),
             std::pair<const at::Tensor*, const char*>(&sensor_grid_linear_id, "sensor_grid_linear_id"),
             std::pair<const at::Tensor*, const char*>(&sensor_valid, "sensor_valid"),
         }) {
        TORCH_CHECK(pair.first->size(0) == sensor_count, pair.second, " must match sensor count");
        check_same_device(*pair.first, light_origin, pair.second);
    }
    check_same_device(sensor_origin, light_origin, "sensor_origin");
    const int64_t total = light_count * sensor_count;
    const int64_t count = max_paths < 0 ? total : std::min<int64_t>(max_paths, total);
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m] = allocate_connection_samples(light_origin, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_endpoint_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_pdf_forward.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_component_mask.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_pdf_reverse.data_ptr<float>(),
            sensor_depth.data_ptr<int>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_grid_linear_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            out_light_depth.data_ptr<int>(),
            out_sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m};
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_diffraction_connection_samples_from_tape_cuda(
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    check_bool_cuda(tape_active, "tape_active", 1);
    check_int_cuda(tape_state_idx, "tape_state_idx", 1);
    check_int_cuda(tape_cell, "tape_cell", 1);
    check_int_cuda(tape_material_idx, "tape_material_idx", 1);
    check_float_cuda(tape_edge_u, "tape_edge_u", 1);
    check_int_cuda(state_edge_index, "state_edge_index", 1);
    check_vec3_cuda(state_edge_pos, "state_edge_pos");
    check_vec3_cuda(state_edge_dir, "state_edge_dir");
    check_float_cuda(state_edge_t_min, "state_edge_t_min", 1);
    check_float_cuda(state_edge_t_max, "state_edge_t_max", 1);
    check_float_cuda(state_exterior_angle, "state_exterior_angle", 1);
    check_vec3_cuda(state_src, "state_src");
    check_float_cuda(state_src_power, "state_src_power", 1);
    check_float_cuda(material_gain, "material_gain", 1);
    check_bool_cuda(material_valid, "material_valid", 1);
    TORCH_CHECK(tx_index >= 0, "tx_index must be non-negative");
    TORCH_CHECK(state_count >= 0, "state_count must be non-negative");
    TORCH_CHECK(grid_axis >= 0 && grid_axis <= 2, "grid_axis must be 0, 1, or 2");
    TORCH_CHECK(grid_resolution0 > 0 && grid_resolution1 > 0, "grid resolutions must be positive");
    TORCH_CHECK(grid_cell_area > 0.0, "grid_cell_area must be positive");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(direct_samples >= 0 && keller_samples >= 0, "sample counts must be non-negative");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_diffraction_mis_args(mode_id, strategy_count, direct_samples, keller_samples);
    const int64_t count = tape_active.size(0);
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&tape_state_idx, "tape_state_idx"),
             std::pair<const at::Tensor*, const char*>(&tape_cell, "tape_cell"),
             std::pair<const at::Tensor*, const char*>(&tape_material_idx, "tape_material_idx"),
             std::pair<const at::Tensor*, const char*>(&tape_edge_u, "tape_edge_u"),
         }) {
        TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match tape_active");
        check_same_device(*pair.first, tape_active, pair.second);
    }
    const int64_t physical_state_count = state_edge_index.size(0);
    TORCH_CHECK(state_count <= physical_state_count, "state_count exceeds state payload width");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&state_edge_pos, "state_edge_pos"),
             std::pair<const at::Tensor*, const char*>(&state_edge_dir, "state_edge_dir"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_min, "state_edge_t_min"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_max, "state_edge_t_max"),
             std::pair<const at::Tensor*, const char*>(&state_exterior_angle, "state_exterior_angle"),
             std::pair<const at::Tensor*, const char*>(&state_src, "state_src"),
             std::pair<const at::Tensor*, const char*>(&state_src_power, "state_src_power"),
         }) {
        TORCH_CHECK(pair.first->size(0) >= state_count, pair.second, " must cover state_count");
        check_same_device(*pair.first, tape_active, pair.second);
    }
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    check_same_device(material_gain, tape_active, "material_gain");
    check_same_device(material_valid, tape_active, "material_valid");
    check_same_device(state_edge_index, tape_active, "state_edge_index");
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m] = allocate_connection_samples(tape_edge_u, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tape_active.get_device()).stream();
        bdpt_diffraction_connection_samples_from_tape_kernel<<<blocks, threads, 0, stream>>>(
            count,
            static_cast<int>(tx_index),
            static_cast<int>(state_count),
            static_cast<int>(grid_resolution0),
            static_cast<int>(grid_resolution1),
            static_cast<int>(grid_axis),
            static_cast<float>(grid_position),
            static_cast<float>(grid_coord0_min),
            static_cast<float>(grid_coord0_max),
            static_cast<float>(grid_coord1_min),
            static_cast<float>(grid_coord1_max),
            static_cast<float>(grid_cell_area),
            static_cast<float>(wavelength),
            static_cast<int>(direct_samples),
            static_cast<int>(keller_samples),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            static_cast<int>(material_gain.size(0)),
            tape_active.data_ptr<bool>(),
            tape_state_idx.data_ptr<int>(),
            tape_cell.data_ptr<int>(),
            tape_material_idx.data_ptr<int>(),
            tape_edge_u.data_ptr<float>(),
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_edge_t_min.data_ptr<float>(),
            state_edge_t_max.data_ptr<float>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            out_light_depth.data_ptr<int>(),
            out_sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m};
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_diffraction_point_connection_samples_cuda(
    at::Tensor rx_positions,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t seed,
    double wavelength,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    check_vec3_cuda(rx_positions, "rx_positions");
    check_int_cuda(state_edge_index, "state_edge_index", 1);
    check_vec3_cuda(state_edge_pos, "state_edge_pos");
    check_vec3_cuda(state_edge_dir, "state_edge_dir");
    check_float_cuda(state_edge_t_min, "state_edge_t_min", 1);
    check_float_cuda(state_edge_t_max, "state_edge_t_max", 1);
    check_int_cuda(state_prim0, "state_prim0", 1);
    check_int_cuda(state_prim1, "state_prim1", 1);
    check_float_cuda(state_exterior_angle, "state_exterior_angle", 1);
    check_vec3_cuda(state_src, "state_src");
    check_float_cuda(state_src_power, "state_src_power", 1);
    check_float_cuda(material_gain, "material_gain", 1);
    check_bool_cuda(material_valid, "material_valid", 1);
    TORCH_CHECK(tx_index >= 0, "tx_index must be non-negative");
    TORCH_CHECK(state_count >= 0, "state_count must be non-negative");
    TORCH_CHECK(direct_samples >= 0 && keller_samples >= 0, "diffraction sample counts must be non-negative");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_diffraction_mis_args(mode_id, strategy_count, direct_samples, keller_samples);
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    const int64_t physical_state_count = state_edge_index.size(0);
    TORCH_CHECK(state_count <= physical_state_count, "state_count exceeds state tensors");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&state_edge_pos, "state_edge_pos"),
             std::pair<const at::Tensor*, const char*>(&state_edge_dir, "state_edge_dir"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_min, "state_edge_t_min"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_max, "state_edge_t_max"),
             std::pair<const at::Tensor*, const char*>(&state_prim0, "state_prim0"),
             std::pair<const at::Tensor*, const char*>(&state_prim1, "state_prim1"),
             std::pair<const at::Tensor*, const char*>(&state_exterior_angle, "state_exterior_angle"),
             std::pair<const at::Tensor*, const char*>(&state_src, "state_src"),
             std::pair<const at::Tensor*, const char*>(&state_src_power, "state_src_power"),
         }) {
        TORCH_CHECK(pair.first->size(0) == physical_state_count, pair.second, " must match state_edge_index");
        check_same_device(*pair.first, rx_positions, pair.second);
    }
    check_same_device(state_edge_index, rx_positions, "state_edge_index");
    check_same_device(material_gain, rx_positions, "material_gain");
    check_same_device(material_valid, rx_positions, "material_valid");
    const int64_t rx_count = rx_positions.size(0);
    const int64_t samples_per_rx = direct_samples + keller_samples;
    const int64_t count = state_count > 0 ? rx_count * samples_per_rx : 0;
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m] = allocate_connection_samples(rx_positions, count);
    auto source_start = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto source_end = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto target_start = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto target_end = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto visibility_active = at::empty({count}, rx_positions.options().dtype(at::kBool));
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
        bdpt_diffraction_point_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            static_cast<int>(tx_index),
            static_cast<int>(state_count),
            static_cast<int>(rx_count),
            static_cast<float>(wavelength),
            static_cast<int>(direct_samples),
            static_cast<int>(keller_samples),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            static_cast<int>(material_gain.size(0)),
            static_cast<unsigned long long>(seed),
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_edge_t_min.data_ptr<float>(),
            state_edge_t_max.data_ptr<float>(),
            state_prim0.data_ptr<int>(),
            state_prim1.data_ptr<int>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            out_light_depth.data_ptr<int>(),
            out_sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>(),
            source_start.data_ptr<float>(),
            source_end.data_ptr<float>(),
            target_start.data_ptr<float>(),
            target_end.data_ptr<float>(),
            visibility_active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m,
        source_start,
        source_end,
        target_start,
        target_end,
        visibility_active};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_bdpt_endpoint_connection_visibility_inputs_cuda(
    at::Tensor light_origin,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor sensor_origin,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    int64_t sample_count) {
    check_vec3_cuda(light_origin, "light_origin");
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    TORCH_CHECK(light_tx_id.size(0) == light_count, "light_tx_id must match light count");
    TORCH_CHECK(light_valid.size(0) == light_count, "light_valid must match light count");
    TORCH_CHECK(sensor_rx_id.size(0) == sensor_count, "sensor_rx_id must match sensor count");
    TORCH_CHECK(sensor_valid.size(0) == sensor_count, "sensor_valid must match sensor count");
    check_same_device(light_tx_id, light_origin, "light_tx_id");
    check_same_device(light_valid, light_origin, "light_valid");
    check_same_device(sensor_origin, light_origin, "sensor_origin");
    check_same_device(sensor_rx_id, light_origin, "sensor_rx_id");
    check_same_device(sensor_valid, light_origin, "sensor_valid");
    TORCH_CHECK(
        sensor_count > 0 || sample_count == 0,
        "sensor count must be positive when sample_count is positive");
    TORCH_CHECK(sample_count <= light_count * sensor_count, "sample_count exceeds endpoint pair count");
    auto start = at::empty({sample_count, 3}, light_origin.options());
    auto end = at::empty({sample_count, 3}, light_origin.options());
    auto active = at::empty({sample_count}, light_origin.options().dtype(at::kBool));
    if (sample_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((sample_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_endpoint_connection_visibility_inputs_kernel<<<blocks, threads, 0, stream>>>(
            sample_count,
            sensor_count,
            light_origin.data_ptr<float>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            sensor_origin.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            start.data_ptr<float>(),
            end.data_ptr<float>(),
            active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {start, end, active};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(tx_id.sizes() == contribution.sizes(), "tx_id must match contribution");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(rx_id.sizes() == contribution.sizes(), "rx_id must match contribution");
    TORCH_CHECK(component_id.sizes() == contribution.sizes(), "component_id must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    check_same_device(tx_id, contribution, "tx_id");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(rx_id, contribution, "rx_id");
    check_same_device(component_id, contribution, "component_id");
    check_same_device(valid, contribution, "valid");
    TORCH_CHECK(accumulation_strategy >= 0 && accumulation_strategy <= 2, "accumulation_strategy must be 0, 1, or 2");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto path_gain = at::empty({tx_count, rx_count}, float_options);
    auto los = at::empty({tx_count, rx_count}, float_options);
    auto reflection = at::empty({tx_count, rx_count}, float_options);
    auto diffraction = at::empty({tx_count, rx_count}, float_options);
    auto transmission = at::empty({tx_count, rx_count}, float_options);
    auto scattering = at::empty({tx_count, rx_count}, float_options);
    const int64_t count = contribution.numel();
    const int64_t out_count = tx_count * rx_count;
    if (accumulation_strategy == 1) {
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_accumulate_connection_samples_staged_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                contribution.data_ptr<float>(),
                mis_weight.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {path_gain, los, reflection, diffraction, transmission, scattering};
    }
    if (accumulation_strategy == 2) {
        zero_float_tensor(path_gain);
        zero_float_tensor(los);
        zero_float_tensor(reflection);
        zero_float_tensor(diffraction);
        zero_float_tensor(transmission);
        zero_float_tensor(scattering);
        auto int_options = tx_id.options().dtype(at::kInt);
        auto compact_count = at::empty({}, int_options);
        auto compact_indices = at::empty({count}, int_options);
        zero_int_tensor(compact_count);
        if (count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_compact_valid_connection_indices_kernel<<<blocks, threads, 0, stream>>>(
                count,
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                tx_count,
                rx_count,
                compact_count.data_ptr<int>(),
                compact_indices.data_ptr<int>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            bdpt_accumulate_connection_samples_compacted_kernel<<<blocks, threads, 0, stream>>>(
                count,
                compact_count.data_ptr<int>(),
                compact_indices.data_ptr<int>(),
                contribution.data_ptr<float>(),
                mis_weight.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                rx_count,
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {path_gain, los, reflection, diffraction, transmission, scattering};
    }
    auto double_options = contribution.options().dtype(at::kDouble);
    auto path_gain_sum = at::empty({tx_count, rx_count}, double_options);
    auto los_sum = at::empty({tx_count, rx_count}, double_options);
    auto reflection_sum = at::empty({tx_count, rx_count}, double_options);
    auto diffraction_sum = at::empty({tx_count, rx_count}, double_options);
    auto transmission_sum = at::empty({tx_count, rx_count}, double_options);
    auto scattering_sum = at::empty({tx_count, rx_count}, double_options);
    zero_double_tensor(path_gain_sum);
    zero_double_tensor(los_sum);
    zero_double_tensor(reflection_sum);
    zero_double_tensor(diffraction_sum);
    zero_double_tensor(transmission_sum);
    zero_double_tensor(scattering_sum);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_accumulate_connection_samples_double_kernel<<<blocks, threads, 0, stream>>>(
            count,
            contribution.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_count,
            rx_count,
            path_gain_sum.data_ptr<double>(),
            los_sum.data_ptr<double>(),
            reflection_sum.data_ptr<double>(),
            diffraction_sum.data_ptr<double>(),
            transmission_sum.data_ptr<double>(),
            scattering_sum.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (out_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((out_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_cast_connection_accumulation_kernel<<<blocks, threads, 0, stream>>>(
            out_count,
            path_gain_sum.data_ptr<double>(),
            los_sum.data_ptr<double>(),
            reflection_sum.data_ptr<double>(),
            diffraction_sum.data_ptr<double>(),
            transmission_sum.data_ptr<double>(),
            scattering_sum.data_ptr<double>(),
            path_gain.data_ptr<float>(),
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            transmission.data_ptr<float>(),
            scattering.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {path_gain, los, reflection, diffraction, transmission, scattering};
}

void cn_bdpt_filter_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor valid,
    at::Tensor visible) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(pdf, "pdf", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_bool_cuda(valid, "valid", 1);
    check_bool_cuda(visible, "visible", 1);
    TORCH_CHECK(pdf.sizes() == contribution.sizes(), "pdf must match contribution");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    TORCH_CHECK(visible.sizes() == contribution.sizes(), "visible must match contribution");
    check_same_device(pdf, contribution, "pdf");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(valid, contribution, "valid");
    check_same_device(visible, contribution, "visible");
    const int64_t count = contribution.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_filter_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            visible.data_ptr<bool>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            valid.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

int64_t cn_bdpt_count_valid_connection_samples_cuda(at::Tensor valid) {
    check_bool_cuda(valid, "valid", 1);
    const int64_t count = valid.size(0);
    int valid_count_host = 0;
    if (count > 0) {
        auto compact_count = at::empty({}, valid.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            valid.data_ptr<bool>(),
            compact_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &valid_count_host,
            compact_count.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return static_cast<int64_t>(valid_count_host);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_compact_connection_samples_cuda(
    at::Tensor topology,
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor grid_linear_id,
    at::Tensor light_depth,
    at::Tensor sensor_depth,
    at::Tensor path_length_m,
    int64_t max_paths) {
    check_int_cuda(topology, "topology", 2);
    TORCH_CHECK(topology.size(1) == 4, "topology must have shape (N, 4)");
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(pdf, "pdf", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(grid_linear_id, "grid_linear_id", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(sensor_depth, "sensor_depth", 1);
    check_float_cuda(path_length_m, "path_length_m", 1);
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t count = contribution.size(0);
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&pdf, "pdf"),
             std::pair<const at::Tensor*, const char*>(&mis_weight, "mis_weight"),
             std::pair<const at::Tensor*, const char*>(&component_id, "component_id"),
             std::pair<const at::Tensor*, const char*>(&valid, "valid"),
             std::pair<const at::Tensor*, const char*>(&tx_id, "tx_id"),
             std::pair<const at::Tensor*, const char*>(&rx_id, "rx_id"),
             std::pair<const at::Tensor*, const char*>(&grid_linear_id, "grid_linear_id"),
             std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),
             std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),
             std::pair<const at::Tensor*, const char*>(&path_length_m, "path_length_m"),
         }) {
        TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match contribution");
        check_same_device(*pair.first, contribution, pair.second);
    }
    TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
    check_same_device(topology, contribution, "topology");
    int valid_count_host = 0;
    if (count > 0) {
        auto compact_count = at::empty({}, contribution.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            valid.data_ptr<bool>(),
            compact_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &valid_count_host,
            compact_count.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const int64_t capacity = max_paths < 0
        ? static_cast<int64_t>(valid_count_host)
        : std::min<int64_t>(max_paths, static_cast<int64_t>(valid_count_host));
    auto [
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m] = allocate_connection_samples(contribution, capacity);
    if (capacity > 0 && count > 0) {
        auto compact_count = at::empty({}, contribution.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_compact_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            capacity,
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            light_depth.data_ptr<int>(),
            sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>(),
            compact_count.data_ptr<int>(),
            out_topology.data_ptr<int>(),
            out_contribution.data_ptr<float>(),
            out_pdf.data_ptr<float>(),
            out_mis_weight.data_ptr<float>(),
            out_component_id.data_ptr<int>(),
            out_valid.data_ptr<bool>(),
            out_tx_id.data_ptr<int>(),
            out_rx_id.data_ptr<int>(),
            out_grid_linear_id.data_ptr<int>(),
            out_light_depth.data_ptr<int>(),
            out_sensor_depth.data_ptr<int>(),
            out_path_length_m.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m};
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_concat_connection_samples_cuda(
    std::vector<at::Tensor> topologies,
    std::vector<at::Tensor> contributions,
    std::vector<at::Tensor> pdfs,
    std::vector<at::Tensor> mis_weights,
    std::vector<at::Tensor> component_ids,
    std::vector<at::Tensor> valids,
    std::vector<at::Tensor> tx_ids,
    std::vector<at::Tensor> rx_ids,
    std::vector<at::Tensor> grid_linear_ids,
    std::vector<at::Tensor> light_depths,
    std::vector<at::Tensor> sensor_depths,
    std::vector<at::Tensor> path_lengths_m) {
    const size_t block_count = contributions.size();
    TORCH_CHECK(block_count > 0, "bdpt_concat_connection_samples requires at least one block");
    TORCH_CHECK(topologies.size() == block_count, "topologies must match block count");
    TORCH_CHECK(pdfs.size() == block_count, "pdfs must match block count");
    TORCH_CHECK(mis_weights.size() == block_count, "mis_weights must match block count");
    TORCH_CHECK(component_ids.size() == block_count, "component_ids must match block count");
    TORCH_CHECK(valids.size() == block_count, "valids must match block count");
    TORCH_CHECK(tx_ids.size() == block_count, "tx_ids must match block count");
    TORCH_CHECK(rx_ids.size() == block_count, "rx_ids must match block count");
    TORCH_CHECK(grid_linear_ids.size() == block_count, "grid_linear_ids must match block count");
    TORCH_CHECK(light_depths.size() == block_count, "light_depths must match block count");
    TORCH_CHECK(sensor_depths.size() == block_count, "sensor_depths must match block count");
    TORCH_CHECK(path_lengths_m.size() == block_count, "path_lengths_m must match block count");
    const at::Tensor& reference = contributions[0];
    check_float_cuda(reference, "contribution[0]", 1);
    int64_t total = 0;
    for (size_t block = 0; block < block_count; ++block) {
        at::Tensor& topology = topologies[block];
        at::Tensor& contribution = contributions[block];
        at::Tensor& pdf = pdfs[block];
        at::Tensor& mis_weight = mis_weights[block];
        at::Tensor& component_id = component_ids[block];
        at::Tensor& valid = valids[block];
        at::Tensor& tx_id = tx_ids[block];
        at::Tensor& rx_id = rx_ids[block];
        at::Tensor& grid_linear_id = grid_linear_ids[block];
        at::Tensor& light_depth = light_depths[block];
        at::Tensor& sensor_depth = sensor_depths[block];
        at::Tensor& path_length_m = path_lengths_m[block];
        check_int_cuda(topology, "topology", 2);
        TORCH_CHECK(topology.size(1) == 4, "topology must have shape (N, 4)");
        check_float_cuda(contribution, "contribution", 1);
        check_float_cuda(pdf, "pdf", 1);
        check_float_cuda(mis_weight, "mis_weight", 1);
        check_int_cuda(component_id, "component_id", 1);
        check_bool_cuda(valid, "valid", 1);
        check_int_cuda(tx_id, "tx_id", 1);
        check_int_cuda(rx_id, "rx_id", 1);
        check_int_cuda(grid_linear_id, "grid_linear_id", 1);
        check_int_cuda(light_depth, "light_depth", 1);
        check_int_cuda(sensor_depth, "sensor_depth", 1);
        check_float_cuda(path_length_m, "path_length_m", 1);
        const int64_t count = contribution.size(0);
        TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
        check_same_device(contribution, reference, "contribution");
        for (const auto& pair : {
                 std::pair<const at::Tensor*, const char*>(&topology, "topology"),
                 std::pair<const at::Tensor*, const char*>(&pdf, "pdf"),
                 std::pair<const at::Tensor*, const char*>(&mis_weight, "mis_weight"),
                 std::pair<const at::Tensor*, const char*>(&component_id, "component_id"),
                 std::pair<const at::Tensor*, const char*>(&valid, "valid"),
                 std::pair<const at::Tensor*, const char*>(&tx_id, "tx_id"),
                 std::pair<const at::Tensor*, const char*>(&rx_id, "rx_id"),
                 std::pair<const at::Tensor*, const char*>(&grid_linear_id, "grid_linear_id"),
                 std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),
                 std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),
                 std::pair<const at::Tensor*, const char*>(&path_length_m, "path_length_m"),
             }) {
            TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match contribution");
            check_same_device(*pair.first, reference, pair.second);
        }
        total += count;
    }
    auto [
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m] = allocate_connection_samples(reference, total);
    int64_t offset = 0;
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    for (size_t block = 0; block < block_count; ++block) {
        const int64_t count = contributions[block].size(0);
        if (count > 0) {
            int grid = static_cast<int>((count + threads - 1) / threads);
            bdpt_copy_connection_samples_kernel<<<grid, threads, 0, stream>>>(
                count,
                offset,
                topologies[block].data_ptr<int>(),
                contributions[block].data_ptr<float>(),
                pdfs[block].data_ptr<float>(),
                mis_weights[block].data_ptr<float>(),
                component_ids[block].data_ptr<int>(),
                valids[block].data_ptr<bool>(),
                tx_ids[block].data_ptr<int>(),
                rx_ids[block].data_ptr<int>(),
                grid_linear_ids[block].data_ptr<int>(),
                light_depths[block].data_ptr<int>(),
                sensor_depths[block].data_ptr<int>(),
                path_lengths_m[block].data_ptr<float>(),
                out_topology.data_ptr<int>(),
                out_contribution.data_ptr<float>(),
                out_pdf.data_ptr<float>(),
                out_mis_weight.data_ptr<float>(),
                out_component_id.data_ptr<int>(),
                out_valid.data_ptr<bool>(),
                out_tx_id.data_ptr<int>(),
                out_rx_id.data_ptr<int>(),
                out_grid_linear_id.data_ptr<int>(),
                out_light_depth.data_ptr<int>(),
                out_sensor_depth.data_ptr<int>(),
                out_path_length_m.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        offset += count;
    }
    return {
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m};
}

at::Tensor cn_bdpt_connection_variance_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(tx_id.sizes() == contribution.sizes(), "tx_id must match contribution");
    TORCH_CHECK(rx_id.sizes() == contribution.sizes(), "rx_id must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(tx_id, contribution, "tx_id");
    check_same_device(rx_id, contribution, "rx_id");
    check_same_device(valid, contribution, "valid");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto double_options = contribution.options().dtype(at::kDouble);
    auto int_options = contribution.options().dtype(at::kInt);
    auto sum = at::empty({tx_count, rx_count}, double_options);
    auto sum_square_unweighted = at::empty({tx_count, rx_count}, double_options);
    auto sample_count = at::empty({tx_count, rx_count}, int_options);
    auto variance = at::empty({tx_count, rx_count}, float_options);
    zero_double_tensor(sum);
    zero_double_tensor(sum_square_unweighted);
    zero_int_tensor(sample_count);
    const int64_t in_count = contribution.numel();
    if (in_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((in_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_connection_variance_accum_double_kernel<<<blocks, threads, 0, stream>>>(
            in_count,
            contribution.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            rx_count,
            static_cast<double>(samples_per_tx),
            sum.data_ptr<double>(),
            sum_square_unweighted.data_ptr<double>(),
            sample_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t out_count = tx_count * rx_count;
    if (out_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((out_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_connection_variance_finalize_double_kernel<<<blocks, threads, 0, stream>>>(
            out_count,
            sum.data_ptr<double>(),
            sum_square_unweighted.data_ptr<double>(),
            sample_count.data_ptr<int>(),
            static_cast<double>(samples_per_tx),
            variance.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return variance;
}
