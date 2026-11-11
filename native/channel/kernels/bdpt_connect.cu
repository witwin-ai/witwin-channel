// Copyright Xingyu Chen.
// Implements BDPT connection CUDA operations.

// ==== Section: BDPT MIS ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"

#include <rayd/field_transport.cuh>

#include <cub/cub.cuh>

#include "bdpt_wavefront.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace utd = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;

constexpr float kLightSpeedMPerS = 299792458.0f;
constexpr float kPi = 3.14159265358979323846f;

// BDPT component_mask bits (component classification): 1=los, 2=reflection,
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

// ==== Section: BDPT connection samples ====

namespace {

struct BdptConnectionGeometry {
    float dx;
    float dy;
    float dz;
    float distance;
    bool direction_valid;
};

// Connection segment from a light vertex to a sensor. A vertex past the
// endpoint (depth > 0) connects only into the half space of its outgoing
// direction.
__device__ __forceinline__ BdptConnectionGeometry bdpt_connection_geometry(
    const float* light_origin,
    const float* light_direction,
    int light_path_depth,
    const float* sensor_origin) {
    const float dx = sensor_origin[0] - light_origin[0];
    const float dy = sensor_origin[1] - light_origin[1];
    const float dz = sensor_origin[2] - light_origin[2];
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1.0e-6f);
    const float dir_dot = light_path_depth > 0
        ? (dx * light_direction[0] + dy * light_direction[1] + dz * light_direction[2]) / distance
        : 1.0f;
    return {dx, dy, dz, distance, dir_dot > 0.0f};
}

struct BdptConnectionRow {
    bool valid;
    float pdf;
    float contribution;
    float total_distance;
};

// One light-vertex to sensor connection row: the projected received power of
// the vertex Jones field over the unfolded path, weighted by the source power
// and 1/samples_per_tx. ``endpoints_valid`` carries the vertex/sensor validity.
__device__ BdptConnectionRow bdpt_connection_row(
    const float* light_origin,
    const float* light_direction,
    int light_path_depth,
    const float* light_field_real,
    const float* light_field_imag,
    float light_source_power,
    float light_pdf_forward,
    float light_path_length,
    const float* sensor_origin,
    const float* sensor_polarization,
    float sensor_pdf_reverse,
    bool endpoints_valid,
    float frequency_hz,
    float inv_samples_per_tx) {
    const BdptConnectionGeometry geometry =
        bdpt_connection_geometry(light_origin, light_direction, light_path_depth, sensor_origin);
    const float dx = geometry.dx;
    const float dy = geometry.dy;
    const float dz = geometry.dz;
    const float distance = geometry.distance;
    const bool row_valid = endpoints_valid && geometry.direction_valid;
    // Proposal density excludes free-space geometry. The deterministic
    // endpoint connection has unit discrete mass; inverse-square spreading
    // belongs to the contribution, not to the sampling PDF.
    const float row_pdf = row_valid
        ? fmaxf(light_pdf_forward, 0.0f) * fmaxf(sensor_pdf_reverse, 0.0f)
        : 0.0f;
    // The free-space spreading acts over the unfolded path (light-subpath
    // prefix + connection segment), not the last segment alone.
    const float total_distance = distance + fmaxf(light_path_length, 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float amplitude = 1.0f /
        (2.0f * fmaxf(wave_number, 1.0e-12f) * fmaxf(total_distance, 1.0e-6f));
    const utd::Complex propagation = utd::cplx_mul_real(
        utd::cplx_exp_phase(transport::precise_neg_kd(wave_number, total_distance)),
        amplitude);
    const utd::Complex3 incident_field = {
        utd::cplx(light_field_real[0], light_field_imag[0]),
        utd::cplx(light_field_real[1], light_field_imag[1]),
        utd::cplx(light_field_real[2], light_field_imag[2])};
    const utd::Complex3 received_field = utd::c3_scale(incident_field, propagation);
    const utd::float3a connection_direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    const utd::float3a receiver_polarization = utd::make_f3(
        sensor_polarization[0], sensor_polarization[1], sensor_polarization[2]);
    const utd::Complex coefficient = transport::project_receiver(
        received_field, connection_direction, receiver_polarization);
    const float coefficient_power = utd::cplx_abs_sqr(coefficient);
    const float row_contribution = row_valid
        ? light_source_power * coefficient_power * inv_samples_per_tx
        : 0.0f;
    return {row_valid, row_pdf, row_contribution, total_distance};
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

    const BdptConnectionRow connection = bdpt_connection_row(
        light_origin + light_index * 3,
        light_direction + light_index * 3,
        light_depth[light_index],
        light_field_real + light_index * 3,
        light_field_imag + light_index * 3,
        light_source_power[light_index],
        light_pdf_forward[light_index],
        light_path_length[light_index],
        sensor_origin + sensor_index * 3,
        sensor_field_real + sensor_index * 3,
        sensor_pdf_reverse[sensor_index],
        is_valid,
        frequency_hz,
        inv_samples_per_tx);
    const bool row_valid = connection.valid;
    const float row_pdf = connection.pdf;
    const float row_contribution = connection.contribution;
    const float total_distance = connection.total_distance;

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
}  // namespace

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
channel_bdpt_endpoint_connection_samples_cuda(
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

// ==== Section: BDPT visibility ====

#define CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS()                                     \
    check_int_cuda(topology, "topology", 2);                                          \
    TORCH_CHECK(topology.size(1) == 4, "topology must have shape (N, 4)");             \
    check_float_cuda(contribution, "contribution", 1);                                \
    check_float_cuda(pdf, "pdf", 1);                                                  \
    check_float_cuda(mis_weight, "mis_weight", 1);                                    \
    check_int_cuda(component_id, "component_id", 1);                                  \
    check_bool_cuda(valid, "valid", 1);                                               \
    check_int_cuda(tx_id, "tx_id", 1);                                                \
    check_int_cuda(rx_id, "rx_id", 1);                                                \
    check_int_cuda(grid_linear_id, "grid_linear_id", 1);                              \
    check_int_cuda(light_depth, "light_depth", 1);                                    \
    check_int_cuda(sensor_depth, "sensor_depth", 1);                                  \
    check_float_cuda(path_length_m, "path_length_m", 1)

#define CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(REFERENCE)                               \
    for (const auto& pair : {                                                          \
             std::pair<const at::Tensor*, const char*>(&pdf, "pdf"),                  \
             std::pair<const at::Tensor*, const char*>(&mis_weight, "mis_weight"),    \
             std::pair<const at::Tensor*, const char*>(&component_id, "component_id"),\
             std::pair<const at::Tensor*, const char*>(&valid, "valid"),              \
             std::pair<const at::Tensor*, const char*>(&tx_id, "tx_id"),              \
             std::pair<const at::Tensor*, const char*>(&rx_id, "rx_id"),              \
             std::pair<const at::Tensor*, const char*>(&grid_linear_id, "grid_linear_id"),\
             std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),  \
             std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),\
             std::pair<const at::Tensor*, const char*>(&path_length_m, "path_length_m"),\
         }) {                                                                          \
        TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match contribution");\
        check_same_device(*pair.first, REFERENCE, pair.second);                        \
    }

#define CHANNEL_BDPT_CONNECTION_OUTPUT_POINTERS()                                           \
    out_topology.data_ptr<int>(),                                                      \
    out_contribution.data_ptr<float>(),                                                \
    out_pdf.data_ptr<float>(),                                                         \
    out_mis_weight.data_ptr<float>(),                                                  \
    out_component_id.data_ptr<int>(),                                                  \
    out_valid.data_ptr<bool>(),                                                        \
    out_tx_id.data_ptr<int>(),                                                         \
    out_rx_id.data_ptr<int>(),                                                         \
    out_grid_linear_id.data_ptr<int>(),                                                \
    out_light_depth.data_ptr<int>(),                                                   \
    out_sensor_depth.data_ptr<int>(),                                                  \
    out_path_length_m.data_ptr<float>()

namespace {

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

template <typename Counter>
__global__ void bdpt_count_valid_connection_samples_kernel(
    int64_t count,
    const bool* valid,
    Counter* compact_count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    atomicAdd(compact_count, Counter{1});
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_bdpt_endpoint_connection_visibility_inputs_cuda(
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

void channel_bdpt_filter_connection_samples_cuda(
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

at::Tensor channel_bdpt_count_valid_connection_samples_cuda(at::Tensor valid) {
    check_bool_cuda(valid, "valid", 1);
    const int64_t count = valid.size(0);
    // The count stays on the device; the caller reads it with the solve's other
    // metadata counts in one copy.
    auto valid_count = at::empty({}, valid.options().dtype(at::kLong));
    C10_CUDA_CHECK(cudaMemsetAsync(
        valid_count.data_ptr(), 0, sizeof(int64_t),
        at::cuda::getCurrentCUDAStream(valid.get_device()).stream()));
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<unsigned long long><<<blocks, threads, 0, stream>>>(
            count,
            valid.data_ptr<bool>(),
            reinterpret_cast<unsigned long long*>(valid_count.data_ptr<int64_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return valid_count;
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
channel_bdpt_compact_connection_samples_cuda(
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
    CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS();
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t count = contribution.size(0);
    CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(contribution);
    TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
    check_same_device(topology, contribution, "topology");
    int valid_count_host = 0;
    if (count > 0) {
        auto compact_count = at::empty({}, contribution.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<int><<<blocks, threads, 0, stream>>>(
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
            CHANNEL_BDPT_CONNECTION_OUTPUT_POINTERS());
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
channel_bdpt_concat_connection_samples_cuda(
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
        CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS();
        const int64_t count = contribution.size(0);
        TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
        check_same_device(contribution, reference, "contribution");
        TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
        check_same_device(topology, reference, "topology");
        CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(reference);
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
                CHANNEL_BDPT_CONNECTION_OUTPUT_POINTERS());
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

#undef CHANNEL_BDPT_CONNECTION_OUTPUT_POINTERS
#undef CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_ROWS
#undef CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS

// ==== Section: BDPT connection accumulation ====

namespace {

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

// coherent combination (opt-in, DEFAULT OFF). Sum the complex per-row
// projected field coefficient into per-(tx, rx, component) phasor bins, then
// finalize |sum|^2. Coherent-eligible rows are the enumerated delta/UTD
// discrete connections (los / reflection / diffraction / coupled->diffraction)
// which carry unit forward/reverse mass, so the phasor is summed with UNIT
// weight (mis_weight is identically 1 for those rows) and the estimate is
// MIS-invariant by construction. This path is only reached when
// combine_domain == 1; combine_domain == 0 keeps the power-domain incoherent
// accumulation bit-identical and never touches the coefficient buffers.
__global__ void bdpt_accumulate_connection_samples_coherent_kernel(
    int64_t count,
    const float* coeff_real,
    const float* coeff_imag,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    double* los_real,
    double* los_imag,
    double* reflection_real,
    double* reflection_imag,
    double* diffraction_real,
    double* diffraction_imag,
    double* transmission_real,
    double* transmission_imag,
    double* scattering_real,
    double* scattering_imag) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count ||
        !bdpt_component_accumulable(component)) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double re = static_cast<double>(coeff_real[index]);
    const double im = static_cast<double>(coeff_imag[index]);
    if (component == kComponentLos) {
        atomicAdd(los_real + out_index, re);
        atomicAdd(los_imag + out_index, im);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection_real + out_index, re);
        atomicAdd(reflection_imag + out_index, im);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction_real + out_index, re);
        atomicAdd(diffraction_imag + out_index, im);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission_real + out_index, re);
        atomicAdd(transmission_imag + out_index, im);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering_real + out_index, re);
        atomicAdd(scattering_imag + out_index, im);
    }
}

__global__ void bdpt_finalize_coherent_accumulation_kernel(
    int64_t count,
    const double* los_real,
    const double* los_imag,
    const double* reflection_real,
    const double* reflection_imag,
    const double* diffraction_real,
    const double* diffraction_imag,
    const double* transmission_real,
    const double* transmission_imag,
    const double* scattering_real,
    const double* scattering_imag,
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
    const double los_power =
        los_real[index] * los_real[index] + los_imag[index] * los_imag[index];
    const double reflection_power =
        reflection_real[index] * reflection_real[index] +
        reflection_imag[index] * reflection_imag[index];
    const double diffraction_power =
        diffraction_real[index] * diffraction_real[index] +
        diffraction_imag[index] * diffraction_imag[index];
    const double transmission_power =
        transmission_real[index] * transmission_real[index] +
        transmission_imag[index] * transmission_imag[index];
    const double scattering_power =
        scattering_real[index] * scattering_real[index] +
        scattering_imag[index] * scattering_imag[index];
    // Paths within one component combine coherently; components combine
    // incoherently into path_gain (matches the deterministic per-component
    // coherent power the coherent combination acceptance gate compares against).
    los[index] = static_cast<float>(los_power);
    reflection[index] = static_cast<float>(reflection_power);
    diffraction[index] = static_cast<float>(diffraction_power);
    transmission[index] = static_cast<float>(transmission_power);
    scattering[index] = static_cast<float>(scattering_power);
    path_gain[index] = static_cast<float>(
        los_power + reflection_power + diffraction_power + transmission_power +
        scattering_power);
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

}  // namespace

// ==== Section: BDPT sorted segmented reduction ====
//
// Deterministic per-(tx, rx, component) sums. Every accumulable row gets the
// key (tx * rx_count + rx) * 5 + slot (slot = los, reflection, diffraction,
// transmission, scattering); other rows get the sentinel 5 * cells. A stable
// radix sort orders rows by key (rows of one key keep their input order) and
// a reduce-by-key sums each key's run, so every bin is the same fixed-order
// sum on every run, in O(rows log rows) instead of O(cells x rows).

namespace {

constexpr int kSortedSlots = 5;

__device__ __forceinline__ int bdpt_component_slot(int component) {
    if (component == kComponentLos) return 0;
    if (component == kComponentReflection) return 1;
    if (component == kComponentDiffraction) return 2;
    if (component == kComponentTransmission) return 3;
    if (component == kComponentScattering) return 4;
    return -1;
}

__global__ void bdpt_sorted_keys_kernel(
    int64_t count,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    unsigned long long* keys,
    int* rows) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned long long sentinel =
        static_cast<unsigned long long>(tx_count * rx_count * kSortedSlots);
    unsigned long long key = sentinel;
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int slot = bdpt_component_slot(component_id[index]);
    if (valid[index] && tx >= 0 && tx < tx_count && rx >= 0 && rx < rx_count && slot >= 0) {
        key = static_cast<unsigned long long>((static_cast<int64_t>(tx) * rx_count + rx) * kSortedSlots + slot);
    }
    keys[index] = key;
    rows[index] = static_cast<int>(index);
}

// sorted_values[i] = first[row] * second[row] (second may be null) in sorted order.
__global__ void bdpt_sorted_gather_kernel(
    int64_t count, const int* rows, const float* first, const float* second, double* sorted_values) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int row = rows[index];
    const double value = static_cast<double>(first[row]);
    sorted_values[index] = second == nullptr ? value : value * static_cast<double>(second[row]);
}

__global__ void bdpt_sorted_scatter_kernel(
    int64_t capacity,
    const int* run_count,
    const unsigned long long* unique_keys,
    const double* run_sums,
    unsigned long long bins_count,
    double* bins) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= capacity || index >= static_cast<int64_t>(run_count[0])) {
        return;
    }
    const unsigned long long key = unique_keys[index];
    if (key < bins_count) {
        bins[key] = run_sums[index];
    }
}

__global__ void bdpt_sorted_power_finalize_kernel(
    int64_t cells,
    const double* bins,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= cells) {
        return;
    }
    const double* bin = bins + cell * kSortedSlots;
    los[cell] = static_cast<float>(bin[0]);
    reflection[cell] = static_cast<float>(bin[1]);
    diffraction[cell] = static_cast<float>(bin[2]);
    transmission[cell] = static_cast<float>(bin[3]);
    scattering[cell] = static_cast<float>(bin[4]);
    path_gain[cell] = static_cast<float>(bin[0] + bin[1] + bin[2] + bin[3] + bin[4]);
}

// Coherent tangent power t_P_c = 2 Re(conj(S_c) t_S_c) per component.
__global__ void bdpt_sorted_coherent_jvp_finalize_kernel(
    int64_t cells,
    const double* bins_re,
    const double* bins_im,
    const double* los_re,
    const double* los_im,
    const double* reflection_re,
    const double* reflection_im,
    const double* diffraction_re,
    const double* diffraction_im,
    const double* transmission_re,
    const double* transmission_im,
    const double* scattering_re,
    const double* scattering_im,
    float* t_path_gain,
    float* t_los,
    float* t_reflection,
    float* t_diffraction,
    float* t_transmission,
    float* t_scattering) {
    const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= cells) {
        return;
    }
    const double* re = bins_re + cell * kSortedSlots;
    const double* im = bins_im + cell * kSortedSlots;
    const double tp_los = 2.0 * (los_re[cell] * re[0] + los_im[cell] * im[0]);
    const double tp_reflection = 2.0 * (reflection_re[cell] * re[1] + reflection_im[cell] * im[1]);
    const double tp_diffraction = 2.0 * (diffraction_re[cell] * re[2] + diffraction_im[cell] * im[2]);
    const double tp_transmission = 2.0 * (transmission_re[cell] * re[3] + transmission_im[cell] * im[3]);
    const double tp_scattering = 2.0 * (scattering_re[cell] * re[4] + scattering_im[cell] * im[4]);
    t_los[cell] = static_cast<float>(tp_los);
    t_reflection[cell] = static_cast<float>(tp_reflection);
    t_diffraction[cell] = static_cast<float>(tp_diffraction);
    t_transmission[cell] = static_cast<float>(tp_transmission);
    t_scattering[cell] = static_cast<float>(tp_scattering);
    t_path_gain[cell] = static_cast<float>(
        tp_los + tp_reflection + tp_diffraction + tp_transmission + tp_scattering);
}

int sorted_key_bits(unsigned long long sentinel) {
    int bits = 1;
    while (bits < 64 && (sentinel >> bits) != 0ULL) {
        ++bits;
    }
    return bits;
}

// Rows stably sorted by bin key, ready for one or more value reductions.
struct SortedBins {
    at::Tensor keys;
    at::Tensor rows;
    int64_t count = 0;
    unsigned long long bins_count = 0;
};

SortedBins sorted_bins(
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    const at::Tensor& component_id,
    const at::Tensor& valid,
    int64_t tx_count,
    int64_t rx_count) {
    SortedBins out;
    out.count = valid.numel();
    out.bins_count = static_cast<unsigned long long>(tx_count * rx_count * kSortedSlots);
    const auto long_options = valid.options().dtype(at::kLong);
    const auto int_options = valid.options().dtype(at::kInt);
    out.keys = at::empty({out.count}, long_options);
    out.rows = at::empty({out.count}, int_options);
    if (out.count == 0) {
        return out;
    }
    TORCH_CHECK(out.count <= std::numeric_limits<int>::max(), "sorted accumulation supports at most 2^31-1 rows");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((out.count + threads - 1) / threads);
    auto keys_in = at::empty({out.count}, long_options);
    auto rows_in = at::empty({out.count}, int_options);
    bdpt_sorted_keys_kernel<<<blocks, threads, 0, stream>>>(
        out.count,
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        component_id.data_ptr<int>(),
        valid.data_ptr<bool>(),
        tx_count,
        rx_count,
        reinterpret_cast<unsigned long long*>(keys_in.data_ptr<int64_t>()),
        rows_in.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int end_bit = sorted_key_bits(out.bins_count);
    size_t temp_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        reinterpret_cast<const unsigned long long*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(out.keys.data_ptr<int64_t>()),
        rows_in.data_ptr<int>(), out.rows.data_ptr<int>(),
        static_cast<int>(out.count), 0, end_bit, stream));
    auto temp = at::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, valid.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.data_ptr(), temp_bytes,
        reinterpret_cast<const unsigned long long*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(out.keys.data_ptr<int64_t>()),
        rows_in.data_ptr<int>(), out.rows.data_ptr<int>(),
        static_cast<int>(out.count), 0, end_bit, stream));
    return out;
}

// bins[key] = fixed-order sum of first[row] * second[row] over the key's rows.
at::Tensor sorted_bin_sums(const SortedBins& sorted, const float* first, const float* second,
                           const at::Tensor& reference) {
    auto bins = at::empty({static_cast<int64_t>(sorted.bins_count)}, reference.options().dtype(at::kDouble));
    zero_double_tensor(bins);
    if (sorted.count == 0) {
        return bins;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((sorted.count + threads - 1) / threads);
    const auto double_options = reference.options().dtype(at::kDouble);
    auto sorted_values = at::empty({sorted.count}, double_options);
    bdpt_sorted_gather_kernel<<<blocks, threads, 0, stream>>>(
        sorted.count, sorted.rows.data_ptr<int>(), first, second, sorted_values.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto unique_keys = at::empty({sorted.count}, reference.options().dtype(at::kLong));
    auto run_sums = at::empty({sorted.count}, double_options);
    auto run_count = at::empty({1}, reference.options().dtype(at::kInt));
    size_t temp_bytes = 0;
    const auto* keys = reinterpret_cast<const unsigned long long*>(sorted.keys.data_ptr<int64_t>());
    auto* unique = reinterpret_cast<unsigned long long*>(unique_keys.data_ptr<int64_t>());
    C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_bytes, keys, unique, sorted_values.data_ptr<double>(), run_sums.data_ptr<double>(),
        run_count.data_ptr<int>(), cub::Sum(), static_cast<int>(sorted.count), stream));
    auto temp = at::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, reference.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        temp.data_ptr(), temp_bytes, keys, unique, sorted_values.data_ptr<double>(), run_sums.data_ptr<double>(),
        run_count.data_ptr<int>(), cub::Sum(), static_cast<int>(sorted.count), stream));
    bdpt_sorted_scatter_kernel<<<blocks, threads, 0, stream>>>(
        sorted.count, run_count.data_ptr<int>(), unique, run_sums.data_ptr<double>(), sorted.bins_count,
        bins.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bins;
}

void sorted_power_finalize(
    const at::Tensor& bins, int64_t cells, at::Tensor& path_gain, at::Tensor& los, at::Tensor& reflection,
    at::Tensor& diffraction, at::Tensor& transmission, at::Tensor& scattering) {
    if (cells == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(bins.get_device()).stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((cells + threads - 1) / threads);
    bdpt_sorted_power_finalize_kernel<<<blocks, threads, 0, stream>>>(
        cells, bins.data_ptr<double>(), path_gain.data_ptr<float>(), los.data_ptr<float>(),
        reflection.data_ptr<float>(), diffraction.data_ptr<float>(), transmission.data_ptr<float>(),
        scattering.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Coherent BDPT accumulation: the coherent forward (combine_domain == 1) returns the
// per-component complex bin-sum buffers (S_b) as ten extra non-differentiable
// outputs so the accumulate backward reads them directly instead of re-reducing
// the atomic-double phasor sum. combine_domain == 0 returns those ten trailing
// slots undefined (empty), keeping the power-domain output byte-identical.
std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor coeff_real,
    at::Tensor coeff_imag,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy,
    int64_t combine_domain) {
    // Ten trailing bin-sum outputs; only assigned on the coherent branch.
    at::Tensor bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
        bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
        bin_transmission_im, bin_scattering_re, bin_scattering_im;
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
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 (power) or 1 (coherent)");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto path_gain = at::empty({tx_count, rx_count}, float_options);
    auto los = at::empty({tx_count, rx_count}, float_options);
    auto reflection = at::empty({tx_count, rx_count}, float_options);
    auto diffraction = at::empty({tx_count, rx_count}, float_options);
    auto transmission = at::empty({tx_count, rx_count}, float_options);
    auto scattering = at::empty({tx_count, rx_count}, float_options);
    const int64_t count = contribution.numel();
    const int64_t out_count = tx_count * rx_count;
    if (combine_domain == 1) {
        // coherent combination. accumulation_strategy is a power-domain
        // reduction perf axis and stays orthogonal: the coherent phasor sum
        // always uses the atomic-double reduction regardless of its value.
        check_float_cuda(coeff_real, "coeff_real", 1);
        check_float_cuda(coeff_imag, "coeff_imag", 1);
        TORCH_CHECK(coeff_real.sizes() == contribution.sizes(), "coeff_real must match contribution");
        TORCH_CHECK(coeff_imag.sizes() == contribution.sizes(), "coeff_imag must match contribution");
        check_same_device(coeff_real, contribution, "coeff_real");
        check_same_device(coeff_imag, contribution, "coeff_imag");
        auto double_options = contribution.options().dtype(at::kDouble);
        auto los_re = at::empty({tx_count, rx_count}, double_options);
        auto los_im = at::empty({tx_count, rx_count}, double_options);
        auto reflection_re = at::empty({tx_count, rx_count}, double_options);
        auto reflection_im = at::empty({tx_count, rx_count}, double_options);
        auto diffraction_re = at::empty({tx_count, rx_count}, double_options);
        auto diffraction_im = at::empty({tx_count, rx_count}, double_options);
        auto transmission_re = at::empty({tx_count, rx_count}, double_options);
        auto transmission_im = at::empty({tx_count, rx_count}, double_options);
        auto scattering_re = at::empty({tx_count, rx_count}, double_options);
        auto scattering_im = at::empty({tx_count, rx_count}, double_options);
        zero_double_tensor(los_re);
        zero_double_tensor(los_im);
        zero_double_tensor(reflection_re);
        zero_double_tensor(reflection_im);
        zero_double_tensor(diffraction_re);
        zero_double_tensor(diffraction_im);
        zero_double_tensor(transmission_re);
        zero_double_tensor(transmission_im);
        zero_double_tensor(scattering_re);
        zero_double_tensor(scattering_im);
        if (count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_accumulate_connection_samples_coherent_kernel<<<blocks, threads, 0, stream>>>(
                count,
                coeff_real.data_ptr<float>(),
                coeff_imag.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                tx_count,
                rx_count,
                los_re.data_ptr<double>(),
                los_im.data_ptr<double>(),
                reflection_re.data_ptr<double>(),
                reflection_im.data_ptr<double>(),
                diffraction_re.data_ptr<double>(),
                diffraction_im.data_ptr<double>(),
                transmission_re.data_ptr<double>(),
                transmission_im.data_ptr<double>(),
                scattering_re.data_ptr<double>(),
                scattering_im.data_ptr<double>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_finalize_coherent_accumulation_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                los_re.data_ptr<double>(),
                los_im.data_ptr<double>(),
                reflection_re.data_ptr<double>(),
                reflection_im.data_ptr<double>(),
                diffraction_re.data_ptr<double>(),
                diffraction_im.data_ptr<double>(),
                transmission_re.data_ptr<double>(),
                transmission_im.data_ptr<double>(),
                scattering_re.data_ptr<double>(),
                scattering_im.data_ptr<double>(),
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        // Retain the phasor bin sums S_b for the coherent backward/jvp (ruling
        // 6.4): no in-backward re-reduction of the atomic-double accumulation.
        bin_los_re = los_re;
        bin_los_im = los_im;
        bin_reflection_re = reflection_re;
        bin_reflection_im = reflection_im;
        bin_diffraction_re = diffraction_re;
        bin_diffraction_im = diffraction_im;
        bin_transmission_re = transmission_re;
        bin_transmission_im = transmission_im;
        bin_scattering_re = scattering_re;
        bin_scattering_im = scattering_im;
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
    }
    if (accumulation_strategy == 1) {
        const SortedBins sorted = sorted_bins(tx_id, rx_id, component_id, valid, tx_count, rx_count);
        const at::Tensor bins = sorted_bin_sums(
            sorted, contribution.data_ptr<float>(), mis_weight.data_ptr<float>(), contribution);
        sorted_power_finalize(bins, out_count, path_gain, los, reflection, diffraction, transmission, scattering);
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
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
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
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
    return {path_gain, los, reflection, diffraction, transmission, scattering,
            bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
            bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
            bin_transmission_im, bin_scattering_re, bin_scattering_im};
}

at::Tensor channel_bdpt_connection_variance_cuda(
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

// ===========================================================================
// BDPT AD: bdpt_accumulate_connection_samples backward + jvp companions.
//
// The accumulate op is linear per component; MIS weights, the connection
// topology (tx_id/rx_id/component_id/valid), and combine_domain are frozen.
// * Power domain M[b] = sum_r contribution_r * mis_r
// backward: grad_contribution_r = mis_r * (grad_path_gain[b] +
// grad_<component_r>[b]) (a deterministic gather)
// * Coherent P_c[b] = |S_c[b]|^2, path_gain[b] = sum_c P_c[b]
// backward: grad_coeff_r = 2 * (grad_<c>[b] + grad_path_gain[b]) * S_c[b]
// reading the forward-retained bin sums S_c (retained forward bin sums).
// The forward's per-component phasor bins are atomic-double (perf axis); the
// JVP recomputes the tangent bin sums in fixed order so it stays deterministic
// with no float atomics (the primal/JVP determinism rule).
// ===========================================================================

namespace {

__device__ __forceinline__ const float* bdpt_component_matrix(
    int component,
    const float* los,
    const float* reflection,
    const float* diffraction,
    const float* transmission,
    const float* scattering) {
    if (component == kComponentLos) return los;
    if (component == kComponentReflection) return reflection;
    if (component == kComponentDiffraction) return diffraction;
    if (component == kComponentTransmission) return transmission;
    if (component == kComponentScattering) return scattering;
    return nullptr;
}

__global__ void bdpt_accumulate_power_backward_kernel(
    int64_t count,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    const float* grad_path_gain,
    const float* grad_los,
    const float* grad_reflection,
    const float* grad_diffraction,
    const float* grad_transmission,
    const float* grad_scattering,
    float* grad_contribution) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float g = 0.0f;
    if (valid[index]) {
        const int tx = tx_id[index];
        const int rx = rx_id[index];
        const int component = component_id[index];
        if (tx >= 0 && tx < tx_count && rx >= 0 && rx < rx_count &&
            bdpt_component_accumulable(component)) {
            const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
            float grad = grad_path_gain != nullptr ? grad_path_gain[out_index] : 0.0f;
            const float* comp = bdpt_component_matrix(
                component, grad_los, grad_reflection, grad_diffraction,
                grad_transmission, grad_scattering);
            if (comp != nullptr) {
                grad += comp[out_index];
            }
            g = mis_weight[index] * grad;
        }
    }
    grad_contribution[index] = g;
}

__global__ void bdpt_accumulate_coherent_backward_kernel(
    int64_t count,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    const float* grad_path_gain,
    const float* grad_los,
    const float* grad_reflection,
    const float* grad_diffraction,
    const float* grad_transmission,
    const float* grad_scattering,
    const double* los_re,
    const double* los_im,
    const double* reflection_re,
    const double* reflection_im,
    const double* diffraction_re,
    const double* diffraction_im,
    const double* transmission_re,
    const double* transmission_im,
    const double* scattering_re,
    const double* scattering_im,
    float* grad_coeff_real,
    float* grad_coeff_imag) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float gr = 0.0f;
    float gi = 0.0f;
    if (valid[index]) {
        const int tx = tx_id[index];
        const int rx = rx_id[index];
        const int component = component_id[index];
        if (tx >= 0 && tx < tx_count && rx >= 0 && rx < rx_count &&
            bdpt_component_accumulable(component)) {
            const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
            const float grad_pg =
                grad_path_gain != nullptr ? grad_path_gain[out_index] : 0.0f;
            const float* comp = bdpt_component_matrix(
                component, grad_los, grad_reflection, grad_diffraction,
                grad_transmission, grad_scattering);
            const float grad_comp = comp != nullptr ? comp[out_index] : 0.0f;
            const float g_power = grad_pg + grad_comp;
            const double* bin_re = nullptr;
            const double* bin_im = nullptr;
            if (component == kComponentLos) {
                bin_re = los_re;
                bin_im = los_im;
            } else if (component == kComponentReflection) {
                bin_re = reflection_re;
                bin_im = reflection_im;
            } else if (component == kComponentDiffraction) {
                bin_re = diffraction_re;
                bin_im = diffraction_im;
            } else if (component == kComponentTransmission) {
                bin_re = transmission_re;
                bin_im = transmission_im;
            } else if (component == kComponentScattering) {
                bin_re = scattering_re;
                bin_im = scattering_im;
            }
            if (bin_re != nullptr) {
                const float s_re = static_cast<float>(bin_re[out_index]);
                const float s_im = static_cast<float>(bin_im[out_index]);
                gr = 2.0f * g_power * s_re;
                gi = 2.0f * g_power * s_im;
            }
        }
    }
    grad_coeff_real[index] = gr;
    grad_coeff_imag[index] = gi;
}

// Optional-tensor helper: None -> nullptr, else validate and expose contiguous
// storage. Shape/dtype/device are enforced against the reference structure.
const at::Tensor* accumulate_optional(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none()) {
        return nullptr;
    }
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* accumulate_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

}  // namespace

pybind11::dict channel_bdpt_accumulate_connection_samples_backward_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object grad_path_gain,
    pybind11::object grad_los,
    pybind11::object grad_reflection,
    pybind11::object grad_diffraction,
    pybind11::object grad_transmission,
    pybind11::object grad_scattering,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im,
    bool need_grad_contribution,
    bool need_grad_coeff) {
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 or 1");
    TORCH_CHECK(tx_id.sizes() == mis_weight.sizes(), "tx_id must match mis_weight");
    TORCH_CHECK(rx_id.sizes() == mis_weight.sizes(), "rx_id must match mis_weight");
    TORCH_CHECK(component_id.sizes() == mis_weight.sizes(), "component_id must match mis_weight");
    TORCH_CHECK(valid.sizes() == mis_weight.sizes(), "valid must match mis_weight");
    check_same_device(tx_id, mis_weight, "tx_id");
    check_same_device(rx_id, mis_weight, "rx_id");
    check_same_device(component_id, mis_weight, "component_id");
    check_same_device(valid, mis_weight, "valid");
    const int64_t count = mis_weight.numel();
    const std::vector<int64_t> matrix_shape = {tx_count, rx_count};

    at::Tensor gpg_s, glos_s, gref_s, gdif_s, gtra_s, gsca_s;
    const at::Tensor* gpg = accumulate_optional(
        std::move(grad_path_gain), gpg_s, "grad_path_gain", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* glos = accumulate_optional(
        std::move(grad_los), glos_s, "grad_los", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gref = accumulate_optional(
        std::move(grad_reflection), gref_s, "grad_reflection", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gdif = accumulate_optional(
        std::move(grad_diffraction), gdif_s, "grad_diffraction", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gtra = accumulate_optional(
        std::move(grad_transmission), gtra_s, "grad_transmission", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gsca = accumulate_optional(
        std::move(grad_scattering), gsca_s, "grad_scattering", at::kFloat, matrix_shape, mis_weight);

    at::Tensor grad_contribution;
    at::Tensor grad_coeff_real;
    at::Tensor grad_coeff_imag;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mis_weight.get_device()).stream();
    if (combine_domain == 0) {
        if (need_grad_contribution) {
            grad_contribution = at::empty({count}, mis_weight.options());
            if (count > 0) {
                constexpr int threads = 256;
                int blocks = static_cast<int>((count + threads - 1) / threads);
                bdpt_accumulate_power_backward_kernel<<<blocks, threads, 0, stream>>>(
                    count,
                    mis_weight.data_ptr<float>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    valid.data_ptr<bool>(),
                    tx_count,
                    rx_count,
                    accumulate_ptr<float>(gpg),
                    accumulate_ptr<float>(glos),
                    accumulate_ptr<float>(gref),
                    accumulate_ptr<float>(gdif),
                    accumulate_ptr<float>(gtra),
                    accumulate_ptr<float>(gsca),
                    grad_contribution.data_ptr<float>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    } else {
        at::Tensor lre_s, lim_s, rre_s, rim_s, dre_s, dim_s, tre_s, tim_s, sre_s, sim_s;
        const at::Tensor* lre = accumulate_optional(
            std::move(los_re), lre_s, "los_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* lim = accumulate_optional(
            std::move(los_im), lim_s, "los_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rre = accumulate_optional(
            std::move(reflection_re), rre_s, "reflection_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rim = accumulate_optional(
            std::move(reflection_im), rim_s, "reflection_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dre = accumulate_optional(
            std::move(diffraction_re), dre_s, "diffraction_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dim = accumulate_optional(
            std::move(diffraction_im), dim_s, "diffraction_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tre = accumulate_optional(
            std::move(transmission_re), tre_s, "transmission_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tim = accumulate_optional(
            std::move(transmission_im), tim_s, "transmission_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sre = accumulate_optional(
            std::move(scattering_re), sre_s, "scattering_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sim = accumulate_optional(
            std::move(scattering_im), sim_s, "scattering_im", at::kDouble, matrix_shape, mis_weight);
        TORCH_CHECK(
            lre && lim && rre && rim && dre && dim && tre && tim && sre && sim,
            "coherent accumulate backward requires all ten bin-sum buffers");
        if (need_grad_coeff) {
            grad_coeff_real = at::empty({count}, mis_weight.options());
            grad_coeff_imag = at::empty({count}, mis_weight.options());
            if (count > 0) {
                constexpr int threads = 256;
                int blocks = static_cast<int>((count + threads - 1) / threads);
                bdpt_accumulate_coherent_backward_kernel<<<blocks, threads, 0, stream>>>(
                    count,
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    valid.data_ptr<bool>(),
                    tx_count,
                    rx_count,
                    accumulate_ptr<float>(gpg),
                    accumulate_ptr<float>(glos),
                    accumulate_ptr<float>(gref),
                    accumulate_ptr<float>(gdif),
                    accumulate_ptr<float>(gtra),
                    accumulate_ptr<float>(gsca),
                    lre->data_ptr<double>(),
                    lim->data_ptr<double>(),
                    rre->data_ptr<double>(),
                    rim->data_ptr<double>(),
                    dre->data_ptr<double>(),
                    dim->data_ptr<double>(),
                    tre->data_ptr<double>(),
                    tim->data_ptr<double>(),
                    sre->data_ptr<double>(),
                    sim->data_ptr<double>(),
                    grad_coeff_real.data_ptr<float>(),
                    grad_coeff_imag.data_ptr<float>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    }
    pybind11::dict out;
    out["grad_contribution"] = grad_contribution.defined()
        ? pybind11::cast(grad_contribution)
        : pybind11::object(pybind11::none());
    out["grad_coeff_real"] = grad_coeff_real.defined()
        ? pybind11::cast(grad_coeff_real)
        : pybind11::object(pybind11::none());
    out["grad_coeff_imag"] = grad_coeff_imag.defined()
        ? pybind11::cast(grad_coeff_imag)
        : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_bdpt_accumulate_connection_samples_jvp_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object tangent_contribution,
    pybind11::object tangent_coeff_real,
    pybind11::object tangent_coeff_imag,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im) {
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 or 1");
    TORCH_CHECK(tx_id.sizes() == mis_weight.sizes(), "tx_id must match mis_weight");
    TORCH_CHECK(rx_id.sizes() == mis_weight.sizes(), "rx_id must match mis_weight");
    TORCH_CHECK(component_id.sizes() == mis_weight.sizes(), "component_id must match mis_weight");
    TORCH_CHECK(valid.sizes() == mis_weight.sizes(), "valid must match mis_weight");
    check_same_device(tx_id, mis_weight, "tx_id");
    check_same_device(rx_id, mis_weight, "rx_id");
    check_same_device(component_id, mis_weight, "component_id");
    check_same_device(valid, mis_weight, "valid");
    const int64_t count = mis_weight.numel();
    const int64_t out_count = tx_count * rx_count;
    const std::vector<int64_t> sample_shape = {count};
    const std::vector<int64_t> matrix_shape = {tx_count, rx_count};
    auto float_options = mis_weight.options().dtype(at::kFloat);
    auto t_path_gain = at::empty({tx_count, rx_count}, float_options);
    auto t_los = at::empty({tx_count, rx_count}, float_options);
    auto t_reflection = at::empty({tx_count, rx_count}, float_options);
    auto t_diffraction = at::empty({tx_count, rx_count}, float_options);
    auto t_transmission = at::empty({tx_count, rx_count}, float_options);
    auto t_scattering = at::empty({tx_count, rx_count}, float_options);
    zero_float_tensor(t_path_gain);
    zero_float_tensor(t_los);
    zero_float_tensor(t_reflection);
    zero_float_tensor(t_diffraction);
    zero_float_tensor(t_transmission);
    zero_float_tensor(t_scattering);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mis_weight.get_device()).stream();
    if (combine_domain == 0) {
        at::Tensor tc_s;
        const at::Tensor* tc = accumulate_optional(
            std::move(tangent_contribution), tc_s, "tangent_contribution",
            at::kFloat, sample_shape, mis_weight);
        TORCH_CHECK(tc != nullptr, "power-domain accumulate jvp requires tangent_contribution");
        // Row-parallel: one stable key sort of the rows, then a fixed-order
        // reduce-by-key of mis_weight * tangent per (tx, rx, component).
        const SortedBins sorted = sorted_bins(tx_id, rx_id, component_id, valid, tx_count, rx_count);
        const at::Tensor bins = sorted_bin_sums(
            sorted, mis_weight.data_ptr<float>(), tc->data_ptr<float>(), mis_weight);
        sorted_power_finalize(
            bins, out_count, t_path_gain, t_los, t_reflection, t_diffraction, t_transmission, t_scattering);
    } else {
        at::Tensor tcr_s, tci_s;
        const at::Tensor* tcr = accumulate_optional(
            std::move(tangent_coeff_real), tcr_s, "tangent_coeff_real",
            at::kFloat, sample_shape, mis_weight);
        const at::Tensor* tci = accumulate_optional(
            std::move(tangent_coeff_imag), tci_s, "tangent_coeff_imag",
            at::kFloat, sample_shape, mis_weight);
        TORCH_CHECK(tcr && tci, "coherent accumulate jvp requires tangent_coeff_real/imag");
        at::Tensor lre_s, lim_s, rre_s, rim_s, dre_s, dim_s, tre_s, tim_s, sre_s, sim_s;
        const at::Tensor* lre = accumulate_optional(
            std::move(los_re), lre_s, "los_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* lim = accumulate_optional(
            std::move(los_im), lim_s, "los_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rre = accumulate_optional(
            std::move(reflection_re), rre_s, "reflection_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rim = accumulate_optional(
            std::move(reflection_im), rim_s, "reflection_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dre = accumulate_optional(
            std::move(diffraction_re), dre_s, "diffraction_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dim = accumulate_optional(
            std::move(diffraction_im), dim_s, "diffraction_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tre = accumulate_optional(
            std::move(transmission_re), tre_s, "transmission_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tim = accumulate_optional(
            std::move(transmission_im), tim_s, "transmission_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sre = accumulate_optional(
            std::move(scattering_re), sre_s, "scattering_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sim = accumulate_optional(
            std::move(scattering_im), sim_s, "scattering_im", at::kDouble, matrix_shape, mis_weight);
        TORCH_CHECK(
            lre && lim && rre && rim && dre && dim && tre && tim && sre && sim,
            "coherent accumulate jvp requires all ten bin-sum buffers");
        const SortedBins sorted = sorted_bins(tx_id, rx_id, component_id, valid, tx_count, rx_count);
        const at::Tensor bins_re = sorted_bin_sums(sorted, tcr->data_ptr<float>(), nullptr, mis_weight);
        const at::Tensor bins_im = sorted_bin_sums(sorted, tci->data_ptr<float>(), nullptr, mis_weight);
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            bdpt_sorted_coherent_jvp_finalize_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                bins_re.data_ptr<double>(),
                bins_im.data_ptr<double>(),
                lre->data_ptr<double>(),
                lim->data_ptr<double>(),
                rre->data_ptr<double>(),
                rim->data_ptr<double>(),
                dre->data_ptr<double>(),
                dim->data_ptr<double>(),
                tre->data_ptr<double>(),
                tim->data_ptr<double>(),
                sre->data_ptr<double>(),
                sim->data_ptr<double>(),
                t_path_gain.data_ptr<float>(),
                t_los.data_ptr<float>(),
                t_reflection.data_ptr<float>(),
                t_diffraction.data_ptr<float>(),
                t_transmission.data_ptr<float>(),
                t_scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    pybind11::dict out;
    out["tangent_path_gain"] = t_path_gain;
    out["tangent_los"] = t_los;
    out["tangent_reflection"] = t_reflection;
    out["tangent_diffraction"] = t_diffraction;
    out["tangent_transmission"] = t_transmission;
    out["tangent_scattering"] = t_scattering;
    return out;
}

// ==== Section: BDPT connection AD ====

#include <src/field_transport_ad.cuh>

#include <algorithm>

// BDPT AD: backward + jvp companions for bdpt_endpoint_connection_samples.
//
// Forward (per connection row): the light endpoint field F and the sensor
// polarization project through the frozen free-space carrier into
// contribution = P_src * |coeff|^2 / N,
// coeff = <F * propagation, rx_axis>, rx_axis = project(sensor_pol, dir).
// Differentiable: the light field F, the sensor polarization (through rx_axis),
// the carrier frequency (through the propagation amplitude and phase), and the
// source power P_src (tx_power). Frozen: the connection geometry (distance,
// direction, total path length), N (samples_per_tx), visibility, MIS, and the
// component/topology structure. The backward recomputes the forward carrier in
// primal expression order; light/sensor field grads accumulate with atomicAdd
// because the light x sensor connection grid shares each endpoint field across
// its row, and the tx_power / frequency grads are scalar atomic reductions.

namespace {

namespace ad = rayd::torch::field_transport_ad;

// Recompute the frozen carrier for one connection row exactly as
// bdpt_endpoint_connection_samples_kernel; returns whether the row contributes.
struct ConnectionCarrier {
    bool row_valid;
    int tx;
    int rx;
    utd::Complex propagation;
    utd::Complex d_propagation_df;  // d propagation / d frequency
    utd::float3a rx_axis;           // project(sensor_pol, direction)
    utd::float3a direction;         // connection direction (frozen)
    utd::float3a sensor_pol;        // sensor polarization (differentiable)
    utd::Complex3 incident_field;   // light field F
    float source_power;
    float inv_samples_per_tx;
};

__device__ ConnectionCarrier bdpt_connection_carrier(
    int64_t index,
    int64_t sensor_count,
    float frequency_hz,
    float inv_samples_per_tx,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid) {
    ConnectionCarrier out;
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const int tx = light_tx_id[light_index];
    const int rx = sensor_rx_id[sensor_index];
    const bool is_valid =
        light_valid[light_index] && sensor_valid[sensor_index] && tx >= 0 && rx >= 0;

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
           dz * light_direction[light_index * 3 + 2]) / distance
        : 1.0f;
    const bool direction_valid = dir_dot > 0.0f;
    out.row_valid = is_valid && direction_valid;
    out.tx = tx;
    out.rx = rx;
    out.inv_samples_per_tx = inv_samples_per_tx;
    out.source_power = light_source_power[light_index];

    const float total_distance = distance + fmaxf(light_path_length[light_index], 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float k_clamped = fmaxf(wave_number, 1.0e-12f);
    const float l_clamped = fmaxf(total_distance, 1.0e-6f);
    const float amplitude = 1.0f / (2.0f * k_clamped * l_clamped);
    const float phase_angle = transport::precise_neg_kd(wave_number, total_distance);
    const utd::Complex carrier_phase = utd::cplx_exp_phase(phase_angle);
    out.propagation = utd::cplx_mul_real(carrier_phase, amplitude);

    // d propagation / d frequency (amplitude and phase chain, matching the
    // free-space carrier convention). dk/df = 2*pi/c; the fmod phase reduction
    // has unit slope so d(phase_angle)/dk = -total_distance.
    const float dk_df = 2.0f * kPi / kLightSpeedMPerS;
    const float d_amp_df =
        wave_number > 1.0e-12f ? (-amplitude / k_clamped) * dk_df : 0.0f;
    const float d_phase_df = -total_distance * dk_df;
    out.d_propagation_df = utd::cplx(
        d_amp_df * carrier_phase.re - amplitude * carrier_phase.im * d_phase_df,
        d_amp_df * carrier_phase.im + amplitude * carrier_phase.re * d_phase_df);

    out.direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    out.sensor_pol = utd::make_f3(
        sensor_field_real[sensor_index * 3 + 0],
        sensor_field_real[sensor_index * 3 + 1],
        sensor_field_real[sensor_index * 3 + 2]);
    out.rx_axis = utd::project_to_wedge_plane(out.sensor_pol, out.direction);
    const int64_t field_offset = light_index * 3;
    out.incident_field = {
        utd::cplx(light_field_real[field_offset], light_field_imag[field_offset]),
        utd::cplx(light_field_real[field_offset + 1], light_field_imag[field_offset + 1]),
        utd::cplx(light_field_real[field_offset + 2], light_field_imag[field_offset + 2])};
    return out;
}

__global__ void bdpt_endpoint_connection_backward_kernel(
    int64_t count,
    int64_t sensor_count,
    int64_t tx_count,
    float frequency_hz,
    float inv_samples_per_tx,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    const float* grad_contribution,
    float* grad_light_field_real,
    float* grad_light_field_imag,
    float* grad_sensor_field_real,
    float* grad_frequency,
    float* grad_tx_power) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ConnectionCarrier carrier = bdpt_connection_carrier(
            index, sensor_count, frequency_hz, inv_samples_per_tx, light_origin,
            light_direction, light_field_real, light_field_imag,
            light_source_power, light_depth, light_tx_id, light_valid,
            light_path_length, sensor_origin, sensor_field_real, sensor_rx_id,
            sensor_valid);
        if (!carrier.row_valid) {
            continue;  // forward wrote contribution = 0; every gradient is zero
        }
        const float g = grad_contribution[index];
        const int64_t light_index = index / sensor_count;
        const int64_t sensor_index = index - light_index * sensor_count;

        // Recompute the forward chain (primal expression order).
        const utd::Complex3 received = utd::c3_scale(
            carrier.incident_field, carrier.propagation);
        const utd::Complex coeff = transport::complex3_dot_real(received, carrier.rx_axis);
        const float coeff_power = utd::cplx_abs_sqr(coeff);
        const float scale = carrier.source_power * carrier.inv_samples_per_tx;

        // contribution = source_power * |coeff|^2 * inv_N.
        // d/dcoeff (pair) = source_power * inv_N * 2 * conj-pair(coeff).
        const utd::Complex g_coeff = utd::cplx(
            g * scale * 2.0f * coeff.re, g * scale * 2.0f * coeff.im);
        // coeff = <received, rx_axis>: split into received and rx_axis.
        utd::Complex3 g_received = utd::c3_zero();
        utd::float3a g_rx_axis = utd::f3_zero();
        utd::adj_cplx_dot_real(received, carrier.rx_axis, g_coeff, g_received, g_rx_axis);
        // received = incident_field * propagation (per axis).
        utd::Complex g_propagation = utd::cplx_zero();
        utd::Complex g_field_x = utd::cplx_zero();
        utd::Complex g_field_y = utd::cplx_zero();
        utd::Complex g_field_z = utd::cplx_zero();
        utd::adj_cplx_mul(
            carrier.incident_field.x, carrier.propagation, g_received.x,
            g_field_x, g_propagation);
        utd::adj_cplx_mul(
            carrier.incident_field.y, carrier.propagation, g_received.y,
            g_field_y, g_propagation);
        utd::adj_cplx_mul(
            carrier.incident_field.z, carrier.propagation, g_received.z,
            g_field_z, g_propagation);

        if (grad_light_field_real != nullptr) {
            const int64_t base = light_index * 3;
            atomicAdd(grad_light_field_real + base, g_field_x.re);
            atomicAdd(grad_light_field_real + base + 1, g_field_y.re);
            atomicAdd(grad_light_field_real + base + 2, g_field_z.re);
            atomicAdd(grad_light_field_imag + base, g_field_x.im);
            atomicAdd(grad_light_field_imag + base + 1, g_field_y.im);
            atomicAdd(grad_light_field_imag + base + 2, g_field_z.im);
        }
        if (grad_sensor_field_real != nullptr) {
            // rx_axis = project(sensor_pol, direction); the direction cotangent
            // is discarded (frozen geometry).
            utd::float3a g_sensor_pol = utd::f3_zero();
            utd::float3a g_dir_dump = utd::f3_zero();
            ad::adj_transverse_project(
                carrier.direction, carrier.sensor_pol, g_rx_axis,
                g_dir_dump, g_sensor_pol);
            const int64_t base = sensor_index * 3;
            atomicAdd(grad_sensor_field_real + base, g_sensor_pol.x);
            atomicAdd(grad_sensor_field_real + base + 1, g_sensor_pol.y);
            atomicAdd(grad_sensor_field_real + base + 2, g_sensor_pol.z);
        }
        if (grad_frequency != nullptr) {
            atomicAdd(
                grad_frequency,
                ad::adj_dot(g_propagation, carrier.d_propagation_df));
        }
        if (grad_tx_power != nullptr && carrier.tx >= 0 && carrier.tx < tx_count) {
            atomicAdd(
                grad_tx_power + carrier.tx,
                g * coeff_power * carrier.inv_samples_per_tx);
        }
    }
}

__global__ void bdpt_endpoint_connection_jvp_kernel(
    int64_t count,
    int64_t sensor_count,
    int64_t tx_count,
    float frequency_hz,
    float inv_samples_per_tx,
    float tangent_frequency,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    const float* tangent_light_field_real,
    const float* tangent_light_field_imag,
    const float* tangent_sensor_field_real,
    const float* tangent_tx_power,
    float* tangent_contribution) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ConnectionCarrier carrier = bdpt_connection_carrier(
            index, sensor_count, frequency_hz, inv_samples_per_tx, light_origin,
            light_direction, light_field_real, light_field_imag,
            light_source_power, light_depth, light_tx_id, light_valid,
            light_path_length, sensor_origin, sensor_field_real, sensor_rx_id,
            sensor_valid);
        if (!carrier.row_valid) {
            tangent_contribution[index] = 0.0f;
            continue;
        }
        const int64_t light_index = index / sensor_count;
        const int64_t sensor_index = index - light_index * sensor_count;
        const int64_t light_base = light_index * 3;
        const int64_t sensor_base = sensor_index * 3;

        const utd::Complex3 t_field = {
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base] : 0.0f),
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base + 1] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base + 1] : 0.0f),
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base + 2] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base + 2] : 0.0f)};
        const utd::float3a t_sensor_pol = utd::make_f3(
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base] : 0.0f,
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base + 1] : 0.0f,
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base + 2] : 0.0f);
        // rx_axis = project(sensor_pol, dir) is linear in sensor_pol (dir frozen).
        const utd::float3a t_rx_axis = utd::project_to_wedge_plane(
            t_sensor_pol, carrier.direction);
        const utd::Complex t_propagation = utd::cplx_mul_real(
            carrier.d_propagation_df, tangent_frequency);

        const utd::Complex3 received = utd::c3_scale(
            carrier.incident_field, carrier.propagation);
        const utd::Complex3 t_received = utd::c3_add(
            utd::c3_scale(t_field, carrier.propagation),
            utd::c3_scale(carrier.incident_field, t_propagation));
        const utd::Complex coeff = transport::complex3_dot_real(received, carrier.rx_axis);
        const utd::Complex t_coeff = utd::cplx_add(
            transport::complex3_dot_real(t_received, carrier.rx_axis),
            transport::complex3_dot_real(received, t_rx_axis));
        const float coeff_power = utd::cplx_abs_sqr(coeff);
        const float t_coeff_power =
            2.0f * (coeff.re * t_coeff.re + coeff.im * t_coeff.im);
        const float t_source_power =
            (tangent_tx_power != nullptr && carrier.tx >= 0 && carrier.tx < tx_count)
                ? tangent_tx_power[carrier.tx]
                : 0.0f;
        tangent_contribution[index] = carrier.inv_samples_per_tx *
            (t_source_power * coeff_power + carrier.source_power * t_coeff_power);
    }
}

const at::Tensor* connect_optional(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none()) {
        return nullptr;
    }
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
T* connect_ptr(at::Tensor& tensor) {
    return tensor.defined() ? tensor.data_ptr<T>() : nullptr;
}

at::Tensor connect_zero(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

int64_t connection_count(int64_t light_count, int64_t sensor_count, int64_t max_paths) {
    const int64_t total = light_count * sensor_count;
    return max_paths < 0 ? total : std::min<int64_t>(max_paths, total);
}

}  // namespace

pybind11::dict channel_bdpt_endpoint_connection_samples_backward_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    at::Tensor grad_contribution,
    bool need_grad_field,
    bool need_grad_frequency,
    bool need_grad_tx_power) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_float_cuda(light_path_length, "light_path_length", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    check_float_cuda(grad_contribution, "grad_contribution", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    const int64_t count = connection_count(light_count, sensor_count, max_paths);
    TORCH_CHECK(grad_contribution.numel() == count, "grad_contribution must match connection count");

    at::Tensor grad_light_field_real;
    at::Tensor grad_light_field_imag;
    at::Tensor grad_sensor_field_real;
    at::Tensor grad_sensor_field_imag;
    at::Tensor grad_frequency;
    at::Tensor grad_tx_power;
    if (need_grad_field) {
        grad_light_field_real = connect_zero({light_count, 3}, light_origin.options());
        grad_light_field_imag = connect_zero({light_count, 3}, light_origin.options());
        grad_sensor_field_real = connect_zero({sensor_count, 3}, light_origin.options());
        // The sensor imaginary field never enters the forward; its derivative is
        // exactly zero (reported explicitly, not a silent-zero substitute).
        grad_sensor_field_imag = connect_zero({sensor_count, 3}, light_origin.options());
    }
    if (need_grad_frequency) {
        grad_frequency = connect_zero({1}, light_origin.options());
    }
    if (need_grad_tx_power) {
        grad_tx_power = connect_zero({tx_count}, light_origin.options());
    }
    if (count > 0 && (need_grad_field || need_grad_frequency || need_grad_tx_power)) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_endpoint_connection_backward_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            tx_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            grad_contribution.data_ptr<float>(),
            connect_ptr<float>(grad_light_field_real),
            connect_ptr<float>(grad_light_field_imag),
            connect_ptr<float>(grad_sensor_field_real),
            connect_ptr<float>(grad_frequency),
            connect_ptr<float>(grad_tx_power));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_light_field_real"] = need_grad_field
        ? pybind11::cast(grad_light_field_real) : pybind11::object(pybind11::none());
    out["grad_light_field_imag"] = need_grad_field
        ? pybind11::cast(grad_light_field_imag) : pybind11::object(pybind11::none());
    out["grad_sensor_field_real"] = need_grad_field
        ? pybind11::cast(grad_sensor_field_real) : pybind11::object(pybind11::none());
    out["grad_sensor_field_imag"] = need_grad_field
        ? pybind11::cast(grad_sensor_field_imag) : pybind11::object(pybind11::none());
    out["grad_frequency"] = need_grad_frequency
        ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    out["grad_tx_power"] = need_grad_tx_power
        ? pybind11::cast(grad_tx_power) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_bdpt_endpoint_connection_samples_jvp_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_sensor_field_real,
    double tangent_frequency,
    pybind11::object tangent_tx_power) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_float_cuda(light_path_length, "light_path_length", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    const int64_t count = connection_count(light_count, sensor_count, max_paths);

    at::Tensor tlr_s, tli_s, tsr_s, ttx_s;
    const at::Tensor* tlr = connect_optional(
        std::move(tangent_light_field_real), tlr_s, "tangent_light_field_real",
        at::kFloat, {light_count, 3}, light_origin);
    const at::Tensor* tli = connect_optional(
        std::move(tangent_light_field_imag), tli_s, "tangent_light_field_imag",
        at::kFloat, {light_count, 3}, light_origin);
    const at::Tensor* tsr = connect_optional(
        std::move(tangent_sensor_field_real), tsr_s, "tangent_sensor_field_real",
        at::kFloat, {sensor_count, 3}, light_origin);
    const at::Tensor* ttx = connect_optional(
        std::move(tangent_tx_power), ttx_s, "tangent_tx_power",
        at::kFloat, {tx_count}, light_origin);

    auto tangent_contribution = at::empty({count}, light_origin.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_endpoint_connection_jvp_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            tx_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            static_cast<float>(tangent_frequency),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            tlr != nullptr ? tlr->data_ptr<float>() : nullptr,
            tli != nullptr ? tli->data_ptr<float>() : nullptr,
            tsr != nullptr ? tsr->data_ptr<float>() : nullptr,
            ttx != nullptr ? ttx->data_ptr<float>() : nullptr,
            tangent_contribution.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_contribution"] = tangent_contribution;
    return out;
}

// ==== Section: BDPT wavefront connections ====

namespace {

namespace wf = channel::bdpt_wavefront;

constexpr int kWavefrontMixedMask = kMaskReflection | kMaskTransmission;

// Optional connection-row output (export and diagnostics): rows land at atomic
// slots below capacity with a (sample, bounce, receiver) key for a stable sort.
struct BdptWavefrontRows {
    int64_t capacity;
    unsigned long long* cursor;
    int64_t* key;
    int* tx_id;
    int* rx_id;
    int* component_id;
    int* light_depth;
    float* contribution;
    float* pdf;
    float* mis_weight;
    float* path_length;
};

__device__ void bdpt_wavefront_write_row(
    const BdptWavefrontRows& rows, int64_t key, int tx, int rx, int component, int depth, float contribution,
    float pdf, float mis_weight, float path_length) {
    if (rows.capacity <= 0) {
        return;
    }
    const unsigned long long slot = atomicAdd(rows.cursor, 1ULL);
    if (slot >= static_cast<unsigned long long>(rows.capacity)) {
        return;
    }
    rows.key[slot] = key;
    rows.tx_id[slot] = tx;
    rows.rx_id[slot] = rx;
    rows.component_id[slot] = component;
    rows.light_depth[slot] = depth;
    rows.contribution[slot] = contribution;
    rows.pdf[slot] = pdf;
    rows.mis_weight[slot] = mis_weight;
    rows.path_length[slot] = path_length;
}

// Radiometric constants of the scatter next-event row.
struct BdptNeeConstants {
    float amplitude_sq;   // (lambda / (4 pi))^2
    float four_pi;
    float samples;        // samples per transmitter
};

// Scatter next-event row of a recorded scatter vertex to one receiver:
// source_power * (P_te f_te + P_tm f_tm) * cos_o * (lambda/4pi)^2 * 4pi /
// (r^2 * max(p_s, 1e-4) * samples). Returns false when the receiver lies behind
// the mean plane or the row carries no power.
__device__ bool bdpt_nee_row(
    const wf::State& state,
    const wf::ScatterTables& tables,
    int64_t light,
    const float* rx_position,
    float source_power,
    const BdptNeeConstants& constants,
    float* wo_world,
    float& r2,
    float& contribution,
    float* pdf_omega = nullptr) {
    const float* position = state.nee_position + light * 3;
    const float* normal = state.nee_normal + light * 3;
    const float delta[3] = {
        rx_position[0] - position[0], rx_position[1] - position[1], rx_position[2] - position[2]};
    r2 = fmaxf(sqrtf(wf::dot_values(delta, delta)), 1.0e-6f);
    for (int axis = 0; axis < 3; ++axis) {
        wo_world[axis] = delta[axis] / r2;
    }
    const float cos_o = wf::dot_values(wo_world, normal);
    contribution = 0.0f;
    if (!(cos_o > 0.0f)) {
        return false;
    }
    const int material = state.nee_material[light];
    float t1[3];
    float t2[3];
    wf::local_frame(normal, tables.rough_axis_rad[material], t1, t2);
    const float wo_local[3] = {wf::dot_values(wo_world, t1), wf::dot_values(wo_world, t2), wf::dot_values(wo_world, normal)};
    const wf::TableView table = wf::table_view(tables, tables.material_slot[material]);
    float f_te = 0.0f;
    float f_tm = 0.0f;
    rayd::shared::scattering::eval_te_tm(
        table.f_te, table.f_tm, table.nti, table.npi, table.nto, table.npo, state.nee_wi_local + light * 3,
        wo_local, f_te, f_tm);
    if (pdf_omega != nullptr) {
        *pdf_omega = rayd::shared::scattering::table_pdf(
            table.density, table.nti, table.npi, table.nto, table.npo, state.nee_wi_local + light * 3, wo_local);
    }
    const float power_kernel = state.nee_p_te[light] * f_te + state.nee_p_tm[light] * f_tm;
    const float geometry = fmaxf(cos_o, 0.0f) * constants.amplitude_sq * constants.four_pi / (r2 * r2);
    contribution = source_power * power_kernel * geometry /
        (fmaxf(state.nee_p_scatter[light], 1.0e-4f) * constants.samples);
    return contribution > 0.0f;
}

// Free-space carrier of a mixed connection for the derivative passes: the
// propagation factor and its frequency derivative, the receiver axis (the
// projected receiver polarization) and the received coefficient
// <propagation * F, rx_axis> whose squared magnitude is the row power.
struct WavefrontMixedCarrier {
    utd::Complex propagation;
    utd::Complex d_propagation_df;
    utd::float3a rx_axis;
    utd::Complex3 field;
    utd::Complex coeff;
};

__device__ WavefrontMixedCarrier wavefront_mixed_carrier(
    const float* origin, const float* field_real, const float* field_imag, float path_length,
    const float* rx_position, const float* rx_polarization, float frequency_hz) {
    WavefrontMixedCarrier out;
    const float dx = rx_position[0] - origin[0];
    const float dy = rx_position[1] - origin[1];
    const float dz = rx_position[2] - origin[2];
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1.0e-6f);
    const float total_distance = distance + fmaxf(path_length, 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float k_clamped = fmaxf(wave_number, 1.0e-12f);
    const float amplitude = 1.0f / (2.0f * k_clamped * fmaxf(total_distance, 1.0e-6f));
    const utd::Complex carrier_phase = utd::cplx_exp_phase(transport::precise_neg_kd(wave_number, total_distance));
    out.propagation = utd::cplx_mul_real(carrier_phase, amplitude);
    const float dk_df = 2.0f * kPi / kLightSpeedMPerS;
    const float d_amp_df = wave_number > 1.0e-12f ? (-amplitude / k_clamped) * dk_df : 0.0f;
    const float d_phase_df = -total_distance * dk_df;
    out.d_propagation_df = utd::cplx(
        d_amp_df * carrier_phase.re - amplitude * carrier_phase.im * d_phase_df,
        d_amp_df * carrier_phase.im + amplitude * carrier_phase.re * d_phase_df);
    const utd::float3a direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    out.rx_axis = utd::project_to_wedge_plane(
        utd::make_f3(rx_polarization[0], rx_polarization[1], rx_polarization[2]), direction);
    out.field = {
        utd::cplx(field_real[0], field_imag[0]),
        utd::cplx(field_real[1], field_imag[1]),
        utd::cplx(field_real[2], field_imag[2])};
    out.coeff = transport::complex3_dot_real(utd::c3_scale(out.field, out.propagation), out.rx_axis);
    return out;
}

// Scatter next-event terms for the derivative passes: the local outgoing
// direction toward the receiver and the factor
// cos_o (lambda/4pi)^2 4pi / (r^2 max(p_s, 1e-4) samples), so that a row is
// source_power * (P_te f_te + P_tm f_tm) * factor.
__device__ float wavefront_nee_factor(
    const wf::State& state, const wf::ScatterTables& tables, int64_t light, const float* rx_position,
    const BdptNeeConstants& constants, float* wo_local) {
    const float* position = state.nee_position + light * 3;
    const float* normal = state.nee_normal + light * 3;
    const float delta[3] = {
        rx_position[0] - position[0], rx_position[1] - position[1], rx_position[2] - position[2]};
    const float r = fmaxf(sqrtf(wf::dot_values(delta, delta)), 1.0e-6f);
    float wo_world[3];
    for (int axis = 0; axis < 3; ++axis) {
        wo_world[axis] = delta[axis] / r;
    }
    const float cos_o = wf::dot_values(wo_world, normal);
    float t1[3];
    float t2[3];
    wf::local_frame(normal, tables.rough_axis_rad[state.nee_material[light]], t1, t2);
    wo_local[0] = wf::dot_values(wo_world, t1);
    wo_local[1] = wf::dot_values(wo_world, t2);
    wo_local[2] = cos_o;
    return fmaxf(cos_o, 0.0f) * constants.amplitude_sq * constants.four_pi / (r * r) /
        (fmaxf(state.nee_p_scatter[light], 1.0e-4f) * constants.samples);
}

// Tile pair p connects light vertex p / rx_tile to receiver rx_begin + p % rx_tile.
// A pair is active for a live mixed reflection+transmission vertex facing the
// receiver (segment from the vertex) or for a recorded scatter vertex with a
// powered next-event row (segment from the vertex offset toward the receiver).
// Inactive pairs are skipped by the visibility query.
__global__ void bdpt_wavefront_visibility_inputs_kernel(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int light_depth,
    const unsigned long long* alive,
    wf::State state,
    wf::ScatterTables tables,
    BdptNeeConstants constants,
    const float* tx_power,
    const float* rx_positions,
    float* start,
    float* end,
    bool* active) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const int64_t light = pair / rx_tile;
    const int64_t rx = rx_begin + (pair - light * rx_tile);
    const int mask = state.component_mask[light];
    bool connect = false;
    float offset[3] = {0.0f, 0.0f, 0.0f};
    const float* origin = state.origin + light * 3;
    if (alive[0] > 0ULL) {
        if ((mask & wf::kMaskScatterVertex) != 0) {
            float wo[3];
            float r2;
            float contribution;
            const int64_t tx = (first + light) / samples;
            connect = bdpt_nee_row(state, tables, light, rx_positions + rx * 3, tx_power[tx], constants, wo, r2,
                                   contribution);
            origin = state.nee_position + light * 3;
            const float epsilon = wf::scale_epsilon(origin, tables.epsilon_floor);
            for (int axis = 0; axis < 3; ++axis) {
                offset[axis] = wo[axis] * epsilon;
            }
        } else if (tables.emit_mixed && state.valid[light] &&
                   (mask & kWavefrontMixedMask) == kWavefrontMixedMask &&
                   (mask & kMaskScattering) == 0) {
            connect = bdpt_connection_geometry(
                origin, state.direction + light * 3, light_depth, rx_positions + rx * 3).direction_valid;
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        start[pair * 3 + axis] = origin[axis] + offset[axis];
        end[pair * 3 + axis] = rx_positions[rx * 3 + axis];
    }
    active[pair] = connect;
}

// Visible active pairs evaluate their row and add contribution * MIS weight
// straight into the double [tx, rx] transmission (mixed chains) or scattering
// (next-event rows) accumulator; no row is stored. rows[0] / rows[1] count the
// accumulated mixed / scatter rows. With kTangent the kernel also adds each
// row's forward-mode tangent into the double tangent accumulators.
template <bool kTangent>
__global__ void bdpt_wavefront_connect_accumulate_kernel(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t rx_count,
    int light_depth,
    const bool* active,
    const bool* visible,
    wf::State state,
    wf::ScatterTables tables,
    BdptNeeConstants constants,
    const float* tx_power,
    const float* rx_positions,
    const float* rx_polarization,
    float frequency_hz,
    float inv_samples_per_tx,
    int mode_id,
    float beta,
    double* transmission,
    double* scattering,
    unsigned long long* rows,
    BdptWavefrontRows row_out,
    int max_depth,
    wf::Tangent tangent,
    double* t_transmission,
    double* t_scattering) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool mixed_row = false;
    bool scatter_row = false;
    if (pair < pairs && active[pair] && visible[pair]) {
        const int64_t light = pair / rx_tile;
        const int64_t rx = rx_begin + (pair - light * rx_tile);
        const int64_t tx = (first + light) / samples;
        const int64_t key = ((first + light) * max_depth + (light_depth - 1)) * rx_count + rx;
        if ((state.component_mask[light] & wf::kMaskScatterVertex) != 0) {
            float wo[3];
            float r2;
            float contribution;
            float pdf_omega = 0.0f;
            if (bdpt_nee_row(state, tables, light, rx_positions + rx * 3, tx_power[tx], constants, wo, r2,
                             contribution, row_out.capacity > 0 ? &pdf_omega : nullptr)) {
                atomicAdd(scattering + tx * rx_count + rx, static_cast<double>(contribution));
                scatter_row = true;
                if constexpr (kTangent) {
                    float wo_local[3];
                    const float factor =
                        wavefront_nee_factor(state, tables, light, rx_positions + rx * 3, constants, wo_local);
                    const int material = state.nee_material[light];
                    const int table_slot = tables.material_slot[material];
                    const wf::TableView table = wf::table_view(tables, table_slot);
                    const float* wi_local = state.nee_wi_local + light * 3;
                    float f_te = 0.0f;
                    float f_tm = 0.0f;
                    rayd::shared::scattering::eval_te_tm(
                        table.f_te, table.f_tm, table.nti, table.npi, table.nto, table.npo, wi_local, wo_local, f_te,
                        f_tm);
                    float df_te = 0.0f;
                    float df_tm = 0.0f;
                    if (tangent.f_te != nullptr) {
                        const int64_t offset = tables.table_offset[table_slot];
                        rayd::shared::scattering::eval_te_tm(
                            tangent.f_te + offset, tangent.f_tm + offset, table.nti, table.npi, table.nto,
                            table.npo, wi_local, wo_local, df_te, df_tm);
                    }
                    const float p_te = state.nee_p_te[light];
                    const float p_tm = state.nee_p_tm[light];
                    const float kernel = p_te * f_te + p_tm * f_tm;
                    const float d_kernel = tangent.nee_p_te[light] * f_te + p_te * df_te +
                        tangent.nee_p_tm[light] * f_tm + p_tm * df_tm;
                    const float t_power = tangent.tx_power != nullptr ? tangent.tx_power[tx] : 0.0f;
                    const float t_row = (t_power * kernel + tx_power[tx] * d_kernel) * factor -
                        2.0f * contribution * tangent.frequency / frequency_hz;
                    atomicAdd(t_scattering + tx * rx_count + rx, static_cast<double>(t_row));
                }
                bdpt_wavefront_write_row(
                    row_out, key, static_cast<int>(tx), static_cast<int>(rx), kComponentScattering, light_depth,
                    contribution, fmaxf(state.nee_p_scatter[light], 1.0e-4f) * pdf_omega, 1.0f,
                    state.nee_path_length[light] + r2);
            }
        } else {
            const BdptConnectionRow connection = bdpt_connection_row(
                state.origin + light * 3,
                state.direction + light * 3,
                light_depth,
                state.field_real + light * 3,
                state.field_imag + light * 3,
                tx_power[tx],
                state.pdf_forward[light],
                state.path_length[light],
                rx_positions + rx * 3,
                rx_polarization + rx * 3,
                1.0f,
                true,
                frequency_hz,
                inv_samples_per_tx);
            if (connection.valid) {
                const float weight = bdpt_single_strategy_mis_weight(connection.pdf, mode_id, beta);
                atomicAdd(
                    transmission + tx * rx_count + rx,
                    static_cast<double>(connection.contribution) * static_cast<double>(weight));
                mixed_row = true;
                if constexpr (kTangent) {
                    const WavefrontMixedCarrier carrier = wavefront_mixed_carrier(
                        state.origin + light * 3, state.field_real + light * 3, state.field_imag + light * 3,
                        state.path_length[light], rx_positions + rx * 3, rx_polarization + rx * 3, frequency_hz);
                    const utd::Complex3 t_field = {
                        utd::cplx(tangent.field_real[light * 3], tangent.field_imag[light * 3]),
                        utd::cplx(tangent.field_real[light * 3 + 1], tangent.field_imag[light * 3 + 1]),
                        utd::cplx(tangent.field_real[light * 3 + 2], tangent.field_imag[light * 3 + 2])};
                    const utd::Complex t_propagation =
                        utd::cplx_mul_real(carrier.d_propagation_df, tangent.frequency);
                    const utd::Complex3 t_received = utd::c3_add(
                        utd::c3_scale(t_field, carrier.propagation), utd::c3_scale(carrier.field, t_propagation));
                    const utd::Complex t_coeff = transport::complex3_dot_real(t_received, carrier.rx_axis);
                    const float coeff_power = utd::cplx_abs_sqr(carrier.coeff);
                    const float t_coeff_power = 2.0f * (carrier.coeff.re * t_coeff.re + carrier.coeff.im * t_coeff.im);
                    const float t_power = tangent.tx_power != nullptr ? tangent.tx_power[tx] : 0.0f;
                    const float t_row =
                        inv_samples_per_tx * (t_power * coeff_power + tx_power[tx] * t_coeff_power) * weight;
                    atomicAdd(t_transmission + tx * rx_count + rx, static_cast<double>(t_row));
                }
                bdpt_wavefront_write_row(
                    row_out, key, static_cast<int>(tx), static_cast<int>(rx),
                    bdpt_component_from_mask(state.component_mask[light] & 0x1f), light_depth,
                    connection.contribution, connection.pdf, weight, connection.total_distance);
            }
        }
    }
    const int block_mixed = __syncthreads_count(mixed_row);
    const int block_scatter = __syncthreads_count(scatter_row);
    if (threadIdx.x == 0) {
        if (block_mixed > 0) atomicAdd(rows, static_cast<unsigned long long>(block_mixed));
        if (block_scatter > 0) atomicAdd(rows + 1, static_cast<unsigned long long>(block_scatter));
    }
}

// Reverse connection stage of one bounce and receiver tile: one thread per light
// vertex walks the tile's receivers in order and, for each visible row with a
// nonzero upstream cotangent, adds the row's cotangent of the vertex field
// (mixed chains) or of P_te/P_tm (scatter vertices) into the tape slot of this
// bounce, plus its direct table, frequency and transmitter-power gradients.
__global__ void bdpt_wavefront_connect_backward_kernel(
    int64_t count,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t rx_count,
    int light_depth,
    const bool* active,
    const bool* visible,
    wf::State state,
    wf::ScatterTables tables,
    BdptNeeConstants constants,
    const float* tx_power,
    const float* rx_positions,
    const float* rx_polarization,
    float frequency_hz,
    float inv_samples_per_tx,
    int mode_id,
    float beta,
    wf::Tape tape,
    wf::Gradients grads) {
    const int64_t light = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (light >= count) {
        return;
    }
    const int64_t tx = (first + light) / samples;
    const float power = tx_power[tx];
    const bool scatter_vertex = (state.component_mask[light] & wf::kMaskScatterVertex) != 0;
    const int64_t slot = static_cast<int64_t>(light_depth - 1) * tape.capacity + light;
    utd::Complex3 g_field = {};
    float g_p_te = 0.0f;
    float g_p_tm = 0.0f;
    float g_frequency = 0.0f;
    float g_power = 0.0f;
    for (int64_t j = 0; j < rx_tile && rx_begin + j < rx_count; ++j) {
        const int64_t pair = light * rx_tile + j;
        if (!(active[pair] && visible[pair])) {
            continue;
        }
        const int64_t rx = rx_begin + j;
        if (scatter_vertex) {
            const float upstream = grads.grad_scattering[tx * rx_count + rx];
            float wo[3];
            float r2;
            float contribution;
            if (upstream == 0.0f ||
                !bdpt_nee_row(state, tables, light, rx_positions + rx * 3, power, constants, wo, r2, contribution)) {
                continue;
            }
            float wo_local[3];
            const float factor = wavefront_nee_factor(state, tables, light, rx_positions + rx * 3, constants, wo_local);
            const int table_slot = tables.material_slot[state.nee_material[light]];
            const wf::TableView table = wf::table_view(tables, table_slot);
            rayd::shared::scattering::TableEvalGrad eval;
            rayd::shared::scattering::eval_te_tm_grad(
                table.f_te, table.f_tm, table.nti, table.npi, table.nto, table.npo, state.nee_wi_local + light * 3,
                wo_local, eval);
            const float scaled = upstream * power * factor;
            g_p_te += scaled * eval.te;
            g_p_tm += scaled * eval.tm;
            if (eval.active) {
                const int64_t offset = tables.table_offset[table_slot];
                const float te_weight = scaled * state.nee_p_te[light];
                const float tm_weight = scaled * state.nee_p_tm[light];
                for (int k = 0; k < 16; ++k) {
                    if (grads.f_te != nullptr && te_weight * eval.cw[k] != 0.0f)
                        atomicAdd(grads.f_te + offset + eval.idx[k], static_cast<double>(te_weight * eval.cw[k]));
                    if (grads.f_tm != nullptr && tm_weight * eval.cw[k] != 0.0f)
                        atomicAdd(grads.f_tm + offset + eval.idx[k], static_cast<double>(tm_weight * eval.cw[k]));
                }
            }
            g_frequency += -2.0f * upstream * contribution / frequency_hz;
            g_power += upstream * (state.nee_p_te[light] * eval.te + state.nee_p_tm[light] * eval.tm) * factor;
        } else {
            const float upstream = grads.grad_transmission[tx * rx_count + rx];
            if (upstream == 0.0f) {
                continue;
            }
            const BdptConnectionRow connection = bdpt_connection_row(
                state.origin + light * 3, state.direction + light * 3, light_depth, state.field_real + light * 3,
                state.field_imag + light * 3, power, state.pdf_forward[light], state.path_length[light],
                rx_positions + rx * 3, rx_polarization + rx * 3, 1.0f, true, frequency_hz, inv_samples_per_tx);
            if (!connection.valid) {
                continue;
            }
            const float weight = bdpt_single_strategy_mis_weight(connection.pdf, mode_id, beta);
            const WavefrontMixedCarrier carrier = wavefront_mixed_carrier(
                state.origin + light * 3, state.field_real + light * 3, state.field_imag + light * 3,
                state.path_length[light], rx_positions + rx * 3, rx_polarization + rx * 3, frequency_hz);
            const float unit = upstream * inv_samples_per_tx * weight;
            const float scaled = unit * power;
            // row = scaled |c|^2 with c = sum_k rx_axis_k propagation F_k:
            // cotangent of F_k is 2 c conj(propagation) rx_axis_k.
            const utd::Complex c = carrier.coeff;
            const utd::Complex p = carrier.propagation;
            const utd::Complex c_conj_p =
                utd::cplx(2.0f * scaled * (c.re * p.re + c.im * p.im), 2.0f * scaled * (c.im * p.re - c.re * p.im));
            g_field.x = utd::cplx_add(g_field.x, utd::cplx_mul_real(c_conj_p, carrier.rx_axis.x));
            g_field.y = utd::cplx_add(g_field.y, utd::cplx_mul_real(c_conj_p, carrier.rx_axis.y));
            g_field.z = utd::cplx_add(g_field.z, utd::cplx_mul_real(c_conj_p, carrier.rx_axis.z));
            const utd::Complex d_coeff = transport::complex3_dot_real(
                utd::c3_scale(carrier.field, carrier.d_propagation_df), carrier.rx_axis);
            g_frequency += 2.0f * scaled * (c.re * d_coeff.re + c.im * d_coeff.im);
            g_power += unit * utd::cplx_abs_sqr(c);
        }
    }
    tape.g_field_real[slot * 3] += g_field.x.re;
    tape.g_field_real[slot * 3 + 1] += g_field.y.re;
    tape.g_field_real[slot * 3 + 2] += g_field.z.re;
    tape.g_field_imag[slot * 3] += g_field.x.im;
    tape.g_field_imag[slot * 3 + 1] += g_field.y.im;
    tape.g_field_imag[slot * 3 + 2] += g_field.z.im;
    tape.g_p[slot * 2] += g_p_te;
    tape.g_p[slot * 2 + 1] += g_p_tm;
    if (grads.frequency != nullptr && g_frequency != 0.0f) {
        atomicAdd(grads.frequency, static_cast<double>(g_frequency));
    }
    if (grads.tx_power != nullptr && g_power != 0.0f) {
        atomicAdd(grads.tx_power + tx, static_cast<double>(g_power));
    }
}

__global__ void bdpt_wavefront_cast_kernel(int64_t count, const double* values, float* out) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = static_cast<float>(values[index]);
    }
}

__global__ void bdpt_wavefront_finalize_kernel(
    int64_t count, const double* accumulated, const float* base, float* out) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    out[index] = static_cast<float>(accumulated[index] + static_cast<double>(base[index]));
}

}  // namespace

void channel_bdpt_wavefront_visibility_inputs_cuda(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t light_depth,
    const at::Tensor& alive,
    const wf::State& state,
    const wf::ScatterTables& tables,
    double amplitude_sq,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    at::Tensor& start,
    at::Tensor& end,
    at::Tensor& active) {
    if (pairs == 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
    const BdptNeeConstants constants{
        static_cast<float>(amplitude_sq), static_cast<float>(4.0 * 3.14159265358979323846),
        static_cast<float>(samples)};
    bdpt_wavefront_visibility_inputs_kernel<<<blocks, threads, 0, stream>>>(
        pairs,
        rx_tile,
        rx_begin,
        first,
        samples,
        static_cast<int>(light_depth),
        reinterpret_cast<const unsigned long long*>(alive.data_ptr<int64_t>()),
        state,
        tables,
        constants,
        tx_power.data_ptr<float>(),
        rx_positions.data_ptr<float>(),
        start.data_ptr<float>(),
        end.data_ptr<float>(),
        active.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void channel_bdpt_wavefront_connect_accumulate_cuda(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t light_depth,
    const at::Tensor& active,
    const at::Tensor& visible,
    const wf::State& state,
    const wf::ScatterTables& tables,
    double amplitude_sq,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    int64_t mode_id,
    double beta,
    at::Tensor& transmission,
    at::Tensor& scattering,
    at::Tensor& rows,
    const std::vector<at::Tensor>& row_out,
    int64_t max_depth,
    const wf::Tangent* tangent,
    at::Tensor* t_transmission,
    at::Tensor* t_scattering) {
    if (pairs == 0) {
        return;
    }
    // row_out: cursor, key, tx, rx, component, depth, contribution, pdf, mis, path length.
    BdptWavefrontRows row_view{};
    if (!row_out.empty()) {
        row_view = {
            row_out[1].size(0),
            reinterpret_cast<unsigned long long*>(row_out[0].data_ptr<int64_t>()),
            row_out[1].data_ptr<int64_t>(),
            row_out[2].data_ptr<int>(),
            row_out[3].data_ptr<int>(),
            row_out[4].data_ptr<int>(),
            row_out[5].data_ptr<int>(),
            row_out[6].data_ptr<float>(),
            row_out[7].data_ptr<float>(),
            row_out[8].data_ptr<float>(),
            row_out[9].data_ptr<float>(),
        };
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
    const BdptNeeConstants constants{
        static_cast<float>(amplitude_sq), static_cast<float>(4.0 * 3.14159265358979323846),
        static_cast<float>(samples)};
    const wf::Tangent no_tangent{};
    auto kernel = tangent != nullptr ? bdpt_wavefront_connect_accumulate_kernel<true>
                                     : bdpt_wavefront_connect_accumulate_kernel<false>;
    kernel<<<blocks, threads, 0, stream>>>(
        pairs,
        rx_tile,
        rx_begin,
        first,
        samples,
        rx_positions.size(0),
        static_cast<int>(light_depth),
        active.data_ptr<bool>(),
        visible.data_ptr<bool>(),
        state,
        tables,
        constants,
        tx_power.data_ptr<float>(),
        rx_positions.data_ptr<float>(),
        rx_polarization.data_ptr<float>(),
        static_cast<float>(frequency_hz),
        1.0f / static_cast<float>(samples),
        static_cast<int>(mode_id),
        static_cast<float>(beta),
        transmission.data_ptr<double>(),
        scattering.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(rows.data_ptr<int64_t>()),
        row_view,
        static_cast<int>(max_depth),
        tangent != nullptr ? *tangent : no_tangent,
        tangent != nullptr ? t_transmission->data_ptr<double>() : nullptr,
        tangent != nullptr ? t_scattering->data_ptr<double>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void channel_bdpt_wavefront_connect_backward_cuda(
    int64_t count,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t light_depth,
    const at::Tensor& active,
    const at::Tensor& visible,
    const wf::State& state,
    const wf::ScatterTables& tables,
    double amplitude_sq,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    int64_t mode_id,
    double beta,
    const wf::Tape& tape,
    const wf::Gradients& grads) {
    if (count == 0) {
        return;
    }
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
    const BdptNeeConstants constants{
        static_cast<float>(amplitude_sq), static_cast<float>(4.0 * 3.14159265358979323846),
        static_cast<float>(samples)};
    bdpt_wavefront_connect_backward_kernel<<<blocks, threads, 0, stream>>>(
        count,
        rx_tile,
        rx_begin,
        first,
        samples,
        rx_positions.size(0),
        static_cast<int>(light_depth),
        active.data_ptr<bool>(),
        visible.data_ptr<bool>(),
        state,
        tables,
        constants,
        tx_power.data_ptr<float>(),
        rx_positions.data_ptr<float>(),
        rx_polarization.data_ptr<float>(),
        static_cast<float>(frequency_hz),
        1.0f / static_cast<float>(samples),
        static_cast<int>(mode_id),
        static_cast<float>(beta),
        tape,
        grads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Float copy of a double gradient accumulator.
at::Tensor channel_bdpt_wavefront_cast_cuda(const at::Tensor& values) {
    auto out = at::empty(values.sizes(), values.options().dtype(at::kFloat));
    const int64_t count = out.numel();
    if (count > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device()).stream();
        bdpt_wavefront_cast_kernel<<<blocks, threads, 0, stream>>>(
            count, values.data_ptr<double>(), out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// Zero-filled CUDA storage for the wavefront accumulators and counters.
at::Tensor channel_bdpt_wavefront_zeros_cuda(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(), 0, static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream()));
    }
    return tensor;
}

at::Tensor channel_bdpt_wavefront_finalize_cuda(const at::Tensor& accumulated, const at::Tensor& base) {
    auto out = at::empty(base.sizes(), base.options());
    const int64_t count = out.numel();
    if (count > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(base.get_device()).stream();
        bdpt_wavefront_finalize_kernel<<<blocks, threads, 0, stream>>>(
            count, accumulated.data_ptr<double>(), base.data_ptr<float>(), out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

namespace {

__global__ void bdpt_wavefront_pack_rows_kernel(
    int64_t count,
    const int* order,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const int* light_depth,
    const float* contribution,
    const float* pdf,
    const float* mis_weight,
    const float* path_length,
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
    float* out_path_length) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int row = order[index];
    out_topology[index * 4 + 0] = tx_id[row];
    out_topology[index * 4 + 1] = rx_id[row];
    out_topology[index * 4 + 2] = component_id[row];
    out_topology[index * 4 + 3] = light_depth[row];
    out_contribution[index] = contribution[row];
    out_pdf[index] = pdf[row];
    out_mis_weight[index] = mis_weight[row];
    out_component_id[index] = component_id[row];
    out_valid[index] = true;
    out_tx_id[index] = tx_id[row];
    out_rx_id[index] = rx_id[row];
    out_grid_linear_id[index] = rx_id[row];
    out_light_depth[index] = light_depth[row];
    out_sensor_depth[index] = 0;
    out_path_length[index] = path_length[row];
}

__global__ void bdpt_iota_kernel(int64_t count, int* out) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = static_cast<int>(index);
    }
}

}  // namespace

// Sort the wavefront rows by their (sample, bounce, receiver) key and pack the
// 12-field connection-sample schema in that stable order.
std::vector<at::Tensor> channel_bdpt_wavefront_pack_rows_cuda(const std::vector<at::Tensor>& row_out) {
    const int64_t count = row_out[1].size(0);
    TORCH_CHECK(count <= std::numeric_limits<int>::max(), "wavefront rows support at most 2^31-1 rows");
    const at::Tensor& reference = row_out[6];
    auto [topology, contribution, pdf, mis_weight, component_id, valid, tx_id, rx_id, grid_linear_id, light_depth,
          sensor_depth, path_length] = allocate_connection_samples(reference, count);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        auto rows_in = at::empty({count}, reference.options().dtype(at::kInt));
        auto order = at::empty({count}, reference.options().dtype(at::kInt));
        auto keys_out = at::empty({count}, reference.options().dtype(at::kLong));
        bdpt_iota_kernel<<<blocks, threads, 0, stream>>>(count, rows_in.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        size_t temp_bytes = 0;
        const auto* keys_in = reinterpret_cast<const unsigned long long*>(row_out[1].data_ptr<int64_t>());
        auto* keys_sorted = reinterpret_cast<unsigned long long*>(keys_out.data_ptr<int64_t>());
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes, keys_in, keys_sorted, rows_in.data_ptr<int>(), order.data_ptr<int>(),
            static_cast<int>(count), 0, 64, stream));
        auto temp = at::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))},
                              reference.options().dtype(at::kByte));
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp.data_ptr(), temp_bytes, keys_in, keys_sorted, rows_in.data_ptr<int>(), order.data_ptr<int>(),
            static_cast<int>(count), 0, 64, stream));
        bdpt_wavefront_pack_rows_kernel<<<blocks, threads, 0, stream>>>(
            count, order.data_ptr<int>(), row_out[2].data_ptr<int>(), row_out[3].data_ptr<int>(),
            row_out[4].data_ptr<int>(), row_out[5].data_ptr<int>(), row_out[6].data_ptr<float>(),
            row_out[7].data_ptr<float>(), row_out[8].data_ptr<float>(), row_out[9].data_ptr<float>(),
            topology.data_ptr<int>(), contribution.data_ptr<float>(), pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(), component_id.data_ptr<int>(), valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), grid_linear_id.data_ptr<int>(),
            light_depth.data_ptr<int>(), sensor_depth.data_ptr<int>(), path_length.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {topology, contribution, pdf, mis_weight, component_id, valid, tx_id, rx_id, grid_linear_id, light_depth,
            sensor_depth, path_length};
}
