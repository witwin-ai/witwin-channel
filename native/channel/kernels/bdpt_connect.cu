// Copyright Xingyu Chen.
// Implements BDPT connection CUDA operations.

// ==== Section: BDPT MIS ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"

#include <rayd/field_transport.cuh>

#include <algorithm>
#include <cmath>
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

__device__ float bdpt_free_space_gain(float tx_power, float distance, float frequency_hz) {
    const float wavelength = kLightSpeedMPerS / fmaxf(frequency_hz, 1.0f);
    const float denom = 4.0f * kPi * fmaxf(distance, 1.0e-6f) / wavelength;
    return tx_power / fmaxf(denom * denom, 1.0e-30f);
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

namespace {

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

}  // namespace

at::Tensor channel_bdpt_mis_weights_cuda(
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

// ==== Section: BDPT connection samples ====


namespace {

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

int64_t channel_bdpt_count_valid_connection_samples_cuda(at::Tensor valid) {
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

__global__ void bdpt_accumulate_power_jvp_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* mis_weight,
    const float* tangent_contribution,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
    float* t_path_gain,
    float* t_los,
    float* t_reflection,
    float* t_diffraction,
    float* t_transmission,
    float* t_scattering) {
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
        const double value = static_cast<double>(mis_weight[index]) *
            static_cast<double>(tangent_contribution[index]);
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
    t_path_gain[out_index] = static_cast<float>(path_sum);
    t_los[out_index] = static_cast<float>(los_sum);
    t_reflection[out_index] = static_cast<float>(reflection_sum);
    t_diffraction[out_index] = static_cast<float>(diffraction_sum);
    t_transmission[out_index] = static_cast<float>(transmission_sum);
    t_scattering[out_index] = static_cast<float>(scattering_sum);
}

__global__ void bdpt_accumulate_coherent_jvp_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* tangent_coeff_real,
    const float* tangent_coeff_imag,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
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
    int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_index >= out_count) {
        return;
    }
    const int out_tx = static_cast<int>(out_index / rx_count);
    const int out_rx = static_cast<int>(out_index - static_cast<int64_t>(out_tx) * rx_count);
    // Fixed-order tangent bin sums t_S_c per component (deterministic, no atomics).
    double t_los_re = 0.0, t_los_im = 0.0;
    double t_reflection_re = 0.0, t_reflection_im = 0.0;
    double t_diffraction_re = 0.0, t_diffraction_im = 0.0;
    double t_transmission_re = 0.0, t_transmission_im = 0.0;
    double t_scattering_re = 0.0, t_scattering_im = 0.0;
    for (int64_t index = 0; index < sample_count; ++index) {
        if (!valid[index] || tx_id[index] != out_tx || rx_id[index] != out_rx) {
            continue;
        }
        const int component = component_id[index];
        if (!bdpt_component_accumulable(component)) {
            continue;
        }
        const double tr = static_cast<double>(tangent_coeff_real[index]);
        const double ti = static_cast<double>(tangent_coeff_imag[index]);
        if (component == kComponentLos) {
            t_los_re += tr;
            t_los_im += ti;
        } else if (component == kComponentReflection) {
            t_reflection_re += tr;
            t_reflection_im += ti;
        } else if (component == kComponentDiffraction) {
            t_diffraction_re += tr;
            t_diffraction_im += ti;
        } else if (component == kComponentTransmission) {
            t_transmission_re += tr;
            t_transmission_im += ti;
        } else if (component == kComponentScattering) {
            t_scattering_re += tr;
            t_scattering_im += ti;
        }
    }
    // t_P_c = 2 Re(conj(S_c) t_S_c); path_gain tangent sums the component powers.
    const double tp_los =
        2.0 * (los_re[out_index] * t_los_re + los_im[out_index] * t_los_im);
    const double tp_reflection = 2.0 *
        (reflection_re[out_index] * t_reflection_re +
         reflection_im[out_index] * t_reflection_im);
    const double tp_diffraction = 2.0 *
        (diffraction_re[out_index] * t_diffraction_re +
         diffraction_im[out_index] * t_diffraction_im);
    const double tp_transmission = 2.0 *
        (transmission_re[out_index] * t_transmission_re +
         transmission_im[out_index] * t_transmission_im);
    const double tp_scattering = 2.0 *
        (scattering_re[out_index] * t_scattering_re +
         scattering_im[out_index] * t_scattering_im);
    t_los[out_index] = static_cast<float>(tp_los);
    t_reflection[out_index] = static_cast<float>(tp_reflection);
    t_diffraction[out_index] = static_cast<float>(tp_diffraction);
    t_transmission[out_index] = static_cast<float>(tp_transmission);
    t_scattering[out_index] = static_cast<float>(tp_scattering);
    t_path_gain[out_index] = static_cast<float>(
        tp_los + tp_reflection + tp_diffraction + tp_transmission + tp_scattering);
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
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            bdpt_accumulate_power_jvp_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                mis_weight.data_ptr<float>(),
                tc->data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
                t_path_gain.data_ptr<float>(),
                t_los.data_ptr<float>(),
                t_reflection.data_ptr<float>(),
                t_diffraction.data_ptr<float>(),
                t_transmission.data_ptr<float>(),
                t_scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
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
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            bdpt_accumulate_coherent_jvp_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                tcr->data_ptr<float>(),
                tci->data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
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
