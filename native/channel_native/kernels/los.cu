#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>

#include <tuple>
#include <vector>

namespace {

constexpr double kLightSpeedMetersPerSecond = 299792458.0;
constexpr int kLosBlockSize = 256;

void check_cuda_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_tensor_rank(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
}

float los_gain_scale(double frequency_hz) {
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const double wavelength = kLightSpeedMetersPerSecond / frequency_hz;
    const double scale = wavelength / 12.566370614359172;
    return static_cast<float>(scale * scale);
}

// d(gain_scale)/d(frequency): gain_scale = (c / (4 pi f))^2, so the frequency
// derivative is -2 * gain_scale / f (computed in double before narrowing).
float los_gain_scale_dfreq(double frequency_hz) {
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const double wavelength = kLightSpeedMetersPerSecond / frequency_hz;
    const double scale = wavelength / 12.566370614359172;
    return static_cast<float>(-2.0 * scale * scale / frequency_hz);
}

__global__ void path_los_export_kernel(
    const float *__restrict__ tx_positions,
    const float *__restrict__ tx_power,
    const float *__restrict__ rx_positions,
    int *__restrict__ tx_id,
    int *__restrict__ rx_id,
    float *__restrict__ path_length,
    float *__restrict__ delay,
    float *__restrict__ path_gain,
    float *__restrict__ path_gain_matrix,
    int64_t tx_count,
    int64_t rx_count,
    float wavelength) {
    const int64_t path_count = tx_count * rx_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t path = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         path < path_count;
         path += stride) {
        const int rx = static_cast<int>(path / tx_count);
        const int tx = static_cast<int>(path - static_cast<int64_t>(rx) * tx_count);
        const float *tx_p = tx_positions + static_cast<int64_t>(tx) * 3;
        const float *rx_p = rx_positions + static_cast<int64_t>(rx) * 3;
        const float dx = tx_p[0] - rx_p[0];
        const float dy = tx_p[1] - rx_p[1];
        const float dz = tx_p[2] - rx_p[2];
        const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1.0e-6f);
        const float gain_base = wavelength / (12.566370614359172f * distance);

        tx_id[path] = tx;
        rx_id[path] = rx;
        path_length[path] = distance;
        delay[path] = distance / static_cast<float>(kLightSpeedMetersPerSecond);
        const float gain = tx_power[tx] * gain_base * gain_base;
        path_gain[path] = gain;
        path_gain_matrix[static_cast<int64_t>(tx) * rx_count + rx] = gain;
    }
}

__global__ void los_path_gain_backward_kernel(
    const float *__restrict__ tx_positions,
    const float *__restrict__ tx_power,
    const float *__restrict__ rx_positions,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_tx,
    float *__restrict__ grad_power,
    float *__restrict__ grad_rx,
    float *__restrict__ grad_frequency,
    int64_t tx_count,
    int64_t rx_count,
    int64_t grad_stride0,
    int64_t grad_stride1,
    float gain_scale,
    float gain_scale_dfreq) {
    const int64_t pair_count = tx_count * rx_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        const int64_t tx = pair / rx_count;
        const int64_t rx = pair - tx * rx_count;
        const float go = grad_output[tx * grad_stride0 + rx * grad_stride1];
        const float *tx_p = tx_positions + tx * 3;
        const float *rx_p = rx_positions + rx * 3;
        const float dx = tx_p[0] - rx_p[0];
        const float dy = tx_p[1] - rx_p[1];
        const float dz = tx_p[2] - rx_p[2];
        const float distance_sq = dx * dx + dy * dy + dz * dz;
        const float distance = sqrtf(distance_sq);
        const float safe_distance = fmaxf(distance, 1.0e-6f);
        const float inv_d2 = 1.0f / (safe_distance * safe_distance);
        atomicAdd(grad_power + tx, go * gain_scale * inv_d2);
        if (grad_frequency != nullptr) {
            // gain = P * gain_scale(f) / d^2; only gain_scale carries f.
            atomicAdd(grad_frequency, go * tx_power[tx] * gain_scale_dfreq * inv_d2);
        }
        // Position gradients follow the clamp_min pass-through convention
        // (>= keeps the boundary subgradient, matching the tests/ad oracle).
        if (distance < 1.0e-6f) {
            continue;
        }
        const float coeff = go * 2.0f * tx_power[tx] * gain_scale * inv_d2 * inv_d2;
        const float gx = coeff * dx;
        const float gy = coeff * dy;
        const float gz = coeff * dz;
        atomicAdd(grad_rx + rx * 3 + 0, gx);
        atomicAdd(grad_rx + rx * 3 + 1, gy);
        atomicAdd(grad_rx + rx * 3 + 2, gz);
        atomicAdd(grad_tx + tx * 3 + 0, -gx);
        atomicAdd(grad_tx + tx * 3 + 1, -gy);
        atomicAdd(grad_tx + tx * 3 + 2, -gz);
    }
}

__global__ void los_path_gain_jvp_kernel(
    const float *__restrict__ tx_positions,
    const float *__restrict__ tx_power,
    const float *__restrict__ rx_positions,
    const float *__restrict__ tx_tangent,
    const float *__restrict__ power_tangent,
    const float *__restrict__ rx_tangent,
    float *__restrict__ out,
    int64_t tx_count,
    int64_t rx_count,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    float gain_scale,
    float gain_scale_dfreq_tangent) {
    const int64_t pair_count = tx_count * rx_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        const int64_t tx = pair / rx_count;
        const int64_t rx = pair - tx * rx_count;
        const float *tx_p = tx_positions + tx * 3;
        const float *rx_p = rx_positions + rx * 3;
        const float dx = tx_p[0] - rx_p[0];
        const float dy = tx_p[1] - rx_p[1];
        const float dz = tx_p[2] - rx_p[2];
        const float distance_sq = dx * dx + dy * dy + dz * dz;
        const float distance = sqrtf(distance_sq);
        const float safe_distance = fmaxf(distance, 1.0e-6f);
        const float inv_d2 = 1.0f / (safe_distance * safe_distance);
        float tangent = 0.0f;
        if (has_power_tangent) {
            tangent += power_tangent[tx] * gain_scale * inv_d2;
        }
        // Frequency tangent, pre-folded on the host into
        // d(gain_scale)/df * t_f so a zero tangent costs one fma.
        tangent += tx_power[tx] * gain_scale_dfreq_tangent * inv_d2;
        if (distance >= 1.0e-6f) {
            const float coeff = 2.0f * tx_power[tx] * gain_scale * inv_d2 * inv_d2;
            if (has_rx_tangent) {
                const float *rx_t = rx_tangent + rx * 3;
                tangent += coeff * (dx * rx_t[0] + dy * rx_t[1] + dz * rx_t[2]);
            }
            if (has_tx_tangent) {
                const float *tx_t = tx_tangent + tx * 3;
                tangent -= coeff * (dx * tx_t[0] + dy * tx_t[1] + dz * tx_t[2]);
            }
        }
        out[tx * rx_count + rx] = tangent;
    }
}

__global__ void los_component_maps_kernel(
    const float *__restrict__ los,
    float *__restrict__ maps,
    int64_t tx_count,
    int64_t rows,
    int64_t cols) {
    const int64_t element_count = tx_count * rows * cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < element_count;
         idx += stride) {
        const int64_t cell = idx % (rows * cols);
        const int64_t tx = idx / (rows * cols);
        const int64_t row = cell / cols;
        const int64_t col = cell - row * cols;
        maps[(tx * cols + col) * rows + row] = los[idx];
    }
}

__global__ void los_component_maps_public_layout_kernel(
    const float *__restrict__ los,
    float *__restrict__ maps,
    int64_t element_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < element_count;
         idx += stride) {
        maps[idx] = los[idx];
    }
}

__global__ void apply_los_visibility_kernel(
    const float *__restrict__ los,
    const bool *__restrict__ visible,
    float *__restrict__ maps,
    int64_t tx_index,
    int64_t rows,
    int64_t cols) {
    const int64_t cell_count = rows * cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        const int64_t row = cell / cols;
        const int64_t col = cell - row * cols;
        const float gain = visible[cell] ? los[(tx_index * rows + row) * cols + col] : 0.0f;
        maps[(tx_index * cols + col) * rows + row] = gain;
    }
}

__global__ void apply_los_visibility_public_layout_kernel(
    const float *__restrict__ los,
    const bool *__restrict__ visible,
    float *__restrict__ maps,
    int64_t tx_index,
    int64_t rows,
    int64_t cols) {
    const int64_t cell_count = rows * cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        const int64_t row = cell / cols;
        const int64_t col = cell - row * cols;
        const float gain = visible[cell] ? los[(tx_index * rows + row) * cols + col] : 0.0f;
        maps[(tx_index * rows + row) * cols + col] = gain;
    }
}

// Adjoint of the (visibility-masked) LoS component-map layout: the forward is
// maps[tx, col, row] = visible[tx, cell] * los[tx, cell] with
// cell = row * cols + col (identity mask when visible == nullptr), a pure
// permutation times a frozen 0/1 mask, so the adjoint gathers the map
// cotangent back into matrix layout under the same mask.
__global__ void los_component_maps_adjoint_kernel(
    const float *__restrict__ grad_maps,
    const bool *__restrict__ visible,
    float *__restrict__ grad_matrix,
    int64_t tx_count,
    int64_t rows,
    int64_t cols,
    int64_t grad_stride0,
    int64_t grad_stride1,
    int64_t grad_stride2) {
    const int64_t element_count = tx_count * rows * cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < element_count;
         idx += stride) {
        const int64_t cell = idx % (rows * cols);
        const int64_t tx = idx / (rows * cols);
        const int64_t row = cell / cols;
        const int64_t col = cell - row * cols;
        const bool pass =
            visible == nullptr || visible[tx * rows * cols + cell];
        grad_matrix[idx] =
            pass ? grad_maps[tx * grad_stride0 + col * grad_stride1 + row * grad_stride2]
                 : 0.0f;
    }
}

__global__ void los_visibility_inputs_kernel(
    const float *__restrict__ tx_positions,
    float *__restrict__ start,
    bool *__restrict__ active,
    int64_t tx_index,
    int64_t rx_count) {
    const float tx_x = tx_positions[tx_index * 3 + 0];
    const float tx_y = tx_positions[tx_index * 3 + 1];
    const float tx_z = tx_positions[tx_index * 3 + 2];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t rx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         rx < rx_count;
         rx += stride) {
        float *out = start + rx * 3;
        out[0] = tx_x;
        out[1] = tx_y;
        out[2] = tx_z;
        active[rx] = true;
    }
}

__global__ void receiver_grid_points_kernel(
    float *__restrict__ points,
    int64_t rows,
    int64_t cols,
    float origin_x,
    float origin_y,
    float origin_z,
    float x_axis_x,
    float x_axis_y,
    float x_axis_z,
    float y_axis_x,
    float y_axis_y,
    float y_axis_z,
    float spacing0,
    float spacing1) {
    const int64_t count = rows * cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        const int64_t row = idx / cols;
        const int64_t col = idx - row * cols;
        const float u = static_cast<float>(row) * spacing0;
        const float v = static_cast<float>(col) * spacing1;
        float *out = points + idx * 3;
        out[0] = origin_x + x_axis_x * u + y_axis_x * v;
        out[1] = origin_y + x_axis_y * u + y_axis_y * v;
        out[2] = origin_z + x_axis_z * u + y_axis_z * v;
    }
}

__global__ void pack_vec3_kernel(
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ z,
    float *__restrict__ out,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        float *dst = out + idx * 3;
        dst[0] = x[idx];
        dst[1] = y[idx];
        dst[2] = z[idx];
    }
}

using LosExport =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

LosExport los_export_cuda_impl(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.size(0);
    const int64_t path_count = tx_count * rx_count;
    auto float_options = tx_positions.options();
    auto int_options = tx_positions.options().dtype(at::kInt);
    auto tx_id = at::empty({path_count}, int_options);
    auto rx_id = at::empty({path_count}, int_options);
    auto path_length = at::empty({path_count}, float_options);
    auto delay = at::empty({path_count}, float_options);
    auto path_gain = at::empty({path_count}, float_options);
    auto path_gain_matrix = at::empty({tx_count, rx_count}, float_options);

    if (path_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((path_count + kLosBlockSize - 1) / kLosBlockSize);
        const float wavelength = static_cast<float>(kLightSpeedMetersPerSecond / frequency_hz);
        path_los_export_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            path_gain_matrix.data_ptr<float>(),
            tx_count,
            rx_count,
            wavelength);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {tx_id, rx_id, path_length, delay, path_gain, path_gain_matrix};
}

at::Tensor launch_los_component_maps(
    const at::Tensor &los,
    int64_t rows,
    int64_t cols) {
    const int64_t tx_count = los.size(0);
    auto maps = at::empty({tx_count, cols, rows}, los.options());
    const int64_t element_count = los.numel();
    if (element_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int block_count = static_cast<int>((element_count + kLosBlockSize - 1) / kLosBlockSize);
        los_component_maps_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            los.data_ptr<float>(),
            maps.data_ptr<float>(),
            tx_count,
            rows,
            cols);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

at::Tensor los_component_maps_cuda_impl(at::Tensor los) {
    check_cuda_tensor(los, "los", at::kFloat, 3);
    return launch_los_component_maps(los, los.size(1), los.size(2));
}

at::Tensor los_component_maps_from_matrix_cuda_impl(
    at::Tensor los,
    int64_t rows,
    int64_t cols) {
    check_cuda_tensor(los, "los", at::kFloat, 2);
    TORCH_CHECK(rows >= 0 && cols >= 0, "rows and cols must be non-negative");
    TORCH_CHECK(los.size(1) == rows * cols, "los columns must match rows * cols");
    return launch_los_component_maps(los, rows, cols);
}

std::tuple<at::Tensor, at::Tensor> los_visibility_inputs_cuda_impl(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");
    TORCH_CHECK(rx_count >= 0, "rx_count must be non-negative");

    auto start = at::empty({rx_count, 3}, tx_positions.options());
    auto active = at::empty({rx_count}, tx_positions.options().dtype(at::kBool));
    if (rx_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((rx_count + kLosBlockSize - 1) / kLosBlockSize);
        los_visibility_inputs_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            start.data_ptr<float>(),
            active.data_ptr<bool>(),
            tx_index,
            rx_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {start, active};
}

at::Tensor receiver_grid_points_cuda_impl(
    at::Tensor reference,
    int64_t rows,
    int64_t cols,
    double origin_x,
    double origin_y,
    double origin_z,
    double x_axis_x,
    double x_axis_y,
    double x_axis_z,
    double y_axis_x,
    double y_axis_y,
    double y_axis_z,
    double spacing0,
    double spacing1) {
    check_cuda_tensor(reference, "reference", at::kFloat, 2);
    TORCH_CHECK(rows >= 0 && cols >= 0, "rows and cols must be non-negative");
    const int64_t count = rows * cols;
    auto points = at::empty({count, 3}, reference.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        const int block_count = static_cast<int>((count + kLosBlockSize - 1) / kLosBlockSize);
        receiver_grid_points_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            points.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(origin_x),
            static_cast<float>(origin_y),
            static_cast<float>(origin_z),
            static_cast<float>(x_axis_x),
            static_cast<float>(x_axis_y),
            static_cast<float>(x_axis_z),
            static_cast<float>(y_axis_x),
            static_cast<float>(y_axis_y),
            static_cast<float>(y_axis_z),
            static_cast<float>(spacing0),
            static_cast<float>(spacing1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return points;
}

std::tuple<at::Tensor, at::Tensor> transmitter_tensors_cuda_impl(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host) {
    TORCH_CHECK(positions_host.size() % 3 == 0, "transmitter positions must be flat xyz triples");
    const int64_t tx_count = static_cast<int64_t>(power_host.size());
    TORCH_CHECK(
        static_cast<int64_t>(positions_host.size()) == tx_count * 3,
        "transmitter positions and powers must have the same length");

    const int device = c10::cuda::current_device();
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
    auto positions = at::empty({tx_count, 3}, options);
    auto power = at::empty({tx_count}, options);
    if (tx_count > 0) {
        C10_CUDA_CHECK(cudaMemcpy(
            positions.data_ptr<float>(),
            positions_host.data(),
            positions_host.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        C10_CUDA_CHECK(cudaMemcpy(
            power.data_ptr<float>(),
            power_host.data(),
            power_host.size() * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    return {positions, power};
}

at::Tensor pack_vec3_cuda_impl(at::Tensor x, at::Tensor y, at::Tensor z) {
    check_cuda_tensor(x, "x", at::kFloat, 1);
    check_cuda_tensor(y, "y", at::kFloat, 1);
    check_cuda_tensor(z, "z", at::kFloat, 1);
    TORCH_CHECK(y.size(0) == x.size(0), "y must match x length");
    TORCH_CHECK(z.size(0) == x.size(0), "z must match x length");
    const int64_t count = x.size(0);
    auto out = at::empty({count, 3}, x.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
        const int block_count = static_cast<int>((count + kLosBlockSize - 1) / kLosBlockSize);
        pack_vec3_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            z.data_ptr<float>(),
            out.data_ptr<float>(),
            count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_path_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz) {
    return los_export_cuda_impl(tx_positions, tx_power, rx_positions, frequency_hz);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_los_path_gain_backward_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor grad_output,
    double frequency_hz) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    check_cuda_tensor_rank(grad_output, "grad_output", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(grad_output.size(0) == tx_positions.size(0), "grad_output tx dimension mismatch");
    TORCH_CHECK(grad_output.size(1) == rx_positions.size(0), "grad_output rx dimension mismatch");
    TORCH_CHECK(tx_power.get_device() == tx_positions.get_device(), "tx_power must be on tx_positions device");
    TORCH_CHECK(rx_positions.get_device() == tx_positions.get_device(), "rx_positions must be on tx_positions device");
    TORCH_CHECK(grad_output.get_device() == tx_positions.get_device(), "grad_output must be on tx_positions device");

    auto grad_tx = at::empty_like(tx_positions);
    auto grad_power = at::empty_like(tx_power);
    auto grad_rx = at::empty_like(rx_positions);
    auto grad_frequency = at::empty({1}, tx_positions.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
    if (grad_tx.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(grad_tx.data_ptr<float>(), 0, grad_tx.numel() * sizeof(float), stream));
    }
    if (grad_power.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(grad_power.data_ptr<float>(), 0, grad_power.numel() * sizeof(float), stream));
    }
    if (grad_rx.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(grad_rx.data_ptr<float>(), 0, grad_rx.numel() * sizeof(float), stream));
    }
    C10_CUDA_CHECK(cudaMemsetAsync(grad_frequency.data_ptr<float>(), 0, sizeof(float), stream));

    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.size(0);
    const int64_t pair_count = tx_count * rx_count;
    if (pair_count > 0) {
        const int block_count = static_cast<int>((pair_count + kLosBlockSize - 1) / kLosBlockSize);
        los_path_gain_backward_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_tx.data_ptr<float>(),
            grad_power.data_ptr<float>(),
            grad_rx.data_ptr<float>(),
            grad_frequency.data_ptr<float>(),
            tx_count,
            rx_count,
            grad_output.stride(0),
            grad_output.stride(1),
            los_gain_scale(frequency_hz),
            los_gain_scale_dfreq(frequency_hz));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_tx, grad_power, grad_rx, grad_frequency};
}

at::Tensor cn_mc_los_path_gain_jvp_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor tx_tangent,
    at::Tensor power_tangent,
    at::Tensor rx_tangent,
    bool has_tx_tangent,
    bool has_power_tangent,
    bool has_rx_tangent,
    double frequency_hz,
    double frequency_tangent) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(tx_power.get_device() == tx_positions.get_device(), "tx_power must be on tx_positions device");
    TORCH_CHECK(rx_positions.get_device() == tx_positions.get_device(), "rx_positions must be on tx_positions device");
    if (has_tx_tangent) {
        check_cuda_tensor(tx_tangent, "tx_tangent", at::kFloat, 2);
        TORCH_CHECK(tx_tangent.sizes() == tx_positions.sizes(), "tx_tangent must match tx_positions");
        TORCH_CHECK(tx_tangent.get_device() == tx_positions.get_device(), "tx_tangent must be on tx_positions device");
    }
    if (has_power_tangent) {
        check_cuda_tensor(power_tangent, "power_tangent", at::kFloat, 1);
        TORCH_CHECK(power_tangent.sizes() == tx_power.sizes(), "power_tangent must match tx_power");
        TORCH_CHECK(power_tangent.get_device() == tx_positions.get_device(), "power_tangent must be on tx_positions device");
    }
    if (has_rx_tangent) {
        check_cuda_tensor(rx_tangent, "rx_tangent", at::kFloat, 2);
        TORCH_CHECK(rx_tangent.sizes() == rx_positions.sizes(), "rx_tangent must match rx_positions");
        TORCH_CHECK(rx_tangent.get_device() == tx_positions.get_device(), "rx_tangent must be on tx_positions device");
    }

    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.size(0);
    const int64_t pair_count = tx_count * rx_count;
    auto out = at::empty({tx_count, rx_count}, tx_positions.options());
    if (pair_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((pair_count + kLosBlockSize - 1) / kLosBlockSize);
        los_path_gain_jvp_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_tangent.data_ptr<float>(),
            power_tangent.data_ptr<float>(),
            rx_tangent.data_ptr<float>(),
            out.data_ptr<float>(),
            tx_count,
            rx_count,
            has_tx_tangent,
            has_power_tangent,
            has_rx_tangent,
            los_gain_scale(frequency_hz),
            static_cast<float>(
                static_cast<double>(los_gain_scale_dfreq(frequency_hz)) *
                frequency_tangent));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_mc_los_component_maps_cuda(at::Tensor los) {
    return los_component_maps_cuda_impl(los);
}

at::Tensor cn_mc_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols) {
    return los_component_maps_from_matrix_cuda_impl(los, rows, cols);
}

at::Tensor cn_mc_los_component_maps_adjoint_cuda(
    at::Tensor grad_maps,
    at::Tensor visible) {
    check_cuda_tensor_rank(grad_maps, "grad_maps", at::kFloat, 3);
    const int64_t tx_count = grad_maps.size(0);
    const int64_t cols = grad_maps.size(1);
    const int64_t rows = grad_maps.size(2);
    const bool has_visible = visible.numel() > 0;
    if (has_visible) {
        check_cuda_tensor(visible, "visible", at::kBool, 2);
        TORCH_CHECK(
            visible.size(0) == tx_count && visible.size(1) == rows * cols,
            "visible must have shape (tx, rows * cols)");
        TORCH_CHECK(
            visible.get_device() == grad_maps.get_device(),
            "visible must share grad_maps device");
    }
    auto grad_matrix = at::empty({tx_count, rows * cols}, grad_maps.options());
    const int64_t element_count = grad_matrix.numel();
    if (element_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(grad_maps.get_device()).stream();
        const int block_count = static_cast<int>(
            (element_count + kLosBlockSize - 1) / kLosBlockSize);
        los_component_maps_adjoint_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            grad_maps.data_ptr<float>(),
            has_visible ? visible.data_ptr<bool>() : nullptr,
            grad_matrix.data_ptr<float>(),
            tx_count,
            rows,
            cols,
            grad_maps.stride(0),
            grad_maps.stride(1),
            grad_maps.stride(2));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_matrix;
}

at::Tensor cn_mc_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index) {
    check_cuda_tensor(maps, "maps", at::kFloat, 3);
    check_cuda_tensor(los, "los", at::kFloat, 2);
    check_cuda_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(maps.size(0) == los.size(0), "maps and los must have the same tx dimension");
    TORCH_CHECK(los.size(1) == maps.size(1) * maps.size(2), "los columns must match map cells");
    TORCH_CHECK(tx_index >= 0 && tx_index < los.size(0), "tx_index is out of range");
    TORCH_CHECK(los.get_device() == maps.get_device(), "los must share maps device");
    TORCH_CHECK(visible.get_device() == maps.get_device(), "visible must share maps device");
    const int64_t rows = maps.size(2);
    const int64_t cols = maps.size(1);
    const int64_t cell_count = rows * cols;
    TORCH_CHECK(visible.size(0) == cell_count, "visible length must match one receiver grid");

    if (cell_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int block_count = static_cast<int>((cell_count + kLosBlockSize - 1) / kLosBlockSize);
        apply_los_visibility_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            los.data_ptr<float>(),
            visible.data_ptr<bool>(),
            maps.data_ptr<float>(),
            tx_index,
            rows,
            cols);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

std::tuple<at::Tensor, at::Tensor> cn_mc_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    return los_visibility_inputs_cuda_impl(tx_positions, tx_index, rx_count);
}

at::Tensor cn_mc_receiver_grid_points_cuda(
    at::Tensor reference,
    int64_t rows,
    int64_t cols,
    double origin_x,
    double origin_y,
    double origin_z,
    double x_axis_x,
    double x_axis_y,
    double x_axis_z,
    double y_axis_x,
    double y_axis_y,
    double y_axis_z,
    double spacing0,
    double spacing1) {
    return receiver_grid_points_cuda_impl(
        reference,
        rows,
        cols,
        origin_x,
        origin_y,
        origin_z,
        x_axis_x,
        x_axis_y,
        x_axis_z,
        y_axis_x,
        y_axis_y,
        y_axis_z,
        spacing0,
        spacing1);
}

std::tuple<at::Tensor, at::Tensor> cn_mc_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host) {
    return transmitter_tensors_cuda_impl(positions_host, power_host);
}

at::Tensor cn_mc_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z) {
    return pack_vec3_cuda_impl(x, y, z);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz) {
    return los_export_cuda_impl(tx_positions, tx_power, rx_positions, frequency_hz);
}

at::Tensor cn_bdpt_los_component_maps_cuda(at::Tensor los) {
    return los_component_maps_cuda_impl(los);
}

at::Tensor cn_bdpt_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols) {
    return los_component_maps_from_matrix_cuda_impl(los, rows, cols);
}

at::Tensor cn_bdpt_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index) {
    check_cuda_tensor(maps, "maps", at::kFloat, 3);
    check_cuda_tensor(los, "los", at::kFloat, 2);
    check_cuda_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(maps.size(0) == los.size(0), "maps and los must have the same tx dimension");
    TORCH_CHECK(los.size(1) == maps.size(1) * maps.size(2), "los columns must match map cells");
    TORCH_CHECK(tx_index >= 0 && tx_index < los.size(0), "tx_index is out of range");
    TORCH_CHECK(los.get_device() == maps.get_device(), "los must share maps device");
    TORCH_CHECK(visible.get_device() == maps.get_device(), "visible must share maps device");
    const int64_t rows = maps.size(1);
    const int64_t cols = maps.size(2);
    const int64_t cell_count = rows * cols;
    TORCH_CHECK(visible.size(0) == cell_count, "visible length must match one receiver grid");

    if (cell_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int block_count = static_cast<int>((cell_count + kLosBlockSize - 1) / kLosBlockSize);
        apply_los_visibility_public_layout_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            los.data_ptr<float>(),
            visible.data_ptr<bool>(),
            maps.data_ptr<float>(),
            tx_index,
            rows,
            cols);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

std::tuple<at::Tensor, at::Tensor> cn_bdpt_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    return los_visibility_inputs_cuda_impl(tx_positions, tx_index, rx_count);
}

at::Tensor cn_bdpt_receiver_grid_points_cuda(
    at::Tensor reference,
    int64_t rows,
    int64_t cols,
    double origin_x,
    double origin_y,
    double origin_z,
    double x_axis_x,
    double x_axis_y,
    double x_axis_z,
    double y_axis_x,
    double y_axis_y,
    double y_axis_z,
    double spacing0,
    double spacing1) {
    return receiver_grid_points_cuda_impl(
        reference,
        rows,
        cols,
        origin_x,
        origin_y,
        origin_z,
        x_axis_x,
        x_axis_y,
        x_axis_z,
        y_axis_x,
        y_axis_y,
        y_axis_z,
        spacing0,
        spacing1);
}

std::tuple<at::Tensor, at::Tensor> cn_bdpt_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host) {
    return transmitter_tensors_cuda_impl(positions_host, power_host);
}

at::Tensor cn_bdpt_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z) {
    return pack_vec3_cuda_impl(x, y, z);
}
