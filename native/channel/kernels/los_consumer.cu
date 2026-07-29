// ADR-044 consolidated CUDA translation unit.
// Physical co-location only: ABI, launches, synchronization, and numerical order are unchanged.

// ---- Consolidated from los.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>

#include <tuple>
#include <vector>

#define CHANNEL_LOS_CHECK_VISIBILITY_APPLICATION()                                         \
    check_cuda_tensor(maps, "maps", at::kFloat, 3);                                  \
    check_cuda_tensor(los, "los", at::kFloat, 2);                                    \
    check_cuda_tensor(visible, "visible", at::kBool, 1);                             \
    TORCH_CHECK(maps.size(0) == los.size(0), "maps and los must have the same tx dimension");\
    TORCH_CHECK(los.size(1) == maps.size(1) * maps.size(2), "los columns must match map cells");\
    TORCH_CHECK(tx_index >= 0 && tx_index < los.size(0), "tx_index is out of range"); \
    TORCH_CHECK(los.get_device() == maps.get_device(), "los must share maps device"); \
    TORCH_CHECK(visible.get_device() == maps.get_device(), "visible must share maps device")

#define CHANNEL_LOS_VISIBILITY_LAUNCH_ARGUMENTS()                                          \
    los.data_ptr<float>(),                                                            \
    visible.data_ptr<bool>(),                                                         \
    maps.data_ptr<float>(),                                                           \
    tx_index,                                                                         \
    rows,                                                                             \
    cols

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
    float wavelength,
    const float *__restrict__ tx_pol) {
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
        // R5 polarization consistency: short-dipole sin^2(theta) pattern via the
        // squared transverse projection |p_t|^2 = |tx_pol|^2 - (tx_pol . k_hat)^2
        // of the true TX polarization onto the tx->rx direction. Matches the
        // reflection/diffraction seed conventions (and the deterministic field
        // export). tx_pol is unit, so this is exactly sin^2(theta).
        const float *tx_pol_p = tx_pol + static_cast<int64_t>(tx) * 3;
        const float inv_distance = 1.0f / distance;
        const float khx = dx * inv_distance;
        const float khy = dy * inv_distance;
        const float khz = dz * inv_distance;
        const float pol_dot = tx_pol_p[0] * khx + tx_pol_p[1] * khy + tx_pol_p[2] * khz;
        const float pol_mag2 =
            tx_pol_p[0] * tx_pol_p[0] + tx_pol_p[1] * tx_pol_p[1] + tx_pol_p[2] * tx_pol_p[2];
        const float pattern = fmaxf(pol_mag2 - pol_dot * pol_dot, 0.0f);

        tx_id[path] = tx;
        rx_id[path] = rx;
        path_length[path] = distance;
        delay[path] = distance / static_cast<float>(kLightSpeedMetersPerSecond);
        const float gain = tx_power[tx] * gain_base * gain_base * pattern;
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
    float gain_scale_dfreq,
    const float *__restrict__ tx_pol) {
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
        const float *tx_pol_p = tx_pol + tx * 3;
        const float dx = tx_p[0] - rx_p[0];
        const float dy = tx_p[1] - rx_p[1];
        const float dz = tx_p[2] - rx_p[2];
        const float distance_sq = dx * dx + dy * dy + dz * dz;
        const float distance = sqrtf(distance_sq);
        const float safe_distance = fmaxf(distance, 1.0e-6f);
        const float inv_distance = 1.0f / safe_distance;
        const float inv_d2 = inv_distance * inv_distance;
        // R5 dipole pattern: gain = P * gain_scale(f) / d^2 * sin2, with
        // sin2 = |p|^2 - (p . k_hat)^2 and k_hat = (tx - rx)/d.
        const float pol_dot =
            (tx_pol_p[0] * dx + tx_pol_p[1] * dy + tx_pol_p[2] * dz) * inv_distance;
        const float pol_mag2 = tx_pol_p[0] * tx_pol_p[0] +
                               tx_pol_p[1] * tx_pol_p[1] + tx_pol_p[2] * tx_pol_p[2];
        const float pattern = fmaxf(pol_mag2 - pol_dot * pol_dot, 0.0f);
        atomicAdd(grad_power + tx, go * gain_scale * inv_d2 * pattern);
        if (grad_frequency != nullptr) {
            // gain = P * gain_scale(f) / d^2 * sin2; only gain_scale carries f.
            atomicAdd(
                grad_frequency,
                go * tx_power[tx] * gain_scale_dfreq * inv_d2 * pattern);
        }
        // Position gradients follow the clamp_min pass-through convention
        // (>= keeps the boundary subgradient, matching the tests/ad oracle).
        if (distance < 1.0e-6f) {
            continue;
        }
        // d(gain)/drx = 2 P S ( sin2 * r / d^4 + c * p_t / d^3 ), with
        // p_t = p - c*k_hat the transverse projection; grad_tx = -grad_rx.
        const float ptx = tx_pol_p[0] - pol_dot * dx * inv_distance;
        const float pty = tx_pol_p[1] - pol_dot * dy * inv_distance;
        const float ptz = tx_pol_p[2] - pol_dot * dz * inv_distance;
        const float base = go * 2.0f * tx_power[tx] * gain_scale * inv_d2;
        const float gx = base * (pattern * dx * inv_d2 + pol_dot * ptx * inv_distance);
        const float gy = base * (pattern * dy * inv_d2 + pol_dot * pty * inv_distance);
        const float gz = base * (pattern * dz * inv_d2 + pol_dot * ptz * inv_distance);
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
    float gain_scale_dfreq_tangent,
    const float *__restrict__ tx_pol) {
    const int64_t pair_count = tx_count * rx_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        const int64_t tx = pair / rx_count;
        const int64_t rx = pair - tx * rx_count;
        const float *tx_p = tx_positions + tx * 3;
        const float *rx_p = rx_positions + rx * 3;
        const float *tx_pol_p = tx_pol + tx * 3;
        const float dx = tx_p[0] - rx_p[0];
        const float dy = tx_p[1] - rx_p[1];
        const float dz = tx_p[2] - rx_p[2];
        const float distance_sq = dx * dx + dy * dy + dz * dz;
        const float distance = sqrtf(distance_sq);
        const float safe_distance = fmaxf(distance, 1.0e-6f);
        const float inv_distance = 1.0f / safe_distance;
        const float inv_d2 = inv_distance * inv_distance;
        // R5 dipole pattern sin2 = |p|^2 - (p . k_hat)^2, k_hat = (tx - rx)/d.
        const float pol_dot =
            (tx_pol_p[0] * dx + tx_pol_p[1] * dy + tx_pol_p[2] * dz) * inv_distance;
        const float pol_mag2 = tx_pol_p[0] * tx_pol_p[0] +
                               tx_pol_p[1] * tx_pol_p[1] + tx_pol_p[2] * tx_pol_p[2];
        const float pattern = fmaxf(pol_mag2 - pol_dot * pol_dot, 0.0f);
        float tangent = 0.0f;
        if (has_power_tangent) {
            tangent += power_tangent[tx] * gain_scale * inv_d2 * pattern;
        }
        // Frequency tangent, pre-folded on the host into
        // d(gain_scale)/df * t_f so a zero tangent costs one fma.
        tangent += tx_power[tx] * gain_scale_dfreq_tangent * inv_d2 * pattern;
        if (distance >= 1.0e-6f) {
            // d(gain)/dr . t = 2 P S ( sin2 * (r . t) / d^4 + c * (p_t . t) / d^3 ).
            const float ptx = tx_pol_p[0] - pol_dot * dx * inv_distance;
            const float pty = tx_pol_p[1] - pol_dot * dy * inv_distance;
            const float ptz = tx_pol_p[2] - pol_dot * dz * inv_distance;
            const float base = 2.0f * tx_power[tx] * gain_scale * inv_d2;
            if (has_rx_tangent) {
                const float *rx_t = rx_tangent + rx * 3;
                const float r_dot = dx * rx_t[0] + dy * rx_t[1] + dz * rx_t[2];
                const float p_dot = ptx * rx_t[0] + pty * rx_t[1] + ptz * rx_t[2];
                tangent += base * (pattern * r_dot * inv_d2 + pol_dot * p_dot * inv_distance);
            }
            if (has_tx_tangent) {
                const float *tx_t = tx_tangent + tx * 3;
                const float r_dot = dx * tx_t[0] + dy * tx_t[1] + dz * tx_t[2];
                const float p_dot = ptx * tx_t[0] + pty * tx_t[1] + ptz * tx_t[2];
                tangent -= base * (pattern * r_dot * inv_d2 + pol_dot * p_dot * inv_distance);
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
    double frequency_hz,
    at::Tensor tx_pol) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_pol, "tx_pol", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(
        tx_pol.sizes() == tx_positions.sizes(),
        "tx_pol must match tx_positions shape (N, 3)");
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
            wavelength,
            tx_pol.data_ptr<float>());
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_path_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz,
    at::Tensor tx_pol) {
    return los_export_cuda_impl(tx_positions, tx_power, rx_positions, frequency_hz, tx_pol);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_los_path_gain_backward_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    at::Tensor grad_output,
    double frequency_hz,
    at::Tensor tx_pol) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_pol, "tx_pol", at::kFloat, 2);
    check_cuda_tensor_rank(grad_output, "grad_output", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(tx_pol.sizes() == tx_positions.sizes(), "tx_pol must match tx_positions shape");
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
            los_gain_scale_dfreq(frequency_hz),
            tx_pol.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_tx, grad_power, grad_rx, grad_frequency};
}

at::Tensor channel_mc_los_path_gain_jvp_cuda(
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
    double frequency_tangent,
    at::Tensor tx_pol) {
    check_cuda_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(rx_positions, "rx_positions", at::kFloat, 2);
    check_cuda_tensor(tx_pol, "tx_pol", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(rx_positions.size(1) == 3, "rx_positions must have shape (M, 3)");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(tx_pol.sizes() == tx_positions.sizes(), "tx_pol must match tx_positions shape");
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
                frequency_tangent),
            tx_pol.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor channel_mc_los_component_maps_cuda(at::Tensor los) {
    return los_component_maps_cuda_impl(los);
}

at::Tensor channel_mc_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols) {
    return los_component_maps_from_matrix_cuda_impl(los, rows, cols);
}

at::Tensor channel_mc_los_component_maps_adjoint_cuda(
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

at::Tensor channel_mc_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index) {
    CHANNEL_LOS_CHECK_VISIBILITY_APPLICATION();
    const int64_t rows = maps.size(2);
    const int64_t cols = maps.size(1);
    const int64_t cell_count = rows * cols;
    TORCH_CHECK(visible.size(0) == cell_count, "visible length must match one receiver grid");

    if (cell_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int block_count = static_cast<int>((cell_count + kLosBlockSize - 1) / kLosBlockSize);
        apply_los_visibility_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            CHANNEL_LOS_VISIBILITY_LAUNCH_ARGUMENTS());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

std::tuple<at::Tensor, at::Tensor> channel_mc_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    return los_visibility_inputs_cuda_impl(tx_positions, tx_index, rx_count);
}

at::Tensor channel_mc_receiver_grid_points_cuda(
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

std::tuple<at::Tensor, at::Tensor> channel_mc_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host) {
    return transmitter_tensors_cuda_impl(positions_host, power_host);
}

at::Tensor channel_mc_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z) {
    return pack_vec3_cuda_impl(x, y, z);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_los_export_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz,
    at::Tensor tx_pol) {
    return los_export_cuda_impl(tx_positions, tx_power, rx_positions, frequency_hz, tx_pol);
}

at::Tensor channel_bdpt_los_component_maps_cuda(at::Tensor los) {
    return los_component_maps_cuda_impl(los);
}

at::Tensor channel_bdpt_los_component_maps_from_matrix_cuda(at::Tensor los, int64_t rows, int64_t cols) {
    return los_component_maps_from_matrix_cuda_impl(los, rows, cols);
}

at::Tensor channel_bdpt_apply_los_visibility_cuda(
    at::Tensor maps,
    at::Tensor los,
    at::Tensor visible,
    int64_t tx_index) {
    CHANNEL_LOS_CHECK_VISIBILITY_APPLICATION();
    const int64_t rows = maps.size(1);
    const int64_t cols = maps.size(2);
    const int64_t cell_count = rows * cols;
    TORCH_CHECK(visible.size(0) == cell_count, "visible length must match one receiver grid");

    if (cell_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int block_count = static_cast<int>((cell_count + kLosBlockSize - 1) / kLosBlockSize);
        apply_los_visibility_public_layout_kernel<<<block_count, kLosBlockSize, 0, stream>>>(
            CHANNEL_LOS_VISIBILITY_LAUNCH_ARGUMENTS());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

#undef CHANNEL_LOS_VISIBILITY_LAUNCH_ARGUMENTS
#undef CHANNEL_LOS_CHECK_VISIBILITY_APPLICATION

std::tuple<at::Tensor, at::Tensor> channel_bdpt_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t rx_count) {
    return los_visibility_inputs_cuda_impl(tx_positions, tx_index, rx_count);
}

at::Tensor channel_bdpt_receiver_grid_points_cuda(
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

std::tuple<at::Tensor, at::Tensor> channel_bdpt_transmitter_tensors_cuda(
    const std::vector<float> &positions_host,
    const std::vector<float> &power_host) {
    return transmitter_tensors_cuda_impl(positions_host, power_host);
}

at::Tensor channel_bdpt_pack_vec3_cuda(at::Tensor x, at::Tensor y, at::Tensor z) {
    return pack_vec3_cuda_impl(x, y, z);
}

// ---- Consolidated from los_silhouette_clearance.cu ----
// ISB boundary taper (ADR-017), LoS member. Native, channel owned.
//
// Two per-(tx,rx) CUDA operations that implement the DEFAULT-OFF joint ISB
// taper's line-of-sight half:
//
//   channel_los_silhouette_clearance:  for each (source, target) segment, find the
//     nearest occluding axis-aligned box the segment grazes, and return the
//     C1 membership factor tau(c_plane / (width * w_F)) where
//       - c   = signed clearance of the segment past the box silhouette
//               (positive when the segment clears the box / lit, negative when
//               it penetrates / shadow), taken as the signed AABB distance at
//               the segment's closest-approach sample (measured at the
//               occluder);
//       - c_plane = c * (d1 + d2) / d1 magnifies that occluder-plane miss
//               distance into the receiver plane, where the point-source shadow
//               of the silhouette edge is enlarged by the same factor. The
//               accepted projection (artifacts/isb-taper/stage2.py) scores the
//               clearance as an in-receiver-plane distance transform, so the
//               native band must cover the same receiver-plane extent;
//       - w_F = sqrt(lambda * d1 * d2 / (d1 + d2)) is the Fresnel penumbra of
//               the grazed edge (d1 = |grazing - source|, d2 = |target -
//               grazing|); the exact form and the signed-distance / grazing
//               conventions match artifacts/isb-taper/common.py + stage1_geom.py;
//       - tau(w) = smoothstep01(0.5 * (w + 1)) is the C1 step through 1/2 at
//               c = 0 (artifacts/isb-taper/stage2.py tau_smoothstep).
//     A segment that grazes no box (empty scene) returns tau = 1 (fully lit).
//
//   channel_los_taper_apply: scale a LoS field bundle (complex3 vector, complex
//     coefficient, complex path_field, real path_gain) by the real per-row
//     factor tau. tau multiplies the field amplitude, so path_gain (a power)
//     is scaled by tau*tau. No torch hot-path math; the scale runs in-kernel.
//
// Both ops are only ever launched when isb_boundary_taper is on; the off path
// never reaches this translation unit, so the default solve is bit-identical.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kBlockSize = 256;
// Segment samples used to locate the closest-approach point to each box.
// Matches artifacts/isb-taper/stage1_geom.py occluding_edge_geom (400 samples).
constexpr int kSegmentSamples = 400;

__device__ __forceinline__ float smoothstep01(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Signed distance of point p to an axis-aligned box [bmin, bmax]:
//   q = max(bmin - p, p - bmax);  sd = ||max(q, 0)|| + min(max(q), 0)
// negative inside the box, positive outside. Matches common.py conventions.
__device__ __forceinline__ float aabb_signed_distance(
    float px, float py, float pz,
    float minx, float miny, float minz,
    float maxx, float maxy, float maxz) {
    const float qx = fmaxf(minx - px, px - maxx);
    const float qy = fmaxf(miny - py, py - maxy);
    const float qz = fmaxf(minz - pz, pz - maxz);
    const float ox = fmaxf(qx, 0.0f);
    const float oy = fmaxf(qy, 0.0f);
    const float oz = fmaxf(qz, 0.0f);
    const float outside = sqrtf(ox * ox + oy * oy + oz * oz);
    const float inside = fminf(fmaxf(fmaxf(qx, qy), qz), 0.0f);
    return outside + inside;
}

__global__ void los_silhouette_clearance_kernel(
    const float *__restrict__ source,
    const float *__restrict__ target,
    const float *__restrict__ box_min,
    const float *__restrict__ box_max,
    float *__restrict__ tau,
    int64_t pair_count,
    int64_t box_count,
    float wavelength,
    float width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < pair_count;
         idx += stride) {
        const float sx = source[idx * 3 + 0];
        const float sy = source[idx * 3 + 1];
        const float sz = source[idx * 3 + 2];
        const float tx = target[idx * 3 + 0];
        const float ty = target[idx * 3 + 1];
        const float tz = target[idx * 3 + 2];
        const float dx = tx - sx;
        const float dy = ty - sy;
        const float dz = tz - sz;

        // Nearest grazed box: minimize |signed AABB distance| along the segment.
        float best_slack = 3.4e38f;
        float best_c = 1.0e30f;     // signed clearance at the closest approach
        float best_d1 = 0.0f;       // |grazing - source|
        float best_d2 = 0.0f;       // |target  - grazing|
        bool found = false;
        for (int64_t b = 0; b < box_count; ++b) {
            const float minx = box_min[b * 3 + 0];
            const float miny = box_min[b * 3 + 1];
            const float minz = box_min[b * 3 + 2];
            const float maxx = box_max[b * 3 + 0];
            const float maxy = box_max[b * 3 + 1];
            const float maxz = box_max[b * 3 + 2];
            float box_slack = 3.4e38f;
            float box_c = 0.0f;
            float box_t = 0.0f;
            for (int s = 0; s < kSegmentSamples; ++s) {
                const float u = static_cast<float>(s) /
                                static_cast<float>(kSegmentSamples - 1);
                const float px = sx + u * dx;
                const float py = sy + u * dy;
                const float pz = sz + u * dz;
                const float sd = aabb_signed_distance(
                    px, py, pz, minx, miny, minz, maxx, maxy, maxz);
                const float slack = fabsf(sd);
                if (slack < box_slack) {
                    box_slack = slack;
                    box_c = sd;
                    box_t = u;
                }
            }
            if (box_slack < best_slack) {
                best_slack = box_slack;
                best_c = box_c;
                const float gx = sx + box_t * dx;
                const float gy = sy + box_t * dy;
                const float gz = sz + box_t * dz;
                best_d1 = sqrtf((gx - sx) * (gx - sx) + (gy - sy) * (gy - sy) +
                                (gz - sz) * (gz - sz));
                best_d2 = sqrtf((tx - gx) * (tx - gx) + (ty - gy) * (ty - gy) +
                                (tz - gz) * (tz - gz));
                found = true;
            }
        }

        if (!found) {
            // No occluder: the segment is fully lit.
            tau[idx] = 1.0f;
            continue;
        }
        // Shadow magnification: best_c is the 3D miss distance measured at the
        // occluder (closest-approach sample), but the accepted projection scores
        // the clearance in the RECEIVER PLANE, where the point-source shadow of
        // the silhouette edge is magnified by (d1 + d2) / d1. Convert so the
        // native taper band covers the same receiver-plane extent as the
        // projection (artifacts/isb-taper/stage2.py in-plane distance transform).
        const float mag = (best_d1 + best_d2) / fmaxf(best_d1, 1.0e-6f);
        const float c_plane = best_c * mag;
        const float w_F = sqrtf(fmaxf(
            wavelength * best_d1 * best_d2 / fmaxf(best_d1 + best_d2, 1.0e-12f),
            0.0f));
        const float w = fmaxf(width * w_F, 1.0e-6f);
        tau[idx] = smoothstep01(0.5f * (c_plane / w + 1.0f));
    }
}

__global__ void los_taper_apply_kernel(
    const c10::complex<float> *__restrict__ field_vector,
    const c10::complex<float> *__restrict__ coefficient,
    const c10::complex<float> *__restrict__ path_field,
    const float *__restrict__ path_gain,
    const float *__restrict__ tau,
    c10::complex<float> *__restrict__ out_field_vector,
    c10::complex<float> *__restrict__ out_coefficient,
    c10::complex<float> *__restrict__ out_path_field,
    float *__restrict__ out_path_gain,
    int64_t row_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < row_count;
         idx += stride) {
        const float s = tau[idx];
        out_field_vector[idx * 3 + 0] = field_vector[idx * 3 + 0] * s;
        out_field_vector[idx * 3 + 1] = field_vector[idx * 3 + 1] * s;
        out_field_vector[idx * 3 + 2] = field_vector[idx * 3 + 2] * s;
        out_coefficient[idx] = coefficient[idx] * s;
        out_path_field[idx] = path_field[idx] * s;
        // tau scales the field amplitude, so a power scales by tau^2.
        out_path_gain[idx] = path_gain[idx] * s * s;
    }
}

void check_cuda(const at::Tensor &t, const char *name, c10::ScalarType dtype,
                int64_t ndim) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(t.dim() == ndim, name, " has the wrong rank");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

at::Tensor channel_los_silhouette_clearance(
    at::Tensor source,
    at::Tensor target,
    at::Tensor box_min,
    at::Tensor box_max,
    double wavelength,
    double width) {
    check_cuda(source, "source", at::kFloat, 2);
    check_cuda(target, "target", at::kFloat, 2);
    check_cuda(box_min, "box_min", at::kFloat, 2);
    check_cuda(box_max, "box_max", at::kFloat, 2);
    TORCH_CHECK(source.size(1) == 3, "source must have shape (N, 3)");
    TORCH_CHECK(target.sizes() == source.sizes(), "target must match source");
    TORCH_CHECK(box_min.size(1) == 3, "box_min must have shape (B, 3)");
    TORCH_CHECK(box_max.sizes() == box_min.sizes(), "box_max must match box_min");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(width > 0.0, "width must be positive");
    TORCH_CHECK(
        source.get_device() == box_min.get_device() &&
            target.get_device() == source.get_device() &&
            box_max.get_device() == source.get_device(),
        "silhouette clearance tensors must share one CUDA device");

    const int64_t pair_count = source.size(0);
    const int64_t box_count = box_min.size(0);
    auto tau = at::empty({pair_count}, source.options());
    if (pair_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        const int blocks =
            static_cast<int>((pair_count + kBlockSize - 1) / kBlockSize);
        los_silhouette_clearance_kernel<<<blocks, kBlockSize, 0, stream>>>(
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            box_min.data_ptr<float>(),
            box_max.data_ptr<float>(),
            tau.data_ptr<float>(),
            pair_count,
            box_count,
            static_cast<float>(wavelength),
            static_cast<float>(width));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return tau;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_los_taper_apply(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor tau) {
    check_cuda(field_vector, "field_vector", at::kComplexFloat, 2);
    check_cuda(coefficient, "coefficient", at::kComplexFloat, 1);
    check_cuda(path_field, "path_field", at::kComplexFloat, 1);
    check_cuda(path_gain, "path_gain", at::kFloat, 1);
    check_cuda(tau, "tau", at::kFloat, 1);
    const int64_t row_count = tau.size(0);
    TORCH_CHECK(field_vector.size(0) == row_count && field_vector.size(1) == 3,
                "field_vector must have shape (N, 3)");
    TORCH_CHECK(coefficient.size(0) == row_count, "coefficient must match tau");
    TORCH_CHECK(path_field.size(0) == row_count, "path_field must match tau");
    TORCH_CHECK(path_gain.size(0) == row_count, "path_gain must match tau");

    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    if (row_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tau.get_device()).stream();
        const int blocks =
            static_cast<int>((row_count + kBlockSize - 1) / kBlockSize);
        los_taper_apply_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const c10::complex<float> *>(
                field_vector.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const c10::complex<float> *>(
                coefficient.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const c10::complex<float> *>(
                path_field.data_ptr<c10::complex<float>>()),
            path_gain.data_ptr<float>(),
            tau.data_ptr<float>(),
            reinterpret_cast<c10::complex<float> *>(
                out_field_vector.data_ptr<c10::complex<float>>()),
            reinterpret_cast<c10::complex<float> *>(
                out_coefficient.data_ptr<c10::complex<float>>()),
            reinterpret_cast<c10::complex<float> *>(
                out_path_field.data_ptr<c10::complex<float>>()),
            out_path_gain.data_ptr<float>(),
            row_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_field_vector, out_coefficient, out_path_field, out_path_gain};
}

// ---- Consolidated from consumer_fixed_los_gather.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../tensor_checks.h"

#include <cstdint>
#include <limits>
#include <optional>

#define kBlockSize kFixedLosGatherBlockSize

namespace {

constexpr int kBlockSize = 256;
constexpr int kLoSComponentId = 0;

enum ContractError : int {
    kIndexBounds = 1 << 0,
    kPairOrder = 1 << 1,
    kDepth = 1 << 2,
    kComponent = 1 << 3,
    kStableId = 1 << 4,
};

struct OptionalVector {
    const float* data;
    int64_t stride0;
    int64_t stride1;
    bool present;
};

struct OptionalScalar {
    const float* data;
    int64_t stride0;
    bool present;
};

__global__ void validate_fixed_los_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    const int64_t* __restrict__ source_id,
    const int64_t* __restrict__ sink_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int64_t* __restrict__ source_stable_ids,
    const int64_t* __restrict__ sink_stable_ids,
    int64_t row_count,
    int64_t source_count,
    int64_t sink_count,
    int64_t* __restrict__ pair_index,
    int* __restrict__ contract_error) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source = source_index[row];
        const int sink = sink_index[row];
        const bool in_bounds =
            source >= 0 && static_cast<int64_t>(source) < source_count &&
            sink >= 0 && static_cast<int64_t>(sink) < sink_count;
        int error = 0;
        int64_t pair = -1;
        if (!in_bounds) {
            error |= kIndexBounds;
        } else {
            pair = static_cast<int64_t>(sink) * source_count + source;
            if (source_id[row] != source_stable_ids[source] ||
                sink_id[row] != sink_stable_ids[sink]) {
                error |= kStableId;
            }
            if (row > 0) {
                const int previous_source = source_index[row - 1];
                const int previous_sink = sink_index[row - 1];
                const bool previous_in_bounds =
                    previous_source >= 0 &&
                    static_cast<int64_t>(previous_source) < source_count &&
                    previous_sink >= 0 &&
                    static_cast<int64_t>(previous_sink) < sink_count;
                if (previous_in_bounds) {
                    const int64_t previous_pair =
                        static_cast<int64_t>(previous_sink) * source_count +
                        previous_source;
                    if (pair < previous_pair) {
                        error |= kPairOrder;
                    }
                }
            }
        }
        pair_index[row] = pair;
        if (depth[row] != 0) {
            error |= kDepth;
        }
        if (component_id[row] != kLoSComponentId) {
            error |= kComponent;
        }
        if (error != 0) {
            atomicOr(contract_error, error);
        }
    }
}

__global__ void gather_fixed_los_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    const int64_t* __restrict__ pair_index,
    const float* __restrict__ source_positions,
    const float* __restrict__ sink_positions,
    const float* __restrict__ source_powers,
    const float* __restrict__ source_polarizations,
    const float* __restrict__ sink_polarizations,
    int64_t row_count,
    float* __restrict__ source,
    float* __restrict__ target,
    float* __restrict__ tx_power,
    float* __restrict__ tx_polarization,
    float* __restrict__ rx_polarization,
    int64_t* __restrict__ pair_offsets) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t output_base = row * 3;
        const int64_t source_base = static_cast<int64_t>(source_row) * 3;
        const int64_t sink_base = static_cast<int64_t>(sink_row) * 3;
        for (int component = 0; component < 3; ++component) {
            source[output_base + component] =
                source_positions[source_base + component];
            target[output_base + component] =
                sink_positions[sink_base + component];
            tx_polarization[output_base + component] =
                source_polarizations[source_base + component];
            rx_polarization[output_base + component] =
                sink_polarizations[sink_base + component];
        }
        tx_power[row] = source_powers[source_row];
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                pair_offsets + pair_index[row] + 1),
            1ULL);
    }
}

__device__ __forceinline__ float read_vector(
    const OptionalVector& view,
    int64_t row,
    int component) {
    return view.present
        ? view.data[row * view.stride0 + component * view.stride1]
        : 0.0f;
}

__device__ __forceinline__ float read_scalar(
    const OptionalScalar& view,
    int64_t row) {
    return view.present ? view.data[row * view.stride0] : 0.0f;
}

__global__ void fixed_los_backward_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    OptionalVector grad_source,
    OptionalVector grad_target,
    OptionalScalar grad_tx_power,
    OptionalVector grad_tx_polarization,
    OptionalVector grad_rx_polarization,
    int64_t row_count,
    float* __restrict__ grad_source_positions,
    float* __restrict__ grad_sink_positions,
    float* __restrict__ grad_source_powers,
    float* __restrict__ grad_source_polarizations,
    float* __restrict__ grad_sink_polarizations) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t source_base = static_cast<int64_t>(source_row) * 3;
        const int64_t sink_base = static_cast<int64_t>(sink_row) * 3;
        for (int component = 0; component < 3; ++component) {
            if (grad_source.present) {
                atomicAdd(
                    grad_source_positions + source_base + component,
                    read_vector(grad_source, row, component));
            }
            if (grad_target.present) {
                atomicAdd(
                    grad_sink_positions + sink_base + component,
                    read_vector(grad_target, row, component));
            }
            if (grad_tx_polarization.present) {
                atomicAdd(
                    grad_source_polarizations + source_base + component,
                    read_vector(grad_tx_polarization, row, component));
            }
            if (grad_rx_polarization.present) {
                atomicAdd(
                    grad_sink_polarizations + sink_base + component,
                    read_vector(grad_rx_polarization, row, component));
            }
        }
        if (grad_tx_power.present) {
            atomicAdd(
                grad_source_powers + source_row,
                read_scalar(grad_tx_power, row));
        }
    }
}

__global__ void fixed_los_jvp_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    OptionalVector tangent_source_positions,
    OptionalVector tangent_sink_positions,
    OptionalScalar tangent_source_powers,
    OptionalVector tangent_source_polarizations,
    OptionalVector tangent_sink_polarizations,
    int64_t row_count,
    float* __restrict__ tangent_source,
    float* __restrict__ tangent_target,
    float* __restrict__ tangent_tx_power,
    float* __restrict__ tangent_tx_polarization,
    float* __restrict__ tangent_rx_polarization) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t output_base = row * 3;
        for (int component = 0; component < 3; ++component) {
            tangent_source[output_base + component] = read_vector(
                tangent_source_positions, source_row, component);
            tangent_target[output_base + component] = read_vector(
                tangent_sink_positions, sink_row, component);
            tangent_tx_polarization[output_base + component] = read_vector(
                tangent_source_polarizations, source_row, component);
            tangent_rx_polarization[output_base + component] = read_vector(
                tangent_sink_polarizations, sink_row, component);
        }
        tangent_tx_power[row] =
            read_scalar(tangent_source_powers, source_row);
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

void check_row_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rows,
    int device) {
    channel::check_tensor(tensor, name, dtype, 1);
    TORCH_CHECK(tensor.size(0) == rows, name, " must share topology rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

void check_endpoint_vec3(
    const at::Tensor& tensor,
    const char* name,
    int64_t rows,
    int device) {
    channel::check_vec3_table(tensor, name);
    TORCH_CHECK(tensor.size(0) == rows, name, " has the wrong endpoint count");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

void check_endpoint_vector(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rows,
    int device) {
    channel::check_tensor(tensor, name, dtype, 1);
    TORCH_CHECK(tensor.size(0) == rows, name, " has the wrong endpoint count");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

OptionalVector optional_vector(
    const std::optional<at::Tensor>& tensor,
    const char* name,
    int64_t rows,
    int device) {
    if (!tensor.has_value()) {
        return {nullptr, 0, 0, false};
    }
    TORCH_CHECK(tensor->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor->scalar_type() == at::kFloat, name, " must use float32");
    TORCH_CHECK(
        tensor->dim() == 2 && tensor->size(0) == rows && tensor->size(1) == 3,
        name,
        " must have shape (K, 3)");
    TORCH_CHECK(tensor->get_device() == device, name, " must share topology device");
    return {
        tensor->data_ptr<float>(),
        tensor->stride(0),
        tensor->stride(1),
        true};
}

OptionalScalar optional_scalar(
    const std::optional<at::Tensor>& tensor,
    const char* name,
    int64_t rows,
    int device) {
    if (!tensor.has_value()) {
        return {nullptr, 0, false};
    }
    TORCH_CHECK(tensor->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor->scalar_type() == at::kFloat, name, " must use float32");
    TORCH_CHECK(
        tensor->dim() == 1 && tensor->size(0) == rows,
        name,
        " must have shape (K,)");
    TORCH_CHECK(tensor->get_device() == device, name, " must share topology device");
    return {tensor->data_ptr<float>(), tensor->stride(0), true};
}

void check_selection(
    const at::Tensor& source_index,
    const at::Tensor& sink_index) {
    channel::check_tensor(source_index, "source_index", at::kInt, 1);
    check_row_tensor(
        sink_index,
        "sink_index",
        at::kInt,
        source_index.size(0),
        source_index.get_device());
}

}  // namespace

pybind11::dict channel_consumer_fixed_los_gather(
    at::Tensor source_index,
    at::Tensor sink_index,
    at::Tensor source_id,
    at::Tensor sink_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor source_positions,
    at::Tensor sink_positions,
    at::Tensor source_powers,
    at::Tensor source_polarizations,
    at::Tensor sink_polarizations,
    at::Tensor source_stable_ids,
    at::Tensor sink_stable_ids) {
    check_selection(source_index, sink_index);
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    check_row_tensor(source_id, "source_id", at::kLong, row_count, device);
    check_row_tensor(sink_id, "sink_id", at::kLong, row_count, device);
    check_row_tensor(depth, "depth", at::kInt, row_count, device);
    check_row_tensor(component_id, "component_id", at::kInt, row_count, device);
    channel::check_vec3_table(source_positions, "source_positions");
    channel::check_vec3_table(sink_positions, "sink_positions");
    const int64_t source_count = source_positions.size(0);
    const int64_t sink_count = sink_positions.size(0);
    check_endpoint_vec3(
        source_polarizations,
        "source_polarizations",
        source_count,
        device);
    check_endpoint_vec3(
        sink_polarizations,
        "sink_polarizations",
        sink_count,
        device);
    check_endpoint_vector(
        source_powers,
        "source_powers",
        at::kFloat,
        source_count,
        device);
    check_endpoint_vector(
        source_stable_ids,
        "source_stable_ids",
        at::kLong,
        source_count,
        device);
    check_endpoint_vector(
        sink_stable_ids,
        "sink_stable_ids",
        at::kLong,
        sink_count,
        device);
    TORCH_CHECK(
        source_positions.get_device() == device &&
            sink_positions.get_device() == device,
        "endpoint positions must share topology device");
    TORCH_CHECK(
        source_count == 0 ||
            sink_count <= std::numeric_limits<int64_t>::max() / source_count,
        "source/sink pair count overflows int64");
    const int64_t pair_count = source_count * sink_count;

    auto float_options = source_positions.options();
    auto long_options = source_index.options().dtype(at::kLong);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    auto source = at::empty({row_count, 3}, float_options);
    auto target = at::empty({row_count, 3}, float_options);
    auto tx_power = at::empty({row_count}, float_options);
    auto tx_polarization = at::empty({row_count, 3}, float_options);
    auto rx_polarization = at::empty({row_count, 3}, float_options);
    auto pair_index = at::empty({row_count}, long_options);
    auto pair_offsets = channel::empty_zero_cuda(
        {pair_count + 1}, long_options, stream);

    if (row_count > 0) {
        auto contract_error = channel::empty_zero_cuda(
            {1}, source_index.options().dtype(at::kInt), stream);
        validate_fixed_los_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_id.data_ptr<int64_t>(),
            sink_id.data_ptr<int64_t>(),
            depth.data_ptr<int>(),
            component_id.data_ptr<int>(),
            source_stable_ids.data_ptr<int64_t>(),
            sink_stable_ids.data_ptr<int64_t>(),
            row_count,
            source_count,
            sink_count,
            pair_index.data_ptr<int64_t>(),
            contract_error.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int host_error = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &host_error,
            contract_error.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK(
            host_error == 0,
            "fixed LoS topology validation failed (error bitmask ",
            host_error,
            ")");

        gather_fixed_los_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            pair_index.data_ptr<int64_t>(),
            source_positions.data_ptr<float>(),
            sink_positions.data_ptr<float>(),
            source_powers.data_ptr<float>(),
            source_polarizations.data_ptr<float>(),
            sink_polarizations.data_ptr<float>(),
            row_count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            pair_offsets.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        thrust::inclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()),
            thrust::device_pointer_cast(
                pair_offsets.data_ptr<int64_t>() + pair_count + 1),
            thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()));
    }

    pybind11::dict result;
    result["source"] = source;
    result["target"] = target;
    result["tx_power"] = tx_power;
    result["tx_polarization"] = tx_polarization;
    result["rx_polarization"] = rx_polarization;
    result["pair_index"] = pair_index;
    result["pair_offsets"] = pair_offsets;
    return result;
}

pybind11::dict channel_consumer_fixed_los_gather_backward(
    at::Tensor source_index,
    at::Tensor sink_index,
    std::optional<at::Tensor> grad_source,
    std::optional<at::Tensor> grad_target,
    std::optional<at::Tensor> grad_tx_power,
    std::optional<at::Tensor> grad_tx_polarization,
    std::optional<at::Tensor> grad_rx_polarization,
    int64_t source_count,
    int64_t sink_count) {
    check_selection(source_index, sink_index);
    TORCH_CHECK(source_count >= 0, "source_count must be non-negative");
    TORCH_CHECK(sink_count >= 0, "sink_count must be non-negative");
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    const OptionalVector source_view =
        optional_vector(grad_source, "grad_source", row_count, device);
    const OptionalVector target_view =
        optional_vector(grad_target, "grad_target", row_count, device);
    const OptionalScalar power_view =
        optional_scalar(grad_tx_power, "grad_tx_power", row_count, device);
    const OptionalVector tx_pol_view = optional_vector(
        grad_tx_polarization,
        "grad_tx_polarization",
        row_count,
        device);
    const OptionalVector rx_pol_view = optional_vector(
        grad_rx_polarization,
        "grad_rx_polarization",
        row_count,
        device);
    auto float_options = source_index.options().dtype(at::kFloat);
    auto grad_source_positions = channel::empty_zero_cuda(
        {source_count, 3}, float_options, stream);
    auto grad_sink_positions = channel::empty_zero_cuda(
        {sink_count, 3}, float_options, stream);
    auto grad_source_powers = channel::empty_zero_cuda(
        {source_count}, float_options, stream);
    auto grad_source_polarizations =
        channel::empty_zero_cuda({source_count, 3}, float_options, stream);
    auto grad_sink_polarizations =
        channel::empty_zero_cuda({sink_count, 3}, float_options, stream);
    if (row_count > 0) {
        fixed_los_backward_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_view,
            target_view,
            power_view,
            tx_pol_view,
            rx_pol_view,
            row_count,
            grad_source_positions.data_ptr<float>(),
            grad_sink_positions.data_ptr<float>(),
            grad_source_powers.data_ptr<float>(),
            grad_source_polarizations.data_ptr<float>(),
            grad_sink_polarizations.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["source_positions"] = grad_source_positions;
    result["sink_positions"] = grad_sink_positions;
    result["source_powers"] = grad_source_powers;
    result["source_polarizations"] = grad_source_polarizations;
    result["sink_polarizations"] = grad_sink_polarizations;
    return result;
}

pybind11::dict channel_consumer_fixed_los_gather_jvp(
    at::Tensor source_index,
    at::Tensor sink_index,
    std::optional<at::Tensor> tangent_source_positions,
    std::optional<at::Tensor> tangent_sink_positions,
    std::optional<at::Tensor> tangent_source_powers,
    std::optional<at::Tensor> tangent_source_polarizations,
    std::optional<at::Tensor> tangent_sink_polarizations,
    int64_t source_count,
    int64_t sink_count) {
    check_selection(source_index, sink_index);
    TORCH_CHECK(source_count >= 0, "source_count must be non-negative");
    TORCH_CHECK(sink_count >= 0, "sink_count must be non-negative");
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    const OptionalVector source_view = optional_vector(
        tangent_source_positions,
        "tangent_source_positions",
        source_count,
        device);
    const OptionalVector target_view = optional_vector(
        tangent_sink_positions,
        "tangent_sink_positions",
        sink_count,
        device);
    const OptionalScalar power_view = optional_scalar(
        tangent_source_powers,
        "tangent_source_powers",
        source_count,
        device);
    const OptionalVector tx_pol_view = optional_vector(
        tangent_source_polarizations,
        "tangent_source_polarizations",
        source_count,
        device);
    const OptionalVector rx_pol_view = optional_vector(
        tangent_sink_polarizations,
        "tangent_sink_polarizations",
        sink_count,
        device);
    auto float_options = source_index.options().dtype(at::kFloat);
    auto tangent_source = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_target = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_tx_power = channel::empty_zero_cuda(
        {row_count}, float_options, stream);
    auto tangent_tx_polarization = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_rx_polarization = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    if (row_count > 0) {
        fixed_los_jvp_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_view,
            target_view,
            power_view,
            tx_pol_view,
            rx_pol_view,
            row_count,
            tangent_source.data_ptr<float>(),
            tangent_target.data_ptr<float>(),
            tangent_tx_power.data_ptr<float>(),
            tangent_tx_polarization.data_ptr<float>(),
            tangent_rx_polarization.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["source"] = tangent_source;
    result["target"] = tangent_target;
    result["tx_power"] = tangent_tx_power;
    result["tx_polarization"] = tangent_tx_polarization;
    result["rx_polarization"] = tangent_rx_polarization;
    return result;
}

#undef kBlockSize

// ---- Consolidated from consumer_los_jones.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include <rayd/shared/rf/field_transport.cuh>

#include "../tensor_checks.h"

#include <cmath>
#include <cstdint>

namespace {

constexpr int kJonesBlockSize = 256;
namespace field = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;

__device__ __forceinline__ field::float3a load3(
    const float *values,
    int64_t row) {
    const int64_t base = row * 3;
    return field::make_f3(
        values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::Basis3 endpoint_basis(
    const float *values,
    int64_t endpoint,
    field::float3a direction) {
    const int64_t base = endpoint * 6;
    const field::float3a first = field::make_f3(
        values[base], values[base + 1], values[base + 2]);
    const field::float3a second = field::make_f3(
        values[base + 3], values[base + 4], values[base + 5]);
    field::Basis3 basis = field::basis_from_first_vector(
        direction,
        first,
        field::stable_perp_basis(direction, second));
    if (field::f3_dot(basis.v, second) < 0.0f) {
        basis.v = field::f3_neg(basis.v);
    }
    return basis;
}

__device__ __forceinline__ void store_basis(
    float *values,
    int64_t row,
    field::Basis3 basis) {
    const int64_t base = row * 6;
    values[base] = basis.u.x;
    values[base + 1] = basis.u.y;
    values[base + 2] = basis.u.z;
    values[base + 3] = basis.v.x;
    values[base + 4] = basis.v.y;
    values[base + 5] = basis.v.z;
}

__device__ __forceinline__ c10::complex<float> to_complex(
    field::Complex value) {
    return {value.re, value.im};
}

__global__ void consumer_los_jones_kernel(
    const int64_t *__restrict__ pair_index,
    const float *__restrict__ source_positions,
    const float *__restrict__ sink_positions,
    const float *__restrict__ source_reference_basis,
    const float *__restrict__ sink_reference_basis,
    int64_t row_count,
    int64_t num_tx,
    int64_t pair_count,
    float frequency_hz,
    c10::complex<float> *__restrict__ matrix,
    float *__restrict__ source_basis,
    float *__restrict__ sink_basis) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int64_t pair = pair_index[row];
        CUDA_KERNEL_ASSERT(pair >= 0 && pair < pair_count);
        const int64_t tx = pair % num_tx;
        const int64_t rx = pair / num_tx;
        const field::float3a source = load3(source_positions, tx);
        const field::float3a sink = load3(sink_positions, rx);
        const field::float3a direction = field::safe_normalize(
            field::f3_sub(sink, source),
            field::make_f3(0.0f, 0.0f, 1.0f));
        const field::Basis3 source_frame = endpoint_basis(
            source_reference_basis, tx, direction);
        const field::Basis3 sink_frame = endpoint_basis(
            sink_reference_basis, rx, direction);
        store_basis(source_basis, row, source_frame);
        store_basis(sink_basis, row, sink_frame);

        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz /
            transport::kSpeedOfLight;
        const field::Complex3 source_u =
            transport::free_space_complex3(
                source, sink, wave_number, source_frame.u);
        const field::Complex3 source_v =
            transport::free_space_complex3(
                source, sink, wave_number, source_frame.v);
        const int64_t base = row * 4;
        matrix[base] = to_complex(transport::project_receiver(
            source_u, direction, sink_frame.u));
        matrix[base + 1] = to_complex(transport::project_receiver(
            source_v, direction, sink_frame.u));
        matrix[base + 2] = to_complex(transport::project_receiver(
            source_u, direction, sink_frame.v));
        matrix[base + 3] = to_complex(transport::project_receiver(
            source_v, direction, sink_frame.v));
    }
}

void check_positions(
    const at::Tensor& values,
    const char *name,
    int device) {
    channel::check_tensor(values, name, at::kFloat, 2);
    TORCH_CHECK(values.size(1) == 3, name, " must have shape (N, 3)");
    TORCH_CHECK(values.get_device() == device, name, " must share pair device");
}

void check_basis(
    const at::Tensor& values,
    const char *name,
    int64_t rows,
    int device) {
    channel::check_tensor(values, name, at::kFloat, 3);
    TORCH_CHECK(
        values.sizes() == at::IntArrayRef({rows, 2, 3}),
        name,
        " must have shape (N, 2, 3)");
    TORCH_CHECK(values.get_device() == device, name, " must share pair device");
}

}  // namespace

pybind11::dict channel_consumer_los_jones(
    at::Tensor pair_index,
    at::Tensor source_positions,
    at::Tensor sink_positions,
    at::Tensor source_reference_basis,
    at::Tensor sink_reference_basis,
    double frequency_hz) {
    channel::check_tensor(pair_index, "pair_index", at::kLong, 1);
    const int device = pair_index.get_device();
    check_positions(source_positions, "source_positions", device);
    check_positions(sink_positions, "sink_positions", device);
    const int64_t num_tx = source_positions.size(0);
    const int64_t num_rx = sink_positions.size(0);
    check_basis(
        source_reference_basis,
        "source_reference_basis",
        num_tx,
        device);
    check_basis(
        sink_reference_basis,
        "sink_reference_basis",
        num_rx,
        device);
    TORCH_CHECK(
        std::isfinite(frequency_hz) && frequency_hz > 0.0,
        "frequency_hz must be finite and positive");
    TORCH_CHECK(
        pair_index.size(0) == 0 || num_tx > 0,
        "non-empty pair_index requires sources");

    const int64_t row_count = pair_index.size(0);
    auto complex_options = pair_index.options().dtype(at::kComplexFloat);
    auto float_options = pair_index.options().dtype(at::kFloat);
    auto matrix = at::empty({row_count, 2, 2}, complex_options);
    auto out_source_basis = at::empty({row_count, 2, 3}, float_options);
    auto out_sink_basis = at::empty({row_count, 2, 3}, float_options);
    if (row_count > 0) {
        const int64_t pair_count = num_tx * num_rx;
        TORCH_CHECK(pair_count > 0, "non-empty pair_index requires endpoint pairs");
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        const int blocks = static_cast<int>(
            (row_count + kJonesBlockSize - 1) / kJonesBlockSize);
        consumer_los_jones_kernel<<<
            blocks, kJonesBlockSize, 0, stream>>>(
            pair_index.data_ptr<int64_t>(),
            source_positions.data_ptr<float>(),
            sink_positions.data_ptr<float>(),
            source_reference_basis.data_ptr<float>(),
            sink_reference_basis.data_ptr<float>(),
            row_count,
            num_tx,
            pair_count,
            static_cast<float>(frequency_hz),
            matrix.data_ptr<c10::complex<float>>(),
            out_source_basis.data_ptr<float>(),
            out_sink_basis.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["matrix"] = matrix;
    result["source_basis"] = out_source_basis;
    result["sink_basis"] = out_sink_basis;
    return result;
}
