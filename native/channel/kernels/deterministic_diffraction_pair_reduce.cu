#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;
constexpr unsigned kFullWarpMask = 0xffffffffu;

using channel::check_tensor;
using Complex = c10::complex<float>;

struct SixFields {
    const float* x_re;
    const float* x_im;
    const float* y_re;
    const float* y_im;
    const float* z_re;
    const float* z_im;
};

struct SixOutputs {
    float* x_re;
    float* x_im;
    float* y_re;
    float* y_im;
    float* z_re;
    float* z_im;
};

template <typename T>
struct OptionalView {
    const T* data;
    int64_t stride0;
    int64_t stride1;
    bool conjugated;
};

struct SixTangents {
    OptionalView<float> x_re;
    OptionalView<float> x_im;
    OptionalView<float> y_re;
    OptionalView<float> y_im;
    OptionalView<float> z_re;
    OptionalView<float> z_im;
};

__device__ __forceinline__ float optional_load(
    OptionalView<float> value,
    int64_t row) {
    return value.data == nullptr ? 0.0f : value.data[row * value.stride0];
}

__device__ __forceinline__ float optional_load(
    OptionalView<float> value,
    int64_t row,
    int64_t column) {
    return value.data == nullptr
        ? 0.0f
        : value.data[row * value.stride0 + column * value.stride1];
}

__device__ __forceinline__ Complex optional_load(
    OptionalView<Complex> value,
    int64_t row,
    int64_t column) {
    if (value.data == nullptr) {
        return Complex(0.0f, 0.0f);
    }
    const Complex stored =
        value.data[row * value.stride0 + column * value.stride1];
    return value.conjugated
        ? Complex(stored.real(), -stored.imag())
        : stored;
}

__global__ void deterministic_diffraction_pair_reduce_status_kernel(
    int32_t* failure_state,
    const int32_t* reported_count,
    const bool* valid,
    int64_t rows) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || failure_state[0] != 0) {
        return;
    }
    int64_t selected_count = 0;
    for (int64_t row = 0; row < rows; ++row) {
        selected_count += valid[row] ? 1 : 0;
    }
    if (static_cast<int64_t>(reported_count[0]) != selected_count) {
        atomicOr(
            failure_state,
            channel::capacity::kDiffractionPathContractError);
    }
}

__global__ void deterministic_diffraction_pair_reduce_kernel(
    const int32_t* failure_state,
    const bool* valid,
    SixFields fields,
    int64_t pair_count,
    int64_t state_capacity,
    Complex* field_xyz,
    float* power) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int64_t pair =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (pair >= pair_count) {
        return;
    }

    // ADR-030: transaction failure is observed before validity or payload.
    if (failure_state[0] != 0) {
        if (lane == 0) {
            field_xyz[pair * 3] = Complex(0.0f, 0.0f);
            field_xyz[pair * 3 + 1] = Complex(0.0f, 0.0f);
            field_xyz[pair * 3 + 2] = Complex(0.0f, 0.0f);
            power[pair] = 0.0f;
        }
        return;
    }

    float sx_re = 0.0f;
    float sx_im = 0.0f;
    float sy_re = 0.0f;
    float sy_im = 0.0f;
    float sz_re = 0.0f;
    float sz_im = 0.0f;
    const int64_t pair_base = pair * state_capacity;

    for (int64_t group = 0; group < state_capacity; group += kWarpSize) {
        const int64_t state = group + lane;
        float lane_x_re = 0.0f;
        float lane_x_im = 0.0f;
        float lane_y_re = 0.0f;
        float lane_y_im = 0.0f;
        float lane_z_re = 0.0f;
        float lane_z_im = 0.0f;
        int lane_valid = 0;
        if (state < state_capacity) {
            const int64_t row = pair_base + state;
            if (valid[row]) {
                lane_valid = 1;
                lane_x_re = fields.x_re[row];
                lane_x_im = fields.x_im[row];
                lane_y_re = fields.y_re[row];
                lane_y_im = fields.y_im[row];
                lane_z_re = fields.z_re[row];
                lane_z_im = fields.z_im[row];
            }
        }

#pragma unroll
        for (int source_lane = 0; source_lane < kWarpSize; ++source_lane) {
            const int source_valid =
                __shfl_sync(kFullWarpMask, lane_valid, source_lane);
            const float value_x_re =
                __shfl_sync(kFullWarpMask, lane_x_re, source_lane);
            const float value_x_im =
                __shfl_sync(kFullWarpMask, lane_x_im, source_lane);
            const float value_y_re =
                __shfl_sync(kFullWarpMask, lane_y_re, source_lane);
            const float value_y_im =
                __shfl_sync(kFullWarpMask, lane_y_im, source_lane);
            const float value_z_re =
                __shfl_sync(kFullWarpMask, lane_z_re, source_lane);
            const float value_z_im =
                __shfl_sync(kFullWarpMask, lane_z_im, source_lane);
            if (lane == 0 && group + source_lane < state_capacity &&
                source_valid != 0) {
                sx_re = sx_re + value_x_re;
                sx_im = sx_im + value_x_im;
                sy_re = sy_re + value_y_re;
                sy_im = sy_im + value_y_im;
                sz_re = sz_re + value_z_re;
                sz_im = sz_im + value_z_im;
            }
        }
    }

    if (lane == 0) {
        field_xyz[pair * 3] = Complex(sx_re, sx_im);
        field_xyz[pair * 3 + 1] = Complex(sy_re, sy_im);
        field_xyz[pair * 3 + 2] = Complex(sz_re, sz_im);
        const float px = sx_re * sx_re + sx_im * sx_im;
        const float py = sy_re * sy_re + sy_im * sy_im;
        const float pz = sz_re * sz_re + sz_im * sz_im;
        power[pair] = (px + py) + pz;
    }
}

__global__ void deterministic_diffraction_pair_reduce_backward_kernel(
    const int32_t* failure_state,
    const bool* valid,
    const Complex* field_xyz,
    OptionalView<Complex> grad_field_xyz,
    OptionalView<float> grad_power,
    int64_t rows,
    int64_t state_capacity,
    SixOutputs gradients) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    gradients.x_re[row] = 0.0f;
    gradients.x_im[row] = 0.0f;
    gradients.y_re[row] = 0.0f;
    gradients.y_im[row] = 0.0f;
    gradients.z_re[row] = 0.0f;
    gradients.z_im[row] = 0.0f;
    // ADR-030: do not read validity or payload after a transaction failure.
    if (failure_state[0] != 0 || !valid[row]) {
        return;
    }

    const int64_t pair = row / state_capacity;
    const Complex grad_x = optional_load(grad_field_xyz, pair, 0);
    const Complex grad_y = optional_load(grad_field_xyz, pair, 1);
    const Complex grad_z = optional_load(grad_field_xyz, pair, 2);
    if (grad_power.data == nullptr) {
        gradients.x_re[row] = grad_x.real();
        gradients.x_im[row] = grad_x.imag();
        gradients.y_re[row] = grad_y.real();
        gradients.y_im[row] = grad_y.imag();
        gradients.z_re[row] = grad_z.real();
        gradients.z_im[row] = grad_z.imag();
        return;
    }

    const Complex sum_x = field_xyz[pair * 3];
    const Complex sum_y = field_xyz[pair * 3 + 1];
    const Complex sum_z = field_xyz[pair * 3 + 2];
    const float power_cotangent = optional_load(grad_power, pair);
    // Parentheses and separate operations are frozen by ADR-030.
    gradients.x_re[row] =
        grad_x.real() + (2.0f * sum_x.real()) * power_cotangent;
    gradients.x_im[row] =
        grad_x.imag() + (2.0f * sum_x.imag()) * power_cotangent;
    gradients.y_re[row] =
        grad_y.real() + (2.0f * sum_y.real()) * power_cotangent;
    gradients.y_im[row] =
        grad_y.imag() + (2.0f * sum_y.imag()) * power_cotangent;
    gradients.z_re[row] =
        grad_z.real() + (2.0f * sum_z.real()) * power_cotangent;
    gradients.z_im[row] =
        grad_z.imag() + (2.0f * sum_z.imag()) * power_cotangent;
}

__global__ void deterministic_diffraction_pair_reduce_jvp_kernel(
    const int32_t* failure_state,
    const bool* valid,
    const Complex* field_xyz,
    SixTangents tangents,
    int64_t pair_count,
    int64_t state_capacity,
    Complex* tangent_field_xyz,
    float* tangent_power) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int64_t pair =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (pair >= pair_count) {
        return;
    }
    if (failure_state[0] != 0) {
        if (lane == 0) {
            tangent_field_xyz[pair * 3] = Complex(0.0f, 0.0f);
            tangent_field_xyz[pair * 3 + 1] = Complex(0.0f, 0.0f);
            tangent_field_xyz[pair * 3 + 2] = Complex(0.0f, 0.0f);
            tangent_power[pair] = 0.0f;
        }
        return;
    }
    const bool has_tangent =
        tangents.x_re.data != nullptr || tangents.x_im.data != nullptr ||
        tangents.y_re.data != nullptr || tangents.y_im.data != nullptr ||
        tangents.z_re.data != nullptr || tangents.z_im.data != nullptr;
    if (!has_tangent) {
        if (lane == 0) {
            tangent_field_xyz[pair * 3] = Complex(0.0f, 0.0f);
            tangent_field_xyz[pair * 3 + 1] = Complex(0.0f, 0.0f);
            tangent_field_xyz[pair * 3 + 2] = Complex(0.0f, 0.0f);
            tangent_power[pair] = 0.0f;
        }
        return;
    }

    float dsx_re = 0.0f;
    float dsx_im = 0.0f;
    float dsy_re = 0.0f;
    float dsy_im = 0.0f;
    float dsz_re = 0.0f;
    float dsz_im = 0.0f;
    const int64_t pair_base = pair * state_capacity;

    for (int64_t group = 0; group < state_capacity; group += kWarpSize) {
        const int64_t state = group + lane;
        float lane_x_re = 0.0f;
        float lane_x_im = 0.0f;
        float lane_y_re = 0.0f;
        float lane_y_im = 0.0f;
        float lane_z_re = 0.0f;
        float lane_z_im = 0.0f;
        int lane_valid = 0;
        if (state < state_capacity) {
            const int64_t row = pair_base + state;
            if (valid[row]) {
                lane_valid = 1;
                lane_x_re = optional_load(tangents.x_re, row);
                lane_x_im = optional_load(tangents.x_im, row);
                lane_y_re = optional_load(tangents.y_re, row);
                lane_y_im = optional_load(tangents.y_im, row);
                lane_z_re = optional_load(tangents.z_re, row);
                lane_z_im = optional_load(tangents.z_im, row);
            }
        }

#pragma unroll
        for (int source_lane = 0; source_lane < kWarpSize; ++source_lane) {
            const int source_valid =
                __shfl_sync(kFullWarpMask, lane_valid, source_lane);
            const float value_x_re =
                __shfl_sync(kFullWarpMask, lane_x_re, source_lane);
            const float value_x_im =
                __shfl_sync(kFullWarpMask, lane_x_im, source_lane);
            const float value_y_re =
                __shfl_sync(kFullWarpMask, lane_y_re, source_lane);
            const float value_y_im =
                __shfl_sync(kFullWarpMask, lane_y_im, source_lane);
            const float value_z_re =
                __shfl_sync(kFullWarpMask, lane_z_re, source_lane);
            const float value_z_im =
                __shfl_sync(kFullWarpMask, lane_z_im, source_lane);
            if (lane == 0 && group + source_lane < state_capacity &&
                source_valid != 0) {
                dsx_re = dsx_re + value_x_re;
                dsx_im = dsx_im + value_x_im;
                dsy_re = dsy_re + value_y_re;
                dsy_im = dsy_im + value_y_im;
                dsz_re = dsz_re + value_z_re;
                dsz_im = dsz_im + value_z_im;
            }
        }
    }

    if (lane == 0) {
        tangent_field_xyz[pair * 3] = Complex(dsx_re, dsx_im);
        tangent_field_xyz[pair * 3 + 1] = Complex(dsy_re, dsy_im);
        tangent_field_xyz[pair * 3 + 2] = Complex(dsz_re, dsz_im);
        const Complex sum_x = field_xyz[pair * 3];
        const Complex sum_y = field_xyz[pair * 3 + 1];
        const Complex sum_z = field_xyz[pair * 3 + 2];
        const float t0 = sum_x.real() * dsx_re;
        const float t1 = sum_x.imag() * dsx_im;
        const float t2 = sum_y.real() * dsy_re;
        const float t3 = sum_y.imag() * dsy_im;
        const float t4 = sum_z.real() * dsz_re;
        const float t5 = sum_z.imag() * dsz_im;
        const float dot = (((((t0 + t1) + t2) + t3) + t4) + t5);
        tangent_power[pair] = 2.0f * dot;
    }
}

int launch_pair_blocks(int64_t pair_count) {
    return static_cast<int>(
        (pair_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

int launch_row_blocks(int64_t rows) {
    return static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
}

void check_same_device(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name,
        " must share valid device");
}

int64_t check_capacity_contract(
    const at::Tensor& failure_state,
    const at::Tensor& valid,
    int64_t pair_count,
    int64_t state_capacity) {
    check_tensor(valid, "valid", at::kBool, 1);
    channel::capacity::validate_failure_state(failure_state, valid);
    TORCH_CHECK(pair_count >= 0, "pair_count must be non-negative");
    TORCH_CHECK(state_capacity >= 0, "state_capacity must be non-negative");
    TORCH_CHECK(
        pair_count == 0 ||
            state_capacity <= std::numeric_limits<int64_t>::max() / pair_count,
        "pair-major state capacity overflows int64");
    const int64_t rows = pair_count * state_capacity;
    TORCH_CHECK(valid.numel() == rows, "valid must match pair_count * state_capacity");
    return rows;
}

void check_row_field(
    const at::Tensor& tensor,
    const at::Tensor& valid,
    const char* name) {
    check_tensor(tensor, name, at::kFloat, 1);
    check_same_device(tensor, valid, name);
    TORCH_CHECK(tensor.numel() == valid.numel(), name, " must match valid shape");
}

void check_pair_field(
    const at::Tensor& tensor,
    const at::Tensor& valid,
    const char* name,
    int64_t pair_count) {
    check_tensor(tensor, name, at::kComplexFloat, 2);
    check_same_device(tensor, valid, name);
    TORCH_CHECK(
        tensor.size(0) == pair_count && tensor.size(1) == 3,
        name,
        " must have shape (pair_count, 3)");
}

template <typename T>
OptionalView<T> optional_view(
    const std::optional<at::Tensor>& value,
    const at::Tensor& valid,
    const char* name,
    at::ScalarType dtype,
    at::IntArrayRef shape) {
    if (!value.has_value()) {
        return {nullptr, 0, 0, false};
    }
    TORCH_CHECK(value->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value->scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(
        value->dim() == static_cast<int64_t>(shape.size()),
        name,
        " has wrong rank");
    check_same_device(*value, valid, name);
    TORCH_CHECK(value->sizes() == shape, name, " has wrong shape");
    return {
        value->data_ptr<T>(),
        value->stride(0),
        value->dim() > 1 ? value->stride(1) : 0,
        value->is_conj()};
}

pybind11::dict reduction_result(at::Tensor field_xyz, at::Tensor power) {
    pybind11::dict out;
    out["field_xyz"] = std::move(field_xyz);
    out["power"] = std::move(power);
    return out;
}

pybind11::dict gradient_result(
    at::Tensor x_re,
    at::Tensor x_im,
    at::Tensor y_re,
    at::Tensor y_im,
    at::Tensor z_re,
    at::Tensor z_im) {
    pybind11::dict out;
    out["grad_x_re"] = std::move(x_re);
    out["grad_x_im"] = std::move(x_im);
    out["grad_y_re"] = std::move(y_re);
    out["grad_y_im"] = std::move(y_im);
    out["grad_z_re"] = std::move(z_re);
    out["grad_z_im"] = std::move(z_im);
    return out;
}

}  // namespace

pybind11::dict channel_deterministic_diffraction_pair_reduce(
    at::Tensor failure_state,
    at::Tensor reported_count,
    at::Tensor valid,
    at::Tensor x_re,
    at::Tensor x_im,
    at::Tensor y_re,
    at::Tensor y_im,
    at::Tensor z_re,
    at::Tensor z_im,
    int64_t pair_count,
    int64_t state_capacity) {
    const int64_t rows =
        check_capacity_contract(failure_state, valid, pair_count, state_capacity);
    check_tensor(reported_count, "reported_count", at::kInt, 1);
    check_same_device(reported_count, valid, "reported_count");
    TORCH_CHECK(
        reported_count.numel() == 1,
        "reported_count must have shape (1,)");
    for (auto named : std::initializer_list<std::pair<at::Tensor*, const char*>>{
             {&x_re, "x_re"},
             {&x_im, "x_im"},
             {&y_re, "y_re"},
             {&y_im, "y_im"},
             {&z_re, "z_re"},
             {&z_im, "z_im"}}) {
        check_row_field(*named.first, valid, named.second);
    }

    c10::cuda::CUDAGuard device_guard(valid.device());
    auto field_xyz = at::empty(
        {pair_count, 3}, valid.options().dtype(at::kComplexFloat));
    auto power = at::empty({pair_count}, valid.options().dtype(at::kFloat));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    deterministic_diffraction_pair_reduce_status_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int32_t>(),
        reported_count.data_ptr<int32_t>(),
        valid.data_ptr<bool>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (pair_count > 0) {
        deterministic_diffraction_pair_reduce_kernel<<<
            launch_pair_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int32_t>(),
            valid.data_ptr<bool>(),
            {x_re.data_ptr<float>(),
             x_im.data_ptr<float>(),
             y_re.data_ptr<float>(),
             y_im.data_ptr<float>(),
             z_re.data_ptr<float>(),
             z_im.data_ptr<float>()},
            pair_count,
            state_capacity,
            field_xyz.data_ptr<Complex>(),
            power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return reduction_result(std::move(field_xyz), std::move(power));
}

pybind11::dict channel_deterministic_diffraction_pair_reduce_backward(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor field_xyz,
    std::optional<at::Tensor> grad_field_xyz,
    std::optional<at::Tensor> grad_power,
    int64_t pair_count,
    int64_t state_capacity) {
    const int64_t rows =
        check_capacity_contract(failure_state, valid, pair_count, state_capacity);
    check_pair_field(field_xyz, valid, "field_xyz", pair_count);
    const int64_t pair_field_shape[] = {pair_count, 3};
    const int64_t pair_shape[] = {pair_count};
    const OptionalView<Complex> grad_field = optional_view<Complex>(
        grad_field_xyz,
        valid,
        "grad_field_xyz",
        at::kComplexFloat,
        pair_field_shape);
    const OptionalView<float> grad_power_view = optional_view<float>(
        grad_power, valid, "grad_power", at::kFloat, pair_shape);

    c10::cuda::CUDAGuard device_guard(valid.device());
    auto options = valid.options().dtype(at::kFloat);
    auto grad_x_re = at::empty({rows}, options);
    auto grad_x_im = at::empty({rows}, options);
    auto grad_y_re = at::empty({rows}, options);
    auto grad_y_im = at::empty({rows}, options);
    auto grad_z_re = at::empty({rows}, options);
    auto grad_z_im = at::empty({rows}, options);
    if (rows > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        deterministic_diffraction_pair_reduce_backward_kernel<<<
            launch_row_blocks(rows), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int32_t>(),
            valid.data_ptr<bool>(),
            field_xyz.data_ptr<Complex>(),
            grad_field,
            grad_power_view,
            rows,
            state_capacity,
            {grad_x_re.data_ptr<float>(),
             grad_x_im.data_ptr<float>(),
             grad_y_re.data_ptr<float>(),
             grad_y_im.data_ptr<float>(),
             grad_z_re.data_ptr<float>(),
             grad_z_im.data_ptr<float>()});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return gradient_result(
        std::move(grad_x_re),
        std::move(grad_x_im),
        std::move(grad_y_re),
        std::move(grad_y_im),
        std::move(grad_z_re),
        std::move(grad_z_im));
}

pybind11::dict channel_deterministic_diffraction_pair_reduce_jvp(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor field_xyz,
    std::optional<at::Tensor> tangent_x_re,
    std::optional<at::Tensor> tangent_x_im,
    std::optional<at::Tensor> tangent_y_re,
    std::optional<at::Tensor> tangent_y_im,
    std::optional<at::Tensor> tangent_z_re,
    std::optional<at::Tensor> tangent_z_im,
    int64_t pair_count,
    int64_t state_capacity) {
    const int64_t rows =
        check_capacity_contract(failure_state, valid, pair_count, state_capacity);
    check_pair_field(field_xyz, valid, "field_xyz", pair_count);
    const int64_t row_shape[] = {rows};
    const SixTangents tangents{
        optional_view<float>(
            tangent_x_re, valid, "tangent_x_re", at::kFloat, row_shape),
        optional_view<float>(
            tangent_x_im, valid, "tangent_x_im", at::kFloat, row_shape),
        optional_view<float>(
            tangent_y_re, valid, "tangent_y_re", at::kFloat, row_shape),
        optional_view<float>(
            tangent_y_im, valid, "tangent_y_im", at::kFloat, row_shape),
        optional_view<float>(
            tangent_z_re, valid, "tangent_z_re", at::kFloat, row_shape),
        optional_view<float>(
            tangent_z_im, valid, "tangent_z_im", at::kFloat, row_shape)};

    c10::cuda::CUDAGuard device_guard(valid.device());
    auto tangent_field_xyz = at::empty(
        {pair_count, 3}, valid.options().dtype(at::kComplexFloat));
    auto tangent_power =
        at::empty({pair_count}, valid.options().dtype(at::kFloat));
    if (pair_count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        deterministic_diffraction_pair_reduce_jvp_kernel<<<
            launch_pair_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int32_t>(),
            valid.data_ptr<bool>(),
            field_xyz.data_ptr<Complex>(),
            tangents,
            pair_count,
            state_capacity,
            tangent_field_xyz.data_ptr<Complex>(),
            tangent_power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return reduction_result(
        std::move(tangent_field_xyz), std::move(tangent_power));
}
