#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <cstdint>
#include <limits>

namespace {

constexpr int kDiffractionPathBlockSize = 256;
using channel_native::check_tensor;
using channel_native::check_vec3_table;

struct DiffractionPathInputView {
    const int *rx_id;
    const int *depth;
    const int *edge_id;
    const float *delay;
    const float *x_re;
    const float *x_im;
    const float *y_re;
    const float *y_im;
    const float *z_re;
    const float *z_im;
    const float *interaction_position;
};

struct DiffractionPathOutputView {
    bool *valid;
    int *rx_id;
    int *depth;
    int *edge_id;
    float *delay;
    float *x_re;
    float *x_im;
    float *y_re;
    float *y_im;
    float *z_re;
    float *z_im;
    float *interaction_position;
};

__global__ void diffraction_path_capacity_flags_kernel(
    const int *__restrict__ failure_state,
    const bool *__restrict__ valid,
    int *__restrict__ flags,
    int64_t input_capacity) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < input_capacity;
         row += stride) {
        flags[row] = failure_state[0] == 0 && valid[row] ? 1 : 0;
    }
}

__global__ void diffraction_path_capacity_init_kernel(
    DiffractionPathOutputView output,
    int64_t output_capacity) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < output_capacity;
         row += stride) {
        output.valid[row] = false;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.edge_id[row] = -1;
        output.delay[row] = -1.0f;
        output.x_re[row] = 0.0f;
        output.x_im[row] = 0.0f;
        output.y_re[row] = 0.0f;
        output.y_im[row] = 0.0f;
        output.z_re[row] = 0.0f;
        output.z_im[row] = 0.0f;
        const int64_t base = row * 3;
        output.interaction_position[base + 0] = 0.0f;
        output.interaction_position[base + 1] = 0.0f;
        output.interaction_position[base + 2] = 0.0f;
    }
}

__global__ void diffraction_path_capacity_status_kernel(
    int *__restrict__ failure_state,
    const int *__restrict__ reported_count,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    int64_t input_capacity,
    int64_t output_capacity,
    int *__restrict__ num_paths,
    bool *__restrict__ overflow) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (failure_state[0] != 0) {
        num_paths[0] = 0;
        overflow[0] = false;
        return;
    }
    int selected_count = 0;
    if (input_capacity > 0) {
        const int last = static_cast<int>(input_capacity - 1);
        selected_count = offsets[last] + flags[last];
    }
    const bool invalid_count = reported_count[0] != selected_count;
    const bool did_overflow = invalid_count || selected_count > output_capacity;
    num_paths[0] = did_overflow ? 0 : selected_count;
    overflow[0] = did_overflow;
    if (invalid_count) {
        atomicOr(
            failure_state,
            channel_native::capacity::kDiffractionPathContractError);
    } else if (selected_count > output_capacity) {
        atomicOr(
            failure_state,
            channel_native::capacity::kDiffractionPathOverflow);
    }
}

__global__ void diffraction_path_capacity_gather_kernel(
    const int *__restrict__ failure_state,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    const bool *__restrict__ overflow,
    DiffractionPathInputView input,
    DiffractionPathOutputView output,
    int64_t input_capacity) {
    if (failure_state[0] != 0 || overflow[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < input_capacity;
         row += stride) {
        if (flags[row] == 0) {
            continue;
        }
        const int dst = offsets[row];
        output.valid[dst] = true;
        output.rx_id[dst] = input.rx_id[row];
        output.depth[dst] = input.depth[row];
        output.edge_id[dst] = input.edge_id[row];
        output.delay[dst] = input.delay[row];
        output.x_re[dst] = input.x_re[row];
        output.x_im[dst] = input.x_im[row];
        output.y_re[dst] = input.y_re[row];
        output.y_im[dst] = input.y_im[row];
        output.z_re[dst] = input.z_re[row];
        output.z_im[dst] = input.z_im[row];
        const int64_t source_base = row * 3;
        const int64_t destination_base = static_cast<int64_t>(dst) * 3;
        output.interaction_position[destination_base + 0] =
            input.interaction_position[source_base + 0];
        output.interaction_position[destination_base + 1] =
            input.interaction_position[source_base + 1];
        output.interaction_position[destination_base + 2] =
            input.interaction_position[source_base + 2];
    }
}

void check_diffraction_path_input(
    const at::Tensor& tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t input_capacity,
    int device) {
    check_tensor(tensor, name, dtype, 1);
    TORCH_CHECK(tensor.size(0) == input_capacity, name, " must match valid capacity");
    TORCH_CHECK(tensor.get_device() == device, name, " must share valid device");
}

}  // namespace

pybind11::dict cn_deterministic_diffraction_order1_capacity_block(
    at::Tensor failure_state,
    at::Tensor count,
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor x_re,
    at::Tensor x_im,
    at::Tensor y_re,
    at::Tensor y_im,
    at::Tensor z_re,
    at::Tensor z_im,
    at::Tensor interaction_position,
    int64_t output_capacity) {
    check_tensor(count, "count", at::kInt, 1);
    TORCH_CHECK(count.size(0) == 1, "count must have shape (1,)");
    check_tensor(valid, "valid", at::kBool, 1);
    channel_native::capacity::validate_failure_state(failure_state, valid);
    TORCH_CHECK(output_capacity >= 0, "output_capacity must be non-negative");
    TORCH_CHECK(
        output_capacity <= std::numeric_limits<int>::max(),
        "output_capacity exceeds int32 indexing capacity");

    const int64_t input_capacity = valid.size(0);
    TORCH_CHECK(
        input_capacity <= std::numeric_limits<int>::max(),
        "input diffraction capacity exceeds int32 indexing capacity");
    const int device = valid.get_device();
    TORCH_CHECK(count.get_device() == device, "count must share valid device");
    check_diffraction_path_input(rx_id, "rx_id", at::kInt, input_capacity, device);
    check_diffraction_path_input(depth, "depth", at::kInt, input_capacity, device);
    check_diffraction_path_input(edge_id, "edge_id", at::kInt, input_capacity, device);
    check_diffraction_path_input(delay, "delay_s", at::kFloat, input_capacity, device);
    check_diffraction_path_input(x_re, "x_re", at::kFloat, input_capacity, device);
    check_diffraction_path_input(x_im, "x_im", at::kFloat, input_capacity, device);
    check_diffraction_path_input(y_re, "y_re", at::kFloat, input_capacity, device);
    check_diffraction_path_input(y_im, "y_im", at::kFloat, input_capacity, device);
    check_diffraction_path_input(z_re, "z_re", at::kFloat, input_capacity, device);
    check_diffraction_path_input(z_im, "z_im", at::kFloat, input_capacity, device);
    check_vec3_table(interaction_position, "interaction_position");
    TORCH_CHECK(
        interaction_position.size(0) == input_capacity,
        "interaction_position must match valid capacity");
    TORCH_CHECK(
        interaction_position.get_device() == device,
        "interaction_position must share valid device");

    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto float_options = valid.options().dtype(at::kFloat);
    auto out_valid = at::empty({output_capacity}, bool_options);
    auto out_rx_id = at::empty({output_capacity}, int_options);
    auto out_depth = at::empty({output_capacity}, int_options);
    auto out_edge_id = at::empty({output_capacity}, int_options);
    auto out_delay = at::empty({output_capacity}, float_options);
    auto out_x_re = at::empty({output_capacity}, float_options);
    auto out_x_im = at::empty({output_capacity}, float_options);
    auto out_y_re = at::empty({output_capacity}, float_options);
    auto out_y_im = at::empty({output_capacity}, float_options);
    auto out_z_re = at::empty({output_capacity}, float_options);
    auto out_z_im = at::empty({output_capacity}, float_options);
    auto out_interaction_position = at::empty({output_capacity, 3}, float_options);
    auto num_paths = at::empty({1}, int_options);
    auto overflow = at::empty({1}, bool_options);

    const DiffractionPathInputView input{
        rx_id.data_ptr<int>(),
        depth.data_ptr<int>(),
        edge_id.data_ptr<int>(),
        delay.data_ptr<float>(),
        x_re.data_ptr<float>(),
        x_im.data_ptr<float>(),
        y_re.data_ptr<float>(),
        y_im.data_ptr<float>(),
        z_re.data_ptr<float>(),
        z_im.data_ptr<float>(),
        interaction_position.data_ptr<float>()};
    const DiffractionPathOutputView output{
        out_valid.data_ptr<bool>(),
        out_rx_id.data_ptr<int>(),
        out_depth.data_ptr<int>(),
        out_edge_id.data_ptr<int>(),
        out_delay.data_ptr<float>(),
        out_x_re.data_ptr<float>(),
        out_x_im.data_ptr<float>(),
        out_y_re.data_ptr<float>(),
        out_y_im.data_ptr<float>(),
        out_z_re.data_ptr<float>(),
        out_z_im.data_ptr<float>(),
        out_interaction_position.data_ptr<float>()};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (output_capacity > 0) {
        const int output_blocks = static_cast<int>(
            (output_capacity + kDiffractionPathBlockSize - 1) /
            kDiffractionPathBlockSize);
        diffraction_path_capacity_init_kernel<<<
            output_blocks, kDiffractionPathBlockSize, 0, stream>>>(
            output, output_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto flags = at::empty({input_capacity}, int_options);
    auto offsets = at::empty({input_capacity}, int_options);
    if (input_capacity > 0) {
        const int input_blocks = static_cast<int>(
            (input_capacity + kDiffractionPathBlockSize - 1) /
            kDiffractionPathBlockSize);
        diffraction_path_capacity_flags_kernel<<<
            input_blocks, kDiffractionPathBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(),
            valid.data_ptr<bool>(),
            flags.data_ptr<int>(),
            input_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t scratch_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(input_capacity),
            stream));
        auto scratch = at::empty(
            {static_cast<int64_t>(scratch_bytes)},
            valid.options().dtype(at::kByte));
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            scratch.data_ptr<uint8_t>(),
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(input_capacity),
            stream));
    }

    diffraction_path_capacity_status_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int>(),
        count.data_ptr<int>(),
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        input_capacity,
        output_capacity,
        num_paths.data_ptr<int>(),
        overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (input_capacity > 0) {
        const int input_blocks = static_cast<int>(
            (input_capacity + kDiffractionPathBlockSize - 1) /
            kDiffractionPathBlockSize);
        diffraction_path_capacity_gather_kernel<<<
            input_blocks, kDiffractionPathBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(),
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            overflow.data_ptr<bool>(),
            input,
            output,
            input_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["valid"] = out_valid;
    result["rx_id"] = out_rx_id;
    result["depth"] = out_depth;
    result["edge_id"] = out_edge_id;
    result["delay_s"] = out_delay;
    result["x_re"] = out_x_re;
    result["x_im"] = out_x_im;
    result["y_re"] = out_y_re;
    result["y_im"] = out_y_im;
    result["z_re"] = out_z_re;
    result["z_im"] = out_z_im;
    result["interaction_position"] = out_interaction_position;
    result["num_paths"] = num_paths;
    result["overflow"] = overflow;
    return result;
}
