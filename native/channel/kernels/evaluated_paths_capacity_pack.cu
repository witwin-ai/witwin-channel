#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "deterministic_capacity_finalize.h"
#include "evaluated_paths_payload_plumbing.h"

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kPackBlockSize = 256;
using channel::check_tensor;

using PackInput = channel::evaluated_paths::PayloadInputView;

struct PackOutput {
    int64_t *selected_row_index;
    bool *valid;
    int *num_paths;
    bool *overflow;
    channel::evaluated_paths::PayloadOutputView payload;
};

__global__ void evaluated_paths_capacity_init_kernel(
    PackOutput output,
    int64_t row_capacity,
    int64_t pair_count,
    int64_t sequence_width) {
    const int64_t count = row_capacity > pair_count ? row_capacity : pair_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count;
         row += stride) {
        if (row < pair_count) {
            output.num_paths[row] = 0;
        }
        if (row >= row_capacity) {
            continue;
        }
        output.selected_row_index[row] = -1;
        output.valid[row] = false;
        channel::evaluated_paths::initialize_row(
            output.payload, row, sequence_width);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output.overflow[0] = false;
    }
}

__global__ void evaluated_paths_capacity_gather_kernel(
    PackInput input,
    PackOutput output,
    const int *__restrict__ failure_state,
    const int *__restrict__ overflow_flag,
    const int *__restrict__ contract_error,
    int64_t row_capacity,
    int64_t sequence_width) {
    if (failure_state[0] != 0 || overflow_flag[0] || contract_error[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!output.valid[destination]) {
            continue;
        }
        const int64_t source = output.selected_row_index[destination];
        channel::evaluated_paths::copy_row(
            input, output.payload, source, destination, sequence_width);
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kPackBlockSize - 1) / kPackBlockSize);
}

}  // namespace

pybind11::dict channel_evaluated_paths_capacity_pack(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor material_id,
    at::Tensor primitive_sequence,
    at::Tensor material_sequence,
    at::Tensor interaction_type,
    at::Tensor path_length_m,
    at::Tensor delay_s,
    at::Tensor field_direction,
    at::Tensor interaction_position,
    at::Tensor interaction_normal,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor path_gain,
    at::Tensor path_field,
    at::Tensor field_xyz,
    at::Tensor coefficient,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_tensor(valid, "valid", at::kBool, 1);
    const int64_t candidate_count = valid.size(0);
    const int device = valid.get_device();
    const channel::evaluated_paths::PayloadTensors payload{
        tx_id, rx_id, depth, component_id, primitive_id, edge_id, material_id,
        primitive_sequence, material_sequence, interaction_type, path_length_m,
        delay_s, field_direction, interaction_position, interaction_normal,
        interaction_positions, interaction_normals, path_gain, path_field,
        field_xyz, coefficient};
    const int64_t sequence_width =
        channel::evaluated_paths::validate_payload(
            payload, candidate_count, device);
    const int64_t row_capacity =
        channel::capacity::deterministic_capacity_validate(
            valid,
            tx_id,
            rx_id,
            pair_count,
            num_tx,
            num_rx,
            path_capacity_per_pair);

    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto long_options = valid.options().dtype(at::kLong);
    auto selected_row_index = at::empty({row_capacity}, long_options);
    auto out_valid = at::empty({row_capacity}, bool_options);
    auto num_paths = at::empty({pair_count}, int_options);
    auto overflow = at::empty({1}, bool_options);
    auto out = channel::evaluated_paths::allocate_payload(
        valid, row_capacity, sequence_width);

    const PackInput input = channel::evaluated_paths::input_view(payload);
    const PackOutput output{
        selected_row_index.data_ptr<int64_t>(), out_valid.data_ptr<bool>(),
        num_paths.data_ptr<int>(), overflow.data_ptr<bool>(),
        channel::evaluated_paths::output_view(out)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = std::max<int64_t>(1, std::max(row_capacity, pair_count));
    evaluated_paths_capacity_init_kernel<<<
        launch_blocks(init_count), kPackBlockSize, 0, stream>>>(
        output, row_capacity, pair_count, sequence_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto state = channel::capacity::deterministic_capacity_finalize_no_trap(
        failure_state,
        valid,
        tx_id,
        rx_id,
        pair_count,
        num_tx,
        num_rx,
        path_capacity_per_pair,
        selected_row_index,
        out_valid,
        num_paths,
        false);
    if (row_capacity > 0) {
        evaluated_paths_capacity_gather_kernel<<<
            launch_blocks(row_capacity), kPackBlockSize, 0, stream>>>(
            input,
            output,
            state.failure_state.data_ptr<int>(),
            state.overflow_flag.data_ptr<int>(),
            state.contract_error.data_ptr<int>(),
            row_capacity,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    channel::capacity::deterministic_capacity_publish_status(
        state, overflow, stream);

    pybind11::dict result;
    result["selected_row_index"] = selected_row_index;
    result["valid"] = out_valid;
    result["num_paths"] = num_paths;
    result["overflow"] = overflow;
    out.append_to(result);
    return result;
}
