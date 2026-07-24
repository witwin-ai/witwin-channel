#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"
#include "capacity_failure_state.h"
#include "evaluated_paths_payload_plumbing.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

constexpr int kGatherBlockSize = 256;
using channel::check_tensor;

struct GatherInput {
    const bool *valid;
    channel::evaluated_paths::PayloadInputView payload;
};

struct GatherOutput {
    int64_t *selected_row_index;
    bool *valid;
    int *num_selected;
    int *num_paths;
    channel::evaluated_paths::PayloadOutputView payload;
};

__global__ void canonical_gather_init_kernel(
    GatherOutput output,
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
        output.num_selected[0] = 0;
    }
}

__global__ void canonical_selection_structure_kernel(
    int *__restrict__ failure_state,
    const int64_t *__restrict__ selected_row_index,
    const bool *__restrict__ selected_valid,
    const bool *__restrict__ candidate_valid,
    unsigned int *__restrict__ seen_source,
    int *__restrict__ observed_selected,
    int64_t capacity) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < capacity;
         row += stride) {
        if (!selected_valid[row]) {
            continue;
        }
        if (row > 0 && !selected_valid[row - 1]) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
            continue;
        }
        const int64_t source = selected_row_index[row];
        if (source < 0 || source >= capacity || !candidate_valid[source]) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
            continue;
        }
        const int64_t word = source >> 5;
        const unsigned int mask = 1u << static_cast<unsigned int>(source & 31);
        const unsigned int previous = atomicOr(seen_source + word, mask);
        if ((previous & mask) != 0) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
            continue;
        }
        atomicAdd(observed_selected, 1);
    }
}

__global__ void canonical_selection_count_kernel(
    int *__restrict__ failure_state,
    const int *__restrict__ reported_selected,
    const int *__restrict__ observed_selected,
    int64_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || failure_state[0] != 0) {
        return;
    }
    const int reported = reported_selected[0];
    if (reported < 0 || static_cast<int64_t>(reported) > capacity ||
        reported != observed_selected[0]) {
        atomicOr(failure_state, channel::capacity::kPairContractError);
    }
}

__global__ void canonical_selection_pair_kernel(
    int *__restrict__ failure_state,
    const int64_t *__restrict__ selected_row_index,
    const bool *__restrict__ selected_valid,
    const bool *__restrict__ candidate_valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    int *__restrict__ observed_num_paths,
    int64_t capacity,
    int64_t num_tx,
    int64_t num_rx) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < capacity;
         row += stride) {
        if (!selected_valid[row]) {
            continue;
        }
        const int64_t source = selected_row_index[row];
        if (source < 0 || source >= capacity || !candidate_valid[source]) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
            continue;
        }
        const int tx = tx_id[source];
        const int rx = rx_id[source];
        if (tx < 0 || static_cast<int64_t>(tx) >= num_tx ||
            rx < 0 || static_cast<int64_t>(rx) >= num_rx) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
            continue;
        }
        const int64_t pair = static_cast<int64_t>(rx) * num_tx + tx;
        atomicAdd(observed_num_paths + pair, 1);
    }
}

__global__ void canonical_selection_pair_count_kernel(
    int *__restrict__ failure_state,
    const int *__restrict__ reported_num_paths,
    const int *__restrict__ observed_num_paths,
    int64_t pair_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        if (reported_num_paths[pair] < 0 ||
            reported_num_paths[pair] != observed_num_paths[pair]) {
            atomicOr(failure_state, channel::capacity::kPairContractError);
        }
    }
}

__global__ void canonical_gather_publish_kernel(
    const int *__restrict__ failure_state,
    const int64_t *__restrict__ selected_row_index,
    const bool *__restrict__ selected_valid,
    const int *__restrict__ num_selected,
    const int *__restrict__ num_paths,
    GatherInput input,
    GatherOutput output,
    int64_t row_capacity,
    int64_t pair_count,
    int64_t sequence_width) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t count = row_capacity > pair_count ? row_capacity : pair_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < count;
         destination += stride) {
        if (destination < pair_count) {
            output.num_paths[destination] = num_paths[destination];
        }
        if (destination >= row_capacity || !selected_valid[destination]) {
            continue;
        }
        const int64_t source = selected_row_index[destination];
        if (source < 0 || source >= row_capacity || !input.valid[source]) {
            continue;
        }
        output.selected_row_index[destination] = source;
        output.valid[destination] = true;
        channel::evaluated_paths::copy_row(
            input.payload, output.payload, source, destination, sequence_width);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output.num_selected[0] = num_selected[0];
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kGatherBlockSize - 1) / kGatherBlockSize);
}

}  // namespace

pybind11::dict channel_evaluated_paths_canonical_capacity_gather(
    at::Tensor failure_state,
    at::Tensor selected_row_index,
    at::Tensor selected_valid,
    at::Tensor num_selected,
    at::Tensor num_paths,
    at::Tensor candidate_valid,
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
    int64_t num_tx,
    int64_t num_rx) {
    check_tensor(candidate_valid, "candidate_valid", at::kBool, 1);
    const int64_t capacity = candidate_valid.size(0);
    const int device = candidate_valid.get_device();
    check_tensor(selected_row_index, "selected_row_index", at::kLong, 1);
    check_tensor(selected_valid, "selected_valid", at::kBool, 1);
    check_tensor(num_selected, "num_selected", at::kInt, 1);
    check_tensor(num_paths, "num_paths", at::kInt, 1);
    TORCH_CHECK(selected_row_index.size(0) == capacity, "selected_row_index must match capacity");
    TORCH_CHECK(selected_valid.size(0) == capacity, "selected_valid must match capacity");
    TORCH_CHECK(num_selected.size(0) == 1, "num_selected must have shape (1,)");
    for (const auto& item : {selected_row_index, selected_valid, num_selected, num_paths}) {
        TORCH_CHECK(item.get_device() == device, "selection tensors must share candidate device");
    }
    channel::capacity::validate_failure_state(failure_state, candidate_valid);
    TORCH_CHECK(num_tx >= 0 && num_rx >= 0, "endpoint counts must be non-negative");
    TORCH_CHECK(
        num_tx == 0 || num_rx <= std::numeric_limits<int64_t>::max() / num_tx,
        "endpoint pair count overflows int64 indexing");
    const int64_t pair_count = num_tx * num_rx;
    TORCH_CHECK(num_paths.size(0) == pair_count, "num_paths must match endpoint pairs");
    TORCH_CHECK(capacity <= std::numeric_limits<int>::max(), "candidate capacity exceeds int32 status capacity");
    TORCH_CHECK(pair_count <= std::numeric_limits<int>::max(), "pair count exceeds int32 status capacity");

    const channel::evaluated_paths::PayloadTensors payload{
        tx_id, rx_id, depth, component_id, primitive_id, edge_id, material_id,
        primitive_sequence, material_sequence, interaction_type, path_length_m,
        delay_s, field_direction, interaction_position, interaction_normal,
        interaction_positions, interaction_normals, path_gain, path_field,
        field_xyz, coefficient};
    const int64_t sequence_width =
        channel::evaluated_paths::validate_payload(payload, capacity, device);

    auto bool_options = candidate_valid.options().dtype(at::kBool);
    auto int_options = candidate_valid.options().dtype(at::kInt);
    auto long_options = candidate_valid.options().dtype(at::kLong);
    auto out_selected_row_index = at::empty({capacity}, long_options);
    auto out_valid = at::empty({capacity}, bool_options);
    auto out_num_selected = at::empty({1}, int_options);
    auto out_num_paths = at::empty({pair_count}, int_options);
    auto out = channel::evaluated_paths::allocate_payload(
        candidate_valid, capacity, sequence_width);
    auto observed_selected = at::empty({1}, int_options);
    auto observed_num_paths = at::empty({pair_count}, int_options);
    const int64_t seen_words = (capacity + 31) / 32;
    auto seen_source = at::empty({seen_words}, int_options);

    const GatherInput input{
        candidate_valid.data_ptr<bool>(),
        channel::evaluated_paths::input_view(payload)};
    const GatherOutput output{
        out_selected_row_index.data_ptr<int64_t>(), out_valid.data_ptr<bool>(),
        out_num_selected.data_ptr<int>(), out_num_paths.data_ptr<int>(),
        channel::evaluated_paths::output_view(out)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = std::max<int64_t>(1, std::max(capacity, pair_count));
    canonical_gather_init_kernel<<<launch_blocks(init_count), kGatherBlockSize, 0, stream>>>(
        output, capacity, pair_count, sequence_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemsetAsync(observed_selected.data_ptr<int>(), 0, sizeof(int), stream));
    if (pair_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            observed_num_paths.data_ptr<int>(), 0, pair_count * sizeof(int), stream));
    }
    if (seen_words > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            seen_source.data_ptr<int>(), 0, seen_words * sizeof(int), stream));
    }
    if (capacity > 0) {
        canonical_selection_structure_kernel<<<
            launch_blocks(capacity), kGatherBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), selected_row_index.data_ptr<int64_t>(),
            selected_valid.data_ptr<bool>(), candidate_valid.data_ptr<bool>(),
            reinterpret_cast<unsigned int *>(seen_source.data_ptr<int>()),
            observed_selected.data_ptr<int>(), capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    canonical_selection_count_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int>(), num_selected.data_ptr<int>(),
        observed_selected.data_ptr<int>(), capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (capacity > 0) {
        canonical_selection_pair_kernel<<<
            launch_blocks(capacity), kGatherBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), selected_row_index.data_ptr<int64_t>(),
            selected_valid.data_ptr<bool>(), candidate_valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), observed_num_paths.data_ptr<int>(),
            capacity, num_tx, num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (pair_count > 0) {
        canonical_selection_pair_count_kernel<<<
            launch_blocks(pair_count), kGatherBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), num_paths.data_ptr<int>(),
            observed_num_paths.data_ptr<int>(), pair_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    canonical_gather_publish_kernel<<<
        launch_blocks(init_count), kGatherBlockSize, 0, stream>>>(
        failure_state.data_ptr<int>(), selected_row_index.data_ptr<int64_t>(),
        selected_valid.data_ptr<bool>(), num_selected.data_ptr<int>(),
        num_paths.data_ptr<int>(), input, output, capacity, pair_count, sequence_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pybind11::dict result;
    result["selected_row_index"] = out_selected_row_index;
    result["valid"] = out_valid;
    result["num_selected"] = out_num_selected;
    result["num_paths"] = out_num_paths;
    out.append_to(result);
    return result;
}
