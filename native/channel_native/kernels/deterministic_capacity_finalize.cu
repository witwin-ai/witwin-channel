#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "deterministic_capacity_finalize.h"

#include <cstdint>
#include <limits>

namespace {

constexpr int kCapacityFinalizeBlockSize = 256;
using channel_native::check_tensor;

__global__ void capacity_finalize_pair_state_init_kernel(
    int *__restrict__ pair_count,
    int *__restrict__ pair_start,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < count;
         pair += stride) {
        pair_count[pair] = 0;
        pair_start[pair] = std::numeric_limits<int>::max();
    }
}

__global__ void capacity_finalize_keys_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    int64_t *__restrict__ pair_key,
    int64_t *__restrict__ row_index,
    int *__restrict__ contract_error,
    int64_t candidate_count,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < candidate_count;
         row += stride) {
        row_index[row] = row;
        // Invalid rows are canonical poison lanes. Do not read either endpoint id.
        if (!valid[row]) {
            pair_key[row] = pair_count;
            continue;
        }
        const int tx = tx_id[row];
        const int rx = rx_id[row];
        if (tx < 0 || static_cast<int64_t>(tx) >= num_tx ||
            rx < 0 || static_cast<int64_t>(rx) >= num_rx) {
            pair_key[row] = pair_count;
            atomicExch(contract_error, 1);
            continue;
        }
        pair_key[row] = static_cast<int64_t>(rx) * num_tx + tx;
    }
}

__global__ void capacity_finalize_count_kernel(
    const int64_t *__restrict__ sorted_pair_key,
    int *__restrict__ pair_counts,
    int *__restrict__ pair_start,
    int64_t candidate_count,
    int64_t pair_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < candidate_count;
         position += stride) {
        const int64_t pair = sorted_pair_key[position];
        if (pair >= pair_count) {
            continue;
        }
        atomicAdd(pair_counts + pair, 1);
        atomicMin(pair_start + pair, static_cast<int>(position));
    }
}

__global__ void capacity_finalize_overflow_kernel(
    const int *__restrict__ pair_counts,
    int *__restrict__ overflow,
    int64_t pair_count,
    int64_t path_capacity_per_pair) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        if (static_cast<int64_t>(pair_counts[pair]) > path_capacity_per_pair) {
            atomicExch(overflow, 1);
        }
    }
}

__global__ void capacity_finalize_public_status_kernel(
    const int *__restrict__ overflow_flag,
    bool *__restrict__ overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        overflow[0] = overflow_flag[0] != 0;
    }
}

__global__ void capacity_finalize_materialize_counts_kernel(
    const int *__restrict__ pair_counts,
    const int *__restrict__ overflow,
    const int *__restrict__ contract_error,
    int *__restrict__ num_paths,
    int64_t pair_count) {
    if (overflow[0] || contract_error[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        num_paths[pair] = pair_counts[pair];
    }
}

__global__ void capacity_finalize_materialize_rows_kernel(
    const int64_t *__restrict__ sorted_pair_key,
    const int64_t *__restrict__ sorted_row_index,
    const int *__restrict__ pair_start,
    const int *__restrict__ overflow,
    const int *__restrict__ contract_error,
    int64_t *__restrict__ selected_row_index,
    bool *__restrict__ output_valid,
    int64_t candidate_count,
    int64_t pair_count,
    int64_t path_capacity_per_pair) {
    if (overflow[0] || contract_error[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < candidate_count;
         position += stride) {
        const int64_t pair = sorted_pair_key[position];
        if (pair >= pair_count) {
            continue;
        }
        const int64_t rank = position - pair_start[pair];
        const int64_t destination = pair * path_capacity_per_pair + rank;
        selected_row_index[destination] = sorted_row_index[position];
        output_valid[destination] = true;
    }
}

__global__ void capacity_finalize_trap_kernel(
    const int *__restrict__ overflow,
    const int *__restrict__ contract_error) {
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (overflow[0] || contract_error[0])) {
        asm volatile("trap;");
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>(
        (count + kCapacityFinalizeBlockSize - 1) / kCapacityFinalizeBlockSize);
}

}  // namespace

int64_t channel_native::capacity::deterministic_capacity_validate(
    const at::Tensor& valid,
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_tensor(valid, "valid", at::kBool, 1);
    check_tensor(tx_id, "tx_id", at::kInt, 1);
    check_tensor(rx_id, "rx_id", at::kInt, 1);
    const int64_t candidate_count = valid.size(0);
    TORCH_CHECK(tx_id.size(0) == candidate_count, "tx_id must match valid");
    TORCH_CHECK(rx_id.size(0) == candidate_count, "rx_id must match valid");
    const int device = valid.get_device();
    TORCH_CHECK(tx_id.get_device() == device, "tx_id must share valid device");
    TORCH_CHECK(rx_id.get_device() == device, "rx_id must share valid device");
    TORCH_CHECK(pair_count >= 0, "pair_count must be non-negative");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");
    TORCH_CHECK(
        path_capacity_per_pair >= 0,
        "path_capacity_per_pair must be non-negative");
    TORCH_CHECK(
        num_tx == 0 || num_rx <= std::numeric_limits<int64_t>::max() / num_tx,
        "endpoint pair count overflows int64 indexing");
    TORCH_CHECK(
        pair_count == num_tx * num_rx,
        "pair_count must equal num_tx * num_rx");
    TORCH_CHECK(
        candidate_count <= std::numeric_limits<int>::max(),
        "candidate_count exceeds CUB int32 item capacity");
    TORCH_CHECK(
        pair_count <= std::numeric_limits<int>::max(),
        "pair_count exceeds int32 count capacity");
    TORCH_CHECK(
        path_capacity_per_pair <= std::numeric_limits<int>::max(),
        "path_capacity_per_pair exceeds int32 count capacity");
    TORCH_CHECK(
        path_capacity_per_pair == 0 ||
            pair_count <= std::numeric_limits<int64_t>::max() / path_capacity_per_pair,
        "result row capacity overflows int64 indexing");
    return pair_count * path_capacity_per_pair;
}

channel_native::capacity::FinalizeState
channel_native::capacity::deterministic_capacity_finalize_no_trap(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair,
    at::Tensor selected_row_index,
    at::Tensor output_valid,
    at::Tensor num_paths,
    bool initialize_public_outputs) {
    const int64_t candidate_count = valid.size(0);
    const int device = valid.get_device();
    const int64_t row_capacity = pair_count * path_capacity_per_pair;

    auto int_options = valid.options().dtype(at::kInt);
    auto long_options = valid.options().dtype(at::kLong);
    auto byte_options = valid.options().dtype(at::kByte);
    check_tensor(selected_row_index, "selected_row_index", at::kLong, 1);
    check_tensor(output_valid, "output_valid", at::kBool, 1);
    check_tensor(num_paths, "num_paths", at::kInt, 1);
    TORCH_CHECK(
        selected_row_index.size(0) == row_capacity,
        "selected_row_index has the wrong capacity");
    TORCH_CHECK(
        output_valid.size(0) == row_capacity,
        "output_valid has the wrong capacity");
    TORCH_CHECK(num_paths.size(0) == pair_count, "num_paths has the wrong shape");
    TORCH_CHECK(
        selected_row_index.get_device() == device &&
            output_valid.get_device() == device && num_paths.get_device() == device,
        "capacity outputs must share candidate device");
    auto overflow_flag = at::empty({1}, int_options);
    auto contract_error = at::empty({1}, int_options);
    auto pair_counts = at::empty({pair_count}, int_options);
    auto pair_start = at::empty({pair_count}, int_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (initialize_public_outputs && row_capacity > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            selected_row_index.data_ptr<int64_t>(),
            0xff,
            static_cast<size_t>(row_capacity) * sizeof(int64_t),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            output_valid.data_ptr<bool>(),
            0,
            static_cast<size_t>(row_capacity) * sizeof(bool),
            stream));
    }
    if (initialize_public_outputs && pair_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            num_paths.data_ptr<int>(),
            0,
            static_cast<size_t>(pair_count) * sizeof(int),
            stream));
    }
    C10_CUDA_CHECK(cudaMemsetAsync(
        overflow_flag.data_ptr<int>(), 0, sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        contract_error.data_ptr<int>(), 0, sizeof(int), stream));
    if (pair_count > 0) {
        capacity_finalize_pair_state_init_kernel<<<
            launch_blocks(pair_count), kCapacityFinalizeBlockSize, 0, stream>>>(
            pair_counts.data_ptr<int>(),
            pair_start.data_ptr<int>(),
            pair_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (candidate_count > 0) {
        auto pair_key = at::empty({candidate_count}, long_options);
        auto row_index = at::empty({candidate_count}, long_options);
        auto sorted_pair_key = at::empty({candidate_count}, long_options);
        auto sorted_row_index = at::empty({candidate_count}, long_options);
        capacity_finalize_keys_kernel<<<
            launch_blocks(candidate_count), kCapacityFinalizeBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            pair_key.data_ptr<int64_t>(),
            row_index.data_ptr<int64_t>(),
            contract_error.data_ptr<int>(),
            candidate_count,
            pair_count,
            num_tx,
            num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t scratch_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            scratch_bytes,
            pair_key.data_ptr<int64_t>(),
            sorted_pair_key.data_ptr<int64_t>(),
            row_index.data_ptr<int64_t>(),
            sorted_row_index.data_ptr<int64_t>(),
            static_cast<int>(candidate_count),
            0,
            sizeof(int64_t) * 8,
            stream));
        auto scratch = at::empty(
            {static_cast<int64_t>(scratch_bytes)}, byte_options);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            scratch.data_ptr<uint8_t>(),
            scratch_bytes,
            pair_key.data_ptr<int64_t>(),
            sorted_pair_key.data_ptr<int64_t>(),
            row_index.data_ptr<int64_t>(),
            sorted_row_index.data_ptr<int64_t>(),
            static_cast<int>(candidate_count),
            0,
            sizeof(int64_t) * 8,
            stream));

        capacity_finalize_count_kernel<<<
            launch_blocks(candidate_count), kCapacityFinalizeBlockSize, 0, stream>>>(
            sorted_pair_key.data_ptr<int64_t>(),
            pair_counts.data_ptr<int>(),
            pair_start.data_ptr<int>(),
            candidate_count,
            pair_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (pair_count > 0) {
            capacity_finalize_overflow_kernel<<<
                launch_blocks(pair_count), kCapacityFinalizeBlockSize, 0, stream>>>(
                pair_counts.data_ptr<int>(),
                overflow_flag.data_ptr<int>(),
                pair_count,
                path_capacity_per_pair);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            capacity_finalize_materialize_counts_kernel<<<
                launch_blocks(pair_count), kCapacityFinalizeBlockSize, 0, stream>>>(
                pair_counts.data_ptr<int>(),
                overflow_flag.data_ptr<int>(),
                contract_error.data_ptr<int>(),
                num_paths.data_ptr<int>(),
                pair_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (row_capacity > 0) {
            capacity_finalize_materialize_rows_kernel<<<
                launch_blocks(candidate_count), kCapacityFinalizeBlockSize, 0, stream>>>(
                sorted_pair_key.data_ptr<int64_t>(),
                sorted_row_index.data_ptr<int64_t>(),
                pair_start.data_ptr<int>(),
                overflow_flag.data_ptr<int>(),
                contract_error.data_ptr<int>(),
                selected_row_index.data_ptr<int64_t>(),
                output_valid.data_ptr<bool>(),
                candidate_count,
                pair_count,
                path_capacity_per_pair);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    return {
        selected_row_index,
        output_valid,
        num_paths,
        overflow_flag,
        contract_error};
}

void channel_native::capacity::deterministic_capacity_publish_status(
    const FinalizeState& state,
    at::Tensor overflow,
    cudaStream_t stream) {
    check_tensor(overflow, "overflow", at::kBool, 1);
    TORCH_CHECK(overflow.size(0) == 1, "overflow must have shape (1,)");
    TORCH_CHECK(
        overflow.get_device() == state.valid.get_device(),
        "overflow must share capacity device");
    capacity_finalize_public_status_kernel<<<1, 1, 0, stream>>>(
        state.overflow_flag.data_ptr<int>(), overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void channel_native::capacity::deterministic_capacity_trap(
    const FinalizeState& state,
    cudaStream_t stream) {
    capacity_finalize_trap_kernel<<<1, 1, 0, stream>>>(
        state.overflow_flag.data_ptr<int>(), state.contract_error.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict cn_deterministic_capacity_finalize(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    const int64_t row_capacity =
        channel_native::capacity::deterministic_capacity_validate(
            valid,
            tx_id,
            rx_id,
            pair_count,
            num_tx,
            num_rx,
            path_capacity_per_pair);
    auto selected_row_index = at::empty({row_capacity}, valid.options().dtype(at::kLong));
    auto output_valid = at::empty({row_capacity}, valid.options().dtype(at::kBool));
    auto num_paths = at::empty({pair_count}, valid.options().dtype(at::kInt));
    auto overflow = at::empty({1}, valid.options().dtype(at::kBool));
    auto state = channel_native::capacity::deterministic_capacity_finalize_no_trap(
        valid,
        tx_id,
        rx_id,
        pair_count,
        num_tx,
        num_rx,
        path_capacity_per_pair,
        selected_row_index,
        output_valid,
        num_paths,
        true);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    channel_native::capacity::deterministic_capacity_publish_status(
        state, overflow, stream);
    channel_native::capacity::deterministic_capacity_trap(state, stream);

    pybind11::dict result;
    result["selected_row_index"] = selected_row_index;
    result["valid"] = output_valid;
    result["num_paths"] = num_paths;
    result["overflow"] = overflow;
    return result;
}
