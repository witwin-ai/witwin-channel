// Copyright Xingyu Chen.
// Implements evaluated paths CUDA operations.

// ==== Section: Canonical path selection ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"
#include "capacity.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#define launch_blocks canonical_select_launch_blocks

namespace {

constexpr int kBlockSize = 256;
constexpr int kGlobalScope = 0;
constexpr int kPerPairScope = 1;
using channel::check_tensor;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void validate_rows_kernel(
    int* __restrict__ failure_state,
    const bool* __restrict__ valid,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    int* __restrict__ valid_count,
    int64_t count,
    int64_t num_tx,
    int64_t num_rx) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count;
         row += stride) {
        if (!valid[row]) {
            continue;
        }
        atomicAdd(valid_count, 1);
        const int tx = tx_id[row];
        const int rx = rx_id[row];
        if (tx < 0 || static_cast<int64_t>(tx) >= num_tx ||
            rx < 0 || static_cast<int64_t>(rx) >= num_rx) {
            atomicOr(
                failure_state,
                channel::capacity::kPairContractError);
        }
    }
}

__global__ void init_order_kernel(int64_t count, int64_t* __restrict__ order) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count;
         row += stride) {
        order[row] = row;
    }
}

__global__ void sort_key_1d_kernel(
    const int* __restrict__ failure_state,
    const bool* __restrict__ valid,
    const int* __restrict__ values,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count;
         position += stride) {
        const int64_t row = order[position];
        keys[position] = failure_state[0] == 0 && valid[row]
            ? static_cast<int64_t>(values[row])
            : 0;
    }
}

__global__ void sort_key_sequence_kernel(
    const int* __restrict__ failure_state,
    const bool* __restrict__ valid,
    const int* __restrict__ sequence,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys,
    int64_t count,
    int64_t width,
    int64_t column) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count;
         position += stride) {
        const int64_t row = order[position];
        keys[position] = failure_state[0] == 0 && valid[row]
            ? static_cast<int64_t>(sequence[row * width + column])
            : 0;
    }
}

__global__ void sort_key_valid_kernel(
    const int* __restrict__ failure_state,
    const bool* __restrict__ valid,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count;
         position += stride) {
        keys[position] = failure_state[0] == 0 && valid[order[position]] ? 0 : 1;
    }
}

__device__ int event_type(int component, int depth, int64_t slot, int64_t width) {
    if (component == 1 && slot < depth) {
        return 1;
    }
    if (component == 2 && depth > 0 && slot == 0) {
        return 2;
    }
    if (component == 5 && slot < depth) {
        return 4;
    }
    if (component == 6 && depth > 0 && slot == 0) {
        return 8;
    }
    if (width >= 2 && depth >= 2 && slot < 2) {
        if (component == 3) {
            return slot == 0 ? 1 : 2;
        }
        if (component == 4) {
            return slot == 0 ? 2 : 1;
        }
        if (component == 7) {
            return 2;
        }
    }
    return 0;
}

__device__ int object_id(
    int type,
    int component,
    int edge,
    const int* __restrict__ sequence,
    int64_t row,
    int64_t slot,
    int64_t width) {
    if (type == 0) {
        return -1;
    }
    if (component == 2 && slot == 0) {
        return edge;
    }
    return sequence[row * width + slot];
}

__device__ bool same_identity(
    int64_t left,
    int64_t right,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int* __restrict__ edge_id,
    const int* __restrict__ sequence,
    int64_t width) {
    if (tx_id[left] != tx_id[right] || rx_id[left] != rx_id[right]) {
        return false;
    }
    const int left_component = component_id[left];
    const int right_component = component_id[right];
    const int left_depth = depth[left];
    const int right_depth = depth[right];
    for (int64_t slot = 0; slot < width; ++slot) {
        const int left_type = event_type(left_component, left_depth, slot, width);
        const int right_type = event_type(right_component, right_depth, slot, width);
        if (left_type != right_type) {
            return false;
        }
        if (object_id(
                left_type,
                left_component,
                edge_id[left],
                sequence,
                left,
                slot,
                width) !=
            object_id(
                right_type,
                right_component,
                edge_id[right],
                sequence,
                right,
                slot,
                width)) {
            return false;
        }
    }
    return true;
}

__global__ void group_start_kernel(
    const int* __restrict__ failure_state,
    const bool* __restrict__ valid,
    const int64_t* __restrict__ order,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int* __restrict__ edge_id,
    const int* __restrict__ sequence,
    bool* __restrict__ group_start,
    int64_t count,
    int64_t width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count;
         position += stride) {
        if (failure_state[0] != 0) {
            group_start[position] = false;
            continue;
        }
        const int64_t row = order[position];
        if (!valid[row]) {
            group_start[position] = false;
            continue;
        }
        group_start[position] = position == 0 ||
            !valid[order[position - 1]] ||
            !same_identity(
                order[position - 1],
                row,
                tx_id,
                rx_id,
                depth,
                component_id,
                edge_id,
                sequence,
                width);
    }
}

__global__ void shortest_winner_kernel(
    const int* __restrict__ failure_state,
    const int* __restrict__ valid_count,
    const bool* __restrict__ valid,
    const int64_t* __restrict__ order,
    const bool* __restrict__ group_start,
    const float* __restrict__ length,
    bool* __restrict__ winner,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         start < count;
         start += stride) {
        winner[start] = false;
        if (failure_state[0] != 0 || !group_start[start]) {
            continue;
        }
        if (valid_count[0] <= 1) {
            winner[start] = true;
            continue;
        }
        float minimum = std::numeric_limits<float>::infinity();
        bool has_nan = false;
        int64_t end = start;
        while (end < count) {
            if (end != start && group_start[end]) {
                break;
            }
            const int64_t row = order[end];
            if (!valid[row]) {
                break;
            }
            const float value = length[row];
            if (isnan(value)) {
                has_nan = true;
                break;
            }
            if (value < minimum) {
                minimum = value;
            }
            ++end;
        }
        if (has_nan) {
            continue;
        }
        for (int64_t position = start; position < end; ++position) {
            if (length[order[position]] == minimum) {
                winner[position] = true;
                break;
            }
        }
    }
}

__global__ void pair_state_init_kernel(
    int* __restrict__ pair_count,
    int* __restrict__ pair_start,
    int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < count;
         pair += stride) {
        pair_count[pair] = 0;
        pair_start[pair] = std::numeric_limits<int>::max();
    }
}

__global__ void dedup_pair_state_kernel(
    const int* __restrict__ failure_state,
    const int64_t* __restrict__ dedup_index,
    const int* __restrict__ dedup_count,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    int* __restrict__ pair_count,
    int* __restrict__ pair_start,
    int64_t capacity,
    int64_t num_tx) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < capacity;
         position += stride) {
        if (position >= dedup_count[0]) {
            continue;
        }
        const int64_t row = dedup_index[position];
        const int64_t pair = static_cast<int64_t>(rx_id[row]) * num_tx + tx_id[row];
        atomicAdd(pair_count + pair, 1);
        atomicMin(pair_start + pair, static_cast<int>(position));
    }
}

__global__ void max_paths_flags_kernel(
    const int* __restrict__ failure_state,
    const int64_t* __restrict__ dedup_index,
    const int* __restrict__ dedup_count,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ pair_start,
    bool* __restrict__ keep,
    int64_t capacity,
    int64_t num_tx,
    int64_t max_paths,
    int scope) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < capacity;
         position += stride) {
        keep[position] = false;
        if (failure_state[0] != 0 || position >= dedup_count[0]) {
            continue;
        }
        if (max_paths < 0) {
            keep[position] = true;
            continue;
        }
        if (scope == kGlobalScope) {
            keep[position] = position < max_paths;
            continue;
        }
        const int64_t row = dedup_index[position];
        const int64_t pair = static_cast<int64_t>(rx_id[row]) * num_tx + tx_id[row];
        keep[position] = position - pair_start[pair] < max_paths;
    }
}

__global__ void final_pair_count_kernel(
    const int* __restrict__ failure_state,
    const int64_t* __restrict__ selected_index,
    const int* __restrict__ selected_count,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    int* __restrict__ pair_count,
    int64_t capacity,
    int64_t num_tx) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < capacity;
         position += stride) {
        if (position >= selected_count[0]) {
            continue;
        }
        const int64_t row = selected_index[position];
        const int64_t pair = static_cast<int64_t>(rx_id[row]) * num_tx + tx_id[row];
        atomicAdd(pair_count + pair, 1);
    }
}

__global__ void publish_kernel(
    const int* __restrict__ failure_state,
    const int64_t* __restrict__ private_index,
    const int* __restrict__ private_count,
    const int* __restrict__ private_pair_count,
    int64_t* __restrict__ selected_index,
    bool* __restrict__ selected_valid,
    int* __restrict__ num_selected,
    int* __restrict__ num_paths,
    int64_t capacity,
    int64_t pair_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < capacity || index < pair_count;
         index += stride) {
        if (index < capacity && index < private_count[0]) {
            selected_index[index] = private_index[index];
            selected_valid[index] = true;
        }
        if (index < pair_count) {
            num_paths[index] = private_pair_count[index];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        num_selected[0] = private_count[0];
    }
}

__global__ void compact_control_record_kernel(
    const int* __restrict__ failure_state,
    const int* __restrict__ selected_count,
    int64_t candidate_count,
    int64_t* __restrict__ control_record) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int count = selected_count[0];
        control_record[0] =
            failure_state[0] == 0 && count >= 0 &&
                    static_cast<int64_t>(count) <= candidate_count
                ? static_cast<int64_t>(count)
                : -1;
    }
}

__global__ void exact_pair_metadata_kernel(
    const int64_t* __restrict__ selected_row_index,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int64_t* __restrict__ source_stable_ids,
    const int64_t* __restrict__ sink_stable_ids,
    int64_t* __restrict__ pair_index,
    int64_t* __restrict__ source_id,
    int64_t* __restrict__ sink_id,
    int64_t* __restrict__ pair_counts,
    int64_t row_count,
    int64_t num_tx,
    int64_t num_rx,
    bool has_stable_ids) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_count;
         destination += stride) {
        const int64_t source =
            selected_row_index == nullptr ? destination
                                          : selected_row_index[destination];
        const int tx = tx_id[source];
        const int rx = rx_id[source];
        if (tx < 0 || static_cast<int64_t>(tx) >= num_tx ||
            rx < 0 || static_cast<int64_t>(rx) >= num_rx) {
            pair_index[destination] = 0;
            if (has_stable_ids) {
                source_id[destination] = 0;
                sink_id[destination] = 0;
            }
            CUDA_KERNEL_ASSERT(
                false && "exact pair metadata endpoint id is out of range");
            continue;
        }
        const int64_t pair = static_cast<int64_t>(rx) * num_tx + tx;
        pair_index[destination] = pair;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(pair_counts + pair),
            static_cast<unsigned long long>(1));
        if (has_stable_ids) {
            source_id[destination] = source_stable_ids[tx];
            sink_id[destination] = sink_stable_ids[rx];
        }
    }
}

void device_select(
    const at::Tensor& input,
    const at::Tensor& flags,
    at::Tensor output,
    at::Tensor count,
    const at::Tensor& scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
    const int item_count = static_cast<int>(input.numel());
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        scratch.data_ptr<uint8_t>(),
        scratch_bytes,
        input.data_ptr<int64_t>(),
        flags.data_ptr<bool>(),
        output.data_ptr<int64_t>(),
        count.data_ptr<int>(),
        item_count,
        stream));
}

}  // namespace

static pybind11::dict canonical_capacity_select_impl(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor primitive_sequence,
    at::Tensor path_length_m,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t max_paths,
    int64_t max_paths_scope) {
    check_tensor(valid, "valid", at::kBool, 1);
    const std::pair<const char*, at::Tensor> index_tensors[] = {
        {"tx_id", tx_id},
        {"rx_id", rx_id},
        {"depth", depth},
        {"component_id", component_id},
        {"primitive_id", primitive_id},
        {"edge_id", edge_id}};
    for (const auto& item : index_tensors) {
        check_tensor(item.second, item.first, at::kInt, 1);
        TORCH_CHECK(item.second.sizes() == valid.sizes(), item.first, " must match valid");
        TORCH_CHECK(item.second.get_device() == valid.get_device(), item.first, " must share valid device");
    }
    check_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    check_tensor(path_length_m, "path_length_m", at::kFloat, 1);
    const int64_t count = valid.size(0);
    TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must match valid rows");
    TORCH_CHECK(path_length_m.size(0) == count, "path_length_m must match valid");
    TORCH_CHECK(
        primitive_sequence.get_device() == valid.get_device() &&
            path_length_m.get_device() == valid.get_device(),
        "selection inputs must share valid device");
    channel::capacity::validate_failure_state(failure_state, valid);
    TORCH_CHECK(pair_count >= 0, "pair_count must be non-negative");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");
    TORCH_CHECK(max_paths == -1 || max_paths > 0, "max_paths must be -1 or positive");
    TORCH_CHECK(
        max_paths_scope == kGlobalScope || max_paths_scope == kPerPairScope,
        "max_paths_scope must be global or per_pair");
    TORCH_CHECK(
        num_tx == 0 || num_rx <= std::numeric_limits<int64_t>::max() / num_tx,
        "endpoint pair count overflows int64 indexing");
    TORCH_CHECK(pair_count == num_tx * num_rx, "pair_count must equal num_tx * num_rx");
    TORCH_CHECK(count <= std::numeric_limits<int>::max(), "candidate capacity exceeds CUB int32 item capacity");
    TORCH_CHECK(pair_count <= std::numeric_limits<int>::max(), "pair_count exceeds int32 capacity");

    const int device = valid.get_device();
    const int64_t width = primitive_sequence.size(1);
    auto int_options = valid.options().dtype(at::kInt);
    auto long_options = valid.options().dtype(at::kLong);
    auto bool_options = valid.options().dtype(at::kBool);
    auto selected_row_index = at::empty({count}, long_options);
    auto selected_valid = at::empty({count}, bool_options);
    auto num_selected = at::empty({1}, int_options);
    auto num_paths = at::empty({pair_count}, int_options);
    auto valid_count = at::empty({1}, int_options);
    auto order = at::empty({count}, long_options);
    auto keys = at::empty({count}, long_options);
    auto sorted_order = at::empty({count}, long_options);
    auto sorted_keys = at::empty({count}, long_options);
    auto group_start = at::empty({count}, bool_options);
    auto winner = at::empty({count}, bool_options);
    auto dedup_index = at::empty({count}, long_options);
    auto dedup_count = at::empty({1}, int_options);
    auto keep = at::empty({count}, bool_options);
    auto private_selected_index = at::empty({count}, long_options);
    auto private_selected_count = at::empty({1}, int_options);
    auto pair_counts = at::empty({pair_count}, int_options);
    auto pair_starts = at::empty({pair_count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    size_t radix_scratch_bytes = 0;
    size_t select_scratch_bytes = 0;
    if (count > 0) {
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            radix_scratch_bytes,
            keys.data_ptr<int64_t>(),
            sorted_keys.data_ptr<int64_t>(),
            order.data_ptr<int64_t>(),
            sorted_order.data_ptr<int64_t>(),
            static_cast<int>(count),
            0,
            sizeof(int64_t) * 8,
            stream));
        C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr,
            select_scratch_bytes,
            order.data_ptr<int64_t>(),
            winner.data_ptr<bool>(),
            dedup_index.data_ptr<int64_t>(),
            dedup_count.data_ptr<int>(),
            static_cast<int>(count),
            stream));
    }
    const size_t cub_scratch_bytes =
        radix_scratch_bytes > select_scratch_bytes
        ? radix_scratch_bytes
        : select_scratch_bytes;
    auto cub_scratch = at::empty(
        {static_cast<int64_t>(cub_scratch_bytes)},
        valid.options().dtype(at::kByte));

    if (count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(selected_row_index.data_ptr<int64_t>(), 0xff, count * sizeof(int64_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(selected_valid.data_ptr<bool>(), 0, count * sizeof(bool), stream));
    }
    C10_CUDA_CHECK(cudaMemsetAsync(num_selected.data_ptr<int>(), 0, sizeof(int), stream));
    if (pair_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(num_paths.data_ptr<int>(), 0, pair_count * sizeof(int), stream));
    }
    C10_CUDA_CHECK(cudaMemsetAsync(valid_count.data_ptr<int>(), 0, sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(dedup_count.data_ptr<int>(), 0, sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(private_selected_count.data_ptr<int>(), 0, sizeof(int), stream));

    if (count > 0) {
        validate_rows_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(), valid_count.data_ptr<int>(), count, num_tx, num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        init_order_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(count, order.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto stable_sort = [&]() {
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                cub_scratch.data_ptr<uint8_t>(),
                radix_scratch_bytes,
                keys.data_ptr<int64_t>(),
                sorted_keys.data_ptr<int64_t>(),
                order.data_ptr<int64_t>(),
                sorted_order.data_ptr<int64_t>(),
                static_cast<int>(count),
                0,
                sizeof(int64_t) * 8,
                stream));
            std::swap(keys, sorted_keys);
            std::swap(order, sorted_order);
        };
        auto sort_1d = [&](const at::Tensor& values) {
            sort_key_1d_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
                failure_state.data_ptr<int>(), valid.data_ptr<bool>(), values.data_ptr<int>(),
                order.data_ptr<int64_t>(), keys.data_ptr<int64_t>(), count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            stable_sort();
        };
        sort_1d(edge_id);
        sort_1d(primitive_id);
        for (int64_t column = width - 1; column >= 0; --column) {
            sort_key_sequence_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
                failure_state.data_ptr<int>(), valid.data_ptr<bool>(), primitive_sequence.data_ptr<int>(),
                order.data_ptr<int64_t>(), keys.data_ptr<int64_t>(), count, width, column);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            stable_sort();
        }
        sort_1d(component_id);
        sort_1d(depth);
        sort_1d(tx_id);
        sort_1d(rx_id);
        sort_key_valid_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), order.data_ptr<int64_t>(),
            keys.data_ptr<int64_t>(), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        stable_sort();
        group_start_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), order.data_ptr<int64_t>(),
            tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), depth.data_ptr<int>(),
            component_id.data_ptr<int>(), edge_id.data_ptr<int>(), primitive_sequence.data_ptr<int>(),
            group_start.data_ptr<bool>(), count, width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        shortest_winner_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid_count.data_ptr<int>(), valid.data_ptr<bool>(),
            order.data_ptr<int64_t>(), group_start.data_ptr<bool>(), path_length_m.data_ptr<float>(),
            winner.data_ptr<bool>(), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        device_select(
            order,
            winner,
            dedup_index,
            dedup_count,
            cub_scratch,
            select_scratch_bytes,
            stream);
    }

    if (pair_count > 0) {
        pair_state_init_kernel<<<launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            pair_counts.data_ptr<int>(), pair_starts.data_ptr<int>(), pair_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (count > 0) {
        dedup_pair_state_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), dedup_index.data_ptr<int64_t>(), dedup_count.data_ptr<int>(),
            tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), pair_counts.data_ptr<int>(),
            pair_starts.data_ptr<int>(), count, num_tx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        max_paths_flags_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), dedup_index.data_ptr<int64_t>(), dedup_count.data_ptr<int>(),
            tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), pair_starts.data_ptr<int>(), keep.data_ptr<bool>(),
            count, num_tx, max_paths, static_cast<int>(max_paths_scope));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        device_select(
            dedup_index,
            keep,
            private_selected_index,
            private_selected_count,
            cub_scratch,
            select_scratch_bytes,
            stream);
    }
    if (pair_count > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(pair_counts.data_ptr<int>(), 0, pair_count * sizeof(int), stream));
    }
    if (count > 0) {
        final_pair_count_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), private_selected_index.data_ptr<int64_t>(),
            private_selected_count.data_ptr<int>(), tx_id.data_ptr<int>(), rx_id.data_ptr<int>(),
            pair_counts.data_ptr<int>(), count, num_tx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t publish_count = count > pair_count ? count : pair_count;
    if (publish_count > 0) {
        publish_kernel<<<launch_blocks(publish_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), private_selected_index.data_ptr<int64_t>(),
            private_selected_count.data_ptr<int>(), pair_counts.data_ptr<int>(),
            selected_row_index.data_ptr<int64_t>(), selected_valid.data_ptr<bool>(),
            num_selected.data_ptr<int>(), num_paths.data_ptr<int>(), count, pair_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["selected_row_index"] = selected_row_index;
    result["valid"] = selected_valid;
    result["num_selected"] = num_selected;
    result["num_paths"] = num_paths;
    return result;
}

pybind11::dict channel_deterministic_gather_topology_block(
    pybind11::dict block,
    at::Tensor order,
    int64_t max_count,
    int64_t sequence_width);

namespace {

at::Tensor require_block_tensor(
    const pybind11::dict& block,
    const char* name) {
    TORCH_CHECK(block.contains(name), "block must contain ", name);
    return pybind11::cast<at::Tensor>(block[name]);
}

void validate_optional_stable_ids(
    const std::optional<at::Tensor>& source_stable_ids,
    const std::optional<at::Tensor>& sink_stable_ids,
    const at::Tensor& reference,
    int64_t num_tx,
    int64_t num_rx) {
    TORCH_CHECK(
        source_stable_ids.has_value() == sink_stable_ids.has_value(),
        "source_stable_ids and sink_stable_ids must be provided together");
    if (!source_stable_ids.has_value()) {
        return;
    }
    check_tensor(*source_stable_ids, "source_stable_ids", at::kLong, 1);
    check_tensor(*sink_stable_ids, "sink_stable_ids", at::kLong, 1);
    TORCH_CHECK(
        source_stable_ids->size(0) == num_tx,
        "source_stable_ids must match num_tx");
    TORCH_CHECK(
        sink_stable_ids->size(0) == num_rx,
        "sink_stable_ids must match num_rx");
    TORCH_CHECK(
        source_stable_ids->get_device() == reference.get_device() &&
            sink_stable_ids->get_device() == reference.get_device(),
        "stable ID tables must share the path device");
}

void build_pair_offsets(
    const at::Tensor& pair_counts,
    at::Tensor pair_offsets,
    cudaStream_t stream) {
    const int64_t pair_count = pair_counts.size(0);
    C10_CUDA_CHECK(cudaMemsetAsync(
        pair_offsets.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    if (pair_count == 0) {
        return;
    }
    size_t scratch_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr,
        scratch_bytes,
        pair_counts.data_ptr<int64_t>(),
        pair_offsets.data_ptr<int64_t>() + 1,
        static_cast<int>(pair_count),
        stream));
    auto scratch = at::empty(
        {static_cast<int64_t>(scratch_bytes)},
        pair_counts.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scratch.data_ptr<uint8_t>(),
        scratch_bytes,
        pair_counts.data_ptr<int64_t>(),
        pair_offsets.data_ptr<int64_t>() + 1,
        static_cast<int>(pair_count),
        stream));
}

pybind11::dict exact_pair_metadata(
    const at::Tensor& selected_row_index,
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    int64_t row_count,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    const std::optional<at::Tensor>& source_stable_ids,
    const std::optional<at::Tensor>& sink_stable_ids,
    cudaStream_t stream) {
    auto long_options = tx_id.options().dtype(at::kLong);
    auto pair_index = at::empty({row_count}, long_options);
    auto pair_offsets = at::empty({pair_count + 1}, long_options);
    auto pair_counts = channel::empty_zero_cuda(
        {pair_count}, long_options, stream);
    const bool has_stable_ids = source_stable_ids.has_value();
    auto source_id = at::empty(
        {has_stable_ids ? row_count : 0}, long_options);
    auto sink_id = at::empty(
        {has_stable_ids ? row_count : 0}, long_options);
    if (row_count > 0) {
        exact_pair_metadata_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            selected_row_index.defined()
                ? selected_row_index.data_ptr<int64_t>()
                : nullptr,
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            has_stable_ids ? source_stable_ids->data_ptr<int64_t>() : nullptr,
            has_stable_ids ? sink_stable_ids->data_ptr<int64_t>() : nullptr,
            pair_index.data_ptr<int64_t>(),
            has_stable_ids ? source_id.data_ptr<int64_t>() : nullptr,
            has_stable_ids ? sink_id.data_ptr<int64_t>() : nullptr,
            pair_counts.data_ptr<int64_t>(),
            row_count,
            num_tx,
            num_rx,
            has_stable_ids);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    build_pair_offsets(pair_counts, pair_offsets, stream);
    pybind11::dict result;
    result["pair_index"] = pair_index;
    result["pair_offsets"] = pair_offsets;
    result["source_id"] = source_id;
    result["sink_id"] = sink_id;
    return result;
}

}  // namespace

pybind11::dict channel_enumerated_canonical_compact(
    pybind11::dict block,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t max_paths,
    int64_t max_paths_scope,
    int64_t sequence_width,
    std::optional<at::Tensor> source_stable_ids,
    std::optional<at::Tensor> sink_stable_ids) {
    auto valid = require_block_tensor(block, "valid");
    auto tx_id = require_block_tensor(block, "tx_id");
    auto rx_id = require_block_tensor(block, "rx_id");
    auto depth = require_block_tensor(block, "depth");
    auto component_id = require_block_tensor(block, "component_id");
    auto primitive_id = require_block_tensor(block, "primitive_id");
    auto edge_id = require_block_tensor(block, "edge_id");
    auto primitive_sequence = require_block_tensor(block, "primitive_sequence");
    auto path_length_m = require_block_tensor(block, "path_length_m");
    validate_optional_stable_ids(
        source_stable_ids, sink_stable_ids, valid, num_tx, num_rx);

    const int device = valid.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto failure_state = channel::empty_zero_cuda(
        {1}, valid.options().dtype(at::kInt), stream);
    pybind11::dict capacity = canonical_capacity_select_impl(
        failure_state,
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        primitive_sequence,
        path_length_m,
        pair_count,
        num_tx,
        num_rx,
        max_paths,
        max_paths_scope);
    auto capacity_index =
        pybind11::cast<at::Tensor>(capacity["selected_row_index"]);
    auto selected_count = pybind11::cast<at::Tensor>(capacity["num_selected"]);
    const int64_t candidate_count = valid.size(0);
    int64_t path_count = 0;
    if (candidate_count > 0) {
        auto control_record = at::empty(
            {1}, valid.options().dtype(at::kLong));
        compact_control_record_kernel<<<1, 1, 0, stream>>>(
            failure_state.data_ptr<int>(),
            selected_count.data_ptr<int>(),
            candidate_count,
            control_record.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &path_count,
            control_record.data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK(
            path_count >= 0,
            "canonical compact selection failed its native contract");
    }

    auto exact_index = at::empty(
        {path_count}, valid.options().dtype(at::kLong));
    if (path_count > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            exact_index.data_ptr<int64_t>(),
            capacity_index.data_ptr<int64_t>(),
            static_cast<size_t>(path_count) * sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            stream));
    }
    pybind11::dict gathered = channel_deterministic_gather_topology_block(
        block, exact_index, -1, sequence_width);
    pybind11::dict metadata = exact_pair_metadata(
        exact_index,
        tx_id,
        rx_id,
        path_count,
        pair_count,
        num_tx,
        num_rx,
        source_stable_ids,
        sink_stable_ids,
        stream);
    gathered["selected_row_index"] = exact_index;
    gathered["pair_index"] = metadata["pair_index"];
    gathered["pair_offsets"] = metadata["pair_offsets"];
    gathered["source_id"] = metadata["source_id"];
    gathered["sink_id"] = metadata["sink_id"];
    gathered["path_count"] = path_count;
    gathered["count_d2h_copies"] =
        candidate_count > 0 ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    gathered["count_d2h_bytes"] =
        candidate_count > 0 ? static_cast<int64_t>(8) : static_cast<int64_t>(0);
    gathered["count_synchronizations"] =
        candidate_count > 0 ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    return gathered;
}

pybind11::dict channel_enumerated_exact_pair_metadata(
    at::Tensor tx_id,
    at::Tensor rx_id,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    std::optional<at::Tensor> source_stable_ids,
    std::optional<at::Tensor> sink_stable_ids) {
    check_tensor(tx_id, "tx_id", at::kInt, 1);
    check_tensor(rx_id, "rx_id", at::kInt, 1);
    TORCH_CHECK(rx_id.sizes() == tx_id.sizes(), "rx_id must match tx_id");
    TORCH_CHECK(
        rx_id.get_device() == tx_id.get_device(),
        "rx_id must share tx_id device");
    TORCH_CHECK(num_tx >= 0 && num_rx >= 0, "endpoint counts must be non-negative");
    TORCH_CHECK(
        num_tx == 0 || num_rx <= std::numeric_limits<int64_t>::max() / num_tx,
        "endpoint pair count overflows int64 indexing");
    TORCH_CHECK(pair_count == num_tx * num_rx, "pair_count must match endpoints");
    validate_optional_stable_ids(
        source_stable_ids, sink_stable_ids, tx_id, num_tx, num_rx);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
    at::Tensor no_selection;
    pybind11::dict result = exact_pair_metadata(
        no_selection,
        tx_id,
        rx_id,
        tx_id.size(0),
        pair_count,
        num_tx,
        num_rx,
        source_stable_ids,
        sink_stable_ids,
        stream);
    result["path_count"] = tx_id.size(0);
    result["count_d2h_copies"] = static_cast<int64_t>(0);
    result["count_d2h_bytes"] = static_cast<int64_t>(0);
    result["count_synchronizations"] = static_cast<int64_t>(0);
    return result;
}

#undef launch_blocks

// ==== Section: Transmission topology packing ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// Windows RPC headers imported by the earlier CUB section define small as
// char; clear that legacy macro before Torch first reaches this header.
#ifdef small
#undef small
#endif
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"
#include "capacity.h"

#include <cstdint>
#include <limits>
#include <optional>

#define launch_blocks transmission_pack_launch_blocks
#define kBlockSize kTransmissionPackBlockSize

namespace {

constexpr int kBlockSize = 256;
constexpr float kLightSpeedMetersPerSecond = 299792458.0f;

__device__ __forceinline__ float canonical_quiet_nan() {
    return __uint_as_float(0x7fc00000u);
}

struct TopologyOutput {
    bool *valid;
    int *tx_id;
    int *rx_id;
    int *depth;
    int *component_id;
    int *primitive_id;
    int *edge_id;
    float *path_length_m;
    float *delay_s;
    float *path_gain;
    c10::complex<float> *path_field;
    float *interaction_position;
    float *interaction_normal;
    int *material_id;
    int *primitive_sequence;
    int *material_sequence;
    float *interaction_positions;
    float *interaction_normals;
};

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void transmission_topology_init_kernel(
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        output.valid[row] = false;
        output.tx_id[row] = -1;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.component_id[row] = -1;
        output.primitive_id[row] = -1;
        output.edge_id[row] = -1;
        output.path_length_m[row] = -1.0f;
        output.delay_s[row] = -1.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = c10::complex<float>(0.0f, 0.0f);
        output.material_id[row] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            output.interaction_position[row * 3 + axis] = 0.0f;
            output.interaction_normal[row * 3 + axis] = 0.0f;
        }
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            output.primitive_sequence[sequence] = -1;
            output.material_sequence[sequence] = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = 0.0f;
                output.interaction_normals[vector] = 0.0f;
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        candidate_count[0] = 0;
        guardrail_count[0] = 0;
    }
}

__global__ void transmission_topology_preflight_kernel(
    int *failure_state,
    const bool *valid,
    const int *num_hits,
    const bool *reached_target,
    const bool *overflow,
    const int *global_primitive_id,
    int64_t pair_count,
    int64_t hit_capacity,
    int64_t face_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        if (overflow[row]) {
            atomicOr(
                failure_state,
                channel::capacity::kSegmentPenetrationFailure);
            continue;
        }
        const int hits = num_hits[row];
        bool contract_ok =
            hits >= 0 && static_cast<int64_t>(hits) <= hit_capacity &&
            (reached_target[row] || hits == 0);
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const bool expected_valid = slot < hits;
            const int64_t index = row * hit_capacity + slot;
            if (valid[index] != expected_valid) {
                contract_ok = false;
            }
        }
        if (!contract_ok) {
            atomicOr(
                failure_state,
                channel::capacity::kSegmentPenetrationFailure);
            continue;
        }
        for (int slot = 0; slot < hits; ++slot) {
            const int primitive = global_primitive_id[row * hit_capacity + slot];
            if (primitive < 0 || static_cast<int64_t>(primitive) >= face_count) {
                atomicOr(
                    failure_state,
                    channel::capacity::kSegmentPenetrationFailure);
                break;
            }
        }
    }
}

__global__ void transmission_topology_pack_kernel(
    const int *failure_state,
    const bool *reached_target,
    const int *num_hits,
    const float *distance,
    const float *position,
    const float *normal,
    const int *global_primitive_id,
    const int *face_material_id,
    const int *geometry_mode_id,
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity,
    int64_t material_count,
    int64_t rx_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        const int hits = num_hits[row];
        const bool penetrated = reached_target[row] && hits >= 1;
        if (!penetrated) {
            continue;
        }
        atomicAdd(candidate_count, 1);
        bool bad_material = false;
        for (int slot = 0; slot < hits; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            const int primitive = global_primitive_id[sequence];
            const int material = face_material_id[primitive];
            if (material < 0 || static_cast<int64_t>(material) >= material_count ||
                geometry_mode_id[material] != 0) {
                bad_material = true;
            }
        }
        if (bad_material) {
            atomicAdd(guardrail_count, 1);
            continue;
        }
        output.valid[row] = true;
        output.tx_id[row] = static_cast<int>(row / rx_count);
        output.rx_id[row] = static_cast<int>(row % rx_count);
        output.depth[row] = hits;
        output.component_id[row] = 5;
        output.edge_id[row] = -1;
        output.path_length_m[row] = distance[row];
        output.delay_s[row] = distance[row] / kLightSpeedMetersPerSecond;
        const float nan = canonical_quiet_nan();
        output.path_gain[row] = nan;
        output.path_field[row] = c10::complex<float>(nan, nan);
        for (int64_t slot = 0; slot < hits; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            const int primitive = global_primitive_id[sequence];
            const int material = face_material_id[primitive];
            output.primitive_sequence[sequence] = primitive;
            output.material_sequence[sequence] = material;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = position[vector];
                output.interaction_normals[vector] = normal[vector];
                if (slot == 0) {
                    output.interaction_position[row * 3 + axis] = position[vector];
                    output.interaction_normal[row * 3 + axis] = normal[vector];
                }
            }
            if (slot == 0) {
                output.primitive_id[row] = primitive;
                output.material_id[row] = material;
            }
        }
    }
}

__global__ void transmission_topology_sanitize_kernel(
    const int *failure_state,
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity) {
    if (failure_state[0] == 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        output.valid[row] = false;
        output.tx_id[row] = -1;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.component_id[row] = -1;
        output.primitive_id[row] = -1;
        output.edge_id[row] = -1;
        output.path_length_m[row] = -1.0f;
        output.delay_s[row] = -1.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = c10::complex<float>(0.0f, 0.0f);
        output.material_id[row] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            output.interaction_position[row * 3 + axis] = 0.0f;
            output.interaction_normal[row * 3 + axis] = 0.0f;
        }
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            output.primitive_sequence[sequence] = -1;
            output.material_sequence[sequence] = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = 0.0f;
                output.interaction_normals[vector] = 0.0f;
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        candidate_count[0] = 0;
        guardrail_count[0] = 0;
    }
}

void check_rows(
    const at::Tensor& tensor,
    const char *name,
    int64_t pair_count,
    int device) {
    TORCH_CHECK(tensor.size(0) == pair_count, name, " must match segment rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share segment device");
}

void check_optional_float_tensor(
    const std::optional<at::Tensor>& tensor,
    const char *name,
    at::IntArrayRef shape,
    int device) {
    if (!tensor.has_value()) {
        return;
    }
    channel::check_tensor(*tensor, name, at::kFloat, shape.size());
    TORCH_CHECK(tensor->sizes() == shape, name, " has an unexpected shape");
    TORCH_CHECK(tensor->get_device() == device, name, " must share the topology device");
}

__global__ void transmission_topology_pack_backward_kernel(
    const bool *topology_valid,
    const bool *hit_valid,
    const float *grad_path_length_m,
    const float *grad_delay_s,
    const float *grad_interaction_position,
    const float *grad_interaction_normal,
    const float *grad_interaction_positions,
    const float *grad_interaction_normals,
    float *grad_distance,
    float *grad_position,
    float *grad_normal,
    int64_t pair_count,
    int64_t hit_capacity) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= pair_count) {
        return;
    }
    grad_distance[row] = 0.0f;
    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t vector = (row * hit_capacity + slot) * 3 + axis;
            grad_position[vector] = 0.0f;
            grad_normal[vector] = 0.0f;
        }
    }
    if (!topology_valid[row]) {
        return;
    }
    float distance_grad = 0.0f;
    if (grad_path_length_m != nullptr) {
        distance_grad += grad_path_length_m[row];
    }
    if (grad_delay_s != nullptr) {
        distance_grad += grad_delay_s[row] / kLightSpeedMetersPerSecond;
    }
    grad_distance[row] = distance_grad;
    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        const int64_t sequence = row * hit_capacity + slot;
        if (!hit_valid[sequence]) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t vector = sequence * 3 + axis;
            float position_grad = 0.0f;
            float normal_grad = 0.0f;
            if (grad_interaction_positions != nullptr) {
                position_grad += grad_interaction_positions[vector];
            }
            if (grad_interaction_normals != nullptr) {
                normal_grad += grad_interaction_normals[vector];
            }
            if (slot == 0) {
                if (grad_interaction_position != nullptr) {
                    position_grad += grad_interaction_position[row * 3 + axis];
                }
                if (grad_interaction_normal != nullptr) {
                    normal_grad += grad_interaction_normal[row * 3 + axis];
                }
            }
            grad_position[vector] = position_grad;
            grad_normal[vector] = normal_grad;
        }
    }
}

__global__ void transmission_topology_pack_jvp_kernel(
    const bool *topology_valid,
    const bool *hit_valid,
    const float *tangent_distance,
    const float *tangent_position,
    const float *tangent_normal,
    float *tangent_path_length_m,
    float *tangent_delay_s,
    float *tangent_interaction_position,
    float *tangent_interaction_normal,
    float *tangent_interaction_positions,
    float *tangent_interaction_normals,
    int64_t pair_count,
    int64_t hit_capacity) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= pair_count) {
        return;
    }
    tangent_path_length_m[row] = 0.0f;
    tangent_delay_s[row] = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        tangent_interaction_position[row * 3 + axis] = 0.0f;
        tangent_interaction_normal[row * 3 + axis] = 0.0f;
    }
    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t vector = (row * hit_capacity + slot) * 3 + axis;
            tangent_interaction_positions[vector] = 0.0f;
            tangent_interaction_normals[vector] = 0.0f;
        }
    }
    if (!topology_valid[row]) {
        return;
    }
    if (tangent_distance != nullptr) {
        tangent_path_length_m[row] = tangent_distance[row];
        tangent_delay_s[row] = tangent_distance[row] / kLightSpeedMetersPerSecond;
    }
    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        const int64_t sequence = row * hit_capacity + slot;
        if (!hit_valid[sequence]) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t vector = sequence * 3 + axis;
            if (tangent_position != nullptr) {
                tangent_interaction_positions[vector] = tangent_position[vector];
                if (slot == 0) {
                    tangent_interaction_position[row * 3 + axis] =
                        tangent_position[vector];
                }
            }
            if (tangent_normal != nullptr) {
                tangent_interaction_normals[vector] = tangent_normal[vector];
                if (slot == 0) {
                    tangent_interaction_normal[row * 3 + axis] = tangent_normal[vector];
                }
            }
        }
    }
}

}  // namespace

pybind11::dict channel_enumerated_transmission_topology_pack(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor num_hits,
    at::Tensor reached_target,
    at::Tensor overflow,
    at::Tensor distance,
    at::Tensor position,
    at::Tensor normal,
    at::Tensor global_primitive_id,
    at::Tensor face_material_id,
    at::Tensor geometry_mode_id,
    int64_t tx_count,
    int64_t rx_count) {
    channel::check_tensor(valid, "valid", at::kBool, 2);
    channel::capacity::validate_failure_state(failure_state, valid);
    channel::check_tensor(num_hits, "num_hits", at::kInt, 1);
    channel::check_tensor(reached_target, "reached_target", at::kBool, 1);
    channel::check_tensor(overflow, "overflow", at::kBool, 1);
    channel::check_tensor(distance, "distance", at::kFloat, 1);
    channel::check_tensor(position, "position", at::kFloat, 3);
    channel::check_tensor(normal, "normal", at::kFloat, 3);
    channel::check_tensor(
        global_primitive_id, "global_primitive_id", at::kInt, 2);
    channel::check_tensor(
        face_material_id, "face_material_id", at::kInt, 1);
    channel::check_tensor(
        geometry_mode_id, "geometry_mode_id", at::kInt, 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(
        tx_count == 0 || rx_count <= std::numeric_limits<int64_t>::max() / tx_count,
        "endpoint pair count overflows int64");
    const int64_t pair_count = tx_count * rx_count;
    const int64_t hit_capacity = valid.size(1);
    TORCH_CHECK(valid.size(0) == pair_count, "valid rows must equal tx_count * rx_count");
    TORCH_CHECK(pair_count <= std::numeric_limits<int>::max(), "pair count exceeds int32");
    TORCH_CHECK(hit_capacity <= std::numeric_limits<int>::max(), "hit capacity exceeds int32");
    TORCH_CHECK(
        position.sizes() == at::IntArrayRef({pair_count, hit_capacity, 3}),
        "position must have shape (P, D, 3)");
    TORCH_CHECK(normal.sizes() == position.sizes(), "normal must match position");
    TORCH_CHECK(global_primitive_id.sizes() == valid.sizes(), "global_primitive_id must match valid");
    const int device = valid.get_device();
    check_rows(num_hits, "num_hits", pair_count, device);
    check_rows(reached_target, "reached_target", pair_count, device);
    check_rows(overflow, "overflow", pair_count, device);
    check_rows(distance, "distance", pair_count, device);
    check_rows(position, "position", pair_count, device);
    check_rows(normal, "normal", pair_count, device);
    check_rows(
        global_primitive_id, "global_primitive_id", pair_count, device);
    TORCH_CHECK(face_material_id.get_device() == device, "face_material_id must share segment device");
    TORCH_CHECK(geometry_mode_id.get_device() == device, "geometry_mode_id must share segment device");
    const c10::cuda::CUDAGuard device_guard(valid.device());

    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto float_options = valid.options().dtype(at::kFloat);
    auto complex_options = valid.options().dtype(at::kComplexFloat);
    auto out_valid = at::empty({pair_count}, bool_options);
    auto tx_id = at::empty({pair_count}, int_options);
    auto rx_id = at::empty({pair_count}, int_options);
    auto depth = at::empty({pair_count}, int_options);
    auto component_id = at::empty({pair_count}, int_options);
    auto primitive_id = at::empty({pair_count}, int_options);
    auto edge_id = at::empty({pair_count}, int_options);
    auto path_length_m = at::empty({pair_count}, float_options);
    auto delay_s = at::empty({pair_count}, float_options);
    auto path_gain = at::empty({pair_count}, float_options);
    auto path_field = at::empty({pair_count}, complex_options);
    auto interaction_position = at::empty({pair_count, 3}, float_options);
    auto interaction_normal = at::empty({pair_count, 3}, float_options);
    auto material_id = at::empty({pair_count}, int_options);
    auto primitive_sequence = at::empty({pair_count, hit_capacity}, int_options);
    auto material_sequence = at::empty({pair_count, hit_capacity}, int_options);
    auto interaction_positions = at::empty({pair_count, hit_capacity, 3}, float_options);
    auto interaction_normals = at::empty({pair_count, hit_capacity, 3}, float_options);
    auto candidate_count = at::empty({1}, int_options);
    auto guardrail_count = at::empty({1}, int_options);
    const TopologyOutput output{
        out_valid.data_ptr<bool>(), tx_id.data_ptr<int>(), rx_id.data_ptr<int>(),
        depth.data_ptr<int>(), component_id.data_ptr<int>(), primitive_id.data_ptr<int>(),
        edge_id.data_ptr<int>(), path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
        path_gain.data_ptr<float>(), path_field.data_ptr<c10::complex<float>>(),
        interaction_position.data_ptr<float>(), interaction_normal.data_ptr<float>(),
        material_id.data_ptr<int>(), primitive_sequence.data_ptr<int>(),
        material_sequence.data_ptr<int>(), interaction_positions.data_ptr<float>(),
        interaction_normals.data_ptr<float>()};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = pair_count > 0 ? pair_count : 1;
    transmission_topology_init_kernel<<<
        launch_blocks(init_count), kBlockSize, 0, stream>>>(
        output, candidate_count.data_ptr<int>(), guardrail_count.data_ptr<int>(),
        pair_count, hit_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (pair_count > 0) {
        transmission_topology_preflight_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), overflow.data_ptr<bool>(),
            global_primitive_id.data_ptr<int>(), pair_count, hit_capacity,
            face_material_id.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        transmission_topology_pack_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), reached_target.data_ptr<bool>(),
            num_hits.data_ptr<int>(),
            distance.data_ptr<float>(), position.data_ptr<float>(),
            normal.data_ptr<float>(), global_primitive_id.data_ptr<int>(),
            face_material_id.data_ptr<int>(), geometry_mode_id.data_ptr<int>(), output,
            candidate_count.data_ptr<int>(), guardrail_count.data_ptr<int>(), pair_count,
            hit_capacity, geometry_mode_id.numel(), rx_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        transmission_topology_sanitize_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), output, candidate_count.data_ptr<int>(),
            guardrail_count.data_ptr<int>(), pair_count, hit_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["valid"] = out_valid;
    result["tx_id"] = tx_id;
    result["rx_id"] = rx_id;
    result["depth"] = depth;
    result["component_id"] = component_id;
    result["primitive_id"] = primitive_id;
    result["edge_id"] = edge_id;
    result["path_length_m"] = path_length_m;
    result["delay_s"] = delay_s;
    result["path_gain"] = path_gain;
    result["path_field"] = path_field;
    result["interaction_position"] = interaction_position;
    result["interaction_normal"] = interaction_normal;
    result["material_id"] = material_id;
    result["primitive_sequence"] = primitive_sequence;
    result["material_sequence"] = material_sequence;
    result["interaction_positions"] = interaction_positions;
    result["interaction_normals"] = interaction_normals;
    result["device_candidate_count"] = candidate_count;
    result["device_guardrail_count"] = guardrail_count;
    return result;
}

pybind11::dict channel_enumerated_transmission_topology_pack_backward(
    at::Tensor topology_valid,
    at::Tensor hit_valid,
    std::optional<at::Tensor> grad_path_length_m,
    std::optional<at::Tensor> grad_delay_s,
    std::optional<at::Tensor> grad_interaction_position,
    std::optional<at::Tensor> grad_interaction_normal,
    std::optional<at::Tensor> grad_interaction_positions,
    std::optional<at::Tensor> grad_interaction_normals) {
    channel::check_tensor(topology_valid, "topology_valid", at::kBool, 1);
    channel::check_tensor(hit_valid, "hit_valid", at::kBool, 2);
    const int64_t pair_count = topology_valid.size(0);
    const int64_t hit_capacity = hit_valid.size(1);
    const int device = topology_valid.get_device();
    TORCH_CHECK(hit_valid.size(0) == pair_count, "hit_valid rows must match topology_valid");
    TORCH_CHECK(hit_valid.get_device() == device, "hit_valid must share the topology device");
    check_optional_float_tensor(
        grad_path_length_m, "grad_path_length_m", {pair_count}, device);
    check_optional_float_tensor(grad_delay_s, "grad_delay_s", {pair_count}, device);
    check_optional_float_tensor(
        grad_interaction_position,
        "grad_interaction_position",
        {pair_count, 3},
        device);
    check_optional_float_tensor(
        grad_interaction_normal,
        "grad_interaction_normal",
        {pair_count, 3},
        device);
    check_optional_float_tensor(
        grad_interaction_positions,
        "grad_interaction_positions",
        {pair_count, hit_capacity, 3},
        device);
    check_optional_float_tensor(
        grad_interaction_normals,
        "grad_interaction_normals",
        {pair_count, hit_capacity, 3},
        device);
    const c10::cuda::CUDAGuard device_guard(topology_valid.device());
    auto options = topology_valid.options().dtype(at::kFloat);
    auto grad_distance = at::empty({pair_count}, options);
    auto grad_position = at::empty({pair_count, hit_capacity, 3}, options);
    auto grad_normal = at::empty({pair_count, hit_capacity, 3}, options);
    if (pair_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        transmission_topology_pack_backward_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            topology_valid.data_ptr<bool>(),
            hit_valid.data_ptr<bool>(),
            grad_path_length_m.has_value() ? grad_path_length_m->data_ptr<float>() : nullptr,
            grad_delay_s.has_value() ? grad_delay_s->data_ptr<float>() : nullptr,
            grad_interaction_position.has_value()
                ? grad_interaction_position->data_ptr<float>()
                : nullptr,
            grad_interaction_normal.has_value()
                ? grad_interaction_normal->data_ptr<float>()
                : nullptr,
            grad_interaction_positions.has_value()
                ? grad_interaction_positions->data_ptr<float>()
                : nullptr,
            grad_interaction_normals.has_value()
                ? grad_interaction_normals->data_ptr<float>()
                : nullptr,
            grad_distance.data_ptr<float>(),
            grad_position.data_ptr<float>(),
            grad_normal.data_ptr<float>(),
            pair_count,
            hit_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["grad_distance"] = grad_distance;
    result["grad_position"] = grad_position;
    result["grad_normal"] = grad_normal;
    return result;
}

pybind11::dict channel_enumerated_transmission_topology_pack_jvp(
    at::Tensor topology_valid,
    at::Tensor hit_valid,
    std::optional<at::Tensor> tangent_distance,
    std::optional<at::Tensor> tangent_position,
    std::optional<at::Tensor> tangent_normal) {
    channel::check_tensor(topology_valid, "topology_valid", at::kBool, 1);
    channel::check_tensor(hit_valid, "hit_valid", at::kBool, 2);
    const int64_t pair_count = topology_valid.size(0);
    const int64_t hit_capacity = hit_valid.size(1);
    const int device = topology_valid.get_device();
    TORCH_CHECK(hit_valid.size(0) == pair_count, "hit_valid rows must match topology_valid");
    TORCH_CHECK(hit_valid.get_device() == device, "hit_valid must share the topology device");
    check_optional_float_tensor(
        tangent_distance, "tangent_distance", {pair_count}, device);
    check_optional_float_tensor(
        tangent_position, "tangent_position", {pair_count, hit_capacity, 3}, device);
    check_optional_float_tensor(
        tangent_normal, "tangent_normal", {pair_count, hit_capacity, 3}, device);
    const c10::cuda::CUDAGuard device_guard(topology_valid.device());
    auto options = topology_valid.options().dtype(at::kFloat);
    auto tangent_path_length_m = at::empty({pair_count}, options);
    auto tangent_delay_s = at::empty({pair_count}, options);
    auto tangent_interaction_position = at::empty({pair_count, 3}, options);
    auto tangent_interaction_normal = at::empty({pair_count, 3}, options);
    auto tangent_interaction_positions =
        at::empty({pair_count, hit_capacity, 3}, options);
    auto tangent_interaction_normals =
        at::empty({pair_count, hit_capacity, 3}, options);
    if (pair_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        transmission_topology_pack_jvp_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            topology_valid.data_ptr<bool>(),
            hit_valid.data_ptr<bool>(),
            tangent_distance.has_value() ? tangent_distance->data_ptr<float>() : nullptr,
            tangent_position.has_value() ? tangent_position->data_ptr<float>() : nullptr,
            tangent_normal.has_value() ? tangent_normal->data_ptr<float>() : nullptr,
            tangent_path_length_m.data_ptr<float>(),
            tangent_delay_s.data_ptr<float>(),
            tangent_interaction_position.data_ptr<float>(),
            tangent_interaction_normal.data_ptr<float>(),
            tangent_interaction_positions.data_ptr<float>(),
            tangent_interaction_normals.data_ptr<float>(),
            pair_count,
            hit_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["path_length_m"] = tangent_path_length_m;
    result["delay_s"] = tangent_delay_s;
    result["interaction_position"] = tangent_interaction_position;
    result["interaction_normal"] = tangent_interaction_normal;
    result["interaction_positions"] = tangent_interaction_positions;
    result["interaction_normals"] = tangent_interaction_normals;
    return result;
}

#undef launch_blocks
#undef kBlockSize

// ==== Section: Compact path finalization ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../tensor_checks.h"
#include "path_payload.cuh"
#include "path_compaction.cuh"

#include <cstdint>
#include <limits>
#include <utility>

namespace {

constexpr int kFinalizeBlockSize = 256;
using channel::check_tensor;
using PayloadInput = channel::evaluated_paths::PayloadInputView;
using PayloadOutput = channel::evaluated_paths::PayloadOutputView;

__global__ void compact_contract_flags_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    int *__restrict__ flags,
    int *__restrict__ contract_error,
    int64_t row_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        if (!valid[row]) {
            flags[row] = 0;
            continue;
        }
        flags[row] = 1;
        const int tx = tx_id[row];
        const int rx = rx_id[row];
        if (tx < 0 || static_cast<int64_t>(tx) >= num_tx ||
            rx < 0 || static_cast<int64_t>(rx) >= num_rx) {
            atomicExch(contract_error, 1);
        }
    }
}

__global__ void compact_selection_kernel(
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    int64_t *__restrict__ selected_row_index,
    int64_t *__restrict__ pair_index,
    int64_t row_count,
    int64_t num_tx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t source =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         source < row_count;
         source += stride) {
        if (flags[source] == 0) {
            continue;
        }
        const int64_t destination = offsets[source];
        selected_row_index[destination] = source;
        pair_index[destination] =
            static_cast<int64_t>(rx_id[source]) * num_tx + tx_id[source];
    }
}

__global__ void exact_row_metadata_kernel(
    const bool *__restrict__ valid,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int64_t *__restrict__ source_stable_ids,
    const int64_t *__restrict__ sink_stable_ids,
    int64_t *__restrict__ selected_row_index,
    int64_t *__restrict__ pair_index,
    int64_t *__restrict__ source_id,
    int64_t *__restrict__ sink_id,
    int64_t row_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int tx = tx_id[row];
        const int rx = rx_id[row];
        const bool in_range =
            tx >= 0 && static_cast<int64_t>(tx) < num_tx &&
            rx >= 0 && static_cast<int64_t>(rx) < num_rx;
        if (!valid[row] || !in_range) {
            selected_row_index[row] = row;
            pair_index[row] = 0;
            source_id[row] = 0;
            sink_id[row] = 0;
            CUDA_KERNEL_ASSERT(
                false &&
                "exact evaluated rows violate validity/id contract");
            continue;
        }
        const int64_t pair = static_cast<int64_t>(rx) * num_tx + tx;
        if (row > 0) {
            const int previous_tx = tx_id[row - 1];
            const int previous_rx = rx_id[row - 1];
            const bool previous_in_range =
                previous_tx >= 0 &&
                static_cast<int64_t>(previous_tx) < num_tx &&
                previous_rx >= 0 &&
                static_cast<int64_t>(previous_rx) < num_rx;
            const int64_t previous_pair = previous_in_range
                ? static_cast<int64_t>(previous_rx) * num_tx + previous_tx
                : 0;
            CUDA_KERNEL_ASSERT(
                previous_in_range && previous_pair <= pair &&
                "exact evaluated rows must already be canonical pair-major");
        }
        selected_row_index[row] = row;
        pair_index[row] = pair;
        source_id[row] = source_stable_ids[tx];
        sink_id[row] = sink_stable_ids[rx];
    }
}

__global__ void compact_payload_gather_kernel(
    PayloadInput input,
    PayloadOutput output,
    const int64_t *__restrict__ selected_row_index,
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int64_t *__restrict__ source_stable_ids,
    const int64_t *__restrict__ sink_stable_ids,
    int64_t *__restrict__ source_id,
    int64_t *__restrict__ sink_id,
    int64_t row_count,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_count;
         destination += stride) {
        const int64_t source = selected_row_index[destination];
        channel::evaluated_paths::copy_row(
            input, output, source, destination, sequence_width);
        source_id[destination] = source_stable_ids[tx_id[source]];
        sink_id[destination] = sink_stable_ids[rx_id[source]];
    }
}

__global__ void compact_pair_histogram_kernel(
    const int64_t *__restrict__ pair_index,
    int64_t *__restrict__ pair_offsets,
    int64_t row_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(pair_offsets + pair_index[row] + 1),
            1ULL);
    }
}

int finalize_launch_blocks(int64_t count) {
    return static_cast<int>(
        (count + kFinalizeBlockSize - 1) / kFinalizeBlockSize);
}

void check_stable_lookup(
    const at::Tensor& lookup,
    const char *name,
    int device) {
    check_tensor(lookup, name, at::kLong, 1);
    TORCH_CHECK(lookup.get_device() == device, name, " must share valid device");
}

void stable_radix_sort_pairs(
    at::Tensor& keys,
    at::Tensor& values,
    int64_t count,
    cudaStream_t stream) {
    TORCH_CHECK(count > 0, "radix sort requires non-empty compact rows");
    TORCH_CHECK(
        count <= std::numeric_limits<int>::max(),
        "radix sort compact row count exceeds CUB item range");
    const int item_count = static_cast<int>(count);
    auto sorted_keys = at::empty_like(keys);
    auto sorted_values = at::empty_like(values);
    size_t temporary_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        temporary_bytes,
        keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        values.data_ptr<int64_t>(),
        sorted_values.data_ptr<int64_t>(),
        item_count,
        0,
        static_cast<int>(sizeof(int64_t) * 8),
        stream));
    auto temporary = at::empty(
        {static_cast<int64_t>(temporary_bytes)},
        keys.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temporary.data_ptr(),
        temporary_bytes,
        keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        values.data_ptr<int64_t>(),
        sorted_values.data_ptr<int64_t>(),
        item_count,
        0,
        static_cast<int>(sizeof(int64_t) * 8),
        stream));
    keys = std::move(sorted_keys);
    values = std::move(sorted_values);
}

void append_payload_aliases(
    pybind11::dict& result,
    const channel::evaluated_paths::PayloadTensors& payload) {
    result["tx_id"] = payload.tx_id;
    result["rx_id"] = payload.rx_id;
    result["depth"] = payload.depth;
    result["component_id"] = payload.component_id;
    result["primitive_id"] = payload.primitive_id;
    result["edge_id"] = payload.edge_id;
    result["material_id"] = payload.material_id;
    result["primitive_sequence"] = payload.primitive_sequence;
    result["material_sequence"] = payload.material_sequence;
    result["interaction_type"] = payload.interaction_type;
    result["path_length_m"] = payload.path_length_m;
    result["delay_s"] = payload.delay_s;
    result["field_direction"] = payload.field_direction;
    result["interaction_position"] = payload.interaction_position;
    result["interaction_normal"] = payload.interaction_normal;
    result["interaction_positions"] = payload.interaction_positions;
    result["interaction_normals"] = payload.interaction_normals;
    result["path_gain"] = payload.path_gain;
    result["path_field"] = payload.path_field;
    result["field_xyz"] = payload.field_xyz;
    result["coefficient"] = payload.coefficient;
}

}  // namespace

pybind11::dict channel_evaluated_paths_compact_finalize(
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
    at::Tensor source_stable_ids,
    at::Tensor sink_stable_ids,
    bool rows_are_compact) {
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
    check_stable_lookup(
        source_stable_ids, "source_stable_ids", device);
    check_stable_lookup(
        sink_stable_ids, "sink_stable_ids", device);
    const int64_t num_tx = source_stable_ids.size(0);
    const int64_t num_rx = sink_stable_ids.size(0);
    TORCH_CHECK(
        num_tx == 0 ||
            num_rx <= std::numeric_limits<int64_t>::max() / num_tx,
        "source/sink pair count overflows int64");
    const int64_t pair_count = num_tx * num_rx;
    TORCH_CHECK(
        pair_count < std::numeric_limits<int64_t>::max(),
        "source/sink pair offset count overflows int64");

    auto int_options = valid.options().dtype(at::kInt);
    auto long_options = valid.options().dtype(at::kLong);
    int64_t compact_count = rows_are_compact ? candidate_count : 0;
    at::Tensor selected_row_index;
    at::Tensor pair_index;
    auto source_id = at::empty({compact_count}, long_options);
    auto sink_id = at::empty({compact_count}, long_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (candidate_count == 0) {
        selected_row_index = at::empty({0}, long_options);
        pair_index = at::empty({0}, long_options);
    } else if (rows_are_compact) {
        selected_row_index = at::empty({compact_count}, long_options);
        pair_index = at::empty({compact_count}, long_options);
        exact_row_metadata_kernel<<<
            finalize_launch_blocks(compact_count),
            kFinalizeBlockSize,
            0,
            stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            source_stable_ids.data_ptr<int64_t>(),
            sink_stable_ids.data_ptr<int64_t>(),
            selected_row_index.data_ptr<int64_t>(),
            pair_index.data_ptr<int64_t>(),
            source_id.data_ptr<int64_t>(),
            sink_id.data_ptr<int64_t>(),
            compact_count,
            num_tx,
            num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(
            candidate_count <= std::numeric_limits<int>::max(),
            "compact candidate count exceeds int32 scan range");
        auto flags = at::empty({candidate_count}, int_options);
        auto offsets = at::empty({candidate_count}, int_options);
        auto contract_error = channel::empty_zero_cuda(
            {1}, int_options, stream);
        compact_contract_flags_kernel<<<
            finalize_launch_blocks(candidate_count),
            kFinalizeBlockSize,
            0,
            stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            flags.data_ptr<int>(),
            contract_error.data_ptr<int>(),
            candidate_count,
            num_tx,
            num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(flags.data_ptr<int>()),
            thrust::device_pointer_cast(flags.data_ptr<int>() + candidate_count),
            thrust::device_pointer_cast(offsets.data_ptr<int>()));
        const auto observation = observe_compact_count(
            flags, offsets, candidate_count, stream, &contract_error);
        TORCH_CHECK(
            observation.control_error == 0,
            "valid evaluated path contains an out-of-range source/sink id");
        compact_count = observation.count;
        selected_row_index = at::empty({compact_count}, long_options);
        pair_index = at::empty({compact_count}, long_options);
        source_id = at::empty({compact_count}, long_options);
        sink_id = at::empty({compact_count}, long_options);
        if (compact_count > 0) {
            compact_selection_kernel<<<
                finalize_launch_blocks(candidate_count),
                kFinalizeBlockSize,
                0,
                stream>>>(
                flags.data_ptr<int>(),
                offsets.data_ptr<int>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                selected_row_index.data_ptr<int64_t>(),
                pair_index.data_ptr<int64_t>(),
                candidate_count,
                num_tx);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            stable_radix_sort_pairs(
                pair_index,
                selected_row_index,
                compact_count,
                stream);
        }
    }

    auto pair_offsets = channel::empty_zero_cuda(
        {pair_count + 1}, long_options, stream);
    if (compact_count > 0 && pair_count > 0) {
        compact_pair_histogram_kernel<<<
            finalize_launch_blocks(compact_count),
            kFinalizeBlockSize,
            0,
            stream>>>(
            pair_index.data_ptr<int64_t>(),
            pair_offsets.data_ptr<int64_t>(),
            compact_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()),
        thrust::device_pointer_cast(
            pair_offsets.data_ptr<int64_t>() + pair_count + 1),
        thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()));

    std::optional<channel::evaluated_paths::AllocatedPayload> out;
    if (!rows_are_compact) {
        out.emplace(channel::evaluated_paths::allocate_payload(
            valid, compact_count, sequence_width));
    }
    if (!rows_are_compact && compact_count > 0) {
        compact_payload_gather_kernel<<<
            finalize_launch_blocks(compact_count),
            kFinalizeBlockSize,
            0,
            stream>>>(
            channel::evaluated_paths::input_view(payload),
            channel::evaluated_paths::output_view(*out),
            selected_row_index.data_ptr<int64_t>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            source_stable_ids.data_ptr<int64_t>(),
            sink_stable_ids.data_ptr<int64_t>(),
            source_id.data_ptr<int64_t>(),
            sink_id.data_ptr<int64_t>(),
            compact_count,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["path_count"] = compact_count;
    result["selected_row_index"] = selected_row_index;
    result["pair_index"] = pair_index;
    result["pair_offsets"] = pair_offsets;
    result["source_id"] = source_id;
    result["sink_id"] = sink_id;
    if (rows_are_compact) {
        result["valid"] = valid;
        append_payload_aliases(result, payload);
    } else {
        result["valid"] = at::ones({compact_count}, valid.options());
        out->append_to(result);
    }
    return result;
}

// ==== Section: Capacity-pack AD ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"
#include <ATen/ATen.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include <cstdint>
#include <optional>

namespace channel::evaluated_paths_ad {

using cfloat = c10::complex<float>;

struct ContinuousOutputs {
    float *path_length_m;
    float *delay_s;
    float *field_direction;
    float *interaction_position;
    float *interaction_normal;
    float *interaction_positions;
    float *interaction_normals;
    float *path_gain;
    cfloat *path_field;
    cfloat *field_xyz;
    cfloat *coefficient;
};

template <typename View, typename T>
View make_optional_view(
    const std::optional<at::Tensor>& value,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef shape,
    int device,
    bool honor_lazy_conjugation) {
    if (!value.has_value()) {
        return {nullptr, 0, 0, 0, false};
    }
    const at::Tensor& tensor = *value;
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == shape, name, " has the wrong shape");
    TORCH_CHECK(tensor.get_device() == device, name, " must share valid device");
    return {
        tensor.data_ptr<T>(),
        tensor.stride(0),
        tensor.dim() > 1 ? tensor.stride(1) : 0,
        tensor.dim() > 2 ? tensor.stride(2) : 0,
        honor_lazy_conjugation ? tensor.is_conj() : true};
}

template <typename Views, typename FloatView, typename ComplexView>
Views make_continuous_views(
    const std::optional<at::Tensor>& path_length_m,
    const std::optional<at::Tensor>& delay_s,
    const std::optional<at::Tensor>& field_direction,
    const std::optional<at::Tensor>& interaction_position,
    const std::optional<at::Tensor>& interaction_normal,
    const std::optional<at::Tensor>& interaction_positions,
    const std::optional<at::Tensor>& interaction_normals,
    const std::optional<at::Tensor>& path_gain,
    const std::optional<at::Tensor>& path_field,
    const std::optional<at::Tensor>& field_xyz,
    const std::optional<at::Tensor>& coefficient,
    int64_t rows,
    int64_t sequence_width,
    int device,
    bool honor_lazy_conjugation) {
    return {
        make_optional_view<FloatView, float>(path_length_m, "path_length_m", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(delay_s, "delay_s", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(field_direction, "field_direction", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_position, "interaction_position", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_normal, "interaction_normal", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_positions, "interaction_positions", at::kFloat, {rows, sequence_width, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_normals, "interaction_normals", at::kFloat, {rows, sequence_width, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(path_gain, "path_gain", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(path_field, "path_field", at::kComplexFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(field_xyz, "field_xyz", at::kComplexFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(coefficient, "coefficient", at::kComplexFloat, {rows}, device, honor_lazy_conjugation)};
}

struct AllocatedContinuous {
    at::Tensor path_length_m;
    at::Tensor delay_s;
    at::Tensor field_direction;
    at::Tensor interaction_position;
    at::Tensor interaction_normal;
    at::Tensor interaction_positions;
    at::Tensor interaction_normals;
    at::Tensor path_gain;
    at::Tensor path_field;
    at::Tensor field_xyz;
    at::Tensor coefficient;

    ContinuousOutputs view() const {
        return {
            path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
            field_direction.data_ptr<float>(), interaction_position.data_ptr<float>(),
            interaction_normal.data_ptr<float>(), interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(), path_gain.data_ptr<float>(),
            path_field.data_ptr<cfloat>(), field_xyz.data_ptr<cfloat>(),
            coefficient.data_ptr<cfloat>()};
    }

    pybind11::dict dict() const {
        pybind11::dict result;
        result["path_length_m"] = path_length_m;
        result["delay_s"] = delay_s;
        result["field_direction"] = field_direction;
        result["interaction_position"] = interaction_position;
        result["interaction_normal"] = interaction_normal;
        result["interaction_positions"] = interaction_positions;
        result["interaction_normals"] = interaction_normals;
        result["path_gain"] = path_gain;
        result["path_field"] = path_field;
        result["field_xyz"] = field_xyz;
        result["coefficient"] = coefficient;
        return result;
    }
};

inline AllocatedContinuous allocate_continuous(
    const at::Tensor& reference,
    int64_t rows,
    int64_t sequence_width) {
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    return {
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, complex_options),
        at::empty({rows, 3}, complex_options),
        at::empty({rows}, complex_options)};
}

}  // namespace channel::evaluated_paths_ad

#include <cstdint>
#include <optional>

#define launch_blocks evaluated_paths_ad_launch_blocks

namespace {

constexpr int kPackAdBlockSize = 256;
using cfloat = c10::complex<float>;
using channel::check_tensor;
using channel::evaluated_paths_ad::AllocatedContinuous;
using channel::evaluated_paths_ad::ContinuousOutputs;
using channel::evaluated_paths_ad::allocate_continuous;

template <typename T>
struct OptionalView {
    const T *data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    bool present;
};

struct ContinuousViews {
    OptionalView<float> path_length_m;
    OptionalView<float> delay_s;
    OptionalView<float> field_direction;
    OptionalView<float> interaction_position;
    OptionalView<float> interaction_normal;
    OptionalView<float> interaction_positions;
    OptionalView<float> interaction_normals;
    OptionalView<float> path_gain;
    OptionalView<cfloat> path_field;
    OptionalView<cfloat> field_xyz;
    OptionalView<cfloat> coefficient;
};

__global__ void evaluated_paths_capacity_continuous_init_kernel(
    ContinuousOutputs output,
    int64_t rows,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        output.path_length_m[row] = 0.0f;
        output.delay_s[row] = 0.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = cfloat(0.0f, 0.0f);
        output.coefficient[row] = cfloat(0.0f, 0.0f);
        const int64_t vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            output.field_direction[vec + component] = 0.0f;
            output.interaction_position[vec + component] = 0.0f;
            output.interaction_normal[vec + component] = 0.0f;
            output.field_xyz[vec + component] = cfloat(0.0f, 0.0f);
        }
        const int64_t sequence_vec = row * sequence_width * 3;
        for (int64_t item = 0; item < sequence_width * 3; ++item) {
            output.interaction_positions[sequence_vec + item] = 0.0f;
            output.interaction_normals[sequence_vec + item] = 0.0f;
        }
    }
}

template <typename T>
__device__ __forceinline__ T read_scalar(
    const OptionalView<T>& view,
    int64_t row) {
    return view.present ? view.data[row * view.stride0] : T(0);
}

template <typename T>
__device__ __forceinline__ T read_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t component) {
    return view.present
        ? view.data[row * view.stride0 + component * view.stride1]
        : T(0);
}

template <typename T>
__device__ __forceinline__ T read_sequence_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t slot,
    int64_t component) {
    return view.present
        ? view.data[
              row * view.stride0 + slot * view.stride1 + component * view.stride2]
        : T(0);
}

__global__ void evaluated_paths_capacity_backward_scatter_kernel(
    const bool *__restrict__ valid,
    const int64_t *__restrict__ selected_row_index,
    ContinuousViews grad_output,
    ContinuousOutputs grad_input,
    int64_t row_capacity,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!valid[destination]) {
            continue;
        }
        const int64_t source = selected_row_index[destination];
        grad_input.path_length_m[source] =
            read_scalar(grad_output.path_length_m, destination);
        grad_input.delay_s[source] = read_scalar(grad_output.delay_s, destination);
        grad_input.path_gain[source] = read_scalar(grad_output.path_gain, destination);
        grad_input.path_field[source] = read_scalar(grad_output.path_field, destination);
        grad_input.coefficient[source] = read_scalar(grad_output.coefficient, destination);
        const int64_t source_vec = source * 3;
        for (int component = 0; component < 3; ++component) {
            grad_input.field_direction[source_vec + component] =
                read_vector(grad_output.field_direction, destination, component);
            grad_input.interaction_position[source_vec + component] =
                read_vector(grad_output.interaction_position, destination, component);
            grad_input.interaction_normal[source_vec + component] =
                read_vector(grad_output.interaction_normal, destination, component);
            grad_input.field_xyz[source_vec + component] =
                read_vector(grad_output.field_xyz, destination, component);
        }
        const int64_t source_sequence = source * sequence_width * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            for (int component = 0; component < 3; ++component) {
                const int64_t item = source_sequence + slot * 3 + component;
                grad_input.interaction_positions[item] = read_sequence_vector(
                    grad_output.interaction_positions,
                    destination,
                    slot,
                    component);
                grad_input.interaction_normals[item] = read_sequence_vector(
                    grad_output.interaction_normals,
                    destination,
                    slot,
                    component);
            }
        }
    }
}

__global__ void evaluated_paths_capacity_jvp_gather_kernel(
    const bool *__restrict__ valid,
    const int64_t *__restrict__ selected_row_index,
    ContinuousViews tangent_input,
    ContinuousOutputs tangent_output,
    int64_t row_capacity,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!valid[destination]) {
            continue;
        }
        const int64_t source = selected_row_index[destination];
        tangent_output.path_length_m[destination] =
            read_scalar(tangent_input.path_length_m, source);
        tangent_output.delay_s[destination] = read_scalar(tangent_input.delay_s, source);
        tangent_output.path_gain[destination] = read_scalar(tangent_input.path_gain, source);
        tangent_output.path_field[destination] =
            read_scalar(tangent_input.path_field, source);
        tangent_output.coefficient[destination] =
            read_scalar(tangent_input.coefficient, source);
        const int64_t destination_vec = destination * 3;
        for (int component = 0; component < 3; ++component) {
            tangent_output.field_direction[destination_vec + component] =
                read_vector(tangent_input.field_direction, source, component);
            tangent_output.interaction_position[destination_vec + component] =
                read_vector(tangent_input.interaction_position, source, component);
            tangent_output.interaction_normal[destination_vec + component] =
                read_vector(tangent_input.interaction_normal, source, component);
            tangent_output.field_xyz[destination_vec + component] =
                read_vector(tangent_input.field_xyz, source, component);
        }
        const int64_t destination_sequence = destination * sequence_width * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            for (int component = 0; component < 3; ++component) {
                const int64_t item = destination_sequence + slot * 3 + component;
                tangent_output.interaction_positions[item] = read_sequence_vector(
                    tangent_input.interaction_positions, source, slot, component);
                tangent_output.interaction_normals[item] = read_sequence_vector(
                    tangent_input.interaction_normals, source, slot, component);
            }
        }
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kPackAdBlockSize - 1) / kPackAdBlockSize);
}

void check_selection(
    const at::Tensor& valid,
    const at::Tensor& selected_row_index) {
    check_tensor(valid, "valid", at::kBool, 1);
    check_tensor(selected_row_index, "selected_row_index", at::kLong, 1);
    TORCH_CHECK(
        selected_row_index.sizes() == valid.sizes(),
        "selected_row_index must match valid");
    TORCH_CHECK(
        selected_row_index.get_device() == valid.get_device(),
        "selected_row_index must share valid device");
}

}  // namespace

pybind11::dict channel_evaluated_paths_compact_finalize_backward(
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> grad_path_length_m,
    std::optional<at::Tensor> grad_delay_s,
    std::optional<at::Tensor> grad_field_direction,
    std::optional<at::Tensor> grad_interaction_position,
    std::optional<at::Tensor> grad_interaction_normal,
    std::optional<at::Tensor> grad_interaction_positions,
    std::optional<at::Tensor> grad_interaction_normals,
    std::optional<at::Tensor> grad_path_gain,
    std::optional<at::Tensor> grad_path_field,
    std::optional<at::Tensor> grad_field_xyz,
    std::optional<at::Tensor> grad_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    check_selection(valid, selected_row_index);
    TORCH_CHECK(candidate_count >= 0, "candidate_count must be non-negative");
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int device = valid.get_device();
    const int64_t row_capacity = valid.size(0);
    auto views = channel::evaluated_paths_ad::make_continuous_views<
        ContinuousViews, OptionalView<float>, OptionalView<cfloat>>(
        grad_path_length_m,
        grad_delay_s,
        grad_field_direction,
        grad_interaction_position,
        grad_interaction_normal,
        grad_interaction_positions,
        grad_interaction_normals,
        grad_path_gain,
        grad_path_field,
        grad_field_xyz,
        grad_coefficient,
        row_capacity,
        sequence_width,
        device,
        false);
    auto outputs = allocate_continuous(valid, candidate_count, sequence_width);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (candidate_count > 0) {
        evaluated_paths_capacity_continuous_init_kernel<<<
            launch_blocks(candidate_count), kPackAdBlockSize, 0, stream>>>(
            outputs.view(), candidate_count, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (row_capacity > 0) {
        evaluated_paths_capacity_backward_scatter_kernel<<<
            launch_blocks(row_capacity), kPackAdBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            selected_row_index.data_ptr<int64_t>(),
            views,
            outputs.view(),
            row_capacity,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs.dict();
}

pybind11::dict channel_evaluated_paths_compact_finalize_jvp(
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> tangent_path_length_m,
    std::optional<at::Tensor> tangent_delay_s,
    std::optional<at::Tensor> tangent_field_direction,
    std::optional<at::Tensor> tangent_interaction_position,
    std::optional<at::Tensor> tangent_interaction_normal,
    std::optional<at::Tensor> tangent_interaction_positions,
    std::optional<at::Tensor> tangent_interaction_normals,
    std::optional<at::Tensor> tangent_path_gain,
    std::optional<at::Tensor> tangent_path_field,
    std::optional<at::Tensor> tangent_field_xyz,
    std::optional<at::Tensor> tangent_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    check_selection(valid, selected_row_index);
    TORCH_CHECK(candidate_count >= 0, "candidate_count must be non-negative");
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int device = valid.get_device();
    const int64_t row_capacity = valid.size(0);
    auto views = channel::evaluated_paths_ad::make_continuous_views<
        ContinuousViews, OptionalView<float>, OptionalView<cfloat>>(
        tangent_path_length_m,
        tangent_delay_s,
        tangent_field_direction,
        tangent_interaction_position,
        tangent_interaction_normal,
        tangent_interaction_positions,
        tangent_interaction_normals,
        tangent_path_gain,
        tangent_path_field,
        tangent_field_xyz,
        tangent_coefficient,
        candidate_count,
        sequence_width,
        device,
        false);
    auto outputs = allocate_continuous(valid, row_capacity, sequence_width);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (row_capacity > 0) {
        evaluated_paths_capacity_continuous_init_kernel<<<
            launch_blocks(row_capacity), kPackAdBlockSize, 0, stream>>>(
            outputs.view(), row_capacity, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        evaluated_paths_capacity_jvp_gather_kernel<<<
            launch_blocks(row_capacity), kPackAdBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            selected_row_index.data_ptr<int64_t>(),
            views,
            outputs.view(),
            row_capacity,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs.dict();
}

pybind11::dict channel_evaluated_paths_capacity_pack_backward(
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> grad_path_length_m,
    std::optional<at::Tensor> grad_delay_s,
    std::optional<at::Tensor> grad_field_direction,
    std::optional<at::Tensor> grad_interaction_position,
    std::optional<at::Tensor> grad_interaction_normal,
    std::optional<at::Tensor> grad_interaction_positions,
    std::optional<at::Tensor> grad_interaction_normals,
    std::optional<at::Tensor> grad_path_gain,
    std::optional<at::Tensor> grad_path_field,
    std::optional<at::Tensor> grad_field_xyz,
    std::optional<at::Tensor> grad_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    return channel_evaluated_paths_compact_finalize_backward(
        valid,
        selected_row_index,
        grad_path_length_m,
        grad_delay_s,
        grad_field_direction,
        grad_interaction_position,
        grad_interaction_normal,
        grad_interaction_positions,
        grad_interaction_normals,
        grad_path_gain,
        grad_path_field,
        grad_field_xyz,
        grad_coefficient,
        candidate_count,
        sequence_width);
}

pybind11::dict channel_evaluated_paths_capacity_pack_jvp(
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> tangent_path_length_m,
    std::optional<at::Tensor> tangent_delay_s,
    std::optional<at::Tensor> tangent_field_direction,
    std::optional<at::Tensor> tangent_interaction_position,
    std::optional<at::Tensor> tangent_interaction_normal,
    std::optional<at::Tensor> tangent_interaction_positions,
    std::optional<at::Tensor> tangent_interaction_normals,
    std::optional<at::Tensor> tangent_path_gain,
    std::optional<at::Tensor> tangent_path_field,
    std::optional<at::Tensor> tangent_field_xyz,
    std::optional<at::Tensor> tangent_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    return channel_evaluated_paths_compact_finalize_jvp(
        valid,
        selected_row_index,
        tangent_path_length_m,
        tangent_delay_s,
        tangent_field_direction,
        tangent_interaction_position,
        tangent_interaction_normal,
        tangent_interaction_positions,
        tangent_interaction_normals,
        tangent_path_gain,
        tangent_path_field,
        tangent_field_xyz,
        tangent_coefficient,
        candidate_count,
        sequence_width);
}

#undef launch_blocks

