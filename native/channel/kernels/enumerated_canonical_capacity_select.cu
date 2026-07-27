#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

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
