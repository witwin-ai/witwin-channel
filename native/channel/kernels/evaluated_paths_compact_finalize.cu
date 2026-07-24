#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../tensor_checks.h"
#include "evaluated_paths_payload_plumbing.h"
#include "path_compaction_common.cuh"

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
