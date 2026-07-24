#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <cstdint>
#include <limits>

namespace {

constexpr int kReflectionCandidateBlockSize = 256;

struct ReflectionCandidateOutput {
    bool *valid;
    int *candidate_count;
    bool *overflow;
    int *selected_sequences;
    float *selected_hits;
    float *selected_normals;
    int *selected_rx_id;
    float *selected_tx;
    float *selected_rx;
    float *tx_power;
    float *eps_r;
    float *sigma_e;
    float *mu_r;
    float *gain;
    int *first_face;
    int *material_id;
    int *material_sequence;
    float *first_hit;
    float *first_normal;
};

__global__ void reflection_candidate_flags_kernel(
    const int *__restrict__ failure_state,
    const bool *__restrict__ visible,
    int *__restrict__ flags,
    int64_t input_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < input_count;
         row += stride) {
        flags[row] = failure_state[0] == 0 && visible[row] ? 1 : 0;
    }
}

__global__ void reflection_candidate_init_kernel(
    ReflectionCandidateOutput output,
    int64_t candidate_capacity,
    int64_t depth) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < candidate_capacity;
         row += stride) {
        output.valid[row] = false;
        output.selected_rx_id[row] = -1;
        output.tx_power[row] = 0.0f;
        output.first_face[row] = -1;
        output.material_id[row] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t vec_index = row * 3 + axis;
            output.selected_tx[vec_index] = 0.0f;
            output.selected_rx[vec_index] = 0.0f;
            output.first_hit[vec_index] = 0.0f;
            output.first_normal[vec_index] = 0.0f;
        }
        for (int64_t column = 0; column < depth; ++column) {
            const int64_t sequence_index = row * depth + column;
            output.selected_sequences[sequence_index] = -1;
            output.eps_r[sequence_index] = 0.0f;
            output.sigma_e[sequence_index] = 0.0f;
            output.mu_r[sequence_index] = 0.0f;
            output.gain[sequence_index] = 0.0f;
            output.material_sequence[sequence_index] = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vec_index = sequence_index * 3 + axis;
                output.selected_hits[vec_index] = 0.0f;
                output.selected_normals[vec_index] = 0.0f;
            }
        }
    }
}

__global__ void reflection_candidate_status_kernel(
    int *__restrict__ failure_state,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    int64_t input_count,
    int64_t candidate_capacity,
    int *__restrict__ candidate_count,
    bool *__restrict__ overflow) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (failure_state[0] != 0) {
        candidate_count[0] = 0;
        overflow[0] = false;
        return;
    }
    int visible_count = 0;
    if (input_count > 0) {
        const int last = static_cast<int>(input_count - 1);
        visible_count = offsets[last] + flags[last];
    }
    const bool did_overflow = visible_count > candidate_capacity;
    candidate_count[0] = did_overflow ? 0 : visible_count;
    overflow[0] = did_overflow;
    if (did_overflow) {
        atomicOr(
            failure_state,
            channel::capacity::kReflectionCandidateOverflow);
    }
}

template <typename sequence_t>
__global__ void reflection_candidate_gather_kernel(
    const int *__restrict__ failure_state,
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    const bool *__restrict__ overflow,
    const sequence_t *__restrict__ epc_sequences,
    const float *__restrict__ epc_hits,
    const float *__restrict__ epc_normals,
    const int *__restrict__ sequence_batch,
    const int *__restrict__ rx_indices,
    const float *__restrict__ tx,
    const float *__restrict__ rx_positions,
    const float *__restrict__ tx_power,
    int tx_index,
    const float *__restrict__ face_eps_r,
    const float *__restrict__ face_sigma_e,
    const float *__restrict__ face_mu_r,
    const float *__restrict__ face_gain,
    const int *__restrict__ face_material_id,
    ReflectionCandidateOutput output,
    int64_t input_count,
    int64_t depth,
    bool grouped_export) {
    if (failure_state[0] != 0 || overflow[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < input_count;
         row += stride) {
        if (flags[row] == 0) {
            continue;
        }
        const int dst = offsets[row];
        const int rx_id = rx_indices[row];
        output.valid[dst] = true;
        output.selected_rx_id[dst] = rx_id;
        output.tx_power[dst] = tx_power[tx_index];
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t destination_index = static_cast<int64_t>(dst) * 3 + axis;
            output.selected_tx[destination_index] = tx[axis];
            output.selected_rx[destination_index] =
                rx_positions[static_cast<int64_t>(rx_id) * 3 + axis];
        }
        for (int64_t column = 0; column < depth; ++column) {
            const int64_t source_sequence_index = row * depth + column;
            const int64_t destination_sequence_index =
                static_cast<int64_t>(dst) * depth + column;
            const int face = grouped_export
                ? static_cast<int>(epc_sequences[source_sequence_index])
                : sequence_batch[source_sequence_index];
            output.selected_sequences[destination_sequence_index] = face;
            output.eps_r[destination_sequence_index] = face_eps_r[face];
            output.sigma_e[destination_sequence_index] = face_sigma_e[face];
            output.mu_r[destination_sequence_index] = face_mu_r[face];
            output.gain[destination_sequence_index] = face_gain[face];
            output.material_sequence[destination_sequence_index] =
                face_material_id[face];
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t source_vec_index = source_sequence_index * 3 + axis;
                const int64_t destination_vec_index =
                    destination_sequence_index * 3 + axis;
                const float hit_value = epc_hits[source_vec_index];
                const float normal_value = epc_normals[source_vec_index];
                output.selected_hits[destination_vec_index] = hit_value;
                output.selected_normals[destination_vec_index] = normal_value;
                if (column == 0) {
                    const int64_t first_index = static_cast<int64_t>(dst) * 3 + axis;
                    output.first_hit[first_index] = hit_value;
                    output.first_normal[first_index] = normal_value;
                }
            }
        }
        const int first = output.selected_sequences[static_cast<int64_t>(dst) * depth];
        output.first_face[dst] = first;
        output.material_id[dst] = face_material_id[first];
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>(
        (count + kReflectionCandidateBlockSize - 1) /
        kReflectionCandidateBlockSize);
}

void check_row_device(
    const at::Tensor& tensor,
    const char *name,
    int64_t input_count,
    int device) {
    TORCH_CHECK(tensor.size(0) == input_count, name, " must match visible rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share visible device");
}

}  // namespace

pybind11::dict channel_deterministic_reflection_candidate_capacity_block(
    at::Tensor failure_state,
    at::Tensor visible,
    at::Tensor epc_sequences,
    at::Tensor epc_hits,
    at::Tensor epc_normals,
    at::Tensor sequence_batch,
    at::Tensor rx_indices,
    at::Tensor tx,
    at::Tensor rx_positions,
    at::Tensor tx_power,
    int64_t tx_index,
    at::Tensor face_eps_r,
    at::Tensor face_sigma_e,
    at::Tensor face_mu_r,
    at::Tensor face_gain,
    at::Tensor face_material_id,
    bool grouped_export,
    int64_t candidate_capacity) {
    channel::check_tensor(visible, "visible", at::kBool, 1);
    channel::capacity::validate_failure_state(failure_state, visible);
    TORCH_CHECK(epc_sequences.is_cuda(), "epc_sequences must be a CUDA tensor");
    TORCH_CHECK(epc_sequences.is_contiguous(), "epc_sequences must be contiguous");
    TORCH_CHECK(
        epc_sequences.scalar_type() == at::kInt ||
            epc_sequences.scalar_type() == at::kLong,
        "epc_sequences must be int32 or int64");
    TORCH_CHECK(
        epc_sequences.dim() == 2,
        "epc_sequences must have shape (N, depth)");
    channel::check_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    channel::check_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    channel::check_tensor(sequence_batch, "sequence_batch", at::kInt, 2);
    channel::check_tensor(rx_indices, "rx_indices", at::kInt, 1);
    channel::check_tensor(tx, "tx", at::kFloat, 1);
    channel::check_vec3_table(rx_positions, "rx_positions");
    channel::check_tensor(tx_power, "tx_power", at::kFloat, 1);
    channel::check_tensor(face_eps_r, "face_eps_r", at::kFloat, 1);
    channel::check_tensor(face_sigma_e, "face_sigma_e", at::kFloat, 1);
    channel::check_tensor(face_mu_r, "face_mu_r", at::kFloat, 1);
    channel::check_tensor(face_gain, "face_gain", at::kFloat, 1);
    channel::check_tensor(
        face_material_id, "face_material_id", at::kInt, 1);

    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(candidate_capacity >= 0, "candidate_capacity must be non-negative");
    TORCH_CHECK(
        candidate_capacity <= std::numeric_limits<int>::max(),
        "candidate_capacity exceeds int32 indexing capacity");
    const int64_t input_count = visible.size(0);
    const int64_t depth = epc_sequences.size(1);
    TORCH_CHECK(depth > 0, "epc_sequences must have positive depth");
    TORCH_CHECK(
        input_count <= std::numeric_limits<int>::max(),
        "reflection candidate input exceeds int32 indexing capacity");
    TORCH_CHECK(
        depth <= std::numeric_limits<int>::max(),
        "reflection candidate depth exceeds int32 indexing capacity");
    TORCH_CHECK(
        candidate_capacity == 0 ||
            depth <= std::numeric_limits<int64_t>::max() / candidate_capacity,
        "reflection candidate output shape overflows int64");
    TORCH_CHECK(
        input_count == 0 ||
            depth <= std::numeric_limits<int64_t>::max() / input_count,
        "reflection candidate input shape overflows int64");
    TORCH_CHECK(
        epc_hits.sizes() == at::IntArrayRef({input_count, depth, 3}),
        "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(
        epc_normals.sizes() == epc_hits.sizes(),
        "epc_normals must match epc_hits");
    TORCH_CHECK(
        sequence_batch.sizes() == epc_sequences.sizes(),
        "sequence_batch must match epc_sequences");
    TORCH_CHECK(
        rx_indices.size(0) == input_count,
        "rx_indices must match visible rows");
    TORCH_CHECK(
        tx_index >= 0 && tx_index < tx_power.size(0),
        "tx_index is out of range");
    TORCH_CHECK(
        face_sigma_e.sizes() == face_eps_r.sizes(),
        "face_sigma_e must match face_eps_r");
    TORCH_CHECK(
        face_mu_r.sizes() == face_eps_r.sizes(),
        "face_mu_r must match face_eps_r");
    TORCH_CHECK(
        face_gain.sizes() == face_eps_r.sizes(),
        "face_gain must match face_eps_r");
    TORCH_CHECK(
        face_material_id.sizes() == face_eps_r.sizes(),
        "face_material_id must match face_eps_r");

    const int device = visible.get_device();
    check_row_device(epc_sequences, "epc_sequences", input_count, device);
    check_row_device(epc_hits, "epc_hits", input_count, device);
    check_row_device(epc_normals, "epc_normals", input_count, device);
    check_row_device(sequence_batch, "sequence_batch", input_count, device);
    check_row_device(rx_indices, "rx_indices", input_count, device);
    TORCH_CHECK(tx.get_device() == device, "tx must share visible device");
    TORCH_CHECK(
        rx_positions.get_device() == device,
        "rx_positions must share visible device");
    TORCH_CHECK(tx_power.get_device() == device, "tx_power must share visible device");
    TORCH_CHECK(
        face_eps_r.get_device() == device,
        "face_eps_r must share visible device");
    TORCH_CHECK(
        face_sigma_e.get_device() == device,
        "face_sigma_e must share visible device");
    TORCH_CHECK(
        face_mu_r.get_device() == device,
        "face_mu_r must share visible device");
    TORCH_CHECK(
        face_gain.get_device() == device,
        "face_gain must share visible device");
    TORCH_CHECK(
        face_material_id.get_device() == device,
        "face_material_id must share visible device");

    auto bool_options = visible.options().dtype(at::kBool);
    auto int_options = visible.options().dtype(at::kInt);
    auto float_options = visible.options().dtype(at::kFloat);
    auto out_valid = at::empty({candidate_capacity}, bool_options);
    auto candidate_count = at::empty({1}, int_options);
    auto overflow = at::empty({1}, bool_options);
    auto selected_sequences =
        at::empty({candidate_capacity, depth}, int_options);
    auto selected_hits =
        at::empty({candidate_capacity, depth, 3}, float_options);
    auto selected_normals =
        at::empty({candidate_capacity, depth, 3}, float_options);
    auto selected_rx_id = at::empty({candidate_capacity}, int_options);
    auto selected_tx = at::empty({candidate_capacity, 3}, float_options);
    auto selected_rx = at::empty({candidate_capacity, 3}, float_options);
    auto selected_tx_power = at::empty({candidate_capacity}, float_options);
    auto selected_eps_r =
        at::empty({candidate_capacity, depth}, float_options);
    auto selected_sigma_e =
        at::empty({candidate_capacity, depth}, float_options);
    auto selected_mu_r =
        at::empty({candidate_capacity, depth}, float_options);
    auto selected_gain =
        at::empty({candidate_capacity, depth}, float_options);
    auto first_face = at::empty({candidate_capacity}, int_options);
    auto material_id = at::empty({candidate_capacity}, int_options);
    auto material_sequence =
        at::empty({candidate_capacity, depth}, int_options);
    auto first_hit = at::empty({candidate_capacity, 3}, float_options);
    auto first_normal = at::empty({candidate_capacity, 3}, float_options);

    const ReflectionCandidateOutput output{
        out_valid.data_ptr<bool>(),
        candidate_count.data_ptr<int>(),
        overflow.data_ptr<bool>(),
        selected_sequences.data_ptr<int>(),
        selected_hits.data_ptr<float>(),
        selected_normals.data_ptr<float>(),
        selected_rx_id.data_ptr<int>(),
        selected_tx.data_ptr<float>(),
        selected_rx.data_ptr<float>(),
        selected_tx_power.data_ptr<float>(),
        selected_eps_r.data_ptr<float>(),
        selected_sigma_e.data_ptr<float>(),
        selected_mu_r.data_ptr<float>(),
        selected_gain.data_ptr<float>(),
        first_face.data_ptr<int>(),
        material_id.data_ptr<int>(),
        material_sequence.data_ptr<int>(),
        first_hit.data_ptr<float>(),
        first_normal.data_ptr<float>()};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (candidate_capacity > 0) {
        reflection_candidate_init_kernel<<<
            launch_blocks(candidate_capacity),
            kReflectionCandidateBlockSize,
            0,
            stream>>>(output, candidate_capacity, depth);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto flags = at::empty({input_count}, int_options);
    auto offsets = at::empty({input_count}, int_options);
    if (input_count > 0) {
        reflection_candidate_flags_kernel<<<
            launch_blocks(input_count),
            kReflectionCandidateBlockSize,
            0,
            stream>>>(
            failure_state.data_ptr<int>(),
            visible.data_ptr<bool>(),
            flags.data_ptr<int>(),
            input_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t scratch_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(input_count),
            stream));
        auto scratch = at::empty(
            {static_cast<int64_t>(scratch_bytes)},
            visible.options().dtype(at::kByte));
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            scratch.data_ptr<uint8_t>(),
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(input_count),
            stream));
    }

    reflection_candidate_status_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int>(),
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        input_count,
        candidate_capacity,
        candidate_count.data_ptr<int>(),
        overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (input_count > 0) {
        const int blocks = launch_blocks(input_count);
        if (epc_sequences.scalar_type() == at::kLong) {
            reflection_candidate_gather_kernel<int64_t><<<
                blocks, kReflectionCandidateBlockSize, 0, stream>>>(
                failure_state.data_ptr<int>(),
                flags.data_ptr<int>(),
                offsets.data_ptr<int>(),
                overflow.data_ptr<bool>(),
                epc_sequences.data_ptr<int64_t>(),
                epc_hits.data_ptr<float>(),
                epc_normals.data_ptr<float>(),
                sequence_batch.data_ptr<int>(),
                rx_indices.data_ptr<int>(),
                tx.data_ptr<float>(),
                rx_positions.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                static_cast<int>(tx_index),
                face_eps_r.data_ptr<float>(),
                face_sigma_e.data_ptr<float>(),
                face_mu_r.data_ptr<float>(),
                face_gain.data_ptr<float>(),
                face_material_id.data_ptr<int>(),
                output,
                input_count,
                depth,
                grouped_export);
        } else {
            reflection_candidate_gather_kernel<int><<<
                blocks, kReflectionCandidateBlockSize, 0, stream>>>(
                failure_state.data_ptr<int>(),
                flags.data_ptr<int>(),
                offsets.data_ptr<int>(),
                overflow.data_ptr<bool>(),
                epc_sequences.data_ptr<int>(),
                epc_hits.data_ptr<float>(),
                epc_normals.data_ptr<float>(),
                sequence_batch.data_ptr<int>(),
                rx_indices.data_ptr<int>(),
                tx.data_ptr<float>(),
                rx_positions.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                static_cast<int>(tx_index),
                face_eps_r.data_ptr<float>(),
                face_sigma_e.data_ptr<float>(),
                face_mu_r.data_ptr<float>(),
                face_gain.data_ptr<float>(),
                face_material_id.data_ptr<int>(),
                output,
                input_count,
                depth,
                grouped_export);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["valid"] = out_valid;
    result["candidate_count"] = candidate_count;
    result["overflow"] = overflow;
    result["selected_sequences"] = selected_sequences;
    result["selected_hits"] = selected_hits;
    result["selected_normals"] = selected_normals;
    result["selected_rx_id"] = selected_rx_id;
    result["selected_tx"] = selected_tx;
    result["selected_rx"] = selected_rx;
    result["tx_power"] = selected_tx_power;
    result["eps_r"] = selected_eps_r;
    result["sigma_e"] = selected_sigma_e;
    result["mu_r"] = selected_mu_r;
    result["gain"] = selected_gain;
    result["first_face"] = first_face;
    result["material_id"] = material_id;
    result["material_sequence"] = material_sequence;
    result["first_hit"] = first_hit;
    result["first_normal"] = first_normal;
    return result;
}
