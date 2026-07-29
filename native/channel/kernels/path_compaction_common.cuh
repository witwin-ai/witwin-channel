// Copyright Xingyu Chen.
// Shares path compaction common CUDA helpers.

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include <tuple>

namespace {

constexpr int kPathBlockSize = 256;

struct CompactCountObservation {
    int64_t count;
    int control_error;
};

__global__ void compact_count_control_metadata_kernel(
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    const int *__restrict__ control_error,
    int64_t row_count,
    int64_t *__restrict__ metadata) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        metadata[0] = control_error[0] == 0
            ? static_cast<int64_t>(flags[row_count - 1]) +
                static_cast<int64_t>(offsets[row_count - 1])
            : -1;
    }
}

inline CompactCountObservation observe_compact_count(
    const at::Tensor& flags,
    const at::Tensor& offsets,
    int64_t row_count,
    cudaStream_t stream,
    const at::Tensor* control_error = nullptr) {
    TORCH_CHECK(row_count > 0, "compact count observation requires rows");
    if (control_error != nullptr) {
        auto metadata = at::empty({1}, flags.options().dtype(at::kLong));
        compact_count_control_metadata_kernel<<<1, 1, 0, stream>>>(
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            control_error->data_ptr<int>(),
            row_count,
            metadata.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int64_t host_metadata = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &host_metadata,
            metadata.data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        return {
            host_metadata < 0 ? 0 : host_metadata,
            host_metadata < 0 ? 1 : 0};
    }
    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + row_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + row_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {
        static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset),
        0};
}

void check_cuda_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name, at::kFloat, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

void check_path_block_shapes(
    const at::Tensor& valid,
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    const at::Tensor& depth,
    const at::Tensor& component_id,
    const at::Tensor& primitive_id,
    const at::Tensor& edge_id,
    const at::Tensor& path_length,
    const at::Tensor& delay,
    const at::Tensor& path_gain) {
    check_cuda_tensor(valid, "valid", at::kBool, 1);
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(depth, "depth", at::kInt, 1);
    check_cuda_tensor(component_id, "component_id", at::kInt, 1);
    check_cuda_tensor(primitive_id, "primitive_id", at::kInt, 1);
    check_cuda_tensor(edge_id, "edge_id", at::kInt, 1);
    check_cuda_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(path_gain, "path_gain", at::kFloat, 1);
    const int64_t count = valid.size(0);
    TORCH_CHECK(tx_id.size(0) == count, "tx_id must match valid");
    TORCH_CHECK(rx_id.size(0) == count, "rx_id must match valid");
    TORCH_CHECK(depth.size(0) == count, "depth must match valid");
    TORCH_CHECK(component_id.size(0) == count, "component_id must match valid");
    TORCH_CHECK(primitive_id.size(0) == count, "primitive_id must match valid");
    TORCH_CHECK(edge_id.size(0) == count, "edge_id must match valid");
    TORCH_CHECK(path_length.size(0) == count, "path_length_m must match valid");
    TORCH_CHECK(delay.size(0) == count, "delay_s must match valid");
    TORCH_CHECK(path_gain.size(0) == count, "path_gain must match valid");
}

int64_t launch_blocks(int64_t count) {
    return (count + kPathBlockSize - 1) / kPathBlockSize;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
empty_path_block_from(const at::Tensor& reference) {
    auto bool_options = reference.options().dtype(at::kBool);
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    return {
        at::empty({0}, bool_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
    };
}

pybind11::dict empty_deterministic_los_topology_block_from(const at::Tensor& reference, int64_t sequence_width) {
    auto bool_options = reference.options().dtype(at::kBool);
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    pybind11::dict out;
    out["valid"] = at::empty({0}, bool_options);
    out["tx_id"] = at::empty({0}, int_options);
    out["rx_id"] = at::empty({0}, int_options);
    out["depth"] = at::empty({0}, int_options);
    out["component_id"] = at::empty({0}, int_options);
    out["primitive_id"] = at::empty({0}, int_options);
    out["edge_id"] = at::empty({0}, int_options);
    out["path_length_m"] = at::empty({0}, float_options);
    out["delay_s"] = at::empty({0}, float_options);
    out["path_gain"] = at::empty({0}, float_options);
    out["path_field"] = at::empty({0}, complex_options);
    out["interaction_position"] = at::empty({0, 3}, float_options);
    out["interaction_normal"] = at::empty({0, 3}, float_options);
    out["material_id"] = at::empty({0}, int_options);
    out["primitive_sequence"] = at::empty({0, sequence_width}, int_options);
    out["material_sequence"] = at::empty({0, sequence_width}, int_options);
    out["interaction_positions"] = at::empty({0, sequence_width, 3}, float_options);
    out["interaction_normals"] = at::empty({0, sequence_width, 3}, float_options);
    return out;
}

}  // namespace
