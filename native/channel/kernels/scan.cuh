// Copyright Xingyu Chen.
// Runs CUB prefix scans on the caller's stream with Torch-allocated scratch.

#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace channel {
namespace scan {

// Reads flags[i] for i < count and 0 at i == count, so one exclusive scan over
// count + 1 items also writes the total into offsets[count].
struct FlagOrZero {
    const int *flags;
    int64_t count;

    __host__ __device__ __forceinline__ int operator()(int64_t index) const {
        return index < count ? flags[index] : 0;
    }
};

inline at::Tensor scratch_bytes(const at::Tensor &reference, size_t bytes) {
    return at::empty(
        {static_cast<int64_t>(bytes > 0 ? bytes : 1)},
        reference.options().dtype(at::kByte));
}

// offsets must hold count + 1 int32 values; offsets[count] receives the total.
inline void exclusive_flag_offsets(
    const at::Tensor &flags, at::Tensor &offsets, int64_t count, cudaStream_t stream) {
    TORCH_CHECK(
        count >= 0 && count < std::numeric_limits<int>::max(),
        "flag scan row count exceeds the int32 scan range");
    TORCH_CHECK(offsets.numel() >= count + 1, "flag scan offsets must hold count + 1 values");
    cub::CountingInputIterator<int64_t> rows(0);
    cub::TransformInputIterator<int, FlagOrZero, cub::CountingInputIterator<int64_t>> input(
        rows, FlagOrZero{flags.data_ptr<int>(), count});
    const int items = static_cast<int>(count + 1);
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, input, offsets.data_ptr<int>(), items, stream));
    auto scratch = scratch_bytes(offsets, bytes);
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scratch.data_ptr<uint8_t>(), bytes, input, offsets.data_ptr<int>(), items, stream));
}

// In-place inclusive sum over the first count int64 values.
inline void inclusive_sum_in_place(at::Tensor &values, int64_t count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    TORCH_CHECK(
        count <= std::numeric_limits<int>::max(),
        "inclusive scan row count exceeds the int32 scan range");
    int64_t *data = values.data_ptr<int64_t>();
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, bytes, data, data, static_cast<int>(count), stream));
    auto scratch = scratch_bytes(values, bytes);
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scratch.data_ptr<uint8_t>(), bytes, data, data, static_cast<int>(count), stream));
}

}  // namespace scan
}  // namespace channel
