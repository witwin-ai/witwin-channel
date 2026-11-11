// Copyright Xingyu Chen.
// Fixed-order device sum of per-row values into one scalar, reproducible bit for bit.

#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

namespace channel::ordered_sum {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxPartials = 1024;
constexpr int64_t kValuesPerBlock = 4096;

// Block-local tree over kThreads lanes in a fixed pairing; lane 0 returns the total.
template <typename T>
__device__ __forceinline__ T block_tree(T value, T* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int width = kThreads / 2; width > 0; width /= 2) {
        if (static_cast<int>(threadIdx.x) < width)
            shared[threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x + width];
        __syncthreads();
    }
    return shared[0];
}

// Block b sums the contiguous range [b * span, (b + 1) * span): every lane adds its strided
// elements in ascending order, then the block tree combines the lanes.
template <typename T>
__global__ void partial_kernel(const T* values, int64_t count, int64_t span, T* partials) {
    __shared__ T shared[kThreads];
    const int64_t begin = static_cast<int64_t>(blockIdx.x) * span;
    const int64_t end = std::min<int64_t>(count, begin + span);
    T value = T(0);
    for (int64_t index = begin + threadIdx.x; index < end; index += kThreads)
        value = value + values[index];
    const T total = block_tree(value, shared);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = total;
}

// One block folds the partials the same way and adds the total to *destination.
template <typename T>
__global__ void finish_kernel(const T* partials, int64_t count, T* destination) {
    __shared__ T shared[kThreads];
    T value = T(0);
    for (int64_t index = threadIdx.x; index < count; index += kThreads)
        value = value + partials[index];
    const T total = block_tree(value, shared);
    if (threadIdx.x == 0)
        *destination = *destination + total;
}

// Adds sum(values[0:count]) to *destination with an order that depends only on count.
// Two launches; the partition is fixed by count, so repeated calls give the same bits.
template <typename T>
[[maybe_unused]] void add_sum(const T* values, int64_t count, T* destination, const at::TensorOptions& options,
             cudaStream_t stream) {
    if (count <= 0)
        return;
    const int64_t blocks = std::min<int64_t>(
        kMaxPartials, (count + kValuesPerBlock - 1) / kValuesPerBlock);
    const int64_t span = (count + blocks - 1) / blocks;
    auto partials = at::empty({blocks}, options);
    partial_kernel<T><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        values, count, span, partials.template data_ptr<T>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finish_kernel<T><<<1, kThreads, 0, stream>>>(
        partials.template data_ptr<T>(), blocks, destination);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace channel::ordered_sum
