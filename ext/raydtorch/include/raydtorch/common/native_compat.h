#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace raydtorch {

inline void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

inline void *jit_cuda_stream() {
    const int device_index = at::cuda::current_device();
    return at::cuda::getCurrentCUDAStream(device_index).stream();
}

inline void audit_cuda_kernel_launch(
    const char *,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint64_t) {}

inline void audit_cuda_memset_async() {}
inline void audit_cuda_memcpy_async() {}
inline void audit_cuda_stream_synchronize() {}
inline void audit_cub_sort() {}
inline void audit_cub_scan() {}

} // namespace raydtorch
