#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace witwin::channel::native_ext::common {

inline void throw_cuda(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(err));
    }
}

} // namespace witwin::channel::native_ext::common

