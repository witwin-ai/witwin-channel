#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <sample/sample_cuda.h>

namespace witwin::channel::native_ext {
namespace {

__global__ void noop_kernel() {}

void throw_if_cuda_error(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(error)
    );
}

} // namespace

int cuda_runtime_version() {
    int version = 0;
    throw_if_cuda_error(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion()");
    return version;
}

void run_cuda_noop() {
    noop_kernel<<<1, 1>>>();
    throw_if_cuda_error(cudaGetLastError(), "noop_kernel launch");
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
}

} // namespace witwin::channel::native_ext
