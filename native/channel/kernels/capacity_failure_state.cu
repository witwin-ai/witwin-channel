#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

void channel::capacity::validate_failure_state(
    const at::Tensor& failure_state,
    const at::Tensor& reference) {
    channel::check_tensor(
        failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(
        failure_state.size(0) == 1,
        "failure_state must have shape (1,)");
    TORCH_CHECK(
        failure_state.get_device() == reference.get_device(),
        "failure_state must share the input device");
}

at::Tensor channel_capacity_failure_state_create(at::Tensor reference) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    auto failure_state = at::empty({1}, reference.options().dtype(at::kInt));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        failure_state.data_ptr<int>(), 0, sizeof(int), stream));
    return failure_state;
}
