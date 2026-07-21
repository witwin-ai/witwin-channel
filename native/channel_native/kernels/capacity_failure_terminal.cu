#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"

namespace {

__global__ void capacity_failure_terminal_check_kernel(
    const int *__restrict__ failure_state) {
    if (failure_state[0] != 0) {
        asm volatile("trap;");
    }
}

}  // namespace

void cn_capacity_failure_terminal_check(at::Tensor failure_state) {
    channel_native::check_tensor(
        failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(
        failure_state.size(0) == 1,
        "failure_state must have shape (1,)");

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(failure_state.get_device()).stream();
    capacity_failure_terminal_check_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int>());
    // A post-launch status query could observe this deliberately tiny trap
    // before the caller's next normal synchronization boundary.
}
