#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/reflection/kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace raydtorch {

namespace {

__global__ void visibility_from_intersection_kernel(
    const int *__restrict__ prim_id,
    const bool *__restrict__ active,
    int64_t count,
    bool *__restrict__ visible) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    visible[idx] = active[idx] && prim_id[idx] < 0;
}

} // namespace

VisibilityForwardOutputs visibility_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &start,
    const at::Tensor &end,
    const at::Tensor &active) {
    const int64_t count = start.size(0);
    at::Tensor direction = (end - start).contiguous();
    at::Tensor tmax = at::ones({count}, start.options());
    IntersectForwardOutputs hit = intersect_forward_cuda(scene, start, direction, tmax, active);

    VisibilityForwardOutputs out;
    out.visible = at::empty({count}, active.options());
    out.tape_prim_id = hit.tape_prim_id;
    out.tape_t = hit.tape_t;

    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(start.get_device()).stream();
    visibility_from_intersection_kernel<<<blocks, threads, 0, stream>>>(
        hit.global_prim_id.data_ptr<int>(),
        active.data_ptr<bool>(),
        count,
        out.visible.data_ptr<bool>());
    return out;
}

} // namespace raydtorch
