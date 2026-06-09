#include <raydtorch/edge/bvh.h>
#include <raydtorch/common/optix_context.h>

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace raydtorch {

[[noreturn]] inline void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

inline void require_local(bool condition, const std::string &message) {
    if (!condition)
        throw_runtime_error_local(message);
}

namespace {

struct Float3 {
    float x;
    float y;
    float z;

    __host__ __device__ Float3()
        : x(0.f), y(0.f), z(0.f) { }

    __host__ __device__ Float3(float x_, float y_, float z_)
        : x(x_), y(y_), z(z_) { }
};

__host__ __device__ inline Float3 min3(const Float3 &a, const Float3 &b) {
    return Float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline Float3 max3(const Float3 &a, const Float3 &b) {
    return Float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

/// One edge per thread; writes its AABB inflated by `inflation` into the packed OptiX buffer (6 floats/edge).
__global__ void compute_edge_optix_aabbs_kernel(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= primitive_count) {
        return;
    }

    const Float3 p0(edge_p0_x[primitive], edge_p0_y[primitive], edge_p0_z[primitive]);
    const Float3 p1(p0.x + edge_e1_x[primitive],
                    p0.y + edge_e1_y[primitive],
                    p0.z + edge_e1_z[primitive]);
    const Float3 bbox_min = min3(p0, p1);
    const Float3 bbox_max = max3(p0, p1);
    const float radius = fmaxf(inflation, 0.0f);
    const int base = primitive * 6;
    out_aabbs[base + 0] = bbox_min.x - radius;
    out_aabbs[base + 1] = bbox_min.y - radius;
    out_aabbs[base + 2] = bbox_min.z - radius;
    out_aabbs[base + 3] = bbox_max.x + radius;
    out_aabbs[base + 4] = bbox_max.y + radius;
    out_aabbs[base + 5] = bbox_max.z + radius;
}

} // namespace

void compute_edge_optix_aabbs_gpu(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float inflation,
    float *out_aabbs) {
    require_local(primitive_count >= 0, "compute_edge_optix_aabbs_gpu(): primitive_count must be non-negative.");
    if (primitive_count == 0) {
        return;
    }
    require_local(edge_p0_x != nullptr && edge_p0_y != nullptr && edge_p0_z != nullptr,
                  "compute_edge_optix_aabbs_gpu(): edge start pointer is null.");
    require_local(edge_e1_x != nullptr && edge_e1_y != nullptr && edge_e1_z != nullptr,
                  "compute_edge_optix_aabbs_gpu(): edge vector pointer is null.");
    require_local(out_aabbs != nullptr, "compute_edge_optix_aabbs_gpu(): output pointer is null.");

    constexpr int block_size = 256;
    const int block_count = (primitive_count + block_size - 1) / block_size;
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_edge_optix_aabbs_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        primitive_count,
        edge_p0_x,
        edge_p0_y,
        edge_p0_z,
        edge_e1_x,
        edge_e1_y,
        edge_e1_z,
        inflation,
        out_aabbs);
}

void compute_edge_optix_aabbs_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z,
    float radius,
    at::Tensor &out_aabbs) {
    if (edge_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("compute_edge_optix_aabbs_cuda(): edge_count exceeds int32 range.");
    compute_edge_optix_aabbs_gpu(
        static_cast<int>(edge_count),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>(),
        radius,
        out_aabbs.data_ptr<float>());
}

} // namespace raydtorch
