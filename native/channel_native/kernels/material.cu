#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace {

constexpr int kMaterialBlockSize = 256;

void check_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void face_material_tensors_kernel(
    const float *__restrict__ material_eps_r,
    const float *__restrict__ material_sigma_e,
    const float *__restrict__ material_mu_r,
    const int *__restrict__ face_material_id,
    float *__restrict__ face_eps_r,
    float *__restrict__ face_sigma_e,
    float *__restrict__ face_mu_r,
    float *__restrict__ face_gain,
    bool *__restrict__ face_valid,
    int64_t face_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         face < face_count;
         face += stride) {
        const int material_id = face_material_id[face];
        face_eps_r[face] = material_eps_r[material_id];
        face_sigma_e[face] = material_sigma_e[material_id];
        face_mu_r[face] = material_mu_r[material_id];
        face_gain[face] = 1.0f;
        face_valid[face] = true;
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id) {
    check_tensor(material_eps_r, "material_eps_r", at::kFloat, 1);
    check_tensor(material_sigma_e, "material_sigma_e", at::kFloat, 1);
    check_tensor(material_mu_r, "material_mu_r", at::kFloat, 1);
    check_tensor(face_material_id, "face_material_id", at::kInt, 1);
    TORCH_CHECK(material_sigma_e.size(0) == material_eps_r.size(0), "material_sigma_e must match material_eps_r");
    TORCH_CHECK(material_mu_r.size(0) == material_eps_r.size(0), "material_mu_r must match material_eps_r");

    const int64_t face_count = face_material_id.size(0);
    auto face_eps_r = at::empty({face_count}, material_eps_r.options());
    auto face_sigma_e = at::empty({face_count}, material_eps_r.options());
    auto face_mu_r = at::empty({face_count}, material_eps_r.options());
    auto face_gain = at::empty({face_count}, material_eps_r.options());
    auto face_valid = at::empty({face_count}, face_material_id.options().dtype(at::kBool));

    if (face_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(material_eps_r.get_device()).stream();
        const int block_count = static_cast<int>((face_count + kMaterialBlockSize - 1) / kMaterialBlockSize);
        face_material_tensors_kernel<<<block_count, kMaterialBlockSize, 0, stream>>>(
            material_eps_r.data_ptr<float>(),
            material_sigma_e.data_ptr<float>(),
            material_mu_r.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_valid.data_ptr<bool>(),
            face_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid};
}
