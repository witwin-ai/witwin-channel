#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <vector>

// ADR-022 6.5 / 6.6: backward + jvp companions for the BDPT finalize maps.
//
// Both finalize ops are linear:
//   path_gain[i] = los[i] + reflection[i] + diffraction[i] + transmission[i] +
//                  scattering[i]                                   (elementwise)
//   <c>_power    = sum_i <c>[i]                                    (scalar sum)
// Backward is the transpose of that linear map (elementwise, deterministic):
//   grad_<c>[i] = grad_path_gain[i] + grad_<c>_power   (the 0-dim power
//                 cotangent broadcasts to every cell).
// JVP is the forward map on the tangents; the power tangents are fixed-order
// sums (single-block tree reduction, no float atomics) so the JVP stays
// deterministic run-to-run. 6.5 uses 2-D [tx, rx] maps, 6.6 uses 3-D
// [tx, H, W] radiomaps; the algebra is shape-agnostic over the flat elements.

namespace {

constexpr int kBlockSize = 256;

const at::Tensor* map_optional(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none()) {
        return nullptr;
    }
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* map_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

void check_map(const at::Tensor& tensor, const char* name, int64_t rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

at::Tensor zero_map(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

__global__ void finalize_maps_backward_kernel(
    int64_t element_count,
    const float* grad_path_gain,
    const float* grad_los_power,
    const float* grad_reflection_power,
    const float* grad_diffraction_power,
    const float* grad_transmission_power,
    const float* grad_scattering_power,
    float* grad_los,
    float* grad_reflection,
    float* grad_diffraction,
    float* grad_transmission,
    float* grad_scattering) {
    const float los_p = grad_los_power != nullptr ? grad_los_power[0] : 0.0f;
    const float reflection_p =
        grad_reflection_power != nullptr ? grad_reflection_power[0] : 0.0f;
    const float diffraction_p =
        grad_diffraction_power != nullptr ? grad_diffraction_power[0] : 0.0f;
    const float transmission_p =
        grad_transmission_power != nullptr ? grad_transmission_power[0] : 0.0f;
    const float scattering_p =
        grad_scattering_power != nullptr ? grad_scattering_power[0] : 0.0f;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < element_count;
         idx += stride) {
        const float gp = grad_path_gain != nullptr ? grad_path_gain[idx] : 0.0f;
        grad_los[idx] = gp + los_p;
        grad_reflection[idx] = gp + reflection_p;
        grad_diffraction[idx] = gp + diffraction_p;
        grad_transmission[idx] = gp + transmission_p;
        grad_scattering[idx] = gp + scattering_p;
    }
}

// Single block, deterministic. Writes the elementwise path_gain tangent and
// reduces the per-component power tangents in a fixed-order tree reduction.
__global__ void finalize_maps_jvp_kernel(
    int64_t element_count,
    const float* tangent_los,
    const float* tangent_reflection,
    const float* tangent_diffraction,
    const float* tangent_transmission,
    const float* tangent_scattering,
    float* tangent_path_gain,
    float* tangent_los_power,
    float* tangent_reflection_power,
    float* tangent_diffraction_power,
    float* tangent_transmission_power,
    float* tangent_scattering_power) {
    __shared__ float los_sum[kBlockSize];
    __shared__ float reflection_sum[kBlockSize];
    __shared__ float diffraction_sum[kBlockSize];
    __shared__ float transmission_sum[kBlockSize];
    __shared__ float scattering_sum[kBlockSize];
    const int tid = threadIdx.x;
    float local_los = 0.0f;
    float local_reflection = 0.0f;
    float local_diffraction = 0.0f;
    float local_transmission = 0.0f;
    float local_scattering = 0.0f;
    for (int64_t idx = tid; idx < element_count; idx += blockDim.x) {
        const float los_v = tangent_los != nullptr ? tangent_los[idx] : 0.0f;
        const float reflection_v =
            tangent_reflection != nullptr ? tangent_reflection[idx] : 0.0f;
        const float diffraction_v =
            tangent_diffraction != nullptr ? tangent_diffraction[idx] : 0.0f;
        const float transmission_v =
            tangent_transmission != nullptr ? tangent_transmission[idx] : 0.0f;
        const float scattering_v =
            tangent_scattering != nullptr ? tangent_scattering[idx] : 0.0f;
        tangent_path_gain[idx] =
            los_v + reflection_v + diffraction_v + transmission_v + scattering_v;
        local_los += los_v;
        local_reflection += reflection_v;
        local_diffraction += diffraction_v;
        local_transmission += transmission_v;
        local_scattering += scattering_v;
    }
    los_sum[tid] = local_los;
    reflection_sum[tid] = local_reflection;
    diffraction_sum[tid] = local_diffraction;
    transmission_sum[tid] = local_transmission;
    scattering_sum[tid] = local_scattering;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            los_sum[tid] += los_sum[tid + offset];
            reflection_sum[tid] += reflection_sum[tid + offset];
            diffraction_sum[tid] += diffraction_sum[tid + offset];
            transmission_sum[tid] += transmission_sum[tid + offset];
            scattering_sum[tid] += scattering_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        tangent_los_power[0] = los_sum[0];
        tangent_reflection_power[0] = reflection_sum[0];
        tangent_diffraction_power[0] = diffraction_sum[0];
        tangent_transmission_power[0] = transmission_sum[0];
        tangent_scattering_power[0] = scattering_sum[0];
    }
}

pybind11::dict finalize_backward_common(
    const at::Tensor& los,
    int64_t rank,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components) {
    check_map(los, "los", rank);
    const auto map_shape = los.sizes();
    const std::vector<int64_t> scalar_shape = {};
    at::Tensor gpg_s, glp_s, grp_s, gdp_s, gtp_s, gsp_s;
    const at::Tensor* gpg = map_optional(
        std::move(grad_path_gain), gpg_s, "grad_path_gain", at::kFloat, map_shape, los);
    const at::Tensor* glp = map_optional(
        std::move(grad_los_power), glp_s, "grad_los_power", at::kFloat, scalar_shape, los);
    const at::Tensor* grp = map_optional(
        std::move(grad_reflection_power), grp_s, "grad_reflection_power", at::kFloat, scalar_shape, los);
    const at::Tensor* gdp = map_optional(
        std::move(grad_diffraction_power), gdp_s, "grad_diffraction_power", at::kFloat, scalar_shape, los);
    const at::Tensor* gtp = map_optional(
        std::move(grad_transmission_power), gtp_s, "grad_transmission_power", at::kFloat, scalar_shape, los);
    const at::Tensor* gsp = map_optional(
        std::move(grad_scattering_power), gsp_s, "grad_scattering_power", at::kFloat, scalar_shape, los);

    pybind11::dict out;
    if (!need_grad_components) {
        out["grad_los"] = pybind11::none();
        out["grad_reflection"] = pybind11::none();
        out["grad_diffraction"] = pybind11::none();
        out["grad_transmission"] = pybind11::none();
        out["grad_scattering"] = pybind11::none();
        return out;
    }
    auto grad_los = zero_map(map_shape, los.options());
    auto grad_reflection = zero_map(map_shape, los.options());
    auto grad_diffraction = zero_map(map_shape, los.options());
    auto grad_transmission = zero_map(map_shape, los.options());
    auto grad_scattering = zero_map(map_shape, los.options());
    const int64_t element_count = los.numel();
    if (element_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        const int blocks = static_cast<int>(
            (element_count + kBlockSize - 1) / kBlockSize);
        finalize_maps_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
            element_count,
            map_ptr<float>(gpg),
            map_ptr<float>(glp),
            map_ptr<float>(grp),
            map_ptr<float>(gdp),
            map_ptr<float>(gtp),
            map_ptr<float>(gsp),
            grad_los.data_ptr<float>(),
            grad_reflection.data_ptr<float>(),
            grad_diffraction.data_ptr<float>(),
            grad_transmission.data_ptr<float>(),
            grad_scattering.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    out["grad_los"] = pybind11::cast(grad_los);
    out["grad_reflection"] = pybind11::cast(grad_reflection);
    out["grad_diffraction"] = pybind11::cast(grad_diffraction);
    out["grad_transmission"] = pybind11::cast(grad_transmission);
    out["grad_scattering"] = pybind11::cast(grad_scattering);
    return out;
}

pybind11::dict finalize_jvp_common(
    const at::Tensor& los,
    int64_t rank,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering) {
    check_map(los, "los", rank);
    const auto map_shape = los.sizes();
    at::Tensor tl_s, tr_s, td_s, tt_s, ts_s;
    const at::Tensor* tl = map_optional(
        std::move(tangent_los), tl_s, "tangent_los", at::kFloat, map_shape, los);
    const at::Tensor* tr = map_optional(
        std::move(tangent_reflection), tr_s, "tangent_reflection", at::kFloat, map_shape, los);
    const at::Tensor* td = map_optional(
        std::move(tangent_diffraction), td_s, "tangent_diffraction", at::kFloat, map_shape, los);
    const at::Tensor* tt = map_optional(
        std::move(tangent_transmission), tt_s, "tangent_transmission", at::kFloat, map_shape, los);
    const at::Tensor* ts = map_optional(
        std::move(tangent_scattering), ts_s, "tangent_scattering", at::kFloat, map_shape, los);

    auto tangent_path_gain = zero_map(map_shape, los.options());
    auto scalar_options = los.options().dtype(at::kFloat);
    auto tangent_los_power = zero_map({}, scalar_options);
    auto tangent_reflection_power = zero_map({}, scalar_options);
    auto tangent_diffraction_power = zero_map({}, scalar_options);
    auto tangent_transmission_power = zero_map({}, scalar_options);
    auto tangent_scattering_power = zero_map({}, scalar_options);
    const int64_t element_count = los.numel();
    if (element_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
        finalize_maps_jvp_kernel<<<1, kBlockSize, 0, stream>>>(
            element_count,
            map_ptr<float>(tl),
            map_ptr<float>(tr),
            map_ptr<float>(td),
            map_ptr<float>(tt),
            map_ptr<float>(ts),
            tangent_path_gain.data_ptr<float>(),
            tangent_los_power.data_ptr<float>(),
            tangent_reflection_power.data_ptr<float>(),
            tangent_diffraction_power.data_ptr<float>(),
            tangent_transmission_power.data_ptr<float>(),
            tangent_scattering_power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_path_gain"] = tangent_path_gain;
    out["tangent_los_power"] = tangent_los_power;
    out["tangent_reflection_power"] = tangent_reflection_power;
    out["tangent_diffraction_power"] = tangent_diffraction_power;
    out["tangent_transmission_power"] = tangent_transmission_power;
    out["tangent_scattering_power"] = tangent_scattering_power;
    return out;
}

}  // namespace

pybind11::dict channel_bdpt_finalize_point_components_backward(
    at::Tensor los,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components) {
    return finalize_backward_common(
        los, 2, std::move(grad_path_gain), std::move(grad_los_power),
        std::move(grad_reflection_power), std::move(grad_diffraction_power),
        std::move(grad_transmission_power), std::move(grad_scattering_power),
        need_grad_components);
}

pybind11::dict channel_bdpt_finalize_point_components_jvp(
    at::Tensor los,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering) {
    return finalize_jvp_common(
        los, 2, std::move(tangent_los), std::move(tangent_reflection),
        std::move(tangent_diffraction), std::move(tangent_transmission),
        std::move(tangent_scattering));
}

pybind11::dict channel_bdpt_finalize_component_maps_backward(
    at::Tensor los,
    pybind11::object grad_path_gain,
    pybind11::object grad_los_power,
    pybind11::object grad_reflection_power,
    pybind11::object grad_diffraction_power,
    pybind11::object grad_transmission_power,
    pybind11::object grad_scattering_power,
    bool need_grad_components) {
    return finalize_backward_common(
        los, 3, std::move(grad_path_gain), std::move(grad_los_power),
        std::move(grad_reflection_power), std::move(grad_diffraction_power),
        std::move(grad_transmission_power), std::move(grad_scattering_power),
        need_grad_components);
}

pybind11::dict channel_bdpt_finalize_component_maps_jvp(
    at::Tensor los,
    pybind11::object tangent_los,
    pybind11::object tangent_reflection,
    pybind11::object tangent_diffraction,
    pybind11::object tangent_transmission,
    pybind11::object tangent_scattering) {
    return finalize_jvp_common(
        los, 3, std::move(tangent_los), std::move(tangent_reflection),
        std::move(tangent_diffraction), std::move(tangent_transmission),
        std::move(tangent_scattering));
}
