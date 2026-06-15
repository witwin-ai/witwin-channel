#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void check_reference(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(tensor.dim() == 2, "reference must have rank 2");
    TORCH_CHECK(tensor.is_contiguous(), "reference must be contiguous");
}

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have rank 2");
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_flat_tensor(const at::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ float uniform01_from_u64(unsigned long long value) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<float>(((value >> 11) & 0x1fffffffffffffULL) * scale);
}

__device__ void direction_from_seed(unsigned long long seed, float* dir) {
    const float u0 = uniform01_from_u64(splitmix64(seed ^ 0x57c5d1f5f8c0a9b3ULL));
    const float u1 = uniform01_from_u64(splitmix64(seed ^ 0xa24baed4963ee407ULL));
    const float z = 1.0f - 2.0f * u0;
    const float phi = static_cast<float>(2.0 * kPi) * u1;
    const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    dir[0] = radial * cosf(phi);
    dir[1] = radial * sinf(phi);
    dir[2] = z;
}

std::vector<at::Tensor> allocate_subpath_state(const at::Tensor& reference, int64_t count) {
    auto float_options = reference.options().dtype(at::kFloat);
    auto int_options = reference.options().dtype(at::kInt);
    auto bool_options = reference.options().dtype(at::kBool);
    return {
        at::empty({count, 3}, float_options),
        at::empty({count, 3}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, bool_options),
    };
}

__global__ void bdpt_light_endpoint_subpaths_kernel(
    int64_t count,
    const float* tx_positions,
    const float* tx_power,
    const int* launch_tx_id,
    const int64_t* light_seed,
    int64_t tx_count,
    float* origin,
    float* direction,
    float* throughput_real,
    float* throughput_imag,
    float* pdf_forward,
    float* pdf_reverse,
    int* depth,
    int* component_mask,
    int* primitive_id,
    int* edge_id,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    bool* valid) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    int tx = launch_tx_id[index];
    const bool is_valid = tx >= 0 && tx < tx_count;
    const float* src = tx_positions + static_cast<int64_t>(is_valid ? tx : 0) * 3;
    float* dst = origin + index * 3;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    float* dir = direction + index * 3;
    if (is_valid) {
        direction_from_seed(static_cast<unsigned long long>(light_seed[index]), dir);
    } else {
        dir[0] = 0.0f;
        dir[1] = 0.0f;
        dir[2] = 0.0f;
    }
    throughput_real[index] = is_valid ? tx_power[tx] : 0.0f;
    throughput_imag[index] = 0.0f;
    pdf_forward[index] = is_valid ? static_cast<float>(1.0 / (4.0 * kPi)) : 0.0f;
    pdf_reverse[index] = 0.0f;
    depth[index] = 0;
    component_mask[index] = 1;
    primitive_id[index] = -1;
    edge_id[index] = -1;
    tx_id[index] = tx;
    rx_id[index] = -1;
    grid_linear_id[index] = -1;
    valid[index] = is_valid;
}

__global__ void bdpt_sensor_endpoint_subpaths_kernel(
    int64_t count,
    const float* rx_positions,
    float* origin,
    float* direction,
    float* throughput_real,
    float* throughput_imag,
    float* pdf_forward,
    float* pdf_reverse,
    int* depth,
    int* component_mask,
    int* primitive_id,
    int* edge_id,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    bool* valid) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float* src = rx_positions + index * 3;
    float* dst = origin + index * 3;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    float* dir = direction + index * 3;
    dir[0] = 0.0f;
    dir[1] = 0.0f;
    dir[2] = -1.0f;
    throughput_real[index] = 1.0f;
    throughput_imag[index] = 0.0f;
    pdf_forward[index] = 0.0f;
    pdf_reverse[index] = 1.0f;
    depth[index] = 0;
    component_mask[index] = 1;
    primitive_id[index] = -1;
    edge_id[index] = -1;
    tx_id[index] = -1;
    rx_id[index] = static_cast<int>(index);
    grid_linear_id[index] = static_cast<int>(index);
    valid[index] = true;
}

__global__ void bdpt_reflected_light_subpaths_kernel(
    int64_t count,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_throughput_imag,
    const float* light_pdf_forward,
    const float* light_pdf_reverse,
    const int* light_depth,
    const int* light_component_mask,
    const int* light_tx_id,
    const int* light_rx_id,
    const int* light_grid_linear_id,
    const bool* light_valid,
    const float* hit_t,
    const float* hit_p,
    const float* hit_n,
    const int* hit_global_prim_id,
    const float* material_gain,
    const bool* material_valid,
    int64_t material_count,
    float* origin,
    float* direction,
    float* throughput_real,
    float* throughput_imag,
    float* pdf_forward,
    float* pdf_reverse,
    int* depth,
    int* component_mask,
    int* primitive_id,
    int* edge_id,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    bool* valid) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int prim = hit_global_prim_id[index];
    const bool prim_in_range = prim >= 0 && static_cast<int64_t>(prim) < material_count;
    const bool material_ok = prim_in_range && material_valid[prim];
    const float gain = material_ok ? fmaxf(material_gain[prim], 0.0f) : 0.0f;
    const bool is_valid = light_valid[index] && prim_in_range && material_ok && hit_t[index] >= 0.0f;
    float* dst_origin = origin + index * 3;
    float* dst_direction = direction + index * 3;
    if (is_valid) {
        const float* src_direction = light_direction + index * 3;
        const float* hit_point = hit_p + index * 3;
        const float* normal = hit_n + index * 3;
        dst_origin[0] = hit_point[0];
        dst_origin[1] = hit_point[1];
        dst_origin[2] = hit_point[2];
        const float dot =
            src_direction[0] * normal[0] +
            src_direction[1] * normal[1] +
            src_direction[2] * normal[2];
        float rx = src_direction[0] - 2.0f * dot * normal[0];
        float ry = src_direction[1] - 2.0f * dot * normal[1];
        float rz = src_direction[2] - 2.0f * dot * normal[2];
        const float inv_len = rsqrtf(fmaxf(rx * rx + ry * ry + rz * rz, 1.0e-20f));
        dst_direction[0] = rx * inv_len;
        dst_direction[1] = ry * inv_len;
        dst_direction[2] = rz * inv_len;
    } else {
        dst_origin[0] = 0.0f;
        dst_origin[1] = 0.0f;
        dst_origin[2] = 0.0f;
        dst_direction[0] = 0.0f;
        dst_direction[1] = 0.0f;
        dst_direction[2] = 0.0f;
    }
    throughput_real[index] = is_valid ? light_throughput_real[index] * gain : 0.0f;
    throughput_imag[index] = is_valid ? light_throughput_imag[index] * gain : 0.0f;
    pdf_forward[index] = is_valid ? light_pdf_forward[index] : 0.0f;
    pdf_reverse[index] = is_valid ? light_pdf_reverse[index] : 0.0f;
    depth[index] = is_valid ? light_depth[index] + 1 : 0;
    component_mask[index] = is_valid ? (light_component_mask[index] | 2) : 0;
    primitive_id[index] = is_valid ? prim : -1;
    edge_id[index] = -1;
    tx_id[index] = is_valid ? light_tx_id[index] : -1;
    rx_id[index] = is_valid ? light_rx_id[index] : -1;
    grid_linear_id[index] = is_valid ? light_grid_linear_id[index] : -1;
    valid[index] = is_valid;
}

}  // namespace

std::vector<at::Tensor> cn_bdpt_empty_subpath_state_cuda(at::Tensor reference) {
    check_reference(reference);
    return allocate_subpath_state(reference, 0);
}

std::vector<at::Tensor> cn_bdpt_light_endpoint_subpath_state_cuda(
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor launch_tx_id,
    at::Tensor light_seed) {
    check_vec3_table(tx_positions, "tx_positions");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_flat_tensor(launch_tx_id, "launch_tx_id", at::kInt);
    check_flat_tensor(light_seed, "light_seed", at::kLong);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(light_seed.size(0) == launch_tx_id.size(0), "light_seed must match launch_tx_id");
    TORCH_CHECK(tx_power.get_device() == tx_positions.get_device(), "tx_power must share tx_positions device");
    TORCH_CHECK(launch_tx_id.get_device() == tx_positions.get_device(), "launch_tx_id must share tx_positions device");
    TORCH_CHECK(light_seed.get_device() == tx_positions.get_device(), "light_seed must share tx_positions device");
    auto state = allocate_subpath_state(tx_positions, launch_tx_id.size(0));
    const int64_t count = launch_tx_id.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        bdpt_light_endpoint_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            launch_tx_id.data_ptr<int>(),
            light_seed.data_ptr<int64_t>(),
            tx_positions.size(0),
            state[0].data_ptr<float>(),
            state[1].data_ptr<float>(),
            state[2].data_ptr<float>(),
            state[3].data_ptr<float>(),
            state[4].data_ptr<float>(),
            state[5].data_ptr<float>(),
            state[6].data_ptr<int>(),
            state[7].data_ptr<int>(),
            state[8].data_ptr<int>(),
            state[9].data_ptr<int>(),
            state[10].data_ptr<int>(),
            state[11].data_ptr<int>(),
            state[12].data_ptr<int>(),
            state[13].data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state;
}

std::vector<at::Tensor> cn_bdpt_sensor_endpoint_subpath_state_cuda(at::Tensor rx_positions) {
    check_vec3_table(rx_positions, "rx_positions");
    auto state = allocate_subpath_state(rx_positions, rx_positions.size(0));
    const int64_t count = rx_positions.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
        bdpt_sensor_endpoint_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            rx_positions.data_ptr<float>(),
            state[0].data_ptr<float>(),
            state[1].data_ptr<float>(),
            state[2].data_ptr<float>(),
            state[3].data_ptr<float>(),
            state[4].data_ptr<float>(),
            state[5].data_ptr<float>(),
            state[6].data_ptr<int>(),
            state[7].data_ptr<int>(),
            state[8].data_ptr<int>(),
            state[9].data_ptr<int>(),
            state[10].data_ptr<int>(),
            state[11].data_ptr<int>(),
            state[12].data_ptr<int>(),
            state[13].data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state;
}

std::vector<at::Tensor> cn_bdpt_reflected_light_subpath_state_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_throughput_imag,
    at::Tensor light_pdf_forward,
    at::Tensor light_pdf_reverse,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_rx_id,
    at::Tensor light_grid_linear_id,
    at::Tensor light_valid,
    at::Tensor hit_t,
    at::Tensor hit_p,
    at::Tensor hit_n,
    at::Tensor hit_global_prim_id,
    at::Tensor material_gain,
    at::Tensor material_valid) {
    check_vec3_table(light_origin, "light.origin");
    check_vec3_table(light_direction, "light.direction");
    check_flat_tensor(light_throughput_real, "light.throughput_real", at::kFloat);
    check_flat_tensor(light_throughput_imag, "light.throughput_imag", at::kFloat);
    check_flat_tensor(light_pdf_forward, "light.pdf_forward", at::kFloat);
    check_flat_tensor(light_pdf_reverse, "light.pdf_reverse", at::kFloat);
    check_flat_tensor(light_depth, "light.depth", at::kInt);
    check_flat_tensor(light_component_mask, "light.component_mask", at::kInt);
    check_flat_tensor(light_tx_id, "light.tx_id", at::kInt);
    check_flat_tensor(light_rx_id, "light.rx_id", at::kInt);
    check_flat_tensor(light_grid_linear_id, "light.grid_linear_id", at::kInt);
    check_flat_tensor(light_valid, "light.valid", at::kBool);
    check_flat_tensor(hit_t, "intersection.t", at::kFloat);
    check_vec3_table(hit_p, "intersection.p");
    check_vec3_table(hit_n, "intersection.n");
    check_flat_tensor(hit_global_prim_id, "intersection.global_prim_id", at::kInt);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    TORCH_CHECK(material_gain.size(0) == material_valid.size(0), "material_gain and material_valid must match");

    const int64_t count = light_origin.size(0);
    for (const auto& tensor : {
             light_direction,
             light_throughput_real,
             light_throughput_imag,
              light_pdf_forward,
              light_pdf_reverse,
              light_depth,
              light_component_mask,
              light_tx_id,
             light_rx_id,
             light_grid_linear_id,
             light_valid,
             hit_t,
              hit_p,
              hit_n,
              hit_global_prim_id,
          }) {
        TORCH_CHECK(tensor.size(0) == count, "reflected light subpath tensors must share batch size");
        TORCH_CHECK(tensor.get_device() == light_origin.get_device(), "reflected light subpath tensors must share device");
    }
    TORCH_CHECK(material_gain.get_device() == light_origin.get_device(), "material_gain must share light device");
    TORCH_CHECK(material_valid.get_device() == light_origin.get_device(), "material_valid must share light device");
    auto state = allocate_subpath_state(light_origin, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_reflected_light_subpaths_kernel<<<blocks, threads, 0, stream>>>(
            count,
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_throughput_imag.data_ptr<float>(),
            light_pdf_forward.data_ptr<float>(),
            light_pdf_reverse.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_component_mask.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_rx_id.data_ptr<int>(),
            light_grid_linear_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            hit_t.data_ptr<float>(),
            hit_p.data_ptr<float>(),
            hit_n.data_ptr<float>(),
            hit_global_prim_id.data_ptr<int>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
            material_gain.size(0),
            state[0].data_ptr<float>(),
            state[1].data_ptr<float>(),
            state[2].data_ptr<float>(),
            state[3].data_ptr<float>(),
            state[4].data_ptr<float>(),
            state[5].data_ptr<float>(),
            state[6].data_ptr<int>(),
            state[7].data_ptr<int>(),
            state[8].data_ptr<int>(),
            state[9].data_ptr<int>(),
            state[10].data_ptr<int>(),
            state[11].data_ptr<int>(),
            state[12].data_ptr<int>(),
            state[13].data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state;
}
