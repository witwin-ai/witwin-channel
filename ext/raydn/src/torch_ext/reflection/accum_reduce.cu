#include <raydn/reflection/accum_reduce.h>
#include <raydn/common/optix_context.h>
#include <raydn/reflection/accum_params.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace raydn {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_i32_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

struct AddReflAccumValue {
    __host__ __device__ ReflAccumStagedValue operator()(
        ReflAccumStagedValue x,
        ReflAccumStagedValue y) const {
        ReflAccumStagedValue out;
        out.a = make_float4(
            x.a.x + y.a.x,
            x.a.y + y.a.y,
            x.a.z + y.a.z,
            x.a.w + y.a.w);
        out.b = make_float4(
            x.b.x + y.b.x,
            x.b.y + y.b.y,
            x.b.z + y.b.z,
            x.b.w + y.b.w);
        return out;
    }
};

__global__ void scatter_refl_accum_reduced_kernel(
    const int *__restrict__ num_runs,
    const int *__restrict__ unique_cells,
    const ReflAccumStagedValue *__restrict__ reduced_values,
    float *__restrict__ out_power,
    float *__restrict__ out_field_x_re,
    float *__restrict__ out_field_x_im,
    float *__restrict__ out_field_y_re,
    float *__restrict__ out_field_y_im,
    float *__restrict__ out_field_z_re,
    float *__restrict__ out_field_z_im,
    int *__restrict__ out_reflection_count) {
    const int n = *num_runs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        const int cell = unique_cells[idx];
        if (cell < 0) {
            continue;
        }
        const ReflAccumStagedValue value = reduced_values[idx];
        if (out_power != nullptr) {
            atomicAdd(out_power + cell, value.a.x);
        }
        if (out_field_x_re != nullptr) {
            atomicAdd(out_field_x_re + cell, value.a.y);
        }
        if (out_field_x_im != nullptr) {
            atomicAdd(out_field_x_im + cell, value.a.z);
        }
        if (out_field_y_re != nullptr) {
            atomicAdd(out_field_y_re + cell, value.a.w);
        }
        if (out_field_y_im != nullptr) {
            atomicAdd(out_field_y_im + cell, value.b.x);
        }
        if (out_field_z_re != nullptr) {
            atomicAdd(out_field_z_re + cell, value.b.y);
        }
        if (out_field_z_im != nullptr) {
            atomicAdd(out_field_z_im + cell, value.b.z);
        }
        const int count = static_cast<int>(value.b.w + 0.5f);
        if (out_reflection_count != nullptr && count != 0) {
            atomicAdd(out_reflection_count, count);
        }
    }
}

} // namespace

void reduce_refl_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_reflection_count) {
    require_i32_count(sample_count, "reduce_refl_accum_staged_cuda(sample_count)");
    if (sample_count == 0) {
        return;
    }

    const int sample_count_i = static_cast<int>(sample_count);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cudaStream_t stream = torch_ctx.stream;
    at::TensorOptions key_options = stage_cell.options();
    at::TensorOptions value_options = stage_value.options().dtype(at::kFloat);
    at::TensorOptions byte_options = stage_cell.options().dtype(at::kByte);

    at::Tensor sorted_cells = at::empty({sample_count}, key_options);
    at::Tensor sorted_values = at::empty({sample_count, 8}, value_options);
    auto *values_in = reinterpret_cast<ReflAccumStagedValue *>(stage_value.data_ptr<float>());
    auto *values_sorted =
        reinterpret_cast<ReflAccumStagedValue *>(sorted_values.data_ptr<float>());

    size_t sort_temp_bytes = 0;
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(refl accum size)");
    at::Tensor sort_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(sort_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_temp_bytes,
            stage_cell.data_ptr<int>(),
            sorted_cells.data_ptr<int>(),
            values_in,
            values_sorted,
            sample_count_i,
            0,
            sizeof(int) * 8,
            stream),
        "cub::DeviceRadixSort::SortPairs(refl accum)");

    at::Tensor unique_cells = at::empty({sample_count}, key_options);
    at::Tensor reduced_values = at::empty({sample_count, 8}, value_options);
    at::Tensor num_runs = at::empty({1}, key_options);
    auto *reduced_values_ptr =
        reinterpret_cast<ReflAccumStagedValue *>(reduced_values.data_ptr<float>());

    size_t reduce_temp_bytes = 0;
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            nullptr,
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddReflAccumValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(refl accum size)");
    at::Tensor reduce_temp = at::empty(
        {std::max<int64_t>(1, static_cast<int64_t>(reduce_temp_bytes))},
        byte_options);
    cuda_check(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp.data_ptr<uint8_t>(),
            reduce_temp_bytes,
            sorted_cells.data_ptr<int>(),
            unique_cells.data_ptr<int>(),
            values_sorted,
            reduced_values_ptr,
            num_runs.data_ptr<int>(),
            AddReflAccumValue{},
            sample_count_i,
            stream),
        "cub::DeviceReduce::ReduceByKey(refl accum)");

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((sample_count + block_size - 1) / block_size);
    scatter_refl_accum_reduced_kernel<<<block_count, block_size, 0, stream>>>(
        num_runs.data_ptr<int>(),
        unique_cells.data_ptr<int>(),
        reduced_values_ptr,
        out_power.data_ptr<float>(),
        out_field_x_re.data_ptr<float>(),
        out_field_x_im.data_ptr<float>(),
        out_field_y_re.data_ptr<float>(),
        out_field_y_im.data_ptr<float>(),
        out_field_z_re.data_ptr<float>(),
        out_field_z_im.data_ptr<float>(),
        out_reflection_count.data_ptr<int>());
    cuda_check(cudaGetLastError(), "scatter_refl_accum_reduced_kernel");
}

} // namespace raydn
