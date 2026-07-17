#include "bdpt_connect_common.cuh"

namespace {

__global__ void bdpt_accumulate_connection_samples_double_kernel(
    int64_t count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    double* path_gain,
    double* los,
    double* reflection,
    double* diffraction,
    double* transmission,
    double* scattering) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count || !bdpt_component_accumulable(component)) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double value = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
    atomicAdd(path_gain + out_index, value);
    if (component == kComponentLos) {
        atomicAdd(los + out_index, value);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection + out_index, value);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction + out_index, value);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission + out_index, value);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering + out_index, value);
    }
}

__global__ void bdpt_compact_valid_connection_indices_kernel(
    int64_t count,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    int* compact_count,
    int* compact_indices) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count || !bdpt_component_accumulable(component)) {
        return;
    }
    const int slot = atomicAdd(compact_count, 1);
    compact_indices[slot] = static_cast<int>(index);
}

__global__ void bdpt_accumulate_connection_samples_compacted_kernel(
    int64_t capacity,
    const int* compact_count,
    const int* compact_indices,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    int64_t rx_count,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t compact_linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact_linear >= capacity || compact_linear >= static_cast<int64_t>(compact_count[0])) {
        return;
    }
    const int index = compact_indices[compact_linear];
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const float value = contribution[index] * mis_weight[index];
    atomicAdd(path_gain + out_index, value);
    if (component == kComponentLos) {
        atomicAdd(los + out_index, value);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection + out_index, value);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction + out_index, value);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission + out_index, value);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering + out_index, value);
    }
}

__global__ void bdpt_accumulate_connection_samples_staged_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_index >= out_count) {
        return;
    }
    const int out_tx = static_cast<int>(out_index / rx_count);
    const int out_rx = static_cast<int>(out_index - static_cast<int64_t>(out_tx) * rx_count);
    double path_sum = 0.0;
    double los_sum = 0.0;
    double reflection_sum = 0.0;
    double diffraction_sum = 0.0;
    double transmission_sum = 0.0;
    double scattering_sum = 0.0;
    for (int64_t index = 0; index < sample_count; ++index) {
        if (!valid[index] || tx_id[index] != out_tx || rx_id[index] != out_rx) {
            continue;
        }
        const int component = component_id[index];
        if (!bdpt_component_accumulable(component)) {
            continue;
        }
        const double value = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
        path_sum += value;
        if (component == kComponentLos) {
            los_sum += value;
        } else if (component == kComponentReflection) {
            reflection_sum += value;
        } else if (component == kComponentDiffraction) {
            diffraction_sum += value;
        } else if (component == kComponentTransmission) {
            transmission_sum += value;
        } else if (component == kComponentScattering) {
            scattering_sum += value;
        }
    }
    path_gain[out_index] = static_cast<float>(path_sum);
    los[out_index] = static_cast<float>(los_sum);
    reflection[out_index] = static_cast<float>(reflection_sum);
    diffraction[out_index] = static_cast<float>(diffraction_sum);
    transmission[out_index] = static_cast<float>(transmission_sum);
    scattering[out_index] = static_cast<float>(scattering_sum);
}

__global__ void bdpt_cast_connection_accumulation_kernel(
    int64_t count,
    const double* path_gain_sum,
    const double* los_sum,
    const double* reflection_sum,
    const double* diffraction_sum,
    const double* transmission_sum,
    const double* scattering_sum,
    float* path_gain,
    float* los,
    float* reflection,
    float* diffraction,
    float* transmission,
    float* scattering) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    path_gain[index] = static_cast<float>(path_gain_sum[index]);
    los[index] = static_cast<float>(los_sum[index]);
    reflection[index] = static_cast<float>(reflection_sum[index]);
    diffraction[index] = static_cast<float>(diffraction_sum[index]);
    transmission[index] = static_cast<float>(transmission_sum[index]);
    scattering[index] = static_cast<float>(scattering_sum[index]);
}

__global__ void bdpt_connection_variance_accum_double_kernel(
    int64_t count,
    const float* contribution,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const bool* valid,
    int64_t rx_count,
    double samples_per_tx,
    double* sum,
    double* sum_square_unweighted,
    int* sample_count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    if (tx < 0 || rx < 0 || rx >= rx_count) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double weighted = static_cast<double>(contribution[index]) * static_cast<double>(mis_weight[index]);
    const double unweighted = weighted * samples_per_tx;
    atomicAdd(sum + out_index, weighted);
    atomicAdd(sum_square_unweighted + out_index, unweighted * unweighted);
    atomicAdd(sample_count + out_index, 1);
}

__global__ void bdpt_connection_variance_finalize_double_kernel(
    int64_t count,
    const double* sum,
    const double* sum_square_unweighted,
    const int* sample_count,
    double samples_per_tx,
    float* variance) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int n = sample_count[index];
    if (n <= 0 || samples_per_tx <= 0.0) {
        variance[index] = 0.0f;
        return;
    }
    const double mean = sum[index];
    const double ex2 = sum_square_unweighted[index] / samples_per_tx;
    const double variance_value = fmax(ex2 - mean * mean, 0.0) / samples_per_tx;
    variance[index] = variance_value <= 1.0e-30 ? 0.0f : static_cast<float>(variance_value);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(tx_id.sizes() == contribution.sizes(), "tx_id must match contribution");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(rx_id.sizes() == contribution.sizes(), "rx_id must match contribution");
    TORCH_CHECK(component_id.sizes() == contribution.sizes(), "component_id must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    check_same_device(tx_id, contribution, "tx_id");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(rx_id, contribution, "rx_id");
    check_same_device(component_id, contribution, "component_id");
    check_same_device(valid, contribution, "valid");
    TORCH_CHECK(accumulation_strategy >= 0 && accumulation_strategy <= 2, "accumulation_strategy must be 0, 1, or 2");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto path_gain = at::empty({tx_count, rx_count}, float_options);
    auto los = at::empty({tx_count, rx_count}, float_options);
    auto reflection = at::empty({tx_count, rx_count}, float_options);
    auto diffraction = at::empty({tx_count, rx_count}, float_options);
    auto transmission = at::empty({tx_count, rx_count}, float_options);
    auto scattering = at::empty({tx_count, rx_count}, float_options);
    const int64_t count = contribution.numel();
    const int64_t out_count = tx_count * rx_count;
    if (accumulation_strategy == 1) {
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_accumulate_connection_samples_staged_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                contribution.data_ptr<float>(),
                mis_weight.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {path_gain, los, reflection, diffraction, transmission, scattering};
    }
    if (accumulation_strategy == 2) {
        zero_float_tensor(path_gain);
        zero_float_tensor(los);
        zero_float_tensor(reflection);
        zero_float_tensor(diffraction);
        zero_float_tensor(transmission);
        zero_float_tensor(scattering);
        auto int_options = tx_id.options().dtype(at::kInt);
        auto compact_count = at::empty({}, int_options);
        auto compact_indices = at::empty({count}, int_options);
        zero_int_tensor(compact_count);
        if (count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_compact_valid_connection_indices_kernel<<<blocks, threads, 0, stream>>>(
                count,
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                tx_count,
                rx_count,
                compact_count.data_ptr<int>(),
                compact_indices.data_ptr<int>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            bdpt_accumulate_connection_samples_compacted_kernel<<<blocks, threads, 0, stream>>>(
                count,
                compact_count.data_ptr<int>(),
                compact_indices.data_ptr<int>(),
                contribution.data_ptr<float>(),
                mis_weight.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                rx_count,
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {path_gain, los, reflection, diffraction, transmission, scattering};
    }
    auto double_options = contribution.options().dtype(at::kDouble);
    auto path_gain_sum = at::empty({tx_count, rx_count}, double_options);
    auto los_sum = at::empty({tx_count, rx_count}, double_options);
    auto reflection_sum = at::empty({tx_count, rx_count}, double_options);
    auto diffraction_sum = at::empty({tx_count, rx_count}, double_options);
    auto transmission_sum = at::empty({tx_count, rx_count}, double_options);
    auto scattering_sum = at::empty({tx_count, rx_count}, double_options);
    zero_double_tensor(path_gain_sum);
    zero_double_tensor(los_sum);
    zero_double_tensor(reflection_sum);
    zero_double_tensor(diffraction_sum);
    zero_double_tensor(transmission_sum);
    zero_double_tensor(scattering_sum);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_accumulate_connection_samples_double_kernel<<<blocks, threads, 0, stream>>>(
            count,
            contribution.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_count,
            rx_count,
            path_gain_sum.data_ptr<double>(),
            los_sum.data_ptr<double>(),
            reflection_sum.data_ptr<double>(),
            diffraction_sum.data_ptr<double>(),
            transmission_sum.data_ptr<double>(),
            scattering_sum.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (out_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((out_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_cast_connection_accumulation_kernel<<<blocks, threads, 0, stream>>>(
            out_count,
            path_gain_sum.data_ptr<double>(),
            los_sum.data_ptr<double>(),
            reflection_sum.data_ptr<double>(),
            diffraction_sum.data_ptr<double>(),
            transmission_sum.data_ptr<double>(),
            scattering_sum.data_ptr<double>(),
            path_gain.data_ptr<float>(),
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            transmission.data_ptr<float>(),
            scattering.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {path_gain, los, reflection, diffraction, transmission, scattering};
}

at::Tensor cn_bdpt_connection_variance_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t samples_per_tx) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(tx_id.sizes() == contribution.sizes(), "tx_id must match contribution");
    TORCH_CHECK(rx_id.sizes() == contribution.sizes(), "rx_id must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(tx_id, contribution, "tx_id");
    check_same_device(rx_id, contribution, "rx_id");
    check_same_device(valid, contribution, "valid");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto double_options = contribution.options().dtype(at::kDouble);
    auto int_options = contribution.options().dtype(at::kInt);
    auto sum = at::empty({tx_count, rx_count}, double_options);
    auto sum_square_unweighted = at::empty({tx_count, rx_count}, double_options);
    auto sample_count = at::empty({tx_count, rx_count}, int_options);
    auto variance = at::empty({tx_count, rx_count}, float_options);
    zero_double_tensor(sum);
    zero_double_tensor(sum_square_unweighted);
    zero_int_tensor(sample_count);
    const int64_t in_count = contribution.numel();
    if (in_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((in_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_connection_variance_accum_double_kernel<<<blocks, threads, 0, stream>>>(
            in_count,
            contribution.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            rx_count,
            static_cast<double>(samples_per_tx),
            sum.data_ptr<double>(),
            sum_square_unweighted.data_ptr<double>(),
            sample_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t out_count = tx_count * rx_count;
    if (out_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((out_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_connection_variance_finalize_double_kernel<<<blocks, threads, 0, stream>>>(
            out_count,
            sum.data_ptr<double>(),
            sum_square_unweighted.data_ptr<double>(),
            sample_count.data_ptr<int>(),
            static_cast<double>(samples_per_tx),
            variance.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return variance;
}
