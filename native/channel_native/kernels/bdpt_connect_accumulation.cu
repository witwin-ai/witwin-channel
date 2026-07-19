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

// ADR-019 coherent combine (opt-in, DEFAULT OFF). Sum the complex per-row
// projected field coefficient into per-(tx, rx, component) phasor bins, then
// finalize |sum|^2. Coherent-eligible rows are the enumerated delta/UTD
// discrete connections (los / reflection / diffraction / coupled->diffraction)
// which carry unit forward/reverse mass, so the phasor is summed with UNIT
// weight (mis_weight is identically 1 for those rows) and the estimate is
// MIS-invariant by construction. This path is only reached when
// combine_domain == 1; combine_domain == 0 keeps the power-domain incoherent
// accumulation bit-identical and never touches the coefficient buffers.
__global__ void bdpt_accumulate_connection_samples_coherent_kernel(
    int64_t count,
    const float* coeff_real,
    const float* coeff_imag,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    double* los_real,
    double* los_imag,
    double* reflection_real,
    double* reflection_imag,
    double* diffraction_real,
    double* diffraction_imag,
    double* transmission_real,
    double* transmission_imag,
    double* scattering_real,
    double* scattering_imag) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int tx = tx_id[index];
    const int rx = rx_id[index];
    const int component = component_id[index];
    if (tx < 0 || tx >= tx_count || rx < 0 || rx >= rx_count ||
        !bdpt_component_accumulable(component)) {
        return;
    }
    const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
    const double re = static_cast<double>(coeff_real[index]);
    const double im = static_cast<double>(coeff_imag[index]);
    if (component == kComponentLos) {
        atomicAdd(los_real + out_index, re);
        atomicAdd(los_imag + out_index, im);
    } else if (component == kComponentReflection) {
        atomicAdd(reflection_real + out_index, re);
        atomicAdd(reflection_imag + out_index, im);
    } else if (component == kComponentDiffraction) {
        atomicAdd(diffraction_real + out_index, re);
        atomicAdd(diffraction_imag + out_index, im);
    } else if (component == kComponentTransmission) {
        atomicAdd(transmission_real + out_index, re);
        atomicAdd(transmission_imag + out_index, im);
    } else if (component == kComponentScattering) {
        atomicAdd(scattering_real + out_index, re);
        atomicAdd(scattering_imag + out_index, im);
    }
}

__global__ void bdpt_finalize_coherent_accumulation_kernel(
    int64_t count,
    const double* los_real,
    const double* los_imag,
    const double* reflection_real,
    const double* reflection_imag,
    const double* diffraction_real,
    const double* diffraction_imag,
    const double* transmission_real,
    const double* transmission_imag,
    const double* scattering_real,
    const double* scattering_imag,
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
    const double los_power =
        los_real[index] * los_real[index] + los_imag[index] * los_imag[index];
    const double reflection_power =
        reflection_real[index] * reflection_real[index] +
        reflection_imag[index] * reflection_imag[index];
    const double diffraction_power =
        diffraction_real[index] * diffraction_real[index] +
        diffraction_imag[index] * diffraction_imag[index];
    const double transmission_power =
        transmission_real[index] * transmission_real[index] +
        transmission_imag[index] * transmission_imag[index];
    const double scattering_power =
        scattering_real[index] * scattering_real[index] +
        scattering_imag[index] * scattering_imag[index];
    // Paths within one component combine coherently; components combine
    // incoherently into path_gain (matches the deterministic per-component
    // coherent power the ADR-019 acceptance gate compares against).
    los[index] = static_cast<float>(los_power);
    reflection[index] = static_cast<float>(reflection_power);
    diffraction[index] = static_cast<float>(diffraction_power);
    transmission[index] = static_cast<float>(transmission_power);
    scattering[index] = static_cast<float>(scattering_power);
    path_gain[index] = static_cast<float>(
        los_power + reflection_power + diffraction_power + transmission_power +
        scattering_power);
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

// ADR-022 ruling 6.4: the coherent forward (combine_domain == 1) returns the
// per-component complex bin-sum buffers (S_b) as ten extra non-differentiable
// outputs so the accumulate backward reads them directly instead of re-reducing
// the atomic-double phasor sum. combine_domain == 0 returns those ten trailing
// slots undefined (empty), keeping the power-domain output byte-identical.
std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_bdpt_accumulate_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor coeff_real,
    at::Tensor coeff_imag,
    int64_t tx_count,
    int64_t rx_count,
    int64_t accumulation_strategy,
    int64_t combine_domain) {
    // Ten trailing bin-sum outputs; only assigned on the coherent branch.
    at::Tensor bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
        bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
        bin_transmission_im, bin_scattering_re, bin_scattering_im;
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
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 (power) or 1 (coherent)");
    auto float_options = contribution.options().dtype(at::kFloat);
    auto path_gain = at::empty({tx_count, rx_count}, float_options);
    auto los = at::empty({tx_count, rx_count}, float_options);
    auto reflection = at::empty({tx_count, rx_count}, float_options);
    auto diffraction = at::empty({tx_count, rx_count}, float_options);
    auto transmission = at::empty({tx_count, rx_count}, float_options);
    auto scattering = at::empty({tx_count, rx_count}, float_options);
    const int64_t count = contribution.numel();
    const int64_t out_count = tx_count * rx_count;
    if (combine_domain == 1) {
        // ADR-019 coherent combine. accumulation_strategy is a power-domain
        // reduction perf axis and stays orthogonal: the coherent phasor sum
        // always uses the atomic-double reduction regardless of its value.
        check_float_cuda(coeff_real, "coeff_real", 1);
        check_float_cuda(coeff_imag, "coeff_imag", 1);
        TORCH_CHECK(coeff_real.sizes() == contribution.sizes(), "coeff_real must match contribution");
        TORCH_CHECK(coeff_imag.sizes() == contribution.sizes(), "coeff_imag must match contribution");
        check_same_device(coeff_real, contribution, "coeff_real");
        check_same_device(coeff_imag, contribution, "coeff_imag");
        auto double_options = contribution.options().dtype(at::kDouble);
        auto los_re = at::empty({tx_count, rx_count}, double_options);
        auto los_im = at::empty({tx_count, rx_count}, double_options);
        auto reflection_re = at::empty({tx_count, rx_count}, double_options);
        auto reflection_im = at::empty({tx_count, rx_count}, double_options);
        auto diffraction_re = at::empty({tx_count, rx_count}, double_options);
        auto diffraction_im = at::empty({tx_count, rx_count}, double_options);
        auto transmission_re = at::empty({tx_count, rx_count}, double_options);
        auto transmission_im = at::empty({tx_count, rx_count}, double_options);
        auto scattering_re = at::empty({tx_count, rx_count}, double_options);
        auto scattering_im = at::empty({tx_count, rx_count}, double_options);
        zero_double_tensor(los_re);
        zero_double_tensor(los_im);
        zero_double_tensor(reflection_re);
        zero_double_tensor(reflection_im);
        zero_double_tensor(diffraction_re);
        zero_double_tensor(diffraction_im);
        zero_double_tensor(transmission_re);
        zero_double_tensor(transmission_im);
        zero_double_tensor(scattering_re);
        zero_double_tensor(scattering_im);
        if (count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_accumulate_connection_samples_coherent_kernel<<<blocks, threads, 0, stream>>>(
                count,
                coeff_real.data_ptr<float>(),
                coeff_imag.data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                tx_count,
                rx_count,
                los_re.data_ptr<double>(),
                los_im.data_ptr<double>(),
                reflection_re.data_ptr<double>(),
                reflection_im.data_ptr<double>(),
                diffraction_re.data_ptr<double>(),
                diffraction_im.data_ptr<double>(),
                transmission_re.data_ptr<double>(),
                transmission_im.data_ptr<double>(),
                scattering_re.data_ptr<double>(),
                scattering_im.data_ptr<double>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
            bdpt_finalize_coherent_accumulation_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                los_re.data_ptr<double>(),
                los_im.data_ptr<double>(),
                reflection_re.data_ptr<double>(),
                reflection_im.data_ptr<double>(),
                diffraction_re.data_ptr<double>(),
                diffraction_im.data_ptr<double>(),
                transmission_re.data_ptr<double>(),
                transmission_im.data_ptr<double>(),
                scattering_re.data_ptr<double>(),
                scattering_im.data_ptr<double>(),
                path_gain.data_ptr<float>(),
                los.data_ptr<float>(),
                reflection.data_ptr<float>(),
                diffraction.data_ptr<float>(),
                transmission.data_ptr<float>(),
                scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        // Retain the phasor bin sums S_b for the coherent backward/jvp (ruling
        // 6.4): no in-backward re-reduction of the atomic-double accumulation.
        bin_los_re = los_re;
        bin_los_im = los_im;
        bin_reflection_re = reflection_re;
        bin_reflection_im = reflection_im;
        bin_diffraction_re = diffraction_re;
        bin_diffraction_im = diffraction_im;
        bin_transmission_re = transmission_re;
        bin_transmission_im = transmission_im;
        bin_scattering_re = scattering_re;
        bin_scattering_im = scattering_im;
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
    }
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
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
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
        return {path_gain, los, reflection, diffraction, transmission, scattering,
                bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
                bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
                bin_transmission_im, bin_scattering_re, bin_scattering_im};
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
    return {path_gain, los, reflection, diffraction, transmission, scattering,
            bin_los_re, bin_los_im, bin_reflection_re, bin_reflection_im,
            bin_diffraction_re, bin_diffraction_im, bin_transmission_re,
            bin_transmission_im, bin_scattering_re, bin_scattering_im};
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

// ===========================================================================
// ADR-022 6.4: bdpt_accumulate_connection_samples backward + jvp companions.
//
// The accumulate op is linear per component; MIS weights, the connection
// topology (tx_id/rx_id/component_id/valid), and combine_domain are frozen.
//   * Power domain  M[b] = sum_r contribution_r * mis_r
//         backward: grad_contribution_r = mis_r * (grad_path_gain[b] +
//                   grad_<component_r>[b])   (a deterministic gather)
//   * Coherent      P_c[b] = |S_c[b]|^2, path_gain[b] = sum_c P_c[b]
//         backward: grad_coeff_r = 2 * (grad_<c>[b] + grad_path_gain[b]) * S_c[b]
//                   reading the forward-retained bin sums S_c (ruling 6.4).
// The forward's per-component phasor bins are atomic-double (perf axis); the
// JVP recomputes the tangent bin sums in fixed order so it stays deterministic
// with no float atomics (the primal/JVP determinism rule).
// ===========================================================================

namespace {

__device__ __forceinline__ const float* bdpt_component_matrix(
    int component,
    const float* los,
    const float* reflection,
    const float* diffraction,
    const float* transmission,
    const float* scattering) {
    if (component == kComponentLos) return los;
    if (component == kComponentReflection) return reflection;
    if (component == kComponentDiffraction) return diffraction;
    if (component == kComponentTransmission) return transmission;
    if (component == kComponentScattering) return scattering;
    return nullptr;
}

__global__ void bdpt_accumulate_power_backward_kernel(
    int64_t count,
    const float* mis_weight,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    const float* grad_path_gain,
    const float* grad_los,
    const float* grad_reflection,
    const float* grad_diffraction,
    const float* grad_transmission,
    const float* grad_scattering,
    float* grad_contribution) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float g = 0.0f;
    if (valid[index]) {
        const int tx = tx_id[index];
        const int rx = rx_id[index];
        const int component = component_id[index];
        if (tx >= 0 && tx < tx_count && rx >= 0 && rx < rx_count &&
            bdpt_component_accumulable(component)) {
            const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
            float grad = grad_path_gain != nullptr ? grad_path_gain[out_index] : 0.0f;
            const float* comp = bdpt_component_matrix(
                component, grad_los, grad_reflection, grad_diffraction,
                grad_transmission, grad_scattering);
            if (comp != nullptr) {
                grad += comp[out_index];
            }
            g = mis_weight[index] * grad;
        }
    }
    grad_contribution[index] = g;
}

__global__ void bdpt_accumulate_coherent_backward_kernel(
    int64_t count,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t tx_count,
    int64_t rx_count,
    const float* grad_path_gain,
    const float* grad_los,
    const float* grad_reflection,
    const float* grad_diffraction,
    const float* grad_transmission,
    const float* grad_scattering,
    const double* los_re,
    const double* los_im,
    const double* reflection_re,
    const double* reflection_im,
    const double* diffraction_re,
    const double* diffraction_im,
    const double* transmission_re,
    const double* transmission_im,
    const double* scattering_re,
    const double* scattering_im,
    float* grad_coeff_real,
    float* grad_coeff_imag) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float gr = 0.0f;
    float gi = 0.0f;
    if (valid[index]) {
        const int tx = tx_id[index];
        const int rx = rx_id[index];
        const int component = component_id[index];
        if (tx >= 0 && tx < tx_count && rx >= 0 && rx < rx_count &&
            bdpt_component_accumulable(component)) {
            const int64_t out_index = static_cast<int64_t>(tx) * rx_count + rx;
            const float grad_pg =
                grad_path_gain != nullptr ? grad_path_gain[out_index] : 0.0f;
            const float* comp = bdpt_component_matrix(
                component, grad_los, grad_reflection, grad_diffraction,
                grad_transmission, grad_scattering);
            const float grad_comp = comp != nullptr ? comp[out_index] : 0.0f;
            const float g_power = grad_pg + grad_comp;
            const double* bin_re = nullptr;
            const double* bin_im = nullptr;
            if (component == kComponentLos) {
                bin_re = los_re;
                bin_im = los_im;
            } else if (component == kComponentReflection) {
                bin_re = reflection_re;
                bin_im = reflection_im;
            } else if (component == kComponentDiffraction) {
                bin_re = diffraction_re;
                bin_im = diffraction_im;
            } else if (component == kComponentTransmission) {
                bin_re = transmission_re;
                bin_im = transmission_im;
            } else if (component == kComponentScattering) {
                bin_re = scattering_re;
                bin_im = scattering_im;
            }
            if (bin_re != nullptr) {
                const float s_re = static_cast<float>(bin_re[out_index]);
                const float s_im = static_cast<float>(bin_im[out_index]);
                gr = 2.0f * g_power * s_re;
                gi = 2.0f * g_power * s_im;
            }
        }
    }
    grad_coeff_real[index] = gr;
    grad_coeff_imag[index] = gi;
}

__global__ void bdpt_accumulate_power_jvp_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* mis_weight,
    const float* tangent_contribution,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
    float* t_path_gain,
    float* t_los,
    float* t_reflection,
    float* t_diffraction,
    float* t_transmission,
    float* t_scattering) {
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
        const double value = static_cast<double>(mis_weight[index]) *
            static_cast<double>(tangent_contribution[index]);
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
    t_path_gain[out_index] = static_cast<float>(path_sum);
    t_los[out_index] = static_cast<float>(los_sum);
    t_reflection[out_index] = static_cast<float>(reflection_sum);
    t_diffraction[out_index] = static_cast<float>(diffraction_sum);
    t_transmission[out_index] = static_cast<float>(transmission_sum);
    t_scattering[out_index] = static_cast<float>(scattering_sum);
}

__global__ void bdpt_accumulate_coherent_jvp_kernel(
    int64_t out_count,
    int64_t sample_count,
    const float* tangent_coeff_real,
    const float* tangent_coeff_imag,
    const int* tx_id,
    const int* rx_id,
    const int* component_id,
    const bool* valid,
    int64_t rx_count,
    const double* los_re,
    const double* los_im,
    const double* reflection_re,
    const double* reflection_im,
    const double* diffraction_re,
    const double* diffraction_im,
    const double* transmission_re,
    const double* transmission_im,
    const double* scattering_re,
    const double* scattering_im,
    float* t_path_gain,
    float* t_los,
    float* t_reflection,
    float* t_diffraction,
    float* t_transmission,
    float* t_scattering) {
    int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_index >= out_count) {
        return;
    }
    const int out_tx = static_cast<int>(out_index / rx_count);
    const int out_rx = static_cast<int>(out_index - static_cast<int64_t>(out_tx) * rx_count);
    // Fixed-order tangent bin sums t_S_c per component (deterministic, no atomics).
    double t_los_re = 0.0, t_los_im = 0.0;
    double t_reflection_re = 0.0, t_reflection_im = 0.0;
    double t_diffraction_re = 0.0, t_diffraction_im = 0.0;
    double t_transmission_re = 0.0, t_transmission_im = 0.0;
    double t_scattering_re = 0.0, t_scattering_im = 0.0;
    for (int64_t index = 0; index < sample_count; ++index) {
        if (!valid[index] || tx_id[index] != out_tx || rx_id[index] != out_rx) {
            continue;
        }
        const int component = component_id[index];
        if (!bdpt_component_accumulable(component)) {
            continue;
        }
        const double tr = static_cast<double>(tangent_coeff_real[index]);
        const double ti = static_cast<double>(tangent_coeff_imag[index]);
        if (component == kComponentLos) {
            t_los_re += tr;
            t_los_im += ti;
        } else if (component == kComponentReflection) {
            t_reflection_re += tr;
            t_reflection_im += ti;
        } else if (component == kComponentDiffraction) {
            t_diffraction_re += tr;
            t_diffraction_im += ti;
        } else if (component == kComponentTransmission) {
            t_transmission_re += tr;
            t_transmission_im += ti;
        } else if (component == kComponentScattering) {
            t_scattering_re += tr;
            t_scattering_im += ti;
        }
    }
    // t_P_c = 2 Re(conj(S_c) t_S_c); path_gain tangent sums the component powers.
    const double tp_los =
        2.0 * (los_re[out_index] * t_los_re + los_im[out_index] * t_los_im);
    const double tp_reflection = 2.0 *
        (reflection_re[out_index] * t_reflection_re +
         reflection_im[out_index] * t_reflection_im);
    const double tp_diffraction = 2.0 *
        (diffraction_re[out_index] * t_diffraction_re +
         diffraction_im[out_index] * t_diffraction_im);
    const double tp_transmission = 2.0 *
        (transmission_re[out_index] * t_transmission_re +
         transmission_im[out_index] * t_transmission_im);
    const double tp_scattering = 2.0 *
        (scattering_re[out_index] * t_scattering_re +
         scattering_im[out_index] * t_scattering_im);
    t_los[out_index] = static_cast<float>(tp_los);
    t_reflection[out_index] = static_cast<float>(tp_reflection);
    t_diffraction[out_index] = static_cast<float>(tp_diffraction);
    t_transmission[out_index] = static_cast<float>(tp_transmission);
    t_scattering[out_index] = static_cast<float>(tp_scattering);
    t_path_gain[out_index] = static_cast<float>(
        tp_los + tp_reflection + tp_diffraction + tp_transmission + tp_scattering);
}

// Optional-tensor helper: None -> nullptr, else validate and expose contiguous
// storage. Shape/dtype/device are enforced against the reference structure.
const at::Tensor* accumulate_optional(
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
const T* accumulate_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

}  // namespace

pybind11::dict cn_bdpt_accumulate_connection_samples_backward_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object grad_path_gain,
    pybind11::object grad_los,
    pybind11::object grad_reflection,
    pybind11::object grad_diffraction,
    pybind11::object grad_transmission,
    pybind11::object grad_scattering,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im,
    bool need_grad_contribution,
    bool need_grad_coeff) {
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 or 1");
    TORCH_CHECK(tx_id.sizes() == mis_weight.sizes(), "tx_id must match mis_weight");
    TORCH_CHECK(rx_id.sizes() == mis_weight.sizes(), "rx_id must match mis_weight");
    TORCH_CHECK(component_id.sizes() == mis_weight.sizes(), "component_id must match mis_weight");
    TORCH_CHECK(valid.sizes() == mis_weight.sizes(), "valid must match mis_weight");
    check_same_device(tx_id, mis_weight, "tx_id");
    check_same_device(rx_id, mis_weight, "rx_id");
    check_same_device(component_id, mis_weight, "component_id");
    check_same_device(valid, mis_weight, "valid");
    const int64_t count = mis_weight.numel();
    const std::vector<int64_t> matrix_shape = {tx_count, rx_count};

    at::Tensor gpg_s, glos_s, gref_s, gdif_s, gtra_s, gsca_s;
    const at::Tensor* gpg = accumulate_optional(
        std::move(grad_path_gain), gpg_s, "grad_path_gain", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* glos = accumulate_optional(
        std::move(grad_los), glos_s, "grad_los", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gref = accumulate_optional(
        std::move(grad_reflection), gref_s, "grad_reflection", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gdif = accumulate_optional(
        std::move(grad_diffraction), gdif_s, "grad_diffraction", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gtra = accumulate_optional(
        std::move(grad_transmission), gtra_s, "grad_transmission", at::kFloat, matrix_shape, mis_weight);
    const at::Tensor* gsca = accumulate_optional(
        std::move(grad_scattering), gsca_s, "grad_scattering", at::kFloat, matrix_shape, mis_weight);

    at::Tensor grad_contribution;
    at::Tensor grad_coeff_real;
    at::Tensor grad_coeff_imag;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mis_weight.get_device()).stream();
    if (combine_domain == 0) {
        if (need_grad_contribution) {
            grad_contribution = at::empty({count}, mis_weight.options());
            if (count > 0) {
                constexpr int threads = 256;
                int blocks = static_cast<int>((count + threads - 1) / threads);
                bdpt_accumulate_power_backward_kernel<<<blocks, threads, 0, stream>>>(
                    count,
                    mis_weight.data_ptr<float>(),
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    valid.data_ptr<bool>(),
                    tx_count,
                    rx_count,
                    accumulate_ptr<float>(gpg),
                    accumulate_ptr<float>(glos),
                    accumulate_ptr<float>(gref),
                    accumulate_ptr<float>(gdif),
                    accumulate_ptr<float>(gtra),
                    accumulate_ptr<float>(gsca),
                    grad_contribution.data_ptr<float>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    } else {
        at::Tensor lre_s, lim_s, rre_s, rim_s, dre_s, dim_s, tre_s, tim_s, sre_s, sim_s;
        const at::Tensor* lre = accumulate_optional(
            std::move(los_re), lre_s, "los_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* lim = accumulate_optional(
            std::move(los_im), lim_s, "los_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rre = accumulate_optional(
            std::move(reflection_re), rre_s, "reflection_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rim = accumulate_optional(
            std::move(reflection_im), rim_s, "reflection_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dre = accumulate_optional(
            std::move(diffraction_re), dre_s, "diffraction_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dim = accumulate_optional(
            std::move(diffraction_im), dim_s, "diffraction_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tre = accumulate_optional(
            std::move(transmission_re), tre_s, "transmission_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tim = accumulate_optional(
            std::move(transmission_im), tim_s, "transmission_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sre = accumulate_optional(
            std::move(scattering_re), sre_s, "scattering_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sim = accumulate_optional(
            std::move(scattering_im), sim_s, "scattering_im", at::kDouble, matrix_shape, mis_weight);
        TORCH_CHECK(
            lre && lim && rre && rim && dre && dim && tre && tim && sre && sim,
            "coherent accumulate backward requires all ten bin-sum buffers");
        if (need_grad_coeff) {
            grad_coeff_real = at::empty({count}, mis_weight.options());
            grad_coeff_imag = at::empty({count}, mis_weight.options());
            if (count > 0) {
                constexpr int threads = 256;
                int blocks = static_cast<int>((count + threads - 1) / threads);
                bdpt_accumulate_coherent_backward_kernel<<<blocks, threads, 0, stream>>>(
                    count,
                    tx_id.data_ptr<int>(),
                    rx_id.data_ptr<int>(),
                    component_id.data_ptr<int>(),
                    valid.data_ptr<bool>(),
                    tx_count,
                    rx_count,
                    accumulate_ptr<float>(gpg),
                    accumulate_ptr<float>(glos),
                    accumulate_ptr<float>(gref),
                    accumulate_ptr<float>(gdif),
                    accumulate_ptr<float>(gtra),
                    accumulate_ptr<float>(gsca),
                    lre->data_ptr<double>(),
                    lim->data_ptr<double>(),
                    rre->data_ptr<double>(),
                    rim->data_ptr<double>(),
                    dre->data_ptr<double>(),
                    dim->data_ptr<double>(),
                    tre->data_ptr<double>(),
                    tim->data_ptr<double>(),
                    sre->data_ptr<double>(),
                    sim->data_ptr<double>(),
                    grad_coeff_real.data_ptr<float>(),
                    grad_coeff_imag.data_ptr<float>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    }
    pybind11::dict out;
    out["grad_contribution"] = grad_contribution.defined()
        ? pybind11::cast(grad_contribution)
        : pybind11::object(pybind11::none());
    out["grad_coeff_real"] = grad_coeff_real.defined()
        ? pybind11::cast(grad_coeff_real)
        : pybind11::object(pybind11::none());
    out["grad_coeff_imag"] = grad_coeff_imag.defined()
        ? pybind11::cast(grad_coeff_imag)
        : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_bdpt_accumulate_connection_samples_jvp_cuda(
    at::Tensor mis_weight,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor valid,
    int64_t tx_count,
    int64_t rx_count,
    int64_t combine_domain,
    pybind11::object tangent_contribution,
    pybind11::object tangent_coeff_real,
    pybind11::object tangent_coeff_imag,
    pybind11::object los_re,
    pybind11::object los_im,
    pybind11::object reflection_re,
    pybind11::object reflection_im,
    pybind11::object diffraction_re,
    pybind11::object diffraction_im,
    pybind11::object transmission_re,
    pybind11::object transmission_im,
    pybind11::object scattering_re,
    pybind11::object scattering_im) {
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_int_cuda(tx_id, "tx_id", 1);
    check_int_cuda(rx_id, "rx_id", 1);
    check_int_cuda(component_id, "component_id", 1);
    check_bool_cuda(valid, "valid", 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(combine_domain == 0 || combine_domain == 1, "combine_domain must be 0 or 1");
    TORCH_CHECK(tx_id.sizes() == mis_weight.sizes(), "tx_id must match mis_weight");
    TORCH_CHECK(rx_id.sizes() == mis_weight.sizes(), "rx_id must match mis_weight");
    TORCH_CHECK(component_id.sizes() == mis_weight.sizes(), "component_id must match mis_weight");
    TORCH_CHECK(valid.sizes() == mis_weight.sizes(), "valid must match mis_weight");
    check_same_device(tx_id, mis_weight, "tx_id");
    check_same_device(rx_id, mis_weight, "rx_id");
    check_same_device(component_id, mis_weight, "component_id");
    check_same_device(valid, mis_weight, "valid");
    const int64_t count = mis_weight.numel();
    const int64_t out_count = tx_count * rx_count;
    const std::vector<int64_t> sample_shape = {count};
    const std::vector<int64_t> matrix_shape = {tx_count, rx_count};
    auto float_options = mis_weight.options().dtype(at::kFloat);
    auto t_path_gain = at::empty({tx_count, rx_count}, float_options);
    auto t_los = at::empty({tx_count, rx_count}, float_options);
    auto t_reflection = at::empty({tx_count, rx_count}, float_options);
    auto t_diffraction = at::empty({tx_count, rx_count}, float_options);
    auto t_transmission = at::empty({tx_count, rx_count}, float_options);
    auto t_scattering = at::empty({tx_count, rx_count}, float_options);
    zero_float_tensor(t_path_gain);
    zero_float_tensor(t_los);
    zero_float_tensor(t_reflection);
    zero_float_tensor(t_diffraction);
    zero_float_tensor(t_transmission);
    zero_float_tensor(t_scattering);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(mis_weight.get_device()).stream();
    if (combine_domain == 0) {
        at::Tensor tc_s;
        const at::Tensor* tc = accumulate_optional(
            std::move(tangent_contribution), tc_s, "tangent_contribution",
            at::kFloat, sample_shape, mis_weight);
        TORCH_CHECK(tc != nullptr, "power-domain accumulate jvp requires tangent_contribution");
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            bdpt_accumulate_power_jvp_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                mis_weight.data_ptr<float>(),
                tc->data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
                t_path_gain.data_ptr<float>(),
                t_los.data_ptr<float>(),
                t_reflection.data_ptr<float>(),
                t_diffraction.data_ptr<float>(),
                t_transmission.data_ptr<float>(),
                t_scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        at::Tensor tcr_s, tci_s;
        const at::Tensor* tcr = accumulate_optional(
            std::move(tangent_coeff_real), tcr_s, "tangent_coeff_real",
            at::kFloat, sample_shape, mis_weight);
        const at::Tensor* tci = accumulate_optional(
            std::move(tangent_coeff_imag), tci_s, "tangent_coeff_imag",
            at::kFloat, sample_shape, mis_weight);
        TORCH_CHECK(tcr && tci, "coherent accumulate jvp requires tangent_coeff_real/imag");
        at::Tensor lre_s, lim_s, rre_s, rim_s, dre_s, dim_s, tre_s, tim_s, sre_s, sim_s;
        const at::Tensor* lre = accumulate_optional(
            std::move(los_re), lre_s, "los_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* lim = accumulate_optional(
            std::move(los_im), lim_s, "los_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rre = accumulate_optional(
            std::move(reflection_re), rre_s, "reflection_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* rim = accumulate_optional(
            std::move(reflection_im), rim_s, "reflection_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dre = accumulate_optional(
            std::move(diffraction_re), dre_s, "diffraction_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* dim = accumulate_optional(
            std::move(diffraction_im), dim_s, "diffraction_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tre = accumulate_optional(
            std::move(transmission_re), tre_s, "transmission_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* tim = accumulate_optional(
            std::move(transmission_im), tim_s, "transmission_im", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sre = accumulate_optional(
            std::move(scattering_re), sre_s, "scattering_re", at::kDouble, matrix_shape, mis_weight);
        const at::Tensor* sim = accumulate_optional(
            std::move(scattering_im), sim_s, "scattering_im", at::kDouble, matrix_shape, mis_weight);
        TORCH_CHECK(
            lre && lim && rre && rim && dre && dim && tre && tim && sre && sim,
            "coherent accumulate jvp requires all ten bin-sum buffers");
        if (out_count > 0) {
            constexpr int threads = 256;
            int blocks = static_cast<int>((out_count + threads - 1) / threads);
            bdpt_accumulate_coherent_jvp_kernel<<<blocks, threads, 0, stream>>>(
                out_count,
                count,
                tcr->data_ptr<float>(),
                tci->data_ptr<float>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                component_id.data_ptr<int>(),
                valid.data_ptr<bool>(),
                rx_count,
                lre->data_ptr<double>(),
                lim->data_ptr<double>(),
                rre->data_ptr<double>(),
                rim->data_ptr<double>(),
                dre->data_ptr<double>(),
                dim->data_ptr<double>(),
                tre->data_ptr<double>(),
                tim->data_ptr<double>(),
                sre->data_ptr<double>(),
                sim->data_ptr<double>(),
                t_path_gain.data_ptr<float>(),
                t_los.data_ptr<float>(),
                t_reflection.data_ptr<float>(),
                t_diffraction.data_ptr<float>(),
                t_transmission.data_ptr<float>(),
                t_scattering.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    pybind11::dict out;
    out["tangent_path_gain"] = t_path_gain;
    out["tangent_los"] = t_los;
    out["tangent_reflection"] = t_reflection;
    out["tangent_diffraction"] = t_diffraction;
    out["tangent_transmission"] = t_transmission;
    out["tangent_scattering"] = t_scattering;
    return out;
}
