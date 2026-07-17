#include "field_transport_ad_common.cuh"

namespace {

// ---------------------------------------------------------------------------
// Free space (frequency is the only differentiable input in AD-1).
// ---------------------------------------------------------------------------

template <typename T>
__global__ void free_space_fwd_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    c10::complex<T>* field_vector,
    c10::complex<T>* coefficient,
    c10::complex<T>* path_field,
    T* path_gain,
    T* path_length,
    T* delay,
    T* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        field_vector[base] = eval.carrier * eval.tx_axis.x;
        field_vector[base + 1] = eval.carrier * eval.tx_axis.y;
        field_vector[base + 2] = eval.carrier * eval.tx_axis.z;
        const c10::complex<T> scalar = eval.carrier * eval.projection;
        coefficient[index] = scalar;
        const c10::complex<T> received = scalar * eval.amplitude_scale;
        path_field[index] = received;
        path_gain[index] = received.real() * received.real() +
                           received.imag() * received.imag();
        path_length[index] = eval.distance;
        delay[index] = eval.distance / T(ad::kSpeedOfLight);
        direction_out[base] = eval.direction.x;
        direction_out[base + 1] = eval.direction.y;
        direction_out[base + 2] = eval.direction.z;
    }
}

template <typename T>
__global__ void free_space_backward_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    const c10::complex<T>* grad_field_vector,
    const c10::complex<T>* grad_coefficient,
    const c10::complex<T>* grad_path_field,
    const T* grad_path_gain,
    const T* grad_path_length,
    const T* grad_delay,
    T* grad_frequency,
    T* grad_source,
    T* grad_target) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        // Fold the field-output cotangents onto the carrier P; the geometry
        // enters through the carrier distance, the tx/rx bases and the raw
        // straight length (path_length_m / delay_s).
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        c10::complex<T> g_scalar(T(0), T(0));
        if (grad_coefficient != nullptr)
            g_scalar += grad_coefficient[index];
        if (grad_path_field != nullptr)
            g_scalar += grad_path_field[index] * eval.amplitude_scale;
        if (grad_path_gain != nullptr)
            g_scalar += path_field_value *
                        (T(2) * grad_path_gain[index] * eval.amplitude_scale);
        c10::complex<T> g_carrier = g_scalar * eval.projection;
        if (grad_field_vector != nullptr) {
            g_carrier += grad_field_vector[base] * eval.tx_axis.x;
            g_carrier += grad_field_vector[base + 1] * eval.tx_axis.y;
            g_carrier += grad_field_vector[base + 2] * eval.tx_axis.z;
        }
        if (grad_frequency != nullptr) {
            const T g_freq = g_carrier.real() * eval.carrier_dfreq.real() +
                             g_carrier.imag() * eval.carrier_dfreq.imag();
            atomicAdd(grad_frequency, g_freq);
        }
        if (grad_source == nullptr && grad_target == nullptr)
            continue;
        // Real-pair cotangents of the tx/rx bases: coefficient = P *
        // <tx_axis, rx_axis> and field_vector = P * tx_axis.
        const T g_projection = g_scalar.real() * eval.carrier.real() +
                               g_scalar.imag() * eval.carrier.imag();
        ad::Vec3<T> g_tx_axis = {T(0), T(0), T(0)};
        ad::Vec3<T> g_rx_axis = {T(0), T(0), T(0)};
        g_tx_axis = ad::v3_add(g_tx_axis, ad::v3_scale(eval.rx_axis, g_projection));
        g_rx_axis = ad::v3_add(g_rx_axis, ad::v3_scale(eval.tx_axis, g_projection));
        if (grad_field_vector != nullptr) {
            g_tx_axis.x += grad_field_vector[base].real() * eval.carrier.real() +
                           grad_field_vector[base].imag() * eval.carrier.imag();
            g_tx_axis.y +=
                grad_field_vector[base + 1].real() * eval.carrier.real() +
                grad_field_vector[base + 1].imag() * eval.carrier.imag();
            g_tx_axis.z +=
                grad_field_vector[base + 2].real() * eval.carrier.real() +
                grad_field_vector[base + 2].imag() * eval.carrier.imag();
        }
        T g_distance = g_carrier.real() * eval.carrier_ddist.real() +
                       g_carrier.imag() * eval.carrier_ddist.imag();
        if (grad_path_length != nullptr)
            g_distance += grad_path_length[index];
        if (grad_delay != nullptr)
            g_distance += grad_delay[index] / T(ad::kSpeedOfLight);
        const ad::Vec3<T> offset =
            ad::v3_sub(ad::v3_load(target, index), ad::v3_load(source, index));
        ad::Vec3<T> g_direction = {T(0), T(0), T(0)};
        ad::adj_v3_transverse_project(
            eval.direction, ad::v3_load(tx_polarization, index), g_tx_axis,
            g_direction);
        ad::adj_v3_transverse_project(
            eval.direction, ad::v3_load(rx_polarization, index), g_rx_axis,
            g_direction);
        ad::Vec3<T> g_offset = {T(0), T(0), T(0)};
        ad::Vec3<T> g_alternate = {T(0), T(0), T(0)};
        ad::adj_v3_safe_normalize(
            offset, ad::Vec3<T>{T(0), T(0), T(1)}, g_direction, g_offset,
            g_alternate);
        ad::adj_v3_length(offset, g_distance, g_offset);
        if (grad_target != nullptr) {
            grad_target[base] = g_offset.x;
            grad_target[base + 1] = g_offset.y;
            grad_target[base + 2] = g_offset.z;
        }
        if (grad_source != nullptr) {
            grad_source[base] = -g_offset.x;
            grad_source[base + 1] = -g_offset.y;
            grad_source[base + 2] = -g_offset.z;
        }
    }
}

template <typename T>
__global__ void free_space_jvp_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    T tangent_frequency,
    const T* tangent_source,
    const T* tangent_target,
    c10::complex<T>* t_field_vector,
    c10::complex<T>* t_coefficient,
    c10::complex<T>* t_path_field,
    T* t_path_gain,
    T* t_path_length,
    T* t_delay) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            ad::v3_load(source, index),
            ad::v3_load(target, index),
            ad::v3_load(tx_polarization, index),
            ad::v3_load(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const ad::Vec3<T> zero3 = {T(0), T(0), T(0)};
        const ad::Vec3<T> d_source =
            tangent_source != nullptr ? ad::v3_load(tangent_source, index) : zero3;
        const ad::Vec3<T> d_target =
            tangent_target != nullptr ? ad::v3_load(tangent_target, index) : zero3;
        const ad::DualV3<T> offset = {
            ad::v3_sub(ad::v3_load(target, index), ad::v3_load(source, index)),
            ad::v3_sub(d_target, d_source)};
        T d_distance = T(0);
        (void)ad::dual_v3_length(offset, d_distance);
        const ad::DualV3<T> direction = ad::dual_v3_safe_normalize(
            offset, ad::dv3_const(ad::Vec3<T>{T(0), T(0), T(1)}));
        const ad::DualV3<T> tx_axis = ad::dual_v3_transverse_project(
            direction, ad::v3_load(tx_polarization, index));
        const ad::DualV3<T> rx_axis = ad::dual_v3_transverse_project(
            direction, ad::v3_load(rx_polarization, index));
        const c10::complex<T> d_carrier =
            eval.carrier_dfreq * tangent_frequency +
            eval.carrier_ddist * d_distance;
        const int64_t base = index * 3;
        t_field_vector[base] =
            d_carrier * eval.tx_axis.x + eval.carrier * tx_axis.d.x;
        t_field_vector[base + 1] =
            d_carrier * eval.tx_axis.y + eval.carrier * tx_axis.d.y;
        t_field_vector[base + 2] =
            d_carrier * eval.tx_axis.z + eval.carrier * tx_axis.d.z;
        const T d_projection = ad::v3_dot(tx_axis.d, eval.rx_axis) +
                               ad::v3_dot(eval.tx_axis, rx_axis.d);
        const c10::complex<T> d_scalar =
            d_carrier * eval.projection + eval.carrier * d_projection;
        t_coefficient[index] = d_scalar;
        const c10::complex<T> d_path_field = d_scalar * eval.amplitude_scale;
        t_path_field[index] = d_path_field;
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        t_path_gain[index] =
            T(2) * (path_field_value.real() * d_path_field.real() +
                    path_field_value.imag() * d_path_field.imag());
        t_path_length[index] = d_distance;
        t_delay[index] = d_distance / T(ad::kSpeedOfLight);
    }
}

void check_free_space_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    c10::ScalarType real_dtype) {
    using channel_native::check_tensor;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}}) {
        check_tensor(named.first, named.second, real_dtype, 2);
        TORCH_CHECK(
            named.first.size(1) == 3, named.second, " must have shape (N, 3)");
    }
    check_tensor(tx_power, "tx_power", real_dtype, 1);
    const int64_t count = source.size(0);
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "free-space field tensors must have matching rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

}  // namespace

pybind11::dict cn_field_free_space_fwd64(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, at::kDouble);
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexDouble);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        free_space_fwd_kernel<double><<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<double>(),
            target.data_ptr<double>(),
            tx_power.data_ptr<double>(),
            tx_polarization.data_ptr<double>(),
            rx_polarization.data_ptr<double>(),
            frequency_hz,
            field_vector.data_ptr<c10::complex<double>>(),
            coefficient.data_ptr<c10::complex<double>>(),
            path_field.data_ptr<c10::complex<double>>(),
            path_gain.data_ptr<double>(),
            path_length.data_ptr<double>(),
            delay.data_ptr<double>(),
            direction.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict cn_field_free_space_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_backward supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        complex_dtype, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        complex_dtype, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        complex_dtype, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        real_dtype, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        real_dtype, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        real_dtype, {count}, source);

    pybind11::dict out;
    if (!need_grad_frequency && !need_grad_geometry) {
        out["grad_frequency"] = pybind11::none();
        out["grad_source"] = pybind11::none();
        out["grad_target"] = pybind11::none();
        return out;
    }
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    at::Tensor grad_source = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    at::Tensor grad_target = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    const bool any_grad = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                          gpg != nullptr || gpl != nullptr || gd != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_backward_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                    grad_ptr<float>(gpg),
                    grad_ptr<float>(gpl),
                    grad_ptr<float>(gd),
                    need_grad_frequency ? grad_frequency.data_ptr<float>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<float>() : nullptr);
        } else {
            free_space_backward_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    gfv ? gfv->data_ptr<c10::complex<double>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<double>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<double>>() : nullptr,
                    grad_ptr<double>(gpg),
                    grad_ptr<double>(gpl),
                    grad_ptr<double>(gd),
                    need_grad_frequency ? grad_frequency.data_ptr<double>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<double>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<double>()
                                       : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_free_space_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_jvp supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor ts_storage;
    at::Tensor tt_storage;
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), ts_storage, "tangent_source",
        real_dtype, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), tt_storage, "tangent_target",
        real_dtype, {count, 3}, source);
    auto complex_options = source.options().dtype(complex_dtype);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_jvp_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    static_cast<float>(tangent_frequency),
                    grad_ptr<float>(t_source),
                    grad_ptr<float>(t_target),
                    t_field_vector.data_ptr<c10::complex<float>>(),
                    t_coefficient.data_ptr<c10::complex<float>>(),
                    t_path_field.data_ptr<c10::complex<float>>(),
                    t_path_gain.data_ptr<float>(),
                    t_path_length.data_ptr<float>(),
                    t_delay.data_ptr<float>());
        } else {
            free_space_jvp_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    tangent_frequency,
                    grad_ptr<double>(t_source),
                    grad_ptr<double>(t_target),
                    t_field_vector.data_ptr<c10::complex<double>>(),
                    t_coefficient.data_ptr<c10::complex<double>>(),
                    t_path_field.data_ptr<c10::complex<double>>(),
                    t_path_gain.data_ptr<double>(),
                    t_path_length.data_ptr<double>(),
                    t_delay.data_ptr<double>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    return out;
}
