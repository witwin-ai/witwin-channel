#include "bdpt_connect_common.cuh"

#include "../field_transport_ad.cuh"

#include <algorithm>

// ADR-022 6.3: backward + jvp companions for bdpt_endpoint_connection_samples.
//
// Forward (per connection row): the light endpoint field F and the sensor
// polarization project through the frozen free-space carrier into
//   contribution = P_src * |coeff|^2 / N,
//   coeff = <F * propagation, rx_axis>,  rx_axis = project(sensor_pol, dir).
// Differentiable: the light field F, the sensor polarization (through rx_axis),
// the carrier frequency (through the propagation amplitude and phase), and the
// source power P_src (tx_power). Frozen: the connection geometry (distance,
// direction, total path length), N (samples_per_tx), visibility, MIS, and the
// component/topology structure. The backward recomputes the forward carrier in
// primal expression order; light/sensor field grads accumulate with atomicAdd
// because the light x sensor connection grid shares each endpoint field across
// its row, and the tx_power / frequency grads are scalar atomic reductions.

namespace {

namespace ad = channel_native::field_transport_ad;

// Recompute the frozen carrier for one connection row exactly as
// bdpt_endpoint_connection_samples_kernel; returns whether the row contributes.
struct ConnectionCarrier {
    bool row_valid;
    int tx;
    int rx;
    utd::Complex propagation;
    utd::Complex d_propagation_df;  // d propagation / d frequency
    utd::float3a rx_axis;           // project(sensor_pol, direction)
    utd::float3a direction;         // connection direction (frozen)
    utd::float3a sensor_pol;        // sensor polarization (differentiable)
    utd::Complex3 incident_field;   // light field F
    float source_power;
    float inv_samples_per_tx;
};

__device__ ConnectionCarrier bdpt_connection_carrier(
    int64_t index,
    int64_t sensor_count,
    float frequency_hz,
    float inv_samples_per_tx,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid) {
    ConnectionCarrier out;
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const int tx = light_tx_id[light_index];
    const int rx = sensor_rx_id[sensor_index];
    const bool is_valid =
        light_valid[light_index] && sensor_valid[sensor_index] && tx >= 0 && rx >= 0;

    const float lx = light_origin[light_index * 3 + 0];
    const float ly = light_origin[light_index * 3 + 1];
    const float lz = light_origin[light_index * 3 + 2];
    const float sx = sensor_origin[sensor_index * 3 + 0];
    const float sy = sensor_origin[sensor_index * 3 + 1];
    const float sz = sensor_origin[sensor_index * 3 + 2];
    const float dx = sx - lx;
    const float dy = sy - ly;
    const float dz = sz - lz;
    const float distance = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1.0e-6f);
    const int light_path_depth = light_depth[light_index];
    const float dir_dot = light_path_depth > 0
        ? (dx * light_direction[light_index * 3 + 0] +
           dy * light_direction[light_index * 3 + 1] +
           dz * light_direction[light_index * 3 + 2]) / distance
        : 1.0f;
    const bool direction_valid = dir_dot > 0.0f;
    out.row_valid = is_valid && direction_valid;
    out.tx = tx;
    out.rx = rx;
    out.inv_samples_per_tx = inv_samples_per_tx;
    out.source_power = light_source_power[light_index];

    const float total_distance = distance + fmaxf(light_path_length[light_index], 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float k_clamped = fmaxf(wave_number, 1.0e-12f);
    const float l_clamped = fmaxf(total_distance, 1.0e-6f);
    const float amplitude = 1.0f / (2.0f * k_clamped * l_clamped);
    const float phase_angle = transport::precise_neg_kd(wave_number, total_distance);
    const utd::Complex carrier_phase = utd::cplx_exp_phase(phase_angle);
    out.propagation = utd::cplx_mul_real(carrier_phase, amplitude);

    // d propagation / d frequency (amplitude and phase chain, matching the
    // free-space carrier convention). dk/df = 2*pi/c; the fmod phase reduction
    // has unit slope so d(phase_angle)/dk = -total_distance.
    const float dk_df = 2.0f * kPi / kLightSpeedMPerS;
    const float d_amp_df =
        wave_number > 1.0e-12f ? (-amplitude / k_clamped) * dk_df : 0.0f;
    const float d_phase_df = -total_distance * dk_df;
    out.d_propagation_df = utd::cplx(
        d_amp_df * carrier_phase.re - amplitude * carrier_phase.im * d_phase_df,
        d_amp_df * carrier_phase.im + amplitude * carrier_phase.re * d_phase_df);

    out.direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    out.sensor_pol = utd::make_f3(
        sensor_field_real[sensor_index * 3 + 0],
        sensor_field_real[sensor_index * 3 + 1],
        sensor_field_real[sensor_index * 3 + 2]);
    out.rx_axis = utd::project_to_wedge_plane(out.sensor_pol, out.direction);
    const int64_t field_offset = light_index * 3;
    out.incident_field = {
        utd::cplx(light_field_real[field_offset], light_field_imag[field_offset]),
        utd::cplx(light_field_real[field_offset + 1], light_field_imag[field_offset + 1]),
        utd::cplx(light_field_real[field_offset + 2], light_field_imag[field_offset + 2])};
    return out;
}

__global__ void bdpt_endpoint_connection_backward_kernel(
    int64_t count,
    int64_t sensor_count,
    int64_t tx_count,
    float frequency_hz,
    float inv_samples_per_tx,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    const float* grad_contribution,
    float* grad_light_field_real,
    float* grad_light_field_imag,
    float* grad_sensor_field_real,
    float* grad_frequency,
    float* grad_tx_power) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ConnectionCarrier carrier = bdpt_connection_carrier(
            index, sensor_count, frequency_hz, inv_samples_per_tx, light_origin,
            light_direction, light_field_real, light_field_imag,
            light_source_power, light_depth, light_tx_id, light_valid,
            light_path_length, sensor_origin, sensor_field_real, sensor_rx_id,
            sensor_valid);
        if (!carrier.row_valid) {
            continue;  // forward wrote contribution = 0; every gradient is zero
        }
        const float g = grad_contribution[index];
        const int64_t light_index = index / sensor_count;
        const int64_t sensor_index = index - light_index * sensor_count;

        // Recompute the forward chain (primal expression order).
        const utd::Complex3 received = utd::c3_scale(
            carrier.incident_field, carrier.propagation);
        const utd::Complex coeff = transport::complex3_dot_real(received, carrier.rx_axis);
        const float coeff_power = utd::cplx_abs_sqr(coeff);
        const float scale = carrier.source_power * carrier.inv_samples_per_tx;

        // contribution = source_power * |coeff|^2 * inv_N.
        // d/dcoeff (pair) = source_power * inv_N * 2 * conj-pair(coeff).
        const utd::Complex g_coeff = utd::cplx(
            g * scale * 2.0f * coeff.re, g * scale * 2.0f * coeff.im);
        // coeff = <received, rx_axis>: split into received and rx_axis.
        utd::Complex3 g_received = utd::c3_zero();
        utd::float3a g_rx_axis = utd::f3_zero();
        utd::adj_cplx_dot_real(received, carrier.rx_axis, g_coeff, g_received, g_rx_axis);
        // received = incident_field * propagation (per axis).
        utd::Complex g_propagation = utd::cplx_zero();
        utd::Complex g_field_x = utd::cplx_zero();
        utd::Complex g_field_y = utd::cplx_zero();
        utd::Complex g_field_z = utd::cplx_zero();
        utd::adj_cplx_mul(
            carrier.incident_field.x, carrier.propagation, g_received.x,
            g_field_x, g_propagation);
        utd::adj_cplx_mul(
            carrier.incident_field.y, carrier.propagation, g_received.y,
            g_field_y, g_propagation);
        utd::adj_cplx_mul(
            carrier.incident_field.z, carrier.propagation, g_received.z,
            g_field_z, g_propagation);

        if (grad_light_field_real != nullptr) {
            const int64_t base = light_index * 3;
            atomicAdd(grad_light_field_real + base, g_field_x.re);
            atomicAdd(grad_light_field_real + base + 1, g_field_y.re);
            atomicAdd(grad_light_field_real + base + 2, g_field_z.re);
            atomicAdd(grad_light_field_imag + base, g_field_x.im);
            atomicAdd(grad_light_field_imag + base + 1, g_field_y.im);
            atomicAdd(grad_light_field_imag + base + 2, g_field_z.im);
        }
        if (grad_sensor_field_real != nullptr) {
            // rx_axis = project(sensor_pol, direction); the direction cotangent
            // is discarded (frozen geometry).
            utd::float3a g_sensor_pol = utd::f3_zero();
            utd::float3a g_dir_dump = utd::f3_zero();
            ad::adj_transverse_project(
                carrier.direction, carrier.sensor_pol, g_rx_axis,
                g_dir_dump, g_sensor_pol);
            const int64_t base = sensor_index * 3;
            atomicAdd(grad_sensor_field_real + base, g_sensor_pol.x);
            atomicAdd(grad_sensor_field_real + base + 1, g_sensor_pol.y);
            atomicAdd(grad_sensor_field_real + base + 2, g_sensor_pol.z);
        }
        if (grad_frequency != nullptr) {
            atomicAdd(
                grad_frequency,
                ad::adj_dot(g_propagation, carrier.d_propagation_df));
        }
        if (grad_tx_power != nullptr && carrier.tx >= 0 && carrier.tx < tx_count) {
            atomicAdd(
                grad_tx_power + carrier.tx,
                g * coeff_power * carrier.inv_samples_per_tx);
        }
    }
}

__global__ void bdpt_endpoint_connection_jvp_kernel(
    int64_t count,
    int64_t sensor_count,
    int64_t tx_count,
    float frequency_hz,
    float inv_samples_per_tx,
    float tangent_frequency,
    const float* light_origin,
    const float* light_direction,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const int* light_depth,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    const float* tangent_light_field_real,
    const float* tangent_light_field_imag,
    const float* tangent_sensor_field_real,
    const float* tangent_tx_power,
    float* tangent_contribution) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ConnectionCarrier carrier = bdpt_connection_carrier(
            index, sensor_count, frequency_hz, inv_samples_per_tx, light_origin,
            light_direction, light_field_real, light_field_imag,
            light_source_power, light_depth, light_tx_id, light_valid,
            light_path_length, sensor_origin, sensor_field_real, sensor_rx_id,
            sensor_valid);
        if (!carrier.row_valid) {
            tangent_contribution[index] = 0.0f;
            continue;
        }
        const int64_t light_index = index / sensor_count;
        const int64_t sensor_index = index - light_index * sensor_count;
        const int64_t light_base = light_index * 3;
        const int64_t sensor_base = sensor_index * 3;

        const utd::Complex3 t_field = {
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base] : 0.0f),
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base + 1] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base + 1] : 0.0f),
            utd::cplx(
                tangent_light_field_real != nullptr ? tangent_light_field_real[light_base + 2] : 0.0f,
                tangent_light_field_imag != nullptr ? tangent_light_field_imag[light_base + 2] : 0.0f)};
        const utd::float3a t_sensor_pol = utd::make_f3(
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base] : 0.0f,
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base + 1] : 0.0f,
            tangent_sensor_field_real != nullptr ? tangent_sensor_field_real[sensor_base + 2] : 0.0f);
        // rx_axis = project(sensor_pol, dir) is linear in sensor_pol (dir frozen).
        const utd::float3a t_rx_axis = utd::project_to_wedge_plane(
            t_sensor_pol, carrier.direction);
        const utd::Complex t_propagation = utd::cplx_mul_real(
            carrier.d_propagation_df, tangent_frequency);

        const utd::Complex3 received = utd::c3_scale(
            carrier.incident_field, carrier.propagation);
        const utd::Complex3 t_received = utd::c3_add(
            utd::c3_scale(t_field, carrier.propagation),
            utd::c3_scale(carrier.incident_field, t_propagation));
        const utd::Complex coeff = transport::complex3_dot_real(received, carrier.rx_axis);
        const utd::Complex t_coeff = utd::cplx_add(
            transport::complex3_dot_real(t_received, carrier.rx_axis),
            transport::complex3_dot_real(received, t_rx_axis));
        const float coeff_power = utd::cplx_abs_sqr(coeff);
        const float t_coeff_power =
            2.0f * (coeff.re * t_coeff.re + coeff.im * t_coeff.im);
        const float t_source_power =
            (tangent_tx_power != nullptr && carrier.tx >= 0 && carrier.tx < tx_count)
                ? tangent_tx_power[carrier.tx]
                : 0.0f;
        tangent_contribution[index] = carrier.inv_samples_per_tx *
            (t_source_power * coeff_power + carrier.source_power * t_coeff_power);
    }
}

const at::Tensor* connect_optional(
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
T* connect_ptr(at::Tensor& tensor) {
    return tensor.defined() ? tensor.data_ptr<T>() : nullptr;
}

at::Tensor connect_zero(at::IntArrayRef sizes, const at::TensorOptions& options) {
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

int64_t connection_count(int64_t light_count, int64_t sensor_count, int64_t max_paths) {
    const int64_t total = light_count * sensor_count;
    return max_paths < 0 ? total : std::min<int64_t>(max_paths, total);
}

}  // namespace

pybind11::dict cn_bdpt_endpoint_connection_samples_backward_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    at::Tensor grad_contribution,
    bool need_grad_field,
    bool need_grad_frequency,
    bool need_grad_tx_power) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_float_cuda(light_path_length, "light_path_length", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    check_float_cuda(grad_contribution, "grad_contribution", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    const int64_t count = connection_count(light_count, sensor_count, max_paths);
    TORCH_CHECK(grad_contribution.numel() == count, "grad_contribution must match connection count");

    at::Tensor grad_light_field_real;
    at::Tensor grad_light_field_imag;
    at::Tensor grad_sensor_field_real;
    at::Tensor grad_sensor_field_imag;
    at::Tensor grad_frequency;
    at::Tensor grad_tx_power;
    if (need_grad_field) {
        grad_light_field_real = connect_zero({light_count, 3}, light_origin.options());
        grad_light_field_imag = connect_zero({light_count, 3}, light_origin.options());
        grad_sensor_field_real = connect_zero({sensor_count, 3}, light_origin.options());
        // The sensor imaginary field never enters the forward; its derivative is
        // exactly zero (reported, not a fallback silent-zero).
        grad_sensor_field_imag = connect_zero({sensor_count, 3}, light_origin.options());
    }
    if (need_grad_frequency) {
        grad_frequency = connect_zero({1}, light_origin.options());
    }
    if (need_grad_tx_power) {
        grad_tx_power = connect_zero({tx_count}, light_origin.options());
    }
    if (count > 0 && (need_grad_field || need_grad_frequency || need_grad_tx_power)) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_endpoint_connection_backward_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            tx_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            grad_contribution.data_ptr<float>(),
            connect_ptr<float>(grad_light_field_real),
            connect_ptr<float>(grad_light_field_imag),
            connect_ptr<float>(grad_sensor_field_real),
            connect_ptr<float>(grad_frequency),
            connect_ptr<float>(grad_tx_power));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_light_field_real"] = need_grad_field
        ? pybind11::cast(grad_light_field_real) : pybind11::object(pybind11::none());
    out["grad_light_field_imag"] = need_grad_field
        ? pybind11::cast(grad_light_field_imag) : pybind11::object(pybind11::none());
    out["grad_sensor_field_real"] = need_grad_field
        ? pybind11::cast(grad_sensor_field_real) : pybind11::object(pybind11::none());
    out["grad_sensor_field_imag"] = need_grad_field
        ? pybind11::cast(grad_sensor_field_imag) : pybind11::object(pybind11::none());
    out["grad_frequency"] = need_grad_frequency
        ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    out["grad_tx_power"] = need_grad_tx_power
        ? pybind11::cast(grad_tx_power) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_bdpt_endpoint_connection_samples_jvp_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_depth,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t tx_count,
    int64_t max_paths,
    pybind11::object tangent_light_field_real,
    pybind11::object tangent_light_field_imag,
    pybind11::object tangent_sensor_field_real,
    double tangent_frequency,
    pybind11::object tangent_tx_power) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_float_cuda(light_path_length, "light_path_length", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    const int64_t count = connection_count(light_count, sensor_count, max_paths);

    at::Tensor tlr_s, tli_s, tsr_s, ttx_s;
    const at::Tensor* tlr = connect_optional(
        std::move(tangent_light_field_real), tlr_s, "tangent_light_field_real",
        at::kFloat, {light_count, 3}, light_origin);
    const at::Tensor* tli = connect_optional(
        std::move(tangent_light_field_imag), tli_s, "tangent_light_field_imag",
        at::kFloat, {light_count, 3}, light_origin);
    const at::Tensor* tsr = connect_optional(
        std::move(tangent_sensor_field_real), tsr_s, "tangent_sensor_field_real",
        at::kFloat, {sensor_count, 3}, light_origin);
    const at::Tensor* ttx = connect_optional(
        std::move(tangent_tx_power), ttx_s, "tangent_tx_power",
        at::kFloat, {tx_count}, light_origin);

    auto tangent_contribution = at::empty({count}, light_origin.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_endpoint_connection_jvp_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            tx_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            static_cast<float>(tangent_frequency),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            tlr != nullptr ? tlr->data_ptr<float>() : nullptr,
            tli != nullptr ? tli->data_ptr<float>() : nullptr,
            tsr != nullptr ? tsr->data_ptr<float>() : nullptr,
            ttx != nullptr ? ttx->data_ptr<float>() : nullptr,
            tangent_contribution.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_contribution"] = tangent_contribution;
    return out;
}
