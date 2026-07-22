#include "bdpt_connect_common.cuh"

namespace {

__global__ void bdpt_endpoint_connection_samples_kernel(
    int64_t count,
    int64_t sensor_count,
    float frequency_hz,
    float inv_samples_per_tx,
    int mode_id,
    float beta,
    int strategy_count,
    const float* light_origin,
    const float* light_direction,
    const float* light_throughput_real,
    const float* light_field_real,
    const float* light_field_imag,
    const float* light_source_power,
    const float* light_pdf_forward,
    const int* light_depth,
    const int* light_component_mask,
    const int* light_tx_id,
    const bool* light_valid,
    const float* light_path_length,
    const float* sensor_origin,
    const float* sensor_field_real,
    const float* sensor_pdf_reverse,
    const int* sensor_depth,
    const int* sensor_rx_id,
    const int* sensor_grid_linear_id,
    const bool* sensor_valid,
    int* topology,
    float* contribution,
    float* pdf,
    float* mis_weight,
    int* component_id,
    bool* valid,
    int* tx_id,
    int* rx_id,
    int* grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const int tx = light_tx_id[light_index];
    const int rx = sensor_rx_id[sensor_index];
    const int grid = sensor_grid_linear_id[sensor_index];
    const bool is_valid = light_valid[light_index] && sensor_valid[sensor_index] && tx >= 0 && rx >= 0;

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
              dz * light_direction[light_index * 3 + 2]) /
            distance
        : 1.0f;
    const bool direction_valid = dir_dot > 0.0f;
    const bool row_valid = is_valid && direction_valid;
    // Proposal density excludes free-space geometry. The deterministic
    // endpoint connection has unit discrete mass; inverse-square spreading
    // belongs to the contribution, not to the sampling PDF.
    const float row_pdf = row_valid
        ? fmaxf(light_pdf_forward[light_index], 0.0f) *
            fmaxf(sensor_pdf_reverse[sensor_index], 0.0f)
        : 0.0f;
    // The free-space spreading acts over the unfolded path (light-subpath
    // prefix + connection segment), not the last segment alone.
    const float total_distance = distance + fmaxf(light_path_length[light_index], 0.0f);
    const float wave_number = 2.0f * kPi * frequency_hz / kLightSpeedMPerS;
    const float amplitude = 1.0f /
        (2.0f * fmaxf(wave_number, 1.0e-12f) * fmaxf(total_distance, 1.0e-6f));
    const utd::Complex propagation = utd::cplx_mul_real(
        utd::cplx_exp_phase(transport::precise_neg_kd(wave_number, total_distance)),
        amplitude);
    const int64_t field_offset = light_index * 3;
    const utd::Complex3 incident_field = {
        utd::cplx(light_field_real[field_offset], light_field_imag[field_offset]),
        utd::cplx(light_field_real[field_offset + 1], light_field_imag[field_offset + 1]),
        utd::cplx(light_field_real[field_offset + 2], light_field_imag[field_offset + 2])};
    const utd::Complex3 received_field = utd::c3_scale(incident_field, propagation);
    const utd::float3a connection_direction = utd::make_f3(dx / distance, dy / distance, dz / distance);
    const int64_t sensor_field_offset = sensor_index * 3;
    const utd::float3a receiver_polarization = utd::make_f3(
        sensor_field_real[sensor_field_offset],
        sensor_field_real[sensor_field_offset + 1],
        sensor_field_real[sensor_field_offset + 2]);
    const utd::Complex coefficient = transport::project_receiver(
        received_field, connection_direction, receiver_polarization);
    const float coefficient_power = utd::cplx_abs_sqr(coefficient);
    const float row_contribution = row_valid
        ? light_source_power[light_index] * coefficient_power * inv_samples_per_tx
        : 0.0f;

    tx_id[index] = tx;
    rx_id[index] = rx;
    grid_linear_id[index] = grid;
    const int light_component = light_component_mask[light_index];
    const int sample_component = bdpt_component_from_mask(light_component);
    component_id[index] = sample_component;
    out_light_depth[index] = light_depth[light_index];
    out_sensor_depth[index] = sensor_depth[sensor_index];
    contribution[index] = row_contribution;
    pdf[index] = row_pdf;
    mis_weight[index] = row_valid ? bdpt_single_strategy_mis_weight(row_pdf, mode_id, beta) : 0.0f;
    valid[index] = row_valid;
    path_length_m[index] = total_distance;
    const int row = static_cast<int>(index * 4);
    topology[row + 0] = tx;
    topology[row + 1] = rx;
    topology[row + 2] = sample_component;
    topology[row + 3] = light_depth[light_index] + sensor_depth[sensor_index];
}
}  // namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
channel_bdpt_endpoint_connection_samples_cuda(
    at::Tensor light_origin,
    at::Tensor light_direction,
    at::Tensor light_throughput_real,
    at::Tensor light_field_real,
    at::Tensor light_field_imag,
    at::Tensor light_source_power,
    at::Tensor light_pdf_forward,
    at::Tensor light_depth,
    at::Tensor light_component_mask,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor light_path_length,
    at::Tensor sensor_origin,
    at::Tensor sensor_field_real,
    at::Tensor sensor_pdf_reverse,
    at::Tensor sensor_depth,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_grid_linear_id,
    at::Tensor sensor_valid,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    int64_t strategy_count,
    int64_t max_paths) {
    check_vec3_cuda(light_origin, "light_origin");
    check_vec3_cuda(light_direction, "light_direction");
    check_float_cuda(light_throughput_real, "light_throughput_real", 1);
    check_vec3_cuda(light_field_real, "light_field_real");
    check_vec3_cuda(light_field_imag, "light_field_imag");
    check_float_cuda(light_source_power, "light_source_power", 1);
    check_float_cuda(light_pdf_forward, "light_pdf_forward", 1);
    check_int_cuda(light_depth, "light_depth", 1);
    check_int_cuda(light_component_mask, "light_component_mask", 1);
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_vec3_cuda(sensor_field_real, "sensor_field_real");
    check_float_cuda(sensor_pdf_reverse, "sensor_pdf_reverse", 1);
    check_int_cuda(sensor_depth, "sensor_depth", 1);
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_int_cuda(sensor_grid_linear_id, "sensor_grid_linear_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(samples_per_tx > 0, "samples_per_tx must be positive");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_mis_args(mode_id, strategy_count);
    TORCH_CHECK(strategy_count == 1, "endpoint connections support exactly one strategy");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    TORCH_CHECK(light_direction.size(0) == light_count, "light_direction must match light count");
    check_same_device(light_direction, light_origin, "light_direction");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&light_throughput_real, "light_throughput_real"),
             std::pair<const at::Tensor*, const char*>(&light_field_real, "light_field_real"),
             std::pair<const at::Tensor*, const char*>(&light_field_imag, "light_field_imag"),
             std::pair<const at::Tensor*, const char*>(&light_source_power, "light_source_power"),
             std::pair<const at::Tensor*, const char*>(&light_pdf_forward, "light_pdf_forward"),
             std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),
             std::pair<const at::Tensor*, const char*>(&light_component_mask, "light_component_mask"),
             std::pair<const at::Tensor*, const char*>(&light_tx_id, "light_tx_id"),
             std::pair<const at::Tensor*, const char*>(&light_valid, "light_valid"),
             std::pair<const at::Tensor*, const char*>(&light_path_length, "light_path_length"),
         }) {
        TORCH_CHECK(pair.first->size(0) == light_count, pair.second, " must match light count");
        check_same_device(*pair.first, light_origin, pair.second);
    }
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&sensor_pdf_reverse, "sensor_pdf_reverse"),
             std::pair<const at::Tensor*, const char*>(&sensor_field_real, "sensor_field_real"),
             std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),
             std::pair<const at::Tensor*, const char*>(&sensor_rx_id, "sensor_rx_id"),
             std::pair<const at::Tensor*, const char*>(&sensor_grid_linear_id, "sensor_grid_linear_id"),
             std::pair<const at::Tensor*, const char*>(&sensor_valid, "sensor_valid"),
         }) {
        TORCH_CHECK(pair.first->size(0) == sensor_count, pair.second, " must match sensor count");
        check_same_device(*pair.first, light_origin, pair.second);
    }
    check_same_device(sensor_origin, light_origin, "sensor_origin");
    const int64_t total = light_count * sensor_count;
    const int64_t count = max_paths < 0 ? total : std::min<int64_t>(max_paths, total);
    auto [
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m] = allocate_connection_samples(light_origin, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_endpoint_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sensor_count,
            static_cast<float>(frequency_hz),
            1.0f / static_cast<float>(samples_per_tx),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            light_origin.data_ptr<float>(),
            light_direction.data_ptr<float>(),
            light_throughput_real.data_ptr<float>(),
            light_field_real.data_ptr<float>(),
            light_field_imag.data_ptr<float>(),
            light_source_power.data_ptr<float>(),
            light_pdf_forward.data_ptr<float>(),
            light_depth.data_ptr<int>(),
            light_component_mask.data_ptr<int>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            light_path_length.data_ptr<float>(),
            sensor_origin.data_ptr<float>(),
            sensor_field_real.data_ptr<float>(),
            sensor_pdf_reverse.data_ptr<float>(),
            sensor_depth.data_ptr<int>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_grid_linear_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            out_light_depth.data_ptr<int>(),
            out_sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        topology,
        contribution,
        pdf,
        mis_weight,
        component_id,
        valid,
        tx_id,
        rx_id,
        grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        path_length_m};
}
