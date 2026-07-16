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

__global__ void bdpt_diffraction_connection_samples_from_tape_kernel(
    int64_t count,
    int tx_index,
    int state_count,
    int grid_resolution0,
    int grid_resolution1,
    int grid_axis,
    float grid_position,
    float grid_coord0_min,
    float grid_coord0_max,
    float grid_coord1_min,
    float grid_coord1_max,
    float grid_cell_area,
    float wavelength,
    int direct_samples,
    int keller_samples,
    int mode_id,
    float beta,
    int strategy_count,
    int material_count,
    const bool* tape_active,
    const int* tape_state_idx,
    const int* tape_cell,
    const int* tape_material_idx,
    const float* tape_edge_u,
    const int* state_edge_index,
    const float* state_edge_pos,
    const float* state_edge_dir,
    const float* state_edge_t_min,
    const float* state_edge_t_max,
    const float* state_exterior_angle,
    const float* state_src,
    const float* state_src_power,
    const float* material_gain,
    const bool* material_valid,
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
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= count) {
        return;
    }
    const int total_samples = direct_samples + keller_samples;
    const bool lane_direct = static_cast<int>(lane) < direct_samples;
    const bool lane_keller =
        !lane_direct && static_cast<int>(lane) < total_samples;
    const int strategy_samples = lane_direct ? direct_samples : (lane_keller ? keller_samples : 0);
    const int state_idx = tape_state_idx[lane];
    const int raydn_cell = tape_cell[lane];
    const int material_idx = tape_material_idx[lane];
    const int cell_count = grid_resolution0 * grid_resolution1;
    const bool row_valid = tape_active[lane] &&
        strategy_samples > 0 &&
        state_idx >= 0 &&
        state_idx < state_count &&
        raydn_cell >= 0 &&
        raydn_cell < cell_count &&
        material_idx >= 0 &&
        material_idx < material_count &&
        material_valid[material_idx];

    const int row = raydn_cell % grid_resolution0;
    const int col = raydn_cell / grid_resolution0;
    const int bdpt_cell = row_valid ? row * grid_resolution1 + col : -1;
    float row_contribution = 0.0f;
    float row_pdf = 0.0f;
    float direct_pdf = 0.0f;
    float keller_pdf = 0.0f;
    float row_path_length = 0.0f;
    if (row_valid) {
        const float t_min = state_edge_t_min[state_idx];
        const float t_max = state_edge_t_max[state_idx];
        const float edge_length = fmaxf(t_max - t_min, 0.0f);
        const float edge_t = t_min + tape_edge_u[lane] * (t_max - t_min);
        const float3 edge_origin = bdpt_vec3_at(state_edge_pos, state_idx);
        const float3 edge_dir = bdpt_normalize3(bdpt_vec3_at(state_edge_dir, state_idx));
        const float3 edge_point = bdpt_add3(edge_origin, bdpt_scale3(edge_dir, edge_t));
        const float3 source = bdpt_vec3_at(state_src, state_idx);
        const float3 target = bdpt_grid_cell_center(
            raydn_cell,
            grid_axis,
            grid_position,
            grid_coord0_min,
            grid_coord0_max,
            grid_coord1_min,
            grid_coord1_max,
            grid_resolution0,
            grid_resolution1);
        // The launch samples a state and a receiver cell in addition to the
        // continuous edge coordinate.  Preserve those discrete proposal
        // probabilities when reconstructing the exported sample; omitting
        // state_count * cell_count made the tape disagree with the map
        // accumulator by exactly that factor.
        const float discrete_domain =
            static_cast<float>(state_count) * static_cast<float>(cell_count);
        const float edge_measure_weight =
            edge_length * discrete_domain /
            fmaxf(static_cast<float>(strategy_samples), 1.0f);
        const float edge_pdf_base = 1.0f /
            fmaxf(edge_length * discrete_domain * grid_cell_area, 1.0e-30f);
        direct_pdf = direct_samples > 0 ? static_cast<float>(direct_samples) * edge_pdf_base : 0.0f;
        keller_pdf = keller_samples > 0 ? static_cast<float>(keller_samples) * edge_pdf_base : 0.0f;
        row_contribution = bdpt_diffraction_contribution(
            state_src_power[state_idx],
            material_gain[material_idx],
            wavelength,
            edge_measure_weight,
            grid_cell_area,
            state_exterior_angle[state_idx],
            source,
            edge_point,
            target);
        row_pdf = static_cast<float>(strategy_samples) * edge_pdf_base;
        row_path_length =
            bdpt_norm3(bdpt_sub3(edge_point, source)) +
            bdpt_norm3(bdpt_sub3(target, edge_point));
    }

    tx_id[lane] = tx_index;
    rx_id[lane] = bdpt_cell;
    grid_linear_id[lane] = bdpt_cell;
    component_id[lane] = 2;
    out_light_depth[lane] = 1;
    out_sensor_depth[lane] = 0;
    contribution[lane] = row_valid ? row_contribution : 0.0f;
    pdf[lane] = row_valid ? row_pdf : 0.0f;
    mis_weight[lane] = row_valid
        ? bdpt_diffraction_strategy_mis_weight(row_pdf, direct_pdf, keller_pdf, strategy_count, mode_id, beta)
        : 0.0f;
    valid[lane] = row_valid;
    path_length_m[lane] = row_path_length;
    const int64_t top = lane * 4;
    topology[top + 0] = tx_index;
    topology[top + 1] = bdpt_cell;
    topology[top + 2] = 2;
    topology[top + 3] = 1;
}

__global__ void bdpt_diffraction_point_connection_samples_kernel(
    int64_t count,
    int tx_index,
    int state_count,
    int rx_count,
    float wavelength,
    int direct_samples,
    int keller_samples,
    int mode_id,
    float beta,
    int strategy_count,
    int material_count,
    unsigned long long seed,
    const int* state_edge_index,
    const float* state_edge_pos,
    const float* state_edge_dir,
    const float* state_edge_t_min,
    const float* state_edge_t_max,
    const int* state_prim0,
    const int* state_prim1,
    const float* state_exterior_angle,
    const float* state_src,
    const float* state_src_power,
    const float* rx_positions,
    const float* material_gain,
    const bool* material_valid,
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
    float* path_length_m,
    float* source_start,
    float* source_end,
    float* target_start,
    float* target_end,
    bool* visibility_active) {
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= count) {
        return;
    }
    const int sample_id = rx_count > 0 ? static_cast<int>(lane / rx_count) : 0;
    const int rx = rx_count > 0 ? static_cast<int>(lane % rx_count) : -1;
    const bool lane_direct = sample_id < direct_samples;
    const bool lane_keller = !lane_direct && sample_id < direct_samples + keller_samples;
    const int strategy_samples = lane_direct ? direct_samples : (lane_keller ? keller_samples : 0);
    const unsigned long long lane_seed =
        seed ^ (static_cast<unsigned long long>(lane) * 0xd1b54a32d192ed03ULL);
    const float u_state = bdpt_uniform01_from_u64(bdpt_splitmix64(lane_seed ^ 0x6a09e667f3bcc909ULL));
    const float u_edge = bdpt_uniform01_from_u64(bdpt_splitmix64(lane_seed ^ 0xbb67ae8584caa73bULL));
    int state_idx = state_count > 0 ? static_cast<int>(floorf(u_state * static_cast<float>(state_count))) : -1;
    if (state_idx >= state_count) {
        state_idx = state_count - 1;
    }
    const int prim0 = state_idx >= 0 ? state_prim0[state_idx] : -1;
    const int prim1 = state_idx >= 0 ? state_prim1[state_idx] : -1;
    const int material_idx = (prim0 >= 0 && prim0 < material_count) ? prim0 : prim1;
    const bool material_ok =
        material_idx >= 0 &&
        material_idx < material_count &&
        material_valid[material_idx];
    const bool row_valid =
        strategy_samples > 0 &&
        state_idx >= 0 &&
        state_idx < state_count &&
        rx >= 0 &&
        rx < rx_count &&
        material_ok;

    float3 source = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float3 edge_point = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float3 target = bdpt_make_float3(0.0f, 0.0f, 0.0f);
    float row_contribution = 0.0f;
    float row_pdf = 0.0f;
    float direct_pdf = 0.0f;
    float keller_pdf = 0.0f;
    float row_path_length = 0.0f;
    if (row_valid) {
        const float t_min = state_edge_t_min[state_idx];
        const float t_max = state_edge_t_max[state_idx];
        const float edge_length = fmaxf(t_max - t_min, 0.0f);
        const float edge_t = t_min + u_edge * (t_max - t_min);
        const float3 edge_origin = bdpt_vec3_at(state_edge_pos, state_idx);
        const float3 edge_dir = bdpt_normalize3(bdpt_vec3_at(state_edge_dir, state_idx));
        edge_point = bdpt_add3(edge_origin, bdpt_scale3(edge_dir, edge_t));
        source = bdpt_vec3_at(state_src, state_idx);
        target = bdpt_vec3_at(rx_positions, rx);
        const float edge_measure_weight =
            edge_length * fmaxf(static_cast<float>(state_count), 1.0f) /
            fmaxf(static_cast<float>(strategy_samples), 1.0f);
        const float edge_pdf_base =
            1.0f / fmaxf(static_cast<float>(state_count) * edge_length, 1.0e-30f);
        direct_pdf = direct_samples > 0 ? static_cast<float>(direct_samples) * edge_pdf_base : 0.0f;
        keller_pdf = keller_samples > 0 ? static_cast<float>(keller_samples) * edge_pdf_base : 0.0f;
        row_contribution = bdpt_diffraction_contribution(
            state_src_power[state_idx],
            material_gain[material_idx],
            wavelength,
            edge_measure_weight,
            1.0f,
            state_exterior_angle[state_idx],
            source,
            edge_point,
            target);
        row_pdf = static_cast<float>(strategy_samples) * edge_pdf_base;
        row_path_length =
            bdpt_norm3(bdpt_sub3(edge_point, source)) +
            bdpt_norm3(bdpt_sub3(target, edge_point));
    }

    tx_id[lane] = tx_index;
    rx_id[lane] = row_valid ? rx : -1;
    grid_linear_id[lane] = row_valid ? rx : -1;
    component_id[lane] = 2;
    out_light_depth[lane] = 1;
    out_sensor_depth[lane] = 0;
    contribution[lane] = row_valid ? row_contribution : 0.0f;
    pdf[lane] = row_valid ? row_pdf : 0.0f;
    mis_weight[lane] = row_valid
        ? bdpt_diffraction_strategy_mis_weight(row_pdf, direct_pdf, keller_pdf, strategy_count, mode_id, beta)
        : 0.0f;
    valid[lane] = row_valid;
    path_length_m[lane] = row_valid ? row_path_length : 0.0f;
    const int64_t top = lane * 4;
    topology[top + 0] = tx_index;
    topology[top + 1] = row_valid ? rx : -1;
    topology[top + 2] = 2;
    topology[top + 3] = 1;
    const int64_t vec = lane * 3;
    source_start[vec + 0] = source.x;
    source_start[vec + 1] = source.y;
    source_start[vec + 2] = source.z;
    source_end[vec + 0] = edge_point.x;
    source_end[vec + 1] = edge_point.y;
    source_end[vec + 2] = edge_point.z;
    target_start[vec + 0] = edge_point.x;
    target_start[vec + 1] = edge_point.y;
    target_start[vec + 2] = edge_point.z;
    target_end[vec + 0] = target.x;
    target_end[vec + 1] = target.y;
    target_end[vec + 2] = target.z;
    visibility_active[lane] = row_valid;
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
cn_bdpt_endpoint_connection_samples_cuda(
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
cn_bdpt_diffraction_connection_samples_from_tape_cuda(
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    check_bool_cuda(tape_active, "tape_active", 1);
    check_int_cuda(tape_state_idx, "tape_state_idx", 1);
    check_int_cuda(tape_cell, "tape_cell", 1);
    check_int_cuda(tape_material_idx, "tape_material_idx", 1);
    check_float_cuda(tape_edge_u, "tape_edge_u", 1);
    check_int_cuda(state_edge_index, "state_edge_index", 1);
    check_vec3_cuda(state_edge_pos, "state_edge_pos");
    check_vec3_cuda(state_edge_dir, "state_edge_dir");
    check_float_cuda(state_edge_t_min, "state_edge_t_min", 1);
    check_float_cuda(state_edge_t_max, "state_edge_t_max", 1);
    check_float_cuda(state_exterior_angle, "state_exterior_angle", 1);
    check_vec3_cuda(state_src, "state_src");
    check_float_cuda(state_src_power, "state_src_power", 1);
    check_float_cuda(material_gain, "material_gain", 1);
    check_bool_cuda(material_valid, "material_valid", 1);
    TORCH_CHECK(tx_index >= 0, "tx_index must be non-negative");
    TORCH_CHECK(state_count >= 0, "state_count must be non-negative");
    TORCH_CHECK(grid_axis >= 0 && grid_axis <= 2, "grid_axis must be 0, 1, or 2");
    TORCH_CHECK(grid_resolution0 > 0 && grid_resolution1 > 0, "grid resolutions must be positive");
    TORCH_CHECK(grid_cell_area > 0.0, "grid_cell_area must be positive");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(direct_samples >= 0 && keller_samples >= 0, "sample counts must be non-negative");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_diffraction_mis_args(mode_id, strategy_count, direct_samples, keller_samples);
    const int64_t count = tape_active.size(0);
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&tape_state_idx, "tape_state_idx"),
             std::pair<const at::Tensor*, const char*>(&tape_cell, "tape_cell"),
             std::pair<const at::Tensor*, const char*>(&tape_material_idx, "tape_material_idx"),
             std::pair<const at::Tensor*, const char*>(&tape_edge_u, "tape_edge_u"),
         }) {
        TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match tape_active");
        check_same_device(*pair.first, tape_active, pair.second);
    }
    const int64_t physical_state_count = state_edge_index.size(0);
    TORCH_CHECK(state_count <= physical_state_count, "state_count exceeds state payload width");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&state_edge_pos, "state_edge_pos"),
             std::pair<const at::Tensor*, const char*>(&state_edge_dir, "state_edge_dir"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_min, "state_edge_t_min"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_max, "state_edge_t_max"),
             std::pair<const at::Tensor*, const char*>(&state_exterior_angle, "state_exterior_angle"),
             std::pair<const at::Tensor*, const char*>(&state_src, "state_src"),
             std::pair<const at::Tensor*, const char*>(&state_src_power, "state_src_power"),
         }) {
        TORCH_CHECK(pair.first->size(0) >= state_count, pair.second, " must cover state_count");
        check_same_device(*pair.first, tape_active, pair.second);
    }
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    check_same_device(material_gain, tape_active, "material_gain");
    check_same_device(material_valid, tape_active, "material_valid");
    check_same_device(state_edge_index, tape_active, "state_edge_index");
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
        path_length_m] = allocate_connection_samples(tape_edge_u, count);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tape_active.get_device()).stream();
        bdpt_diffraction_connection_samples_from_tape_kernel<<<blocks, threads, 0, stream>>>(
            count,
            static_cast<int>(tx_index),
            static_cast<int>(state_count),
            static_cast<int>(grid_resolution0),
            static_cast<int>(grid_resolution1),
            static_cast<int>(grid_axis),
            static_cast<float>(grid_position),
            static_cast<float>(grid_coord0_min),
            static_cast<float>(grid_coord0_max),
            static_cast<float>(grid_coord1_min),
            static_cast<float>(grid_coord1_max),
            static_cast<float>(grid_cell_area),
            static_cast<float>(wavelength),
            static_cast<int>(direct_samples),
            static_cast<int>(keller_samples),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            static_cast<int>(material_gain.size(0)),
            tape_active.data_ptr<bool>(),
            tape_state_idx.data_ptr<int>(),
            tape_cell.data_ptr<int>(),
            tape_material_idx.data_ptr<int>(),
            tape_edge_u.data_ptr<float>(),
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_edge_t_min.data_ptr<float>(),
            state_edge_t_max.data_ptr<float>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
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
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cn_bdpt_diffraction_point_connection_samples_cuda(
    at::Tensor rx_positions,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t tx_index,
    int64_t state_count,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t seed,
    double wavelength,
    int64_t mode_id,
    double beta,
    int64_t strategy_count) {
    check_vec3_cuda(rx_positions, "rx_positions");
    check_int_cuda(state_edge_index, "state_edge_index", 1);
    check_vec3_cuda(state_edge_pos, "state_edge_pos");
    check_vec3_cuda(state_edge_dir, "state_edge_dir");
    check_float_cuda(state_edge_t_min, "state_edge_t_min", 1);
    check_float_cuda(state_edge_t_max, "state_edge_t_max", 1);
    check_int_cuda(state_prim0, "state_prim0", 1);
    check_int_cuda(state_prim1, "state_prim1", 1);
    check_float_cuda(state_exterior_angle, "state_exterior_angle", 1);
    check_vec3_cuda(state_src, "state_src");
    check_float_cuda(state_src_power, "state_src_power", 1);
    check_float_cuda(material_gain, "material_gain", 1);
    check_bool_cuda(material_valid, "material_valid", 1);
    TORCH_CHECK(tx_index >= 0, "tx_index must be non-negative");
    TORCH_CHECK(state_count >= 0, "state_count must be non-negative");
    TORCH_CHECK(direct_samples >= 0 && keller_samples >= 0, "diffraction sample counts must be non-negative");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    check_diffraction_mis_args(mode_id, strategy_count, direct_samples, keller_samples);
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    const int64_t physical_state_count = state_edge_index.size(0);
    TORCH_CHECK(state_count <= physical_state_count, "state_count exceeds state tensors");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>(&state_edge_pos, "state_edge_pos"),
             std::pair<const at::Tensor*, const char*>(&state_edge_dir, "state_edge_dir"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_min, "state_edge_t_min"),
             std::pair<const at::Tensor*, const char*>(&state_edge_t_max, "state_edge_t_max"),
             std::pair<const at::Tensor*, const char*>(&state_prim0, "state_prim0"),
             std::pair<const at::Tensor*, const char*>(&state_prim1, "state_prim1"),
             std::pair<const at::Tensor*, const char*>(&state_exterior_angle, "state_exterior_angle"),
             std::pair<const at::Tensor*, const char*>(&state_src, "state_src"),
             std::pair<const at::Tensor*, const char*>(&state_src_power, "state_src_power"),
         }) {
        TORCH_CHECK(pair.first->size(0) == physical_state_count, pair.second, " must match state_edge_index");
        check_same_device(*pair.first, rx_positions, pair.second);
    }
    check_same_device(state_edge_index, rx_positions, "state_edge_index");
    check_same_device(material_gain, rx_positions, "material_gain");
    check_same_device(material_valid, rx_positions, "material_valid");
    const int64_t rx_count = rx_positions.size(0);
    const int64_t samples_per_rx = direct_samples + keller_samples;
    const int64_t count = state_count > 0 ? rx_count * samples_per_rx : 0;
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
        path_length_m] = allocate_connection_samples(rx_positions, count);
    auto source_start = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto source_end = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto target_start = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto target_end = at::empty({count, 3}, rx_positions.options().dtype(at::kFloat));
    auto visibility_active = at::empty({count}, rx_positions.options().dtype(at::kBool));
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(rx_positions.get_device()).stream();
        bdpt_diffraction_point_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            static_cast<int>(tx_index),
            static_cast<int>(state_count),
            static_cast<int>(rx_count),
            static_cast<float>(wavelength),
            static_cast<int>(direct_samples),
            static_cast<int>(keller_samples),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            static_cast<int>(strategy_count),
            static_cast<int>(material_gain.size(0)),
            static_cast<unsigned long long>(seed),
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_edge_t_min.data_ptr<float>(),
            state_edge_t_max.data_ptr<float>(),
            state_prim0.data_ptr<int>(),
            state_prim1.data_ptr<int>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            material_gain.data_ptr<float>(),
            material_valid.data_ptr<bool>(),
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
            path_length_m.data_ptr<float>(),
            source_start.data_ptr<float>(),
            source_end.data_ptr<float>(),
            target_start.data_ptr<float>(),
            target_end.data_ptr<float>(),
            visibility_active.data_ptr<bool>());
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
        path_length_m,
        source_start,
        source_end,
        target_start,
        target_end,
        visibility_active};
}
