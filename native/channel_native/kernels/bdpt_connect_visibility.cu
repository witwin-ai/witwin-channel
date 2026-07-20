#include "bdpt_connect_common.cuh"

#define CN_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS()                                     \
    check_int_cuda(topology, "topology", 2);                                          \
    TORCH_CHECK(topology.size(1) == 4, "topology must have shape (N, 4)");             \
    check_float_cuda(contribution, "contribution", 1);                                \
    check_float_cuda(pdf, "pdf", 1);                                                  \
    check_float_cuda(mis_weight, "mis_weight", 1);                                    \
    check_int_cuda(component_id, "component_id", 1);                                  \
    check_bool_cuda(valid, "valid", 1);                                               \
    check_int_cuda(tx_id, "tx_id", 1);                                                \
    check_int_cuda(rx_id, "rx_id", 1);                                                \
    check_int_cuda(grid_linear_id, "grid_linear_id", 1);                              \
    check_int_cuda(light_depth, "light_depth", 1);                                    \
    check_int_cuda(sensor_depth, "sensor_depth", 1);                                  \
    check_float_cuda(path_length_m, "path_length_m", 1)

#define CN_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(REFERENCE)                               \
    for (const auto& pair : {                                                          \
             std::pair<const at::Tensor*, const char*>(&pdf, "pdf"),                  \
             std::pair<const at::Tensor*, const char*>(&mis_weight, "mis_weight"),    \
             std::pair<const at::Tensor*, const char*>(&component_id, "component_id"),\
             std::pair<const at::Tensor*, const char*>(&valid, "valid"),              \
             std::pair<const at::Tensor*, const char*>(&tx_id, "tx_id"),              \
             std::pair<const at::Tensor*, const char*>(&rx_id, "rx_id"),              \
             std::pair<const at::Tensor*, const char*>(&grid_linear_id, "grid_linear_id"),\
             std::pair<const at::Tensor*, const char*>(&light_depth, "light_depth"),  \
             std::pair<const at::Tensor*, const char*>(&sensor_depth, "sensor_depth"),\
             std::pair<const at::Tensor*, const char*>(&path_length_m, "path_length_m"),\
         }) {                                                                          \
        TORCH_CHECK(pair.first->size(0) == count, pair.second, " must match contribution");\
        check_same_device(*pair.first, REFERENCE, pair.second);                        \
    }

#define CN_BDPT_CONNECTION_OUTPUT_POINTERS()                                           \
    out_topology.data_ptr<int>(),                                                      \
    out_contribution.data_ptr<float>(),                                                \
    out_pdf.data_ptr<float>(),                                                         \
    out_mis_weight.data_ptr<float>(),                                                  \
    out_component_id.data_ptr<int>(),                                                  \
    out_valid.data_ptr<bool>(),                                                        \
    out_tx_id.data_ptr<int>(),                                                         \
    out_rx_id.data_ptr<int>(),                                                         \
    out_grid_linear_id.data_ptr<int>(),                                                \
    out_light_depth.data_ptr<int>(),                                                   \
    out_sensor_depth.data_ptr<int>(),                                                  \
    out_path_length_m.data_ptr<float>()

namespace {

__global__ void bdpt_endpoint_connection_visibility_inputs_kernel(
    int64_t count,
    int64_t sensor_count,
    const float* light_origin,
    const int* light_tx_id,
    const bool* light_valid,
    const float* sensor_origin,
    const int* sensor_rx_id,
    const bool* sensor_valid,
    float* start,
    float* end,
    bool* active) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t light_index = index / sensor_count;
    const int64_t sensor_index = index - light_index * sensor_count;
    const float* src = light_origin + light_index * 3;
    const float* dst = sensor_origin + sensor_index * 3;
    float* out_start = start + index * 3;
    float* out_end = end + index * 3;
    out_start[0] = src[0];
    out_start[1] = src[1];
    out_start[2] = src[2];
    out_end[0] = dst[0];
    out_end[1] = dst[1];
    out_end[2] = dst[2];
    active[index] = light_valid[light_index] && sensor_valid[sensor_index] &&
        light_tx_id[light_index] >= 0 && sensor_rx_id[sensor_index] >= 0;
}

__global__ void bdpt_filter_connection_samples_kernel(
    int64_t count,
    const bool* visible,
    float* contribution,
    float* pdf,
    float* mis_weight,
    bool* valid) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const bool keep = valid[index] && visible[index];
    valid[index] = keep;
    if (!keep) {
        contribution[index] = 0.0f;
        pdf[index] = 0.0f;
        mis_weight[index] = 0.0f;
    }
}

__global__ void bdpt_compact_connection_samples_kernel(
    int64_t count,
    int64_t capacity,
    const int* topology,
    const float* contribution,
    const float* pdf,
    const float* mis_weight,
    const int* component_id,
    const bool* valid,
    const int* tx_id,
    const int* rx_id,
    const int* grid_linear_id,
    const int* light_depth,
    const int* sensor_depth,
    const float* path_length_m,
    int* compact_count,
    int* out_topology,
    float* out_contribution,
    float* out_pdf,
    float* out_mis_weight,
    int* out_component_id,
    bool* out_valid,
    int* out_tx_id,
    int* out_rx_id,
    int* out_grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* out_path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    const int slot = atomicAdd(compact_count, 1);
    if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
        return;
    }
    const int64_t src_row = index * 4;
    const int64_t dst_row = static_cast<int64_t>(slot) * 4;
    out_topology[dst_row + 0] = topology[src_row + 0];
    out_topology[dst_row + 1] = topology[src_row + 1];
    out_topology[dst_row + 2] = topology[src_row + 2];
    out_topology[dst_row + 3] = topology[src_row + 3];
    out_contribution[slot] = contribution[index];
    out_pdf[slot] = pdf[index];
    out_mis_weight[slot] = mis_weight[index];
    out_component_id[slot] = component_id[index];
    out_valid[slot] = true;
    out_tx_id[slot] = tx_id[index];
    out_rx_id[slot] = rx_id[index];
    out_grid_linear_id[slot] = grid_linear_id[index];
    out_light_depth[slot] = light_depth[index];
    out_sensor_depth[slot] = sensor_depth[index];
    out_path_length_m[slot] = path_length_m[index];
}

__global__ void bdpt_copy_connection_samples_kernel(
    int64_t count,
    int64_t dst_offset,
    const int* topology,
    const float* contribution,
    const float* pdf,
    const float* mis_weight,
    const int* component_id,
    const bool* valid,
    const int* tx_id,
    const int* rx_id,
    const int* grid_linear_id,
    const int* light_depth,
    const int* sensor_depth,
    const float* path_length_m,
    int* out_topology,
    float* out_contribution,
    float* out_pdf,
    float* out_mis_weight,
    int* out_component_id,
    bool* out_valid,
    int* out_tx_id,
    int* out_rx_id,
    int* out_grid_linear_id,
    int* out_light_depth,
    int* out_sensor_depth,
    float* out_path_length_m) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t dst = dst_offset + index;
    const int64_t src_row = index * 4;
    const int64_t dst_row = dst * 4;
    out_topology[dst_row + 0] = topology[src_row + 0];
    out_topology[dst_row + 1] = topology[src_row + 1];
    out_topology[dst_row + 2] = topology[src_row + 2];
    out_topology[dst_row + 3] = topology[src_row + 3];
    out_contribution[dst] = contribution[index];
    out_pdf[dst] = pdf[index];
    out_mis_weight[dst] = mis_weight[index];
    out_component_id[dst] = component_id[index];
    out_valid[dst] = valid[index];
    out_tx_id[dst] = tx_id[index];
    out_rx_id[dst] = rx_id[index];
    out_grid_linear_id[dst] = grid_linear_id[index];
    out_light_depth[dst] = light_depth[index];
    out_sensor_depth[dst] = sensor_depth[index];
    out_path_length_m[dst] = path_length_m[index];
}

__global__ void bdpt_count_valid_connection_samples_kernel(
    int64_t count,
    const bool* valid,
    int* compact_count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !valid[index]) {
        return;
    }
    atomicAdd(compact_count, 1);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_bdpt_endpoint_connection_visibility_inputs_cuda(
    at::Tensor light_origin,
    at::Tensor light_tx_id,
    at::Tensor light_valid,
    at::Tensor sensor_origin,
    at::Tensor sensor_rx_id,
    at::Tensor sensor_valid,
    int64_t sample_count) {
    check_vec3_cuda(light_origin, "light_origin");
    check_int_cuda(light_tx_id, "light_tx_id", 1);
    check_bool_cuda(light_valid, "light_valid", 1);
    check_vec3_cuda(sensor_origin, "sensor_origin");
    check_int_cuda(sensor_rx_id, "sensor_rx_id", 1);
    check_bool_cuda(sensor_valid, "sensor_valid", 1);
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");
    const int64_t light_count = light_origin.size(0);
    const int64_t sensor_count = sensor_origin.size(0);
    TORCH_CHECK(light_tx_id.size(0) == light_count, "light_tx_id must match light count");
    TORCH_CHECK(light_valid.size(0) == light_count, "light_valid must match light count");
    TORCH_CHECK(sensor_rx_id.size(0) == sensor_count, "sensor_rx_id must match sensor count");
    TORCH_CHECK(sensor_valid.size(0) == sensor_count, "sensor_valid must match sensor count");
    check_same_device(light_tx_id, light_origin, "light_tx_id");
    check_same_device(light_valid, light_origin, "light_valid");
    check_same_device(sensor_origin, light_origin, "sensor_origin");
    check_same_device(sensor_rx_id, light_origin, "sensor_rx_id");
    check_same_device(sensor_valid, light_origin, "sensor_valid");
    TORCH_CHECK(
        sensor_count > 0 || sample_count == 0,
        "sensor count must be positive when sample_count is positive");
    TORCH_CHECK(sample_count <= light_count * sensor_count, "sample_count exceeds endpoint pair count");
    auto start = at::empty({sample_count, 3}, light_origin.options());
    auto end = at::empty({sample_count, 3}, light_origin.options());
    auto active = at::empty({sample_count}, light_origin.options().dtype(at::kBool));
    if (sample_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((sample_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(light_origin.get_device()).stream();
        bdpt_endpoint_connection_visibility_inputs_kernel<<<blocks, threads, 0, stream>>>(
            sample_count,
            sensor_count,
            light_origin.data_ptr<float>(),
            light_tx_id.data_ptr<int>(),
            light_valid.data_ptr<bool>(),
            sensor_origin.data_ptr<float>(),
            sensor_rx_id.data_ptr<int>(),
            sensor_valid.data_ptr<bool>(),
            start.data_ptr<float>(),
            end.data_ptr<float>(),
            active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {start, end, active};
}

void cn_bdpt_filter_connection_samples_cuda(
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor valid,
    at::Tensor visible) {
    check_float_cuda(contribution, "contribution", 1);
    check_float_cuda(pdf, "pdf", 1);
    check_float_cuda(mis_weight, "mis_weight", 1);
    check_bool_cuda(valid, "valid", 1);
    check_bool_cuda(visible, "visible", 1);
    TORCH_CHECK(pdf.sizes() == contribution.sizes(), "pdf must match contribution");
    TORCH_CHECK(mis_weight.sizes() == contribution.sizes(), "mis_weight must match contribution");
    TORCH_CHECK(valid.sizes() == contribution.sizes(), "valid must match contribution");
    TORCH_CHECK(visible.sizes() == contribution.sizes(), "visible must match contribution");
    check_same_device(pdf, contribution, "pdf");
    check_same_device(mis_weight, contribution, "mis_weight");
    check_same_device(valid, contribution, "valid");
    check_same_device(visible, contribution, "visible");
    const int64_t count = contribution.size(0);
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_filter_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            visible.data_ptr<bool>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            valid.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

int64_t cn_bdpt_count_valid_connection_samples_cuda(at::Tensor valid) {
    check_bool_cuda(valid, "valid", 1);
    const int64_t count = valid.size(0);
    int valid_count_host = 0;
    if (count > 0) {
        auto compact_count = at::empty({}, valid.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            valid.data_ptr<bool>(),
            compact_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &valid_count_host,
            compact_count.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return static_cast<int64_t>(valid_count_host);
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
cn_bdpt_compact_connection_samples_cuda(
    at::Tensor topology,
    at::Tensor contribution,
    at::Tensor pdf,
    at::Tensor mis_weight,
    at::Tensor component_id,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor grid_linear_id,
    at::Tensor light_depth,
    at::Tensor sensor_depth,
    at::Tensor path_length_m,
    int64_t max_paths) {
    CN_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS();
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    const int64_t count = contribution.size(0);
    CN_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(contribution);
    TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
    check_same_device(topology, contribution, "topology");
    int valid_count_host = 0;
    if (count > 0) {
        auto compact_count = at::empty({}, contribution.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_count_valid_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            valid.data_ptr<bool>(),
            compact_count.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &valid_count_host,
            compact_count.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const int64_t capacity = max_paths < 0
        ? static_cast<int64_t>(valid_count_host)
        : std::min<int64_t>(max_paths, static_cast<int64_t>(valid_count_host));
    auto [
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m] = allocate_connection_samples(contribution, capacity);
    if (capacity > 0 && count > 0) {
        auto compact_count = at::empty({}, contribution.options().dtype(at::kInt));
        zero_int_tensor(compact_count);
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(contribution.get_device()).stream();
        bdpt_compact_connection_samples_kernel<<<blocks, threads, 0, stream>>>(
            count,
            capacity,
            topology.data_ptr<int>(),
            contribution.data_ptr<float>(),
            pdf.data_ptr<float>(),
            mis_weight.data_ptr<float>(),
            component_id.data_ptr<int>(),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            grid_linear_id.data_ptr<int>(),
            light_depth.data_ptr<int>(),
            sensor_depth.data_ptr<int>(),
            path_length_m.data_ptr<float>(),
            compact_count.data_ptr<int>(),
            CN_BDPT_CONNECTION_OUTPUT_POINTERS());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m};
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
cn_bdpt_concat_connection_samples_cuda(
    std::vector<at::Tensor> topologies,
    std::vector<at::Tensor> contributions,
    std::vector<at::Tensor> pdfs,
    std::vector<at::Tensor> mis_weights,
    std::vector<at::Tensor> component_ids,
    std::vector<at::Tensor> valids,
    std::vector<at::Tensor> tx_ids,
    std::vector<at::Tensor> rx_ids,
    std::vector<at::Tensor> grid_linear_ids,
    std::vector<at::Tensor> light_depths,
    std::vector<at::Tensor> sensor_depths,
    std::vector<at::Tensor> path_lengths_m) {
    const size_t block_count = contributions.size();
    TORCH_CHECK(block_count > 0, "bdpt_concat_connection_samples requires at least one block");
    TORCH_CHECK(topologies.size() == block_count, "topologies must match block count");
    TORCH_CHECK(pdfs.size() == block_count, "pdfs must match block count");
    TORCH_CHECK(mis_weights.size() == block_count, "mis_weights must match block count");
    TORCH_CHECK(component_ids.size() == block_count, "component_ids must match block count");
    TORCH_CHECK(valids.size() == block_count, "valids must match block count");
    TORCH_CHECK(tx_ids.size() == block_count, "tx_ids must match block count");
    TORCH_CHECK(rx_ids.size() == block_count, "rx_ids must match block count");
    TORCH_CHECK(grid_linear_ids.size() == block_count, "grid_linear_ids must match block count");
    TORCH_CHECK(light_depths.size() == block_count, "light_depths must match block count");
    TORCH_CHECK(sensor_depths.size() == block_count, "sensor_depths must match block count");
    TORCH_CHECK(path_lengths_m.size() == block_count, "path_lengths_m must match block count");
    const at::Tensor& reference = contributions[0];
    check_float_cuda(reference, "contribution[0]", 1);
    int64_t total = 0;
    for (size_t block = 0; block < block_count; ++block) {
        at::Tensor& topology = topologies[block];
        at::Tensor& contribution = contributions[block];
        at::Tensor& pdf = pdfs[block];
        at::Tensor& mis_weight = mis_weights[block];
        at::Tensor& component_id = component_ids[block];
        at::Tensor& valid = valids[block];
        at::Tensor& tx_id = tx_ids[block];
        at::Tensor& rx_id = rx_ids[block];
        at::Tensor& grid_linear_id = grid_linear_ids[block];
        at::Tensor& light_depth = light_depths[block];
        at::Tensor& sensor_depth = sensor_depths[block];
        at::Tensor& path_length_m = path_lengths_m[block];
        CN_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS();
        const int64_t count = contribution.size(0);
        TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
        check_same_device(contribution, reference, "contribution");
        TORCH_CHECK(topology.size(0) == count, "topology must match contribution");
        check_same_device(topology, reference, "topology");
        CN_BDPT_CHECK_CONNECTION_SAMPLE_ROWS(reference);
        total += count;
    }
    auto [
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m] = allocate_connection_samples(reference, total);
    int64_t offset = 0;
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    for (size_t block = 0; block < block_count; ++block) {
        const int64_t count = contributions[block].size(0);
        if (count > 0) {
            int grid = static_cast<int>((count + threads - 1) / threads);
            bdpt_copy_connection_samples_kernel<<<grid, threads, 0, stream>>>(
                count,
                offset,
                topologies[block].data_ptr<int>(),
                contributions[block].data_ptr<float>(),
                pdfs[block].data_ptr<float>(),
                mis_weights[block].data_ptr<float>(),
                component_ids[block].data_ptr<int>(),
                valids[block].data_ptr<bool>(),
                tx_ids[block].data_ptr<int>(),
                rx_ids[block].data_ptr<int>(),
                grid_linear_ids[block].data_ptr<int>(),
                light_depths[block].data_ptr<int>(),
                sensor_depths[block].data_ptr<int>(),
                path_lengths_m[block].data_ptr<float>(),
                CN_BDPT_CONNECTION_OUTPUT_POINTERS());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        offset += count;
    }
    return {
        out_topology,
        out_contribution,
        out_pdf,
        out_mis_weight,
        out_component_id,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_grid_linear_id,
        out_light_depth,
        out_sensor_depth,
        out_path_length_m};
}

#undef CN_BDPT_CONNECTION_OUTPUT_POINTERS
#undef CN_BDPT_CHECK_CONNECTION_SAMPLE_ROWS
#undef CN_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS
