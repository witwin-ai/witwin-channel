#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace raydn {

struct OptixPipelineConfig {
    const char *ptx = nullptr;
    size_t ptx_size = 0;
    std::vector<const char *> raygen_entries;
    const char *miss_entry = nullptr;
    const char *closesthit_entry = nullptr;
    const char *anyhit_entry = nullptr;
    int num_payload_values = 0;
    size_t params_size = 0;
};

class OptixLaunchPipeline {
public:
    OptixLaunchPipeline() = default;
    ~OptixLaunchPipeline();

    OptixLaunchPipeline(const OptixLaunchPipeline &) = delete;
    OptixLaunchPipeline &operator=(const OptixLaunchPipeline &) = delete;

    void build(
        OptixDeviceContext context,
        int device_index,
        int hitgroup_record_count,
        const OptixPipelineConfig &config);

    bool is_ready() const {
        return ready_;
    }

    template <typename Params>
    void launch(int raygen_index, const Params &params, unsigned int n_rays, cudaStream_t stream) {
        launch_impl(raygen_index, &params, sizeof(Params), n_rays, stream);
    }

private:
    void launch_impl(
        int raygen_index,
        const void *params,
        size_t actual_params_size,
        unsigned int n_rays,
        cudaStream_t stream);

    bool ready_ = false;
    int device_index_ = 0;
    int hitgroup_record_count_ = 0;
    size_t params_size_ = 0;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    std::vector<OptixProgramGroup> raygen_groups_;
    OptixProgramGroup miss_group_ = nullptr;
    OptixProgramGroup hitgroup_ = nullptr;
    std::vector<at::Tensor> raygen_records_;
    at::Tensor miss_record_;
    at::Tensor hitgroup_records_;
    at::Tensor params_buffer_;
};

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int device_index,
    int hitgroup_record_count,
    const OptixPipelineConfig &config);

} // namespace raydn
