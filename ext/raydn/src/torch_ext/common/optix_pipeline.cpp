#include <raydn/common/optix_pipeline.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <raydn/common/optix_context.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace raydn {

namespace {

struct EmptySbtData {
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using EmptySbtRecord = SbtRecord<EmptySbtData>;

using PipelineCacheKey = std::tuple<
    OptixDeviceContext,
    const char *,
    size_t,
    std::string,
    std::string,
    std::string,
    std::string,
    int,
    int,
    size_t>;

std::mutex &pipeline_cache_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> &pipeline_cache() {
    static std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> *cache =
        new std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>();
    return *cache;
}

std::string entry_key(const char *entry) {
    return entry != nullptr ? std::string(entry) : std::string();
}

std::string raygen_key(const std::vector<const char *> &entries) {
    std::string key;
    for (const char *entry : entries) {
        if (!key.empty())
            key.push_back('\n');
        key += entry_key(entry);
    }
    return key;
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void create_program_group(
    OptixDeviceContext context,
    const OptixProgramGroupDesc &desc,
    OptixProgramGroup *out_group) {
    OptixProgramGroupOptions options = {};
    char log[4096] = {};
    size_t log_size = sizeof(log);
    raydn_OPTIX_CHECK(
        optixProgramGroupCreate(context, &desc, 1, &options, log, &log_size, out_group));
}

at::Tensor make_sbt_record(
    int device_index,
    OptixProgramGroup program_group,
    cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    raydn_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
    at::Tensor record = at::empty(
        {static_cast<int64_t>(sizeof(EmptySbtRecord))},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(
        cudaMemcpyAsync(
            record.data_ptr<uint8_t>(),
            &host_record,
            sizeof(host_record),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(multipath SBT record)");
    return record;
}

int hitgroup_record_capacity(int hitgroup_record_count) {
    constexpr int kMinHitgroupRecordCapacity = 64;
    int capacity = kMinHitgroupRecordCapacity;
    while (capacity < hitgroup_record_count)
        capacity *= 2;
    return capacity;
}

} // namespace

OptixLaunchPipeline::~OptixLaunchPipeline() {
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr)
        optixPipelineDestroy(pipeline_);
    if (hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr)
        optixProgramGroupDestroy(hitgroup_);
    if (miss_group_ != nullptr && optixProgramGroupDestroy != nullptr)
        optixProgramGroupDestroy(miss_group_);
    for (OptixProgramGroup group : raygen_groups_) {
        if (group != nullptr && optixProgramGroupDestroy != nullptr)
            optixProgramGroupDestroy(group);
    }
    if (module_ != nullptr && optixModuleDestroy != nullptr)
        optixModuleDestroy(module_);
}

void OptixLaunchPipeline::build(
    OptixDeviceContext context,
    int device_index,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
    if (context == nullptr)
        throw std::runtime_error("OptixLaunchPipeline::build(): invalid OptiX context.");
    if (hitgroup_record_count <= 0)
        throw std::runtime_error("OptixLaunchPipeline::build(): invalid hitgroup count.");
    if (config.raygen_entries.empty())
        throw std::runtime_error("OptixLaunchPipeline::build(): missing raygen entry.");

    c10::cuda::CUDAGuard guard(device_index);
    device_index_ = device_index;

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = config.num_payload_values;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = false;

    char log[8192] = {};
    size_t log_size = sizeof(log);
    OptixResult result = optixModuleCreate(
        context,
        &module_options,
        &pipeline_options,
        config.ptx,
        config.ptx_size,
        log,
        &log_size,
        &module_);
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string("OptiX error in optixModuleCreate(multipath): code=") +
            std::to_string(static_cast<int>(result)) + " log=" + std::string(log, log_size));
    }

    for (const char *entry : config.raygen_entries) {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = entry;
        OptixProgramGroup group = nullptr;
        create_program_group(context, desc, &group);
        raygen_groups_.push_back(group);
    }

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = config.miss_entry;
    create_program_group(context, miss_desc, &miss_group_);

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = config.closesthit_entry;
    hitgroup_desc.hitgroup.moduleAH = config.anyhit_entry != nullptr ? module_ : nullptr;
    hitgroup_desc.hitgroup.entryFunctionNameAH = config.anyhit_entry;
    create_program_group(context, hitgroup_desc, &hitgroup_);

    std::vector<OptixProgramGroup> program_groups = raygen_groups_;
    program_groups.push_back(miss_group_);
    program_groups.push_back(hitgroup_);

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 2;

    log_size = sizeof(log);
    result = optixPipelineCreate(
        context,
        &pipeline_options,
        &link_options,
        program_groups.data(),
        static_cast<unsigned int>(program_groups.size()),
        log,
        &log_size,
        &pipeline_);
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string("OptiX error in optixPipelineCreate(multipath): code=") +
            std::to_string(static_cast<int>(result)) + " log=" + std::string(log, log_size));
    }
    raydn_OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    for (OptixProgramGroup group : raygen_groups_)
        raygen_records_.push_back(make_sbt_record(device_index, group, stream));
    miss_record_ = make_sbt_record(device_index, miss_group_, stream);

    std::vector<EmptySbtRecord> hitgroup_host(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_host)
        raydn_OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_, &record));
    hitgroup_records_ = at::empty(
        {static_cast<int64_t>(sizeof(EmptySbtRecord) * hitgroup_host.size())},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(
        cudaMemcpyAsync(
            hitgroup_records_.data_ptr<uint8_t>(),
            hitgroup_host.data(),
            sizeof(EmptySbtRecord) * hitgroup_host.size(),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(multipath hitgroup records)");

    params_size_ = config.params_size;
    params_buffer_ = at::empty(
        {static_cast<int64_t>(std::max<size_t>(params_size_, 1024))},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int device_index,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
    const int hitgroup_capacity = hitgroup_record_capacity(hitgroup_record_count);
    PipelineCacheKey key{
        context,
        config.ptx,
        config.ptx_size,
        raygen_key(config.raygen_entries),
        entry_key(config.miss_entry),
        entry_key(config.closesthit_entry),
        entry_key(config.anyhit_entry),
        hitgroup_capacity,
        config.num_payload_values,
        config.params_size,
    };

    std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
    auto &cache = pipeline_cache();
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    auto pipeline = std::make_shared<OptixLaunchPipeline>();
    pipeline->build(context, device_index, hitgroup_capacity, config);
    cache[key] = pipeline;
    return pipeline;
}

void OptixLaunchPipeline::launch_impl(
    int raygen_index,
    const void *params,
    size_t actual_params_size,
    unsigned int n_rays,
    cudaStream_t stream) {
    if (!ready_)
        throw std::runtime_error("OptixLaunchPipeline::launch(): pipeline is not ready.");
    if (raygen_index < 0 || raygen_index >= static_cast<int>(raygen_records_.size()))
        throw std::runtime_error("OptixLaunchPipeline::launch(): raygen index out of range.");
    const size_t launch_params_size = (std::max)(params_size_, actual_params_size);
    if (params_buffer_.numel() < static_cast<int64_t>(launch_params_size))
        throw std::runtime_error("OptixLaunchPipeline::launch(): params buffer is too small.");

    cuda_check(
        cudaMemcpyAsync(
            params_buffer_.data_ptr<uint8_t>(),
            params,
            launch_params_size,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(multipath params)");

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(raygen_records_[raygen_index].data_ptr<uint8_t>());
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record_.data_ptr<uint8_t>());
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records_.data_ptr<uint8_t>());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    raydn_OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream,
        reinterpret_cast<CUdeviceptr>(params_buffer_.data_ptr<uint8_t>()),
        launch_params_size,
        &sbt,
        n_rays,
        1,
        1));
}

} // namespace raydn
