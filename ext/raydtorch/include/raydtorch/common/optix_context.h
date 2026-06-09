#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include <ATen/ATen.h>

#include <cstdint>

namespace raydtorch {

enum class EdgeOptixLaunchKind {
    Point,
    Ray,
    PointTopK,
};

struct TorchCudaContext {
    int device_index = 0;
    cudaStream_t stream = nullptr;
};

struct OptixDeviceContextEntry {
    int device_index = 0;
    CUcontext cuda_context = nullptr;
    OptixDeviceContext optix_context = nullptr;
    OptixModule intersect_module = nullptr;
    OptixPipeline intersect_pipeline = nullptr;
    OptixProgramGroup intersect_raygen_group = nullptr;
    OptixProgramGroup intersect_miss_group = nullptr;
    OptixProgramGroup intersect_hitgroup = nullptr;
    OptixShaderBindingTable intersect_sbt = {};
    at::Tensor intersect_raygen_record;
    at::Tensor intersect_miss_record;
    at::Tensor intersect_hitgroup_record;
    OptixModule edge_module = nullptr;
    OptixModule edge_topk_module = nullptr;
    OptixPipeline edge_pipeline = nullptr;
    OptixPipeline edge_topk_pipeline = nullptr;
    OptixProgramGroup edge_raygen_point_group = nullptr;
    OptixProgramGroup edge_raygen_ray_group = nullptr;
    OptixProgramGroup edge_raygen_topk_group = nullptr;
    OptixProgramGroup edge_miss_group = nullptr;
    OptixProgramGroup edge_topk_miss_group = nullptr;
    OptixProgramGroup edge_hit_point_group = nullptr;
    OptixProgramGroup edge_hit_ray_group = nullptr;
    OptixProgramGroup edge_hit_topk_group = nullptr;
    OptixShaderBindingTable edge_point_sbt = {};
    OptixShaderBindingTable edge_ray_sbt = {};
    OptixShaderBindingTable edge_topk_sbt = {};
    at::Tensor edge_raygen_point_record;
    at::Tensor edge_raygen_ray_record;
    at::Tensor edge_raygen_topk_record;
    at::Tensor edge_miss_record;
    at::Tensor edge_topk_miss_record;
    at::Tensor edge_hitgroup_records;
    at::Tensor edge_topk_hitgroup_record;
    at::Tensor edge_params_buffer;
    OptixModule reflection_trace_module = nullptr;
    OptixPipeline reflection_trace_pipeline = nullptr;
    OptixProgramGroup reflection_trace_raygen_group = nullptr;
    OptixProgramGroup reflection_trace_miss_group = nullptr;
    OptixProgramGroup reflection_trace_hitgroup = nullptr;
    OptixShaderBindingTable reflection_trace_sbt = {};
    at::Tensor reflection_trace_raygen_record;
    at::Tensor reflection_trace_miss_record;
    at::Tensor reflection_trace_hitgroup_record;
};

TorchCudaContext current_torch_cuda_context();
OptixDeviceContextEntry &get_optix_context(int device_index);
void ensure_intersect_pipeline(OptixDeviceContextEntry &entry);
void ensure_edge_pipeline(OptixDeviceContextEntry &entry);
OptixPipeline edge_pipeline(const OptixDeviceContextEntry &entry, EdgeOptixLaunchKind kind);
const OptixShaderBindingTable &edge_sbt(const OptixDeviceContextEntry &entry, EdgeOptixLaunchKind kind);
void ensure_reflection_trace_pipeline(OptixDeviceContextEntry &entry);
void optix_check(OptixResult result, const char *expr, const char *file, int line);

} // namespace raydtorch

#define raydtorch_OPTIX_CHECK(expr) ::raydtorch::optix_check((expr), #expr, __FILE__, __LINE__)
