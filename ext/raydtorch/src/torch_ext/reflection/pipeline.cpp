#include <raydtorch/reflection/pipeline.h>

#include <raydtorch/reflection/accum_params.h>
#include <raydtorch/reflection/epc_params.h>
#include <raydtorch/reflection/trace_params.h>
#include <raydtorch/reflection/visibility_params.h>
#include <raydtorch/reflection_accumulation_optix_ptx.h>
#include <raydtorch/reflection_epc_optix_ptx.h>
#include <raydtorch/reflection_trace_optix_ptx.h>
#include <raydtorch/segment_visibility_optix_ptx.h>

namespace raydtorch {

OptixPipelineConfig refl_trace_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_trace_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_trace_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_trace"};
    config.miss_entry = "__miss__reflection";
    config.closesthit_entry = "__closesthit__reflection";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflectionTraceParams);
    return config;
}

OptixPipelineConfig refl_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_segment_visibility_optix_ptx;
    config.ptx_size = sizeof(raydtorch_segment_visibility_optix_ptx);
    config.raygen_entries = {
        "__raygen__segment_visibility",
        "__raygen__segment_pair_visibility",
        "__raygen__axial_edge_visibility",
        "__raygen__segment_chain_visibility",
    };
    config.miss_entry = "__miss__segment_visibility";
    config.closesthit_entry = "__closesthit__segment_visibility";
    config.anyhit_entry = "__anyhit__segment_visibility";
    config.num_payload_values = 3;
    config.params_size = sizeof(SegmentVisibilityParams);
    return config;
}

OptixPipelineConfig refl_epc_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_epc_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_epc_optix_ptx);
    config.raygen_entries = {
        "__raygen__reflection_epc",
        "__raygen__reflection_epc_direct",
        "__raygen__reflection_epc_direct_primary",
    };
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig refl_accum_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_accumulation_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_accumulation_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_accumulation"};
    config.miss_entry = "__miss__reflection_accumulation";
    config.closesthit_entry = "__closesthit__reflection_accumulation";
    config.num_payload_values = 6;
    config.params_size = sizeof(AccumParams);
    return config;
}

} // namespace raydtorch
