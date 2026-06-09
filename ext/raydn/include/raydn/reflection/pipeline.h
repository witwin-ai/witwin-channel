#pragma once

#include <raydn/common/optix_pipeline.h>

namespace raydn {

OptixPipelineConfig refl_trace_pipeline_config();
OptixPipelineConfig refl_visibility_pipeline_config();
OptixPipelineConfig refl_epc_pipeline_config();
OptixPipelineConfig refl_accum_pipeline_config();

inline OptixPipelineConfig reflection_trace_pipeline_config() {
    return refl_trace_pipeline_config();
}

inline OptixPipelineConfig segment_visibility_pipeline_config() {
    return refl_visibility_pipeline_config();
}

inline OptixPipelineConfig reflection_epc_pipeline_config() {
    return refl_epc_pipeline_config();
}

inline OptixPipelineConfig reflection_accumulation_pipeline_config() {
    return refl_accum_pipeline_config();
}

} // namespace raydn
