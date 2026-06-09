#pragma once

#include <raydtorch/common/optix_pipeline.h>

namespace raydtorch {

OptixPipelineConfig dfr_paths_pipeline_config();
OptixPipelineConfig dfr_accum_pipeline_config();

inline OptixPipelineConfig diffraction_paths_pipeline_config() {
    return dfr_paths_pipeline_config();
}

inline OptixPipelineConfig diffraction_accumulation_pipeline_config() {
    return dfr_accum_pipeline_config();
}

} // namespace raydtorch
