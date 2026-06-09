#include <raydn/diffraction/pipeline.h>

#include <raydn/diffraction/accum_params.h>
#include <raydn/diffraction/paths_params.h>
#include <raydn/diffraction_accumulation_optix_ptx.h>
#include <raydn/diffraction_paths_optix_ptx.h>

namespace raydn {

OptixPipelineConfig dfr_paths_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydn_diffraction_paths_optix_ptx;
    config.ptx_size = sizeof(raydn_diffraction_paths_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_paths_order1_primary",
        "__raygen__diffraction_paths_order1",
        "__raygen__diffraction_paths_order1_source_visibility_primary",
        "__raygen__diffraction_paths_order1_target_export_primary",
    };
    config.miss_entry = "__miss__diffraction_paths";
    config.closesthit_entry = "__closesthit__diffraction_paths";
    config.num_payload_values = 4;
    config.params_size = sizeof(DfrPathParams);
    return config;
}

OptixPipelineConfig dfr_accum_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydn_diffraction_accumulation_optix_ptx;
    config.ptx_size = sizeof(raydn_diffraction_accumulation_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_order1_accumulation",
        "__raygen__diffraction_order1_accumulation_primary",
        "__raygen__diffraction_order1_accumulation_no_suffix",
        "__raygen__diffraction_order1_accumulation_no_suffix_primary",
        "__raygen__diffraction_order1_accumulation_suffix",
        "__raygen__diffraction_order1_accumulation_suffix_primary",
        "__raygen__diffraction_order1_source_visibility_primary",
        "__raygen__diffraction_order1_no_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_suffix_first_visibility_primary",
        "__raygen__diffraction_order1_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_coherent_accumulation",
        "__raygen__diffraction_order1_coherent_accumulation_primary",
        "__raygen__diffraction_chain_accumulation",
        "__raygen__diffraction_chain_accumulation_primary",
    };
    config.miss_entry = "__miss__diffraction_accumulation";
    config.closesthit_entry = "__closesthit__diffraction_accumulation";
    config.num_payload_values = 4;
    config.params_size = sizeof(DfrAccumParams);
    return config;
}

} // namespace raydn
