#include "registry.h"

PYBIND11_MODULE(_channel_native, module) {
    module.doc() = "Channel Native Torch/CUDA extension.";
    register_build(module);
    register_bdpt_subpaths(module);
    register_materials(module);
    register_bdpt_connections(module);
    register_path_core(module);
    register_bdpt_components(module);
    register_rayd_geometry(module);
    register_fields(module);
    register_rayd_accumulation(module);
    register_bdpt_diffraction_support(module);
    register_path_diffraction_state(module);
    register_bdpt_material_helpers(module);
    register_path(module);
    register_montecarlo(module);
    register_path_deterministic(module);
}
