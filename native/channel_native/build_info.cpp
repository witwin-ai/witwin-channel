#include <torch/extension.h>

#include <torch/cuda.h>

pybind11::dict cn_build_info() {
    pybind11::dict info;
    info["backend"] = "channel-native";
    info["uses_dr_jit"] = false;
#if CHANNEL_NATIVE_WITH_RAYD
    info["uses_raydn_native"] = true;
    info["rayd_integration"] = "source-linked";
#else
    info["uses_raydn_native"] = false;
    info["rayd_integration"] = "unavailable";
#endif
    info["uses_path_native"] = true;
    info["cuda_available"] = torch::cuda::is_available();
#if CHANNEL_NATIVE_WITH_OPTIX
    info["optix_available"] = true;
#else
    info["optix_available"] = false;
#endif
    return info;
}
