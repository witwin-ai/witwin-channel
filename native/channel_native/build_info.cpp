#include <torch/extension.h>

#include <torch/cuda.h>

pybind11::dict cn_build_info() {
    pybind11::dict info;
    info["backend"] = "channel-native";
    info["uses_dr_jit"] = false;
    info["uses_raydn_native"] = false;
    info["uses_path_native"] = true;
    info["cuda_available"] = torch::cuda::is_available();
    info["optix_available"] = false;
    return info;
}
