#include <torch/extension.h>

#include <torch/cuda.h>

#include <sstream>
#include <string>

namespace {

pybind11::list cuda_architectures() {
    pybind11::list values;
    std::stringstream stream(CHANNEL_NATIVE_CUDA_ARCHITECTURES);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            values.append(value);
        }
    }
    return values;
}

}  // namespace

pybind11::dict cn_build_info() {
    pybind11::dict info;
    info["backend"] = "channel-native";
    info["uses_dr_jit"] = false;
#if CHANNEL_NATIVE_WITH_RAYD
    info["uses_rayd_native"] = true;
    info["rayd_integration"] = "source-linked";
#else
    info["uses_rayd_native"] = false;
    info["rayd_integration"] = "unavailable";
#endif
    info["uses_path_native"] = true;
    info["material_abi_version"] = 3;
    info["cuda_available"] = torch::cuda::is_available();
#if CHANNEL_NATIVE_WITH_OPTIX
    info["optix_available"] = true;
#else
    info["optix_available"] = false;
#endif
    info["channel_native_abi_version"] = CHANNEL_NATIVE_ABI_VERSION;
    info["channel_native_git_sha"] = CHANNEL_NATIVE_GIT_SHA;
    info["channel_native_git_dirty"] = static_cast<bool>(CHANNEL_NATIVE_GIT_DIRTY);
    info["rayd_repository_url"] = CHANNEL_NATIVE_RAYD_REPOSITORY_URL;
    info["rayd_commit"] = CHANNEL_NATIVE_RAYD_COMMIT;
    info["rayd_dirty"] = static_cast<bool>(CHANNEL_NATIVE_RAYD_DIRTY);
    info["rayd_integration_abi_kind"] = CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_KIND;
    info["rayd_integration_abi_path"] = CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_PATH;
    info["rayd_integration_abi_sha256"] = CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_SHA256;
    info["torch_version"] = CHANNEL_NATIVE_TORCH_VERSION;
    info["cuda_version"] = CHANNEL_NATIVE_CUDA_VERSION;
    info["cuda_compiler_version"] = CHANNEL_NATIVE_CUDA_COMPILER_VERSION;
    info["compiler"] = CHANNEL_NATIVE_COMPILER;
    info["cxx_abi"] = CHANNEL_NATIVE_CXX_ABI;
    info["cuda_architectures"] = cuda_architectures();
    info["build_type"] = CHANNEL_NATIVE_BUILD_TYPE;
    info["build_fingerprint"] = CHANNEL_NATIVE_BUILD_FINGERPRINT;
    return info;
}
