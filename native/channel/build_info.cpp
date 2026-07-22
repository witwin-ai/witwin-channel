#include <torch/extension.h>

#include <torch/cuda.h>

#include <sstream>
#include <string>

namespace {

pybind11::list cuda_architectures() {
    pybind11::list values;
    std::stringstream stream(CHANNEL_CUDA_ARCHITECTURES);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            values.append(value);
        }
    }
    return values;
}

}  // namespace

pybind11::dict channel_build_info() {
    pybind11::dict info;
    info["backend"] = "channel";
    info["uses_dr_jit"] = false;
#if CHANNEL_WITH_RAYD
    info["uses_rayd_native"] = true;
    info["rayd_integration"] = "source-linked";
#else
    info["uses_rayd_native"] = false;
    info["rayd_integration"] = "unavailable";
#endif
    info["uses_path_native"] = true;
    info["material_abi_version"] = 3;
    info["cuda_available"] = torch::cuda::is_available();
#if CHANNEL_WITH_OPTIX
    info["optix_available"] = true;
#else
    info["optix_available"] = false;
#endif
    info["channel_abi_version"] = CHANNEL_ABI_VERSION;
    info["channel_git_sha"] = CHANNEL_GIT_SHA;
    info["channel_git_dirty"] = static_cast<bool>(CHANNEL_GIT_DIRTY);
    info["rayd_repository_url"] = CHANNEL_RAYD_REPOSITORY_URL;
    info["rayd_commit"] = CHANNEL_RAYD_COMMIT;
    info["rayd_dirty"] = static_cast<bool>(CHANNEL_RAYD_DIRTY);
    info["rayd_integration_abi_kind"] = CHANNEL_RAYD_INTEGRATION_ABI_KIND;
    info["rayd_integration_abi_path"] = CHANNEL_RAYD_INTEGRATION_ABI_PATH;
    info["rayd_integration_abi_sha256"] = CHANNEL_RAYD_INTEGRATION_ABI_SHA256;
    info["rayd_source_kind"] = CHANNEL_RAYD_SOURCE_KIND;
    info["rayd_source_manifest_sha256"] = CHANNEL_RAYD_SOURCE_MANIFEST_SHA256;
    info["torch_version"] = CHANNEL_TORCH_VERSION;
    info["cuda_version"] = CHANNEL_CUDA_VERSION;
    info["cuda_compiler_version"] = CHANNEL_CUDA_COMPILER_VERSION;
    info["compiler"] = CHANNEL_COMPILER;
    info["cxx_abi"] = CHANNEL_CXX_ABI;
    info["cuda_architectures"] = cuda_architectures();
    info["build_type"] = CHANNEL_BUILD_TYPE;
    info["build_fingerprint"] = CHANNEL_BUILD_FINGERPRINT;
    return info;
}
