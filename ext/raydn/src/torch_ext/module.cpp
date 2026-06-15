#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace raydn {

py::dict build_info() {
    py::dict info;
    info["backend"] = "rayd-native";
    info["uses_dr_jit"] = false;
    return info;
}

uintptr_t native_module_handle() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&native_module_handle),
            &module)) {
        throw std::runtime_error(
            "failed to resolve loaded _raydn module handle: Windows error " + std::to_string(GetLastError()));
    }
    return reinterpret_cast<uintptr_t>(module);
#else
    Dl_info info = {};
    if (dladdr(reinterpret_cast<void *>(&native_module_handle), &info) == 0 || info.dli_fbase == nullptr)
        throw std::runtime_error("failed to resolve loaded _raydn module handle");
    return reinterpret_cast<uintptr_t>(info.dli_fbase);
#endif
}

PYBIND11_MODULE(_raydn, m) {
    m.doc() = "RayDN CUDA/OptiX backend.";
    m.def("build_info", &build_info);
    m.def("native_module_handle", &native_module_handle);
}

} // namespace raydn
