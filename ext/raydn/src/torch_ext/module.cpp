#include <torch/extension.h>

namespace raydn {

py::dict build_info() {
    py::dict info;
    info["backend"] = "rayd-native";
    info["uses_dr_jit"] = false;
    return info;
}

PYBIND11_MODULE(_raydn, m) {
    m.doc() = "RayDN CUDA/OptiX backend.";
    m.def("build_info", &build_info);
}

} // namespace raydn
