#include <torch/extension.h>

namespace raydtorch {

void bind_scene_ops(py::module_ &m);
void bind_intersect_ops(py::module_ &m);
void bind_edge_ops(py::module_ &m);
void bind_reflection_ops(py::module_ &m);
void bind_diffraction_ops(py::module_ &m);

py::dict build_info() {
    py::dict info;
    info["backend"] = "raydtorch-native";
    info["uses_dr_jit"] = false;
    return info;
}

PYBIND11_MODULE(_raydtorch, m) {
    m.doc() = "RayDTorch CUDA/OptiX backend.";
    m.def("build_info", &build_info);
    bind_scene_ops(m);
    bind_intersect_ops(m);
    bind_edge_ops(m);
    bind_reflection_ops(m);
    bind_diffraction_ops(m);
}

} // namespace raydtorch
