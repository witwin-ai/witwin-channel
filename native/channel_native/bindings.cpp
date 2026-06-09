#include <torch/extension.h>

pybind11::dict cn_build_info();
pybind11::dict cn_path_los_export(
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    double frequency_hz);

PYBIND11_MODULE(_channel_native, module) {
    module.doc() = "Channel Native Torch/CUDA extension.";
    module.def("build_info", &cn_build_info, "Return Channel Native build metadata.");
    module.def(
        "path_los_export",
        &cn_path_los_export,
        "Export empty-space LoS paths from CUDA tensors.");
}
