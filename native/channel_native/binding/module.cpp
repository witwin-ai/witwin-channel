#include "registry.h"

PYBIND11_MODULE(_channel_native, module) {
    module.doc() = "Channel Native Torch/CUDA extension.";
    register_all(module);
}
