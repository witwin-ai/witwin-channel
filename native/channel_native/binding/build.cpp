#include <torch/extension.h>

#include "registry.h"

#include <cstdint>
#include <string>
#include <vector>

pybind11::dict cn_build_info();

void register_build(pybind11::module_ &module) {
    module.def("build_info", &cn_build_info, "Return Channel Native build metadata.");
}
