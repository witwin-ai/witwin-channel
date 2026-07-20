#include <torch/extension.h>

#include "registry.h"

torch::Tensor cn_capacity_failure_state_create(torch::Tensor reference);

void register_runtime(pybind11::module_ &module) {
    module.def(
        "capacity_failure_state_create",
        &cn_capacity_failure_state_create,
        "Create a zeroed device-resident capacity failure state.",
        pybind11::arg("reference"));
}
