// Copyright Xingyu Chen.
// Implements runtime native integration.

#include <torch/extension.h>

#include "registry.h"

torch::Tensor channel_capacity_failure_state_create(torch::Tensor reference);
pybind11::tuple channel_capacity_failure_state_read(torch::Tensor record);

void register_runtime(pybind11::module_ &module) {
    module.def(
        "capacity_failure_state_create",
        &channel_capacity_failure_state_create,
        "Create a zeroed device-resident capacity failure record.",
        pybind11::arg("reference"));
    module.def(
        "capacity_failure_state_read",
        &channel_capacity_failure_state_read,
        "Copy a capacity failure record to the host once and return (bits, dropped_rows).",
        pybind11::arg("record"));
}
