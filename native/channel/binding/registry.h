// Copyright Xingyu Chen.
// Declares registry native contracts.

#pragma once

#include <pybind11/pybind11.h>

void register_build(pybind11::module_ &module);
void register_runtime(pybind11::module_ &module);
void register_bdpt_subpaths(pybind11::module_ &module);
void register_materials(pybind11::module_ &module);
void register_bdpt_connections(pybind11::module_ &module);
void register_path_core(pybind11::module_ &module);
void register_bdpt_components(pybind11::module_ &module);
void register_rayd_geometry(pybind11::module_ &module);
void register_fields(pybind11::module_ &module);
void register_rayd_accumulation(pybind11::module_ &module);
void register_bdpt_diffraction_support(pybind11::module_ &module);
void register_path_diffraction_state(pybind11::module_ &module);
void register_bdpt_material_helpers(pybind11::module_ &module);
void register_path(pybind11::module_ &module);
void register_montecarlo(pybind11::module_ &module);
void register_montecarlo_transmission(pybind11::module_ &module);
void register_path_deterministic(pybind11::module_ &module);
