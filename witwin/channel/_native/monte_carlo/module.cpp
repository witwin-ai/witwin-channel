#include "drjit_common.h"
#include <sparse_coeff/bind.h>
#include <transport_grid/bind.h>
#include <transport_vertex/bind.h>
#include <diffraction_builder/bind.h>

#include <cuda_runtime.h>

NB_MODULE(_monte_carlo_radiomap_native, m) {
    nb::module_::import_("drjit");
    nb::module_::import_("drjit.cuda");
    nb::module_::import_("drjit.cuda.ad");

    m.def("drjit_data_ptr_inplace", [](nb::handle value) {
        if (!value.is_valid()) {
            throw nb::type_error("drjit_data_ptr_inplace expects a Dr.Jit array instance");
        }
        try {
            return drjit_data_ptr_handle(value);
        } catch (const std::exception &exc) {
            throw nb::type_error(exc.what());
        }
    });
    m.def("drjit_data_ptr", [](const Float &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_data_ptr", [](const DiffFloat &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_data_ptr", [](const Int32 &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_data_ptr", [](const DiffInt32 &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_data_ptr", [](const UInt32 &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_data_ptr", [](const DiffUInt32 &value) {
        return reinterpret_cast<std::uintptr_t>(::drjit_data_ptr(value));
    });
    m.def("drjit_index_data_ptr", [](uint32_t index) {
        void *ptr = nullptr;
        jit_var_data(index, &ptr);
        return reinterpret_cast<std::uintptr_t>(ptr);
    });

    register_monte_carlo_native_bindings(m);
    register_transport_grid_bindings(m);
    register_transport_vertex_bindings(m);
    register_diffraction_builder_bindings(m);
}
