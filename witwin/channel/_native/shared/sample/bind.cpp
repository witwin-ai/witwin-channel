#include "drjit_common.h"
#include <sample/bind.h>

#include <sample/sample_cuda.h>

// ---------------------------------------------------------------------------
// sample_add_one implementations
// ---------------------------------------------------------------------------

inline Float sample_add_one_impl(const Float &value) {
    return value + 1.f;
}

inline DiffFloat sample_add_one_impl(const DiffFloat &value) {
    return value + 1.f;
}

// ---------------------------------------------------------------------------
// register_sample_bindings — DrJit type init + sample / debug helpers
// ---------------------------------------------------------------------------

void register_sample_bindings(nb::module_ &m) {
    // Initialize DrJit array types so nanobind knows about them.
    {
        drjit::ArrayBinding binding;
        drjit::bind_all<Float>(binding);
        drjit::bind_all<DiffFloat>(binding);
        drjit::bind_all<Int32>(binding);
    }

    m.doc() = "Native Dr.Jit/CUDA sample bindings for Witwin.";

    m.def(
        "cuda_runtime_version",
        &witwin::channel::native_ext::cuda_runtime_version,
        "Return the CUDA runtime version detected by the bundled native helper."
    );
    m.def(
        "run_cuda_noop",
        &witwin::channel::native_ext::run_cuda_noop,
        "Launch and synchronize a trivial CUDA kernel to validate the native CUDA path."
    );
    m.def(
        "sample_add_one",
        nb::overload_cast<const Float &>(&sample_add_one_impl),
        nb::arg("value"),
        "Add one to a drjit.cuda.Float array and return the result."
    );
    m.def(
        "sample_add_one",
        nb::overload_cast<const DiffFloat &>(&sample_add_one_impl),
        nb::arg("value"),
        "Add one to a drjit.cuda.ad.Float array and return the result."
    );
}
