#include "drjit_common.h"
#include <shadow_boundary/bind.h>

#include <cuda_runtime.h>

NB_MODULE(_channel_utils_native, m) {
    nb::module_::import_("drjit");
    nb::module_::import_("drjit.cuda");
    nb::module_::import_("drjit.cuda.ad");

    register_shadow_boundary_bindings(m);
}
