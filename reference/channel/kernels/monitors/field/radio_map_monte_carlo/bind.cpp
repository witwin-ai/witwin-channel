#include "drjit_common.h"
#include <monitors/field/radio_map_monte_carlo/bind.h>

#include <monitors/field/radio_map_monte_carlo/radio_map_monte_carlo.h>

void register_radio_map_monte_carlo_bindings(nb::module_ &m) {
    m.def(
        "radiomap_monte_carlo_scatter_axis_aligned_into",
        [](
            nb::handle coord_0,
            nb::handle coord_1,
            nb::handle los_power,
            nb::handle reflection_power,
            nb::handle diffraction_power,
            nb::handle out_los,
            nb::handle out_reflection,
            nb::handle out_diffraction,
            int n_samples,
            float coord_0_min,
            float coord_0_max,
            float coord_1_min,
            float coord_1_max,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            witwin::channel::native_ext::radiomap_monte_carlo_scatter_axis_aligned(
                ptr<float>(drjit_data_ptr_handle(coord_0)),
                ptr<float>(drjit_data_ptr_handle(coord_1)),
                ptr<float>(drjit_data_ptr_handle(los_power)),
                ptr<float>(drjit_data_ptr_handle(reflection_power)),
                ptr<float>(drjit_data_ptr_handle(diffraction_power)),
                ptr_mut<float>(drjit_data_ptr_handle(out_los)),
                ptr_mut<float>(drjit_data_ptr_handle(out_reflection)),
                ptr_mut<float>(drjit_data_ptr_handle(out_diffraction)),
                n_samples,
                coord_0_min,
                coord_0_max,
                coord_1_min,
                coord_1_max,
                cell_size_0,
                cell_size_1,
                n_coord_0,
                n_coord_1
            );
        },
        "Scatter weighted Monte Carlo radio-map samples into axis-aligned cell buffers."
    );
}
