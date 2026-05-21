#include "drjit_common.h"
#include <transport_grid/bind.h>

#include <transport_grid/transport_grid.h>

void register_transport_grid_bindings(nb::module_ &m) {
    m.def(
        "monte_carlo_transport_grid_forward_raw",
        [](
            const Float &coord_0,
            const Float &coord_1,
            const Float &power,
            const Int32 &active_mask,
            int n_samples,
            float coord_0_min,
            float coord_1_min,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            size_t n_cells = static_cast<size_t>(n_coord_0) * static_cast<size_t>(n_coord_1);
            Float out_grid = drjit::zeros<Float>(n_cells);
            drjit::eval(coord_0, coord_1, power, active_mask, out_grid);
            witwin::channel::native_ext::monte_carlo_transport_grid_forward(
                drjit_data_ptr(coord_0),
                drjit_data_ptr(coord_1),
                drjit_data_ptr(power),
                drjit_data_ptr(active_mask),
                drjit_data_ptr_mut(out_grid),
                n_samples,
                coord_0_min,
                coord_1_min,
                cell_size_0,
                cell_size_1,
                n_coord_0,
                n_coord_1
            );
            return out_grid;
        },
        "Launch the native transport-grid forward accumulation kernel."
    );

    m.def(
        "monte_carlo_transport_grid_jvp_raw",
        [](
            const Float &coord_0,
            const Float &coord_1,
            const Float &power,
            const Int32 &active_mask,
            const Float &t_coord_0,
            const Float &t_coord_1,
            const Float &t_power,
            int n_samples,
            float coord_0_min,
            float coord_1_min,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            size_t n_cells = static_cast<size_t>(n_coord_0) * static_cast<size_t>(n_coord_1);
            Float out_grid = drjit::zeros<Float>(n_cells);
            drjit::eval(
                coord_0,
                coord_1,
                power,
                active_mask,
                t_coord_0,
                t_coord_1,
                t_power,
                out_grid
            );
            witwin::channel::native_ext::monte_carlo_transport_grid_jvp(
                drjit_data_ptr(coord_0),
                drjit_data_ptr(coord_1),
                drjit_data_ptr(power),
                drjit_data_ptr(active_mask),
                drjit_data_ptr(t_coord_0),
                drjit_data_ptr(t_coord_1),
                drjit_data_ptr(t_power),
                drjit_data_ptr_mut(out_grid),
                n_samples,
                coord_0_min,
                coord_1_min,
                cell_size_0,
                cell_size_1,
                n_coord_0,
                n_coord_1
            );
            return out_grid;
        },
        "Launch the native transport-grid JVP kernel."
    );

    m.def(
        "monte_carlo_transport_grid_backward_raw",
        [](
            const Float &coord_0,
            const Float &coord_1,
            const Float &power,
            const Int32 &active_mask,
            const Float &upstream_grid,
            int n_samples,
            float coord_0_min,
            float coord_1_min,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            size_t n_ray = static_cast<size_t>(n_samples);
            Float grad_coord_0 = drjit::zeros<Float>(n_ray);
            Float grad_coord_1 = drjit::zeros<Float>(n_ray);
            Float grad_power = drjit::zeros<Float>(n_ray);
            drjit::eval(
                coord_0,
                coord_1,
                power,
                active_mask,
                upstream_grid,
                grad_coord_0,
                grad_coord_1,
                grad_power
            );
            witwin::channel::native_ext::monte_carlo_transport_grid_backward(
                drjit_data_ptr(coord_0),
                drjit_data_ptr(coord_1),
                drjit_data_ptr(power),
                drjit_data_ptr(active_mask),
                drjit_data_ptr(upstream_grid),
                drjit_data_ptr_mut(grad_coord_0),
                drjit_data_ptr_mut(grad_coord_1),
                drjit_data_ptr_mut(grad_power),
                n_samples,
                coord_0_min,
                coord_1_min,
                cell_size_0,
                cell_size_1,
                n_coord_0,
                n_coord_1
            );
            return nb::make_tuple(grad_coord_0, grad_coord_1, grad_power);
        },
        "Launch the native transport-grid backward kernel."
    );
}
