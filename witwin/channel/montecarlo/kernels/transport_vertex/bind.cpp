#include "drjit_common.h"
#include <transport_vertex/bind.h>

#include <transport_vertex/transport_vertex.h>

void register_transport_vertex_bindings(nb::module_ &m) {
    m.def(
        "monte_carlo_transport_vertex_jvp_into",
        [](
            nb::handle coord_0,
            nb::handle coord_1,
            nb::handle power,
            nb::handle active_mask,
            nb::handle vertex_indices,
            nb::handle coord_0_coeff_x,
            nb::handle coord_0_coeff_y,
            nb::handle coord_0_coeff_z,
            nb::handle coord_1_coeff_x,
            nb::handle coord_1_coeff_y,
            nb::handle coord_1_coeff_z,
            int vertex_slot_count,
            nb::handle vertex_tangent_x,
            nb::handle vertex_tangent_y,
            nb::handle vertex_tangent_z,
            int n_samples,
            int n_outputs,
            float coord_0_min,
            float coord_1_min,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            Float out_grid = drjit::zeros<Float>(static_cast<size_t>(n_outputs));
            drjit::eval(out_grid);
            witwin::channel::native_ext::monte_carlo_transport_vertex_jvp_into(
                ptr<float>(drjit_data_ptr_handle(coord_0)),
                ptr<float>(drjit_data_ptr_handle(coord_1)),
                ptr<float>(drjit_data_ptr_handle(power)),
                ptr<int>(drjit_data_ptr_handle(active_mask)),
                ptr<int>(drjit_data_ptr_handle(vertex_indices)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_z)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_z)),
                vertex_slot_count,
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_x)),
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_y)),
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_z)),
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
        "Apply transport vertex coefficients to tangents and accumulate a grid JVP."
    );

    m.def(
        "monte_carlo_transport_vertex_vjp_into",
        [](
            nb::handle coord_0,
            nb::handle coord_1,
            nb::handle power,
            nb::handle active_mask,
            nb::handle vertex_indices,
            nb::handle coord_0_coeff_x,
            nb::handle coord_0_coeff_y,
            nb::handle coord_0_coeff_z,
            nb::handle coord_1_coeff_x,
            nb::handle coord_1_coeff_y,
            nb::handle coord_1_coeff_z,
            int vertex_slot_count,
            nb::handle upstream_grid,
            int n_samples,
            int n_vertices,
            float coord_0_min,
            float coord_1_min,
            float cell_size_0,
            float cell_size_1,
            int n_coord_0,
            int n_coord_1
        ) {
            Float out_vertex_grad_x = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            Float out_vertex_grad_y = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            Float out_vertex_grad_z = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            drjit::eval(out_vertex_grad_x, out_vertex_grad_y, out_vertex_grad_z);
            witwin::channel::native_ext::monte_carlo_transport_vertex_vjp_into(
                ptr<float>(drjit_data_ptr_handle(coord_0)),
                ptr<float>(drjit_data_ptr_handle(coord_1)),
                ptr<float>(drjit_data_ptr_handle(power)),
                ptr<int>(drjit_data_ptr_handle(active_mask)),
                ptr<int>(drjit_data_ptr_handle(vertex_indices)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(coord_0_coeff_z)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(coord_1_coeff_z)),
                vertex_slot_count,
                ptr<float>(drjit_data_ptr_handle(upstream_grid)),
                drjit_data_ptr_mut(out_vertex_grad_x),
                drjit_data_ptr_mut(out_vertex_grad_y),
                drjit_data_ptr_mut(out_vertex_grad_z),
                n_samples,
                coord_0_min,
                coord_1_min,
                cell_size_0,
                cell_size_1,
                n_coord_0,
                n_coord_1
            );
            return nb::make_tuple(out_vertex_grad_x, out_vertex_grad_y, out_vertex_grad_z);
        },
        "Apply transport vertex coefficients to output adjoints and accumulate vertex VJPs."
    );
}
