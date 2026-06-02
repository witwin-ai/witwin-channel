#include "drjit_common.h"
#include <monitors/radio_map/monte_carlo/native/bind.h>

#include <monitors/radio_map/monte_carlo/native/monte_carlo_sparse_coeff.h>

void register_monte_carlo_native_bindings(nb::module_ &m) {
    m.def(
        "monte_carlo_sparse_coeff_jvp_into",
        [](
            nb::handle cell_idx,
            nb::handle tx_coeff_x,
            nb::handle tx_coeff_y,
            nb::handle tx_coeff_z,
            nb::handle vertex_indices,
            nb::handle vertex_coeff_x,
            nb::handle vertex_coeff_y,
            nb::handle vertex_coeff_z,
            int vertex_slot_count,
            nb::handle material_indices,
            nb::handle material_coeff_eps,
            nb::handle material_coeff_sigma,
            int material_slot_count,
            nb::handle tx_tangent_x,
            nb::handle tx_tangent_y,
            nb::handle tx_tangent_z,
            nb::handle vertex_tangent_x,
            nb::handle vertex_tangent_y,
            nb::handle vertex_tangent_z,
            nb::handle material_tangent_eps,
            nb::handle material_tangent_sigma,
            int n_samples,
            int n_outputs
        ) {
            Float out_component = drjit::zeros<Float>(static_cast<size_t>(n_outputs));
            drjit::eval(out_component);
            witwin::channel::native_ext::monte_carlo_sparse_coeff_jvp_into(
                ptr<unsigned int>(drjit_data_ptr_handle(cell_idx)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_z)),
                ptr<int>(drjit_data_ptr_handle(vertex_indices)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_z)),
                vertex_slot_count,
                ptr<int>(drjit_data_ptr_handle(material_indices)),
                ptr<float>(drjit_data_ptr_handle(material_coeff_eps)),
                ptr<float>(drjit_data_ptr_handle(material_coeff_sigma)),
                material_slot_count,
                ptr<float>(drjit_data_ptr_handle(tx_tangent_x)),
                ptr<float>(drjit_data_ptr_handle(tx_tangent_y)),
                ptr<float>(drjit_data_ptr_handle(tx_tangent_z)),
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_x)),
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_y)),
                ptr<float>(drjit_data_ptr_handle(vertex_tangent_z)),
                ptr<float>(drjit_data_ptr_handle(material_tangent_eps)),
                ptr<float>(drjit_data_ptr_handle(material_tangent_sigma)),
                drjit_data_ptr_mut(out_component),
                n_samples
            );
            return out_component;
        },
        "Apply sparse Monte Carlo coefficient tape to input tangents and accumulate a component JVP."
    );

    m.def(
        "monte_carlo_sparse_coeff_vjp_into",
        [](
            nb::handle cell_idx,
            nb::handle tx_coeff_x,
            nb::handle tx_coeff_y,
            nb::handle tx_coeff_z,
            nb::handle vertex_indices,
            nb::handle vertex_coeff_x,
            nb::handle vertex_coeff_y,
            nb::handle vertex_coeff_z,
            int vertex_slot_count,
            nb::handle material_indices,
            nb::handle material_coeff_eps,
            nb::handle material_coeff_sigma,
            int material_slot_count,
            nb::handle upstream_component,
            int n_samples,
            int n_vertices,
            int n_materials
        ) {
            Float out_tx_grad_x = drjit::zeros<Float>(static_cast<size_t>(n_samples));
            Float out_tx_grad_y = drjit::zeros<Float>(static_cast<size_t>(n_samples));
            Float out_tx_grad_z = drjit::zeros<Float>(static_cast<size_t>(n_samples));
            Float out_vertex_grad_x = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            Float out_vertex_grad_y = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            Float out_vertex_grad_z = drjit::zeros<Float>(static_cast<size_t>(n_vertices));
            Float out_material_grad_eps = drjit::zeros<Float>(static_cast<size_t>(n_materials));
            Float out_material_grad_sigma = drjit::zeros<Float>(static_cast<size_t>(n_materials));
            drjit::eval(
                out_tx_grad_x,
                out_tx_grad_y,
                out_tx_grad_z,
                out_vertex_grad_x,
                out_vertex_grad_y,
                out_vertex_grad_z,
                out_material_grad_eps,
                out_material_grad_sigma
            );
            witwin::channel::native_ext::monte_carlo_sparse_coeff_vjp_into(
                ptr<unsigned int>(drjit_data_ptr_handle(cell_idx)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(tx_coeff_z)),
                ptr<int>(drjit_data_ptr_handle(vertex_indices)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_x)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_y)),
                ptr<float>(drjit_data_ptr_handle(vertex_coeff_z)),
                vertex_slot_count,
                ptr<int>(drjit_data_ptr_handle(material_indices)),
                ptr<float>(drjit_data_ptr_handle(material_coeff_eps)),
                ptr<float>(drjit_data_ptr_handle(material_coeff_sigma)),
                material_slot_count,
                ptr<float>(drjit_data_ptr_handle(upstream_component)),
                drjit_data_ptr_mut(out_tx_grad_x),
                drjit_data_ptr_mut(out_tx_grad_y),
                drjit_data_ptr_mut(out_tx_grad_z),
                drjit_data_ptr_mut(out_vertex_grad_x),
                drjit_data_ptr_mut(out_vertex_grad_y),
                drjit_data_ptr_mut(out_vertex_grad_z),
                drjit_data_ptr_mut(out_material_grad_eps),
                drjit_data_ptr_mut(out_material_grad_sigma),
                n_samples
            );
            return nb::make_tuple(
                out_tx_grad_x,
                out_tx_grad_y,
                out_tx_grad_z,
                out_vertex_grad_x,
                out_vertex_grad_y,
                out_vertex_grad_z,
                out_material_grad_eps,
                out_material_grad_sigma
            );
        },
        "Apply sparse Monte Carlo coefficient tape to output adjoints and accumulate a VJP."
    );
}
