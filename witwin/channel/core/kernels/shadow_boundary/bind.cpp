#include "drjit_common.h"
#include <shadow_boundary/bind.h>

#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <shadow_boundary/shadow_boundary.h>

void register_shadow_boundary_bindings(nb::module_ &m) {
    using witwin::channel::native_ext::common::throw_cuda;

    m.def(
        "shadow_boundary_candidate_accumulate",
        [](
            const Float &edge_pos_x,
            const Float &edge_pos_y,
            const Float &edge_pos_z,
            const Float &edge_dir_x,
            const Float &edge_dir_y,
            const Float &edge_dir_z,
            const Float &edge_n0_x,
            const Float &edge_n0_y,
            const Float &edge_n0_z,
            const Float &edge_nn_x,
            const Float &edge_nn_y,
            const Float &edge_nn_z,
            const Float &edge_wedge_n,
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Float &source_pos_x,
            const Float &source_pos_y,
            const Float &source_pos_z,
            const UInt32 &direct_los_visible,
            const Int32 &direct_blocker_group,
            const Int32 &edge_adjacent_group0,
            const Int32 &edge_adjacent_group1,
            const Float &cell_x,
            const Float &cell_y,
            const Float &cell_z,
            int n_edges,
            int grid_nx,
            int grid_ny,
            int tile_nx,
            int tile_ny,
            float k,
            float wavelength,
            float band_width_wavelengths,
            float max_candidate_factor
        ) {
            const size_t n_cells = static_cast<size_t>(grid_nx) * static_cast<size_t>(grid_ny);
            Float out_incident_weight = drjit::zeros<Float>(n_cells);
            Float out_reflection_weight = drjit::zeros<Float>(n_cells);
            Float out_incident_response_real = drjit::zeros<Float>(n_cells);
            Float out_incident_response_imag = drjit::zeros<Float>(n_cells);
            Float out_reflection_response_real = drjit::zeros<Float>(n_cells);
            Float out_reflection_response_imag = drjit::zeros<Float>(n_cells);
            UInt32 out_candidate_tile_count = drjit::zeros<UInt32>(1);
            UInt32 out_candidate_cell_count = drjit::zeros<UInt32>(1);
            drjit::eval(
                edge_pos_x,
                edge_pos_y,
                edge_pos_z,
                edge_dir_x,
                edge_dir_y,
                edge_dir_z,
                edge_n0_x,
                edge_n0_y,
                edge_n0_z,
                edge_nn_x,
                edge_nn_y,
                edge_nn_z,
                edge_wedge_n,
                edge_line_min,
                edge_line_max,
                source_pos_x,
                source_pos_y,
                source_pos_z,
                direct_los_visible,
                direct_blocker_group,
                edge_adjacent_group0,
                edge_adjacent_group1,
                cell_x,
                cell_y,
                cell_z,
                out_incident_weight,
                out_reflection_weight,
                out_incident_response_real,
                out_incident_response_imag,
                out_reflection_response_real,
                out_reflection_response_imag,
                out_candidate_tile_count,
                out_candidate_cell_count
            );
            throw_cuda(
                cudaMemset(drjit_data_ptr_mut(out_candidate_tile_count), 0, sizeof(unsigned int)),
                "shadow_boundary candidate count memset"
            );
            throw_cuda(
                cudaMemset(drjit_data_ptr_mut(out_candidate_cell_count), 0, sizeof(unsigned int)),
                "shadow_boundary candidate cell count memset"
            );
            witwin::channel::native_ext::shadow_boundary_candidate_accumulate(
                drjit_data_ptr(edge_pos_x),
                drjit_data_ptr(edge_pos_y),
                drjit_data_ptr(edge_pos_z),
                drjit_data_ptr(edge_dir_x),
                drjit_data_ptr(edge_dir_y),
                drjit_data_ptr(edge_dir_z),
                drjit_data_ptr(edge_n0_x),
                drjit_data_ptr(edge_n0_y),
                drjit_data_ptr(edge_n0_z),
                drjit_data_ptr(edge_nn_x),
                drjit_data_ptr(edge_nn_y),
                drjit_data_ptr(edge_nn_z),
                drjit_data_ptr(edge_wedge_n),
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_pos_x),
                drjit_data_ptr(source_pos_y),
                drjit_data_ptr(source_pos_z),
                drjit_data_ptr(direct_los_visible),
                drjit_data_ptr(direct_blocker_group),
                drjit_data_ptr(edge_adjacent_group0),
                drjit_data_ptr(edge_adjacent_group1),
                drjit_data_ptr(cell_x),
                drjit_data_ptr(cell_y),
                drjit_data_ptr(cell_z),
                n_edges,
                grid_nx,
                grid_ny,
                tile_nx,
                tile_ny,
                k,
                wavelength,
                band_width_wavelengths,
                max_candidate_factor,
                drjit_data_ptr_mut(out_incident_weight),
                drjit_data_ptr_mut(out_reflection_weight),
                drjit_data_ptr_mut(out_incident_response_real),
                drjit_data_ptr_mut(out_incident_response_imag),
                drjit_data_ptr_mut(out_reflection_response_real),
                drjit_data_ptr_mut(out_reflection_response_imag),
                drjit_data_ptr_mut(out_candidate_tile_count),
                drjit_data_ptr_mut(out_candidate_cell_count)
            );
            return nb::make_tuple(
                out_incident_weight,
                out_reflection_weight,
                out_incident_response_real,
                out_incident_response_imag,
                out_reflection_response_real,
                out_reflection_response_imag,
                out_candidate_tile_count,
                out_candidate_cell_count
            );
        },
        "Evaluate UTD shadow-boundary smoothing over conservative native edge-tile candidates."
    );
}
