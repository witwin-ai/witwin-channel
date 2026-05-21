#include "drjit_common.h"
#include <diffraction_builder/bind.h>

#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <diffraction_builder/diffraction_builder.h>

void register_diffraction_builder_bindings(nb::module_ &m) {
    using witwin::channel::native_ext::common::throw_cuda;

    m.def(
        "monte_carlo_diffraction_sample_slots",
        [](
            const UInt32 &sample_index,
            const Float &cdf,
            int n_samples,
            int n_states,
            float total_length_scalar,
            int seed
        ) {
            UInt32 out_slots = drjit::zeros<UInt32>(static_cast<size_t>(n_samples));
            drjit::eval(sample_index, cdf, out_slots);
            witwin::channel::native_ext::monte_carlo_diffraction_sample_slots(
                drjit_data_ptr(sample_index),
                drjit_data_ptr(cdf),
                drjit_data_ptr_mut(out_slots),
                n_samples,
                n_states,
                total_length_scalar,
                seed
            );
            return out_slots;
        },
        "Sample diffraction state slots from a length-weighted CDF with a native CUDA binary search."
    );

    m.def(
        "monte_carlo_diffraction_best_edge_indices",
        [](
            float tx_x,
            float tx_y,
            float tx_z,
            const Float &ray_dir_x,
            const Float &ray_dir_y,
            const Float &ray_dir_z,
            const Float &hit_p_x,
            const Float &hit_p_y,
            const Float &hit_p_z,
            const Float &hit_n_x,
            const Float &hit_n_y,
            const Float &hit_n_z,
            const Float &hit_geo_n_x,
            const Float &hit_geo_n_y,
            const Float &hit_geo_n_z,
            const Int32 &hit_mask,
            const UInt32 &triangle_edge_count,
            const Int32 &triangle_edge_indices,
            int max_triangle_edge_slots,
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
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Int32 &edge_adjacent_face1,
            int n_edges,
            int n_rays
        ) {
            Int32 out_best_edge_idx = drjit::zeros<Int32>(static_cast<size_t>(n_rays));
            drjit::eval(
                ray_dir_x,
                ray_dir_y,
                ray_dir_z,
                hit_p_x,
                hit_p_y,
                hit_p_z,
                hit_n_x,
                hit_n_y,
                hit_n_z,
                hit_geo_n_x,
                hit_geo_n_y,
                hit_geo_n_z,
                hit_mask,
                triangle_edge_count,
                triangle_edge_indices,
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
                edge_line_min,
                edge_line_max,
                edge_adjacent_face1,
                out_best_edge_idx
            );
            witwin::channel::native_ext::monte_carlo_diffraction_best_edge_indices(
                tx_x,
                tx_y,
                tx_z,
                drjit_data_ptr(ray_dir_x),
                drjit_data_ptr(ray_dir_y),
                drjit_data_ptr(ray_dir_z),
                drjit_data_ptr(hit_p_x),
                drjit_data_ptr(hit_p_y),
                drjit_data_ptr(hit_p_z),
                drjit_data_ptr(hit_n_x),
                drjit_data_ptr(hit_n_y),
                drjit_data_ptr(hit_n_z),
                drjit_data_ptr(hit_geo_n_x),
                drjit_data_ptr(hit_geo_n_y),
                drjit_data_ptr(hit_geo_n_z),
                drjit_data_ptr(hit_mask),
                drjit_data_ptr(triangle_edge_count),
                drjit_data_ptr(triangle_edge_indices),
                max_triangle_edge_slots,
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
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(edge_adjacent_face1),
                n_edges,
                n_rays,
                drjit_data_ptr_mut(out_best_edge_idx)
            );
            return out_best_edge_idx;
        },
        "Select the nearest silhouette-valid diffraction edge per hit ray with a native CUDA helper."
    );

    m.def(
        "monte_carlo_diffraction_discover_edges",
        [](
            float tx_x,
            float tx_y,
            float tx_z,
            const Float &ray_dir_x,
            const Float &ray_dir_y,
            const Float &ray_dir_z,
            const Int32 &prim_index,
            const Float &hit_p_x,
            const Float &hit_p_y,
            const Float &hit_p_z,
            const Float &hit_n_x,
            const Float &hit_n_y,
            const Float &hit_n_z,
            const Float &hit_geo_n_x,
            const Float &hit_geo_n_y,
            const Float &hit_geo_n_z,
            int n_hits,
            const UInt32 &triangle_edge_count,
            const Int32 &triangle_edge_indices,
            int max_triangle_edge_slots,
            int n_triangles,
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
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Int32 &edge_adjacent_face1,
            int n_edges
        ) {
            UInt32 seen_edge_mask = drjit::zeros<UInt32>(static_cast<size_t>(n_edges));
            drjit::eval(
                ray_dir_x,
                ray_dir_y,
                ray_dir_z,
                prim_index,
                hit_p_x,
                hit_p_y,
                hit_p_z,
                hit_n_x,
                hit_n_y,
                hit_n_z,
                hit_geo_n_x,
                hit_geo_n_y,
                hit_geo_n_z,
                triangle_edge_count,
                triangle_edge_indices,
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
                edge_line_min,
                edge_line_max,
                edge_adjacent_face1,
                seen_edge_mask
            );
            throw_cuda(
                cudaMemset(
                    drjit_data_ptr_mut(seen_edge_mask),
                    0,
                    sizeof(uint32_t) * static_cast<size_t>(n_edges)
                ),
                "diffraction_discover_edges memset seen mask"
            );
            witwin::channel::native_ext::monte_carlo_diffraction_discover_edges(
                tx_x,
                tx_y,
                tx_z,
                drjit_data_ptr(ray_dir_x),
                drjit_data_ptr(ray_dir_y),
                drjit_data_ptr(ray_dir_z),
                drjit_data_ptr(prim_index),
                drjit_data_ptr(hit_p_x),
                drjit_data_ptr(hit_p_y),
                drjit_data_ptr(hit_p_z),
                drjit_data_ptr(hit_n_x),
                drjit_data_ptr(hit_n_y),
                drjit_data_ptr(hit_n_z),
                drjit_data_ptr(hit_geo_n_x),
                drjit_data_ptr(hit_geo_n_y),
                drjit_data_ptr(hit_geo_n_z),
                n_hits,
                drjit_data_ptr(triangle_edge_count),
                drjit_data_ptr(triangle_edge_indices),
                max_triangle_edge_slots,
                n_triangles,
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
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(edge_adjacent_face1),
                n_edges,
                drjit_data_ptr_mut(seen_edge_mask)
            );
            std::vector<uint32_t> host_seen(static_cast<size_t>(n_edges), 0u);
            throw_cuda(
                cudaMemcpy(
                    host_seen.data(),
                    drjit_data_ptr(seen_edge_mask),
                    sizeof(uint32_t) * static_cast<size_t>(n_edges),
                    cudaMemcpyDeviceToHost
                ),
                "diffraction_discover_edges copy seen mask"
            );
            std::vector<uint32_t> host_edges;
            host_edges.reserve(static_cast<size_t>(n_edges));
            for (int edge_idx = 0; edge_idx < n_edges; ++edge_idx) {
                if (host_seen[static_cast<size_t>(edge_idx)] != 0u) {
                    host_edges.push_back(static_cast<uint32_t>(edge_idx));
                }
            }
            UInt32 out_edges = drjit::zeros<UInt32>(host_edges.size());
            if (!host_edges.empty()) {
                drjit::eval(out_edges);
                throw_cuda(
                    cudaMemcpy(
                        drjit_data_ptr_mut(out_edges),
                        host_edges.data(),
                        sizeof(uint32_t) * host_edges.size(),
                        cudaMemcpyHostToDevice
                    ),
                    "diffraction_discover_edges copy compact edges"
                );
            }
            return out_edges;
        },
        "Discover unique diffraction edges directly from stored hit data with a native CUDA helper."
    );

    m.def(
        "monte_carlo_diffraction_build_state_arrays",
        [](
            const UInt32 &edge_idx,
            int n_states,
            float tx_x,
            float tx_y,
            float tx_z,
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
            const Int32 &edge_adjacent_face0,
            const Int32 &edge_adjacent_face1
        ) {
            Int32 out_edge_index = drjit::zeros<Int32>(static_cast<size_t>(n_states));
            Float out_edge_pos_x = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_edge_pos_y = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_edge_pos_z = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_edge_dir_x = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_edge_dir_y = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_edge_dir_z = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_n0_x = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_n0_y = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_n0_z = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_nn_x = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_nn_y = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_nn_z = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_wedge_n = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_line_min = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_line_max = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_source_pos_x = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_source_pos_y = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Float out_source_pos_z = drjit::zeros<Float>(static_cast<size_t>(n_states));
            Int32 out_adjacent_face0 = drjit::zeros<Int32>(static_cast<size_t>(n_states));
            Int32 out_adjacent_face1 = drjit::zeros<Int32>(static_cast<size_t>(n_states));
            drjit::eval(
                edge_idx,
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
                edge_adjacent_face0,
                edge_adjacent_face1,
                out_edge_index,
                out_edge_pos_x,
                out_edge_pos_y,
                out_edge_pos_z,
                out_edge_dir_x,
                out_edge_dir_y,
                out_edge_dir_z,
                out_n0_x,
                out_n0_y,
                out_n0_z,
                out_nn_x,
                out_nn_y,
                out_nn_z,
                out_wedge_n,
                out_line_min,
                out_line_max,
                out_source_pos_x,
                out_source_pos_y,
                out_source_pos_z,
                out_adjacent_face0,
                out_adjacent_face1
            );
            witwin::channel::native_ext::monte_carlo_diffraction_build_state_arrays(
                drjit_data_ptr(edge_idx),
                n_states,
                tx_x,
                tx_y,
                tx_z,
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
                drjit_data_ptr(edge_adjacent_face0),
                drjit_data_ptr(edge_adjacent_face1),
                drjit_data_ptr_mut(out_edge_index),
                drjit_data_ptr_mut(out_edge_pos_x),
                drjit_data_ptr_mut(out_edge_pos_y),
                drjit_data_ptr_mut(out_edge_pos_z),
                drjit_data_ptr_mut(out_edge_dir_x),
                drjit_data_ptr_mut(out_edge_dir_y),
                drjit_data_ptr_mut(out_edge_dir_z),
                drjit_data_ptr_mut(out_n0_x),
                drjit_data_ptr_mut(out_n0_y),
                drjit_data_ptr_mut(out_n0_z),
                drjit_data_ptr_mut(out_nn_x),
                drjit_data_ptr_mut(out_nn_y),
                drjit_data_ptr_mut(out_nn_z),
                drjit_data_ptr_mut(out_wedge_n),
                drjit_data_ptr_mut(out_line_min),
                drjit_data_ptr_mut(out_line_max),
                drjit_data_ptr_mut(out_source_pos_x),
                drjit_data_ptr_mut(out_source_pos_y),
                drjit_data_ptr_mut(out_source_pos_z),
                drjit_data_ptr_mut(out_adjacent_face0),
                drjit_data_ptr_mut(out_adjacent_face1)
            );
            nb::dict out;
            out["edge_index"] = out_edge_index;
            out["edge_pos_x"] = out_edge_pos_x;
            out["edge_pos_y"] = out_edge_pos_y;
            out["edge_pos_z"] = out_edge_pos_z;
            out["edge_dir_x"] = out_edge_dir_x;
            out["edge_dir_y"] = out_edge_dir_y;
            out["edge_dir_z"] = out_edge_dir_z;
            out["n0_x"] = out_n0_x;
            out["n0_y"] = out_n0_y;
            out["n0_z"] = out_n0_z;
            out["nn_x"] = out_nn_x;
            out["nn_y"] = out_nn_y;
            out["nn_z"] = out_nn_z;
            out["wedge_n"] = out_wedge_n;
            out["line_min"] = out_line_min;
            out["line_max"] = out_line_max;
            out["source_pos_x"] = out_source_pos_x;
            out["source_pos_y"] = out_source_pos_y;
            out["source_pos_z"] = out_source_pos_z;
            out["adjacent_face0"] = out_adjacent_face0;
            out["adjacent_face1"] = out_adjacent_face1;
            return out;
        },
        "Gather diffraction state arrays from unique edge indices with a native CUDA helper."
    );
}
