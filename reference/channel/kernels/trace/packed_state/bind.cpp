#include "drjit_common.h"
#include <trace/packed_state/bind.h>

#include <trace/packed_state/packed_state.h>

void register_packed_state_bindings(nb::module_ &m) {
    m.def(
        "gather_packed_states_raw",
        [](
            const DiffFloat &src,
            const Int32 &indices,
            int n_out,
            int stride
        ) {
            DiffFloat dst = drjit::zeros<DiffFloat>(static_cast<size_t>(n_out) * static_cast<size_t>(stride));
            drjit::eval(src, indices, dst);
            witwin::channel::native_ext::gather_packed_states(
                drjit_data_ptr(src),
                drjit_data_ptr(indices),
                drjit_data_ptr_mut(dst),
                n_out,
                stride
            );
            return dst;
        },
        nb::arg("src"), nb::arg("indices"),
        nb::arg("n_out"), nb::arg("stride"),
        "Gather packed states by index. Single bulk memcpy replaces 52x dr.gather()."
    );

    m.def(
        "concat_packed_states_raw",
        [](
            nb::list src_arrays,
            nb::list sizes,
            int stride
        ) {
            size_t n_sources = src_arrays.size();
            if (sizes.size() != n_sources) {
                throw std::runtime_error("concat_packed_states_raw: src_arrays/sizes length mismatch");
            }

            std::vector<std::uintptr_t> src_ptrs = array_pointer_list(
                src_arrays,
                "concat_packed_states_raw(src_arrays)"
            );
            std::vector<const float*> host_srcs;
            std::vector<int> host_sizes;
            host_srcs.reserve(n_sources);
            host_sizes.reserve(n_sources);
            int total_states = 0;
            for (size_t i = 0; i < n_sources; ++i) {
                host_srcs.push_back(ptr<const float>(src_ptrs[i]));
                int size = nb::cast<int>(sizes[i]);
                host_sizes.push_back(size);
                total_states += size;
            }

            DiffFloat dst = drjit::zeros<DiffFloat>(static_cast<size_t>(total_states) * static_cast<size_t>(stride));
            drjit::eval(dst);
            witwin::channel::native_ext::concat_packed_states(
                host_srcs.data(),
                host_sizes.data(),
                drjit_data_ptr_mut(dst),
                static_cast<int>(n_sources),
                stride
            );
            return dst;
        },
        nb::arg("src_arrays"), nb::arg("sizes"), nb::arg("stride"),
        "Concatenate packed state buffers with one CUDA kernel launch."
    );

    m.def(
        "pack_state_arrays_raw",
        [](
            nb::list core_arrays,
            int n_states,
            int stride
        ) {
            auto host_core = array_pointer_list(core_arrays, "pack_state_arrays_raw(core_arrays)");
            if (host_core.size() != witwin::channel::native_ext::PACKED_CORE_POINTER_COUNT) {
                throw std::runtime_error("pack_state_arrays_raw: core_arrays length mismatch");
            }

            DiffFloat dst = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states) * static_cast<size_t>(stride));
            drjit::eval(dst);
            witwin::channel::native_ext::pack_state_arrays(
                host_core.data(),
                drjit_data_ptr_mut(dst),
                n_states,
                stride
            );
            return dst;
        },
        nb::arg("core_arrays"), nb::arg("n_states"), nb::arg("stride"),
        "Pack SoA diffraction state arrays into one contiguous packed-state buffer."
    );

    m.def(
        "unpack_state_arrays_raw",
        [](
            const DiffFloat &src,
            int n_states,
            int stride
        ) {
            size_t count = static_cast<size_t>(n_states);

            DiffUInt32 edge_idx = drjit::zeros<DiffUInt32>(count);
            DiffFloat edge_pos_x = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_pos_y = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_pos_z = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_dir_x = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_dir_y = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_dir_z = drjit::zeros<DiffFloat>(count);
            DiffFloat n0_x = drjit::zeros<DiffFloat>(count);
            DiffFloat n0_y = drjit::zeros<DiffFloat>(count);
            DiffFloat n0_z = drjit::zeros<DiffFloat>(count);
            DiffFloat nn_x = drjit::zeros<DiffFloat>(count);
            DiffFloat nn_y = drjit::zeros<DiffFloat>(count);
            DiffFloat nn_z = drjit::zeros<DiffFloat>(count);
            DiffFloat wedge_n = drjit::zeros<DiffFloat>(count);
            DiffInt32 adjacent_face0 = drjit::zeros<DiffInt32>(count);
            DiffInt32 adjacent_face1 = drjit::zeros<DiffInt32>(count);
            DiffFloat source_pos_x = drjit::zeros<DiffFloat>(count);
            DiffFloat source_pos_y = drjit::zeros<DiffFloat>(count);
            DiffFloat source_pos_z = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_field_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_field_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_nderiv_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_nderiv_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_jones_u_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_jones_u_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_jones_v_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_jones_v_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_djones_u_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_djones_u_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_djones_v_re = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_djones_v_im = drjit::zeros<DiffFloat>(count);
            DiffFloat r0_re = drjit::zeros<DiffFloat>(count);
            DiffFloat r0_im = drjit::zeros<DiffFloat>(count);
            DiffFloat rn_re = drjit::zeros<DiffFloat>(count);
            DiffFloat rn_im = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_u_x = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_u_y = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_u_z = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_v_x = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_v_y = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_v_z = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_k_x = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_k_y = drjit::zeros<DiffFloat>(count);
            DiffFloat inc_basis_k_z = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m00_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m00_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m01_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m01_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m10_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m10_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m11_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_op_m11_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m00_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m00_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m01_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m01_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m10_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m10_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m11_re = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_op_m11_im = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_eta_r = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_sigma = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_gain = drjit::zeros<DiffFloat>(count);
            DiffFloat f0_use_fresnel = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_eta_r = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_sigma = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_gain = drjit::zeros<DiffFloat>(count);
            DiffFloat f1_use_fresnel = drjit::zeros<DiffFloat>(count);
            DiffUInt32 prefix_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 inter_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 suffix_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 order = drjit::zeros<DiffUInt32>(count);

            nb::list core_arrays;
            std::vector<std::uintptr_t> core_ptrs;
            core_ptrs.reserve(witwin::channel::native_ext::PACKED_CORE_POINTER_COUNT);
            auto push_core = [&](auto &array) {
                core_ptrs.push_back(reinterpret_cast<std::uintptr_t>(drjit_data_ptr_mut(array)));
                core_arrays.append(array);
            };

            push_core(edge_idx);
            push_core(edge_pos_x); push_core(edge_pos_y); push_core(edge_pos_z);
            push_core(edge_dir_x); push_core(edge_dir_y); push_core(edge_dir_z);
            push_core(n0_x); push_core(n0_y); push_core(n0_z);
            push_core(nn_x); push_core(nn_y); push_core(nn_z);
            push_core(wedge_n);
            push_core(adjacent_face0); push_core(adjacent_face1);
            push_core(source_pos_x); push_core(source_pos_y); push_core(source_pos_z);
            push_core(inc_field_re); push_core(inc_field_im);
            push_core(inc_nderiv_re); push_core(inc_nderiv_im);
            push_core(inc_jones_u_re); push_core(inc_jones_u_im);
            push_core(inc_jones_v_re); push_core(inc_jones_v_im);
            push_core(inc_djones_u_re); push_core(inc_djones_u_im);
            push_core(inc_djones_v_re); push_core(inc_djones_v_im);
            push_core(r0_re); push_core(r0_im);
            push_core(rn_re); push_core(rn_im);
            push_core(inc_basis_u_x); push_core(inc_basis_u_y); push_core(inc_basis_u_z);
            push_core(inc_basis_v_x); push_core(inc_basis_v_y); push_core(inc_basis_v_z);
            push_core(inc_basis_k_x); push_core(inc_basis_k_y); push_core(inc_basis_k_z);
            push_core(f0_op_m00_re); push_core(f0_op_m00_im);
            push_core(f0_op_m01_re); push_core(f0_op_m01_im);
            push_core(f0_op_m10_re); push_core(f0_op_m10_im);
            push_core(f0_op_m11_re); push_core(f0_op_m11_im);
            push_core(f1_op_m00_re); push_core(f1_op_m00_im);
            push_core(f1_op_m01_re); push_core(f1_op_m01_im);
            push_core(f1_op_m10_re); push_core(f1_op_m10_im);
            push_core(f1_op_m11_re); push_core(f1_op_m11_im);
            push_core(f0_eta_r); push_core(f0_sigma); push_core(f0_gain);
            push_core(f0_use_fresnel);
            push_core(f1_eta_r); push_core(f1_sigma); push_core(f1_gain);
            push_core(f1_use_fresnel);
            push_core(prefix_refl_depth);
            push_core(inter_refl_depth);
            push_core(suffix_refl_depth);
            push_core(order);

            witwin::channel::native_ext::unpack_state_arrays(
                drjit_data_ptr(src),
                core_ptrs.data(),
                n_states,
                stride
            );

            return core_arrays;
        },
        nb::arg("src"), nb::arg("n_states"), nb::arg("stride"),
        "Unpack a packed-state buffer back into SoA diffraction state arrays."
    );

    m.def(
        "gather_inserted_reflection_state_fields_raw",
        [](
            const DiffFloat &src,
            const Int32 &indices,
            int n_out,
            int stride
        ) {
            size_t count = static_cast<size_t>(n_out);

            DiffFloat edge_pos_x = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_pos_y = drjit::zeros<DiffFloat>(count);
            DiffFloat edge_pos_z = drjit::zeros<DiffFloat>(count);
            DiffUInt32 prefix_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 inter_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 suffix_refl_depth = drjit::zeros<DiffUInt32>(count);
            DiffUInt32 order = drjit::zeros<DiffUInt32>(count);

            witwin::channel::native_ext::gather_inserted_reflection_state_fields(
                drjit_data_ptr(src),
                drjit_data_ptr(indices),
                drjit_data_ptr_mut(edge_pos_x),
                drjit_data_ptr_mut(edge_pos_y),
                drjit_data_ptr_mut(edge_pos_z),
                drjit_data_ptr_mut(prefix_refl_depth),
                drjit_data_ptr_mut(inter_refl_depth),
                drjit_data_ptr_mut(suffix_refl_depth),
                drjit_data_ptr_mut(order),
                n_out,
                stride
            );

            return nb::make_tuple(
                edge_pos_x,
                edge_pos_y,
                edge_pos_z,
                prefix_refl_depth,
                inter_refl_depth,
                suffix_refl_depth,
                order
            );
        },
        nb::arg("src"), nb::arg("indices"), nb::arg("n_out"), nb::arg("stride"),
        "Gather only the inserted-reflection builder fields from a packed-state buffer."
    );

    m.def(
        "build_diffraction_path_slots_raw",
        [](
            const DiffInt32 &prefix_depth,
            const DiffInt32 &order,
            nb::list path_edge_slots,
            nb::list inserted_depth_slots,
            int history_size,
            int n_states,
            int max_depth,
            bool return_geometry,
            nb::handle first_interaction_pos_x,
            nb::handle first_interaction_pos_y,
            nb::handle first_interaction_pos_z,
            nb::handle edge_pos_x,
            nb::handle edge_pos_y,
            nb::handle edge_pos_z,
            nb::handle edge_n0_x,
            nb::handle edge_n0_y,
            nb::handle edge_n0_z,
            nb::handle edge_object_idx,
            int n_edges,
            int reflection_code,
            int diffraction_code
        ) {
            if (history_size < 0 || history_size > witwin::channel::native_ext::PACKED_MAX_HISTORY_SLOTS) {
                throw std::runtime_error("build_diffraction_path_slots_raw: invalid history_size");
            }
            if (max_depth < 1 || max_depth > witwin::channel::native_ext::DIFFRACTION_PATH_SLOT_MAX_DEPTH) {
                throw std::runtime_error("build_diffraction_path_slots_raw: invalid max_depth");
            }

            auto path_edge_ptrs = array_pointer_list(
                path_edge_slots,
                "build_diffraction_path_slots_raw(path_edge_slots)"
            );
            auto inserted_ptrs = array_pointer_list(
                inserted_depth_slots,
                "build_diffraction_path_slots_raw(inserted_depth_slots)"
            );
            size_t expected_inserted = history_size > 0 ? static_cast<size_t>(history_size - 1) : 0;
            if (path_edge_ptrs.size() != static_cast<size_t>(history_size)
                || inserted_ptrs.size() != expected_inserted) {
                throw std::runtime_error(
                    "build_diffraction_path_slots_raw: slot pointer length mismatch"
                );
            }

            auto maybe_float_ptr = [](nb::handle value) -> const float* {
                if (!value.is_valid() || value.is_none()) {
                    return nullptr;
                }
                return ptr<const float>(drjit_data_ptr_handle(value));
            };
            auto maybe_int_ptr = [](nb::handle value) -> const int* {
                if (!value.is_valid() || value.is_none()) {
                    return nullptr;
                }
                return ptr<const int>(drjit_data_ptr_handle(value));
            };

            witwin::channel::native_ext::DiffractionPathSlotInputs in{};
            in.prefix_depth = drjit_data_ptr(prefix_depth);
            in.order = drjit_data_ptr(order);
            in.first_interaction_pos_x = maybe_float_ptr(first_interaction_pos_x);
            in.first_interaction_pos_y = maybe_float_ptr(first_interaction_pos_y);
            in.first_interaction_pos_z = maybe_float_ptr(first_interaction_pos_z);
            in.edge_pos_x = maybe_float_ptr(edge_pos_x);
            in.edge_pos_y = maybe_float_ptr(edge_pos_y);
            in.edge_pos_z = maybe_float_ptr(edge_pos_z);
            in.edge_n0_x = maybe_float_ptr(edge_n0_x);
            in.edge_n0_y = maybe_float_ptr(edge_n0_y);
            in.edge_n0_z = maybe_float_ptr(edge_n0_z);
            in.edge_object_idx = maybe_int_ptr(edge_object_idx);
            in.history_size = history_size;
            in.n_edges = n_edges;
            for (int slot = 0; slot < history_size; ++slot) {
                in.path_edge_slots[slot] = ptr<const int>(path_edge_ptrs[slot]);
            }
            for (int slot = 0; slot < history_size - 1; ++slot) {
                in.inserted_depth_slots[slot] = ptr<const int>(inserted_ptrs[slot]);
            }

            nb::list type_slots;
            nb::list vertex_x_slots;
            nb::list vertex_y_slots;
            nb::list vertex_z_slots;
            nb::list normal_x_slots;
            nb::list normal_y_slots;
            nb::list normal_z_slots;
            nb::list object_slots;
            witwin::channel::native_ext::DiffractionPathSlotOutputs out{};

            drjit::eval(prefix_depth, order);
            for (int depth = 0; depth < max_depth; ++depth) {
                DiffInt32 type_slot = drjit::zeros<DiffInt32>(static_cast<size_t>(n_states));
                drjit::eval(type_slot);
                out.type_slots[depth] = drjit_data_ptr_mut(type_slot);
                type_slots.append(type_slot);

                if (!return_geometry) {
                    continue;
                }

                DiffFloat vertex_x = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffFloat vertex_y = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffFloat vertex_z = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffFloat normal_x = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffFloat normal_y = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffFloat normal_z = drjit::zeros<DiffFloat>(static_cast<size_t>(n_states));
                DiffInt32 object_slot = drjit::zeros<DiffInt32>(static_cast<size_t>(n_states));
                drjit::eval(
                    vertex_x,
                    vertex_y,
                    vertex_z,
                    normal_x,
                    normal_y,
                    normal_z,
                    object_slot
                );
                out.vertex_x_slots[depth] = drjit_data_ptr_mut(vertex_x);
                out.vertex_y_slots[depth] = drjit_data_ptr_mut(vertex_y);
                out.vertex_z_slots[depth] = drjit_data_ptr_mut(vertex_z);
                out.normal_x_slots[depth] = drjit_data_ptr_mut(normal_x);
                out.normal_y_slots[depth] = drjit_data_ptr_mut(normal_y);
                out.normal_z_slots[depth] = drjit_data_ptr_mut(normal_z);
                out.object_slots[depth] = drjit_data_ptr_mut(object_slot);
                vertex_x_slots.append(vertex_x);
                vertex_y_slots.append(vertex_y);
                vertex_z_slots.append(vertex_z);
                normal_x_slots.append(normal_x);
                normal_y_slots.append(normal_y);
                normal_z_slots.append(normal_z);
                object_slots.append(object_slot);
            }

            witwin::channel::native_ext::build_diffraction_path_slots(
                in,
                out,
                n_states,
                max_depth,
                return_geometry,
                reflection_code,
                diffraction_code
            );

            if (!return_geometry) {
                return nb::make_tuple(
                    type_slots,
                    nb::none(),
                    nb::none(),
                    nb::none(),
                    nb::none(),
                    nb::none(),
                    nb::none(),
                    nb::none()
                );
            }
            return nb::make_tuple(
                type_slots,
                vertex_x_slots,
                vertex_y_slots,
                vertex_z_slots,
                normal_x_slots,
                normal_y_slots,
                normal_z_slots,
                object_slots
            );
        },
        nb::arg("prefix_depth"),
        nb::arg("order"),
        nb::arg("path_edge_slots"),
        nb::arg("inserted_depth_slots"),
        nb::arg("history_size"),
        nb::arg("n_states"),
        nb::arg("max_depth"),
        nb::arg("return_geometry"),
        nb::arg("first_interaction_pos_x") = nb::none(),
        nb::arg("first_interaction_pos_y") = nb::none(),
        nb::arg("first_interaction_pos_z") = nb::none(),
        nb::arg("edge_pos_x") = nb::none(),
        nb::arg("edge_pos_y") = nb::none(),
        nb::arg("edge_pos_z") = nb::none(),
        nb::arg("edge_n0_x") = nb::none(),
        nb::arg("edge_n0_y") = nb::none(),
        nb::arg("edge_n0_z") = nb::none(),
        nb::arg("edge_object_idx") = nb::none(),
        nb::arg("n_edges") = 0,
        nb::arg("reflection_code") = 1,
        nb::arg("diffraction_code") = 2,
        "Build diffraction path type/depth slots on the GPU from replay history."
    );

    m.attr("PACKED_CORE_FLOATS") = witwin::channel::native_ext::PACKED_CORE_FLOATS;
    m.attr("PACKED_MAX_HISTORY_SLOTS") = witwin::channel::native_ext::PACKED_MAX_HISTORY_SLOTS;
    m.attr("DIFFRACTION_PATH_SLOT_MAX_DEPTH") = witwin::channel::native_ext::DIFFRACTION_PATH_SLOT_MAX_DEPTH;
}
