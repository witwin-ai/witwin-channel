#include "drjit_common.h"
#include <scene_build/edge_geometry/bind.h>

#include <scene_build/edge_geometry/edge_geometry.h>

void register_edge_geometry_bindings(nb::module_ &m) {
    m.def(
        "batch_edge_geometry_arrays",
        [](
            Float vertices_x,
            Float vertices_y,
            Float vertices_z,
            Float face_normals_x,
            Float face_normals_y,
            Float face_normals_z,
            Int32 edge_v0,
            Int32 edge_v1,
            Int32 edge_face0,
            Int32 edge_face1,
            int n_edges
        ) {
            if (n_edges <= 0) {
                return nb::make_tuple(
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0),
                    drjit::zeros<Float>(0)
                );
            }

            size_t size = static_cast<size_t>(n_edges);
            Float out_pos_x = drjit::zeros<Float>(size);
            Float out_pos_y = drjit::zeros<Float>(size);
            Float out_pos_z = drjit::zeros<Float>(size);
            Float out_dir_x = drjit::zeros<Float>(size);
            Float out_dir_y = drjit::zeros<Float>(size);
            Float out_dir_z = drjit::zeros<Float>(size);
            Float out_n0_x = drjit::zeros<Float>(size);
            Float out_n0_y = drjit::zeros<Float>(size);
            Float out_n0_z = drjit::zeros<Float>(size);
            Float out_nn_x = drjit::zeros<Float>(size);
            Float out_nn_y = drjit::zeros<Float>(size);
            Float out_nn_z = drjit::zeros<Float>(size);
            Float out_wedge_n = drjit::zeros<Float>(size);
            Float out_length = drjit::zeros<Float>(size);

            drjit::eval(
                vertices_x,
                vertices_y,
                vertices_z,
                face_normals_x,
                face_normals_y,
                face_normals_z,
                edge_v0,
                edge_v1,
                edge_face0,
                edge_face1,
                out_pos_x,
                out_pos_y,
                out_pos_z,
                out_dir_x,
                out_dir_y,
                out_dir_z,
                out_n0_x,
                out_n0_y,
                out_n0_z,
                out_nn_x,
                out_nn_y,
                out_nn_z,
                out_wedge_n,
                out_length
            );

            witwin::channel::native_ext::batch_edge_geometry(
                drjit_data_ptr(vertices_x),
                drjit_data_ptr(vertices_y),
                drjit_data_ptr(vertices_z),
                drjit_data_ptr(face_normals_x),
                drjit_data_ptr(face_normals_y),
                drjit_data_ptr(face_normals_z),
                drjit_data_ptr(edge_v0),
                drjit_data_ptr(edge_v1),
                drjit_data_ptr(edge_face0),
                drjit_data_ptr(edge_face1),
                drjit_data_ptr_mut(out_pos_x),
                drjit_data_ptr_mut(out_pos_y),
                drjit_data_ptr_mut(out_pos_z),
                drjit_data_ptr_mut(out_dir_x),
                drjit_data_ptr_mut(out_dir_y),
                drjit_data_ptr_mut(out_dir_z),
                drjit_data_ptr_mut(out_n0_x),
                drjit_data_ptr_mut(out_n0_y),
                drjit_data_ptr_mut(out_n0_z),
                drjit_data_ptr_mut(out_nn_x),
                drjit_data_ptr_mut(out_nn_y),
                drjit_data_ptr_mut(out_nn_z),
                drjit_data_ptr_mut(out_wedge_n),
                drjit_data_ptr_mut(out_length),
                n_edges
            );

            return nb::make_tuple(
                out_pos_x,
                out_pos_y,
                out_pos_z,
                out_dir_x,
                out_dir_y,
                out_dir_z,
                out_n0_x,
                out_n0_y,
                out_n0_z,
                out_nn_x,
                out_nn_y,
                out_nn_z,
                out_wedge_n,
                out_length
            );
        },
        "Batch edge geometry computation returning freshly materialized Dr.Jit arrays."
    );
}
