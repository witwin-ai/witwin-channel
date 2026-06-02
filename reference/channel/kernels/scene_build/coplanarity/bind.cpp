#include "drjit_common.h"
#include <scene_build/coplanarity/bind.h>

#include <scene_build/coplanarity/coplanarity.h>

void register_coplanarity_bindings(nb::module_ &m) {
    m.def(
        "batch_coplanarity_check_arrays",
        [](
            Float face_normals_x,
            Float face_normals_y,
            Float face_normals_z,
            Int32 edge_face_a,
            Int32 edge_face_b,
            Float vertices_x,
            Float vertices_y,
            Float vertices_z,
            Int32 faces_x,
            Int32 faces_y,
            Int32 faces_z,
            int n_edges,
            int n_faces,
            int n_verts,
            float normal_cos_tol,
            float plane_tol
        ) {
            if (n_edges <= 0) {
                return drjit::zeros<Int32>(0);
            }

            Int32 is_coplanar = drjit::zeros<Int32>(static_cast<size_t>(n_edges));
            drjit::eval(
                face_normals_x,
                face_normals_y,
                face_normals_z,
                edge_face_a,
                edge_face_b,
                vertices_x,
                vertices_y,
                vertices_z,
                faces_x,
                faces_y,
                faces_z,
                is_coplanar
            );

            witwin::channel::native_ext::batch_coplanarity_check(
                drjit_data_ptr(face_normals_x),
                drjit_data_ptr(face_normals_y),
                drjit_data_ptr(face_normals_z),
                drjit_data_ptr(edge_face_a),
                drjit_data_ptr(edge_face_b),
                drjit_data_ptr(vertices_x),
                drjit_data_ptr(vertices_y),
                drjit_data_ptr(vertices_z),
                drjit_data_ptr(faces_x),
                drjit_data_ptr(faces_y),
                drjit_data_ptr(faces_z),
                drjit_data_ptr_mut(is_coplanar),
                n_edges,
                n_faces,
                n_verts,
                normal_cos_tol,
                plane_tol
            );

            return is_coplanar;
        },
        "Batch coplanarity check returning a freshly materialized Dr.Jit integer mask."
    );
    m.attr("SURFACE_GROUP_NORMAL_COS_TOL") = witwin::channel::native_ext::SURFACE_GROUP_NORMAL_COS_TOL;
    m.attr("SURFACE_GROUP_PLANE_TOL") = witwin::channel::native_ext::SURFACE_GROUP_PLANE_TOL;
}
