#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// =========================================================================
// Batch Surface Coplanarity Check
//
// Replaces the per-edge-pair `scalar()` GPU->CPU sync loop in compile.py
// with a single vectorized kernel. For each edge that has exactly 2
// adjacent faces, checks:
//   1. Normal angle: |dot(n_a, n_b)| >= normal_cos_tol
//   2. Plane distance: both non-shared vertices within plane_tol of
//      the shared edge's plane
//
// The Python side then reads back the bool array and runs CPU union-find
// to group coplanar surfaces. This reduces N edge-pair GPU->CPU syncs
// to a single kernel launch + one bulk D2H copy.
// =========================================================================

// Default thresholds from compile.py
constexpr float SURFACE_GROUP_NORMAL_COS_TOL = 0.99999f;  // 1.0 - 1e-5
constexpr float SURFACE_GROUP_PLANE_TOL      = 1.0e-5f;

// -------------------------------------------------------------------------
// Batch coplanarity check for all edges with 2 adjacent faces.
//
// Input arrays use per-edge indexing. Only edges with 2 adjacent faces
// should be included; boundary edges should be excluded by the caller.
//
// face_normals_x/y/z: [n_faces] precomputed unit face normals
// edge_face_a:  [n_edges] — face index of the first adjacent face
// edge_face_b:  [n_edges] — face index of the second adjacent face
// vertices_x/y/z: [n_verts] mesh vertex positions
// faces_x/y/z:    [n_faces] triangle vertex index triplets
//
// Output:
// is_coplanar:  [n_edges] — 1 if coplanar, 0 otherwise (int, not bool,
//               for easy D2H copy and nanobind compatibility)
// -------------------------------------------------------------------------
void batch_coplanarity_check(
    const float*    face_normals_x,  // [n_faces]
    const float*    face_normals_y,  // [n_faces]
    const float*    face_normals_z,  // [n_faces]
    const int*      edge_face_a,     // [n_edges]
    const int*      edge_face_b,     // [n_edges]
    const float*    vertices_x,      // [n_verts]
    const float*    vertices_y,      // [n_verts]
    const float*    vertices_z,      // [n_verts]
    const int*      faces_x,         // [n_faces]
    const int*      faces_y,         // [n_faces]
    const int*      faces_z,         // [n_faces]
    int*            is_coplanar,     // [n_edges] output
    int             n_edges,
    int             n_faces,
    int             n_verts,
    float           normal_cos_tol,  // default SURFACE_GROUP_NORMAL_COS_TOL
    float           plane_tol        // default SURFACE_GROUP_PLANE_TOL
);

} // namespace witwin::channel::native_ext
