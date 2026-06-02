#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// =========================================================================
// Batch Edge Geometry Computation
//
// Replaces the per-edge Python loop in compile.py that calls
// compute_edge_geometry() and preload_diffraction_edge_geometry().
//
// For each interior edge (2 adjacent faces), computes:
//   - Edge midpoint position
//   - Normalized edge direction
//   - Both adjacent face normals (n0, nn), oriented outward
//   - Wedge angle parameter (wedge_n = exterior_angle / pi)
//   - Edge length
//
// All results are written to contiguous GPU buffers, eliminating the
// per-edge scalar() GPU->CPU sync in preload_diffraction_edge_geometry().
// =========================================================================

// -------------------------------------------------------------------------
// Batch compute diffraction edge geometry for all interior edges.
//
// Input (all device pointers):
//   vertices_x/y/z:     [n_verts] mesh vertex positions, SoA layout
//   face_normals_x/y/z: [n_faces] unit face normals, SoA layout
//   edge_v0:      [n_edges] — first vertex index per edge
//   edge_v1:      [n_edges] — second vertex index per edge
//   edge_face0:   [n_edges] — first adjacent face index
//   edge_face1:   [n_edges] — second adjacent face index
//
// Output (all device pointers):
//   out_pos_x/y/z: [n_edges] edge midpoint xyz
//   out_dir_x/y/z: [n_edges] normalized edge direction xyz
//   out_n0_x/y/z:  [n_edges] face 0 normal (oriented outward) xyz
//   out_nn_x/y/z:  [n_edges] face 1 normal (oriented outward) xyz
//   out_wedge_n:  [n_edges]     — wedge angle parameter
//   out_length:   [n_edges]     — edge length
// -------------------------------------------------------------------------
void batch_edge_geometry(
    const float*  vertices_x,     // [n_verts]
    const float*  vertices_y,     // [n_verts]
    const float*  vertices_z,     // [n_verts]
    const float*  face_normals_x, // [n_faces]
    const float*  face_normals_y, // [n_faces]
    const float*  face_normals_z, // [n_faces]
    const int*    edge_v0,        // [n_edges]
    const int*    edge_v1,        // [n_edges]
    const int*    edge_face0,     // [n_edges]
    const int*    edge_face1,     // [n_edges]
    float*        out_pos_x,      // [n_edges]
    float*        out_pos_y,      // [n_edges]
    float*        out_pos_z,      // [n_edges]
    float*        out_dir_x,      // [n_edges]
    float*        out_dir_y,      // [n_edges]
    float*        out_dir_z,      // [n_edges]
    float*        out_n0_x,       // [n_edges]
    float*        out_n0_y,       // [n_edges]
    float*        out_n0_z,       // [n_edges]
    float*        out_nn_x,       // [n_edges]
    float*        out_nn_y,       // [n_edges]
    float*        out_nn_z,       // [n_edges]
    float*        out_wedge_n,    // [n_edges]
    float*        out_length,     // [n_edges]
    int           n_edges
);

} // namespace witwin::channel::native_ext
