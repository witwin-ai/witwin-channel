#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <common/soa_ops.h>
#include <scene_build/coplanarity/coplanarity.h>

namespace witwin::channel::native_ext {
namespace {

using common::Vec3f;
using common::dot;
using common::load_xyz;
using common::throw_cuda;

// =========================================================================
// Batch coplanarity check kernel
//
// One thread per edge. Checks:
//   1. |dot(normal_a, normal_b)| >= normal_cos_tol
//   2. For each face's non-shared vertex, distance to the plane of the
//      other face is <= plane_tol
// =========================================================================
__global__ void coplanarity_check_kernel(
    const float* __restrict__ face_normals_x, // [n_faces]
    const float* __restrict__ face_normals_y, // [n_faces]
    const float* __restrict__ face_normals_z, // [n_faces]
    const int*   __restrict__ edge_face_a,    // [n_edges]
    const int*   __restrict__ edge_face_b,    // [n_edges]
    const float* __restrict__ vertices_x,     // [n_verts]
    const float* __restrict__ vertices_y,     // [n_verts]
    const float* __restrict__ vertices_z,     // [n_verts]
    const int*   __restrict__ faces_x,        // [n_faces]
    const int*   __restrict__ faces_y,        // [n_faces]
    const int*   __restrict__ faces_z,        // [n_faces]
    int*         __restrict__ is_coplanar,    // [n_edges] output
    int n_edges,
    float normal_cos_tol,
    float plane_tol)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_edges) return;

    int fa = edge_face_a[eid];
    int fb = edge_face_b[eid];

    // Load face normals
    Vec3f na = load_xyz(face_normals_x, face_normals_y, face_normals_z, fa);
    Vec3f nb = load_xyz(face_normals_x, face_normals_y, face_normals_z, fb);

    // Stage 1: normal angle check
    float dot_normals = dot(na, nb);
    if (fabsf(dot_normals) < normal_cos_tol) {
        is_coplanar[eid] = 0;
        return;
    }

    // Load face vertex indices
    int va0 = faces_x[fa], va1 = faces_y[fa], va2 = faces_z[fa];
    int vb0 = faces_x[fb], vb1 = faces_y[fb], vb2 = faces_z[fb];

    // Find shared vertices (triangles sharing an edge must share exactly 2 vertices)
    // and identify the non-shared vertex from each face
    int other_a = -1;  // vertex in face_a not shared with face_b
    int other_b = -1;  // vertex in face_b not shared with face_a

    // For face_a vertices, check if they appear in face_b
    int va[3] = {va0, va1, va2};
    int vb[3] = {vb0, vb1, vb2};

    for (int i = 0; i < 3; ++i) {
        bool shared = false;
        for (int j = 0; j < 3; ++j) {
            if (va[i] == vb[j]) { shared = true; break; }
        }
        if (!shared) other_a = va[i];
    }
    for (int i = 0; i < 3; ++i) {
        bool shared = false;
        for (int j = 0; j < 3; ++j) {
            if (vb[i] == va[j]) { shared = true; break; }
        }
        if (!shared) other_b = vb[i];
    }

    if (other_a < 0 || other_b < 0) {
        // Degenerate: faces share all 3 vertices or share < 2
        is_coplanar[eid] = 0;
        return;
    }

    // Stage 2: plane distance check
    // Use a shared vertex as the plane point (any vertex of face_a)
    Vec3f plane_pt = load_xyz(vertices_x, vertices_y, vertices_z, va0);
    Vec3f pt_a = load_xyz(vertices_x, vertices_y, vertices_z, other_a);
    Vec3f pt_b = load_xyz(vertices_x, vertices_y, vertices_z, other_b);

    float dist_a = fabsf(
        (pt_a.x - plane_pt.x) * na.x +
        (pt_a.y - plane_pt.y) * na.y +
        (pt_a.z - plane_pt.z) * na.z
    );
    float dist_b = fabsf(
        (pt_b.x - plane_pt.x) * na.x +
        (pt_b.y - plane_pt.y) * na.y +
        (pt_b.z - plane_pt.z) * na.z
    );

    is_coplanar[eid] = (dist_a <= plane_tol && dist_b <= plane_tol) ? 1 : 0;
}

} // anonymous namespace

// =========================================================================
// Host launcher
// =========================================================================
void batch_coplanarity_check(
    const float* face_normals_x,
    const float* face_normals_y,
    const float* face_normals_z,
    const int*   edge_face_a,
    const int*   edge_face_b,
    const float* vertices_x,
    const float* vertices_y,
    const float* vertices_z,
    const int*   faces_x,
    const int*   faces_y,
    const int*   faces_z,
    int*         is_coplanar,
    int          n_edges,
    int          n_faces,
    int          n_verts,
    float        normal_cos_tol,
    float        plane_tol)
{
    if (n_edges <= 0) return;

    constexpr int BLOCK = 256;
    int grid = (n_edges + BLOCK - 1) / BLOCK;

    coplanarity_check_kernel<<<grid, BLOCK>>>(
        face_normals_x, face_normals_y, face_normals_z,
        edge_face_a, edge_face_b,
        vertices_x, vertices_y, vertices_z,
        faces_x, faces_y, faces_z,
        is_coplanar,
        n_edges, normal_cos_tol, plane_tol);

    throw_cuda(cudaGetLastError(), "coplanarity_check_kernel launch");
}

} // namespace witwin::channel::native_ext
