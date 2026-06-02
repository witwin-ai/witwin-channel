#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <common/soa_ops.h>
#include <scene_build/edge_geometry/edge_geometry.h>

namespace witwin::channel::native_ext {
namespace {

using common::Vec3f;
using common::add;
using common::cross;
using common::dot;
using common::load_xyz;
using common::mul;
using common::neg;
using common::norm;
using common::store_xyz;
using common::sub;
using common::throw_cuda;

constexpr float EG_PI  = 3.14159265358979323846f;
constexpr float EG_EPS = 1.0e-10f;

using f3 = Vec3f;

__device__ __forceinline__ float len3(f3 a) { return norm(a); }
__device__ __forceinline__ f3 norm3(f3 a) {
    float l = len3(a) + EG_EPS;
    return {a.x/l, a.y/l, a.z/l};
}

// =========================================================================
// Batch edge geometry kernel
//
// One thread per edge. Computes midpoint, direction, oriented face
// normals, wedge angle, and length.
//
// Normal orientation: n0 and nn are oriented so that they point into the
// exterior wedge region. The convention follows the Slang reference:
//   - edge_hat = normalize(p1 - p0)
//   - to_hat = cross(n0, edge_hat) should point from the edge outward
//   - If it points inward (dot with mesh center direction < 0), flip n0
//   - Same logic for nn
//
// Simplified approach: orient n0 so that cross(n0, edge_hat) points
// "away" from nn, and vice versa. This gives the standard UTD wedge
// convention where the exterior angle is between n0 and nn.
// =========================================================================
__global__ void edge_geometry_kernel(
    const float* __restrict__ vertices_x,
    const float* __restrict__ vertices_y,
    const float* __restrict__ vertices_z,
    const float* __restrict__ face_normals_x,
    const float* __restrict__ face_normals_y,
    const float* __restrict__ face_normals_z,
    const int*   __restrict__ edge_v0,
    const int*   __restrict__ edge_v1,
    const int*   __restrict__ edge_face0,
    const int*   __restrict__ edge_face1,
    float*       __restrict__ out_pos_x,
    float*       __restrict__ out_pos_y,
    float*       __restrict__ out_pos_z,
    float*       __restrict__ out_dir_x,
    float*       __restrict__ out_dir_y,
    float*       __restrict__ out_dir_z,
    float*       __restrict__ out_n0_x,
    float*       __restrict__ out_n0_y,
    float*       __restrict__ out_n0_z,
    float*       __restrict__ out_nn_x,
    float*       __restrict__ out_nn_y,
    float*       __restrict__ out_nn_z,
    float*       __restrict__ out_wedge_n,
    float*       __restrict__ out_length,
    int n_edges)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_edges) return;

    // Load vertex positions
    int v0i = edge_v0[eid];
    int v1i = edge_v1[eid];
    f3 p0 = load_xyz(vertices_x, vertices_y, vertices_z, v0i);
    f3 p1 = load_xyz(vertices_x, vertices_y, vertices_z, v1i);

    // Edge vector and length
    f3 edge_vec = sub(p1, p0);
    float edge_len = len3(edge_vec);
    f3 edge_hat = norm3(edge_vec);

    // Midpoint
    f3 mid = mul(add(p0, p1), 0.5f);

    // Load face normals
    int f0i = edge_face0[eid];
    int f1i = edge_face1[eid];
    f3 fn0 = load_xyz(face_normals_x, face_normals_y, face_normals_z, f0i);
    f3 fn1 = load_xyz(face_normals_x, face_normals_y, face_normals_z, f1i);

    // Orient normals for consistent UTD wedge convention.
    // We want n0 and nn such that cross(n0, edge_hat) points toward
    // the exterior region. The tangent vectors t0 = cross(n0, edge_hat)
    // and t1 = cross(nn, edge_hat) should point into opposite half-planes.
    f3 t0 = cross(fn0, edge_hat);
    f3 t1 = cross(fn1, edge_hat);

    // If t0 and t1 point in the same direction (dot > 0), flip fn1
    if (dot(t0, t1) > 0.f) {
        fn1 = neg(fn1);
        t1 = neg(t1);
    }

    // Ensure t0 points "outward" (away from fn1 side): dot(t0, fn1) should be >= 0
    // If not, flip both normals
    if (dot(t0, fn1) < 0.f) {
        fn0 = neg(fn0);
        fn1 = neg(fn1);
    }

    // Compute wedge angle: exterior_angle = 2*pi - interior_angle
    // interior_angle = acos(clamp(-dot(n0, nn), -1, 1))
    float cos_interior = fminf(fmaxf(-dot(fn0, fn1), -1.f), 1.f);
    float interior_angle = acosf(cos_interior);
    float exterior_angle = 2.f * EG_PI - interior_angle;
    float wedge_n = exterior_angle / EG_PI;

    // Store results
    store_xyz(out_pos_x, out_pos_y, out_pos_z, eid, mid);
    store_xyz(out_dir_x, out_dir_y, out_dir_z, eid, edge_hat);
    store_xyz(out_n0_x, out_n0_y, out_n0_z, eid, fn0);
    store_xyz(out_nn_x, out_nn_y, out_nn_z, eid, fn1);
    out_wedge_n[eid] = wedge_n;
    out_length[eid]  = edge_len;
}

} // anonymous namespace

// =========================================================================
// Host launcher
// =========================================================================
void batch_edge_geometry(
    const float* vertices_x,
    const float* vertices_y,
    const float* vertices_z,
    const float* face_normals_x,
    const float* face_normals_y,
    const float* face_normals_z,
    const int*   edge_v0,
    const int*   edge_v1,
    const int*   edge_face0,
    const int*   edge_face1,
    float*       out_pos_x,
    float*       out_pos_y,
    float*       out_pos_z,
    float*       out_dir_x,
    float*       out_dir_y,
    float*       out_dir_z,
    float*       out_n0_x,
    float*       out_n0_y,
    float*       out_n0_z,
    float*       out_nn_x,
    float*       out_nn_y,
    float*       out_nn_z,
    float*       out_wedge_n,
    float*       out_length,
    int          n_edges)
{
    if (n_edges <= 0) return;

    constexpr int BLOCK = 256;
    int grid = (n_edges + BLOCK - 1) / BLOCK;

    edge_geometry_kernel<<<grid, BLOCK>>>(
        vertices_x, vertices_y, vertices_z,
        face_normals_x, face_normals_y, face_normals_z,
        edge_v0, edge_v1, edge_face0, edge_face1,
        out_pos_x, out_pos_y, out_pos_z,
        out_dir_x, out_dir_y, out_dir_z,
        out_n0_x, out_n0_y, out_n0_z,
        out_nn_x, out_nn_y, out_nn_z,
        out_wedge_n, out_length,
        n_edges);

    throw_cuda(cudaGetLastError(), "edge_geometry_kernel launch");
}

} // namespace witwin::channel::native_ext
